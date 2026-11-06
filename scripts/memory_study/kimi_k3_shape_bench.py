# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure Kimi K3's MXFP8 matmuls at the shapes each parallelism config
produces, to check the roofline's 60%-of-FP8-peak assumption.

Shapes per GPU per microbatch:
  - dense linears of one KDA + MoE and one MLA + MoE layer: CP sets the rows
    (seq_len / CP tokens after the sequence-parallel all-gather); TP splits
    column-parallel weights on their output dimension and row-parallel weights
    on their input dimension. Replicated weights (MLA's low-rank wq_a / wkv_a,
    the router, the routed down / up projections) keep their full size; the
    router and routed down / up run on the TP-local tokens.
    Timed as forward, input-gradient and weight-gradient GEMMs on
    pre-quantized MXFP8 operands, replayed from a CUDA graph (GEMM kernels
    only).
  - routed experts as grouped GEMMs (w13 then w2, forward + backward) through
    torchao's MXFP8 grouped GEMM, as TorchTitan's MXFP8 experts run them, not
    DistMoE: EP sets the local experts (896 / EP), CP x TP the local tokens, so
    each expert gets local tokens x 16 / local experts rows (balanced routing).
    A profiler splits the time into GEMM kernels and quantization / layout
    kernels. torchao's weight-gradient kernel needs a power-of-2 group count,
    so the groups are padded with empty experts (same rows and FLOPs).

Writes bench/shapes_<gpu>.json.

Usage:
  PYTHONPATH=.:scripts/memory_study python scripts/memory_study/kimi_k3_shape_bench.py
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from kimi_k3_kernel_bench import bench_graph_ms, gemm_times, guarded
from torch.profiler import profile, ProfilerActivity

D, L, FE, V = 7168, 3584, 3072, 163840
NUM_EXPERTS, TOP_K = 896, 16
FP8_PEAK = 5e15
BF16_PEAK = 2.5e15

# (name, in_features, out_features, TP layout): "col" splits out, "row" splits
# in, "rep" keeps the weight whole on the full tokens, "rep_local" keeps it
# whole on the TP-local tokens.
KDA_LINEARS = [
    ("q_proj", D, 12288, "col"), ("k_proj", D, 12288, "col"), ("v_proj", D, 12288, "col"),
    ("output_gate", D, 12288, "col"), ("output_proj", 12288, D, "row"),
]
MLA_LINEARS = [
    ("wq_a", D, 1536, "rep"), ("wq_b", 1536, 18432, "col"), ("wkv_a", D, 576, "rep"),
    ("wkv_b", 512, 24576, "col"), ("gate", D, 12288, "col"), ("wo", 12288, D, "row"),
]
MOE_DENSE = [
    ("shared_w13", D, 12288, "col"), ("shared_w2", 6144, D, "row"),
    ("router", D, NUM_EXPERTS, "rep_local"),
    ("routed_down", D, L, "rep_local"), ("routed_up", L, D, "rep_local"),
]

# (context, seq_len) x (CP, TP) of the fit map; EP 64 (FSDP 512 and PP 8) and
# EP 32 (PP 16)
SPLITS = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4)]
CONTEXTS = {"8K": 8192, "32K": 32768}
EPS = (64, 32)


def linear_shape(m_full: int, tp: int, k: int, n: int, layout: str) -> tuple[int, int, int]:
    if layout == "col":
        return m_full, k, n // tp
    if layout == "row":
        return m_full, k // tp, n
    if layout == "rep":
        return m_full, k, n
    return m_full // tp, k, n  # rep_local


def bench_dense(m_full: int, tp: int, dtype: str = "mxfp8") -> dict:
    """The dense linears of a KDA + MoE and an MLA + MoE layer at one split, in
    MXFP8 (efficiency against the FP8 peak) or bf16 (against the bf16 peak)."""
    peak = FP8_PEAK if dtype == "mxfp8" else BF16_PEAK
    out = {}
    for scope, shapes in (("kda_layer", KDA_LINEARS + MOE_DENSE), ("mla_layer", MLA_LINEARS + MOE_DENSE)):
        rows, ms, flops = [], 0.0, 0.0
        for name, k0, n0, layout in shapes:
            m, k, n = linear_shape(m_full, tp, k0, n0, layout)
            t = gemm_times(m, k, n, dtype)
            f = 3 * 2 * m * k * n
            rows.append({"name": name, "m": m, "k": k, "n": n, **t,
                         "eff": f / (sum(t.values()) * 1e-3) / peak})
            ms += sum(t.values())
            flops += f
        out[scope] = {"ms": ms, "eff": flops / (ms * 1e-3) / peak, "gemms": rows}
    return out


def _is_gemm(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in ("gemm", "cutlass", "nvjet", "sm100", "sm103", "grouped_mm", "xmma"))


def bench_experts(local_tokens: int, ep: int) -> dict:
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
    from torchao.prototype.moe_training.utils import _quantize_then_scaled_grouped_mm

    cfg = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe("mxfp8_rceil"))
    experts = NUM_EXPERTS // ep
    groups = 1 << (experts - 1).bit_length()  # torchao's wgrad kernel: power-of-2 groups
    per = local_tokens * TOP_K // experts // 128 * 128  # 128-row scale tiles
    rows = per * experts
    dev, bf = "cuda", torch.bfloat16
    offs = torch.tensor([per] * experts + [0] * (groups - experts), device=dev).cumsum(0).to(torch.int32)
    x = torch.randn(rows, L, device=dev, dtype=bf, requires_grad=True)
    w13 = (0.02 * torch.randn(groups, 2 * FE, L, device=dev)).to(bf).requires_grad_()
    w2 = (0.02 * torch.randn(groups, L, FE, device=dev)).to(bf).requires_grad_()
    dy = torch.randn(rows, L, device=dev, dtype=bf)

    def gmm(a, w):
        return _quantize_then_scaled_grouped_mm(a, w.transpose(-2, -1), config=cfg, offs=offs)

    def step():
        g, u = gmm(x, w13).chunk(2, dim=-1)
        torch.autograd.backward(gmm(F.silu(g) * u, w2), dy)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    reps = 10
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            step()
        torch.cuda.synchronize()
    gemm_us = other_us = 0.0
    kinds: dict[str, float] = {}
    for e in prof.key_averages():
        us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if us <= 0 or e.key.startswith(("ProfilerStep", "aten::", "autograd::", "cuda")):
            continue
        if _is_gemm(e.key):
            gemm_us += us
        else:
            other_us += us
            kinds[e.key[:80]] = kinds.get(e.key[:80], 0.0) + us / reps / 1e3
    flops = 3 * 2 * rows * (L * 2 * FE + FE * L)
    gemm_ms, other_ms = gemm_us / reps / 1e3, other_us / reps / 1e3
    top = dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:6])
    return {"local_experts": experts, "rows_per_expert": per, "gemm_ms": gemm_ms, "other_ms": other_ms,
            "gemm_eff": flops / (gemm_ms * 1e-3) / FP8_PEAK,
            "total_eff": flops / ((gemm_ms + other_ms) * 1e-3) / FP8_PEAK, "other_kernels_ms": top}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sections", nargs="+", default=["dense", "experts"],
                        choices=["dense", "dense_bf16", "experts"])
    args = parser.parse_args()
    gpu = torch.cuda.get_device_name().replace("NVIDIA ", "").replace(" ", "_")
    out_path = args.out or Path(__file__).resolve().parent / "bench" / f"shapes_{gpu}.json"
    # keep the sections this run does not measure
    result: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    result.update({"gpu": gpu, "torch": torch.__version__, "fp8_peak": FP8_PEAK, "bf16_peak": BF16_PEAK})

    if "dense" in args.sections:
        dense_shapes = sorted({(T // cp, tp) for T in CONTEXTS.values() for cp, tp in SPLITS})

        def run_dense():
            out = {}
            for m, tp in dense_shapes:
                r = bench_dense(m, tp)
                out[f"m{m}_tp{tp}"] = {"m": m, "tp": tp, **r}
                print(f"dense M={m:6d} TP={tp}: KDA layer {r['kda_layer']['eff']:.0%}, "
                      f"MLA layer {r['mla_layer']['eff']:.0%} of FP8 peak")
            return out
        result["dense"] = guarded("dense", run_dense)

    if "dense_bf16" in args.sections:
        dense_shapes = sorted({(T // cp, tp) for T in CONTEXTS.values() for cp, tp in SPLITS})

        def run_dense_bf16():
            out = {}
            for m, tp in dense_shapes:
                r = bench_dense(m, tp, "bf16")
                out[f"m{m}_tp{tp}"] = {"m": m, "tp": tp, **r}
                print(f"dense bf16 M={m:6d} TP={tp}: KDA layer {r['kda_layer']['eff']:.0%}, "
                      f"MLA layer {r['mla_layer']['eff']:.0%} of bf16 peak")
            return out
        result["dense_bf16"] = guarded("dense_bf16", run_dense_bf16)

    if "experts" in args.sections:
        expert_shapes = sorted({(T // (cp * tp), ep) for T in CONTEXTS.values() for cp, tp in SPLITS for ep in EPS})

        def run_experts():
            out = {}
            for tl, ep in expert_shapes:
                r = bench_experts(tl, ep)
                out[f"t{tl}_ep{ep}"] = {"local_tokens": tl, "ep": ep, **r}
                print(f"experts tokens={tl:6d} EP={ep}: {r['local_experts']} experts x {r['rows_per_expert']} rows, "
                      f"GEMM {r['gemm_eff']:.0%}, with quantization {r['total_eff']:.0%} of FP8 peak")
            return out
        result["experts"] = guarded("experts", run_experts)

    out_path.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
