# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure Kimi K3 kernel throughput on the local GPU, for the roofline.

Times, at the per-GPU token counts the parallelism configs produce:
  - dense linears of one KDA + MoE and one MLA + MoE layer, and the LM head:
    forward, input-gradient and weight-gradient GEMMs, in bf16 and in MXFP8
    (GEMM only, operands quantized beforehand);
  - routed-expert GEMMs (w13, w2) at a range of tokens per expert, as dense
    GEMMs: torch._grouped_mm does not run on SM103 yet, so this is an
    optimistic stand-in for the grouped kernel;
  - MLA attention (cuDNN SDPA, causal) and KDA (Attention Gym chunk kernel,
    the backend TorchTitan selects), forward and forward + backward.

Writes bench/kernels_<gpu>.json next to this script.

Usage:
  PYTHONPATH=. python scripts/memory_study/kimi_k3_kernel_bench.py
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from kimi_k3_params import (
    build_model_config,
    ParallelismConfig,
    replace,
    skip_flash_attention_activation,
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchtitan.models.common.attention.linear_attention import chunk_kernel_options
from torchtitan.models.common.attn_gym import chunk_kda
from triton.testing import do_bench

# Tokens per GPU per microbatch: 8K / 32K contexts split by CP x TP up to 8.
TOKENS = (1024, 2048, 4096, 8192, 16384, 32768)
# Tokens per routed expert per microbatch (balanced routing spans these).
EXPERT_TOKENS = (512, 1024, 2048, 4096, 8192, 16384)
SEQ_LENS = (4096, 8192, 16384, 32768)
# KDA + MoE layer and MLA + MoE layer (layer 0 has a dense FFN instead).
KDA_LAYER, MLA_LAYER = 1, 3
MIN_LINEAR_NUMEL = 1 << 20  # skip tiny projections


def bench_ms(fn) -> float:
    return do_bench(fn, warmup=25, rep=100)


def bench_graph_ms(fn) -> float:
    """GPU time of fn replayed from a CUDA graph. Eager torchao MXTensor calls
    cost ~75 us of CPU dispatch each, which would otherwise dominate small GEMMs;
    training hides that overhead (compile), so the roofline needs kernel time."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return do_bench(graph.replay, warmup=25, rep=100)


def linear_shapes(cfg) -> dict[str, list[tuple[str, int, int]]]:
    """(name, in_features, out_features) of each large dense linear, per scope.
    A 3D (2, F, D) weight is a fused gate + up projection: one D -> 2F GEMM."""
    with torch.device("meta"), skip_flash_attention_activation():
        model = cfg.build()
    scopes: dict[str, list[tuple[str, int, int]]] = {
        "kda_layer": [],
        "mla_layer": [],
        "lm_head": [],
    }
    for name, p in model.named_parameters():
        if p.numel() < MIN_LINEAR_NUMEL or "embed" in name or "routed_experts" in name:
            continue
        if p.ndim == 3 and p.shape[0] == 2:
            out_f, in_f = 2 * p.shape[1], p.shape[2]
        elif p.ndim == 2:
            out_f, in_f = p.shape
        else:
            continue
        if name.startswith(f"layers.{KDA_LAYER}."):
            scopes["kda_layer"].append((name.split(".", 2)[2], in_f, out_f))
        elif name.startswith(f"layers.{MLA_LAYER}."):
            scopes["mla_layer"].append((name.split(".", 2)[2], in_f, out_f))
        elif "layers." not in name and out_f == cfg.vocab_size:
            scopes["lm_head"].append((name, in_f, out_f))
    return scopes


def to_mx(t: torch.Tensor) -> MXTensor:
    return MXTensor.to_mx(
        t,
        torch.float8_e4m3fn,
        32,
        kernel_preference=KernelPreference.AUTO,
        is_swizzled_scales=True,
    )


def gemm_times(m: int, k: int, n: int, dtype: str) -> dict[str, float]:
    """Forward (m,k)x(k,n), input-gradient (m,n)x(n,k) and weight-gradient
    (n,m)x(m,k) GEMM times in ms; each contracts its inner dimension."""
    dev, bf = "cuda", torch.bfloat16
    x = torch.randn(m, k, device=dev, dtype=bf)
    w = torch.randn(n, k, device=dev, dtype=bf)
    dy = torch.randn(m, n, device=dev, dtype=bf)
    if dtype == "bf16":
        wt, dyt, xt = w.t().contiguous(), dy.t().contiguous(), x.t().contiguous()
        return {
            "fwd": bench_graph_ms(lambda: x @ w.t()),
            "dgrad": bench_graph_ms(lambda: dy @ wt.t()),
            "wgrad": bench_graph_ms(lambda: dyt @ xt.t()),
        }
    # MXFP8 operands quantized along each GEMM's contraction dimension
    x_q, w_q = to_mx(x), to_mx(w)
    dy_q, wt_q = to_mx(dy), to_mx(w.t().contiguous())
    dyt_q, xt_q = to_mx(dy.t().contiguous()), to_mx(x.t().contiguous())
    return {
        "fwd": bench_graph_ms(lambda: torch.mm(x_q, w_q.t())),
        "dgrad": bench_graph_ms(lambda: torch.mm(dy_q, wt_q.t())),
        "wgrad": bench_graph_ms(lambda: torch.mm(dyt_q, xt_q.t())),
    }


def bench_linears(scopes) -> dict:
    out = {}
    for dtype in ("bf16", "mxfp8"):
        for scope, shapes in scopes.items():
            for m in TOKENS:
                rows, total_ms, total_flops = [], 0.0, 0.0
                for name, k, n in shapes:
                    t = gemm_times(m, k, n, dtype)
                    flops = 2 * m * k * n
                    rows.append(
                        {"name": name, "k": k, "n": n, **t, "flops_each": flops}
                    )
                    total_ms += sum(t.values())
                    total_flops += 3 * flops
                out.setdefault(dtype, {}).setdefault(scope, {})[m] = {
                    "ms": total_ms,
                    "tflops": total_flops / total_ms / 1e9,
                    "gemms": rows,
                }
                print(
                    f"{dtype:5s} {scope:9s} M={m:6d}: {total_ms:8.3f} ms fwd+bwd, "
                    f"{total_flops / total_ms / 1e9:7.1f} TFLOPS"
                )
    return out


def bench_experts(hidden: int, inter: int) -> dict:
    out = {}
    for dtype in ("bf16", "mxfp8"):
        for n_tok in EXPERT_TOKENS:
            w13 = gemm_times(n_tok, hidden, 2 * inter, dtype)
            w2 = gemm_times(n_tok, inter, hidden, dtype)
            ms = sum(w13.values()) + sum(w2.values())
            flops = 3 * 2 * n_tok * (hidden * 2 * inter + inter * hidden)
            out.setdefault(dtype, {})[n_tok] = {
                "ms": ms,
                "tflops": flops / ms / 1e9,
                "w13": w13,
                "w2": w2,
            }
            print(
                f"{dtype:5s} expert   tok={n_tok:6d}: {ms:8.3f} ms fwd+bwd per expert, "
                f"{flops / ms / 1e9:7.1f} TFLOPS"
            )
    return out


def bench_mla(heads: int, d_qk: int, d_v: int) -> dict:
    out = {}
    for t in SEQ_LENS:
        q = torch.randn(
            1, heads, t, d_qk, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn(
            1, heads, t, d_v, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        do = torch.randn(1, heads, t, d_v, device="cuda", dtype=torch.bfloat16)

        def fwd():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        def fwd_bwd():
            torch.autograd.backward(fwd(), do)

        f_ms, fb_ms = bench_ms(fwd), bench_ms(fwd_bwd)
        fwd_flops = t * t * heads * (d_qk + d_v)  # causal: half of 2*T^2*H*(dqk+dv)
        out[t] = {
            "fwd_ms": f_ms,
            "fwd_bwd_ms": fb_ms,
            "fwd_tflops": fwd_flops / f_ms / 1e9,
            "fwd_bwd_tflops": 3.5 * fwd_flops / fb_ms / 1e9,
        }
        print(
            f"mla attn T={t:6d}: fwd {f_ms:7.3f} ms ({fwd_flops / f_ms / 1e9:6.1f} TFLOPS), "
            f"fwd+bwd {fb_ms:7.3f} ms ({3.5 * fwd_flops / fb_ms / 1e9:6.1f} TFLOPS, bwd = 2.5x fwd FLOPs)"
        )
    return out


def bench_kda(heads: int, d_k: int, d_v: int) -> dict:
    out = {}
    for t in SEQ_LENS:
        dev, bf = "cuda", torch.bfloat16
        q = (
            F.normalize(torch.randn(1, t, heads, d_k, device=dev), dim=-1)
            .to(bf)
            .requires_grad_()
        )
        k = (
            F.normalize(torch.randn(1, t, heads, d_k, device=dev), dim=-1)
            .to(bf)
            .requires_grad_()
        )
        v = torch.randn(1, t, heads, d_v, device=dev, dtype=bf, requires_grad=True)
        g = (
            -F.softplus(torch.randn(1, t, heads, d_k, device=dev)) * 0.1
        ).requires_grad_()
        beta = torch.sigmoid(torch.randn(1, t, heads, device=dev)).requires_grad_()
        opts = chunk_kernel_options(q)
        do = torch.randn(1, t, heads, d_v, device=dev, dtype=bf)

        def fwd():
            return chunk_kda(q, k, v, g, beta, autotune=True, kernel_options=opts)[0]

        def fwd_bwd():
            torch.autograd.backward(fwd(), do)

        f_ms, fb_ms = bench_ms(fwd), bench_ms(fwd_bwd)
        out[t] = {"fwd_ms": f_ms, "fwd_bwd_ms": fb_ms, "backend": opts.get("backend")}
        print(
            f"kda      T={t:6d}: fwd {f_ms:7.3f} ms, fwd+bwd {fb_ms:7.3f} ms "
            f"({1e3 * fb_ms / t:.2f} us/token, backend {opts.get('backend')})"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flavor", default="Kimi-K3")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = build_model_config(args.flavor, parallelism=ParallelismConfig())
    cfg = replace(cfg, vision_encoder=None)
    moe = cfg.layers[KDA_LAYER].moe
    mla = cfg.layers[MLA_LAYER].attention
    kda = cfg.layers[KDA_LAYER].delta_attention
    gpu = torch.cuda.get_device_name().replace("NVIDIA ", "").replace(" ", "_")

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "notes": "GEMMs are timed as CUDA-graph replays (kernel time, no CPU dispatch); expert GEMMs "
        "are dense per-expert GEMMs (grouped_mm unavailable on SM103); MXFP8 is GEMM-only with "
        "pre-quantized operands",
        "linears": bench_linears(linear_shapes(cfg)),
        # w13 fuses gate and up: (tokens, L) x (L, 2 Fe); w13.out_features is Fe
        "experts": bench_experts(
            moe.routed_down.out_features, moe.routed_experts.w13.out_features
        ),
        "mla_attention": bench_mla(
            mla.n_heads, mla.qk_nope_head_dim + mla.qk_rope_head_dim, mla.v_head_dim
        ),
        "kda": bench_kda(kda.num_heads, kda.head_dim, kda.head_dim),
    }
    out = args.out or Path(__file__).with_name("bench") / f"kernels_{gpu}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
