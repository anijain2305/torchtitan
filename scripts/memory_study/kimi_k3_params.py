# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Account for every Kimi K3 decoder parameter from hand-derived formulas.

Step 1 of a DeepSeek-V3-style memory study (cf.
https://deepseek-v3.ezyang.com/studies/02-hopper-memory.html): before modeling
weights/grads/optimizer/activations per rank, make sure every parameter in the
model is explained by a closed-form shape formula.

How it works:
  1. Read the free hyperparameters (D, H, E, ...) from the flavor config.
  2. Build the expected ``{fqn: shape}`` table from symbolic shape formulas
     written by hand below (e.g. ``"H*K, D"``).
  3. Build the real model on the meta device and require an exact match:
     every real parameter is claimed by exactly one formula with the same shape.
  4. Print per-layer-type tables, a category summary, and active-param variants.

The vision encoder (~447M params, <0.02% of the model) is dropped: the study
covers the text decoder only.

Usage:
  PYTHONPATH=. python scripts/memory_study/kimi_k3_params.py [--flavor Kimi-K3] [--buffers]
"""

import argparse
import contextlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from unittest import mock

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import build_model_config


# ---------------------------------------------------------------------------
# Hyperparameters (symbols used in the formulas)
# ---------------------------------------------------------------------------

SYMBOL_DOCS = {
    # Model-wide: embeddings, lm_head, norms, residual projections.
    "D": "decoder hidden size",
    "V": "vocab size",
    # Attention, shared by both kinds of attention layer.
    "H": "attention heads (KDA and MLA)",
    # KDA (non-MLA) attention: 69 linear-attention layers.
    "K": "KDA head dim (also the forget-gate low rank)",
    "W": "KDA short-conv kernel width",
    # MLA (full) attention: 24 layers (every 4th, plus the last).
    "Rq": "MLA q_lora_rank",
    "Rkv": "MLA kv_lora_rank",
    "Dn": "MLA qk_nope_head_dim",
    "Dr": "MLA qk_rope_head_dim",
    "Dv": "MLA v_head_dim",
    # Non-expert FFN: only layer 0 is dense.
    "F": "dense FFN hidden size (layer 0)",
    # Experts (LatentMoE, layers 1+): routed experts run in the latent dim L
    # (D -> L -> experts -> L -> D); shared experts run in the full dim D.
    "L": "LatentMoE latent dim",
    "Fe": "routed-expert hidden size",
    "E": "routed experts",
    "k": "routed experts per token (top-k)",
    "S": "shared experts (fused into one FFN of width S*Fe)",
}


def read_symbols(cfg) -> dict[str, int]:
    """Extract the free hyperparameters from a built KimiK3Model.Config.

    Values that the flavor only passes into sub-module configs (e.g. the MLA
    q_lora_rank) are read back from those sub-configs.
    """
    kda_layer = next(lc for lc in cfg.layers if lc.delta_attention is not None)
    mla_layer = next(lc for lc in cfg.layers if lc.attention is not None)
    dense_layer = next(lc for lc in cfg.layers if lc.feed_forward is not None)
    moe_layer = next(lc for lc in cfg.layers if lc.moe is not None)
    kda, mla, moe = kda_layer.delta_attention, mla_layer.attention, moe_layer.moe

    shared_hidden = moe.shared_experts.w2.in_features
    expert_hidden = moe.routed_experts.w13.out_features
    return {
        "D": cfg.dim,
        "V": cfg.vocab_size,
        "H": kda.num_heads,
        "K": kda.head_dim,
        "W": kda.conv_kernel_size,
        "Rq": mla.wq_a.out_features,
        "Rkv": mla.kv_lora_rank,
        "Dn": mla.qk_nope_head_dim,
        "Dr": mla.qk_rope_head_dim,
        "Dv": mla.v_head_dim,
        "F": dense_layer.feed_forward.w2.in_features,
        "L": moe.routed_down.out_features,
        "Fe": expert_hidden,
        "E": moe.num_experts,
        "k": moe.router.top_k,
        "S": shared_hidden // expert_hidden,
    }


# ---------------------------------------------------------------------------
# Hand-derived parameter formulas
# ---------------------------------------------------------------------------
# Each entry: (local fqn, symbolic shape, category). Shapes are comma-separated
# expressions over the symbols above, evaluated once to get the numeric shape.
#
# Symbol values for the Kimi-K3 flavor (for reading the formulas; the script
# itself reads them from the config):
#   Model-wide:      D = 7168, V = 163840
#   Attention:       H = 96
#   KDA (non-MLA):   K = 128, W = 4
#   MLA:             Rq = 1536, Rkv = 512, Dn = 128, Dr = 64, Dv = 128
#   Non-expert FFN:  F = 33792
#   Experts:         L = 3584, Fe = 3072, E = 896, k = 16, S = 2

# KDA entries are split in two. The five full-width projections are 99.25% of
# a KDA layer (5 x 88.1M = 440.4M of 443.7M); the rest (3.3M/layer, ~230M over
# 69 layers) can be ignored in the memory study. Large entries are ordered input
# projections first, then the output side (output_gate reads x but gates the
# attention output); tiny entries are ordered by size.
# Symbols: D = 7168, H = 96, K = 128, W = 4 (H*K = 12288)
KDA_LARGE_PARAMS = [
    ("delta_attention.q_proj.weight", "H*K, D", "kda"),  # 88.1M
    ("delta_attention.k_proj.weight", "H*K, D", "kda"),  # 88.1M
    ("delta_attention.v_proj.weight", "H*K, D", "kda"),  # 88.1M
    ("delta_attention.output_gate.weight", "H*K, D", "kda"),  # 88.1M
    ("delta_attention.output_proj.weight", "D, H*K", "kda"),  # 88.1M
]

KDA_TINY_PARAMS = [
    ("delta_attention.forget_b.weight", "H*K, K", "kda_tiny"),  # 1.57M
    ("delta_attention.forget_a.weight", "K, D", "kda_tiny"),  # 918K
    ("delta_attention.beta.weight", "H, D", "kda_tiny"),  # 688K
    ("delta_attention.q_conv.weight", "H*K, 1, W", "kda_tiny"),  # 49K
    ("delta_attention.k_conv.weight", "H*K, 1, W", "kda_tiny"),  # 49K
    ("delta_attention.v_conv.weight", "H*K, 1, W", "kda_tiny"),  # 49K
    ("delta_attention.dt_bias", "H, K", "kda_tiny"),  # 12K
    ("delta_attention.output_norm.weight", "K", "kda_tiny"),  # 128
    ("delta_attention.A_log", "H", "kda_tiny"),  # 96
]

KDA_PARAMS = KDA_LARGE_PARAMS + KDA_TINY_PARAMS

# MLA entries are split in two. Unlike KDA, the low-rank q/kv projections are
# not negligible (56.0M, 24% of the layer), so every projection is "large";
# only the two norms (2K/layer) can be ignored. Large entries are ordered input
# projections first (q path, then kv path), then the output side (gate reads x
# but gates the attention output).
# Symbols: D = 7168, H = 96, Rq = 1536, Rkv = 512, Dn = 128, Dr = 64, Dv = 128
MLA_LARGE_PARAMS = [
    ("attention.wq_a.weight", "Rq, D", "mla"),  # 11.0M
    ("attention.wq_b.weight", "H*(Dn+Dr), Rq", "mla"),  # 28.3M
    ("attention.wkv_a.weight", "Rkv+Dr, D", "mla"),  # 4.1M
    ("attention.wkv_b.weight", "H*(Dn+Dv), Rkv", "mla"),  # 12.6M
    ("attention.gate.weight", "H*Dv, D", "mla"),  # 88.1M
    ("attention.wo.weight", "D, H*Dv", "mla"),  # 88.1M
]

MLA_TINY_PARAMS = [
    ("attention.q_norm.weight", "Rq", "mla_tiny"),  # 1536
    ("attention.kv_norm.weight", "Rkv", "mla_tiny"),  # 512
]

MLA_PARAMS = MLA_LARGE_PARAMS + MLA_TINY_PARAMS

# Symbols: D = 7168, F = 33792
DENSE_FFN_PARAMS = [
    ("feed_forward.w13.weight", "2, F, D", "dense_ffn"),  # 484.4M
    ("feed_forward.w2.weight", "D, F", "dense_ffn"),  # 242.2M
]

# Symbols:
#   D  = 7168  decoder hidden size
#   L  = 3584  LatentMoE latent dim (routed experts run at this width)
#   Fe = 3072  routed-expert hidden size
#   E  = 896   routed experts
#   S  = 2     shared experts (fused into one FFN of width S*Fe = 6144)
MOE_PARAMS = [
    ("moe.router.gate.weight", "E, D", "moe_router"),  # 6.4M
    ("moe.routed_down.weight", "L, D", "moe_latent"),  # 25.7M
    ("moe.routed_experts.w13.weight", "E, 2, Fe, L", "moe_routed"),  # 19.73B
    ("moe.routed_experts.w2.weight", "E, L, Fe", "moe_routed"),  # 9.87B
    ("moe.routed_norm.weight", "L", "moe_latent"),  # 3584
    ("moe.routed_up.weight", "D, L", "moe_latent"),  # 25.7M
    ("moe.shared_experts.w13.weight", "2, S*Fe, D", "moe_shared"),  # 88.1M
    ("moe.shared_experts.w2.weight", "D, S*Fe", "moe_shared"),  # 44.0M
]

# Symbols: D = 7168
BLOCK_NORM_PARAMS = [
    ("attention_norm.weight", "D", "norm"),  # 7168
    ("ffn_norm.weight", "D", "norm"),  # 7168
]

# Attention Residuals: a depth-wise softmax over earlier block outputs, scored
# by a D -> 1 projection after an RMSNorm. Layer 0 has no attention residual.
# Symbols: D = 7168
ATTN_RES_PARAMS = [
    ("attention_res_norm.weight", "D", "attn_res"),  # 7168
    ("attention_res_proj.weight", "1, D", "attn_res"),  # 7168
]
FFN_RES_PARAMS = [
    ("ffn_res_norm.weight", "D", "attn_res"),  # 7168
    ("ffn_res_proj.weight", "1, D", "attn_res"),  # 7168
]

# Symbols: D = 7168, V = 163840
GLOBAL_PARAMS = [
    ("tok_embeddings.weight", "V, D", "embedding"),  # 1.17B
    ("norm.weight", "D", "norm"),  # 7168
    ("output_res_norm.weight", "D", "attn_res"),  # 7168
    ("output_res_proj.weight", "1, D", "attn_res"),  # 7168
    ("lm_head.weight", "V, D", "lm_head"),  # 1.17B
]


def decoder_layer_params(cfg, layer_id: int) -> list[tuple[str, str, str]]:
    """Formula entries for one decoder layer, keyed by the layer's structure."""
    lc = cfg.layers[layer_id]
    entries = []
    entries += MLA_PARAMS if lc.attention is not None else KDA_PARAMS
    entries += DENSE_FFN_PARAMS if lc.feed_forward is not None else MOE_PARAMS
    entries += BLOCK_NORM_PARAMS
    if lc.attention_res_norm is not None:
        entries += ATTN_RES_PARAMS
    entries += FFN_RES_PARAMS
    return entries


@dataclass(frozen=True, slots=True)
class Expected:
    fqn: str
    formula: str
    shape: tuple[int, ...]
    category: str

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


def evaluate(formula: str, symbols: dict[str, int]) -> tuple[int, ...]:
    return tuple(
        int(eval(term, {"__builtins__": {}}, symbols)) for term in formula.split(",")
    )


def expected_params(cfg, symbols: dict[str, int]) -> dict[str, Expected]:
    out: dict[str, Expected] = {}

    def add(fqn: str, formula: str, category: str) -> None:
        assert fqn not in out, f"duplicate formula for {fqn}"
        out[fqn] = Expected(fqn, formula, evaluate(formula, symbols), category)

    for local, formula, category in GLOBAL_PARAMS:
        add(local, formula, category)
    for layer_id in range(len(cfg.layers)):
        for local, formula, category in decoder_layer_params(cfg, layer_id):
            add(f"layers.{layer_id}.{local}", formula, category)
    return out


# ---------------------------------------------------------------------------
# Verification against the real module tree
# ---------------------------------------------------------------------------


def verify(expected: dict[str, Expected], model: torch.nn.Module) -> bool:
    actual = {name: tuple(p.shape) for name, p in model.named_parameters()}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name].shape != actual[name]
    )
    for name in missing:
        print(f"  MISSING    {name}: formula {expected[name].formula}")
    for name in unexpected:
        print(f"  UNEXPECTED {name}: shape {actual[name]} has no formula")
    for name in mismatched:
        e = expected[name]
        print(
            f"  MISMATCH   {name}: formula ({e.formula}) = {e.shape}, "
            f"actual {actual[name]}"
        )
    expected_total = sum(e.numel for e in expected.values())
    actual_total = sum(p.numel() for p in model.parameters())
    ok = not (missing or unexpected or mismatched) and expected_total == actual_total
    status = "OK" if ok else "FAILED"
    print(
        f"  {status}: {len(actual)} parameter tensors, "
        f"formula total {expected_total:,} vs module total {actual_total:,}"
    )
    return ok


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_count(n: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.3f}{unit}"
    return f"{n:.0f}"


def compress_ranges(ids: list[int]) -> str:
    """[1, 2, 4, 5, 6, 8] -> '1-2,4-6,8'."""
    parts, start = [], ids[0]
    for prev, cur in zip(ids, ids[1:] + [None]):
        if cur != prev + 1:
            parts.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = cur
    return ",".join(parts)


def print_table(entries: list[Expected], title: str, *, multiplier: int = 1) -> None:
    print(f"\n{title}")
    print(f"  {'parameter':<42} {'formula':<22} {'shape':<26} {'numel':>14}")
    total = 0
    for e in entries:
        total += e.numel
        print(f"  {e.fqn:<42} {e.formula:<22} {str(e.shape):<26} {e.numel:>14,}")
    suffix = (
        f"   (x{multiplier} = {fmt_count(total * multiplier)})"
        if multiplier > 1
        else ""
    )
    print(f"  {'per-instance total':<92} {total:>14,} = {fmt_count(total)}{suffix}")


def report_layer_types(cfg, expected: dict[str, Expected]) -> None:
    """Group decoder layers with identical parameter structure."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for layer_id in range(len(cfg.layers)):
        sig = tuple(
            (local, formula)
            for local, formula, _ in decoder_layer_params(cfg, layer_id)
        )
        groups[sig].append(layer_id)

    for sig, ids in groups.items():
        lc = cfg.layers[ids[0]]
        attn = "MLA" if lc.attention is not None else "KDA"
        ffn = "dense FFN" if lc.feed_forward is not None else "LatentMoE"
        rows = [
            replace_fqn(expected[f"layers.{ids[0]}.{local}"], local) for local, _ in sig
        ]
        print_table(
            rows,
            f"== Decoder layer type: {attn} + {ffn}   "
            f"[{len(ids)} layers: {compress_ranges(ids)}]",
            multiplier=len(ids),
        )


def replace_fqn(e: Expected, fqn: str) -> Expected:
    return Expected(fqn, e.formula, e.shape, e.category)


def report_globals(expected: dict[str, Expected]) -> None:
    print_table(
        [expected[f] for f, _, _ in GLOBAL_PARAMS], "== Embedding / head / final norms"
    )


def report_summary(expected: dict[str, Expected], symbols: dict[str, int]) -> None:
    by_cat: dict[str, int] = defaultdict(int)
    for e in expected.values():
        by_cat[e.category] += e.numel
    total = sum(by_cat.values())

    print("\n== Totals by category")
    print(f"  {'category':<14} {'params':>18} {'':>10} {'share':>7} {'bf16 GiB':>10}")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(
            f"  {cat:<14} {n:>18,} {fmt_count(n):>10} {100 * n / total:>6.2f}% "
            f"{2 * n / 2**30:>10.1f}"
        )
    print(
        f"  {'TOTAL':<14} {total:>18,} {fmt_count(total):>10} {100.0:>6.2f}% "
        f"{2 * total / 2**30:>10.1f}"
    )

    # Active params per text token: routed experts count at k/E.
    routed = by_cat["moe_routed"]
    inactive_routed = routed - routed * symbols["k"] // symbols["E"]
    active = total - inactive_routed
    print("\n== Activated parameters per text token (routed experts weighted by k/E)")
    variants = [
        ("all params", active),
        ("- tok_embeddings", active - by_cat["embedding"]),
    ]
    for label, n in variants:
        print(f"  {label:<28} {n:>18,} = {fmt_count(n)}")
    print("  (model card: 2.8T total incl. ~0.4B vision encoder, 104B activated)")


def report_buffers(model: torch.nn.Module) -> None:
    """Buffers are not parameters, but they occupy memory on every rank."""
    agg: dict[tuple[str, tuple, torch.dtype], int] = defaultdict(int)
    for name, b in model.named_buffers():
        local = re.sub(r"(^|\.)layers\.\d+\.", r"\1layers.*.", name)
        agg[(local, tuple(b.shape), b.dtype)] += 1
    print("\n== Buffers (not parameters; listed for the memory study)")
    total_bytes = 0
    for (local, shape, dtype), count in sorted(agg.items()):
        nbytes = count * torch.Size(shape).numel() * dtype.itemsize
        total_bytes += nbytes
        print(
            f"  {local:<58} {str(shape):<14} {str(dtype):<14} x{count:<4} {nbytes / 2**20:>9.1f} MiB"
        )
    print(f"  {'TOTAL':<93} {total_bytes / 2**20:>9.1f} MiB")


# ---------------------------------------------------------------------------
# Persistent state: weights, gradients, optimizer states
# ---------------------------------------------------------------------------
# Copied from _dist_muon_optimizer in torchtitan_recipes/tests/models/kimi_k3.py
# (importing it needs torchvision for the recipe's multimodal collator). The
# OptimizersContainer matches with re.search, first match wins: DistMuon takes
# the matrix projections, AdamW (pattern ".*") everything else.
MUON_PATTERN = (
    r"(?:"
    r"attention\.(?:wq_a|wq_b|wkv_a|wkv_b|gate|wo)\.weight|"
    r"delta_attention\.(?:q_proj|k_proj|v_proj|forget_a|forget_b|output_gate|"
    r"output_proj)\.weight|"
    r"routed_experts\.w(?:13|2)\.weight|"
    r"feed_forward\.w(?:13|2)\.weight|"
    r"moe\.router\.gate\.weight|"
    r"moe\.shared_experts\.w(?:13|2)\.weight|"
    r"moe\.routed_(?:down|up)\.weight"
    r")$"
)

# Persistent state per parameter follows training.dtype, the dtype parameters
# are stored in:
#   - weights: training.dtype (bf16 means no fp32 master copy).
#   - grads: FSDP2 casts each reduced gradient to the stored parameter dtype
#     (FSDPParam.sharded_grad_dtype = orig_dtype). But with PP and several
#     microbatches, every backward but the last runs with gradient sync off
#     (pipelining/stage.py), and FSDP2 accumulates into the unsharded .grad in
#     mixed_precision_reduce (fp32) until the final reduce-scatter. So
#     grad_dtype="float32" (the default here) models that accumulator.
#   - DistMuon: one momentum buffer per parameter, zeros_like(grad) of the
#     reduced sharded grad, so the parameter dtype (its Newton-Schulz
#     workspace is bf16 and transient).
#   - AdamW: exp_avg + exp_avg_sq; moment_dtype="parameter" (default) follows
#     the parameter dtype.
GB200_HBM_BYTES = 192e9


DTYPE_BYTES = {"bfloat16": 2, "float32": 4}
DTYPE_SHORT = {"bfloat16": "bf16", "float32": "fp32"}


def persistent_bytes_per_param(
    training_dtype: str, grad_dtype: str
) -> dict[str, dict[str, int]]:
    b = DTYPE_BYTES[training_dtype]
    g = DTYPE_BYTES[training_dtype if grad_dtype == "param" else grad_dtype]
    return {
        "Muon": {"weights": b, "grads": g, "state": b},
        "AdamW": {"weights": b, "grads": g, "state": 2 * b},
    }


def optimizer_of(fqn: str) -> str:
    return "Muon" if re.search(MUON_PATTERN, fqn) else "AdamW"


def report_persistent_state(
    expected: dict[str, Expected], training_dtype: str, grad_dtype: str
) -> None:
    per_param = persistent_bytes_per_param(training_dtype, grad_dtype)
    by_opt: dict[str, int] = defaultdict(int)
    families: dict[tuple[str, str], int] = defaultdict(int)
    for e in expected.values():
        opt = optimizer_of(e.fqn)
        by_opt[opt] += e.numel
        families[(opt, re.sub(r"^layers\.\d+\.", "layers.*.", e.fqn))] += e.numel

    dt = DTYPE_SHORT[training_dtype]
    gdt = DTYPE_SHORT[training_dtype if grad_dtype == "param" else grad_dtype]
    print(
        f"\n== Persistent state, training.dtype={training_dtype}: "
        f"weights ({dt}) + grads ({gdt}) + optimizer states ({dt})"
    )
    print(
        f"  {'optimizer':<8} {'params':>18} {'weights':>11} {'grads':>11} "
        f"{'opt state':>11} {'total':>11}   bytes/param"
    )
    totals = defaultdict(float)
    for opt in ("Muon", "AdamW"):
        n = by_opt[opt]
        bpp = per_param[opt]
        w, g, s = n * bpp["weights"], n * bpp["grads"], n * bpp["state"]
        for k, v in (("params", n), ("w", w), ("g", g), ("s", s)):
            totals[k] += v
        state = "momentum" if opt == "Muon" else "exp_avg + exp_avg_sq"
        print(
            f"  {opt:<8} {n:>18,} {w / 1e12:>8.3f} TB {g / 1e12:>8.3f} TB "
            f"{s / 1e12:>8.3f} TB {(w + g + s) / 1e12:>8.3f} TB   "
            f"{sum(bpp.values())} ({state})"
        )
    total = totals["w"] + totals["g"] + totals["s"]
    print(
        f"  {'TOTAL':<8} {int(totals['params']):>18,} {totals['w'] / 1e12:>8.3f} TB "
        f"{totals['g'] / 1e12:>8.3f} TB {totals['s'] / 1e12:>8.3f} TB "
        f"{total / 1e12:>8.3f} TB   = {total / GB200_HBM_BYTES:.1f}x one GB200 (192 GB)"
    )

    print("\n  Parameter families per optimizer (params summed over all layers):")
    for (opt, fam), n in sorted(families.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"    {opt:<6} {fam:<52} {n:>18,}")


def skip_flash_attention_activation() -> contextlib.AbstractContextManager:
    """VarlenInnerAttention (the default attn_backend) activates FlashAttention-4
    on Blackwell at build time, which needs the flash_attn package. Building on
    the meta device or under fake tensors runs no kernel, so skip it."""
    return mock.patch(
        "torchtitan.tools.utils.get_cuda_flash_attention_impl", return_value=None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flavor", default="Kimi-K3")
    parser.add_argument("--buffers", action="store_true", help="also list buffers")
    parser.add_argument(
        "--training-dtype",
        default="float32",
        choices=["bfloat16", "float32"],
        help="training.dtype: the dtype parameters (and optimizer states) are "
        "stored in; this study uses fp32 master weights (the test recipe uses "
        "bfloat16)",
    )
    parser.add_argument(
        "--grad-dtype",
        default="float32",
        choices=["float32", "param"],
        help="float32: FSDP's unsharded accumulator under PP microbatching "
        "(mixed_precision_reduce); param: the reduced sharded grad dtype",
    )
    args = parser.parse_args()

    cfg = build_model_config(args.flavor, parallelism=ParallelismConfig())
    cfg = replace(cfg, vision_encoder=None)
    symbols = read_symbols(cfg)
    expected = expected_params(cfg, symbols)

    print(f"Kimi K3 parameter accounting -- flavor {args.flavor}")
    print("\n== Symbols")
    for sym, value in symbols.items():
        print(f"  {sym:<4} = {value:<8} {SYMBOL_DOCS[sym]}")

    report_layer_types(cfg, expected)
    report_globals(expected)
    report_summary(expected, symbols)
    report_persistent_state(expected, args.training_dtype, args.grad_dtype)

    with torch.device("meta"), skip_flash_attention_activation():
        model = cfg.build()
    if args.buffers:
        report_buffers(model)

    print("\n== Verification: formulas vs. meta-built module")
    return 0 if verify(expected, model) else 1


if __name__ == "__main__":
    sys.exit(main())
