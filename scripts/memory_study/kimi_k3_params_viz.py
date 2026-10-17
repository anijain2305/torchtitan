# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the Kimi K3 parameter accounting as a self-contained HTML page.

Reuses the hand-derived formulas in ``kimi_k3_params.py`` and refuses to write
the page unless they match the meta-built module exactly.

The page shows:
  1. Anatomy of the two repeated decoder layers (KDA + LatentMoE and
     MLA + LatentMoE): every weight in dataflow order with its formula, size,
     and a log-scale bar.
  2. Saved activations (from ``kimi_k3_activations.py``): totals by region and
     the composition of one KDA + LatentMoE and one MLA + LatentMoE layer.
  3. Weights (bf16) and saved activations against one GB200 (192 GB): a linear
     share strip plus a log2-axis bar chart in bytes, for the whole model or a
     single layer.
  4. Persistent state: Muon's momentum, fp32 master weights and gradients, and
     the memory-by-kind chart with saved activations.
  5. Rematerialization: no remat vs the trainer default
     (DefaultActivationRematConfig) vs full (FullActivationRematConfig) in the
     memory-by-kind chart, then full remat + DistMoE for the routed experts
     (saved inputs, the routed output full remat keeps, and scratch at capacity
     factor 4), the baseline for every later section.
  6. Sharding over 512 GPUs: per-GPU memory through ZeRO-1, ZeRO-2 and
     ZeRO-3 / FSDP, then FSDP + EP 64, with chunked loss and the peak transient.
  7. Quantization (MXFP8): what it does to weights and saved activations.
  8. Pipeline parallelism: PP 4 / 8 / 16, per-GPU memory on the worst stage.
  9. Context parallelism: CP 1 / 2 / 4 / 8 at the selected context, per-GPU
     memory with DistMoE, scaled from that context's measurement.
  10. Tensor parallelism: TP 2 / 4 (with sequence parallelism) against CP of
     the same degree, sharding dense weights as well as activations.

Activation passes run at each --context-lengths value with one sequence per
microbatch; the page toggles between them. The global batch is fixed at
--global-tokens per step.

Usage:
  PYTHONPATH=. python scripts/memory_study/kimi_k3_params_viz.py \
      [--flavor Kimi-K3] [--context-lengths 8192 32768] [--out scripts/memory_study/kimi_k3_params.html]
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import torch

from kimi_k3_activations import (
    layer_type,
    local_module,
    mxfp8_exclusive_bytes,
    MXFP8_LINEARS,
    region_of,
    run as run_activations,
)
from kimi_k3_params import (
    build_model_config,
    compress_ranges,
    decoder_layer_params,
    DTYPE_SHORT,
    expected_params,
    optimizer_of,
    ParallelismConfig,
    persistent_bytes_per_param,
    read_symbols,
    skip_flash_attention_activation,
    verify,
)

# Color group of each accounting category. "tiny" weights are the ones the
# memory study ignores (see KDA_TINY_PARAMS / MLA_TINY_PARAMS).
GROUP_OF_CATEGORY = {
    "moe_routed": "experts",
    "moe_shared": "experts",
    "kda": "attention",
    "mla": "attention",
    "moe_latent": "other",
    "moe_router": "other",
    "embedding": "other",
    "lm_head": "other",
    "dense_ffn": "other",
    "kda_tiny": "tiny",
    "mla_tiny": "tiny",
    "attn_res": "tiny",
    "norm": "tiny",
}

# Bars of the "where are the parameters" chart: (key, label, color group,
# categories). Every category lands in exactly one bar, so the bars sum to the
# total.
CHART_ROWS = [
    ("routed", "Routed experts", "experts", ["moe_routed"]),
    ("shared", "Shared experts", "experts", ["moe_shared"]),
    ("attention", "Attention", "attention", ["kda", "mla", "kda_tiny", "mla_tiny"]),
    (
        "others",
        "Others",
        "other",
        [
            "moe_latent",
            "embedding",
            "lm_head",
            "dense_ffn",
            "moe_router",
            "attn_res",
            "norm",
        ],
    ),
]
assert sorted(c for *_, cats in CHART_ROWS for c in cats) == sorted(GROUP_OF_CATEGORY)

# Layer anatomy, in dataflow order. Each block holds subsections of local
# parameter names, plus weightless "note" steps for the core ops.
ATTN_RES_BLOCK = {
    "title": "AttnRes mix -> attention input",
    "kind": "glue",
    "items": [
        {"rows": ["attention_res_norm.weight", "attention_res_proj.weight"]},
        {"rows": ["attention_norm.weight"]},
    ],
}
FFN_RES_BLOCK = {
    "title": "AttnRes mix -> FFN input",
    "kind": "glue",
    "items": [
        {"rows": ["ffn_res_norm.weight", "ffn_res_proj.weight"]},
        {"rows": ["ffn_norm.weight"]},
    ],
}
KDA_BLOCK = {
    "title": "KDA attention",
    "kind": "attention",
    "items": [
        {
            "title": "Input projections (D -> H*K)",
            "rows": [
                "delta_attention.q_proj.weight",
                "delta_attention.k_proj.weight",
                "delta_attention.v_proj.weight",
            ],
        },
        {
            "note": "short conv (W=4, depthwise) + gated delta-rule recurrence "
            "(chunk_kda); the H x K x K state is an activation, not a weight"
        },
        {
            "title": "Output side",
            "rows": [
                "delta_attention.output_gate.weight",
                "delta_attention.output_proj.weight",
            ],
        },
        {
            "title": "Tiny: forget gate, beta, convs, decay params, norm",
            "rows": [
                "delta_attention.forget_b.weight",
                "delta_attention.forget_a.weight",
                "delta_attention.beta.weight",
                "delta_attention.q_conv.weight",
                "delta_attention.k_conv.weight",
                "delta_attention.v_conv.weight",
                "delta_attention.dt_bias",
                "delta_attention.output_norm.weight",
                "delta_attention.A_log",
            ],
        },
    ],
}
MLA_BLOCK = {
    "title": "MLA attention",
    "kind": "attention",
    "items": [
        {
            "title": "q path (D -> Rq -> H*(Dn+Dr))",
            "rows": [
                "attention.wq_a.weight",
                "attention.q_norm.weight",
                "attention.wq_b.weight",
            ],
        },
        {
            "title": "kv path (D -> Rkv+Dr -> H*(Dn+Dv))",
            "rows": [
                "attention.wkv_a.weight",
                "attention.kv_norm.weight",
                "attention.wkv_b.weight",
            ],
        },
        {"note": "softmax attention over H heads (no rotary transform in K3)"},
        {
            "title": "Output side",
            "rows": ["attention.gate.weight", "attention.wo.weight"],
        },
    ],
}
MOE_BLOCK = {
    "title": "LatentMoE",
    "kind": "moe",
    "items": [
        {"title": "Router (scores all E experts)", "rows": ["moe.router.gate.weight"]},
        {
            "title": "Routed path: D -> L -> top-k of E experts -> L -> D",
            "rows": [
                "moe.routed_down.weight",
                "moe.routed_experts.w13.weight",
                "moe.routed_experts.w2.weight",
                "moe.routed_norm.weight",
                "moe.routed_up.weight",
            ],
        },
        {
            "title": "Shared experts (S fused, full width D)",
            "rows": [
                "moe.shared_experts.w13.weight",
                "moe.shared_experts.w2.weight",
            ],
        },
    ],
}


# Persistent state per parameter: fp32 master weights (training.dtype; FSDP
# casts to bf16 for compute), fp32 gradients and fp32 optimizer states, which
# follow the parameter dtype. See kimi_k3_params.py.
TRAINING_DTYPE = "float32"
NUM_GPUS = 512
# ChunkedLossWrapper's default; section 6 assumes chunked loss under FSDP.
CHUNKED_LOSS_CHUNKS = 8
# Tokens per optimizer step, fixed across context lengths (2^24: 2048 x 8K,
# 512 x 32K or 256 x 64K sequences).
GLOBAL_TOKENS = 16 * 1024 * 1024
GRAD_DTYPE = "float32"
# Section 9: CP degrees swept at each context length.
CP_DEGREES = (1, 2, 4, 8)
PERSISTENT_KINDS = ("weights", "grads", "state")
ROW_OF_CATEGORY = {c: key for key, _, _, cats in CHART_ROWS for c in cats}


def persistent_scope(entries) -> dict[str, dict[str, float]]:
    """Persistent bytes by kind, split by chart component (routed, shared, ...)."""
    bpp = persistent_bytes_per_param(TRAINING_DTYPE, GRAD_DTYPE)
    out = {kind: defaultdict(int) for kind in PERSISTENT_KINDS}
    for e in entries:
        opt = optimizer_of(e.fqn)
        row = ROW_OF_CATEGORY[e.category]
        for kind in PERSISTENT_KINDS:
            out[kind][row] += e.numel * bpp[opt][kind]
    return {kind: dict(by_row) for kind, by_row in out.items()}


def layer_kind(lc) -> tuple[str, str]:
    attn = "MLA" if lc.attention is not None else "KDA"
    ffn = "dense FFN" if lc.feed_forward is not None else "LatentMoE"
    return attn, ffn


def active_numel(category: str, numel: int, symbols: dict[str, int]) -> int:
    """Per-token active params: routed experts at k/E, embedding lookup at 0."""
    if category == "embedding":
        return 0
    if category == "moe_routed":
        active = Fraction(numel * symbols["k"], symbols["E"])
        assert active.denominator == 1
        return int(active)
    return numel


def scope_rows(entries, symbols: dict[str, int]) -> dict[str, list[dict]]:
    by_cat: dict[str, int] = defaultdict(int)
    for e in entries:
        by_cat[e.category] += e.numel
    out = {}
    for mode in ("stored", "active"):
        rows = []
        for key, label, group, cats in CHART_ROWS:
            n = sum(
                by_cat[c] if mode == "stored" else active_numel(c, by_cat[c], symbols)
                for c in cats
            )
            if n:
                rows.append(
                    {
                        "key": key,
                        "label": label,
                        "group": group,
                        "numel": n,
                    }
                )
        out[mode] = rows
    return out


def layer_card(cfg, expected, symbols, layer_ids: list[int]) -> dict:
    layer_id = layer_ids[0]
    attn, ffn = layer_kind(cfg.layers[layer_id])
    blocks_def = [
        ATTN_RES_BLOCK,
        KDA_BLOCK if attn == "KDA" else MLA_BLOCK,
        FFN_RES_BLOCK,
        MOE_BLOCK,
    ]

    seen = set()
    blocks = []
    for bdef in blocks_def:
        items = []
        for item in bdef["items"]:
            if "note" in item:
                items.append({"note": item["note"]})
                continue
            rows = []
            for local in item["rows"]:
                e = expected[f"layers.{layer_id}.{local}"]
                seen.add(local)
                rows.append(
                    {
                        "name": local.split(".", 1)[1]
                        if local.startswith(("delta_attention.", "attention.", "moe."))
                        else local,
                        "fqn": e.fqn,
                        "formula": e.formula,
                        "shape": list(e.shape),
                        "numel": e.numel,
                        "group": GROUP_OF_CATEGORY[e.category],
                        "active": active_numel(e.category, e.numel, symbols),
                    }
                )
            items.append({"title": item.get("title"), "rows": rows})
        blocks.append({"title": bdef["title"], "kind": bdef["kind"], "items": items})

    # The diagram must show every parameter of the layer exactly once.
    layer_locals = {local for local, _, _ in decoder_layer_params(cfg, layer_id)}
    assert seen == layer_locals, seen ^ layer_locals

    entries = [expected[f"layers.{layer_id}.{local}"] for local in layer_locals]
    return {
        "attn": attn,
        "title": f"{attn} + {ffn}",
        "layer_ids": compress_ranges(layer_ids),
        "count": len(layer_ids),
        "blocks": blocks,
        "scope": scope_rows(entries, symbols),
        "persistent": persistent_scope(entries),
    }


# Saved-activation components, by the module that first saved each storage
# (layer-relative path): (key, label, color group).
ACT_COMPONENTS = [
    ("routed", "Routed experts", "experts"),
    ("shared", "Shared experts", "experts"),
    ("attention", "Attention", "attention"),
    ("dense_ffn", "Dense FFN", "other"),
    ("attn_res", "AttnRes (block-residual mixing)", "other"),
    ("norms", "Norms", "other"),
]


def act_component(local: str) -> str:
    if local.startswith("moe.shared_experts"):
        return "shared"
    if local == "moe" or local.startswith("moe."):
        return "routed"
    if local == "attention" or local.startswith(("attention.", "delta_attention")):
        return "attention"
    if local.startswith("feed_forward"):
        return "dense_ffn"
    if local == "(block)" or "_res_" in local:
        return "attn_res"
    return "norms"


def _saved_activations(
    flavor: str,
    seq_len: int,
    *,
    recompute_materialized: bool,
    remat: str = "none",
    chunked_loss: int = 0,
    mxfp8: str = "off",
):
    cfg, recorder = run_activations(
        flavor,
        seq_len,
        attn_backend="varlen",
        local_compile="default",
        execution="fake",
        fake_compile="aot_eager_decomp_partition",
        recompute_materialized=recompute_materialized,
        remat_policy=remat,
        chunked_loss=chunked_loss,
        mxfp8=mxfp8,
    )
    return cfg, [r for r in recorder.saved.values() if not r.is_param]


def _attn_res_bytes(acts) -> int:
    """AttnRes saves across the model: per-layer mixes plus the head's final mix."""
    return sum(
        r.nbytes
        for r in acts
        if region_of(r.module) not in ("embedding", "loss")
        and act_component(local_module(r.module)) == "attn_res"
    )


def activation_variant(
    flavor: str, seq_len: int, *, remat: str, chunked_loss: int = 0, mxfp8: str = "off"
) -> dict:
    """Saved-activation totals by region for one setting. With chunked loss, one
    chunk's lm_head + loss saves are reported as a transient ("chunk"), not in
    the total."""
    cfg, acts = _saved_activations(
        flavor,
        seq_len,
        recompute_materialized=True,
        remat=remat,
        chunked_loss=chunked_loss,
        mxfp8=mxfp8,
    )
    chunk = sum(r.nbytes for r in acts if region_of(r.module) == "loss chunk")
    exclusive = mxfp8_exclusive_bytes(acts)
    acts = [r for r in acts if region_of(r.module) != "loss chunk"]
    by_region: dict[str, int] = defaultdict(int)
    for r in acts:
        by_region[region_of(r.module)] += r.nbytes
    types: dict[str, list[int]] = defaultdict(list)
    for layer_id in range(len(cfg.layers)):
        types[layer_type(cfg, layer_id)].append(layer_id)
    regions = []
    for name, ids in types.items():
        per_layer = [by_region[f"layers.{i}"] for i in ids]
        regions.append(
            {
                "label": name,
                "bytes": sum(per_layer),
                "count": len(ids),
                "avg": sum(per_layer) / len(ids),
                "min": min(per_layer),
                "max": max(per_layer),
            }
        )
    regions += [
        {
            "label": "Head (AttnRes + norm + lm_head)",
            "bytes": by_region["head"],
            "count": 0,
        },
        {"label": "Loss (cross-entropy)", "bytes": by_region["loss"], "count": 0},
        {"label": "Embedding", "bytes": by_region["embedding"], "count": 0},
    ]
    total = sum(r.nbytes for r in acts)
    assert total == sum(r["bytes"] for r in regions)
    return {
        "seq_len": seq_len,
        "remat": remat,
        "chunked_loss": chunked_loss,
        "mxfp8": mxfp8,
        "total": total,
        "chunk": chunk,
        "mxfp8_exclusive": exclusive,
        "regions": regions,
    }


def activation_data(flavor: str, seq_len: int) -> dict:
    """Saved-for-backward bytes of one forward + loss (training defaults), with
    the partitioner recomputing materialized upcasts; the default partitioner is
    run too, for the AttnRes before/after note."""
    _, acts_default = _saved_activations(flavor, seq_len, recompute_materialized=False)
    cfg, acts = _saved_activations(flavor, seq_len, recompute_materialized=True)
    by_region: dict[str, int] = defaultdict(int)
    by_layer_comp: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in acts:
        region = region_of(r.module)
        by_region[region] += r.nbytes
        if region.startswith("layers."):
            layer_id = int(region.split(".")[1])
            by_layer_comp[layer_id][act_component(local_module(r.module))] += r.nbytes

    types: dict[str, list[int]] = defaultdict(list)
    for layer_id in range(len(cfg.layers)):
        types[layer_type(cfg, layer_id)].append(layer_id)

    def type_summary(name: str) -> dict:
        ids = types[name]
        per_layer = [by_region[f"layers.{i}"] for i in ids]
        comp = []
        for key, label, group in ACT_COMPONENTS:
            n = sum(by_layer_comp[i][key] for i in ids)
            if n:
                comp.append(
                    {"key": key, "label": label, "group": group, "bytes": n / len(ids)}
                )
        return {
            "name": name,
            "count": len(ids),
            "total": sum(per_layer),
            "avg": sum(per_layer) / len(ids),
            "min": min(per_layer),
            "max": max(per_layer),
            "components": comp,
        }

    regions = [{"label": "Embedding", "bytes": by_region["embedding"], "count": 0}]
    summaries = {name: type_summary(name) for name in types}
    for name, s in summaries.items():
        regions.append(
            {
                "label": name,
                "bytes": s["total"],
                "count": s["count"],
                "avg": s["avg"],
                "min": s["min"],
                "max": s["max"],
            }
        )
    regions += [
        {
            "label": "Head (AttnRes + norm + lm_head)",
            "bytes": by_region["head"],
            "count": 0,
        },
        {"label": "Loss (cross-entropy)", "bytes": by_region["loss"], "count": 0},
    ]
    total = sum(r.nbytes for r in acts)
    assert total == sum(r["bytes"] for r in regions)
    return {
        "seq_len": seq_len,
        "total": total,
        "total_default_partitioner": sum(r.nbytes for r in acts_default),
        "attn_res": _attn_res_bytes(acts),
        "attn_res_default_partitioner": _attn_res_bytes(acts_default),
        "regions": regions,
        "kda_layer": summaries["KDA + LatentMoE"],
        "mla_layer": summaries["MLA + LatentMoE"],
    }


def dist_moe_plans(cfg, local_tokens: int) -> dict | None:
    """Dist-MoE's fixed allocations at one context length, from its pure memory
    planner: saved inputs only (both activation-slot controls unset), at scratch
    capacity factors 1 (balanced routing) and 4 (TorchTitan recipes), for no PP
    and PP 4 / 8 / 16 with EP covering the stage. Also the routed experts' bf16
    output per local MoE layer and microbatch in flight: the dist_moe remat
    region is always saved, so full remat persists its output for the bare
    routed_norm consumer (kimi_k3/moe.py), outside dist_moe's planner. None if
    dist_moe is unavailable."""
    try:
        import dist_moe
    except ImportError:
        return None
    moe = next(lc.moe for lc in cfg.layers if lc.moe is not None)
    num_moe_layers = sum(lc.moe is not None for lc in cfg.layers)
    tokens = local_tokens
    out = {"local_tokens": tokens}
    for name, pp in (("nopp", 1), ("pp4", 4), ("pp8", 8), ("pp16", 16)):
        group = NUM_GPUS // pp  # EP covers the stage (EP 64 without PP)
        ep = 64 if pp == 1 else group
        layers = -(-num_moe_layers // pp)
        slots = pp  # stage 0 holds P microbatches in flight under 1F1B
        for factor in (1.0, 4.0):
            plan = dist_moe.plan_memory(
                dist_moe.Config(
                    max_num_local_input_tokens=tokens,
                    hidden_dim=moe.routed_down.out_features,
                    intermediate_dim=moe.routed_experts.w13.out_features,
                    top_k=moe.router.top_k,
                    num_experts=moe.num_experts,
                    max_moe_layers_per_activation_slot=layers,
                    num_activation_slots=slots,
                    device_scratch_capacity_factor=factor,
                ),
                ep_size=ep,
            )
            out[f"{name}_f{int(factor)}"] = {
                "saved": plan.activation_slot_bytes * slots,
                "scratch": plan.device_scratch_bytes,
                "output": tokens * moe.routed_down.out_features * 2 * layers * slots,
            }
    return out


def context_data(flavor: str, seq_len: int) -> dict:
    """Every activation-dependent measurement at one context length (one
    sequence per microbatch)."""
    cl = CHUNKED_LOSS_CHUNKS
    return {
        "seq_len": seq_len,
        "activations": activation_data(flavor, seq_len),
        "remat": {
            "default": activation_variant(flavor, seq_len, remat="default"),
            "full": activation_variant(flavor, seq_len, remat="full"),
        },
        # Section 6 (chunked loss) and section 7 (MXFP8) variants.
        "quant": {
            remat: {
                mx: activation_variant(
                    flavor, seq_len, remat=remat, chunked_loss=cl, mxfp8=mx
                )
                for mx in ("off", "bf16", "all")
            }
            for remat in ("none", "full")
        },
    }


def build_data(cfg, symbols, expected, flavor: str) -> dict:
    by_kind: dict[tuple[str, str], list[int]] = defaultdict(list)
    for layer_id, lc in enumerate(cfg.layers):
        by_kind[layer_kind(lc)].append(layer_id)
    layers = [
        layer_card(cfg, expected, symbols, by_kind[("KDA", "LatentMoE")]),
        layer_card(cfg, expected, symbols, by_kind[("MLA", "LatentMoE")]),
    ]
    model_scope = scope_rows(expected.values(), symbols)
    return {
        "flavor": flavor,
        "symbols": symbols,
        "num_layers": len(cfg.layers),
        "num_tensors": len(expected),
        "dense_layers": compress_ranges(by_kind[("KDA", "dense FFN")]),
        "layers": layers,
        "scopes": {
            "model": model_scope,
            "kda_layer": layers[0]["scope"],
            "mla_layer": layers[1]["scope"],
        },
        "optimizers": {
            "weight_dtype": DTYPE_SHORT[TRAINING_DTYPE],
            "grad_dtype": DTYPE_SHORT[GRAD_DTYPE],
            "bytes_per_param": persistent_bytes_per_param(TRAINING_DTYPE, GRAD_DTYPE),
            "params": {
                opt: sum(
                    e.numel for e in expected.values() if optimizer_of(e.fqn) == opt
                )
                for opt in ("Muon", "AdamW")
            },
        },
        # Parameters of the projection matrices MXFP8 converts, for the whole
        # model and for one KDA + LatentMoE block.
        "mxfp8_linear_params": {
            "model": sum(
                e.numel
                for e in expected.values()
                if any(f"{name}.weight" in e.fqn for name in MXFP8_LINEARS)
            ),
            "kda_block": sum(
                e.numel
                for fqn, e in expected.items()
                if fqn.startswith(f"layers.{by_kind[('KDA', 'LatentMoE')][0]}.")
                and any(f"{name}.weight" in fqn for name in MXFP8_LINEARS)
            ),
        },
        "persistent": {
            "model": persistent_scope(expected.values()),
            "kda_layer": layers[0]["persistent"],
            "mla_layer": layers[1]["persistent"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flavor", default="Kimi-K3")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[8192, 32768],
        help="context lengths (one sequence per microbatch) to measure; the page "
        "gets a toggle between them",
    )
    parser.add_argument(
        "--global-tokens",
        type=int,
        default=GLOBAL_TOKENS,
        help="tokens per optimizer step, fixed across context lengths",
    )
    parser.add_argument(
        "--dist-moe-path",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "dist_moe",
        help="dist_moe checkout used for its memory planner (sections 5 to 9)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("kimi_k3_params.html"),
    )
    args = parser.parse_args()

    cfg = build_model_config(args.flavor, parallelism=ParallelismConfig())
    cfg = replace(cfg, vision_encoder=None)
    symbols = read_symbols(cfg)
    expected = expected_params(cfg, symbols)

    with torch.device("meta"), skip_flash_attention_activation():
        model = cfg.build()
    print("Verification: formulas vs. meta-built module")
    if not verify(expected, model):
        print("Refusing to render: formulas do not match the module.")
        return 1

    data = build_data(cfg, symbols, expected, args.flavor)
    data["global_tokens"] = args.global_tokens
    data["chunked_loss"] = CHUNKED_LOSS_CHUNKS
    data["contexts"] = {
        f"{seq_len // 1024}K": context_data(args.flavor, seq_len)
        for seq_len in args.context_lengths
    }
    sys.path.insert(0, str(args.dist_moe_path))
    # DistMoE plans per context and CP degree; CP 1 is the baseline (sections
    # 5 to 8), the rest feed the CP sweep (section 9).
    data["cp_sweep"] = {
        f"{seq_len // 1024}K": {
            cp: dist_moe_plans(cfg, seq_len // cp) for cp in CP_DEGREES
        }
        for seq_len in args.context_lengths
    }
    html = HTML_TEMPLATE.replace("/*__DATA__*/null", json.dumps(data))
    args.out.write_text(html)
    print(f"Wrote {args.out}")
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kimi K3 -- parameters and activations</title>
<style>
.viz-root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --wash: rgba(11, 11, 11, 0.04);
  --s-experts: #2a78d6;
  --s-attention: #eb6834;
  --s-other: #1baf7a;
  --s-acts: #eda100;
  --s-acts-full: #9a6400;
  --s-pergpu: #52514e;
  --s-weights: #e87ba4;
  --s-grads: #008300;
  --s-optstate: #4a3aa7;
  --s-tiny: #c3c2b7;
  --note-bg: #fdf1d6;
}
html, body { margin: 0; }
.viz-root {
  background: var(--page);
  color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  min-height: 100vh;
}
.wrap { max-width: 1240px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 24px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 24px; font-weight: 700; margin: 48px 0 6px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; max-width: 860px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.topbar { display: flex; justify-content: space-between; align-items: start; gap: 16px; }
button, .seg label {
  font: inherit; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; cursor: pointer;
}
button:hover, .seg label:hover { background: var(--wash); }
:focus-visible { outline: 2px solid var(--s-experts); outline-offset: 2px; }

/* stat tiles */
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12px; }

/* legend */
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 8px 0; color: var(--text-secondary); font-size: 13px; }
.legend .item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

/* layer anatomy */
.layers { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.card h3 { margin: 0; font-size: 16px; font-weight: 600; }
.card .meta { color: var(--text-secondary); font-size: 13px; margin: 2px 0 10px; }
.card .ids { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.strip { display: flex; gap: 2px; height: 14px; margin: 8px 0 4px; }
.strip > div { border-radius: 2px; min-width: 2px; }
.strip-labels { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; gap: 12px; }
.block { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.block.glue { background: transparent; }
.block-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.block-title { font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
.block-total { color: var(--text-secondary); font-size: 12px; white-space: nowrap; }
/* context-length toggle, pinned to the top-right while scrolling */
.ctx-float {
  position: fixed; top: 12px; right: 16px; z-index: 20;
  display: flex; align-items: center; gap: 8px;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 6px 8px 6px 12px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
}
.ctx-float .ctx-label { color: var(--text-secondary); font-size: 13px; }
.ctx-float .ctx-batch { color: var(--text-secondary); font-size: 12px; line-height: 1.35; white-space: nowrap; }
.subhead { font-size: 17px; font-weight: 600; margin: 24px 0 4px; }
.subtitle { color: var(--muted); font-size: 12px; margin: 6px 0 2px; }
.flow { color: var(--muted); font-size: 12px; text-align: center; line-height: 1; padding: 3px 0; }
.flow::before { content: ""; display: block; width: 1px; height: 8px; background: var(--axis); margin: 0 auto 3px; }
.note-step {
  color: var(--text-secondary); font-size: 12px; background: var(--wash);
  border-radius: 6px; padding: 4px 8px; margin: 6px 0;
}
.row {
  display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr) 64px 112px;
  align-items: center; gap: 8px; padding: 2px 4px; border-radius: 4px; cursor: default;
}
.row:hover, .row:focus { background: var(--wash); outline: none; }
.row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .formula { color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .num { text-align: right; font-variant-numeric: tabular-nums; }
.row.tiny .name, .row.tiny .num { color: var(--muted); }
.minibar { height: 8px; background: transparent; position: relative; }
.minibar > div { height: 8px; border-radius: 0 3px 3px 0; }
.tiny-summary { color: var(--muted); font-size: 12px; padding: 2px 4px; }
.toolbar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 4px 0 12px; }
.toolbar label.check { display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); }

/* chart section */
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 16px 8px; }
.seg { display: inline-flex; gap: 0; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg label { border-radius: 0; margin-left: -1px; }
.seg label:first-of-type { border-radius: 6px 0 0 6px; margin-left: 0; }
.seg label:last-of-type { border-radius: 0 6px 6px 0; }
.seg input:checked + label { background: var(--text-primary); color: var(--surface-1); border-color: var(--text-primary); }
.seg input:focus-visible + label { outline: 2px solid var(--s-experts); outline-offset: 2px; }
.chart-title { font-weight: 600; margin: 6px 0 2px; }
.chart-note { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
#bigstrip, #bigstrip4 { display: flex; gap: 2px; height: 28px; margin: 6px 0 4px; }
#bigstrip > div, #bigstrip4 > div { border-radius: 3px; min-width: 2px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; overflow: visible; white-space: nowrap; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
svg .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .rowlabel { fill: var(--text-primary); font-size: 13px; }
svg .vallabel { fill: var(--text-secondary); font-size: 12px; font-variant-numeric: tabular-nums;
  paint-order: stroke; stroke: var(--surface-1); stroke-width: 4px; stroke-linejoin: round; }
svg .hit { fill: transparent; }
svg g.bar:hover .hit, svg g.bar:focus .hit { fill: var(--wash); }
svg g.bar:focus { outline: none; }
/* linear horizontal bars (section 2) */
.hbars { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; align-items: center; }
.hbars .lbl { font-size: 13px; white-space: nowrap; }
.hbars .lbl .muted { color: var(--muted); }
.hbars .track { display: flex; align-items: center; gap: 8px; min-width: 0; border-radius: 4px; padding: 2px 0; }
.hbars .track:hover, .hbars .track:focus { background: var(--wash); outline: none; }
.hbars .bar { height: 14px; border-radius: 0 4px 4px 0; min-width: 2px; flex: none; }
.hbars .val { color: var(--text-secondary); font-size: 12px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.callout {
  margin: 12px 0 4px; padding: 10px 14px; border-radius: 8px; background: var(--wash);
  font-size: 14px;
}
.callout b { font-weight: 600; }
.callout.note {
  background: var(--note-bg); border-left: 4px solid var(--s-acts);
  margin-top: 16px; padding: 12px 16px;
}
.callout.note .note-label { font-weight: 700; margin-right: 6px; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--text-secondary); }
table { border-collapse: collapse; margin-top: 8px; width: 100%; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid var(--grid); font-size: 13px; }
td.r, th.r { text-align: right; font-variant-numeric: tabular-nums; }
/* section 6: cells that changed from the row above */
td.down { background: rgba(12, 163, 12, 0.10); }
td.up { background: rgba(208, 59, 59, 0.10); }
td .change { display: block; font-size: 11px; font-weight: 600; }
td .row-detail { display: block; font-size: 11px; color: var(--text-secondary); }
#zeroTable td.r, #zeroTable th.r, #zeroBufTable td.r, #zeroBufTable th.r,
#quantZeroTable td.r, #quantZeroTable th.r { white-space: nowrap; }
td.down .change { color: #006300; }
td.up .change { color: #b02a2a; }
.foot { color: var(--muted); font-size: 12px; margin-top: 32px; }

/* tooltip */
#tip {
  position: fixed; z-index: 10; pointer-events: none; display: none;
  background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); font-size: 12px; max-width: 360px;
}
#tip .v { font-size: 15px; font-weight: 600; }
#tip .k { color: var(--text-secondary); }
#tip .line { display: flex; align-items: center; gap: 6px; }
#tip .key { width: 12px; height: 2px; border-radius: 1px; display: inline-block; }
@media (max-width: 900px) {
  .layers { grid-template-columns: 1fr; }
  .tiles { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">
  <div class="topbar">
    <div>
      <h1>Kimi K3: parameters and activations</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <div class="ctx-float">
      <span class="ctx-label">Context</span>
      <div class="seg" role="radiogroup" aria-label="Context length" id="ctxToggle"></div>
      <span class="ctx-batch" id="ctxBatch"></span>
    </div>
  </div>

  <div class="tiles" id="tiles"></div>

  <h2>1. The two repeated layers</h2>
  <p class="sub">Every weight of one decoder layer, in dataflow order. Bars use one
  log2 scale shared by both layers (each 2x adds the same length), so the routed
  experts' lead over everything else is visible without flattening the rest.</p>
  <div class="toolbar">
    <div class="legend" id="legend1"></div>
    <label class="check"><input type="checkbox" id="showTiny"> Show tiny weights</label>
  </div>
  <div class="layers" id="layers"></div>

  <h2>2. Saved activations</h2>
  <p class="sub" id="actSub"></p>
  <div class="panel">
    <div class="chart-title">Saved activations by region (linear)</div>
    <p class="chart-note" id="actRegionNote"></p>
    <div class="hbars" id="actRegions"></div>
  </div>
  <div class="layers" id="actLayers" style="margin-top:16px"></div>
  <div class="callout note" id="attnResNote"></div>

  <h2>3. Weights and activations against one GPU</h2>
  <div class="toolbar">
    <div class="seg" role="radiogroup" aria-label="Scope">
      <input type="radio" name="scope" id="sc-model" value="model" checked><label for="sc-model">Whole model</label>
      <input type="radio" name="scope" id="sc-kda" value="kda_layer"><label for="sc-kda">One KDA + MoE layer</label>
      <input type="radio" name="scope" id="sc-mla" value="mla_layer"><label for="sc-mla">One MLA + MoE layer</label>
    </div>
  </div>
  <div class="panel">
    <div class="callout" id="callout"></div>
    <div class="chart-title" id="stripTitle"></div>
    <div id="bigstrip"></div>
    <div class="legend" id="legend2"></div>
    <div class="chart-title" id="chartTitle" style="margin-top:14px"></div>
    <p class="chart-note" id="chartNote"></p>
    <div id="chart"></div>
    <details>
      <summary>Table view</summary>
      <table id="table"></table>
    </details>
  </div>

  <h2>4. Persistent state</h2>
  <p class="sub" id="persistSub"></p>
  <h3 class="subhead">4.1 Muon: extra memory per weight</h3>
  <p class="sub" id="muonText"></p>
  <div class="callout note" id="adamwNote"></div>
  <h3 class="subhead">4.2 fp32 master weights and gradients</h3>
  <p class="sub" id="gradText"></p>
  <h3 class="subhead">4.3 Weights, gradients, optimizer states and activations</h3>
  <div class="toolbar">
    <div class="seg" role="radiogroup" aria-label="Scope">
      <input type="radio" name="scope4" id="sc4-model" value="model" checked><label for="sc4-model">Whole model</label>
      <input type="radio" name="scope4" id="sc4-kda" value="kda_layer"><label for="sc4-kda">One KDA + MoE layer</label>
      <input type="radio" name="scope4" id="sc4-mla" value="mla_layer"><label for="sc4-mla">One MLA + MoE layer</label>
    </div>
  </div>
  <div class="panel">
    <div class="callout" id="callout4"></div>
    <div class="chart-title" id="stripTitle4"></div>
    <div id="bigstrip4"></div>
    <div class="legend" id="legend3"></div>
    <div class="chart-title" id="chartTitle4" style="margin-top:14px"></div>
    <p class="chart-note" id="chartNote4"></p>
    <div id="chart4"></div>
    <details>
      <summary>Table view</summary>
      <table id="table4"></table>
    </details>
  </div>
  <div class="callout note" id="totalNote"></div>

  <h2>5. Rematerialization</h2>
  <p class="sub" id="acSub"></p>
  <p class="sub" id="acSubDm"></p>
  <div class="panel">
    <div class="chart-title" id="acChartTitle"></div>
    <p class="chart-note" id="acChartNote"></p>
    <div class="legend" id="legendAC"></div>
    <div id="acChart"></div>
  </div>
  <div class="callout note" id="acNote"></div>

  <h2>6. Sharding over 512 GPUs: ZeRO / FSDP, EP and chunked loss</h2>
  <p class="sub" id="zeroSub"></p>
  <div class="panel">
    <table id="zeroTable"></table>
    <div class="chart-title" style="margin-top:18px">Peak transient: the FSDP buffer vs the chunked-loss loop</div>
    <p class="sub" id="zeroBufText" style="margin:2px 0 8px"></p>
    <table id="zeroBufTable"></table>
    <div class="chart-title" style="margin-top:16px">Per-GPU memory at each sharding step (log2 axis, bytes)</div>
    <div id="zeroChart"></div>
  </div>

  <h2>7. Quantization (MXFP8)</h2>
  <p class="sub" id="quantSub"></p>
  <h3 class="subhead">7.1 Saved activations: no remat vs full remat</h3>
  <p class="sub" id="quantActsText"></p>
  <div class="panel">
    <table id="quantTable"></table>
  </div>
  <h3 class="subhead">7.2 Per-GPU memory with MXFP8</h3>
  <p class="sub" id="quantZeroText"></p>
  <div class="panel">
    <table id="quantZeroTable"></table>
  </div>
  <div class="callout note" id="zeroNote"></div>

  <h2>8. Pipeline parallelism</h2>
  <p class="sub" id="ppSub"></p>
  <div class="layers">
    <div class="card">
      <h3 id="ppTitle8"></h3>
      <div class="meta" id="ppMeta8"></div>
      <div id="ppChart8"></div>
    </div>
    <div class="card">
      <h3 id="ppTitle16"></h3>
      <div class="meta" id="ppMeta16"></div>
      <div id="ppChart16"></div>
    </div>
  </div>
  <div class="callout note" id="ppNote"></div>

  <h2>9. Context parallelism</h2>
  <p class="sub" id="cpSub"></p>
  <h3 class="subhead">9.1 PP 8 across CP degrees</h3>
  <p class="sub" id="cpPpSub"></p>
  <div class="layers">
    <div class="card">
      <h3 id="cpPpTitle1"></h3>
      <div class="meta" id="cpPpMeta1"></div>
      <div id="cpPpChart1"></div>
    </div>
    <div class="card">
      <h3 id="cpPpTitle2"></h3>
      <div class="meta" id="cpPpMeta2"></div>
      <div id="cpPpChart2"></div>
    </div>
  </div>
  <div class="layers" style="margin-top: 16px">
    <div class="card">
      <h3 id="cpPpTitle4"></h3>
      <div class="meta" id="cpPpMeta4"></div>
      <div id="cpPpChart4"></div>
    </div>
    <div class="card">
      <h3 id="cpPpTitle8"></h3>
      <div class="meta" id="cpPpMeta8"></div>
      <div id="cpPpChart8"></div>
    </div>
  </div>
  <h3 class="subhead">9.2 Every configuration across CP degrees</h3>
  <p class="sub" id="cpTableSub"></p>
  <div class="panel">
    <table id="cpTable"></table>
  </div>
  <div class="callout note" id="cpNote"></div>

  <h2>10. Tensor parallelism</h2>
  <p class="sub" id="tpSub"></p>
  <div class="panel">
    <table id="tpTable"></table>
  </div>
  <div class="callout note" id="tpNote"></div>

  <p class="foot" id="foot"></p>
</div>
<div id="tip" role="tooltip"></div>
</div>

<script>
const DATA = /*__DATA__*/null;

// Context-length toggle: every activation-dependent number comes from C().
let CTX = Object.keys(DATA.contexts)[0];
const C = () => DATA.contexts[CTX];
// One sequence per microbatch; the global batch is fixed in tokens.
const seqsPerStep = () => DATA.global_tokens / C().seq_len;

const GROUPS = [
  { key: "experts", label: "Expert weights", color: "var(--s-experts)" },
  { key: "attention", label: "Attention weights", color: "var(--s-attention)" },
  { key: "other", label: "Other dense weights", color: "var(--s-other)" },
  { key: "tiny", label: "Tiny (ignored)", color: "var(--s-tiny)" },
  { key: "activations", label: "Saved activations", color: "var(--s-acts)" },
  { key: "weights", label: `Weights (${DATA.optimizers.weight_dtype} master)`, color: "var(--s-weights)" },
  { key: "grads", label: `Gradients (${DATA.optimizers.grad_dtype})`, color: "var(--s-grads)" },
  { key: "state", label: `Optimizer states (${DATA.optimizers.weight_dtype})`, color: "var(--s-optstate)" },
  { key: "acts_noac", label: "Saved activations, no remat", color: "var(--s-tiny)" },
  { key: "acts_full", label: "Saved activations, full remat", color: "var(--s-acts-full)" },
  { key: "acts_dm", label: "Activations, full remat + DistMoE", color: "var(--s-experts)" },
  { key: "pergpu", label: "Per-GPU memory", color: "var(--s-pergpu)" },
  { key: "masteropt", label: `Master + optimizer states (${DATA.optimizers.weight_dtype})`, color: "var(--s-optstate)" },
];
const GROUP = Object.fromEntries(GROUPS.map(g => [g.key, g]));

// ---------- formatting ----------
function fmt(n) {
  const units = [["T", 1e12], ["B", 1e9], ["M", 1e6], ["K", 1e3]];
  for (const [u, s] of units) if (n >= s) return (n / s).toPrecision(4) + u;
  return String(n);
}
const exact = n => n.toLocaleString("en-US");
function pct(x) {
  if (x === 0) return "0%";
  if (x < 0.01) return "<0.01%";
  return x.toFixed(2) + "%";
}
// Binary-unit label for 2^e: 2^20 -> "1Mi", 2^33 -> "8Gi".
function binaryUnit(e) {
  const [u, base] = e >= 40 ? ["Ti", 40] : e >= 30 ? ["Gi", 30] : e >= 20 ? ["Mi", 20] : ["Ki", 10];
  return 2 ** (e - base) + u;
}
const log2s = n => "2^" + Math.log2(n).toFixed(2);
const gib = n => (n / 2 ** 30).toFixed(n / 2 ** 30 < 10 ? 2 : 1) + " GiB";
// Bytes in binary units, 4 significant digits: 5.056 TiB, 445.4 GiB, 96.00 MiB.
function fmtBytes(n) {
  for (const [u, s] of [["TiB", 2 ** 40], ["GiB", 2 ** 30], ["MiB", 2 ** 20], ["KiB", 2 ** 10]])
    if (n >= s) return (n / s).toPrecision(4) + " " + u;
  return n + " B";
}
// Decimal units, for comparing against the GB200's 192 GB.
function fmtDecimal(n) {
  for (const [u, s] of [["TB", 1e12], ["GB", 1e9], ["MB", 1e6]])
    if (n >= s) return (n / s).toPrecision(3) + " " + u;
  return n + " B";
}
const GB200_BYTES = 192e9;

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  }
  if (text !== undefined) e.textContent = text;
  return e;
}
const SVGNS = "http://www.w3.org/2000/svg";
function sel(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------- tooltip ----------
const tip = document.getElementById("tip");
function showTip(lines, x, y) {
  tip.replaceChildren();
  for (const ln of lines) {
    const d = el("div", { class: "line" });
    if (ln.color) d.append(el("span", { class: "key", style: `background:${ln.color}` }));
    const t = el("span", { class: ln.cls || "" }, ln.text);
    d.append(t);
    tip.append(d);
  }
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, top) + "px";
}
function hideTip() { tip.style.display = "none"; }
function attachTip(node, linesFn) {
  node.addEventListener("pointermove", ev => showTip(linesFn(), ev.clientX, ev.clientY));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => {
    const r = node.getBoundingClientRect();
    showTip(linesFn(), r.right - 40, r.top + r.height / 2);
  });
  node.addEventListener("blur", hideTip);
}

// ---------- header ----------
function totals(scope, mode) {
  return DATA.scopes[scope][mode].reduce((a, r) => a + r.numel, 0);
}
function groupTotals(rows) {
  const t = Object.fromEntries(GROUPS.map(g => [g.key, 0]));
  for (const r of rows) t[r.group] += r.numel;
  return t;
}
function renderHeader() {
  const S = DATA.symbols;
  const nKda = DATA.layers[0].count + 1, nMla = DATA.layers[1].count;
  document.getElementById("subtitle").textContent =
    `Flavor ${DATA.flavor}, text decoder only (vision encoder excluded). ` +
    `${DATA.num_layers} layers: ${nKda} KDA + ${nMla} MLA attention; ` +
    `layer ${DATA.dense_layers} has a dense FFN, the rest LatentMoE with ` +
    `${S.E} routed experts (top-${S.k}) + ${S.S} shared. Global batch: ${fmt(DATA.global_tokens)} tokens ` +
    `per step = ${exact(seqsPerStep())} sequences of ${exact(C().seq_len)} tokens; one sequence per microbatch.`;
  const stored = totals("model", "stored"), active = totals("model", "active");
  const g = groupTotals(DATA.scopes.model.stored);
  const tiles = [
    ["Total parameters", fmt(stored), `${exact(stored)}; ${gib(2 * stored)} in bf16`],
    ["Activated per token", fmt(active), `${S.k}/${S.E} routed experts; embedding excluded`],
    ["Expert weights", pct(100 * g.experts / stored), `routed + shared, ${fmt(g.experts)}`],
    ["Attention weights", pct(100 * g.attention / stored), `KDA + MLA, ${fmt(g.attention)}`],
  ];
  const box = document.getElementById("tiles");
  box.replaceChildren();
  for (const [label, value, note] of tiles) {
    const t = el("div", { class: "tile" });
    t.append(el("div", { class: "label" }, label), el("div", { class: "value" }, value),
             el("div", { class: "note" }, note));
    box.append(t);
  }
  for (const [id, keys] of [["legend1", ["experts", "attention", "other", "tiny"]],
                            ["legend2", ["experts", "attention", "other", "activations"]],
                            ["legend3", ["grads", "masteropt", "activations"]]]) {
    const lg = document.getElementById(id);
    lg.replaceChildren();
    for (const gr of keys.map(k => GROUP[k])) {
      const it = el("span", { class: "item", "data-group": gr.key });
      it.append(el("span", { class: "swatch", style: `background:${gr.color}` }),
                el("span", {}, gr.label));
      lg.append(it);
    }
  }
}

// ---------- section 1: layer anatomy ----------
const MINI_LO = 6, MINI_HI = 35;  // log2 domain of the mini bars: 64 .. 34B
function miniWidth(n) {
  const f = (Math.log2(n) - MINI_LO) / (MINI_HI - MINI_LO);
  return Math.max(0.02, Math.min(1, f)) * 100;
}

function renderLayers() {
  const showTiny = document.getElementById("showTiny").checked;
  const box = document.getElementById("layers");
  box.replaceChildren();
  for (const L of DATA.layers) {
    const stored = L.scope.stored.reduce((a, r) => a + r.numel, 0);
    const active = L.scope.active.reduce((a, r) => a + r.numel, 0);
    const card = el("div", { class: "card" });
    card.append(el("h3", {}, L.title));
    card.append(el("div", { class: "meta" },
      `${L.count} layers; ${fmt(stored)} params per layer, ${fmt(active)} activated per token`));
    card.append(el("div", { class: "ids" }, `layers ${L.layer_ids}`));

    // attention vs MoE share of this layer (linear)
    const blockTotals = L.blocks.map(b => ({
      b, n: b.items.flatMap(i => i.rows || []).reduce((a, r) => a + r.numel, 0) }));
    const attnN = blockTotals.filter(x => x.b.kind === "attention").reduce((a, x) => a + x.n, 0);
    const moeN = blockTotals.filter(x => x.b.kind === "moe").reduce((a, x) => a + x.n, 0);
    const glueN = stored - attnN - moeN;
    const strip = el("div", { class: "strip" });
    for (const [n, color, label] of [[attnN, "var(--s-attention)", `${L.attn} attention`],
                                     [moeN, "var(--s-experts)", "LatentMoE"],
                                     [glueN, "var(--s-tiny)", "norms + AttnRes"]]) {
      const seg = el("div", { style: `flex:${n};background:${color}` });
      attachTip(seg, () => [{ text: fmt(n), cls: "v" },
                            { text: `${label}: ${pct(100 * n / stored)} of the layer`, cls: "k" }]);
      strip.append(seg);
    }
    card.append(strip);
    const sl = el("div", { class: "strip-labels" });
    sl.append(el("span", {}, `${L.attn} attention ${pct(100 * attnN / stored)} (${fmt(attnN)})`),
              el("span", {}, `LatentMoE ${pct(100 * moeN / stored)} (${fmt(moeN)})`));
    card.append(sl);

    L.blocks.forEach((b, bi) => {
      if (bi > 0) card.append(el("div", { class: "flow mono" }, "x [T, D]"));
      const n = blockTotals[bi].n;
      const blk = el("div", { class: `block ${b.kind}` });
      const head = el("div", { class: "block-head" });
      const title = el("span", { class: "block-title" });
      if (b.kind !== "glue") {
        const c = b.kind === "attention" ? "var(--s-attention)" : "var(--s-experts)";
        title.append(el("span", { class: "swatch", style: `background:${c}` }));
      }
      title.append(el("span", {}, b.title));
      head.append(title, el("span", { class: "block-total" }, `${fmt(n)} - ${pct(100 * n / stored)}`));
      blk.append(head);

      for (const item of b.items) {
        if (item.note) { blk.append(el("div", { class: "note-step" }, item.note)); continue; }
        const allTiny = item.rows.every(r => r.group === "tiny");
        if (allTiny && !showTiny && b.kind !== "glue") {
          const tn = item.rows.reduce((a, r) => a + r.numel, 0);
          blk.append(el("div", { class: "tiny-summary" },
            `+ ${item.rows.length} tiny weights hidden (${fmt(tn)}, ${pct(100 * tn / stored)})`));
          continue;
        }
        if (item.title) blk.append(el("div", { class: "subtitle" }, item.title));
        const hidden = [];
        for (const r of item.rows) {
          if (r.group === "tiny" && !showTiny) { hidden.push(r); continue; }
          blk.append(weightRow(r, stored, L));
        }
        if (hidden.length && b.kind !== "glue") {
          const hn = hidden.reduce((a, r) => a + r.numel, 0);
          blk.append(el("div", { class: "tiny-summary" },
            `+ ${hidden.map(r => r.name.replace(/\.weight$/, "")).join(", ")} hidden (${fmt(hn)})`));
        }
      }
      if (b.kind === "glue" && !showTiny) {
        const rows = b.items.flatMap(i => i.rows || []);
        blk.append(el("div", { class: "tiny-summary" },
          `${rows.length} tiny weights (norms, D -> 1 depth-mixing projections) - ${fmt(n)}`));
      }
      card.append(blk);
    });
    box.append(card);
  }
}

function weightRow(r, layerTotal, L) {
  const row = el("div", { class: "row" + (r.group === "tiny" ? " tiny" : ""), tabindex: "0" });
  row.append(el("span", { class: "name mono" }, r.name),
             el("span", { class: "formula mono" }, r.formula),
             el("span", { class: "num" }, fmt(r.numel)));
  const mb = el("div", { class: "minibar" });
  mb.append(el("div", { style: `width:${miniWidth(r.numel)}%;background:${GROUP[r.group].color}` }));
  row.append(mb);
  attachTip(row, () => {
    const lines = [
      { text: exact(r.numel) + " params", cls: "v" },
      { text: r.fqn.replace(/^layers\.\d+\./, `layers.*.`), color: GROUP[r.group].color },
      { text: `shape (${r.formula}) = [${r.shape.join(", ")}]`, cls: "k" },
      { text: `${log2s(r.numel)} params`, cls: "k" },
      { text: `${pct(100 * r.numel / layerTotal)} of the layer; x${L.count} layers = ${fmt(r.numel * L.count)}`, cls: "k" },
      { text: `${gib(2 * r.numel)} bf16 / ${gib(4 * r.numel)} fp32 per layer`, cls: "k" },
    ];
    if (r.active !== r.numel)
      lines.push({ text: `activated per token: ${fmt(r.active)} (top-${DATA.symbols.k} of ${DATA.symbols.E})`, cls: "k" });
    return lines;
  });
  return row;
}

// ---------- section 2: saved activations ----------
function hbarRow(box, { label, sub, bytes, max, color, tipLines, val }) {
  const lbl = el("span", { class: "lbl" }, label);
  if (sub) { lbl.append(" "); lbl.append(el("span", { class: "muted" }, sub)); }
  const track = el("div", { class: "track", tabindex: "0" });
  // leave room for the value label at the bar end
  track.append(el("div", { class: "bar", style: `width:calc(${(100 * bytes / max).toFixed(3)}% - ${(150 * bytes / max).toFixed(1)}px);background:${color}` }),
               el("span", { class: "val" }, val));
  attachTip(track, () => tipLines);
  box.append(lbl, track);
}

function renderActivations() {
  const A = C().activations;
  const T = A.seq_len;
  document.getElementById("actSub").textContent =
    `Tensors the forward pass keeps for the backward pass, for one microbatch of ` +
    `${exact(T)} tokens with no rematerialization and no parallelism: ${fmtBytes(A.total)} ` +
    `(${fmtBytes(A.total / T)} per token); ${fmtBytes(A.kda_layer.avg)} per KDA + MoE layer and ` +
    `${fmtBytes(A.mla_layer.avg)} per MLA + MoE layer on average.`;
  document.getElementById("actRegionNote").textContent =
    "Per-layer amounts grow with depth: AttnRes keeps a copy of the stack of earlier " +
    "block outputs, which gains one entry every block of layers.";

  const note = document.getElementById("attnResNote");
  note.replaceChildren();
  const b = t => el("b", {}, t);
  const code = t => el("code", { class: "mono" }, t);
  note.append(
    el("span", { class: "note-label" }, "Note:"),
    "Notice that the AttnRes (residual attention) tensors take a significant share: ",
    b(fmtBytes(A.attn_res)), ` (${pct(100 * A.attn_res / A.total)} of saved activations). `,
    "This was higher before -- ", b(fmtBytes(A.attn_res_default_partitioner)),
    " -- because the compiled AttnRes region saved fp32 copies of its inputs for backward. ",
    "Since this model will be memory-constrained, we set the compiler partitioner flag ",
    code("ban_recompute_materialized_backward=False"), ", which recomputes the upcast in ",
    "backward so only the bf16 tensors are saved. That cuts the AttnRes saved memory from ",
    b(fmtBytes(A.attn_res_default_partitioner)), " to ", b(fmtBytes(A.attn_res)),
    ", and the total from ", b(fmtBytes(A.total_default_partitioner)), " to ",
    b(fmtBytes(A.total)), ".");
  const rb = document.getElementById("actRegions");
  rb.replaceChildren();
  const rmax = Math.max(...A.regions.map(r => r.bytes));
  for (const r of A.regions) {
    const lines = [{ text: fmtBytes(r.bytes), cls: "v" },
                   { text: r.label, color: GROUP.activations.color },
                   { text: `${pct(100 * r.bytes / A.total)} of saved activations`, cls: "k" }];
    if (r.count)
      lines.push({ text: `${r.count} layers; ${fmtBytes(r.avg)} per layer on average ` +
                         `(${fmtBytes(r.min)} to ${fmtBytes(r.max)})`, cls: "k" });
    hbarRow(rb, {
      label: r.label, sub: r.count ? `x${r.count}` : "", bytes: r.bytes, max: rmax,
      color: GROUP.activations.color, tipLines: lines,
      val: `${fmtBytes(r.bytes)} - ${pct(100 * r.bytes / A.total)}`,
    });
  }

  const lb = document.getElementById("actLayers");
  lb.replaceChildren();
  const cmax = Math.max(...[A.kda_layer, A.mla_layer].flatMap(L => L.components.map(c => c.bytes)));
  for (const L of [A.kda_layer, A.mla_layer]) {
    const attn = L.name.split(" ")[0];
    const card = el("div", { class: "card" });
    card.append(el("h3", {}, `One ${L.name} layer`));
    card.append(el("div", { class: "meta" },
      `${fmtBytes(L.avg)} saved per layer (average of ${L.count}; ` +
      `${fmtBytes(L.min)} to ${fmtBytes(L.max)}), ${fmtBytes(L.total)} for all ${L.count}`));
    const box = el("div", { class: "hbars" });
    for (const c of [...L.components].sort((a, b) => b.bytes - a.bytes)) {
      const label = c.key === "attention" ? `${attn} attention` : c.label;
      hbarRow(box, {
        label, bytes: c.bytes, max: cmax, color: GROUP[c.group].color,
        tipLines: [{ text: fmtBytes(c.bytes), cls: "v" },
                   { text: label, color: GROUP[c.group].color },
                   { text: `${pct(100 * c.bytes / L.avg)} of the layer (average)`, cls: "k" },
                   { text: `x${L.count} layers = ${fmtBytes(c.bytes * L.count)}`, cls: "k" }],
        val: `${fmtBytes(c.bytes)} - ${pct(100 * c.bytes / L.avg)}`,
      });
    }
    card.append(box);
    lb.append(card);
  }
}

// ---------- section 3: weights (bf16) + activations vs one GPU ----------
const SCOPE_LABEL = { model: "whole model", kda_layer: "one KDA + MoE layer", mla_layer: "one MLA + MoE layer" };
// Fixed across scopes, like the reference study: dividing a size by 2^k (e.g.
// an FSDP degree) always shortens its bar by exactly k gridlines.
const X_LO_EXP = 26, X_HI_EXP = 44;  // 64 MiB .. 16 TiB, shared by sections 3 and 4
const X_LO = 2 ** X_LO_EXP, X_HI = 2 ** X_HI_EXP;

// Shared log2 bar chart: rows [{label, group, bytes}] against the GB200 line.
function drawLogChart(box, rows, total, ariaLabel, tipFn, { showPct = true, compact = false } = {}) {
  box.replaceChildren();
  // compact: for half-width cards -- narrow margins, ticks labeled every 16x
  const W = compact ? (box.clientWidth || 560) : Math.max(640, box.clientWidth || 1000);
  const m = compact ? { l: 120, r: 120, t: 26, b: 44 } : { l: 250, r: 220, t: 26, b: 44 };
  const labelEvery = compact ? 4 : 2;
  const rowH = 32, barH = 18;
  const H = m.t + rows.length * rowH + m.b;
  const svg = sel("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ariaLabel });
  const x = n => m.l + (Math.log2(n) - Math.log2(X_LO)) /
                 (Math.log2(X_HI) - Math.log2(X_LO)) * (W - m.l - m.r);
  // gridlines at every power of 2; even exponents are stronger and labeled
  // "2^e" over the binary-unit value
  for (let e = X_LO_EXP; e <= X_HI_EXP; e++) {
    const xv = x(2 ** e), major = e % labelEvery === 0;
    svg.append(sel("line", { x1: xv, x2: xv, y1: m.t, y2: H - m.b,
                             stroke: major ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }));
    if (!major) continue;
    const t = sel("text", { x: xv, y: H - m.b + 16, "text-anchor": "middle", class: "tick" }, "2");
    t.append(sel("tspan", { "baseline-shift": "super", "font-size": "8px" }, String(e)));
    svg.append(t);
    svg.append(sel("text", { x: xv, y: H - m.b + 30, "text-anchor": "middle", class: "tick" }, binaryUnit(e) + "B"));
  }
  svg.append(sel("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1 }));
  // one GB200: 192 GB of HBM, drawn under the bars; value labels carry a
  // surface-colored halo so they stay legible where they cross it
  const xg = x(GB200_BYTES);
  svg.append(sel("line", { x1: xg, x2: xg, y1: m.t - 6, y2: H - m.b,
                           stroke: "var(--text-primary)", "stroke-width": 1.5 }));
  svg.append(sel("text", { x: xg, y: m.t - 10, "text-anchor": "middle", class: "rowlabel",
                           "font-weight": "600" }, "one GB200: 192 GB"));
  rows.forEach((r, i) => {
    const y0 = m.t + i * rowH, yc = y0 + rowH / 2;
    const g = sel("g", { class: "bar", tabindex: "0", role: "listitem",
                         "aria-label": `${r.label}: ${fmtBytes(r.bytes)} (${pct(100 * r.bytes / total)})` });
    g.append(sel("rect", { class: "hit", x: 0, y: y0 + 1, width: W, height: rowH - 2, rx: 4 }));
    const x1 = Math.max(x(r.bytes), m.l + 2);
    const w = x1 - m.l, rad = Math.min(4, w / 2), yt = yc - barH / 2;
    // square at the baseline, 4px rounded data-end
    g.append(sel("path", {
      d: `M${m.l},${yt} H${x1 - rad} Q${x1},${yt} ${x1},${yt + rad} V${yt + barH - rad} ` +
         `Q${x1},${yt + barH} ${x1 - rad},${yt + barH} H${m.l} Z`,
      fill: GROUP[r.group].color }));
    g.append(sel("text", { x: m.l - 10, y: yc + 4, "text-anchor": "end", class: "rowlabel" }, r.label));
    g.append(sel("text", { x: x1 + 8, y: yc + 4, class: "vallabel" },
                 compact ? fmtDecimal(r.bytes)
                   : `${fmtBytes(r.bytes)} (${log2s(r.bytes)})` + (showPct ? ` - ${pct(100 * r.bytes / total)}` : "")));
    attachTip(g, () => tipFn(r));
    svg.append(g);
  });
  box.append(svg);
}

// Linear share strip + legend percentages, shared by sections 3 and 4.
function drawStrip(strip, legend, rows, total) {
  const gt = {};
  for (const r of rows) gt[r.group] = (gt[r.group] || 0) + r.bytes;
  strip.replaceChildren();
  const stripW = strip.clientWidth || 1000;
  // segments follow the legend's order
  for (const g of [...legend.children].map(it => GROUP[it.dataset.group])) {
    const n = gt[g.key];
    if (!n) continue;
    const seg = el("div", { style: `flex:${n};background:${g.color}` });
    const label = pct(100 * n / total);
    // label inside only when it fits with padding; white text on saturated fills
    if ((n / total) * stripW > label.length * 7.5 + 16) {
      seg.textContent = label;
      seg.style.color = "#ffffff";
    }
    attachTip(seg, () => [{ text: fmtBytes(n), cls: "v" },
                          { text: `${g.label}: ${pct(100 * n / total)}`, color: g.color }]);
    strip.append(seg);
  }
  for (const it of legend.children) {
    const g = it.dataset.group;
    it.style.display = gt[g] ? "" : "none";
    it.lastChild.textContent = `${GROUP[g].label} ${pct(100 * (gt[g] || 0) / total)}`;
  }
}

function weightRows(scope) {
  const A = C().activations;
  const rows = DATA.scopes[scope].stored.map(r => ({
    key: r.key, label: `${r.label} (weights)`, group: r.group, bytes: 2 * r.numel, numel: r.numel,
  }));
  const acts = scope === "model" ? A.total : A[scope].avg;
  rows.push({ key: "activations", label: "Saved activations", group: "activations", bytes: acts });
  return rows.sort((a, b) => b.bytes - a.bytes);
}

function renderChart() {
  const scope = document.querySelector("input[name=scope]:checked").value;
  const A = C().activations;
  const rows = weightRows(scope);
  const total = rows.reduce((a, r) => a + r.bytes, 0);
  const weights = total - rows.find(r => r.key === "activations").bytes;
  const actsBytes = total - weights;
  const actsNote = scope === "model" ? `one ${exact(A.seq_len)}-token sequence`
                                     : `${exact(A.seq_len)} tokens, layer average`;

  // per-layer callout; the whole-model comparison waits for section 4, which
  // adds gradients and optimizer states
  const callout = document.getElementById("callout");
  callout.replaceChildren();
  callout.style.display = scope === "model" ? "none" : "";
  if (scope !== "model") {
    const b = t => el("b", {}, t);
    callout.append(
      `For ${SCOPE_LABEL[scope]}: weights in bf16 (`, b(fmtDecimal(weights)),
      ") plus saved activations (", b(fmtDecimal(actsBytes)), `, ${actsNote}) come to `,
      b(fmtDecimal(total)), " -- ", b(`${(100 * total / GB200_BYTES).toFixed(1)}%`),
      " of one GB200's 192 GB.");
  }

  document.getElementById("stripTitle").textContent =
    `Share of memory, ${SCOPE_LABEL[scope]} (linear): ${fmtBytes(total)} (${fmtDecimal(total)}) total`;
  drawStrip(document.getElementById("bigstrip"), document.getElementById("legend2"), rows, total);

  document.getElementById("chartTitle").textContent =
    `Memory by component, ${SCOPE_LABEL[scope]} (log2 axis, bytes)`;
  document.getElementById("chartNote").textContent =
    "Log2 axis: one gridline per power of 2, labeled every 4x (binary units, 1 GiB = 2^30 bytes). " +
    "The range is fixed across scopes, so dividing by 2^k -- e.g. sharding over FSDP degree 2^k -- " +
    "shortens a bar by exactly k gridlines. Weights at 2 bytes per parameter (bf16).";

  drawLogChart(document.getElementById("chart"), rows, total,
    document.getElementById("chartTitle").textContent, r => {
      const lines = [
        { text: fmtBytes(r.bytes), cls: "v" },
        { text: r.label, color: GROUP[r.group].color },
        { text: `${fmtDecimal(r.bytes)}; ${log2s(r.bytes)} bytes`, cls: "k" },
        { text: `${pct(100 * r.bytes / total)} of the ${SCOPE_LABEL[scope]}'s memory`, cls: "k" },
        { text: `${(r.bytes / GB200_BYTES).toFixed(2)}x one GB200 (192 GB)`, cls: "k" },
      ];
      if (r.numel) lines.push({ text: `${fmt(r.numel)} params x 2 bytes`, cls: "k" });
      else lines.push({ text: actsNote, cls: "k" });
      return lines;
    });

  const table = document.getElementById("table");
  table.replaceChildren();
  const thead = el("tr");
  for (const [h, rr] of [["Component", 0], ["Bytes", 1], ["Binary", 1], ["Share", 1], ["x GB200", 1]])
    thead.append(el("th", rr ? { class: "r" } : {}, h));
  table.append(thead);
  for (const r of [...rows, { label: "Total", bytes: total }]) {
    const tr = el("tr");
    tr.append(el("td", {}, r.label), el("td", { class: "r" }, exact(Math.round(r.bytes))),
              el("td", { class: "r" }, fmtBytes(r.bytes)), el("td", { class: "r" }, pct(100 * r.bytes / total)),
              el("td", { class: "r" }, (r.bytes / GB200_BYTES).toFixed(2)));
    table.append(tr);
  }
}

// ---------- section 4: persistent state ----------
const COMPONENTS = [["routed", "Routed experts"], ["shared", "Shared experts"],
                    ["attention", "Attention"], ["others", "Others"]];
const KIND_DETAIL = {
  weights: `${DATA.optimizers.weight_dtype} master copy (training.dtype); FSDP casts to bf16 for compute`,
  grads: `${DATA.optimizers.grad_dtype}: accumulated across PP microbatches, reduced to the master dtype`,
  state: `${DATA.optimizers.weight_dtype}: Muon momentum (1 per param) + AdamW exp_avg/exp_avg_sq (2 per param)`,
  masteropt: `${DATA.optimizers.weight_dtype} master weights + Muon momentum / AdamW moments; ZeRO-1 shards them together`,
};

function renderPersistentText() {
  const O = DATA.optimizers, B = O.bytes_per_param, wdt = O.weight_dtype;
  const muon = O.params.Muon, adamw = O.params.AdamW, all = muon + adamw;
  const P = DATA.persistent.model;
  const sum = o => Object.values(o).reduce((a, x) => a + x, 0);
  const b = t => el("b", {}, t);
  const code = t => el("code", { class: "mono" }, t);

  document.getElementById("persistSub").textContent =
    "Weights, gradients and optimizer states stay in memory for the whole training step. " +
    "Section 3 counted bf16 weights only; here the weights are the fp32 master copies, plus " +
    "fp32 gradients and fp32 optimizer states.";

  document.getElementById("muonText").replaceChildren(
    b("Muon"), ` (used for ${pct(100 * muon / all)} of parameters, all the projection matrices) `,
    "saves ", b("one momentum buffer per parameter"), ", the same size and dtype as the gradient, ",
    `so ${O.grad_dtype}: `, b(`+${B.Muon.state} bytes per weight`), ", ",
    b(fmtDecimal(muon * B.Muon.state)), " for the model.");

  const state = muon * B.Muon.state + adamw * B.AdamW.state;
  document.getElementById("adamwNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    "AdamW keeps two moments per parameter, ", b("twice"), " Muon's state -- a nice saving from ",
    "Muon. The few parameters Muon can't take (embeddings, LM head, norms, KDA scalars; ",
    `${pct(100 * adamw / all)} of parameters) still use AdamW: `, b(fmtDecimal(adamw * B.AdamW.state)),
    `, ${pct(100 * adamw * B.AdamW.state / state)} of optimizer state, included in the totals below.`);

  const grads = sum(P.grads), weights = sum(P.weights);
  document.getElementById("gradText").replaceChildren(
    "Weights are kept as ", b(`${wdt} master copies`), " (", code("training.dtype=float32"),
    "), cast to bf16 only for compute. Gradients match the master dtype, and with pipeline ",
    "parallelism FSDP also accumulates them in ", code("mixed_precision_reduce=float32"),
    " across microbatches. Both are ", b(`${B.Muon.weights} bytes per weight`), ", ",
    b(fmtDecimal(weights)), " each.");
}

function renderTotalNote() {
  const rows = persistentRows("model");
  const total = rows.reduce((a, r) => a + r.bytes, 0);
  const b = t => el("b", {}, t);
  document.getElementById("totalNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    "In total, fp32 master weights, fp32 gradients, fp32 optimizer states and the saved ",
    "activations of one microbatch (", b(`microbatch size 1: one sequence of ${exact(C().activations.seq_len)} tokens`),
    ") come to ", b(fmtDecimal(total)), ", a monstrous value: ",
    b(`${(total / GB200_BYTES).toFixed(0)}x`), " the 192 GB of HBM on one GB200.");
}

function persistentRows(scope) {
  const P = DATA.persistent[scope];
  const A = C().activations;
  const masteropt = {};
  for (const k of Object.keys(P.weights)) masteropt[k] = P.weights[k] + (P.state[k] || 0);
  const rows = [["grads", P.grads], ["masteropt", masteropt]].map(([kind, byComponent]) => ({
    key: kind, group: kind, label: GROUP[kind].label, byComponent,
    bytes: Object.values(byComponent).reduce((a, x) => a + x, 0),
  }));
  rows.push({ key: "activations", group: "activations", label: "Saved activations",
              bytes: scope === "model" ? A.total : A[scope].avg });
  return rows;
}

function renderPersistent() {
  const scope = document.querySelector("input[name=scope4]:checked").value;
  const A = C().activations;
  const rows = persistentRows(scope);
  const total = rows.reduce((a, r) => a + r.bytes, 0);
  const actsBytes = rows.find(r => r.key === "activations").bytes;
  const persistent = total - actsBytes;
  const actsNote = scope === "model" ? `one ${exact(A.seq_len)}-token sequence`
                                     : `${exact(A.seq_len)} tokens, layer average`;

  const callout = document.getElementById("callout4");
  callout.replaceChildren();
  const ratio = total / GB200_BYTES;
  const b = t => el("b", {}, t);
  callout.append(
    `${SCOPE_LABEL[scope][0].toUpperCase() + SCOPE_LABEL[scope].slice(1)}: persistent state `,
    "(weights + gradients + optimizer states, ", b(fmtDecimal(persistent)),
    ") plus saved activations (", b(fmtDecimal(actsBytes)), `, ${actsNote}) come to `,
    b(fmtDecimal(total)), " -- ",
    b(ratio >= 1 ? `${ratio.toFixed(1)}x` : `${(100 * ratio).toFixed(1)}%`),
    ratio >= 1 ? " the 192 GB of one GB200." : " of one GB200's 192 GB.");

  document.getElementById("stripTitle4").textContent =
    `Share of memory, ${SCOPE_LABEL[scope]} (linear): ${fmtBytes(total)} (${fmtDecimal(total)}) total`;
  drawStrip(document.getElementById("bigstrip4"), document.getElementById("legend3"), rows, total);

  document.getElementById("chartTitle4").textContent =
    `Memory by kind, ${SCOPE_LABEL[scope]} (log2 axis, bytes)`;
  document.getElementById("chartNote4").textContent =
    `Per parameter: gradients ${DATA.optimizers.grad_dtype} (${DATA.optimizers.bytes_per_param.Muon.grads} B); ` +
    `master weights (${DATA.optimizers.bytes_per_param.Muon.weights} B) + optimizer states ` +
    `(Muon ${DATA.optimizers.bytes_per_param.Muon.state} B, AdamW ${DATA.optimizers.bytes_per_param.AdamW.state} B), ` +
    "one category because ZeRO-1 shards them together. " +
    "Same log2 axis as section 3: dividing by 2^k -- e.g. sharding over FSDP degree 2^k -- " +
    "shortens a bar by exactly k gridlines.";

  drawLogChart(document.getElementById("chart4"), rows, total,
    document.getElementById("chartTitle4").textContent, r => {
      const lines = [
        { text: fmtBytes(r.bytes), cls: "v" },
        { text: r.label, color: GROUP[r.group].color },
        { text: `${fmtDecimal(r.bytes)}; ${(r.bytes / GB200_BYTES).toFixed(2)}x one GB200`, cls: "k" },
      ];
      if (r.byComponent) {
        lines.push({ text: KIND_DETAIL[r.key], cls: "k" });
        for (const [key, label] of COMPONENTS)
          if (r.byComponent[key]) lines.push({ text: `${label}: ${fmtBytes(r.byComponent[key])}`, cls: "k" });
      } else {
        lines.push({ text: actsNote, cls: "k" });
      }
      return lines;
    });

  const table = document.getElementById("table4");
  table.replaceChildren();
  const thead = el("tr");
  thead.append(el("th", {}, "Kind"));
  for (const [, label] of COMPONENTS) thead.append(el("th", { class: "r" }, label));
  for (const h of ["Total", "Share", "x GB200"]) thead.append(el("th", { class: "r" }, h));
  table.append(thead);
  for (const r of [...rows, { label: "Total", bytes: total }]) {
    const tr = el("tr");
    tr.append(el("td", {}, r.label));
    for (const [key] of COMPONENTS)
      tr.append(el("td", { class: "r" }, r.byComponent && r.byComponent[key] ? fmtBytes(r.byComponent[key]) : ""));
    tr.append(el("td", { class: "r" }, fmtBytes(r.bytes)), el("td", { class: "r" }, pct(100 * r.bytes / total)),
              el("td", { class: "r" }, (r.bytes / GB200_BYTES).toFixed(2)));
    table.append(tr);
  }
}

// ---------- shared helpers for section 5 ----------
function persistentTotal() {
  const P = DATA.persistent.model;
  return ["weights", "grads", "state"].reduce(
    (a, k) => a + Object.values(P[k]).reduce((x, y) => x + y, 0), 0);
}

function legendItems(box, items) {
  box.replaceChildren();
  for (const [label, color] of items) {
    const it = el("span", { class: "item" });
    it.append(el("span", { class: "swatch", style: `background:${color}` }), el("span", {}, label));
    box.append(it);
  }
}

function tileRow(box, tiles) {
  box.replaceChildren();
  for (const [label, value, note] of tiles) {
    const t = el("div", { class: "tile" });
    t.append(el("div", { class: "label" }, label), el("div", { class: "value" }, value),
             el("div", { class: "note" }, note));
    box.append(t);
  }
}

// DistMoE plan (section 5 onward) for a layout key ("nopp", "pp4", ...), at a
// context, CP degree and scratch capacity factor: saved = the input slot,
// output = the routed output full remat keeps, scratch = device scratch;
// total = all three, counted as activations from section 6 on.
const DM_FACTOR = 4;
function dmPlan(key, { ctx, cp = 1, factor = DM_FACTOR } = {}) {
  const plan = DATA.cp_sweep[ctx || CTX][cp][`${key}_f${factor}`];
  return { ...plan, total: plan.saved + plan.output + plan.scratch };
}

// ---------- section 5: rematerialization ----------
function renderRemat() {
  const no = C().activations, def = C().remat.default, full = C().remat.full;
  const T = no.seq_len;
  const b = t => el("b", {}, t);
  const code = t => el("code", { class: "mono" }, t);
  const kda = r => r.find(x => x.label === "KDA + LatentMoE");
  document.getElementById("acSub").replaceChildren(
    "Rematerialization trades compute for memory: a transformer block keeps less for the ",
    "backward pass and reruns parts of its forward to rebuild the rest. Sections 2 to 4 used ",
    b("no remat"), ". The trainer's ", b("default remat"), " (", code("DefaultActivationRematConfig"),
    ") keeps every model-declared region except the routed experts' w13 projection; ",
    b("full remat"), " (", code("FullActivationRematConfig"), ") keeps only each block's input and ",
    `the routing decisions. At ${exact(T)} tokens the saved activations are `, b(fmtBytes(no.total)),
    ", ", b(fmtBytes(def.total)), ` (${(no.total / def.total).toFixed(1)}x less) and `,
    b(fmtBytes(full.total)), ` (${(no.total / full.total).toFixed(0)}x less). Per KDA + MoE layer: `,
    `${fmtBytes(kda(no.regions).avg)}, ${fmtBytes(kda(def.regions).avg)} and ${fmtBytes(kda(full.regions).avg)}.`);
  const dm = dmPlan("nopp");
  document.getElementById("acSubDm").replaceChildren(
    "The routed experts run on ", b("DistMoE"), ", one fused operation -- token dispatch, grouped expert ",
    "GEMMs and combine -- that manages its own memory. We assume its kernels gain SiTU-GLU (today they ",
    "implement SwiGLU only). At its default (activation-slot controls unset) it saves only each MoE ",
    "layer's input, in a preallocated activation slot, and recomputes every expert intermediate in ",
    "backward. Full remat never replays the DistMoE call (that would repeat dispatch and the EP ",
    "all-to-alls), so autograd also keeps its output, the same [tokens, L] bf16 size. DistMoE also ",
    "preallocates device scratch for one layer's temporaries, sized by ", code("scratch_capacity_factor"),
    `; we use ${DM_FACTOR}, TorchTitan's recipe value, to cover routing imbalance. At ${exact(T)} tokens `,
    "that is ", b(fmtBytes(dm.saved)), " of inputs and ", b(fmtBytes(dm.output)),
    " of outputs on top of full remat's ", fmtBytes(full.total), ", plus ", b(fmtBytes(dm.scratch)),
    " of scratch (", code("dist_moe.plan_memory"), "; its symmetric dispatch/combine buffers are not ",
    "included). ", b("From here on, every number uses full remat + DistMoE"), ", and ",
    b("activations"), " means all three together: full remat's saves, DistMoE's input and output, ",
    "and its scratch.");

  // memory by kind, whole model, at each remat level
  const P = DATA.persistent.model;
  const kind = key => Object.values(P[key]).reduce((a, x) => a + x, 0);
  const rows = [
    { group: "grads", label: GROUP.grads.label, bytes: kind("grads") },
    { group: "masteropt", label: GROUP.masteropt.label, bytes: kind("weights") + kind("state") },
    { group: "acts_noac", label: "Saved activations, no remat", bytes: no.total },
    { group: "activations", label: "Saved activations, default remat", bytes: def.total },
    { group: "acts_full", label: "Saved activations, full remat", bytes: full.total },
    { group: "acts_dm", label: "Full remat + DistMoE", bytes: full.total + dm.total },
  ];
  document.getElementById("acChartTitle").textContent =
    "Memory by kind, whole model, at each remat level (log2 axis, bytes)";
  document.getElementById("acChartNote").textContent =
    "Same axis as sections 3 and 4. The persistent state does not change with remat or DistMoE.";
  legendItems(document.getElementById("legendAC"), rows.map(r => [r.label, GROUP[r.group].color]));
  drawLogChart(document.getElementById("acChart"), rows, 1,
    document.getElementById("acChartTitle").textContent,
    r => [{ text: fmtBytes(r.bytes), cls: "v" }, { text: r.label, color: GROUP[r.group].color },
          { text: `${fmtDecimal(r.bytes)}; ${(r.bytes / GB200_BYTES).toFixed(2)}x one GB200`, cls: "k" }],
    { showPct: false });

  const persist = persistentTotal();
  const tot = v => persist + v.total;
  const base = persist + full.total + dm.total;
  document.getElementById("acNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    "Even full remat, which cuts saved activations ", `${(no.total / full.total).toFixed(0)}x, only moves `,
    "the model total from ", b(fmtDecimal(tot(no))), " to ", b(fmtDecimal(tot(full))),
    ` (default remat: ${fmtDecimal(tot(def))}); with DistMoE it is `, b(fmtDecimal(base)), ", still ",
    b(`${(base / GB200_BYTES).toFixed(0)}x`), " one GB200. The persistent state dominates, so remat ",
    "alone cannot make this fit; the model has to be partitioned.");
}

// ---------- section 6: ZeRO / FSDP sharding over 512 GPUs ----------
const NUM_GPUS = 512, EP_DEGREE = 64;

// Per-GPU rows for section 6 (mxfp8=false) and its MXFP8 redo in section 7.
// MXFP8: with ZeRO-1/2 the full compute copy is kept only in MXFP8 (32x32 scale
// tiles, 1 + 1/1024 B, one copy for both GEMM orientations) for every quantized
// matrix -- dense projections and routed experts; embedding, lm_head, router
// and norms stay bf16. Under FSDP, a dense weight's bf16 all-gather output is
// still alive while its MXFP8 operands are built, so the block buffer holds
// both. Persistent state stays fp32.
const MXFP8_WEIGHT_BYTES = 1 + 1 / 1024;

// opts (sections 9 and 10): ctx = measured context to scale from, seqLen =
// sequence length, cp = context-parallel degree, tp = tensor-parallel degree
// (with sequence parallelism), factor = DistMoE scratch capacity factor. CP and
// SP both split a microbatch's tokens, and saved activations under full remat
// scale linearly with local tokens; TP also splits the dense weights.
function zeroModel(mxfp8, opts = {}) {
  const P = DATA.persistent.model;
  const kind = key => Object.values(P[key]).reduce((a, x) => a + x, 0);
  const N = DATA.optimizers.params.Muon + DATA.optimizers.params.AdamW;
  const MO = kind("weights") + kind("state"), G = kind("grads");
  const ctx = DATA.contexts[opts.ctx || CTX];
  const act = mxfp8 ? ctx.quant.full.bf16 : ctx.quant.full.off;  // full remat + chunked loss
  const T = opts.seqLen || act.seq_len, seqs = DATA.global_tokens / T;
  // One sequence per microbatch: with fewer sequences per step than GPUs, each
  // sequence must span cp GPUs (context parallelism), splitting its tokens.
  const cp = opts.cp || Math.max(1, NUM_GPUS / seqs), tp = opts.tp || 1;
  const shard = cp * tp;  // GPUs splitting one microbatch's tokens
  const accum = seqs * shard / NUM_GPUS;  // local batch: sequential microbatches per DP rank
  const scale = T / act.seq_len / shard;
  const dm = dmPlan("nopp", { ctx: opts.ctx, cp: shard, factor: opts.factor });
  const A = act.total * scale + dm.total;  // full remat + DistMoE (section 5)
  const n = NUM_GPUS, ep = EP_DEGREE, E = DATA.symbols.E;
  const D = DATA.symbols.D, V = DATA.symbols.V, cl = DATA.chunked_loss;
  const layer = DATA.layers[0].scope.stored;
  const layerN = layer.reduce((a, r) => a + r.numel, 0);
  const routedN = layer.find(r => r.key === "routed").numel;
  const denseN = layerN - routedN;
  const linBlock = DATA.mxfp8_linear_params.kda_block;
  const routedModel = DATA.scopes.model.stored.find(r => r.key === "routed").numel;
  const quantModel = DATA.mxfp8_linear_params.model + routedModel;
  // full compute copy (ZeRO-1/2): bf16, or MXFP8 for every quantized matrix;
  // TP splits the dense part
  const computeCopy = mxfp8 ? 2 * (N - quantModel) + MXFP8_WEIGHT_BYTES * quantModel
                            : 2 * ((N - routedModel) / tp + routedModel);
  // block FSDP buffer: gathered bf16 (2 B) + fp32 gradient (4 B) per parameter,
  // plus the MXFP8 operands of the block's dense linears under MXFP8
  const blockBuf = gathered => 6 * gathered + (mxfp8 ? MXFP8_WEIGHT_BYTES * linBlock : 0);
  // under TP the LM head is vocab-parallel: weights, gradient and logits / tp
  const head = [
    ["lm_head bf16 weights, gathered", 2 * V * D / tp],
    ["lm_head fp32 gradient, unsharded", 4 * V * D / tp],
    [`one chunk's saved tensors (T/${cl} tokens)`, act.chunk * scale],
    ["one chunk's fp32 logits gradient", 4 * (T / cl / shard) * V],
    ["hidden-state gradient accumulator, [T, D] fp32", 4 * T / shard * D],
  ];
  const headT = head.reduce((a, [, v]) => a + v, 0);
  const z3buf = blockBuf(denseN / tp + routedN), epbuf = blockBuf(denseN / tp + routedN / ep);
  const stages = [
    ["No sharding", "No sharding", 0, G, MO, 0],
    ["ZeRO-1", "ZeRO-1: shard master weights + optimizer states", computeCopy, G, MO / n, 0],
    ["ZeRO-2", "ZeRO-2: + shard gradients", computeCopy, G / n, MO / n, 0],
    ["ZeRO-3 / FSDP", `ZeRO-3 / FSDP: + shard ${mxfp8 ? "compute" : "bf16"} weights (gathers all ${E} experts)`,
     0, G / n, MO / n, Math.max(z3buf, headT)],
    [`FSDP + EP ${ep}`, `FSDP + EP ${ep} (gathers only its ${E / ep} experts)`,
     0, G / n, MO / n, Math.max(epbuf, headT)],
  ].map(([label, long, p, g, mo, t]) => ({ label, long, p, g, mo, a: A, t, bytes: p + g + mo + A + t, group: "pergpu" }));
  stages[3].detail = `peak transient: block FSDP buffer, ${fmt(layerN)} params gathered`;
  stages[4].detail = `peak transient: the chunked-loss loop (block buffer only ${fmtDecimal(epbuf)})`;
  return { stages, head, headT, z3buf, epbuf, layerN, routedN, denseN, A, n, ep, E, cl, cp, tp, accum };
}

// Per-GPU table; each changed cell is shaded and annotated against `ref`
// (the row above when ref is null).
function zeroTable(table, stages, ref) {
  const b = t => el("b", {}, t);
  table.replaceChildren();
  const head1 = el("tr");
  head1.append(el("th", {}, "Per GPU"));
  for (const h of ["Compute weights", "Gradients", "Master + optimizer", "Activations", "Peak transient", "Total", "x GB200"])
    head1.append(el("th", { class: "r" }, h));
  table.append(head1);
  const cols = st => [st.p, st.g, st.mo, st.a, st.t, st.bytes];
  const change = (v, prev) => {
    if (!prev) return "new";
    if (!v) return "freed";
    const ratio = prev / v;
    if (Math.abs(ratio - Math.round(ratio)) < 1e-6 && ratio >= 2) return `\u00f7${Math.round(ratio)}`;
    return `${v < prev ? "\u2212" : "+"}${fmtDecimal(Math.abs(v - prev))}`;
  };
  stages.forEach((st, i) => {
    const tr = el("tr");
    const labelCell = el("td", {}, st.long);
    if (st.detail) labelCell.append(el("span", { class: "row-detail" }, st.detail));
    tr.append(labelCell);
    const prevRow = ref ? ref[i] : i > 0 ? stages[i - 1] : null;
    const prev = prevRow ? cols(prevRow) : null;
    cols(st).forEach((v, j) => {
      const changed = prev && Math.abs(v - prev[j]) > 1e-6 * Math.max(v, prev[j]);
      const dir = changed ? (v < prev[j] ? "down" : "up") : "";
      const td = el("td", { class: `r ${dir}` });
      const value = v ? fmtDecimal(v) : "-";
      td.append(j === 5 ? b(value) : value);
      if (changed) td.append(el("span", { class: "change" }, change(v, prev[j])));
      tr.append(td);
    });
    tr.append(el("td", { class: "r" }, (st.bytes / GB200_BYTES).toFixed(2)));
    table.append(tr);
  });
}

function renderZero() {
  const M = zeroModel(false);
  const { stages, head, headT, layerN, routedN, denseN, A, n, ep, E, cl, cp, accum } = M;
  const b = t => el("b", {}, t);

  const batchText = cp > 1
    ? `With ${exact(seqsPerStep())} sequences per step and ${n} GPUs, each sequence must span ${cp} GPUs, ` +
      `so this uses context parallelism ${cp} (each GPU holds 1/${cp} of a sequence's tokens). `
    : `With ${exact(seqsPerStep())} sequences per step over ${n} DP ranks, the local batch is ${accum} ` +
      `sequence${accum === 1 ? "" : "s"}, one sequence per microbatch` +
      (accum > 1 ? " (gradient accumulation, reduce-scattering gradients every microbatch). " : ". ");
  document.getElementById("zeroSub").replaceChildren(
    `Data parallel over ${n} GPUs, no pipeline or tensor parallelism, with chunked loss `,
    `(${cl} chunks). `, batchText, "Activations are per microbatch and not sharded by FSDP; with ",
    "full remat + DistMoE (section 5) they are ", b(fmtDecimal(A)), ". Once the fp32 master weights are sharded (ZeRO-1), each GPU ",
    "needs a full bf16 copy of the weights to compute, until ZeRO-3 shards that too and all-gathers one ",
    `block at a time. EP ${ep} then shards the routed experts ${ep} ways; FSDP covers the remaining `,
    `${n / ep}, so the sharded state per GPU is unchanged.`);

  zeroTable(document.getElementById("zeroTable"), stages, null);

  const state = stages[3].g + stages[3].mo + A;  // identical in the last two rows
  document.getElementById("zeroBufText").replaceChildren(
    "Besides what stays resident, a GPU needs room for the biggest temporary allocation, at a moment ",
    "when nearly all saved activations are still alive. Two candidates: ", b("the FSDP buffer"),
    " -- to compute a block, FSDP all-gathers its bf16 weights and, in backward, builds its full fp32 ",
    "gradient before reduce-scattering it, then frees both -- and ", b("the chunked-loss loop"),
    ", which runs lm_head + loss one chunk at a time after the decoder forward. ",
    b("Without the transient the last two rows would be exactly equal"), `: both hold 1/${n} of `,
    `everything, ${fmtDecimal(state)} per GPU. EP does not change the shards, only what a GPU gathers.`);

  const bt = document.getElementById("zeroBufTable");
  bt.replaceChildren();
  const bh = el("tr");
  for (const [h, r] of [["Transient, per GPU", 0], ["ZeRO-3", 1], [`FSDP + EP ${ep}`, 1]])
    bh.append(el("th", r ? { class: "r" } : {}, h));
  bt.append(bh);
  const row = (label, z3, fe, strong) => {
    const tr = el("tr");
    tr.append(el("td", {}, label));
    for (const v of [z3, fe]) {
      const td = el("td", { class: "r" });
      td.append(strong ? b(fmtDecimal(v)) : fmtDecimal(v));
      tr.append(td);
    }
    bt.append(tr);
  };
  row(`Block FSDP buffer (KDA + MoE): ${fmt(layerN)} vs ${fmt(denseN + routedN / ep)} params gathered x 6 B`,
      M.z3buf, M.epbuf, true);
  for (const [label, v] of head) row(`  ${label}`, v, v, false);
  row("Chunked-loss loop, total", headT, headT, true);
  row("Peak transient = the larger of the two", stages[3].t, stages[4].t, true);

  drawLogChart(document.getElementById("zeroChart"), stages, 1, "Per-GPU memory at each sharding step",
    st => [{ text: fmtDecimal(st.bytes), cls: "v" }, { text: st.long, color: GROUP.pergpu.color },
           { text: `compute weights ${fmtDecimal(st.p)}, grads ${fmtDecimal(st.g)}, master + optimizer ${fmtDecimal(st.mo)}`, cls: "k" },
           { text: `activations ${fmtDecimal(st.a)}, peak transient ${fmtDecimal(st.t)}`, cls: "k" },
           { text: `${(st.bytes / GB200_BYTES).toFixed(2)}x one GB200`, cls: "k" }],
    { showPct: false });

  const last = stages[stages.length - 1], fits = last.bytes <= GB200_BYTES;
  const pctOf = v => `${(100 * v / GB200_BYTES).toFixed(0)}%`;
  document.getElementById("zeroNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    "Quantization does not change the outcome: with full remat it leaves saved activations alone, and ",
    "with ZeRO-3 / FSDP the weight copies it shrinks are short-lived anyway, so MXFP8 is a speed ",
    "optimization here, not a memory one. ",
    `With FSDP + EP ${ep} over ${n} GPUs, chunked loss, full remat and DistMoE, a GPU needs `,
    b(fmtDecimal(last.bytes)), ` at ${CTX}, ${pctOf(last.bytes)} of a GB200's 192 GB. `,
    ...(fits
      ? ["On paper this is a valid setting. It still leaves out other transient memory and workspaces ",
         "-- FSDP prefetching the next block, DistMoE's symmetric dispatch/combine buffers, kernel ",
         "workspaces, allocator fragmentation -- so the real peak can be higher. What we have not ",
         "considered at all is time: FSDP over ", b(`${n} GPUs`), " means every block's all-gather and ",
         "reduce-scatter crosses nodes, which will add a lot of latency to each step."]
      : ["That does not fit: the microbatch's activations alone take ",
         b(fmtDecimal(last.a)), ", and FSDP does not shard them. The next sections look at ",
         "whether pipeline and context parallelism bring it under 192 GB."]));
}

// ---------- section 7: quantization (MXFP8) ----------
function renderQuant() {
  const Q = C().quant, b = t => el("b", {}, t);
  const code = t => el("code", { class: "mono" }, t);
  const P = DATA.persistent.model;
  const kind = key => Object.values(P[key]).reduce((a, x) => a + x, 0);
  const persistent = kind("weights") + kind("grads") + kind("state");
  const bestOf = v => v.off.total - v.off.mxfp8_exclusive * (1 - 33 / 64);  // 1 + 1/32 B instead of 2 B

  document.getElementById("quantSub").replaceChildren(
    "With MXFP8, matmul operands are quantized to float8 (e4m3) with one shared scale per 32 ",
    "elements: about 1 byte per element instead of bf16's 2. TorchTitan applies it with ",
    code("MXFP8LinearConverter"), " for the projection matrices and ", code("MXFP8GroupedLinearConverter"),
    " for the routed experts. The persistent state does not change: master weights, gradients and ",
    `optimizer states stay fp32 (${fmtDecimal(persistent)}). Same setting as section 6: `,
    `${exact(Q.full.off.seq_len)} tokens, chunked loss.`);

  // 7.1 saved activations
  const none = Q.none, full = Q.full;
  document.getElementById("quantActsText").replaceChildren(
    "A linear keeps its input for the weight gradient, in bf16 by default or in MXFP8 if opted in (",
    code("input_activation_format_for_backward=\"mxfp8\""), ", selected per linear with ",
    code("linears_saving_inputs_for_backward_in_mxfp8"), "). That shrinks a saved input only if nothing ",
    "else saves the same bf16 tensor; otherwise the MXFP8 copy is extra. ", b("With no remat"),
    `, ${fmtBytes(none.off.mxfp8_exclusive)} of saved inputs belong to linears alone; opting in exactly `,
    "those (best policy) saves ", b(fmtBytes(none.off.total - bestOf(none))), ". Opting in every linear ",
    "instead adds copies, and the expert grouped GEMMs pad each expert's tokens to a multiple of 32. ",
    "The largest saves -- the routed experts' -- stay bf16 even though their grouped GEMMs run in ",
    "MXFP8: the grouped GEMMs save bf16 operands, with no option to save MXFP8. ",
    b("With full remat"), " a block keeps only its bf16 input, so MXFP8 has nothing to shrink. With ",
    "DistMoE (section 5) the routed experts bypass ", code("MXFP8GroupedLinearConverter"), "; DistMoE has ",
    "its own MXFP8 path, whose block-scaled storage could shrink its saved-input slot. That is not ",
    "modeled here.");

  const table = document.getElementById("quantTable");
  table.replaceChildren();
  const head = el("tr");
  for (const [h, r] of [["Saved activations per GPU", 0], ["bf16 (no MXFP8)", 1],
                        ["MXFP8, linears save bf16", 1], ["MXFP8, all linears save MXFP8", 1],
                        ["MXFP8, best policy", 1]])
    head.append(el("th", r ? { class: "r" } : {}, h));
  table.append(head);
  for (const [remat, label] of [["none", "No remat"], ["full", "Full remat"]]) {
    const v = Q[remat], base = v.off.total;
    const tr = el("tr");
    tr.append(el("td", {}, label));
    [base, v.bf16.total, v.all.total, bestOf(v)].forEach((x, i) => {
      const dir = i === 0 || Math.abs(x - base) < 1e-6 * base ? "" : x < base ? "down" : "up";
      const td = el("td", { class: `r ${dir}` });
      td.append(fmtBytes(x));
      if (i > 0) {
        const d = x - base;
        td.append(el("span", { class: "change" },
          Math.abs(d) < 1e-6 * base ? "no change"
            : `${d < 0 ? "\u2212" : "+"}${fmtBytes(Math.abs(d))} (${Math.abs(100 * d / base) < 0.1 ? "<0.1" : (100 * Math.abs(d) / base).toFixed(1)}%)`));
      }
      tr.append(td);
    });
    table.append(tr);
  }

  // 7.2 per-GPU table vs section 6
  const base6 = zeroModel(false).stages, mx = zeroModel(true);
  document.getElementById("quantZeroText").replaceChildren(
    "Section 6's table redone with MXFP8 (linears save bf16 inputs, TorchTitan's default), each cell ",
    "compared with the same row of section 6. With ZeRO-1/2 every GPU keeps a full compute copy; it can ",
    "be held in MXFP8 for every quantized matrix (dense projections and routed experts, about 1 byte per ",
    "parameter; embedding, LM head, router and norms stay bf16), so it roughly halves. ",
    b("With ZeRO-3 / FSDP there isn't much benefit"), ": the gathered weights are released right after ",
    "each block uses them, so a smaller copy only shrinks a short-lived buffer -- and since FSDP gathers ",
    "in bf16 before quantizing, the buffer briefly holds both copies and grows slightly.");
  zeroTable(document.getElementById("quantZeroTable"), mx.stages, base6);

}

// ---------- section 8: pipeline parallelism ----------
// Worst stage of a P-way pipeline over NUM_GPUS GPUs: ceil(L / P) KDA + MoE-sized
// layers, sharded inside the stage over g = NUM_GPUS / P GPUs with EP g covering
// the stage (each GPU owns E / g experts outright). Stage 0 holds P microbatches
// in flight under 1F1B. What a GPU keeps across the step's microbatches:
//   ZeRO-1 (FSDP's default under PP: no resharding after backward, no gradient
//     sync until the last microbatch): bf16 compute copy (2 B) and fp32
//     gradients (4 B) of the stage's dense part plus its local experts.
//   ZeRO-2: gradients reduce-scattered every microbatch; weights stay gathered.
//   ZeRO-3 (resharding every microbatch): one block gathered at a time.
function ppModel(P, opts = {}) {
  const L = DATA.num_layers, g = NUM_GPUS / P, E = DATA.symbols.E;
  const layer = DATA.layers[0].scope.stored;
  const layerN = layer.reduce((a, r) => a + r.numel, 0);
  const routedN = layer.find(r => r.key === "routed").numel;
  const denseN = layerN - routedN;
  const Ls = Math.ceil(L / P), Ns = Ls * layerN;
  const act = DATA.contexts[opts.ctx || CTX].quant.full.off;  // full remat + chunked loss
  const T = opts.seqLen || act.seq_len, seqs = DATA.global_tokens / T;
  // M = sequences per pipeline replica per step; 1F1B needs M >= P, which may
  // take context parallelism (fewer, wider replicas).
  const cp = opts.cp || Math.max(1, Math.ceil(NUM_GPUS / seqs)), tp = opts.tp || 1;
  const shard = cp * tp;  // GPUs splitting one microbatch's tokens
  const M = seqs * P * shard / NUM_GPUS;
  const bubble = (P - 1) / (M + P - 1);
  const head = act.regions.filter(r => r.label.startsWith("Head") || r.label === "Embedding")
                          .reduce((a, r) => a + r.bytes, 0);
  // P microbatches x Ls layers, 1/(cp x tp) of each, plus DistMoE (planned for
  // the stage's MoE layers and its P microbatches)
  const dm = dmPlan(`pp${P}`, { ctx: opts.ctx, cp: shard, factor: opts.factor });
  const A = P * Ls * (act.total - head) / L * (T / act.seq_len) / shard + dm.total;
  const B = DATA.optimizers.bytes_per_param.Muon;  // fp32 master / grads / state
  const MO = (B.weights + B.state) * Ns / g, Gs = B.grads * Ns / g;
  const local = Ls * (denseN / tp + routedN / g);  // stage dense part (TP-split) + local experts
  const block = denseN / tp + routedN / g;          // one block's dense part + local experts
  const rows = [
    ["No sharding", "No sharding", 0, B.grads * Ns, (B.weights + B.state) * Ns, 0],
    ["ZeRO-1 (default)", "ZeRO-1: FSDP's default in PP", 2 * local, B.grads * local, MO, 0],
    ["ZeRO-2", "ZeRO-2: reduce-scatter gradients every microbatch", 2 * local, Gs, MO, 0],
    ["ZeRO-3", "ZeRO-3: reshard weights and gradients every microbatch", 0, Gs, MO, 6 * block],
  ].map(([label, long, p, gr, mo, t]) => ({ label, long, p, g: gr, mo, a: A, t, bytes: p + gr + mo + A + t, group: "pergpu" }));
  return { P, g, Ls, Ns, A, rows, experts: E / g, local, block, cp, tp, M, bubble };
}

function renderPP() {
  const b = t => el("b", {}, t);
  const m4 = ppModel(4), m8 = ppModel(8), m16 = ppModel(16);
  document.getElementById("ppSub").replaceChildren(
    `We look at three pipeline degrees over the ${NUM_GPUS} GPUs: `, b("PP 4, 8 and 16"),
    ` (${m4.g}, ${m8.g} and ${m16.g} GPUs per stage), each stage sharded with FSDP and EP across its `,
    "GPUs. Larger degrees get worse quickly: with only 93 layers, PP 64 would give some stages 2 layers ",
    "and others 1; finer stages split the 12-layer AttnRes blocks and give stages an uneven mix of MLA ",
    "and KDA layers; and every extra hop carries the AttnRes stack. With the global batch fixed, each ",
    `pipeline gets M = ${exact(seqsPerStep())} x P / ${NUM_GPUS} microbatches per step (one sequence each), `,
    "so the 1F1B bubble, (P - 1) / (M + P - 1), is about the same for every P; shown under each chart. ",
    "The charts show per-GPU memory on the worst stage, stage 0: under 1F1B it holds P microbatches in ",
    "flight, so its activations stay about one full-model microbatch for any P. Under PP, FSDP's default ",
    "keeps the stage's weights gathered and its gradients unsharded until the last microbatch, so it ",
    "behaves like ", b("ZeRO-1"), " within the step; ZeRO-2 and ZeRO-3 shard gradients, and then weights, ",
    "after every microbatch, at the cost of one reduce-scatter (and all-gathers) per microbatch.");

  const tip = r => [{ text: fmtDecimal(r.bytes), cls: "v" }, { text: r.long, color: GROUP.pergpu.color },
                    { text: `compute weights ${fmtDecimal(r.p)}, grads ${fmtDecimal(r.g)}, master + optimizer ${fmtDecimal(r.mo)}`, cls: "k" },
                    { text: `activations ${fmtDecimal(r.a)}, one-block FSDP buffer ${fmtDecimal(r.t)}`, cls: "k" },
                    { text: `${(r.bytes / GB200_BYTES).toFixed(2)}x one GB200`, cls: "k" }];
  for (const m of [m8, m16]) {
    document.getElementById(`ppTitle${m.P}`).textContent = `PP ${m.P}`;
    document.getElementById(`ppMeta${m.P}`).textContent =
      `${m.g} GPUs and up to ${m.Ls} layers (${fmt(m.Ns)} params) per stage; EP ${m.g} = ${m.experts} ` +
      `experts per GPU.${m.cp > 1 ? ` Needs CP ${m.cp} to have enough microbatches.` : ""} ` +
      `Local batch M = ${m.M} sequences (microbatches) per step, 1F1B bubble ${(100 * m.bubble).toFixed(0)}%. ` +
      `Activations ${fmtDecimal(m.A)}. ` +
      `ZeRO-1 (default): ${fmtDecimal(m.rows[1].bytes)}; ZeRO-3: ${fmtDecimal(m.rows[3].bytes)}.`;
    drawLogChart(document.getElementById(`ppChart${m.P}`), m.rows, 1, `PP ${m.P}: per-GPU memory`, tip,
                 { showPct: false, compact: true });
  }

  const pp4Fits = m4.rows[1].bytes <= GB200_BYTES;
  const fitting = [m8, m16].flatMap(m => [[1, "ZeRO-1"], [3, "ZeRO-3"]]
    .filter(([i]) => m.rows[i].bytes <= GB200_BYTES).map(([, z]) => `PP ${m.P} ${z}`));
  document.getElementById("ppNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    `At ${CTX}, `, b(fitting.length ? `${fitting.join(", ")} fit in 192 GB` : "no PP 8 or 16 option fits in 192 GB"),
    fitting.length ? ". " : ": PP shards the persistent state but not the activations of the P microbatches in flight. ",
    pp4Fits ? "PP 4 fits only on paper, with almost no headroom: with "
            : "PP 4 does not fit because memory is so tight: with ",
    `${m4.Ls}-layer stages, FSDP's default `,
    `(ZeRO-1) keeps ${fmtDecimal(6 * m4.local)} of gathered weights and unsharded gradients per GPU even `,
    `with EP ${m4.g}, for `, b(fmtDecimal(m4.rows[1].bytes)),
    ` (${(100 * m4.rows[1].bytes / GB200_BYTES).toFixed(0)}% of 192 GB) on the worst stage. Resharding every `,
    `microbatch (ZeRO-3) would bring it to ${fmtDecimal(m4.rows[3].bytes)}, but with ${m4.g}-GPU stages `,
    "those per-microbatch all-gathers and reduce-scatters cross two 64-GPU groups.");
}

// ---------- section 9: context parallelism ----------
// The configurations swept, with their per-GPU bytes, PP degree and
// microbatches per pipeline (M).
function cpConfigs(opts) {
  const z = zeroModel(false, opts), p8 = ppModel(8, opts), p16 = ppModel(16, opts);
  return [
    { label: "FSDP + EP 64", bytes: z.stages[z.stages.length - 1].bytes, P: 1, M: z.accum },
    { label: "PP 8, ZeRO-1 (default)", bytes: p8.rows[1].bytes, P: 8, M: p8.M },
    { label: "PP 8, ZeRO-3", bytes: p8.rows[3].bytes, P: 8, M: p8.M },
    { label: "PP 16, ZeRO-1 (default)", bytes: p16.rows[1].bytes, P: 16, M: p16.M },
    { label: "PP 16, ZeRO-3", bytes: p16.rows[3].bytes, P: 16, M: p16.M },
  ];
}

const CP_DEGREES = [1, 2, 4, 8];  // CP_DEGREES in Python

// Section 9 sweeps CP at the selected context. A config is valid when every
// replica gets whole microbatches (M >= 1) and, with PP, 1F1B can fill the
// pipeline (M >= P).
function renderCpTable(seqLen) {
  const table = document.getElementById("cpTable");
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, "Per GPU"));
  for (const cp of CP_DEGREES) {
    const th = el("th", { class: "r" }, `CP ${cp}`);
    th.append(el("span", { class: "change", style: "display: block; font-weight: 400" }, `${exact(seqLen / cp)} tokens`));
    head.append(th);
  }
  table.append(head);
  const fits = {};
  const rows = CP_DEGREES.map(cp => cpConfigs({ cp }));
  const rows1 = CP_DEGREES.map(cp => cpConfigs({ cp, factor: 1 }));
  rows[0].forEach((c0, i) => {
    const tr = el("tr");
    tr.append(el("td", {}, c0.label));
    CP_DEGREES.forEach((cp, j) => {
      const c = rows[j][i];
      if (c.M < Math.max(1, c.P)) {
        const td = el("td", { class: "r", style: "color: var(--muted)" }, "--");
        td.append(el("span", { class: "change" },
          c.M < 1 ? `${c.M} sequence per GPU` : `M = ${c.M} < P = ${c.P}`));
        tr.append(td);
        return;
      }
      const t4 = c.bytes, t1 = rows1[j][i].bytes;
      const td = el("td", { class: `r ${t4 > GB200_BYTES ? "up" : "down"}` });
      td.append(el("b", {}, fmtDecimal(t4)));
      const bubble = c.P > 1 ? `; bubble ${(100 * (c.P - 1) / (c.M + c.P - 1)).toFixed(0)}%` : "";
      // the factor-1 line carries its own fit color, which may differ from the cell's
      td.append(el("span", { class: "change", style: `color: ${t1 > GB200_BYTES ? "#b02a2a" : "#006300"}` },
        `factor 1: ${fmtDecimal(t1)} (${t1 > GB200_BYTES ? "over" : "fits"})${bubble}`));
      tr.append(td);
      fits[c0.label] ||= {};
      if (t1 <= GB200_BYTES && !fits[c0.label].f1) fits[c0.label].f1 = cp;
      if (t4 <= GB200_BYTES && !fits[c0.label].f4) fits[c0.label].f4 = cp;
    });
    table.append(tr);
  });
  return fits;
}


// 9.1: section 8's PP 8 chart at each CP degree.
function renderCpPp() {
  const P = 8;
  document.getElementById("cpPpSub").textContent =
    `Section 8's PP ${P} chart at each CP degree; CP 1 is section 8. ` +
    "Only the activation-sized terms move; weights, gradients and optimizer states are the same in " +
    "every chart.";
  const tip = r => [{ text: fmtDecimal(r.bytes), cls: "v" }, { text: r.long, color: GROUP.pergpu.color },
                    { text: `compute weights ${fmtDecimal(r.p)}, grads ${fmtDecimal(r.g)}, master + optimizer ${fmtDecimal(r.mo)}`, cls: "k" },
                    { text: `activations ${fmtDecimal(r.a)}, one-block FSDP buffer ${fmtDecimal(r.t)}`, cls: "k" },
                    { text: `${(r.bytes / GB200_BYTES).toFixed(2)}x one GB200`, cls: "k" }];
  for (const cp of CP_DEGREES) {
    const m = ppModel(P, { cp }), rows = m.rows;
    document.getElementById(`cpPpTitle${cp}`).textContent = `PP ${P}, CP ${cp}`;
    const box = document.getElementById(`cpPpChart${cp}`);
    if (m.M < P) {
      document.getElementById(`cpPpMeta${cp}`).textContent = `M = ${m.M} < P = ${P}: 1F1B cannot fill the pipeline.`;
      box.replaceChildren();
      continue;
    }
    document.getElementById(`cpPpMeta${cp}`).textContent =
      `${exact(C().seq_len / cp)} tokens per microbatch; activations ${fmtDecimal(m.A)}. ` +
      `Local batch M = ${m.M} sequences (microbatches) per step, 1F1B bubble ` +
      `${(100 * m.bubble).toFixed(0)}%. ZeRO-1 (default): ${fmtDecimal(rows[1].bytes)}; ` +
      `ZeRO-3: ${fmtDecimal(rows[3].bytes)}.`;
    drawLogChart(box, rows, 1, `PP ${P}, CP ${cp}: per-GPU memory`, tip, { showPct: false, compact: true });
  }
}

function renderCp() {
  const b = t => el("b", {}, t), code = t => el("code", { class: "mono" }, t);
  const sub = document.getElementById("cpSub");
  sub.replaceChildren(
    "Context parallelism (CP) splits each sequence's tokens across c GPUs. The batch and GPU count stay ",
    `fixed, so every GPU still processes ${exact(DATA.global_tokens / NUM_GPUS)} tokens per step: each `,
    "microbatch shrinks to T/c tokens and there are c times as many of them. CP folds into FSDP's ",
    "storage axis (dp_shard x cp), so ", b("persistent state per GPU does not change"), ". What shrinks is ",
    "everything sized by local tokens: ", b("the activations"), " (full remat + DistMoE)", ". The pipeline also gets c times more microbatches, which shrinks the bubble. The costs ",
    "are time, not memory: attention communication (an all-gather of each MLA layer's K and V, recurrent-state ",
    "passing between ranks for KDA), c times more weight all-gathers under ZeRO-3, and smaller expert ",
    "GEMMs. Activations are scaled from the full-remat measurement at CP 1; every tensor ",
    "full remat saves grows linearly with tokens, so the scaling is exact.");
  renderCpPp();
  const T = C().seq_len, fits = renderCpTable(T), local = seqsPerStep() / NUM_GPUS;
  document.getElementById("cpTableSub").textContent =
    "Each cell is the per-GPU total (DistMoE scratch factor 4); below it, the total at factor 1 and the " +
    `PP bubble. The local batch per DP rank is ${local === 1 ? "" : `${local} x `}CP sequences, times ` +
    "PP with a pipeline. Per-GPU memory does not depend on it; it sets the number of microbatches, and " +
    "with them the bubble.";
  const lowest = Object.entries(fits).map(([label, v]) =>
    `${label}: ${v.f4 ? `CP ${v.f4}` : "none"}`).join("; ");
  const allAtOne = Object.values(fits).every(v => v.f4 === 1);
  // A sequence twice as long at twice the CP has the same local tokens and M.
  const longer = 2 * T;
  document.getElementById("cpNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    ...(allAtOne
      ? [`At ${CTX} every configuration already fits at CP 1, even at scratch factor 4, so CP only `,
         "buys a smaller PP bubble at the cost of communication. "]
      : ["Lowest CP that fits in 192 GB: ", b(lowest), ". ",
         "CP buys memory with communication, so the choice between these comes down to step time. "]),
    `A ${longer / 1024}K context (${exact(DATA.global_tokens / longer)} sequences per step) at CP 2c has `,
    `exactly the same local tokens and microbatches as ${CTX} at CP c, so its numbers are this table `,
    "shifted one column right.");
}

// ---------- section 10: tensor parallelism ----------
const TP_COLUMNS = [{ cp: 2 }, { tp: 2 }, { cp: 4 }, { tp: 4 }, { tp: 2, cp: 2 }];
const tpColumnLabel = c => [c.tp > 1 ? `TP ${c.tp}` : "", c.cp > 1 ? `CP ${c.cp}` : ""].filter(Boolean).join(" x ");

function renderTp() {
  const b = t => el("b", {}, t);
  document.getElementById("tpSub").replaceChildren(
    "With sequence parallelism, TP splits a microbatch's tokens like CP, so activations shrink the same ",
    "way; it also splits the dense weights, which ", b("helps only where they stay gathered: PP with ZeRO-1"),
    ". The cost is ", b("more communication than CP"), ": activations are all-gathered and reduce-scattered ",
    "around every attention and shared-expert block. Each cell is the per-GPU total; below it, the change ",
    "against CP of the same degree and the PP bubble.");

  const table = document.getElementById("tpTable");
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, "Per GPU"));
  for (const c of TP_COLUMNS) head.append(el("th", { class: "r" }, tpColumnLabel(c)));
  table.append(head);
  const cols = TP_COLUMNS.map(c => cpConfigs(c));
  const sameCp = TP_COLUMNS.map(c => cpConfigs({ cp: (c.cp || 1) * (c.tp || 1) }));
  cols[0].forEach((c0, i) => {
    const tr = el("tr");
    tr.append(el("td", {}, c0.label));
    TP_COLUMNS.forEach((col, j) => {
      const c = cols[j][i];
      if (c.M < Math.max(1, c.P)) {
        const td = el("td", { class: "r", style: "color: var(--muted)" }, "--");
        td.append(el("span", { class: "change" }, c.M < 1 ? `${c.M} sequence per GPU` : `M = ${c.M} < P = ${c.P}`));
        tr.append(td);
        return;
      }
      const td = el("td", { class: `r ${c.bytes > GB200_BYTES ? "up" : "down"}` });
      td.append(b(fmtDecimal(c.bytes)));
      const parts = [];
      if ((col.tp || 1) > 1) {
        const d = c.bytes - sameCp[j][i].bytes;
        parts.push(`${d < 0 ? "\u2212" : "+"}${fmtDecimal(Math.abs(d))} vs CP ${(col.cp || 1) * col.tp}`);
      }
      if (c.P > 1) parts.push(`bubble ${(100 * (c.P - 1) / (c.M + c.P - 1)).toFixed(0)}%`);
      td.append(el("span", { class: "change", style: "color: var(--text-secondary)" }, parts.join("; ") || "\u00a0"));
      tr.append(td);
    });
    table.append(tr);
  });

  // TP 4 vs CP 4, per row
  const tp4 = cols[3], cp4 = cols[2];
  const gain = i => cp4[i].bytes - tp4[i].bytes;
  const best = tp4.map((c, i) => [c.label, gain(i)]).sort((x, y) => y[1] - x[1]);
  document.getElementById("tpNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    `At ${CTX}, TP 4 saves `, b(fmtDecimal(best[0][1])), ` more than CP 4 for ${best[0][0]}, `,
    `${fmtDecimal(best[1][1])} for ${best[1][0]} and a few GB elsewhere.`, "  Whether that is worth TP's extra communication, and so the choice between CP and TP, ",
    "depends on the roofline.");
}

// ---------- wiring ----------
document.getElementById("showTiny").addEventListener("change", renderLayers);
for (const inp of document.querySelectorAll("input[name=scope]"))
  inp.addEventListener("change", renderChart);
for (const inp of document.querySelectorAll("input[name=scope4]"))
  inp.addEventListener("change", renderPersistent);
window.addEventListener("resize", () => { renderChart(); renderPersistent(); renderRemat(); renderZero(); renderQuant(); renderPP(); renderCp(); renderTp(); });

// Pinned next to the context toggle: global batch, and the local batch per DP
// rank (DP degree = NUM_GPUS / (PP x CP)).
function renderCtxBatch() {
  const seqs = seqsPerStep(), local = seqs / NUM_GPUS, b = t => el("b", {}, t);
  const box = document.getElementById("ctxBatch");
  box.replaceChildren("Global batch ", b(`${exact(seqs)} seqs`),
    ` (${DATA.global_tokens / 2 ** 20}M tokens)`, el("br"),
    "Local batch per DP rank ", b(`${local === 1 ? "" : `${local} x `}PP x CP seqs`));
}

function renderAll() {
  renderCtxBatch();
  renderHeader();
  renderActivations();
  renderChart();
  renderPersistentText();
  renderPersistent();
  renderTotalNote();
  renderRemat();
  renderZero();
  renderQuant();
  renderPP();
  renderCp();
  renderTp();
}

const ctxToggle = document.getElementById("ctxToggle");
for (const key of Object.keys(DATA.contexts)) {
  const id = `ctx-${key}`;
  const input = el("input", { type: "radio", name: "ctx", id, value: key });
  if (key === CTX) input.checked = true;
  input.addEventListener("change", () => { CTX = key; renderAll(); });
  ctxToggle.append(input, el("label", { for: id }, key));
}

renderLayers();
renderAll();
document.getElementById("foot").textContent =
  `Generated by scripts/memory_study/kimi_k3_params_viz.py. Every one of the ${DATA.num_tensors} ` +
  `parameter tensors was matched by name and shape against the meta-built module before rendering.`;
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
