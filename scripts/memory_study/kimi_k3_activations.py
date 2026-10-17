# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Account for every tensor Kimi K3 saves for backward, deduplicated by storage.

Step 2 of the memory study (step 1 is ``kimi_k3_params.py``): the activation
footprint of one forward pass with no parallelism and no activation
checkpointing.

How it works:
  1. Build the full model under ``FakeTensorMode`` on CUDA in bf16 (FSDP's
     compute dtype), so no memory is allocated.
  2. Run forward + cross-entropy loss inside a saved-tensors pack hook
     (``torch_remat.saved_tensors_hooks``, which matches the native hook outside
     checkpoint regions and composes with remat regions once AC is enabled).
  3. Deduplicate by storage: a storage saved by several ops (or through several
     views) is billed once, at its full storage size, to the first module that
     saved it. Parameter storages are reported separately and excluded from the
     activation total.
  4. Report totals by region (embedding / each decoder layer / head / loss),
     per-layer-type breakdowns, and bytes per token.

Usage:
  PYTHONPATH=. python scripts/memory_study/kimi_k3_activations.py \
      [--flavor Kimi-K3] [--seq-len 4096] [--top 25]
"""

import argparse
import contextlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from unittest import mock

import torch
import torch_remat as remat
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    MXFP8GroupedLinearConverter,
    MXFP8LinearConverter,
)
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.kimi_k3 import build_model_config

LAYER_RE = re.compile(r"^layers\.(\d+)(?:\.|$)")
LOSS_CHUNK = "<loss chunk>"
# Projection matrices converted to MXFP8 (FQN substrings).
MXFP8_LINEARS = [
    "delta_attention.q_proj",
    "delta_attention.k_proj",
    "delta_attention.v_proj",
    "delta_attention.forget_a",
    "delta_attention.forget_b",
    "delta_attention.output_gate",
    "delta_attention.output_proj",
    "attention.wq_a",
    "attention.wq_b",
    "attention.wkv_a",
    "attention.wkv_b",
    "attention.gate",
    "attention.wo",
    "feed_forward.w13",
    "feed_forward.w2",
    "shared_experts.w13",
    "shared_experts.w2",
    "moe.routed_down",
    "moe.routed_up",
]


@dataclass
class SavedStorage:
    """One deduplicated storage saved for backward."""

    nbytes: int
    module: str  # FQN of the innermost module active at the first save
    producer: str  # autograd node that produced the first saved view
    shape: tuple[int, ...]  # shape/dtype of the first saved view
    dtype: torch.dtype
    is_param: bool
    num_saves: int = 1
    views: set[tuple[int, ...]] = field(default_factory=set)
    savers: set[str] = field(default_factory=set)  # every module that saved it


def producer_of(t: torch.Tensor) -> str:
    """Name of the autograd node that produced ``t`` (``MmBackward0`` -> ``Mm``).

    A dispatch mode would give exact aten op names, but Dynamo refuses to
    compile flex attention while a user dispatch mode is active.
    """
    if t.grad_fn is not None:
        return re.sub(r"Backward\d*$", "", type(t.grad_fn).__name__)
    return "<leaf>" if t.requires_grad else "<no grad>"


class SavedTensorRecorder:
    """Pack hook + module stack that records deduplicated saved storages."""

    def __init__(self, model: torch.nn.Module):
        self.fqn = {id(m): name for name, m in model.named_modules()}
        self.param_storages = {
            StorageWeakRef(p.untyped_storage()) for p in model.parameters()
        }
        self.stack: list[str] = []
        self.saved: dict[StorageWeakRef, SavedStorage] = {}
        self.extra_module = ""  # label for saves outside any module (the loss)
        self.num_packs = 0
        self.naive_bytes = 0  # storage bytes summed over every pack, no dedup

    def pre_hook(self, module, args):
        self.stack.append(self.fqn.get(id(module), "?"))

    def post_hook(self, module, args, output):
        self.stack.pop()

    def pack(self, t: torch.Tensor) -> torch.Tensor:
        storage = t.untyped_storage()
        self.num_packs += 1
        self.naive_bytes += storage.nbytes()
        key = StorageWeakRef(storage)
        rec = self.saved.get(key)
        if rec is None:
            rec = SavedStorage(
                nbytes=storage.nbytes(),
                # Saves inside the loss-chunk loop belong to the chunk, even when
                # lm_head is the innermost module.
                module=(
                    LOSS_CHUNK
                    if LOSS_CHUNK in self.stack
                    else self.stack[-1]
                    if self.stack
                    else self.extra_module
                ),
                producer=producer_of(t),
                shape=tuple(t.shape),
                dtype=t.dtype,
                is_param=key in self.param_storages,
            )
            self.saved[key] = rec
        else:
            rec.num_saves += 1
        rec.views.add(tuple(t.shape))
        rec.savers.add(self.stack[-1] if self.stack else self.extra_module)
        return t


# ---------------------------------------------------------------------------
# Model run
# ---------------------------------------------------------------------------


def run(
    flavor: str,
    seq_len: int,
    *,
    attn_backend: str,
    local_compile: str,
    execution: str,
    fake_compile: str,
    recompute_materialized: bool = True,
    remat_policy: str = "none",
    chunked_loss: int = 0,
    mxfp8: str = "off",
) -> tuple[object, "SavedTensorRecorder"]:
    # MXFP8 as TorchTitan applies it: converters swap the projection matrices to
    # MXFP8Linear (every converted linear keeps its backward input in MXFP8) and
    # the routed experts to MXFP8 grouped GEMMs. Residual projections (out=1),
    # the router and lm_head are not MXFP8 targets.
    # mxfp8="all": every converted linear saves its backward input in MXFP8;
    # mxfp8="bf16": TorchTitan's conservative default, linears save bf16.
    converters = None
    if mxfp8 != "off":
        converters = [
            MXFP8LinearConverter.Config(
                fqns=MXFP8_LINEARS,
                linears_saving_inputs_for_backward_in_mxfp8=(
                    MXFP8_LINEARS if mxfp8 == "all" else []
                ),
            ),
            MXFP8GroupedLinearConverter.Config(),
        ]
    cfg = build_model_config(
        flavor,
        attn_backend=attn_backend,
        converters=converters,
        parallelism=ParallelismConfig(),
    )
    cfg = replace(cfg, vision_encoder=None)
    fake = execution == "fake"
    # The min-cut partitioner's ban_recompute_materialized_backward (default
    # True) never recomputes a tensor that backward must materialize, e.g. an
    # upcast feeding a bmm. For AttnRes that saves an fp32 copy of [T, N+1, D]
    # per call instead of the bf16 input. The model is memory-bound, so this
    # study turns the ban off (recompute the upcast in backward). The flag is
    # read at compile time; reset Dynamo so a second run() recompiles.
    torch._functorch.config.ban_recompute_materialized_backward = (
        not recompute_materialized
    )
    torch._dynamo.reset()
    regions = cfg.local_compile_regions if local_compile == "default" else []
    # "default" binds the model's configured regions as parallelize() does;
    # "none" keeps every region eager. Under fake tensors Inductor's Triton
    # kernels would launch on nonexistent storage, so either:
    #   aot_eager_decomp_partition: Inductor's decompositions and min-cut
    #     partitioner (which decide what is saved), graphs run as aten ops; or
    #   inductor_wrap: real Inductor, each compiled graph wrapped in the
    #     inductor_compiled_code HOP (wrap_inductor_compiled_regions) whose fake
    #     impl reruns the stored FX graph. Correct on the compiling call, but on
    #     cached reuse it returns fresh storages for outputs that alias inputs, so
    #     saved views of one storage (SiTU-GLU's gate/up) dedupe as two storages
    #     and the total overcounts (Kimi-K3, T=4096: 522 vs 445 GiB). Checked
    #     against real tensors on the SiTU-GLU region: aot_eager_decomp_partition
    #     matches exactly.
    if fake and fake_compile == "aot_eager_decomp_partition":
        real_compile = torch.compile

        def compile_for_fake_tensors(fn, **kwargs):
            return real_compile(fn, **{**kwargs, "backend": fake_compile})

        with mock.patch.object(torch, "compile", compile_for_fake_tensors):
            apply_local_compile(regions)
    else:
        torch._inductor.config.wrap_inductor_compiled_regions = fake
        apply_local_compile(regions)

    # VarlenInnerAttention activates FlashAttention-4 on Blackwell, which needs
    # the flash_attn package. Under fake tensors no kernel runs: varlen_attn's
    # fake impl gives the shapes and its setup_context fixes what is saved
    # (q, k, v, cu_seqs, out, lse, rng_state) regardless of the kernel, so skip
    # the activation.
    fake_mode = (
        FakeTensorMode(allow_non_fake_inputs=True) if fake else contextlib.nullcontext()
    )
    with fake_mode, mock.patch(
        "torchtitan.tools.utils.get_cuda_flash_attention_impl", return_value=None
    ):
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                model = cfg.build()
                if not fake:
                    model.init_weights(buffer_device=torch.device("cuda"))
        finally:
            torch.set_default_dtype(torch.float32)
    # Activation remat as parallelize() applies it. Both configs wrap every
    # transformer block in torch_remat.checkpoint:
    #   default: the trainer default; keep every model-declared region except
    #     the routed experts' w13 grouped GEMM.
    #   full: recompute everything but the block inputs and the regions model
    #     code always retains.
    remat_config = {
        "none": None,
        "default": DefaultActivationRematConfig,
        "full": FullActivationRematConfig,
    }[remat_policy]
    if remat_config is not None:
        remat_config().apply(model)

    # One packed document of seq_len tokens. Attention metadata depends on the
    # data (document boundaries), so build it from real tensors outside the
    # fake mode; the fake mode accepts them as constants.
    positions = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    attention_metadata = model._get_attention_metadata(positions)

    recorder = SavedTensorRecorder(model)
    handles = [
        torch.nn.modules.module.register_module_forward_pre_hook(recorder.pre_hook),
        torch.nn.modules.module.register_module_forward_hook(
            recorder.post_hook, always_call=True
        ),
    ]
    try:
        with fake_mode:
            tokens = torch.randint(0, cfg.vocab_size, (seq_len,), device="cuda")
            labels = torch.randint(0, cfg.vocab_size, (seq_len,), device="cuda")
            with remat.saved_tensors_hooks(recorder.pack, lambda t: t):
                if chunked_loss:
                    # ChunkedLossWrapper: the decoder stops before lm_head; each
                    # chunk then runs lm_head + loss and its backward right away,
                    # so one chunk's saves are a transient, recorded as
                    # "<loss chunk>" rather than held across the decoder backward.
                    model._skip_lm_head = True
                    hidden_TD = model(
                        tokens,
                        positions=positions,
                        attention_metadata=attention_metadata,
                    )
                    chunk = seq_len // chunked_loss
                    hidden_chunk = hidden_TD[:chunk].detach().requires_grad_()
                    recorder.stack.append(LOSS_CHUNK)
                    loss = cross_entropy_loss(
                        model.lm_head(hidden_chunk), labels[:chunk]
                    )
                    recorder.stack.pop()
                else:
                    logits = model(
                        tokens,
                        positions=positions,
                        attention_metadata=attention_metadata,
                    )
                    recorder.extra_module = "<loss>"
                    loss = cross_entropy_loss(logits, labels)
            assert loss.requires_grad
    finally:
        for h in handles:
            h.remove()
    return cfg, recorder


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_bytes(n: float) -> str:
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def region_of(module: str) -> str:
    """Coarse region: embedding / layers.N / head / loss / loss chunk."""
    if module == "<loss>":
        return "loss"
    if module == LOSS_CHUNK:
        return "loss chunk"
    if module.startswith("tok_embeddings"):
        return "embedding"
    m = LAYER_RE.match(module)
    if m:
        return f"layers.{m.group(1)}"
    return "head"


def layer_type(cfg, layer_id: int) -> str:
    lc = cfg.layers[layer_id]
    attn = "MLA" if lc.attention is not None else "KDA"
    ffn = "dense FFN" if lc.feed_forward is not None else "LatentMoE"
    return f"{attn} + {ffn}"


def local_module(module: str) -> str:
    return LAYER_RE.sub("", module) or "(block)"


def print_breakdown(rows: list[SavedStorage], title: str, seq_len: int, top: int):
    """Per (module, producer, shape) rows, largest first."""
    agg: dict[tuple[str, str, tuple, torch.dtype], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for r in rows:
        key = (local_module(r.module), r.producer, r.shape, r.dtype)
        agg[key][0] += r.nbytes
        agg[key][1] += 1
    total = sum(r.nbytes for r in rows)
    print(f"\n{title}")
    print(
        f"  {'module (first saver)':<44} {'producer':<26} {'shape (first view)':<22} "
        f"{'dtype':<9} {'bytes':>11} {'B/token':>9} {'share':>7}"
    )
    items = sorted(agg.items(), key=lambda kv: -kv[1][0])
    for (mod, prod, shape, dtype), (nbytes, count) in items[:top]:
        dt = str(dtype).replace("torch.", "")
        cnt = f" x{count}" if count > 1 else ""
        print(
            f"  {mod:<44} {prod + cnt:<26} {str(list(shape)):<22} {dt:<9} "
            f"{fmt_bytes(nbytes):>11} {nbytes / seq_len:>9,.0f} {100 * nbytes / total:>6.2f}%"
        )
    rest = items[top:]
    if rest:
        rb = sum(v[0] for _, v in rest)
        print(f"  ... {len(rest)} more rows, {fmt_bytes(rb)}")
    print(f"  {'total':<104} {fmt_bytes(total):>11} {total / seq_len:>9,.0f}")


def report(
    cfg, recorder: SavedTensorRecorder, seq_len: int, top: int, setup: str
) -> None:
    saved = list(recorder.saved.values())
    acts = [r for r in saved if not r.is_param]
    params = [r for r in saved if r.is_param]
    act_total = sum(r.nbytes for r in acts)

    # Per region totals.
    by_region: dict[str, list[SavedStorage]] = defaultdict(list)
    for r in acts:
        by_region[region_of(r.module)].append(r)

    # Group decoder layers by type and check they agree.
    by_type: dict[str, list[int]] = defaultdict(list)
    for layer_id in range(len(cfg.layers)):
        by_type[layer_type(cfg, layer_id)].append(layer_id)

    print(
        f"Kimi K3 saved-for-backward accounting -- {len(cfg.layers)} layers, "
        f"seq_len T = {seq_len:,}, bf16 compute, no parallelism"
    )
    print(f"  {setup}")
    print(
        f"  {recorder.num_packs:,} pack calls -> {len(saved):,} unique storages; "
        f"dedup by storage: {fmt_bytes(recorder.naive_bytes)} -> "
        f"{fmt_bytes(sum(r.nbytes for r in saved))} (activations + parameters)"
    )

    print("\n== Totals by region")
    print(
        f"  {'region':<36} {'layers':>6} {'per layer':>12} {'total':>12} {'B/token':>12} {'share':>7}"
    )
    region_rows = [("embedding", ["embedding"])]
    for t, ids in by_type.items():
        region_rows.append((t, [f"layers.{i}" for i in ids]))
    region_rows += [
        ("head (AttnRes + norm + lm_head)", ["head"]),
        ("loss (cross-entropy)", ["loss"]),
        ("one loss chunk (transient)", ["loss chunk"]),
    ]
    accounted = 0
    for label, regions in region_rows:
        per = [sum(r.nbytes for r in by_region.get(g, [])) for g in regions]
        tot = sum(per)
        accounted += tot
        nl = (
            len(regions)
            if label not in ("embedding",)
            and not label.startswith(("head", "loss", "one loss"))
            else ""
        )
        spread = ""
        if nl and min(per) != max(per):
            spread = f"  (min {fmt_bytes(min(per))}, max {fmt_bytes(max(per))})"
        per_layer = fmt_bytes(tot / len(regions)) if nl else ""
        print(
            f"  {label:<36} {str(nl):>6} {per_layer:>12} {fmt_bytes(tot):>12} "
            f"{tot / seq_len:>12,.0f} {100 * tot / act_total:>6.2f}%{spread}"
        )
    print(
        f"  {'TOTAL activations':<36} {'':>6} {'':>12} {fmt_bytes(act_total):>12} "
        f"{act_total / seq_len:>12,.0f} {100.0:>6.2f}%"
    )
    assert accounted == act_total, (accounted, act_total)

    # One representative layer of each type, plus the non-layer regions.
    print(
        "\n  Breakdown rows merge storages with the same (module, producer, shape, "
        "dtype); 'xN' = N storages."
    )
    for t, ids in by_type.items():
        rep = ids[1] if len(ids) > 1 else ids[0]
        print_breakdown(
            by_region[f"layers.{rep}"],
            f"== {t}: layer {rep} (one of {len(ids)})",
            seq_len,
            top,
        )
    for g, title in (
        ("embedding", "== Embedding"),
        ("head", "== Head"),
        ("loss", "== Loss"),
    ):
        if by_region.get(g):
            print_breakdown(by_region[g], title, seq_len, top)

    print("\n== Parameters saved for backward (excluded from the activation total)")
    pbytes = sum(r.nbytes for r in params)
    print(
        f"  {len(params)} parameter storages, {fmt_bytes(pbytes)} "
        f"(weights referenced by matmul backward; they live in memory anyway)"
    )


def mxfp8_exclusive_bytes(saved: list[SavedStorage]) -> int:
    """Bytes of bf16 activations saved only by MXFP8-convertible linears.

    Those are the only saves an MXFP8 input format can shrink: if any other op
    also saves the storage, a linear's MXFP8 copy is extra memory.
    """

    def is_linear(fqn: str) -> bool:
        return any(fqn.endswith(name) for name in MXFP8_LINEARS)

    return sum(
        r.nbytes
        for r in saved
        if not r.is_param
        and r.dtype == torch.bfloat16
        and r.savers
        and all(is_linear(m) for m in r.savers)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flavor", default="Kimi-K3")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--top", type=int, default=25, help="rows per breakdown")
    parser.add_argument(
        "--attn-backend",
        default="varlen",
        choices=["varlen", "flex", "flex_flash"],
        help="MLA attention backend; varlen is an eager op (no torch.compile)",
    )
    parser.add_argument(
        "--local-compile",
        default="default",
        choices=["default", "none"],
        help="default: the model's local_compile_regions, as in training; none: all eager",
    )
    parser.add_argument(
        "--execution",
        default="fake",
        choices=["fake", "real"],
        help="real allocates and runs the model; only for small flavors (debugmodel)",
    )
    parser.add_argument(
        "--fake-compile",
        default="aot_eager_decomp_partition",
        choices=["aot_eager_decomp_partition", "inductor_wrap"],
        help="how local_compile regions run under fake tensors (see run())",
    )
    parser.add_argument(
        "--chunked-loss",
        type=int,
        default=0,
        help="number of loss chunks (ChunkedLossWrapper); 0 = unchunked loss",
    )
    parser.add_argument(
        "--mxfp8",
        default="off",
        choices=["off", "all", "bf16"],
        help="MXFP8 linear + grouped-GEMM converters: off; all (every converted "
        "linear saves MXFP8 inputs); bf16 (linears save bf16 inputs, the default)",
    )
    parser.add_argument(
        "--remat",
        default="none",
        choices=["none", "default", "full"],
        help="activation remat: none, default (DefaultActivationRematConfig, the "
        "trainer default), or full (FullActivationRematConfig)",
    )
    parser.add_argument(
        "--no-recompute-materialized",
        action="store_true",
        help="keep the partitioner's default ban_recompute_materialized_backward=True",
    )
    args = parser.parse_args()

    cfg, recorder = run(
        args.flavor,
        args.seq_len,
        attn_backend=args.attn_backend,
        local_compile=args.local_compile,
        execution=args.execution,
        fake_compile=args.fake_compile,
        recompute_materialized=not args.no_recompute_materialized,
        remat_policy=args.remat,
        chunked_loss=args.chunked_loss,
        mxfp8=args.mxfp8,
    )
    regions = cfg.local_compile_regions if args.local_compile == "default" else []
    setup = (
        f"attn_backend={args.attn_backend}; local_compile_regions={regions}; "
        f"execution={args.execution}"
        + (
            f" (compiled regions: {args.fake_compile})"
            if args.execution == "fake"
            else ""
        )
        + f"; ban_recompute_materialized_backward={args.no_recompute_materialized}"
        + f"; remat={args.remat}; chunked_loss={args.chunked_loss}; mxfp8={args.mxfp8}"
    )
    report(cfg, recorder, args.seq_len, args.top, setup)
    excl = mxfp8_exclusive_bytes(list(recorder.saved.values()))
    print(
        f"\n== bf16 activations saved only by MXFP8-convertible linears: {fmt_bytes(excl)} "
        f"(an MXFP8 save at 1 + 1/32 B/elem would cut this to {fmt_bytes(excl * 33 / 64)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
