# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inventory Kimi K3's torch_remat regions: bytes each region keeps for
backward and its forward (= recompute) time, per layer.

Runs the memory-study config (FSDP 512 + EP 64, DistMoE, 8K, fake process
group, one GPU) truncated to the first KIMI_K3_INVENTORY_LAYERS layers (default
8: layer 0 is KDA + dense FFN, 1-2 and 4-6 KDA + MoE, 3 and 7 MLA + MoE) under a
save-all policy (save_regions=["*"]), so every region keeps its saves:

  - bytes: at the start of backward of a measured microbatch, each live
    checkpoint region's save records, deduped by storage within the layer
    (a storage is billed to the first region that saved it);
  - time: CUDA events around each region call of the original forward
    (median over the measured microbatches). Regions nest, so a parent's time
    includes its children's.

Usage:
  KIMI_K3_INVENTORY_OUT=scripts/memory_study/bench/remat_regions_8k.json \\
  NGPU=512 PYTHONPATH=.:scripts/memory_study:<dist_moe> torchrun --standalone \\
    --nproc-per-node=1 -m torchtitan.train --module kimi_k3_remat_regions \\
    --config remat_regions_8k --comm-backend fake
"""

import atexit
import functools
import json
import os
import statistics
from collections import defaultdict
from dataclasses import replace

import kimi_k3_fake_configs as base
import torch
import torch_remat as remat

# Private torch_remat internals, used for reporting only (the public
# print_saved_tensors_report returns text).
from torch_remat._region import _iter_live_regions
from torch_remat._reporting import _collect_storages
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

# TODO: remove once the study's PyTorch build (13552b6, Nov 6) is newer than the
# FSDP2 accessor torchtitan's activation offloader calls. Same body as upstream
# FSDPModule.get_all_gather_copy_in_stream.
from torch.distributed.fsdp import FSDPModule as _FSDPModule

if not hasattr(_FSDPModule, "get_all_gather_copy_in_stream"):

    def _get_all_gather_copy_in_stream(self) -> torch.Stream:
        state = self._get_fsdp_state()
        if state._is_root is None:
            raise RuntimeError("call after the root FSDP module's first forward")
        return state._comm_ctx.all_gather_copy_in_stream

    _FSDPModule.get_all_gather_copy_in_stream = _get_all_gather_copy_in_stream

NUM_LAYERS = int(os.environ.get("KIMI_K3_INVENTORY_LAYERS", "8"))
OUT = os.environ.get("KIMI_K3_INVENTORY_OUT")
# microbatch forwards to skip before measuring (step 1 warms up DistMoE, FSDP, allocator)
SKIP_FORWARDS = int(os.environ.get("KIMI_K3_INVENTORY_SKIP", "4"))

_state = {"layer": None, "forward": -1, "stack": []}
_events = []  # (forward, layer, name, parent, recompute, start, end)
_bytes = {}  # layer -> {region name: bytes}
_alloc = {}  # allocator growth across the layer stack of the snapshot microbatch
_alloc_per_layer = {}  # layer -> allocator growth across that block's forward
_block_events = []  # (layer, start, end): whole-block forward


def _measuring() -> bool:
    return _state["forward"] >= SKIP_FORWARDS


_orig_region = remat.region
_orig_checkpoint = remat.checkpoint


def _timed_region(function, name, *, recompute):
    wrapped = _orig_region(function, name, recompute=recompute)

    def call(*args, **kwargs):
        if remat.is_recomputing() or _state["layer"] is None or not _measuring():
            return wrapped(*args, **kwargs)
        parent = _state["stack"][-1] if _state["stack"] else None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _state["stack"].append(name)
        start.record()
        try:
            return wrapped(*args, **kwargs)
        finally:
            end.record()
            _state["stack"].pop()
            _events.append((_state["forward"], _state["layer"], name, parent, recompute, start, end))

    return call


def _snapshot_saved_bytes():
    """Called once, right after the last layer's forward of a measured
    microbatch: every layer's saves are live and backward has not started."""
    for region_state in _iter_live_regions():
        claimed = set()
        per_name = {}
        for name, record in region_state.records.items():
            total = 0
            for key, storage in _collect_storages(record).items():
                if key not in claimed:
                    claimed.add(key)
                    total += storage.nbytes
            per_name[name] = total
        _bytes[region_state.region_name] = per_name


def _checkpoint(*args, region_name=None, **kwargs):
    decorate = _orig_checkpoint(*args, region_name=region_name, **kwargs)

    def wrap(fn):
        inner = decorate(fn)

        def call(*a, **kw):
            if remat.is_recomputing():
                return inner(*a, **kw)
            return _measured_block(region_name, inner, *a, **kw)

        return call

    return wrap


def _measured_block(region_name, fn, *a, **kw):
    """One transformer block's original forward: time it and, on the snapshot
    microbatch, record what it holds from its forward to its backward."""
    if region_name == "layers.0":
        _state["forward"] += 1
        if _measuring() and not _state.get("snap_done"):
            _alloc["before_layers"] = torch.cuda.memory_allocated()
    _state["layer"] = region_name
    snapshot = _measuring() and not _state.get("snap_done")
    before = torch.cuda.memory_allocated() if snapshot else 0
    if _measuring():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    try:
        out = fn(*a, **kw)
    finally:
        _state["layer"] = None
    if _measuring():
        end.record()
        _block_events.append((region_name, start, end))
    if snapshot:
        _alloc_per_layer[region_name] = torch.cuda.memory_allocated() - before
    if region_name == f"layers.{NUM_LAYERS - 1}" and snapshot:
        _alloc["after_layers"] = torch.cuda.memory_allocated()
        _snapshot_saved_bytes()
        _state["snap_done"] = True
        _write()  # now, not only at exit: fake CP runs can crash a step later, which skips atexit
    return out


# Fake CP runs: peers' halves of a sequence are fake, so the loss can come out
# non-finite and the optimizer's finiteness guard would stop the run before the
# first step is measured. These runs only measure memory and time, so skip it.
if int(os.environ.get("KIMI_K3_CP", "1")) > 1:
    torch._assert_async = lambda *a, **kw: None

# KIMI_K3_INVENTORY_PLAIN=1: no instrumentation (CUDA-graphed runs; event
# timing inside a graph capture is not meaningful).
_PLAIN = os.environ.get("KIMI_K3_INVENTORY_PLAIN") == "1"
if not _PLAIN:
    remat.region = _timed_region
    remat.checkpoint = _checkpoint

if os.environ.get("KIMI_K3_INVENTORY_POLICY") == "none" and not _PLAIN:
    # No remat checkpoints to wrap: measure each block's forward directly.
    from torchtitan.models.kimi_k3.model import KimiK3TransformerBlock as _Block

    _orig_block_forward = _Block.forward
    _block_names = {}

    @functools.wraps(_orig_block_forward)  # sharding redistribution binds inputs by forward's signature
    def _block_forward(self, *a, **kw):
        name = _block_names.setdefault(id(self), f"layers.{len(_block_names)}")
        return _measured_block(name, lambda *x, **y: _orig_block_forward(self, *x, **y), *a, **kw)

    _Block.forward = _block_forward


def _write():
    if not OUT or not (_events or _block_events):
        return
    torch.cuda.synchronize()
    times = defaultdict(list)
    meta = {}
    for fwd, layer, name, parent, recompute, start, end in _events:
        times[(layer, name)].append(start.elapsed_time(end))
        meta[(layer, name)] = (parent, recompute)
    layers = defaultdict(list)
    for (layer, name), ms in times.items():
        parent, recompute = meta[(layer, name)]
        layers[layer].append(
            {
                "name": name,
                "parent": parent,
                "saved": not recompute,
                "fwd_ms": statistics.median(ms),
                "calls": len(ms),
                "bytes": _bytes.get(layer, {}).get(name, 0),
            }
        )
    out = {
        "num_layers": NUM_LAYERS,
        "measured_forwards": len({e[0] for e in _events}),
        "policy": os.environ.get("KIMI_K3_INVENTORY_POLICY", "save_all"),
        "recompute_regions": _recompute_regions(),
        "allocated_growth_over_layers": _alloc.get("after_layers", 0) - _alloc.get("before_layers", 0),
        "allocated_growth_per_layer": _alloc_per_layer,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "block_fwd_ms": _block_times(),
        "layers": dict(layers),
        "unattributed_bytes": {
            layer: {n: b for n, b in per.items() if (layer, n) not in times}
            for layer, per in _bytes.items()
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {OUT}")


atexit.register(_write)


def _block_times() -> dict[str, float]:
    per = defaultdict(list)
    for layer, start, end in _block_events:
        per[layer].append(start.elapsed_time(end))
    return {layer: statistics.median(ms) for layer, ms in per.items()}


def _recompute_regions() -> list[str]:
    return [r for r in os.environ.get("KIMI_K3_INVENTORY_RECOMPUTE", "").split(",") if r]


def _policy():
    """KIMI_K3_INVENTORY_POLICY / _RECOMPUTE / _OFFLOAD as a remat config (or None)."""
    policy = os.environ.get("KIMI_K3_INVENTORY_POLICY", "save_all")
    # KIMI_K3_INVENTORY_OFFLOAD: "1" = block inputs + every retained save; otherwise a
    # comma-separated list of region patterns, where "block" means the block inputs
    spec = os.environ.get("KIMI_K3_INVENTORY_OFFLOAD", "")
    items = ["block", "*"] if spec == "1" else [x for x in spec.split(",") if x]
    if policy == "save_all":
        return ActivationRematConfig(
            save_regions=["*"],
            recompute_regions=_recompute_regions(),
            offload_block_input="block" in items,
            offload_regions=[x for x in items if x != "block"],
        )
    if policy == "full":
        return FullActivationRematConfig(offload_block_input="block" in items,
                                         offload_regions=[x for x in items if x != "block"])
    assert not items, "offload is not wired for no remat"
    return None


def remat_perf_32k() -> Trainer.Config:
    """remat_perf_8k at 32K (perf_fsdp_ep64_32k, CP 1)."""
    return remat_perf_8k(base.perf_fsdp_ep64_32k)


def remat_perf_8k(make=None) -> Trainer.Config:
    """perf_fsdp_ep64_8k truncated to NUM_LAYERS (CUDA graphs, last step profiled) with the remat /
    offload policy from the environment; run with KIMI_K3_INVENTORY_PLAIN=1.
    Traces go to outputs/profiling/<KIMI_K3_INVENTORY_PROFILE>."""
    config = (make or base.perf_fsdp_ep64_8k)()
    config.model = replace(config.model, layers=config.model.layers[:NUM_LAYERS])
    config.activation_remat = _policy()
    steps = os.environ.get("KIMI_K3_INVENTORY_STEPS")
    if steps:  # at 32K a GPU runs one microbatch per step
        config.training.steps = int(steps)
        config.profiler = replace(config.profiler, profile_freq=int(steps))
    slot = os.environ.get("KIMI_K3_INVENTORY_DM_SLOT")  # DistMoE activation slot factor
    if slot:
        config.dist_moe_runtime = replace(config.dist_moe_runtime, activation_slot_capacity_factor=float(slot))
    prof = os.environ.get("KIMI_K3_INVENTORY_PROFILE", "remat_perf_8k")
    config.profiler = replace(config.profiler, save_traces_folder=f"profiling/{prof}",
                              enable_memory_snapshot=False)
    return config


def _generic_parallelism():
    """KIMI_K3_SEQ / _CP / _TP / _PP as a ParallelismConfig over 512 GPUs:
    FSDP + EP 64 without PP; with PP, 1F1B, EP = the stage's GPUs."""
    from torchtitan.config.parallelism import ParallelismConfig

    seq = int(os.environ.get("KIMI_K3_SEQ", "8192"))
    cp, tp, pp = (int(os.environ.get(f"KIMI_K3_{k}", "1")) for k in ("CP", "TP", "PP"))
    if pp == 1:
        par = ParallelismConfig(data_parallel_shard_degree=-1, expert_parallel_degree=64,
                                context_parallel_degree=cp, tensor_parallel_degree=tp)
    else:
        dp = 512 // (pp * tp * cp)
        par = ParallelismConfig(pipeline_parallel_degree=pp, pipeline_parallel_schedule="1F1B",
                                num_pp_microbatches=base.GLOBAL_TOKENS // seq // dp,
                                data_parallel_shard_degree=-1, expert_parallel_degree=512 // pp,
                                context_parallel_degree=cp, tensor_parallel_degree=tp)
    return seq, cp, tp, pp, par


def remat_generic() -> Trainer.Config:
    """Any FSDP + EP / ZeRO-1 + PP config from the environment (see
    _generic_parallelism), eager or CUDA-graphed (KIMI_K3_GRAPHS=1), with the
    remat / offload / DistMoE-slot policy and layer truncation of
    remat_regions_8k. With PP, set FAKE_PP_RANK (0 = the first stage)."""
    seq, cp, tp, pp, par = _generic_parallelism()
    name = f"generic_{seq // 1024}k_cp{cp}_tp{tp}_pp{pp}"
    graphs = os.environ.get("KIMI_K3_GRAPHS") == "1"
    make = (lambda: base._perf(name, seq, par, cp=cp > 1)) if graphs else (
        lambda: base._memory_study(name, seq, par, cp=cp > 1))
    config = remat_regions_8k(make)
    if pp > 1:  # stages compile at different times, so the first step's P2P can wait for minutes
        config.comm = replace(config.comm, init_timeout_seconds=3600, train_timeout_seconds=1800)
    if graphs:
        prof = os.environ.get("KIMI_K3_INVENTORY_PROFILE")
        if prof:
            config.profiler = replace(config.profiler, enable_profiling=True,
                                      save_traces_folder=f"profiling/{prof}", enable_memory_snapshot=False,
                                      profile_freq=config.training.steps)
    return config


def remat_regions_32k_cp2() -> Trainer.Config:
    """remat_regions_8k at 32K with CP 2 (mem_fsdp_ep64_32k_cp2; eager)."""
    return remat_regions_8k(base.mem_fsdp_ep64_32k_cp2)


def remat_regions_32k() -> Trainer.Config:
    """remat_regions_8k at 32K (mem_fsdp_ep64_32k)."""
    return remat_regions_8k(base.mem_fsdp_ep64_32k)


def remat_regions_8k(make=None) -> Trainer.Config:
    """mem_fsdp_ep64_8k (or `make`) truncated to NUM_LAYERS layers; KIMI_K3_INVENTORY_POLICY
    = save_all (save_regions=["*"], the default; KIMI_K3_INVENTORY_RECOMPUTE is a
    comma-separated recompute_regions list on top of it), full (full remat: shows
    the regions model code always saves) or none (no remat)."""
    config = (make or base.mem_fsdp_ep64_8k)()
    config.model = replace(config.model, layers=config.model.layers[:NUM_LAYERS])
    policy = os.environ.get("KIMI_K3_INVENTORY_POLICY", "save_all")
    config.activation_remat = {
        "save_all": ActivationRematConfig(save_regions=["*"], recompute_regions=_recompute_regions()),
        "full": FullActivationRematConfig(),
        "none": None,
    }[policy]
    if os.environ.get("KIMI_K3_INVENTORY_OFFLOAD"):  # offload needs the full policy builder
        config.activation_remat = _policy()
    slot = os.environ.get("KIMI_K3_INVENTORY_DM_SLOT")  # DistMoE activation slot factor
    if slot:
        config.dist_moe_runtime = replace(
            config.dist_moe_runtime, activation_slot_capacity_factor=float(slot)
        )
    steps = os.environ.get("KIMI_K3_INVENTORY_STEPS")
    if steps:
        config.training.steps = int(steps)
    prof = os.environ.get("KIMI_K3_INVENTORY_PROFILE")  # trace folder name: profile the last step
    if prof:
        last = config.training.steps
        config.profiler = Profiler.Config(enable_profiling=True, save_traces_folder=f"profiling/{prof}",
                                          profile_freq=last, profiler_warmup=1, profiler_active=1)
    else:
        config.profiler = Profiler.Config()
    return config
