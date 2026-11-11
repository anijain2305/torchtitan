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
            if region_name == "layers.0":
                _state["forward"] += 1
                if _measuring() and not _bytes:
                    _alloc["before_layers"] = torch.cuda.memory_allocated()
            _state["layer"] = region_name
            snapshot = _measuring() and not _bytes
            before = torch.cuda.memory_allocated() if snapshot else 0
            if _measuring():
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            try:
                out = inner(*a, **kw)
            finally:
                _state["layer"] = None
            if _measuring():
                end.record()
                _block_events.append((region_name, start, end))
            if snapshot:  # what this block holds from its forward to its backward
                _alloc_per_layer[region_name] = torch.cuda.memory_allocated() - before
            if region_name == f"layers.{NUM_LAYERS - 1}" and _measuring() and not _bytes:
                _alloc["after_layers"] = torch.cuda.memory_allocated()
                _snapshot_saved_bytes()
            return out

        return call

    return wrap


remat.region = _timed_region
remat.checkpoint = _checkpoint


def _write():
    if not OUT or not _events:
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


def remat_regions_8k() -> Trainer.Config:
    """mem_fsdp_ep64_8k truncated to NUM_LAYERS layers; KIMI_K3_INVENTORY_POLICY
    = save_all (save_regions=["*"], the default; KIMI_K3_INVENTORY_RECOMPUTE is a
    comma-separated recompute_regions list on top of it), full (full remat: shows
    the regions model code always saves) or none (no remat)."""
    config = base.mem_fsdp_ep64_8k()
    config.model = replace(config.model, layers=config.model.layers[:NUM_LAYERS])
    policy = os.environ.get("KIMI_K3_INVENTORY_POLICY", "save_all")
    config.activation_remat = {
        "save_all": ActivationRematConfig(save_regions=["*"], recompute_regions=_recompute_regions()),
        "full": FullActivationRematConfig(),
        "none": None,
    }[policy]
    config.profiler = Profiler.Config()
    return config
