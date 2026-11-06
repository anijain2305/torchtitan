# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Break a CUDA memory snapshot into the blog's memory categories, at the end
of the step and at the step's peak.

Reads a TorchTitan memory snapshot (pickle from Profiler's
enable_memory_snapshot). The live blocks at dump time come from the segments;
replaying the allocator trace backwards from there reconstructs the live set at
every earlier event, so the peak allocated moment and what was live at it can
be found. Each block is classified by the Python frames that allocated it.

Usage:
  python scripts/memory_study/kimi_k3_memsnap.py <snapshot.pickle> [--out bench/x.json]
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

GIB = 2**30

# (category, substrings of an allocating frame's filename or function), first match wins,
# searched from the innermost frame outwards
RULES = [
    ("optimizer state", ["optim", "muon", "adamw"]),
    ("DistMoE buffers (slots, scratch, comm)", ["dist_moe"]),
    (
        "FSDP unsharded params / all-gather",
        [
            "_fsdp_collectives.py:foreach_all_gather",
            "_fsdp_param.py",
            "fsdp_post_all_gather",
            "_fsdp_tensor",
            "all_gather",
        ],
    ),
    ("FSDP gradient reduce-scatter", ["reduce_scatter", "foreach_reduce"]),
    ("loss / LM head", ["loss", "lm_head", "chunked"]),
    (
        "autograd backward (grads, recompute)",
        ["backward", "autograd", "remat", "checkpoint"],
    ),
    ("forward activations", ["forward", "_call_impl"]),
]


def frames_key(frames: list[dict]) -> list[str]:
    return [f"{f.get('filename', '')}:{f.get('name', '')}" for f in frames or []]


def classify(frames: list[dict]) -> str:
    keys = frames_key(frames)
    if not keys:
        return "no stack (allocated before recording)"
    for cat, pats in RULES:
        for k in keys:
            if any(p in k for p in pats):
                return cat
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    snap = pickle.load(open(args.snapshot, "rb"))

    live = {}  # addr -> (size, frames)
    reserved = 0
    device = max(
        range(len(snap.get("device_traces") or [[]])),
        key=lambda d: len(snap["device_traces"][d]),
        default=0,
    )
    for seg in snap["segments"]:
        if seg.get("device", device) != device:
            continue
        reserved += seg["total_size"]
        addr = seg["address"]
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                live[addr] = (blk["size"], blk.get("frames", []))
            addr += blk["size"]
    end_bytes = sum(s for s, _ in live.values())
    end_by = defaultdict(int)
    for s, fr in live.values():
        end_by[classify(fr)] += s

    # each rank records on its own device; pick the device that has the trace
    trace = max(snap.get("device_traces") or [[]], key=len)
    # a free event carries the stack of the free; use the matching alloc's stack
    # when it is inside the trace window
    alloc_frames, frames_at = {}, {}
    for i, e in enumerate(trace):
        if e["action"] == "alloc":
            alloc_frames[e.get("addr")] = e.get("frames", [])
        elif e["action"] == "free_completed" and e.get("addr") in alloc_frames:
            frames_at[i] = alloc_frames.pop(e.get("addr"))
    cur = dict(live)
    total = end_bytes
    peak, peak_live, peak_idx = total, dict(cur), len(trace)
    for i in range(len(trace) - 1, -1, -1):
        e = trace[i]
        act, addr, size = e["action"], e.get("addr"), e.get("size", 0)
        if act == "alloc":
            if addr in cur:
                total -= cur.pop(addr)[0]
        elif act == "free_completed":
            cur[addr] = (size, frames_at.get(i, e.get("frames", [])))
            total += size
            if total > peak:
                peak, peak_live, peak_idx = total, dict(cur), i
    peak_by = defaultdict(int)
    for s, fr in peak_live.values():
        peak_by[classify(fr)] += s

    def show(title, by, tot):
        print(f"{title}: {tot / GIB:.2f} GiB ({tot / 1e9:.1f} GB)")
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"  {k:46s} {v / GIB:8.2f} GiB  {v / 1e9:7.1f} GB")

    print(f"reserved at dump: {reserved / GIB:.2f} GiB; trace events: {len(trace):,}")
    show("allocated at end of step", end_by, end_bytes)
    show(f"peak allocated (trace event {peak_idx:,})", peak_by, peak)
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "snapshot": str(args.snapshot),
                    "reserved_gb": reserved / 1e9,
                    "end_gb": end_bytes / 1e9,
                    "end_by_gb": {k: v / 1e9 for k, v in end_by.items()},
                    "peak_gb": peak / 1e9,
                    "peak_by_gb": {k: v / 1e9 for k, v in peak_by.items()},
                },
                indent=1,
            )
        )
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
