# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Break one profiled training step's GPU time into op classes.

Reads a PyTorch profiler Chrome trace (from kimi_k3_fake_configs' *_profiled
configs), sums GPU kernel durations by kernel-name class, and reports each
class's time and share. Classes are matched in order, first match wins; the
"other" bucket lists its largest kernels so new patterns can be added.

Usage:
  python scripts/memory_study/kimi_k3_trace_breakdown.py <trace.json[.gz]> [--out bench/x.json]
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

# (class, kernel-name regex), first match wins
CLASSES = [
    (
        "FSDP buffer copies (chunk_cat / split copy-out)",
        r"chunk_cat|split_with_sizes_copy|foreach_copy|all_gather_copy",
    ),
    ("Attention: MLA (SDPA / flash)", r"sdpa|flash|fmha|cudnn::fusion"),
    ("Attention: KDA", r"kda|frost|delta|gdn|gla|neumann|gate_scan|recompute_w_u"),
    ("Routed experts: DistMoE kernels", r"dist_moe|grouped_gemm|GroupedGemm"),
    (
        "MoE token movement (index / sort / scatter)",
        r"indexing_backward|index_put|index_select|index_add|scatter|gather|radix|unique|sort",
    ),
    (
        "GEMMs (dense; non-DistMoE expert GEMMs too)",
        r"nvjet|gemm|gemv|cutlass|cublas|xmma|matmul",
    ),
    ("Norms, softmax (RMSNorm, AttnRes depth mixing)", r"rsqrt|softmax|norm|mean"),
    ("Optimizer (Muon / AdamW updates)", r"foreach|multi_tensor|adam|muon|lerp"),
    ("Copies, casts, cat, memset", r"copy|memcpy|memset|cast|cat|fill_"),
    (
        "Elementwise and other fused Triton",
        r"elementwise|vectorized|triton_|unrolled|silu|sigmoid|exp|mul|add",
    ),
]


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    events = load(args.trace).get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    by_class: dict[str, float] = defaultdict(float)
    other: dict[str, float] = defaultdict(float)
    for e in kernels:
        name = e.get("name", "")
        for cls, pattern in CLASSES:
            if re.search(pattern, name, re.IGNORECASE):
                by_class[cls] += e["dur"]
                break
        else:
            by_class["other"] += e["dur"]
            other[name[:120]] += e["dur"]
    total = sum(by_class.values())
    # wall span of GPU activity in the profiled step
    span = (
        (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels))
        if kernels
        else 0.0
    )
    print(
        f"{len(kernels)} kernels, GPU busy {total / 1e6:.2f} s over a {span / 1e6:.2f} s span "
        f"({100 * total / span if span else 0:.0f}% busy)"
    )
    for cls, us in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:48s} {us / 1e6:8.3f} s  {100 * us / total:5.1f}%")
    print("largest 'other' kernels:")
    for name, us in sorted(other.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {us / 1e6:8.3f} s  {name}")
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "trace": str(args.trace),
                    "gpu_busy_s": total / 1e6,
                    "span_s": span / 1e6,
                    "by_class_s": {k: v / 1e6 for k, v in by_class.items()},
                    "top_other_s": {
                        k: v / 1e6
                        for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:30]
                    },
                },
                indent=1,
            )
        )
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
