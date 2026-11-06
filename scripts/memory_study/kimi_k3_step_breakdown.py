# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split one profiled CUDA-graph training step into forward + backward and the
optimizer (on the GPU timeline: everything after the last attention / DistMoE
kernel is the optimizer), and break forward + backward GPU time into the
roofline's buckets: matmuls (dense MXFP8 / bf16 GEMMs, DistMoE, KDA, MLA) and
the memory-bound rest.

Usage:
  python scripts/memory_study/kimi_k3_step_breakdown.py <trace.json.gz> [--out bench/x.json]
"""

import argparse
import collections
import gzip
import json
import re


def cls(n: str) -> str:
    """Kernel class, first match wins. The memory-bound classes match the
    non-matmul model's: AttnRes (its compiled region's fused cat / norm /
    softmax kernels, the batched-matmul GEMVs, and the stack concatenations),
    MXFP8 quantization, FSDP buffer copies, norms, and everything else."""
    if re.search(r"kda|frost|short_conv", n):
        return "KDA"
    if re.search(r"dist_moe", n) and "quantize" not in n:
        return "DistMoE grouped GEMM (+fused dispatch/combine)"
    if re.search(r"FlashAttn|flash|fmha|sdpa", n):
        return "MLA"
    if re.search(r"mxfp8|quantize|scale_swizzle|to_mx", n):
        return "MXFP8 quantize"
    if re.search(r"attention_residual|_attn_res|cat_mean_mul_pow_rsqrt|softmax_backward|gemv|CatArrayBatchedCopy|gpu_kernel_impl_nocast", n):
        return "Attention residuals"
    if re.search(r"nvjet_\w*qq", n):
        return "Dense GEMM MXFP8"
    if re.search(r"nvjet|gemm|cutlass|cublas|xmma|matmul", n, re.I):
        return "Dense GEMM bf16"
    if re.search(r"chunk_cat|split_with_sizes|foreach_copy|all_gather_copy", n):
        return "FSDP buffer copies"
    if re.search(r"rms_norm|l2norm", n):
        return "Norms (RMSNorm, L2)"
    return "Other elementwise, casts, reductions"


def busy(events) -> float:
    total, end = 0.0, -1.0
    for e in events:
        s, f = e["ts"], e["ts"] + e["dur"]
        if f <= end:
            continue
        total += f - max(s, end)
        end = f
    return total / 1e6


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    t = json.load(gzip.open(a.trace, "rt"))
    ev = sorted((e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"),
                key=lambda e: e["ts"])
    fb_end = max(e["ts"] + e["dur"] for e in ev if re.search(r"kda|frost|dist_moe|FlashAttn|flash|fmha", e["name"]))
    t0, tend = ev[0]["ts"], max(e["ts"] + e["dur"] for e in ev)
    fb = [e for e in ev if e["ts"] < fb_end]
    op = [e for e in ev if e["ts"] >= fb_end]
    by = collections.defaultdict(float)
    for e in fb:
        by[cls(e["name"])] += e["dur"] / 1e6
    res = {"trace": a.trace, "fwd_bwd_span_s": (fb_end - t0) / 1e6, "fwd_bwd_busy_s": busy(fb),
           "optimizer_span_s": (tend - fb_end) / 1e6, "optimizer_busy_s": busy(op),
           "fwd_bwd_by_class_s": dict(sorted(by.items(), key=lambda kv: -kv[1]))}
    print(f"fwd+bwd span {res['fwd_bwd_span_s']:.2f} s (busy {res['fwd_bwd_busy_s']:.2f}); "
          f"optimizer span {res['optimizer_span_s']:.2f} s (busy {res['optimizer_busy_s']:.2f})")
    for k, v in res["fwd_bwd_by_class_s"].items():
        print(f"  {v:6.2f} s  {k}")
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
