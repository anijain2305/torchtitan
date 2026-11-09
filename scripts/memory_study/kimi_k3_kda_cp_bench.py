# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time KDA under context parallelism at Kimi K3's shapes, for the roofline.

Each rank holds its T / cp tokens of one T-token document (TorchTitan's
contiguous CP sharding) and runs Attention Gym's context_parallel_kda, the
kernel ContextParallelInnerKDA calls. For comparison it also times plain
chunk_kda on the full T tokens and on a standalone T / cp-token document.
Times are summed GPU kernel time per call (NCCL kernels included), forward and
forward + backward, measured on rank 0.

Run (writes bench/kda_cp_<gpu>_cp<world>.json):
  torchrun --standalone --nproc-per-node=2 scripts/memory_study/kimi_k3_kda_cp_bench.py
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

# attention before context_parallel: the reverse order is a circular import
from torchtitan.models.common.attention.linear_attention import chunk_kernel_options
from torchtitan.distributed.context_parallel import get_token_fragments  # usort: skip
from torchtitan.models.common.attn_gym import (
    chunk_kda,
    context_parallel_kda,
    ContextParallelRouting,
)

HEADS, DIM = 96, 128  # Kimi K3 KDA
SEQ_LENS = (8192, 16384, 32768)


def time_ms(fn, iters: int = 10, warmup: int = 3, repeats: int = 5) -> float:
    """Summed GPU kernel time per call, minimum over repeats. KDA launches many
    small kernels, so wall time on a shared host is dominated by CPU launch
    jitter; the fake-PG trace breakdown also sums kernel time, so this is the
    comparable number. The minimum drops bursts of interference from other jobs
    sharing the GPU (up to 2.4x slower on identical calls)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        dist.barrier()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        kernel_us = sum(
            e.device_time
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        )
        best = min(best, kernel_us / 1e3 / iters)
    return best


def inputs(t: int, device):
    bf = torch.bfloat16
    q = (
        F.normalize(torch.randn(1, t, HEADS, DIM, device=device), dim=-1)
        .to(bf)
        .requires_grad_()
    )
    k = (
        F.normalize(torch.randn(1, t, HEADS, DIM, device=device), dim=-1)
        .to(bf)
        .requires_grad_()
    )
    v = torch.randn(1, t, HEADS, DIM, device=device, dtype=bf, requires_grad=True)
    g = (
        -F.softplus(torch.randn(1, t, HEADS, DIM, device=device)) * 0.1
    ).requires_grad_()
    beta = torch.sigmoid(torch.randn(1, t, HEADS, device=device)).requires_grad_()
    do = torch.randn(1, t, HEADS, DIM, device=device, dtype=bf)
    return (q, k, v, g, beta), do


def fwd_and_fwd_bwd(fwd, do) -> tuple[float, float]:
    def fwd_bwd():
        torch.autograd.backward(fwd(), do)

    with torch.no_grad():
        f_ms = time_ms(fwd)
    return f_ms, time_ms(fwd_bwd)


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    cp, rank = dist.get_world_size(), dist.get_rank()
    results = []
    for t in SEQ_LENS:
        torch.manual_seed(7 + rank)
        r = {"seq_len": t, "cp": cp, "local_tokens": t // cp}
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=[0, t],
            fragments=get_token_fragments(t, cp_size=cp, permutation=None),
            cp_rank=rank,
            device=device,
        )
        x, do = inputs(t // cp, device)
        opts = chunk_kernel_options(x[0])
        r["cp_fwd_ms"], r["cp_fwd_bwd_ms"] = fwd_and_fwd_bwd(
            lambda: context_parallel_kda(
                *x, routing=routing, group=dist.group.WORLD, kernel_options=opts
            )[0],
            do,
        )
        r["local_fwd_ms"], r["local_fwd_bwd_ms"] = fwd_and_fwd_bwd(
            lambda: chunk_kda(*x, autotune=True, kernel_options=opts)[0], do
        )
        del x, do
        x, do = inputs(t, device)
        r["full_fwd_ms"], r["full_fwd_bwd_ms"] = fwd_and_fwd_bwd(
            lambda: chunk_kda(
                *x, autotune=True, kernel_options=chunk_kernel_options(x[0])
            )[0],
            do,
        )
        del x, do
        results.append(r)
        if rank == 0:
            print(json.dumps(r), flush=True)
    if rank == 0:
        gpu = torch.cuda.get_device_name().replace("NVIDIA ", "").replace(" ", "_")
        out = Path(__file__).with_name("bench") / f"kda_cp_{gpu}_cp{cp}.json"
        out.write_text(
            json.dumps(
                {"gpu": torch.cuda.get_device_name(), "results": results}, indent=1
            )
        )
        print(f"Wrote {out}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
