# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What does DistMoE's internal recompute cost? One Kimi K3 routed-experts
layer (latent dim 3584, 896 experts, top-16, MXFP8 blockscaled training) at
EP 64 under a fake process group on one GPU, timed forward + backward at
several activation_slot_capacity_factor settings:

  None -> save the expert inputs only; backward recomputes w13 + SwiGLU;
  f    -> keep the intermediates in slot buffers sized f x balanced rows
          (rows beyond the slot are recomputed).

Reports per-layer forward / backward time (CUDA events, median of --iters
after --warmup) and the bytes the context allocates up front (slots + scratch).

Usage:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<dist_moe> python kimi_k3_dist_moe_slots.py \\
    --tokens 8192 --slots none 0.5 1.25 --out bench/dist_moe_slots_8k.json

With --real, run under torchrun with a real process group (EP = the world
size), so dispatch and combine move data between GPUs instead of aliasing
fake peers:
  PYTHONPATH=<dist_moe> torchrun --nproc-per-node=4 kimi_k3_dist_moe_slots.py \\
    --real --tokens 8192 --slots none 1.0 --out bench/dist_moe_slots_8k_real_ep4.json
"""

import os

import argparse
import json
import statistics

import dist_moe
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

NUM_EXPERTS, TOP_K, DIM, INTER = 896, 16, 3584, 3072  # Kimi K3 latent MoE


def run(tokens: int, ep: int, slot: float | None, iters: int, warmup: int) -> dict:
    dev = torch.device("cuda", torch.cuda.current_device())
    local = NUM_EXPERTS // ep
    torch.manual_seed(0)
    x = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=dev, requires_grad=True)
    t = torch.arange(tokens, device=dev)[:, None]
    ids = (t * TOP_K + torch.arange(TOP_K, device=dev)[None, :]) % NUM_EXPERTS  # balanced
    scores = torch.full((tokens, TOP_K), 1.0 / TOP_K, device=dev, requires_grad=True)
    w13 = torch.randn(local, 2 * INTER, DIM, dtype=torch.bfloat16, device=dev, requires_grad=True)
    w2 = torch.randn(local, DIM, INTER, dtype=torch.bfloat16, device=dev, requires_grad=True)
    grad_out = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=dev)
    config = dist_moe.Config(
        max_num_local_input_tokens=tokens,
        hidden_dim=DIM,
        intermediate_dim=INTER,
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        max_moe_layers_per_activation_slot=1,
        device_scratch_capacity_factor=4.0,  # the study's recipe value
        activation_slot_capacity_factor=slot,
        num_activation_slots=1,
        block_scaled=dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3, pipeline="staged"
        ),
    )
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    ctx = dist_moe.create_context(group=dist.group.WORLD, config=config, device=dev)
    context_bytes = torch.cuda.memory_allocated() - base
    fwd, bwd = [], []
    try:
        for i in range(warmup + iters):
            for p in (x, scores, w13, w2):
                p.grad = None
            e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            e[0].record()
            out = dist_moe.routed_experts(x, ids, scores, w13, w2, ctx)
            e[1].record()
            out.backward(grad_out)
            e[2].record()
            torch.cuda.synchronize()
            dist.barrier()
            if i >= warmup:
                fwd.append(e[0].elapsed_time(e[1]))
                bwd.append(e[1].elapsed_time(e[2]))
    finally:
        ctx.close()
    return {"tokens": tokens, "ep": ep, "slot": slot, "context_bytes": context_bytes,
            "fwd_ms": statistics.median(fwd), "bwd_ms": statistics.median(bwd)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--ep", type=int, default=64)
    p.add_argument("--slots", nargs="+", default=["none", "0.5", "1.25"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--real", action="store_true", help="real process group under torchrun; EP = world size")
    a = p.parse_args()
    if a.real:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        a.ep = dist.get_world_size()
    else:
        dist.init_process_group("fake", rank=0, world_size=a.ep, store=FakeStore())
    rank0 = dist.get_rank() == 0
    rows = []
    for s in a.slots:
        slot = None if s == "none" else float(s)
        r = run(a.tokens, a.ep, slot, a.iters, a.warmup)
        rows.append(r)
        if rank0:
            print(f"tokens {r['tokens']} slot {str(slot):5s}: fwd {r['fwd_ms']:7.2f} ms, bwd {r['bwd_ms']:7.2f} ms, "
                  f"context {r['context_bytes'] / 2**30:6.2f} GiB", flush=True)
    dist.destroy_process_group()
    if a.out and rank0:
        with open(a.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(), "process_group": "real" if a.real else "fake",
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
