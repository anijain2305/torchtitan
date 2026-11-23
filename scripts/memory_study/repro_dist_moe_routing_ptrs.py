# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Repro: DistMoE saves worst-case-sized routing pointers for backward.

One Kimi K3 routed-experts layer (latent dim 3584, 896 experts, top-16, MXFP8
blockscaled training) at EP 64 under a fake process group on one GPU. After one
forward, list every int64 tensor the layer keeps for backward. The routing
pointer arrays are sized for the worst case (EP x local tokens x min(top-k,
local experts)), independent of how many rows the rank actually receives, so
fake communication does not change their size.

Usage:
  CUDA_VISIBLE_DEVICES=0 python repro_dist_moe_routing_ptrs.py [--ep 64] [--tokens 8192]
  (PYTHONPATH must include the dist_moe checkout)
"""

import argparse

import dist_moe
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore


def saved_tensors(fn) -> list[torch.Tensor]:
    """Every tensor saved by the autograd graph reachable from fn."""
    out, seen, stack = [], set(), [fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        for t in getattr(node, "saved_tensors", ()) or ():
            if isinstance(t, torch.Tensor):
                out.append(t)
        stack.extend(n for n, _ in node.next_functions)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ep", type=int, default=64)
    p.add_argument("--tokens", type=int, default=8192, help="local tokens per microbatch")
    p.add_argument("--moe-layers", type=int, default=92)
    a = p.parse_args()

    num_experts, top_k, dim, inter = 896, 16, 3584, 3072  # Kimi K3 latent MoE
    local_experts = num_experts // a.ep
    dist.init_process_group("fake", rank=0, world_size=a.ep, store=FakeStore())
    dev = torch.device("cuda")
    torch.manual_seed(0)

    x_TD = torch.randn(a.tokens, dim, dtype=torch.bfloat16, device=dev, requires_grad=True)
    # balanced routing: token t picks experts t*k .. t*k+k-1 (mod 896)
    t = torch.arange(a.tokens, device=dev)[:, None]
    ids_TK = (t * top_k + torch.arange(top_k, device=dev)[None, :]) % num_experts
    scores_TK = torch.full((a.tokens, top_k), 1.0 / top_k, device=dev, requires_grad=True)
    w13 = torch.randn(local_experts, 2 * inter, dim, dtype=torch.bfloat16, device=dev, requires_grad=True)
    w2 = torch.randn(local_experts, dim, inter, dtype=torch.bfloat16, device=dev, requires_grad=True)

    config = dist_moe.Config(
        max_num_local_input_tokens=a.tokens,
        hidden_dim=dim,
        intermediate_dim=inter,
        top_k=top_k,
        num_experts=num_experts,
        max_moe_layers_per_activation_slot=1,
        device_scratch_capacity_factor=4.0,  # the study's recipe value
        activation_slot_capacity_factor=None,
        num_activation_slots=1,
        block_scaled=dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3, pipeline="staged"
        ),
    )
    ctx = dist_moe.create_context(group=dist.group.WORLD, config=config, device=dev)
    try:
        before = torch.cuda.memory_allocated()
        out = dist_moe.routed_experts(x_TD, ids_TK, scores_TK, w13, w2, ctx)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before - out.numel() * out.element_size()

        ints = [s for s in saved_tensors(out.grad_fn) if s.dtype == torch.int64 and s.numel() > 1024]
        ptrs = [s for s in ints if s.dim() == 1]  # routing pointer arrays (fwd gather, bwd gather, scatter)
        ptr_bytes = sum(s.numel() * 8 for s in ptrs)
        worst = a.ep * a.tokens * min(top_k, local_experts)
        balanced = a.tokens * top_k  # rows a rank receives on average
        print(f"EP {a.ep}, {local_experts} local experts, {a.tokens} local tokens, top-{top_k}")
        for s in ints:
            print(f"  saved int64 {tuple(s.shape)}: {s.numel() * 8 / 2**20:8.1f} MiB")
        print(f"  pointer arrays: {ptr_bytes / 2**20:.1f} MiB per layer, "
              f"{ptr_bytes * a.moe_layers / 2**30:.1f} GiB over {a.moe_layers} MoE layers")
        print(f"  entries per array: {ptrs[0].numel() if ptrs else 0:,} (worst case {worst:,}); "
              f"balanced routing receives ~{balanced:,} rows -> {worst / balanced:.0f}x oversized")
        print(f"  allocator growth across the forward (excl. output; includes other saves): {held / 2**20:.1f} MiB")
    finally:
        ctx.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
