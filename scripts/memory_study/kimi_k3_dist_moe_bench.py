# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time DistMoE's routed-expert op at Kimi K3's shapes, for the roofline.

Each rank owns the same number of local experts as a Kimi K3 config (14 at
EP 64, 28 at EP 32), so with balanced routing every expert receives
local tokens x top-k / local experts tokens -- the same per-expert GEMM size
and per-GPU work as the real config, whatever the world size here. DistMoE
is left at its default (activation slots unset: saves inputs, recomputes in
backward), BF16 or MXFP8, with SwiGLU standing in for SiTU-GLU (same GEMMs).
Times include DistMoE's fused dispatch and combine over this host's NVLink.

Run (writes bench/dist_moe_<gpu>_ep<world>.json):
  torchrun --standalone --nproc-per-node=4 scripts/memory_study/kimi_k3_dist_moe_bench.py
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

HIDDEN, INTERMEDIATE, TOP_K = 3584, 3072, 16  # Kimi K3: L, Fe, routed top-k
LOCAL_TOKENS = (1024, 2048, 4096, 8192)
LOCAL_EXPERTS = {"ep64": 14, "ep32": 28}  # 896 / EP


def time_ms(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    dist.barrier()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(
    dist_moe, device, group, *, local_tokens: int, local_experts: int, mxfp8: bool
) -> dict:
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    num_experts = local_experts * world
    torch.manual_seed(17 + rank)
    x = torch.randn(
        local_tokens, HIDDEN, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    # balanced routing: global token g picks experts g*k .. g*k+k-1 (mod E)
    g = rank * local_tokens + torch.arange(local_tokens, device=device)
    ids = (
        g[:, None] * TOP_K + torch.arange(TOP_K, device=device)[None, :]
    ) % num_experts
    scores = torch.full(
        (local_tokens, TOP_K),
        1.0 / TOP_K,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    w13 = (
        (0.02 * torch.randn(local_experts, 2 * INTERMEDIATE, HIDDEN, device=device))
        .bfloat16()
        .requires_grad_()
    )
    w2 = (
        (0.02 * torch.randn(local_experts, HIDDEN, INTERMEDIATE, device=device))
        .bfloat16()
        .requires_grad_()
    )
    config = dist_moe.Config(
        max_num_local_input_tokens=local_tokens,
        hidden_dim=HIDDEN,
        intermediate_dim=INTERMEDIATE,
        top_k=TOP_K,
        num_experts=num_experts,
        max_moe_layers_per_activation_slot=1,
        device_scratch_capacity_factor=1.0,  # balanced routing
        activation_slot_capacity_factor=None,  # default: save inputs, recompute
        num_activation_slots=1,
        block_scaled=dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3
        )
        if mxfp8
        else None,
        activation="swiglu",
    )
    context = dist_moe.create_context(group=group, config=config, device=device)
    try:
        context.select_activation_slot(activation_slot=0, num_moe_layers_in_slot=1)

        def fwd():
            with torch.no_grad():
                dist_moe.routed_experts(x, ids, scores, w13, w2, context)

        def fwd_bwd():
            dist_moe.routed_experts(
                x, ids, scores, w13, w2, context
            ).float().sum().backward()

        f_ms, fb_ms = time_ms(fwd), time_ms(fwd_bwd)
    finally:
        context.close()
    n_e = local_tokens * TOP_K / local_experts
    params = 3 * HIDDEN * INTERMEDIATE  # w13 (2 L Fe) + w2 (Fe L) per expert
    fwd_flops = 2 * local_tokens * TOP_K * params
    return {
        "local_tokens": local_tokens,
        "local_experts": local_experts,
        "tokens_per_expert": n_e,
        "dtype": "mxfp8" if mxfp8 else "bf16",
        "fwd_ms": f_ms,
        "fwd_bwd_ms": fb_ms,
        "fwd_tflops": fwd_flops / f_ms / 1e9,
        # forward + recompute of both forwards + two backward GEMMs = 4x forward FLOPs
        "fwd_bwd_tflops": 4 * fwd_flops / fb_ms / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dist-moe-path",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "dist_moe",
    )
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(args.dist_moe_path))
    import dist_moe

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    results = []
    try:
        for name, local_experts in LOCAL_EXPERTS.items():
            for mxfp8 in (False, True):
                for tokens in LOCAL_TOKENS:
                    try:
                        r = run_case(
                            dist_moe,
                            device,
                            group,
                            local_tokens=tokens,
                            local_experts=local_experts,
                            mxfp8=mxfp8,
                        )
                        r["layout"] = name
                    except Exception as e:  # record and keep going; one shape failing should not lose the rest
                        r = {
                            "layout": name,
                            "local_tokens": tokens,
                            "dtype": "mxfp8" if mxfp8 else "bf16",
                            "error": repr(e)[:300],
                        }
                    results.append(r)
                    if dist.get_rank() == 0:
                        print(json.dumps(r), flush=True)
    finally:
        if dist.get_rank() == 0:
            gpu = torch.cuda.get_device_name().replace("NVIDIA ", "").replace(" ", "_")
            out = (
                Path(__file__).with_name("bench")
                / f"dist_moe_{gpu}_ep{dist.get_world_size()}.json"
            )
            out.write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "world_size": dist.get_world_size(),
                        "results": results,
                    },
                    indent=1,
                )
            )
            print(f"Wrote {out}", flush=True)
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
