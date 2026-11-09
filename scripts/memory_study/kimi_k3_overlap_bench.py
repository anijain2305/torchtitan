# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How much does a running NCCL collective slow down the compute it overlaps,
and how much does the compute slow down the collective?

On every rank (real NCCL over NVLink), for each compute kernel and each
collective:
  1. time the compute alone and the collective alone;
  2. enqueue enough back-to-back collectives on a side stream to outlast the
     compute, then run the compute on the main stream, and time both;
  3. read the collective kernel's thread-block count (NCCL channels, one SM
     each) from a profiler trace.

Compute kernels: an MXFP8 GEMM at a Kimi K3 projection shape (pre-quantized
operands), a bf16 GEMM of the same shape, and a memory-bound elementwise
kernel. Collectives: all-gather, reduce-scatter (bf16), all-to-all, and a
ring send/recv, at a few message sizes. Run once per NCCL_MAX_NCHANNELS value
to see the trade-off between communication speed and compute slowdown.

Usage (4 GPUs):
  NCCL_MAX_NCHANNELS=<n> PYTHONPATH=.:scripts/memory_study torchrun --standalone \\
      --nproc-per-node=4 scripts/memory_study/kimi_k3_overlap_bench.py --out bench/overlap_<n>.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

M, K, N = 8192, 7168, 12288  # one KDA projection at 8K tokens
MEM_ELEMS = 512 * 1024 * 1024  # bf16 elementwise operands, 1 GiB each
SIZES_MB = (64, 256, 1024)  # per-rank buffer (all-gather: per-rank shard x world)


def timed(stream, fn, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            fn()
        end.record(stream)
    return start, end


def build_compute():
    dev, bf = "cuda", torch.bfloat16
    x = torch.randn(M, K, device=dev, dtype=bf)
    w = torch.randn(N, K, device=dev, dtype=bf)
    a = torch.randn(MEM_ELEMS, device=dev, dtype=bf)
    b = torch.randn(MEM_ELEMS, device=dev, dtype=bf)
    c = torch.empty_like(a)
    out = {}
    try:  # MXFP8 needs torchao; the cluster image may not have it
        from kimi_k3_kernel_bench import to_mx

        x_q, w_q = to_mx(x), to_mx(w)
        out["mxfp8_gemm"] = (lambda: torch.mm(x_q, w_q.t()), 2 * M * K * N, 0)
    except ImportError:
        pass
    out["bf16_gemm"] = (lambda: x @ w.t(), 2 * M * K * N, 0)
    out["memory_bound"] = (lambda: torch.add(a, b, alpha=0.5, out=c), 0, 3 * 2 * MEM_ELEMS)
    return out


def build_comm(size_mb: int, world: int, rank: int, group=None):
    """Collectives over `group` (default: every GPU) of `world` ranks."""
    dev, bf = "cuda", torch.bfloat16
    n = size_mb * 1024 * 1024 // 2
    shard = torch.randn(n // world, device=dev, dtype=bf)
    full = torch.empty(n // world * world, device=dev, dtype=bf)
    big = torch.randn(n, device=dev, dtype=bf)
    out_rs = torch.empty(n // world, device=dev, dtype=bf)
    a2a_in, a2a_out = torch.randn(n, device=dev, dtype=bf), torch.empty(n, device=dev, dtype=bf)
    send, recv = torch.randn(n, device=dev, dtype=bf), torch.empty(n, device=dev, dtype=bf)
    nxt, prv = (rank + 1) % world, (rank - 1) % world

    def sendrecv():
        ops = [dist.P2POp(dist.isend, send, nxt), dist.P2POp(dist.irecv, recv, prv)]
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    return {
        "all_gather": lambda: dist.all_gather_into_tensor(full, shard, group=group),
        "reduce_scatter": lambda: dist.reduce_scatter_tensor(out_rs, big, group=group),
        "all_to_all": lambda: dist.all_to_all_single(a2a_out, a2a_in),
        "send_recv": sendrecv,
    }


def build_one_way(size_mb: int, rank: int, world: int):
    """The first half of the ranks only send, each to its partner in the
    second half (rank + world / 2), which only receives: the sender's and the
    receiver's costs, measured separately. With 2 hosts x 4 GPUs, rank i on host
    0 sends to rank i on host 1, like a PP hop."""
    n = size_mb * 1024 * 1024 // 2
    buf = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    half = world // 2
    if rank < half:
        return {"send_only": lambda: dist.send(buf, rank + half)}
    return {"recv_only": lambda: dist.recv(buf, rank - half)}


def nccl_blocks(fn) -> int | None:
    """Thread blocks (one per channel, one SM each) of the collective's kernel."""
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = f"/tmp/overlap_trace_{os.getpid()}.json"
    prof.export_chrome_trace(path)
    blocks = None
    for e in json.load(open(path))["traceEvents"]:
        if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower():
            g = e.get("args", {}).get("grid", [0, 1, 1])
            blocks = max(blocks or 0, g[0] * g[1] * g[2])
    os.remove(path)
    return blocks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="one small size, few iterations")
    p.add_argument("--one-way", action="store_true",
                   help="first half of the ranks only send to the second half, which only receives")
    p.add_argument("--sizes-mb", type=int, nargs="+", default=None)
    p.add_argument("--rail", action="store_true",
                   help="collectives in groups of one GPU per host (all groups at once)")
    p.add_argument("--collectives", nargs="+", default=None,
                   help="only these collectives (default: all of the mode's)")
    args = p.parse_args()
    global SIZES_MB
    if args.sizes_mb:
        SIZES_MB = tuple(args.sizes_mb)
    if args.smoke:
        SIZES_MB = (16,)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if rank == 0:
        print(f"torch {torch.__version__} nccl {torch.cuda.nccl.version()} world {world} "
              f"MNNVL={os.environ.get('NCCL_MNNVL_ENABLE', 'default')}", flush=True)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rail_group, n_local = None, int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    n_hosts = world // n_local
    if args.rail:  # every rank creates every group, in the same order
        groups = [dist.new_group([h * n_local + j for h in range(n_hosts)]) for j in range(n_local)]
        rail_group = groups[rank % n_local]
    main_s, comm_s = torch.cuda.current_stream(), torch.cuda.Stream()
    compute = build_compute()
    results = {"gpu": torch.cuda.get_device_name(), "sms": sms, "world": world,
               "nccl_max_nchannels": os.environ.get("NCCL_MAX_NCHANNELS", "default"), "rows": []}

    def alone(fn, stream, iters):
        """Solo time per call, agreed across ranks (rank 0's), so every rank
        sizes its loops the same way and issues the same collectives."""
        for _ in range(3):
            with torch.cuda.stream(stream):
                fn()
        torch.cuda.synchronize()
        dist.barrier()
        s, e = timed(stream, fn, iters)
        torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) / iters], device="cuda")
        dist.broadcast(t, 0)
        return t.item()

    for cname, (cfn, flops, nbytes) in compute.items():
        c_ms = alone(cfn, main_s, 20)
        c_iters = max(20, int(200 / c_ms))  # about 200 ms of compute
        for size in SIZES_MB:
            if args.one_way:
                comm = build_one_way(size, rank, world)
            elif args.rail:
                comm = build_comm(size, n_hosts, rank // n_local, rail_group)
            else:
                comm = build_comm(size, world, rank)
            if args.collectives:
                comm = {k: v for k, v in comm.items() if k in args.collectives}
            for kname, kfn in comm.items():
                if rank == 0:
                    print(f"  [{cname} / {kname} / {size} MB] timing alone", flush=True)
                k_ms = alone(kfn, comm_s, 10)
                blocks = nccl_blocks(kfn)
                k_iters = min(int(1.3 * c_iters * c_ms / k_ms) + 2, 2000)  # outlast the compute
                if k_iters * k_ms < 1.1 * c_iters * c_ms:  # too many tiny ops: shorten the compute instead
                    c_run = max(5, int(k_iters * k_ms / (1.3 * c_ms)))
                else:
                    c_run = c_iters
                torch.cuda.synchronize()
                dist.barrier()
                ks, ke = timed(comm_s, kfn, k_iters)
                cs, ce = timed(main_s, cfn, c_run)
                torch.cuda.synchronize()
                c_span = cs.elapsed_time(ce)
                c_ov = c_span / c_run
                # The collectives start with the compute and outlast it; after
                # the compute ends they run alone at the solo rate. So the ops
                # done during the overlap are the total minus those that fit in
                # the tail, and their rate gives the collective's slowdown.
                k_total = ks.elapsed_time(ke)
                tail_ops = max(0.0, k_total - c_span) / k_ms
                ops_overlapped = max(k_iters - tail_ops, 1e-9)
                row = {"compute": cname, "collective": kname, "size_mb": size, "nccl_blocks": blocks,
                       "compute_alone_ms": c_ms, "compute_overlapped_ms": c_ov,
                       "compute_slowdown": c_ov / c_ms, "comm_alone_ms": k_ms,
                       "comm_total_ms": k_total, "comm_iters": k_iters,
                       "comm_slowdown": min(c_span, k_total) / (ops_overlapped * k_ms)}
                row["rank"] = rank
                results["rows"].append(row)
                if rank == 0 or args.one_way:
                    print(f"[rank {rank}] {cname:12s} + {kname:14s} {size:5d} MB  blocks={blocks}  "
                          f"compute x{row['compute_slowdown']:.3f}  comm x{row['comm_slowdown']:.3f} "
                          f"(alone: compute {c_ms:.3f} ms, comm {k_ms:.3f} ms)", flush=True)
            del comm
            torch.cuda.empty_cache()
    gathered = [None] * world
    dist.all_gather_object(gathered, results["rows"])
    if rank == 0:
        results["rows"] = [r for rows in gathered for r in rows] if args.one_way else results["rows"]
        args.out.write_text(json.dumps(results, indent=1))
        print(f"Wrote {args.out}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
