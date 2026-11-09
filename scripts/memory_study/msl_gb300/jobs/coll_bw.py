# Bus bandwidth of NCCL collectives vs message size across every GPU of the job
# (nccl-tests' busbw convention), no compute. Prints one JSON line per row.
import json, os, sys, torch, torch.distributed as dist

dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
tag = os.environ.get("BW_TAG", "")
sizes = [1 << p for p in range(20, 32)]  # 1 MiB .. 2 GiB total
bf = torch.bfloat16


def bench(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) / iters], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() * 1e-3


for size in sizes:
    n = size // 2
    if n % w:
        continue
    full = torch.empty(n, device="cuda", dtype=bf); shard = torch.empty(n // w, device="cuda", dtype=bf)
    a2a_out = torch.empty(n, device="cuda", dtype=bf)
    half = w // 2
    ops = {
        "allgather": (lambda: dist.all_gather_into_tensor(full, shard), (w - 1) / w),
        "reducescatter": (lambda: dist.reduce_scatter_tensor(shard, full), (w - 1) / w),
        "alltoall": (lambda: dist.all_to_all_single(a2a_out, full), (w - 1) / w),
        "sendrecv": (lambda: (dist.send(full, r + half) if r < half else dist.recv(full, r - half)), 1.0),
    }
    only = os.environ.get("BW_OPS")  # comma-separated subset of ops
    for op, (fn, factor) in ops.items():
        if only and op not in only.split(","):
            continue
        iters = max(5, min(50, int(2e9 / size)))
        t = bench(fn, iters)
        if r == 0:
            print("BW " + json.dumps({"tag": tag, "world": w, "op": op, "bytes": size, "s": t,
                                      "busbw_GBps": size * factor / t / 1e9}), flush=True)
    del full, shard, a2a_out
    torch.cuda.empty_cache()
dist.destroy_process_group()
