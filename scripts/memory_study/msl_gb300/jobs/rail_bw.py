# All-gather / reduce-scatter bus bandwidth in groups of one GPU per host (the
# GPUs with the same local rank on every host), all groups at once: how an
# expert shard's FSDP replicas, one per NVLink domain, gather across InfiniBand.
import json, os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
nloc = int(os.environ["LOCAL_WORLD_SIZE"]); nn = w // nloc
groups = [dist.new_group([h * nloc + j for h in range(nn)]) for j in range(nloc)]
g = groups[lr]
tag = os.environ.get("BW_TAG", "")
bf = torch.bfloat16
for size in [1 << p for p in range(24, 31)]:  # 16 MB .. 1 GB
    n = size // 2
    full = torch.empty(n, device="cuda", dtype=bf); shard = torch.empty(n // nn, device="cuda", dtype=bf)
    for op, fn in (("allgather", lambda: dist.all_gather_into_tensor(full, shard, group=g)),
                   ("reducescatter", lambda: dist.reduce_scatter_tensor(shard, full, group=g))):
        for _ in range(3): fn()
        torch.cuda.synchronize(); dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        it = 10; s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) / it], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if r == 0:
            sec = t.item() * 1e-3
            print("RAIL " + json.dumps({"tag": tag, "group": nn, "op": op, "bytes": size, "s": sec,
                                        "busbw_GBps": size * (nn - 1) / nn / sec / 1e9}), flush=True)
dist.destroy_process_group()
