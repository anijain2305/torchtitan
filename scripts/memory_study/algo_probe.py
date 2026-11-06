# Which algorithm / protocol / channel count NCCL picks for large all-gathers
# and reduce-scatters across every GPU of the job (read from NCCL's log).
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for mb in (64, 1024):
    n = mb * (1 << 20) // 2
    full = torch.empty(n // w * w, device="cuda", dtype=torch.bfloat16)
    shard = torch.empty(n // w, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        dist.all_gather_into_tensor(full, shard)
        dist.reduce_scatter_tensor(shard, full)
torch.cuda.synchronize(); dist.barrier()
if r == 0: print("ALGO_PROBE OK world", w, flush=True)
dist.destroy_process_group()
