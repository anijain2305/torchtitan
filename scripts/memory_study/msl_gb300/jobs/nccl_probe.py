# Tiny cross-host NCCL check: one all-gather and one send/recv between rank i and i + world/2.
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(1 << 20, device="cuda"); out = torch.empty(w << 20, device="cuda")
dist.all_gather_into_tensor(out, x)
half = w // 2
(dist.send(x, r + half) if r < half else dist.recv(x, r - half))
torch.cuda.synchronize(); dist.barrier()
if r == 0: print("PROBE OK world", w, "MNNVL", os.environ.get("NCCL_MNNVL_ENABLE", "default"), flush=True)
dist.destroy_process_group()
