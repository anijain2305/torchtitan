# The 72-GPU GB300 cluster (MSL tenant `msl_infra_pytorch_dct`)

18 Nebius GB300 hosts x 4 GPUs = 72 GPUs in one NVL72 rack, with InfiniBand
between hosts as well. This is how to run PyTorch on it. `NOTES.md` has the
job-by-job log of how the recipe was found; the "Verified" column below says
when each fact was last observed.

## 1. What it is

| Property | Value | Verified |
|---|---|---|
| Tenant path | `root/gen_ai/msl/fair_research/fair_prod/FAIR Foundation/MSL_Infra/msl_infra_pytorch_dct` | 2026-10-30 |
| ACL / owner / oncall | `msl_infra_pytorch_dev` / `jathu` / `pytorch_release_engineering` | 2026-10-30 |
| Quota | 18 x `T23_NEBIUS_GB300_288GB_HBM3E_IB_XDR` hosts = 72 GPUs (288 GB HBM3e each) | 2026-10-30 |
| Shared with | other MSL users: on 2026-11-09, 15 of 18 hosts were busy (yichuan 6, jessicazhong 6, anshulsi 3) | 2026-11-09 |
| Region / pool | `ofn` (`*.ofn1.tw.fbinfra.net`), pool `mast_gen_ai_t23_ofn_0` | 2026-10-30 |
| Per host | 4 GB300, 144 CPUs (aarch64 Grace), ~845 GB RAM | 2026-10-30 |
| NVLink | NVL72 switch fabric; multi-host NVLink (MNNVL) only when the job asks for one domain (section 4) | 16 and 32 GPUs, 2026-10-31 |
| InfiniBand | 4 x 800 Gb/s XDR per host (`mlx5_0..3`, one per GPU) | 2026-10-30 |
| Ethernet | 2 x 400 Gb/s (`mlx5_4..5`), frontend only, not for NCCL | 2026-10-30 |

Measured (`../bench/`): 4-GPU NVLink all-to-all about 520 GB/s bus bandwidth
(MSL health check). Across 4 hosts over IB: all-gather 371-378, reduce-scatter
354-361, all-to-all 115 and send/recv 90 GB/s per GPU at large messages
(`ib_collectives_GB300_4hosts.json`). NVLink across hosts:
`nvl_collectives_GB300_{4,8}hosts.json`.

## 2. Before submitting

Check who is using it (multi-host jobs queue behind running ones):

```bash
T='root/gen_ai/msl/fair_research/fair_prod/FAIR Foundation/MSL_Infra/msl_infra_pytorch_dct'
meta ai.msl-capacity list-job-queue --tenant-path="$T" --group-by=owner
```

Access: `meta ai.msl-capacity list-tenants --can-submit --root-path=gen_ai/msl`
lists the tenant if you can submit. `meta ai.mast-job get-permissions` says
`submit_job: no` for MSL tenants; that is wrong, ignore it.

## 3. One-time setup on the devserver

The hosts have no internet and the PyTorch fbpkg `torchtitan_conda_prod` is
x86-64, so jobs ship a small aarch64 conda env, packed by the `msl_conda`
scheduler into an ephemeral fbpkg under the shared `metaconda_demo` package (no
extra permission needed).

```bash
conda create -y -n msl_bench python=3.12
export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
~/.conda/envs/msl_bench/bin/pip install --pre torch numpy nvidia-ml-py \
  --index-url https://download.pytorch.org/whl/nightly/cu130 --extra-index-url https://pypi.org/simple
```

`torchx` (`/usr/local/bin/torchx`, with the `msl` / `msl_conda` schedulers)
comes with the devserver.

## 4. Submitting a job

```bash
cd /tmp   # run torchx from a clean directory
[DOMAIN=1] ~/local/torchtitan/scripts/memory_study/msl_gb300/submit.sh <num_hosts> <script in jobs/> [--dryrun]
# e.g. a 4-host smoke test, all hosts in one NVLink domain:
DOMAIN=1 ~/local/torchtitan/scripts/memory_study/msl_gb300/submit.sh 4 run_smoke.sh --dryrun
DOMAIN=1 ~/local/torchtitan/scripts/memory_study/msl_gb300/submit.sh 4 run_smoke.sh
# prints msl_conda://torchx/<job>; UI: https://msl-runs.internalmeta.com/mast/<job>
```

What `submit.sh` does: activates `msl_bench`, copies `jobs/*` (plus
`../kimi_k3_overlap_bench.py` and `nvlink_fabric_info.py`) into
`$CONDA_PREFIX/msl_ws/`, and runs `jobs/raw_sh_env.py:raw_sh_env` with the
`msl_conda` scheduler, which runs `bash <script>` on every host. Always
`--dryrun` first and check `taskCount`, `serverSubTypes` and, with `DOMAIN=1`,
`topologyConstraints`.

- `DOMAIN=1` adds a MAST `topologyConstraints.domain` constraint (one group,
  all hosts), so all hosts land in one NVL72 domain and NCCL uses multi-host
  NVLink. Without it, hosts can land in different domains, and default NCCL
  hangs (use `NCCL_MNNVL_ENABLE=0` for an InfiniBand-only job).
- Scheduler args used (in `submit.sh`): `hpcIdentity=pytorch_distributed,
  hpcClusterUuid=MastGenAICluster, rmAttribution=msl_infra_pytorch_dct,
  hpcJobOncall=pytorch_distributed, localityConstraints=region;ofn,
  jobPriority=LOW, conda_fbpkg_name=metaconda_demo, activate_conda=True,
  workspace_dir=/packages/metaconda_demo/conda/msl_ws`. Keep `jobPriority=LOW`:
  this is shared team capacity.
- Named resource `nebius_gb300` = one whole host. Jobs run as root
  (`run_as_root`, needed for RDMA device enumeration) with `DISABLE_OILFS=1`
  (region `ofn` has no warm storage).

### Writing a job script

Start every script with `source ./gpu_env.sh`. It fixes the loader path (section
6) and sets the rendezvous variables `MASTER`, `NN` (hosts) and `NODE_RANK`:

```bash
cd "$(dirname "$0")"; source ./gpu_env.sh
NCCL_DEBUG=WARN torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK \
  --master-addr $MASTER --master-port 29600 my_bench.py
```

Put `my_bench.py` in `jobs/` too (everything in `jobs/` is shipped). Examples:
`run_smoke.sh` (GPUs + NCCL transport), `run_nvl.sh` (NVLink collectives +
overlap), `run_bw.sh` / `run_ib_scale.sh` (IB collectives), `run_oneway.sh`
(PP-hop overlap), `run_island8.sh` (NVLink islands with IB between them).

### Monitoring

```bash
torchx status msl_conda://torchx/<job>
torchx log --streams stdout msl_conda://torchx/<job>   # your script's output (only here)
meta ai.mast-job list-tasks --name=<job>                # host + exit code per task
torchx cancel msl_conda://torchx/<job>
```

Hosts spend about 1 minute in MSL health checks before the script starts.

## 5. NVLink or InfiniBand between hosts

| Wanted | How |
|---|---|
| Inter-host NVLink (MNNVL) | `DOMAIN=1` and default NCCL (`P2P/MNNVL` in the NCCL log) |
| Inter-host InfiniBand | `NCCL_MNNVL_ENABLE=0` (NCCL uses `mlx5_0..3`) |
| NVLink islands with IB between them | `DOMAIN=1` plus `NCCL_MNNVL_CLIQUE_ID` per island (`run_island8.sh`) |

Verify every run instead of assuming: `nvlink_fabric_info.py` prints each GPU's
fabric `cluster` + `clique` (the same on all hosts = one NVLink domain; the
image has no `nvidia-smi`), and `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`
shows `via NET/IB` vs `MNNVL` / `P2P`. Use `NCCL_DEBUG=WARN` for timed runs
(the host `/etc/nccl.conf` sets INFO).

## 6. Pitfalls already solved

| Symptom | Fix (in place) |
|---|---|
| `Exec format error` | `torchtitan_conda_prod` is x86-64; use the `msl_bench` env via `msl_conda` |
| `WarmStorageSpec ... missing clusters field` at submit | role env `DISABLE_OILFS=1` (`raw_sh_env.py`) |
| `torch.cuda.device_count() == 0` | `gpu_env.sh` links `libcuda*` / `libnvidia*` from `/usr/local/fbcode/platform010-aarch64/lib` onto `LD_LIBRARY_PATH` |
| NCCL `Failed to initialize any NET plugin`, verbs sees 0 devices | the system `libmlx5-rdmav57.so` is a dangling symlink; `gpu_env.sh` puts the platform dir's `libibverbs.so.1.14.57.0` + `libmlx5.so.1.25.57.0` first on the path |
| Multi-host NCCL hangs | hosts are in different NVLink domains: submit with `DOMAIN=1`, or run with `NCCL_MNNVL_ENABLE=0` |
| Wrong node ranks | take the rank from the host's position in `MAST_HPC_TASK_GROUP_HOSTNAMES` (`gpu_env.sh`), not `TW_TASK_ID` |
| Whole job abandoned | a non-master host exiting non-zero kills the job; don't gate on output only rank 0 prints |
| `torchx utils.sh` scripts exit 127 | it quotes the joined args into one word; `raw_sh_env.py` runs `/bin/bash -c <script>` instead |
| `submit.sh` packs the wrong env | it must run with `msl_bench` active (it activates it and checks `$CONDA_PREFIX`) |

## 7. Not verified yet

- All 18 hosts in one NVLink domain: largest verified is 8 hosts / 32 GPUs.
  A full 72-GPU job needs the tenant idle.
- TorchTitan training itself on this cluster: so far only benchmarks
  (collectives, overlap, PP hops) have run here. A TorchTitan run needs the
  `torchtitan` package and its dependencies (e.g. DistMoE) in `msl_bench`.
