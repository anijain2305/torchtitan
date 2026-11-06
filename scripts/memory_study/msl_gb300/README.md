# Running jobs on the MSL Nebius GB300 tenant (`msl_infra_pytorch_dct`)

Notes for an agent that runs NVLink vs InfiniBand benchmarks (for example PP
send/recv SM contention) on GB300. Everything marked **verified** was
observed on 2026-10-30 from user `anijain`. Everything else is an inference or
an open question; check it before relying on it.

## 1. What the cluster is

| Property | Value | Source |
|---|---|---|
| Tenant | `root/gen_ai/msl/fair_research/fair_prod/FAIR Foundation/MSL_Infra/msl_infra_pytorch_dct` | `meta ai.msl-capacity describe-tenant` (verified) |
| Tenant ACL / owner / oncall | `msl_infra_pytorch_dev` / `jathu` / `pytorch_release_engineering` | same |
| Quota | `MACHINE_TYPE_T23_NEBIUS_GB300_288GB_HBM3E_IB_XDR_GENAI=18` -> 18 hosts x 4 GB300 = 72 GPUs | `meta ai.msl-capacity list-tenants --can-submit` (verified) |
| Over-quota borrowing | `ALLOW` (do not count on it) | describe-tenant |
| Region / host pool | `ofn` (hosts `*.ofn1.tw.fbinfra.net`), pool `mast_gen_ai_t23_ofn_0` | job env (verified) |
| CPU arch | aarch64 (Grace) | GB300 |
| Usage at time of writing | `yichuan` ran 6 one-host jobs; ~12 hosts free | `list-job-queue` |

Check current usage before submitting multi-host jobs:

```bash
T='root/gen_ai/msl/fair_research/fair_prod/FAIR Foundation/MSL_Infra/msl_infra_pytorch_dct'
meta ai.msl-capacity list-job-queue --tenant-path="$T" --group-by=owner
```

### Two networks per host (verified)

Each host has both fabrics. NVLink being present does not remove InfiniBand.

| Fabric | Hardware | Scope |
|---|---|---|
| NVLink (scale-up, MNNVL) | NVL72 NVLink switch fabric | GPUs in the same NVL72 domain (up to 18 hosts) |
| InfiniBand (scale-out) | `mlx5_0..mlx5_3`: 4 x 800 Gb/s 4X XDR, link layer `InfiniBand`, state ACTIVE (one per GPU) | Any host to any host, including across racks |
| Ethernet (frontend/storage) | `mlx5_4..mlx5_5`: 2 x 400 Gb/s, link layer `Ethernet` | Not for NCCL |

`/etc/nccl.conf` on the host sets `NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1`.

### Multi-host NVLink domain (verified 2026-10-31, 16 GPUs)

Job `torchx-raw-sh-rwghh7f3`: 4 hosts submitted with `--one_scaleup_domain True`
(see section 3) landed on `twshared1966/1969/1973/1979.01.ofn1`. All 16 GPUs
reported via NVML: fabric `state=3` (COMPLETED), `status=0`, the same
`clusterUuid=55b45f6f607442fb808c0ab73db42cbf` and `cliqueId=32766` -> one
NVLink domain. Scheduled ~1 min after submit.

**Not verified:** that all 18 quota hosts sit in one NVL72 domain ("18 hosts"
is a quota count), and that NCCL actually picks MNNVL inside the container
(IMEX channels, `/dev/nvidia-caps-imex-channels`, not checked;
`/etc/nvidia-imex` does not exist in the container). Without
`--one_scaleup_domain`, hosts of one job may land in different racks. Always
record each host's fabric IDs (section 4) and NCCL's transport line.

## 2. Access

- `meta ai.msl-capacity list-tenants --can-submit --root-path=gen_ai/msl`
  lists this tenant -> submit works (verified by successful jobs).
- `meta ai.mast-job get-permissions ... --entitlement=<tenant path>` reports
  `submit_job: no` and `get-acl-owner` says the entitlement is "not found".
  These MAST tools are wrong for MSL tenants; ignore them.
- `meta ai.capacity find-capacity` also cannot see this tenant.

## 3. Submitting a job

Scheduler args (derived from a working `yichuan` job and verified by our own
jobs):

```bash
SA='hpcIdentity=pytorch_distributed,hpcClusterUuid=MastGenAICluster,rmAttribution=msl_infra_pytorch_dct,hpcJobOncall=pytorch_distributed,localityConstraints=region;ofn,jobPriority=LOW'
```

- Scheduler is `msl` (not `mast`). Named resource `nebius_gb300` gives one
  whole host: 4 GPUs, 144 CPU, 865280 MB RAM, server subtype
  `T23_NEBIUS_GB300_288GB_HBM3E_IB_XDR`.
- Use `jobPriority=LOW` (or NORMAL) -- this is shared team capacity.
- Always `--dryrun` first and check `command`, `arguments`, `taskCount` and
  `serverSubTypes` in the printed spec.

### Run a shell script on N hosts

Use `raw_sh.py` in this directory. Do NOT use torchx `utils.sh`: it
shlex-quotes the joined args, so `a; b` becomes one word and bash exits 127
(this cost three failed jobs). Also do not rely on
`/packages/conda_mast_core/sh` -- it does not exist; use `/bin/bash`.

```bash
cd /tmp   # torchx run from a clean dir
torchx run --scheduler msl --scheduler_args "$SA" \
  ~/local/torchtitan/scripts/memory_study/msl_gb300/raw_sh.py:raw_sh \
  --num_replicas 2 \
  --script 'hostname; ls /sys/class/infiniband'
# prints msl://torchx/<job>; UI: https://msl-runs.internalmeta.com/mast/<job>
```

To force all hosts into one NVL72 domain add `--one_scaleup_domain True`. It
sets the MAST `topologyConstraints.domain` constraint (same mechanism as
`generator_in_one_scaleup_domain` in `fb/mast_rl/mast.py`; MAST rejects
SCALEUP_DOMAIN as a networkAffinity scope).

Verified example: `torchx-raw-sh-dvdjxl17` (1 host, SUCCEEDED, ~3.5 min wall
clock including ~1 min of MSL host health checks before the script runs).

### Monitoring and logs

```bash
torchx status msl://torchx/<job>
torchx log --streams stdout msl://torchx/<job>   # your script's stdout
torchx log msl://torchx/<job>                     # default: infra + stderr
meta ai.mast-job list-tasks --name=<job>          # host + exit code per task
meta ai.mast-job show-definition --name=<job>     # what MSL actually scheduled
```

Script stdout is ONLY in `--streams stdout`; the default stream is mostly
Tupperware/health-check noise. Lines are prefixed `sh/<task_id>`.

The job env contains `MAST_HPC_TASK_GROUP_HOSTNAMES` (comma-separated, all
hosts of the role) and `TW_TASK_ID` (0..N-1) -- use them for
`MASTER_ADDR`/node rank in a torchrun rendezvous.

## 4. NVLink vs InfiniBand between two hosts

By default NCCL uses MNNVL (NVLink) for inter-host traffic when both hosts are
in the same NVL72 domain, and IB otherwise.

| Path wanted | Env |
|---|---|
| Inter-host over NVLink | default (requires same NVLink clique) |
| Inter-host over InfiniBand | `NCCL_MNNVL_ENABLE=0` (NCCL then uses the `mlx5_0..3` IB HCAs) |

Intra-host GPU pairs always use NVLink regardless; to isolate the inter-host
path, pair rank `i` on host 0 with rank `i` on host 1.

Verify -- do not assume -- the path for every run:

1. Same NVLink domain? Run `nvlink_fabric_info.py` (this directory) on each
   host with `/usr/bin/python3`; same `cluster` + `clique` on all hosts = same
   NVL72 domain. The bare `conda_mast_core` image has NO `nvidia-smi`
   anywhere; NVML is at
   `/usr/local/fbcode/platform010-aarch64/lib/libnvidia-ml.so.1`. To ship the
   script without a package, base64 it into the `--script` string:
   ```bash
   B=$(base64 -w0 nvlink_fabric_info.py)
   ... --script "echo $B | base64 -d > /tmp/f.py; /usr/bin/python3 /tmp/f.py"
   ```
   In a PyTorch image, `nvidia-smi -q | grep -iE 'ClusterUUID|CliqueId'` may
   also work.
2. Which transport NCCL picked: run with
   `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` and look for `via NET/IB/...`
   (InfiniBand) vs `MNNVL` / `P2P/...` (NVLink) on the inter-host
   connections. Record the line in the results.
3. If the two hosts landed in different cliques, the default run is already
   IB, and you cannot measure inter-host NVLink with that pair -- resubmit.

## 5. Open items for the benchmarking agent

- **Image with PyTorch.** `conda_mast_core:stable` has no PyTorch. Options:
  - `mast_conda` / monarch `AllRolesWorkspace` workflow as used by
    `fbcode/pytorch/torchtitan/fb/mast_rl/mast.py` (packs the active aarch64
    conda env + workspace into an ephemeral fbpkg; needs write access to the
    target fbpkg, e.g. `torchtitan_conda_prod`).
  - `fbcode/pytorch/torchtitan/fb/six_cluster/configs/gb300.py` (six cluster
    tooling; its config targets `msl_infra_pytorch_dev`/`gb300_genai` and
    region `nao`, so change tenant/region/host type).
  Neither has been tried on this tenant yet.
- **Benchmark to adapt:** `scripts/memory_study/kimi_k3_overlap_bench.py`
  (single-host NVLink compute vs collective overlap; contains MXFP8/bf16 GEMM,
  memory-bound kernel, ring send/recv, thread-block count from profiler
  trace). For PP: 2 hosts x 4 GPUs, cross-host pairs `i <-> i+4`,
  PP-activation-sized messages, sweep `NCCL_MAX_NCHANNELS` and
  `NCCL_NCHANNELS_PER_NET_PEER`, run once default (NVLink) and once with
  `NCCL_MNNVL_ENABLE=0` (IB). Existing single-host results:
  `scripts/memory_study/bench/overlap_GB300_ch*.json`.
- Use `NCCL_DEBUG=WARN` (not INFO) for the timed runs; INFO logging perturbs
  timing.

## 6. Other options considered (for context)

- **FAIR CoreWeave `fair-cw-use2-1`** (`fbcode/pytorch/torchtitan/fb/fair_coreweave/cw.sh`):
  login works, but as of 2026-10-30 22:23 PDT the Slurm association on
  `faircw-pytorch-access` (QoS `g3_lowest`) had not synced, so GPU jobs are
  rejected. Its scale-out is RoCEv2, not InfiniBand, so this MSL tenant is the
  better fit for an IB measurement anyway.
- **MAST tenants under `root/gen_ai/msl` via `find-capacity`**: none visible.
