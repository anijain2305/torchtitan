# Overnight notes: GB300 MSL cluster runs (2026-10-30 / 31)

What it took to run PyTorch benchmarks on the `msl_infra_pytorch_dct` tenant,
and what each job showed. Complements README.md (cluster facts).

## Getting a PyTorch image (no TEST_ENV_WRITE needed)

- `torchtitan_conda_prod:stable` is an **x86-64** build: it cannot run on the
  aarch64 GB300 hosts (`Exec format error`).
- Working path: a small local conda env (`~/.conda/envs/msl_bench`: Python 3.12,
  PyTorch nightly cu130 aarch64, numpy, nvidia-ml-py), packed by the
  **`msl_conda`** scheduler into an ephemeral fbpkg under the shared
  **`metaconda_demo`** package (ACL `meta_conda_fbpkg_default`, no extra
  permission).
- The benchmark scripts travel *inside* the conda env (`$CONDA_PREFIX/msl_ws/`),
  because the workspace and the conda env cannot both be ephemeral builds of the
  same package; `workspace_dir=/packages/metaconda_demo/conda/msl_ws`.
- Submit with the env **activated** (`eval "$(conda shell.bash hook)"; conda
  activate msl_bench`): with `conda_dir` set, `msl_conda` assumes the env already
  exists on the host and does not pack it; without activation it packs whatever
  env is active (`pytorch1`, which fails on its editable install).
- Scheduler args: `hpcIdentity=pytorch_distributed,hpcClusterUuid=MastGenAICluster,
  rmAttribution=msl_infra_pytorch_dct,hpcJobOncall=pytorch_distributed,
  localityConstraints=region;ofn,jobPriority=LOW,conda_fbpkg_name=metaconda_demo,
  activate_conda=True,workspace_dir=/packages/metaconda_demo/conda/msl_ws`.

## Fixes needed inside the job

| Symptom | Cause | Fix |
|---|---|---|
| `WarmStorageSpec ... missing clusters field` at submit | `msl_conda` mounts warm storage by default; region `ofn` has no WS cluster mapping | role env `DISABLE_OILFS=1` |
| `torch.cuda.device_count() == 0` | the driver's `libcuda.so.1` is in `/usr/local/fbcode/platform010-aarch64/lib`, off the loader path | symlink only `libcuda*` / `libnvidia*` from there into `/tmp/nvdrv`, add to `LD_LIBRARY_PATH` |
| NCCL `Failed to initialize any NET plugin`, `NET/IB : Using [RO]` with no devices; `ibv_get_device_list` returns 0 | the system provider `/usr/lib64/libibverbs/libmlx5-rdmav57.so` is a **dangling symlink** to `../libmlx5.so.1.25.57.0`, which `/usr/lib64` does not have (it has 1.26.62); `VERBS_LOG_LEVEL=4` shows `couldn't load driver`. `/etc/nccl.conf` forces `NCCL_NET=IB`, so no fallback | link the fbcode platform dir's matching pair (`libibverbs.so.1.14.57.0`, `libmlx5.so.1.25.57.0`, the latter also as `libmlx5-rdmav57.so`) into `/tmp/ibv`, first on `LD_LIBRARY_PATH`: verbs then finds the 4 IB HCAs |
| 2-host NCCL hangs with default settings | NCCL tries the multi-node NVLink path, which these hosts do not have | `NCCL_MNNVL_ENABLE=0`: IB connects (`Connected all trees`) |
| (tried) verbs 0 devices | not a permission issue: running as root (`MAST_USERNAME=root` capability) did not change it | kept `run_as_root` anyway, harmless |

The job-side setup is `jobs/gpu_env.sh`, sourced first by every run script. Submit with `./submit.sh <hosts> <script in jobs/>` (add `--dryrun` first).

## Cluster observations

- NVML reports the GPUs' fabric state as "not supported" (cluster UUID 0,
  clique 0): these hosts may not share a multi-node NVLink domain, in which case
  inter-host traffic is InfiniBand only. (To confirm from NCCL's transport log.)
- `/etc/nccl.conf`: `NCCL_NET=IB`, `NCCL_NVLS_ENABLE=0`, `NCCL_PXN_DISABLE=1`,
  `NCCL_NCHANNELS_PER_NET_PEER=4`, `NCCL_DEBUG=INFO` (override with
  `NCCL_DEBUG=WARN` for timed runs).
- MSL's single-host health check measures a 4-GPU NVLink all-to-all at about
  521 GB/s bus bandwidth.

## Jobs

| Job | Hosts | What | Result |
|---|---|---|---|
| torchx-raw-sh-fj7zqzmm | 2 | pip install torch in `conda_mast_core` | no internet from hosts |
| torchx-raw-sh-j2fh2v7c | 1 | `torchtitan_conda_prod:stable` | x86-64, unusable |
| raw-sh-t611lj2v | 2 | first `msl_conda` smoke | torch imports, 0 GPUs (cancelled) |
| raw-sh-dc335c1v | 1 | GPU probe | found `libcuda` location |
| raw-sh-r05rs4fk / raw-sh-dzvwq2ff | 2 | 4 GPUs/host; NCCL probe | NCCL NET plugin failure |
| raw-sh-zxwf6nc0, raw-sh-bx92przv, raw-sh-pmm6tp3r, raw-sh-h3kslhwm | 1 | verbs probes | found the dangling provider symlink |
| raw-sh-qqzskgdg | 1 | verbs with the fix | 4 devices |
| raw-sh-qwdc7cv5, raw-sh-...(b5748) | 2 | NCCL probe, default + MNNVL off | hung (default mode) |
| raw-sh-w1qvclvz | 2 | NCCL probe, MNNVL off | IB connected, exit 0 |
| raw-sh-nkcjktwg | 2 | IB collective bandwidth + one-way PP-hop overlap | hung with no output after 45 min; cancelled (another user's 16-host job was waiting) |
| raw-sh-q4dkqt1q | 4 | IB collective bandwidth, 16 GPUs | **all-gather 371-378, reduce-scatter 354-361, all-to-all 115, send/recv 90 GB/s per GPU** (large messages); `bench/ib_collectives_GB300_4hosts.json` |
| raw-sh-xkr0m04f | 2 | one-way PP-hop overlap over IB, logs kept | (running) |

## Results

**Collectives across hosts over InfiniBand (16 GPUs on 4 hosts).** Send/recv
between rank i and rank i + 8 reaches 90 GB/s per GPU, 0.90 of the 800 Gb/s
NIC's line rate. All-gather and reduce-scatter reach 355-378 GB/s bus bandwidth,
about 4x one NIC: NCCL's ring stripes its channels over all 4 NICs of a host and
moves data inside the host over NVLink, so the host boundary carries the host's
aggregate NIC bandwidth (0.92 x 4 x line rate). For the study's GB200 setup
(64-GPU NVLink domains, 50 GB/s NICs) that means FSDP across domains is
NVLink-bound, not InfiniBand-bound; the step-time model now uses
`min(NVLink, GPUs per domain x NIC x 0.92)` for collectives across domains and
`0.90 x NIC` for PP hops.
