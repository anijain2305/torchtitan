# Methodology

- Kernel benchmarks: GB300 devserver (shared), CUDA-graph replays for GEMMs,
  minimum over repeats where other jobs interfered.
- DistMoE benchmark: EP 4 over NVLink, Kimi K3's 14 / 28 local experts,
  balanced routing, 25% scratch headroom (a benchmark setting; the memory model
  uses a capacity factor of 4); GEMM kernel time separated from helper kernels.
- Fake-process-group runs: one GPU stands in for rank 0 of 512; real kernels,
  collectives move no data; DistMoE with the real 896 experts, SwiGLU standing
  in for SiTU-GLU (same tensor sizes), balanced routing.
- Communication: nccl-tests on 4 GPUs; multi-host NVLink (16, 32 GPUs) and
  InfiniBand measured on a GB300 NVL72 cluster (below).

## Counting the memory-bound bytes

The memory-bound terms of [[compute_roofline]], per GPU per step. With full
remat, forward ops run twice (forward and remat), and backward moves about
twice a forward's bytes.

- **Attention residuals** (TorchTitan's fused Triton kernel). Each layer mixes
  its sources twice (before attention and before the MoE). Over `n` sources of
  width `D`, forward and remat each read the bf16 sources once and write the
  output; backward reads them again with the output gradient and writes their
  gradients: about `(8n + 6) x D` bytes per mix in total. `n` grows from 1 to 9
  across the 8 blocks, and each block's first layer also concatenates the
  stack.
- **MXFP8 quantization** (read bf16, write 8 bits plus one scale per 32
  elements, about 3 bytes per element): each linear's input three times
  (forward, remat, weight gradient) and its output gradient twice (input and
  weight gradients). Its weights: three times per microbatch without PP
  (forward, remat, input gradient), but once per step in both orientations
  under PP's ZeRO-1, where the gathered stage stays alive. The experts' weights
  likewise.
- **FSDP buffer copies.** Each all-gather copies the bf16 weights out of the
  gathered buffer (read and write 2 bytes per parameter); each reduce-scatter
  copies the bf16 gradients into an fp32 buffer (2 + 4 bytes). Without PP:
  two gathers and one reduce-scatter per layer and microbatch. Under PP's
  ZeRO-1: once per step.
- **Norms** (RMSNorm, KDA's L2 and gated norms, MLA's q / kv norms): 4 bytes
  per element forward and 6 backward, 14 over forward, remat and backward.
- **Everything else**: the two residual adds, the output gates and the shared
  expert's SwiGLU. This bucket also holds casts and gradient accumulation we do
  not count, so it is the least accurate.

## Communication experiments

**Hardware.** Within one host: a 4-GPU GB300 devserver (152 SMs per GPU,
NVLink between the GPUs). Across hosts: a GB300 NVL72 cluster with 4 GPUs per
host and one 800 Gb/s XDR InfiniBand NIC per GPU.

- **NVLink across hosts:** the job asks the scheduler to place all its hosts in
  one NVL72 domain. Every GPU reported the same NVLink fabric and clique, and
  NCCL's log showed one node of 16 (or 32) ranks with every connection between
  hosts over NVLink (`P2P/MNNVL`). Enabling NVLink SHARP (`NCCL_NVLS_ENABLE=1`)
  did not change the bandwidth.
- **InfiniBand across hosts:** NCCL's multi-host NVLink mode is turned off
  (`NCCL_MNNVL_ENABLE=0`), so each host's 4 GPUs talk over NVLink and traffic
  between hosts goes over the NICs: islands of 4 GPUs joined by InfiniBand, at
  8, 16 and 32 GPUs (2, 4 and 8 hosts). For islands of 8, 4 hosts in one NVL72
  domain are split into two NVLink cliques of 2 hosts (`NCCL_MNNVL_CLIQUE_ID`),
  so NCCL uses NVLink inside each and InfiniBand between them.
- **How NCCL does it** (from its logs): all-gather and reduce-scatter run as
  rings (`RING`, no NVLS or CollNet), one per channel, with 4 channels per NIC
  of the island up to 32 (16 on islands of 4, 32 on islands of 8). Each ring
  crosses every island boundary through one NIC, and since every GPU is on
  every ring, each GPU sees the island's combined NIC bandwidth. Each channel
  takes one SM, so the compute slows less on islands of 4 (1.18x) than on
  islands of 8 (1.3x, as inside a domain).

PyTorch nightly (2.16, CUDA 13) with its NCCL 2.32. Every timed run used
`NCCL_DEBUG=WARN`: the hosts' default per-call logging distorts timing.

**Bandwidth** (`coll_bw.py`). All-gather, reduce-scatter and all-to-all over
every GPU of the job, and send/recv from rank `i` to rank `i + N/2` (so each pair
spans two hosts), at 1 MB to 2 GB. Each size runs a few warm-up calls, then a
timed loop on CUDA events; the slowest rank's time counts. Bus bandwidth
follows nccl-tests: bytes x (N - 1) / N / time for the collectives, bytes /
time for send/recv.

**Overlap** (`kimi_k3_overlap_bench.py`). For each pair of a compute kernel and a
collective:

1. Time each alone. All ranks adopt rank 0's times, so every rank sizes its
   loops identically and issues the same number of collectives (otherwise the
   ranks deadlock).
2. Queue enough back-to-back collectives on a side stream to outlast about
   200 ms of compute, then run the compute on the main stream, and time both.
3. **Compute slowdown** = the compute's time per call while overlapped over its
   time alone.
4. **Collective slowdown**: the collectives start with the compute and outlast
   it; after the compute ends they run alone at their solo rate. Subtracting
   the calls that fit in that tail gives the calls done during the overlap, and
   the overlap window divided by those calls' solo time is the slowdown.
5. **NCCL SMs**: the thread blocks (one per channel, one SM each) of the
   collective's kernel, from a profiler trace.

The compute kernels are an MXFP8 and a bf16 GEMM at a KDA projection's shape
(8,192 x 7,168 x 12,288) and a memory-bound elementwise kernel over 1 GiB
operands (on the cluster, bf16 only: its image has no torchao). Within one host we also swept NCCL's channel count
(`NCCL_MAX_NCHANNELS` = 32, 16, 8, 4).

**One-way transfers** (PP hops). The first half of the ranks only send, each to
its partner in the second half, which only receives; every rank also computes,
and each reports its own compute slowdown. Within one host the pair is 2 GPUs
over NVLink; across hosts it is rank `i` on one host and rank `i` on the other,
4 pairs at once, so each transfer uses its own NIC.

**Running on the cluster.** The job image is a small conda environment
(PyTorch, numpy, nvidia-ml-py) packed into a package at submission. Inside the
job, the GPU driver library and a consistent pair of InfiniBand userspace
libraries (verbs and its mlx5 provider) had to be put on the loader path; the
host's default provider link was broken, which made InfiniBand invisible to
NCCL. Each node's rendezvous rank comes from its position in the job's host
list.
The scripts and a step-by-step log of the runs are in
`scripts/memory_study/msl_gb300/`.

## Why plain 1F1B exposes its hops

**The in-flight limit.** After warmup, plain 1F1B lets stage `s` hold only
`P - s` microbatches at a time: that is what bounds its activation memory. It is
just enough work to keep the stage busy while a microbatch makes its round trip
down the pipeline and back, if the trip were compute only. In the 1F1B chart of
[[step.1]], the boxed gap on stage 0 shows what happens: after F5, stage 0 has
F6 ready, but starting it would mean holding a fifth microbatch, more
activation memory than 1F1B allows. It has to run B2 first, and B2 waits for
its gradient's hop.

**It is the extra microbatches, not the chunks.** Interleaved 1F1B's warmup
runs `(v - 1) P + 2 (P - 1 - s)` forwards on stage `s`, twice the `P - 1 - s`
extra that keeps the pipeline busy. Swapping the two warmups in the simulation
(four stages, forward 1, backward 2, hop 0.5) shows which one hides the hops:

| Schedule | Warmup forwards on stage `s` | Hops added to the step per extra microbatch |
|---|---|---|
| Plain 1F1B | `P - 1 - s` (standard) | 1.5 |
| Plain 1F1B | `2 (P - 1 - s)` | 0 |
| Interleaved | `(v - 1) P + (P - 1 - s)` | 3 |
| Interleaved | `(v - 1) P + 2 (P - 1 - s)` (PyTorch's) | 0 |

The step without hops is the same in each pair: the doubled warmup only adds
slack, and costs the activation memory of the extra microbatches.

**Receiving early.** A transfer between two GPUs needs both to take part: the
receiver must give NCCL a buffer for the data ("post a receive") before it can
move. We assume both schedules post each receive as soon as they know the data
is coming. A receiver that posts it only when it needs the data makes every hop
start late: PyTorch's plain 1F1B does that for activations, which adds about
0.1 hop per microbatch; its interleaved schedule posts early.

## Notes
