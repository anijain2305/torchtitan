# Methodology

- Kernel benchmarks: a shared 4-GPU GB300 machine, CUDA-graph replays for GEMMs,
  minimum over repeats where other jobs interfered.
- DistMoE benchmark: EP 4 over NVLink, Kimi K3's 14 / 28 local experts,
  balanced routing, 25% scratch headroom (a benchmark setting; the memory model
  uses a capacity factor of 4); GEMM kernel time separated from helper kernels.
- Fake-process-group runs: one GPU stands in for rank 0 of 512; real kernels,
  collectives move no data; DistMoE with the real 896 experts, SwiGLU standing
  in for SiTU-GLU (same tensor sizes), balanced routing.
- Communication: nccl-tests on 4 GPUs; multi-host NVLink (16, 32 GPUs) and
  InfiniBand measured on a GB300 NVL72 cluster (below).

## Measuring Stashed Activations

`kimi_k3_activations.py` builds the full model under `FakeTensorMode` in bf16
(FSDP's compute dtype), so no memory is allocated, and runs one forward pass
plus the cross-entropy loss inside a saved-tensors pack hook. The hook records
every tensor autograd stashes for backward. Tensors are deduplicated by storage:
a storage stashed by several ops, or through several views, is counted once at
its full size. Parameters are excluded. The model's `local_compile` regions run
as in training, so the compiler's partitioner decides what those regions stash.

The attention residuals run on TorchTitan's fused Triton override, a custom op
with a fake implementation, so the same accounting covers it. It reads the
block stack and the partial sum directly and saves only small per-token
statistics, where the compiled region would stash a concatenated `[T, N+1, D]`
copy of its inputs for every call.

## Why Full Remat Keeps DistMoE's Output

Full remat stashes each layer's input in the forward pass. In the backward pass
it first replays the layer's forward from that input, rebuilding the
intermediates the gradients need (norm outputs, attention internals, residual
sums), and then runs the backward.

TorchTitan excludes the DistMoE call from that replay: it is wrapped in
`remat.region(..., recompute=False)` and its output is marked with
`remat.recompute_needs_tensor` (`torchtitan/models/common/dist_moe/routed_experts.py`).
When the replay reaches the MoE, it reuses the output from the original forward
and carries on through the rest of the layer. So autograd keeps that output: a
`[tokens, 3584]` bf16 tensor, about 59 MB per MoE layer at 8K, 5.4 GB over 92
layers.

The reason is that DistMoE already recomputes its own insides. It stashes only
its input, and its backward re-dispatches the tokens and reruns the expert
forward GEMMs to rebuild the intermediates it needs. Replaying the call under
remat as well would do that work twice in every MoE layer:

- an extra dispatch and combine, the expert-parallel all-to-all (about 940 MB
  per GPU per transfer at 8K), and
- an extra expert forward: 10 FLOPs per parameter per token for the routed
  experts instead of 8, about 25% more expert compute, on the quarter of the
  step that is experts.

Keeping the output trades 5.4 GB at 8K for avoiding both.

DistMoE's stashed input is a separate 5.4 GB, not the input full remat already
keeps: the MoE input arrives mid-layer, after attention, the attention-residual
mix, the norm and the down-projection into the 3,584-wide latent.

## Why TP Beats CP Only Under PP (or ZeRO-1)

CP and TP both halve the activations. TP also splits the dense weights
(attention, shared expert, norms), so its edge depends on how much of those a
GPU holds unsharded. The table counts the unsharded gradients (fp32) and
compute weights each GPU holds, per GPU at 8K:

::: figure tp-dense-table

Routed experts are 98% of a layer, but EP 64 leaves each GPU only 14 of 896,
so the dense part ends up the larger share of what it holds (62%, routed 38%;
45% and 55% after TP 2 halves the dense part). That makes the dense part worth
splitting. How much TP saves then depends on how many layers are held: under
FSDP only two, so TP 2 saves just 3.8 GB; under PP, ZeRO-1 holds the whole
12-layer stage all step, and TP 2 saves about 22 GB.

## Counting the Memory-bound Bytes

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
- **MXFP8 quantization** of the routed experts' weights (read bf16, write 8
  bits plus one scale per 32 elements, about 3 bytes per element): three times
  per microbatch without PP (forward, remat, input gradient), but once per step
  in both orientations under PP's ZeRO-1, where the gathered stage stays alive.
  DistMoE quantizes the expert activations inside its fused kernels, so their
  cost is part of its measured time; the dense linears run in bf16 and need no
  quantization.
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

## Communication Experiments

**Hardware.** Within one host: a 4-GPU GB300 machine (152 SMs per GPU,
NVLink between the GPUs). Across hosts: a GB300 NVL72 cluster with 4 GPUs per
host and one 800 Gb/s XDR InfiniBand NIC per GPU.

- **NVLink across hosts:** the job asks the scheduler to place all its hosts in
  one NVL72 domain. Every GPU reported the same NVLink fabric and clique, and
  NCCL's log showed one node of 16 (or 32) ranks with every connection between
  hosts over NVLink (`P2P/MNNVL`). Enabling NVLink SHARP (`NCCL_NVLS_ENABLE=1`)
  did not change the bandwidth.
- **InfiniBand across hosts:** NCCL's multi-host NVLink mode is turned off
  (`NCCL_MNNVL_ENABLE=0`), so each host's 4 GPUs talk over NVLink and traffic
  between hosts goes over the NICs: NVL domains of 4 GPUs joined by InfiniBand, at
  8, 16 and 32 GPUs (2, 4 and 8 hosts). For NVL domains of 8, 4 hosts in one NVL72
  domain are split into two NVLink cliques of 2 hosts (`NCCL_MNNVL_CLIQUE_ID`),
  so NCCL uses NVLink inside each and InfiniBand between them.
- **How NCCL does it** (from its logs): all-gather and reduce-scatter run as
  rings (`RING`, no NVLS or CollNet), one per channel, with 4 channels per NIC
  of the NVL domain up to 32 (16 with NVL domains of 4, 32 with 8). Each ring
  crosses every NVL domain boundary through one NIC, and since every GPU is on
  every ring, each GPU sees the NVL domain's combined NIC bandwidth. Each channel
  takes one SM, so the compute slows less with NVL domains of 4 (1.18x) than
  with 8 (1.3x, as inside a 64-GPU NVL domain).

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

## Kernel Efficiency at Each Config's Shapes

The measurements behind the practical roofline's compute ops ([[compute_roofline.2.1]]),
as a fraction of peak (green: at or above the ideal roofline's assumption; red:
more than 5 points below).

**Dense GEMMs** in bf16: the forward, input-gradient and weight-gradient GEMMs
of every linear in a layer.

::: figure dense-shapes

**Routed experts**: DistMoE's MXFP8 kernels (grouped GEMMs for w13 and w2 with
the token dispatch and combine fused in), forward, remat and backward. At EP 64
they come from the FSDP 512 + EP 64 fake-process-group steps, whose collectives
move no data, so the time is the kernels alone; EP 32 scales them by the
EP 32 / EP 64 ratio of a single-GPU DistMoE benchmark at the same local tokens.
On a real cluster DistMoE runs slower still: the kernels read and write their
peers' tokens over NVLink, and its ranks wait on each other at barriers. We
leave that to the gap between the model and a real run. With balanced routing,
each expert's `M` (rounded down to a multiple of 128, the MXFP8 scale tile)
is:

```
M per expert = local_tokens x top_k / local_experts
             = local_tokens x top_k x EP / num_experts
```

::: figure expert-shapes

**MLA** on FlashAttention 4 in bf16, as the two matmuls' FLOPs over the measured
time (which includes the softmax and the causal masking). Ulysses CP and TP both
split the heads:

::: figure mla-shapes

## Why Plain 1F1B Exposes Its Hops

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
