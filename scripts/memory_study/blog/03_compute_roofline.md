# The Compute Roofline

Memory told us which parallelism configs fit. Now we want to know how fast
each one trains. We start with compute alone: how long one step takes with no
communication and nothing else in the way.

We build that estimate in two passes:

- **The ideal roofline** ([[compute_roofline.1]]) runs every op at a fixed
  fraction of the GPU's peak: one assumed efficiency for all GEMMs, full HBM
  bandwidth for everything else. It is quick to compute and good for finding
  compute bottlenecks.
- **The practical roofline** ([[compute_roofline.2]]) assumes the compute
  kernels are already well optimized (we do not plan to improve cuBLAS's
  matmuls in this study) and times each one at the shapes each config produces.
  It needs profiling on the hardware we care about, but it is more accurate,
  which makes it the right model for ranking configs. It is **not** a tool for
  finding compute bottlenecks: it takes every kernel's measured speed as given.
  This is the model the rest of the post builds on.

## The Ideal Roofline

Every GPU processes the same number of tokens per step, whatever the
parallelism: 16M tokens / 512 GPUs = **32,768 tokens**. With full remat, a
linear layer costs 8 FLOPs per parameter per token: 2 in forward, 2 when remat
reruns it, and 4 in backward.

| Symbol | Meaning | Kimi K3 |
|---|---|---|
| `T_gpu` | tokens per GPU per step | 32,768 |
| `N_dense` | dense linear parameters, summed over all 93 layers | 54.4B |
| `N_expert` | parameters of one routed expert (w13 + w2) | 33.0M |
| `V`, `D` | vocabulary size, model width | 163,840, 7,168 |
| `seq_len` | context length | 8K or 32K |
| `H` | MLA attention heads | 96 |
| `d_qk`, `d_v` | MLA query/key and value head dimensions | 192, 128 |
| `B_hbm` | HBM bandwidth | 8 TB/s |

Per GPU per step:

- **Dense linears**: `8 x T_gpu x N_dense`.
- **Routed experts**: `8 x T_gpu x 16 x N_expert x 92 layers` (16 experts per
  token).
- **LM head**: `6 x T_gpu x V x D` (not rematerialized).
- **MLA attention**: `4.5 x T_gpu x seq_len x H x (d_qk + d_v)` per layer
  (each token attends to half the sequence on average; forward + remat +
  backward is 4.5 forwards). The one term that grows with context.
- **KDA attention**: linear in tokens; its chunked kernels are not plain
  matmuls, so we use their measured time per token.
- **Everything else** (the attention-residual mix, the experts' MXFP8 quantization, FSDP's
  buffer copies, norms, residual adds): memory-bandwidth bound,
  `bytes moved / B_hbm`. Without PP, FSDP re-gathers, copies out and
  re-quantizes every layer's expert weights each microbatch; under PP's ZeRO-1 the
  stage stays gathered, so this happens once per step. [[methodology.4]]
  counts the bytes class by class.

The matmuls run at an assumed fraction of peak: **60% of 5 PFLOPS** in MXFP8
(routed experts) and **70% of 2.5 PFLOPS** in bf16 (dense linears, LM head,
attention).

::: figure roofline-table

15.6 s per step at 8K and 16.4 s at 32K. Two things stand out:

- **The dense linears cost about twice as much as the routed experts.** Experts
  are 98% of the parameters, but a token touches about 54B dense parameters and
  only 49B expert ones, and the dense ones run in bf16 at half the FP8 rate.
- **The matmuls cost the same in every config.** Every GPU does the same
  FLOPs per step; only the memory-bound term changes with the config, so the
  ideal roofline barely separates them.

## A Practical Roofline

Most roofline studies stop at something like the ideal roofline. It is a
first-order model, built to show where the bottlenecks are, and it can be far
off: for FSDP 512 + EP 64 at 8K it predicts 15.6 s, while the same step,
run with a fake process group (real kernels, collectives that move no data),
measures 20.6 s of GPU time ([[compute_roofline.2.5]]). We want more than
bottlenecks: we want to use this model to choose how to run Kimi K3 on our
cluster, so it has to be more accurate.

For that, we can lean on what we know about our framework. Like many
practitioners, we know which parts are already well optimized: in this study,
the compute kernels. So instead of one assumed efficiency for every GEMM, we
run each matmul at the shapes the parallelism configs actually produce and
measure its efficiency. The model still leaves out communication and pipeline bubbles,
which come later.

> INSIGHT: A roofline model is not one-size-fits-all. What you know about your framework and what you need the model for decide how to build it: a first-order model to find bottlenecks, or measured kernels at your exact shapes to choose between configs.

All measurements run on one GB300, which has the same bf16 (2.5 PFLOPS) and FP8
(5 PFLOPS) peaks as GB200, so the percentages carry over. Kernels are timed as
CUDA-graph replays: kernel time, no CPU launch overhead.

### Compute Ops at Each Config's Shapes

The compute kernels are well optimized, so the model uses the efficiency they
actually reach at each config's shapes. Parallelism changes those shapes. For a
linear's forward GEMM `(M x K) @ (K x N)`, `M` is the tokens (rows), `K` the
input width and `N` the output width:

- **CP**: fewer tokens per microbatch, so smaller `M`.
- **TP**: narrower dense weights at the same `M` (a few small ones stay
  whole): smaller `N` for column-parallel linears, smaller `K` for row-parallel
  ones. The MoE runs on TP-local tokens, so like CP it gives each expert a
  smaller `M`.
- **EP**: experts per GPU (14 at EP 64, 28 at EP 32); with the local tokens,
  that sets each expert's `M`.
- **PP**: no shape of its own; PP 16 means EP 32.

In the weight-gradient GEMM the tokens are the reduction dimension, so there
CP shrinks `K` instead.

We time each compute op at each split's shapes on one GPU and compare it with
the ideal roofline. The dense GEMMs are bf16 GEMMs; the routed experts are
DistMoE's own MXFP8 kernels, grouped GEMMs with the token dispatch and combine
fused in, timed in a fake-process-group step (its collectives move no data, so
this is the kernels alone, at the real per-GPU shapes); MLA is FlashAttention 4
in bf16, and KDA is its chunked kernel:

::: figure ideal-gap-table

- **Different ops miss by different amounts.** The dense GEMMs come within about
  5% of their assumed 70%. DistMoE's experts take 1.25x the ideal time: their
  plain GEMMs run at about 60%, but the two that fuse the combine (each token's
  top-k outputs weighted and summed back) run at about half that. MLA takes
  1.3-1.7x the ideal time: its softmax
  costs time but no matmul FLOPs (the exponentials run on slower
  special-function units, and the backward recomputes them).
- **Different splits miss by different amounts.** At 8K, CP 4 or TP 4 leaves
  each expert an `M` of about 2,300 rows, and DistMoE drops to 43% of peak
  (1.38x). KDA does not split cleanly at all: under CP its recurrent state
  passes between ranks, and TP 2 or TP 4 speeds it up only 1.7x at 8K, so TP 4
  takes 2.35x the ideal time.
- **Different contexts miss by different amounts.** At 32K the same splits
  leave larger shapes, and most of the gaps shrink.

One efficiency per op would charge a bad config the same as a good one. The
per-shape measurements behind this table are in [[methodology.6]].

### Memory-bound Work

Here too we measure: we keep the estimated bytes from [[compute_roofline.1]] and
charge each class the bandwidth its kernels reach in a measured step (FSDP 512 + EP
64 at 8K, CUDA graphs):

::: figure nonmatmul-table

- **This is where the ideal roofline is furthest off**: 1.5 s at full
  bandwidth, 5.4 s measured, 28% of the bandwidth on average.
- **MXFP8 quantization and norms** come close: 58-60%.
- **FSDP's copies** reach 39%, and **everything else** (casts, adds, gradient
  accumulation) only 16%.
- **The attention residuals** take 1.6 s: about half in the fused Triton kernel, the
  rest concatenating the block stack at each block's first layer.

> CAVEAT: These memory-bound kernels likely have room to improve, and the practical roofline **deliberately** takes them as they are. So it is not a target for optimizing them: speed them up, and the step can run faster than the practical roofline predicts. That may sound odd for a roofline, but it is our choice: the goal here is not to optimize compute, but to rank the parallelism configs that fit in memory. To optimize compute, the ideal roofline is the better target.

### The Optimizer Step

Every step ends with the optimizer: DistMuon, Muon over the sharded parameters.
Muon orthogonalizes each weight matrix with a few Newton-Schulz iterations, so
it needs whole matrices, and how much it has to move depends on how they are
sharded:

- **Routed experts, 98% of the parameters: nothing to move.** FSDP shards each
  group's 14 experts along the expert dimension, so a GPU's shard of the
  gradient, momentum and weights is two whole experts, and it orthogonalizes
  them where they are.
- **Dense weights: a small redistribution.** They are sharded by rows, so
  DistMuon gathers each matrix (or each head's block) to one GPU, runs
  Newton-Schulz there and sends the update back: about 0.2 GB each way per GPU
  per step, a few milliseconds.

So the optimizer is almost all compute, and about the same per GPU in every
config: each GPU holds about 1/512 of the parameters either way. Measured in
fake-process-group steps at 8K:

| Optimizer step | GPU work (the practical roofline) | Eager, as TorchTitan runs it today |
|---|---|---|
| FSDP 512, EP 64 | 0.8 s | 3.1 s |
| PP 4, first stage | 0.5 s | 0.5 s |

The FLOPs are about the same, but without PP the work comes as many more,
smaller kernels (all 93 layers' parameters), which keep the GPU less busy.
The model charges the GPU work, 0.8 s per step without PP and 0.5 s with it
(PP 8 and 16 hold fewer layers per GPU; we take PP 4's time for them too), on
top of the forward and backward below.

> CAVEAT: Today FSDP 512's optimizer step takes 3.1 s, because DistMuon runs eagerly: for all 93 layers' parameters it launches about 700,000 small kernels, and the GPU waits on the CPU that launches them. A PP rank holds a quarter of the layers or fewer, so the CPU keeps up. The step cannot be captured in a CUDA graph yet (among other things, its learning rate is a Python number, baked into a capture); once it can, FSDP 512 gets those 2.3 s back. Either way, it does not change the ranking.

### Step Time per Config, Sans Communication

Now the forward and backward per GPU, still without communication, pipeline
bubbles or the optimizer step. Red
cells show how much slower a config is than the same family with no split (CP 1,
TP 1). Hatched rows do not fit in memory.

**FSDP 512, EP 64:**

::: figure compute-table-fsdp

**Without PP, CP and TP are expensive.** Nearly all of the slowdown is
memory-bound work, and most of that is FSDP's: it re-gathers, copies and
re-quantizes every layer's expert weights each microbatch, and CP and TP multiply the
microbatches. At 8K, CP 4 grows the memory-bound work from 5.4 to 11.7 s.

**PP 8 (EP 64 per stage):**

::: figure compute-table-pp8

**PP's ZeRO-1 removes that cost.** The stage's weights stay gathered for the
whole step, so the weight work happens once: with TP 2 at 8K, PP 8 needs
18.9 s against 22.1 s for FSDP 512, and CP and TP cost far less. What they still cost is
smaller shapes: KDA under CP and TP, and the experts at CP 4 or TP 4.

**PP 16 (EP 32 per stage):**

::: figure compute-table-pp16

**Small shapes cost up to a second and a half.** PP 16 with CP 4 or TP 4 runs its
experts at 41% of peak (0.5 s more than without a split), and at 8K TP 4 more
than doubles KDA's time.

### Checking Against a Measured Step

A fake-process-group step measures exactly this part: its collectives do no
work, so the GPU time is compute and memory-bound work only. One GPU holds a
single rank of every FSDP 512 + EP 64 config except 32K without a split, so we
ran them all (PP needs one GPU per stage, so its configs are not here):

::: figure sweep-table

- **Within 8% everywhere**: 3% over at 8K without a split, 3-6% over at 32K
  with CP 2, CP 4 or TP 2.
- **Optimistic where KDA splits or the microbatches multiply** (CP 4 and TP 4
  at 8K, TP 4 at 32K), by 6-8%: those configs run KDA at its least efficient
  and redo FSDP's weight work more often, and those kernels run a little slower
  than calibrated.
- **The order holds at 8K**, which is what the ranking needs. At 32K the four
  splits sit within 2.5 s and the model swaps one: it puts TP 4 second, the
  measurement last.

The experts are not an independent check here: their times come from these
same runs ([[compute_roofline.2.1]]).

> NOTE: Why build the practical roofline at all, if a fake-process-group run measures the step directly? Because the step is not one number once communication enters. Each collective overlaps a different part of the compute: FSDP's gathers the next layer's GEMMs, DistMoE's dispatch the expert GEMMs, and TP's and CP's sit between the ops that wait for them. To tell whether a collective is really hidden, and which compute it slows down, we need the time of each op, not one total.

**From here on, "the roofline" means this practical roofline**, not the ideal
one from [[compute_roofline.1]].

## Notes
