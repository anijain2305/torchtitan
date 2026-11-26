# Ranking

**From compute to step time.** Without PP (FSDP 512 + EP 64), every microbatch
runs all 93 layers: FSDP's all-gather and reduce-scatter of the next layer
overlap the current layer's compute, with the penalties of [[communication]];
EP's dispatch and combine run inside DistMoE's expert GEMMs; TP's and CP's
collectives are exposed in full. The step is one microbatch's time, times the
number of microbatches. With PP, the schedule simulation of [[step]] gives the
step from each chunk's time and each hop, and ZeRO-1 adds the stage's
all-gather and reduce-scatter once per step; so does ZeRO-1 + EP 128 + TP,
without a pipeline. Every config then runs the optimizer step
([[compute_roofline.2.3]]).

Each config runs the remat and offload its greedy path picked ([[memory.5]]),
so the model charges only what that pick recomputes: the dense layers' replay
under full remat, the attention kernels' replay when attention is recomputed,
and DistMoE's recompute in the MoE layers whose buffer was not bought back.
Each of these, and offload's slowdown, is calibrated on the measured 8K steps
of [[memory.4.1]] and scaled with each config's shapes.

The configs of [[memory.5]]'s tables that fit in memory, ranked by step time
(the toggle switches context). For PP, each config also names its schedule and its hop
transport. The chart runs TorchTitan's transports, chain for plain 1F1B and
cache for interleaving ([[step]]); whether direct would change anything is
[[ranking.4]]'s question. For PP, each bar shows the rank that sets the pace,
usually the last (it also runs the LM head), so its compute is a little above
[[compute_roofline.2.4]]'s, which averages over the stages. The chart splits
each step into parts (hover a row for the numbers; the buttons filter by
family and schedule, and sort by the parts the sections below look at):

- **Compute**, at the practical roofline ([[compute_roofline.2]]): GEMMs and
  attention; **recompute**, what the config's remat and DistMoE replay in
  backward; offload's slowdown; the memory-bound kernels that scale with tokens (norms, casts, the
  attention-residual mix); and **weight prep**,
  the work that scales with each gather of the weights: FSDP's copies of the
  gathered weights and of the gradients into their buffers, and the expert
  weights' MXFP8 quantization. FSDP 512 (ZeRO-3) repeats it every microbatch, so it grows
  with the microbatch count (CP and TP); PP's ZeRO-1 does it once per step.
- **Communication**, colored by parallelism: solid where it is exposed (it does
  not fit behind the compute it overlaps), hatched for the compute slowdown
  while it overlaps, cross-hatched for the collective's own slowdown when that
  makes it outlast the compute ([[communication]]). TP and CP are exposed in
  full: nothing overlaps them in TorchTitan today.
- **Pipeline**, from the schedule simulation ([[step]]): the **bubble**'s
  warmup and cooldown (filling and draining, plus uneven chunks), about
  `(P - 1) t / v`; the **hops**, the part of the bubble the hops add on top
  (most of their time with plain 1F1B, little with interleaving); and
  **ZeRO-1**, the
  stage's FSDP all-gather of its weights before the first forward and
  reduce-scatter of its gradients after the last backward. Under PP each rank
  keeps its stage's weights gathered and its gradients unsharded for the whole
  step (ZeRO-1: only the master weights and optimizer state stay sharded),
  instead of gathering and reduce-scattering every layer every microbatch
  (ZeRO-3, as FSDP 512 does), so this happens once per step, with nothing to
  hide behind ([[memory.2.4]]).
- **Optimizer**: DistMuon's step, at the GPU time it needs ([[compute_roofline.2.3]]).

The bars share one axis at both contexts; the four slowest 8K configs and the
slowest 32K one run past it (marked &raquo;).

::: figure ranking-chart

First, the winners:

- **At 8K, interleaved PP 8 leads, without a split:** 19.1 s at `v = 4`, 19.8 s
  at `v = 2`. It recomputes attention, offloads, and buys back a fifth to a
  quarter of DistMoE's buffer.
- **At 32K, two very different configs tie** within 0.3 s: FSDP 512 without a
  split (23.1 s), and PP 8 interleaved with CP 4 (23.4 s).

But the chart holds much more than the winners, so let's go through what it
says, one question at a time.

## Why FSDP 512 Without PP Falls Behind

At 8K, even the best FSDP 512 config, with 2 sequences per microbatch, is 3 s
behind the leader, and the others fare much worse. Why?

- **Overlap is not free.** FSDP's collectives almost hide behind each layer's
  compute, but hiding them means running alongside nearly all of it, and
  slowing it down the whole time. On top comes the weight prep (copying the
  gathered weights, quantizing the experts), which FSDP repeats every
  microbatch and PP's ZeRO-1 does once per step.
- **Less remat buys FSDP little.** The compute that remat no longer replays was
  what hid FSDP's collectives. At 8K, the greedy pick saves 2.6 s of compute
  over full remat, but only 0.8 s of the step: the rest comes back as exposed
  communication and overlap penalties. Halving the microbatches helps more:
  with 2 sequences per microbatch, FSDP gathers half as often, and wins 3.6 s
  even though it needs full remat.
- **CP and TP make it much worse.** They multiply the microbatches, and that
  exposes the communication from both sides: each microbatch has fewer tokens,
  so less compute to hide behind, and there are more of them, so more
  all-gathers and reduce-scatters. At CP 4, a layer's forward takes about a
  third of the time its all-gather does.
- **TP is less bad than CP.** Both sit at the bottom of the ranking, but TP also
  splits the dense weights, so FSDP has less to gather.

At 32K, FSDP 512 wins, which may be surprising at first. Without a split, a
32K step is a single microbatch per GPU, so FSDP gathers and reduce-scatters
each layer once per step, as ZeRO-1 does, and full remat with the block inputs
offloaded fits ([[memory.5]]). What is left is the overlap penalty of that one
round, under a second. PP, on the other hand, now pays for its split: CP 4 to
fit, exposed CP and a bigger bubble. FSDP 512 with a split follows close
behind, since each microbatch still has long attention to hide behind.

## Why PP 8 Wins Over PP 16 and PP 4

With FSDP 512 out of the running at 8K, the next questions build up, one at a
time, why the leader is the fastest. First: PP 8 or PP 16? At the same split
and schedule, PP 16 is slower every time, by about a second, and by more with
deep interleaving. Why?

When we raise the PP degree, we usually think of the stages getting smaller.
But it also shrinks everything inside a stage: with 512 GPUs, PP 16 leaves 32
GPUs per stage instead of 64, so the stage's SPMD mesh, and with it EP, halves
(EP 32 instead of EP 64). That is the main reason PP 16 is slower here:

- **Smaller expert GEMMs.** With EP 32 each GPU holds twice the experts (28
  instead of 14) but processes the same tokens, so each expert gets half the
  rows, and the expert kernels run less efficiently. Together with the LM
  head, which weighs more on a 6-layer stage than on a 12-layer one, this
  slows the slowest rank by about a second: most of the gap in the top
  configs.
- **Twice the hops.** A microbatch crosses twice as many stage boundaries, and
  with half as many data-parallel replicas each pipeline runs twice the
  microbatches.
- **Uneven chunks at `v = 4`.** 93 layers over 64 virtual stages leaves chunks
  of one or two layers, and the imbalance stalls the schedule: at `v = 4` this,
  not the experts, is the bigger cost.

> INSIGHT: Raising the PP degree shrinks everything inside a stage too. With a fixed number of GPUs, more stages mean fewer GPUs per stage, so smaller EP and smaller expert GEMMs.

At 32K the same holds: both need CP 4 to fit, and PP 16 is 1.3 s slower at
`v = 2`. PP 16's one advantage, half the layers per stage, buys memory that PP 8
does not need.

**And PP 4?** It holds twice the layers per stage, so it fits only at 8K, and
it is the slowest PP of all: 25.4 s at best. The reason is the mirror image of
PP 16's: with 128 GPUs per stage, EP grows to 128, and that spans two NVLink
domains. Half of DistMoE's dispatch and combine then crosses InfiniBand, much
slower than NVLink and too slow to hide behind the expert GEMMs: 6.3 s of
exposed EP per step.

## Why Interleaved 1F1B Wins Over 1F1B

Interleaving beats plain 1F1B in every PP family, and almost all of the gain
is the bubble:

| PP 8 at 8K | Step | Bubble | Hops | DistMoE buffer bought back |
|---|---|---|---|---|
| 1F1B (chain) | 21.9 s | 3.5 s | 0.3 s | 30% |
| Interleaved, `v = 2` (cache) | 19.8 s | 1.4 s | 0.1 s | 20% |
| Interleaved, `v = 4` (cache) | 19.1 s | 0.5 s | 0.1 s | 25% |

- **The bubble shrinks by about `v`,** as [[step.1]] predicts.
- **The hops barely matter.** With 100 GB/s NICs a hop is short, so even plain
  1F1B, which exposes its hops, loses little to them.
- **The cost is memory:** the first rank holds more chunks in flight, so less
  of DistMoE's buffer can be bought back, and a little more is recomputed.
- **More is not always better.** With PP 16, `v = 4` leaves chunks of one or
  two layers, and the bubble grows back: at both contexts it is more than a
  second slower than `v = 2`.

> INSIGHT: Interleaving shrinks the bubble by about `v` and its extra work in flight hides the hops, at the price of memory; but once chunks shrink to a layer or two, the imbalance between them brings the bubble back.

## Do Attention Residuals Hurt PP?

**The problem.** Attention residuals give PP an unusual property: a stage needs
the blocks of every stage before it, so with chain each hop carries a bigger
stack than the last, and the latest stages, already the slowest (the last
also runs the LM head), trade the
most bytes, on the critical path ([[step.2]]). A natural fix is **direct**,
which TorchTitan does not have: each rank sends every block it finishes
straight to the later stages that use it, asynchronously, so a hop carries
only the hidden state, and the block traffic spreads over all pairs of ranks,
off the critical path, while the stages compute.

::: figure transport-diagram

**The numbers.** With 100 GB/s NICs, hops are cheap to begin with, so there is
little for direct to save:

::: figure transport-compare-table

- **PP 8:** the transport barely matters with any schedule.
- **PP 16:** chain's stack grows long enough to cost up to about a second and a
  half, and direct wins most of that back; cache, under interleaving, gets
  most of it too.

**Verdict:** for the configs we rank, direct does not help. Where the leaders
run, TorchTitan's cache is as good, so the attention residuals cost PP little.

## What If TP and CP Were Overlapped?

The model charges TP and CP as fully exposed, the way TorchTitan runs them
today. But nothing prevents overlapping them, like FSDP's collectives: TP's
behind the dense GEMMs (async TP), CP's behind the rest of the layer. Would
that change the picture? Charging them like FSDP instead, overlapped and with
the same NVLink penalties:

::: figure overlap-tp-cp-table

- **At 8K nothing changes at the top:** the leaders run without a split.
  ZeRO-1 + EP 128 + TP 8 gains the most, 3.6 s, but stays far behind.
- **TP gains more than CP,** since it exposed more: FSDP 128 with TP 4 gains
  1.2 s at 32K, CP 4 configs about half a second.
- **The 32K tie flips:** PP 8 interleaved with CP 4 edges ahead of FSDP 512, and
  FSDP 128 with TP 4 moves up to third.

Overall, overlapping helps TP far more than CP, but it reorders only near-ties.
An overlapped TP would make FSDP with TP 4 a real contender at 32K.

## How Far to Trust the Ranking

We have now worked out, from first principles, which configs to run and why.
But there is a catch: this is a model, and the leaders sit within a few
percent of each other. Small changes to the model can flip their order:

- **What it assumes:** NVLink bandwidth and overlap inside a 64-GPU NVL domain
  (measured up to 32 GPUs), TP and CP fully exposed, a hop moving as soon as
  both sides are ready, the optimizer at its GPU time, and every remat and
  offload cost scaled from the measured 8K FSDP 512 + EP 64 steps. EP 128's
  expert kernels are assumed as efficient as EP 64's (not benchmarked).
- **What it leaves out:** profiles of a real 512-GPU run show DistMoE
  slower than in our fake-process-group steps (about 1.65x the bare grouped
  GEMM instead of about 1.25x, plus waits at its barriers), and FSDP's dense
  collectives across all 512 GPUs slower than modelled. The first adds about
  the same to every config; the second only makes FSDP 512 slower still.

So the practical way to use the ranking is to take a few of the top configs
and run them on the cluster. What the model gives the practitioner is a clear
picture of what to expect and why: absolute times will be off, and the order
of near-ties may change, but the big picture should hold. PP beats FSDP 512 at
8K, FSDP 512 catches up at 32K once a step is a single microbatch, interleaving
(at least `v = 2`) beats plain 1F1B, PP 8 beats PP 16 and PP 4, and the
transport hardly matters. If a run disagrees with that, it
points to an inaccuracy in the model.

## Notes
