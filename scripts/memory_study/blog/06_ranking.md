# Ranking

**From compute to step time.** Without PP (FSDP 512 + EP 64), every microbatch
runs all 93 layers: FSDP's all-gather and reduce-scatter of the next layer
overlap the current layer's compute, with the penalties of [[communication]];
EP's dispatch and combine run inside DistMoE's expert GEMMs; TP's and CP's
collectives are exposed in full. The step is one microbatch's time, times the
number of microbatches. With PP, the schedule simulation of [[step]] gives the
step from each chunk's time and each hop, and ZeRO-1 adds the stage's
all-gather and reduce-scatter once per step. Every config then runs the
optimizer step ([[compute_roofline.2.3]]).

The configs that fit in memory ([[sharding.7]]), ranked by step time (the toggle
switches context). For PP, each config also names its schedule and its hop
transport. The chart runs TorchTitan's transports, chain for plain 1F1B and
cache for interleaving ([[step]]); whether direct would change anything is
[[ranking.4]]'s question. For PP, each bar shows the rank that sets the pace,
usually the last (it also runs the LM head), so its compute is a little above
[[compute_roofline.2.4]]'s, which averages over the stages. The chart splits
each step into parts (hover a row for the numbers; the buttons filter by
family and schedule, and sort by the parts the sections below look at):

- **Compute**, at the practical roofline ([[compute_roofline.2]]): GEMMs and
  attention; the memory-bound kernels that scale with tokens (norms, casts, the
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
  hide behind ([[sharding.5]]).
- **Optimizer**: DistMuon's step, at the GPU time it needs ([[compute_roofline.2.3]]).

The bars share one axis at both contexts; the four slowest 8K configs run past
it (marked &raquo;).

::: figure ranking-chart

First, the winners:

- **At 8K, interleaved PP 8 leads,** with CP 2 or TP 2 (22.1-22.3 s).
- **At 32K, the top configs tie** within about a second: PP 8 interleaved with
  TP 4, PP 16 interleaved with CP 4, and FSDP 512 with TP 4.

But the chart holds much more than the winners, so let's go through what it
says, one question at a time.

## Why FSDP 512 Without PP Falls Behind

At 8K, even the best FSDP 512 config is well behind the leader, and the others
fare much worse. Why?

- **Overlap is not free.** FSDP's collectives almost hide behind each layer's
  compute, but hiding them means running alongside nearly all of it, and
  slowing it down the whole time. On top comes the weight prep (copying the
  gathered weights, quantizing the experts), which FSDP repeats every
  microbatch and PP's ZeRO-1 does once per step.
- **CP and TP make it much worse.** They multiply the microbatches, and that
  exposes the communication from both sides: each microbatch has fewer tokens,
  so less compute to hide behind, and there are more of them, so more
  all-gathers and reduce-scatters. At CP 4, a layer's forward takes about a
  third of the time its all-gather does.
- **TP is less bad than CP.** Both sit at the bottom of the ranking, but TP also
  splits the dense weights, so FSDP has less to gather.

At 32K, FSDP 512 fares much better, which may be surprising at first. Memory
forces every config into CP 4 or TP 4, and that treats the two families very
differently. FSDP 512 gets the same 4 microbatches of 8,192 tokens per GPU as
8K without a split, so its FSDP cost does not grow, while longer attention
gives it more compute to hide behind. PP, on the other hand, now pays for the
split: exposed TP or CP, and a bigger bubble. So the gap closes to about a
second.

## Why PP 8 Wins Over PP 16

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

At 32K, though, PP 16 ties the leaders. The reason is memory, the one thing
PP 16 has going for it: each stage holds half the layers. At 32K every config
needs a 4-way split, and the two splits are not equally cheap. CP 4 exposes
little communication, since only attention communicates, while TP 4 exposes
its collectives around every dense layer. PP 8 does not fit with CP 4, so it has to take TP 4
and pay for it; PP 16, with half the layers per stage, fits CP 4. The cheaper
split makes up for PP 16's smaller expert GEMMs and bigger bubble. With TP 4,
PP 16 loses to PP 8 as usual.

## Why Interleaved 1F1B Wins Over 1F1B

Interleaving with `v = 2` beats plain 1F1B in every PP family that fits it,
and almost all of the gain is the bubble:

| PP 8, CP 1, TP 2 at 8K | Step | Bubble | Hops | Memory |
|---|---|---|---|---|
| 1F1B (chain) | 23.6 s | 2.1 s | 0.1 s | 133 GB |
| Interleaved, `v = 2` (cache) | 22.4 s | 0.7 s | 0.1 s | 142 GB |
| Interleaved, `v = 4` (cache) | 22.3 s | 0.25 s | 0.2 s | 137 GB |

- **The bubble shrinks by about `v`,** as [[step.1]] predicts. It matters most
  at 32K, where the pipeline runs fewer, longer microbatches.
- **The hops barely matter.** With 100 GB/s NICs a hop is short, so even plain
  1F1B, which exposes its hops, loses little to them.
- **The cost is memory:** the first rank holds more chunks in flight
  ([[sharding.6]]).
- **More is not always better.** With PP 16, `v = 4` leaves chunks of one or
  two layers, and the bubble grows back: at 8K it is slower than plain 1F1B
  in most splits.

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
- **PP 16:** chain's stack grows long enough to cost up to about a second, and
  direct wins that back; cache, under interleaving, gets most of it too.

**Verdict:** for the configs we rank, direct does not help. Where the leaders
run, TorchTitan's cache is as good, so the attention residuals cost PP little.

## What If TP and CP Were Overlapped?

The model charges TP and CP as fully exposed, the way TorchTitan runs them
today. But nothing prevents overlapping them, like FSDP's collectives: TP's
behind the dense GEMMs (async TP), CP's behind the rest of the layer. Would
that change the picture? Charging them like FSDP instead, overlapped and with
the same NVLink penalties:

::: figure overlap-tp-cp-table

- **TP gains the most,** since it exposed the most: under a second at 8K,
  about two at 32K, where TP 4's collectives are larger. CP, which exposed
  little to begin with, moves much less.
- **The near-ties reorder.** At 8K the lead passes from CP 2 to TP 2; at 32K,
  PP 8 with TP 4 pulls clear, and FSDP 512 with TP 4 overtakes PP 16 with CP 4.
- **The big picture holds.** PP 8 interleaved still leads at both contexts, and
  FSDP 512 stays well behind at 8K.

Overall, overlapping helps TP far more than CP, but the CP configs sit so
close that the picture barely changes. A rule of thumb for PP: reach for TP
when memory is tight, since it also splits the dense weights; when everything
fits, CP is slightly faster with TorchTitan's TP today. An overlapped TP would
tip that balance toward TP.

## How Far to Trust the Ranking

We have now worked out, from first principles, which configs to run and why.
But there is a catch: this is a model, and the leaders sit within a few
percent of each other. Small changes to the model can flip their order:

- **What it assumes:** NVLink bandwidth and overlap inside a 64-GPU NVL domain
  (measured up to 32 GPUs), TP and CP fully exposed, a hop moving as soon as
  both sides are ready, and the optimizer at its GPU time.
- **What it leaves out:** profiles of a real 512-GPU run show DistMoE
  slower than in our fake-process-group steps (about 1.65x the bare grouped
  GEMM instead of about 1.25x, plus waits at its barriers), and FSDP's dense
  collectives across all 512 GPUs slower than modelled. The first adds about
  the same to every config; the second only makes FSDP 512 slower still.

So the practical way to use the ranking is to take a few of the top configs
and run them on the cluster. What the model gives the practitioner is a clear
picture of what to expect and why: absolute times will be off, and the order
of near-ties may change, but the big picture should hold. PP beats FSDP 512 at
8K, interleaving (at least `v = 2`) beats plain 1F1B, PP 8 beats PP 16 unless memory forces the
split, and the transport hardly matters. If a run disagrees with that, it
points to an inaccuracy in the model.

## Notes
