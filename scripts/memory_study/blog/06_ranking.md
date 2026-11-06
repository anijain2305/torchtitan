# Ranking

**From compute to step time.** Without PP (FSDP 512 + EP 64), every microbatch
runs all 93 layers: FSDP's all-gather and reduce-scatter of the next layer
overlap the current layer's compute, with the penalties of [[communication]];
EP's dispatch and combine run inside DistMoE's expert GEMMs; TP's and CP's
collectives are exposed in full. The step is one microbatch's time, times the
number of microbatches. With PP, the schedule simulation of [[step]] gives the
step from each chunk's time and each hop, and ZeRO-1 adds the stage's
all-gather and reduce-scatter once per step. The optimizer step is left out
([[open_problems]]).

The configs that fit in memory ([[sharding.7]]), ranked by step time (the toggle
switches context). For PP, each config also names its schedule and its hop
transport: chain, cache or direct ([[step]]). The chart splits each step into
parts (hover a row for the numbers; the buttons filter by family, schedule and
transport, and sort by the parts the postmortems below look at):

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
- **Optimizer**: DistMuon's step, at the GPU time it needs ([[ranking.5]]).

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
(EP 32 instead of EP 64). That is **the** reason PP 16 is slower here:

- **Smaller expert GEMMs.** With EP 32 each GPU holds twice the experts (28
  instead of 14) but processes the same tokens, so each expert gets half the
  rows, and the expert kernels run less efficiently. This is most of the gap,
  except at `v = 4`.
- **Twice the hops.** A microbatch crosses twice as many stage boundaries, and
  with half as many data-parallel replicas each pipeline runs twice the
  microbatches.
- **Uneven chunks at `v = 4`.** 93 layers over 64 virtual stages leaves chunks
  of one or two layers, and the imbalance stalls the schedule: at `v = 4` this,
  not the experts, is the bigger cost.

::: figure pp16-vs-pp8-table

What PP 16 has going for it is memory: each stage holds half the layers. That
lets it run CP 4 at 32K, which PP 8 cannot fit, and there it ties the leaders.

## Why Interleaved 1F1B Wins Over 1F1B

Interleaving wins in every PP family that fits it, and almost all of the gain
is the bubble:

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
  two layers, and the bubble grows back.

## Do Attention Residuals Hurt PP?

**The problem.** Attention residuals give PP an unusual property: a stage needs
the blocks of every stage before it, so with chain each hop carries a bigger
stack than the last, and the latest stages, the slowest already, trade the
most bytes, on the critical path ([[step.2]]). A natural fix is **direct**,
which TorchTitan does not have: each rank sends every block it finishes
straight to the later stages that use it, asynchronously, so a hop carries
only the hidden state, and the block traffic spreads over all pairs of ranks,
off the critical path, while the stages compute.

::: figure transport-diagram

With the two schedules, that makes four options to compare:

::: figure transport-matrix

**The numbers.** With 100 GB/s NICs, hops are cheap to begin with, so there is
little for direct to save:

::: figure transport-compare-table

- **PP 8:** the transport barely matters with any schedule.
- **PP 16:** chain's stack grows long enough to cost up to about a second, and
  direct wins that back; cache, under interleaving, gets most of it too.

**Verdict:** for the configs we rank, direct does not help. Where the leaders
run, TorchTitan's cache is as good, so the attention residuals cost PP little.

## The Optimizer Step

Every config ends its step with the optimizer: DistMuon, Muon over the sharded
parameters. Muon orthogonalizes each weight matrix with a few Newton-Schulz
iterations, so it needs whole matrices, and how much it has to move depends on
how they are sharded:

- **Routed experts, 98% of the parameters: nothing to move.** FSDP shards each
  group's 14 experts along the expert dimension, so a GPU's shard of the
  gradient, momentum and weights is two whole experts, and it orthogonalizes
  them where they are.
- **Dense weights: a small redistribution.** They are sharded by rows, so
  DistMuon gathers each matrix (or each head's block) to one GPU, runs
  Newton-Schulz there and sends the update back: about 0.2 GB each way per GPU
  per step, a few milliseconds.

So the optimizer is almost all compute, and about the same per GPU in every
config: each GPU holds about 1/512 of the parameters either way. We measured it
in fake-process-group steps at 8K:

| Optimizer step | GPU work (the model) | Eager, as TorchTitan runs it today |
|---|---|---|
| FSDP 512, EP 64 | 0.8 s | 3.1 s |
| PP 4, first stage | 0.5 s | 0.5 s |

The model charges the GPU work, its practical roofline: 0.8 s without PP, 0.5 s
with it. Today FSDP 512's step takes 3.1 s, because DistMuon runs eagerly: for
all 93 layers' parameters it launches about 700,000 small kernels, and the GPU
waits on the CPU that launches them. A PP rank holds a quarter of the layers or
fewer, so the CPU keeps up. The step cannot be captured in a CUDA graph yet
(among other things, its learning rate is a Python number, baked into a
capture); once it can, FSDP 512 gets those 2.3 s back. Either way, the
optimizer does not change the ranking.

## Postmortem: FSDP 512 Without PP

Every FSDP 512 config is far behind every PP config: 43 s at best, against 17
s at 8K and 22.5 s at 32K. All the splits that fit:

::: figure fsdp-postmortem-table

**Why all of them are slow: the experts' gathers.** FSDP gathers every layer's
weights before using them, and EP 64 leaves each expert shard with only 8
replicas, one per NVL domain, so its all-gather and reduce-scatter run on one
NIC per GPU ([[communication.2]]). Per layer and microbatch, that all-gather
takes 24 ms, 20 ms of it the expert shard, while the layer's forward takes 6.5
ms at 8K without a split: most of the gather cannot hide, 17.5 s per step even
in the best config.

**Why splitting makes it worse.** CP and TP split each microbatch's tokens
over 2 or 4 GPUs. The tokens per step are fixed, so each GPU runs 2 or 4 times
more microbatches (4, then 8, then 16 at 8K). But FSDP gathers every layer's
weights again for every microbatch, however few tokens it has, while the
compute it could hide behind shrinks with the tokens: at CP 4 a layer's
forward takes 2 ms against the same 24 ms gather. So the exposed FSDP time
grows with the microbatch count, from 17.5 s to 49 s with CP 2 and 111 s with
CP 4; so does the weight prep (copies of the gathered weights, MXFP8 quantization of the
expert weights), from 2.2 s to 9.0 s.

**Why CP 4 is worse than TP 4.** Both run 16 microbatches per GPU, with about
the same compute per microbatch. The difference is what each gathers: TP 4
also shards the dense weights 4 ways, so each GPU gathers 0.32 GB of them per
layer instead of 1.27 GB (the 0.92 GB expert shard is the same in both). TP 4
saves 11.6 s of exposed FSDP, 3.6 s of weight prep and 1.7 s of penalties, and
gives back 3.2 s of exposed TP and 0.7 s of slower compute: 137 s against 150
s. At 32K both run 4 microbatches per GPU, like 8K without a split, and land
there too (43-45 s); TP 4 is ahead again, by 1.8 s, for the same reason.

## Why PP 16 Loses to PP 8

At the same split, schedule and transport, PP 16 is slower than PP 8 every
time, by about 1 s, and by 3 s at `v = 4`:

::: figure pp16-vs-pp8-table

- **Smaller expert GEMMs (compute).** PP 16 runs EP 32: each GPU holds 28
  experts instead of 14, so each expert gets half the tokens, and the grouped
  GEMM runs less efficiently (at TP 2, 2,304 rows per expert at 52% of peak,
  against 4,608 at 57%).
- **Twice the hops.** A microbatch's round trip crosses 30 hops instead of 14,
  and with half as many data-parallel replicas, each pipeline runs twice the
  microbatches. More hop time, and more of the compute penalty hops cost.
- **A slightly bigger bubble**, 15 stage-steps of half the size instead of 7,
  and a much bigger one at `v = 4`: 93 layers over 64 virtual stages leaves
  chunks of one or two layers, and the uneven chunks stall the schedule.

What PP 16 has going for it is memory: each stage holds half the layers, so it
needs 13-27 GB less. That lets it run configs PP 8 cannot fit, such as
interleaving with CP 4 at 32K, where it ties the leaders (22.9 s).

## Transports Compared

The same PP configs with each transport ([[step.2]]):

::: figure transport-compare-table

- **Chain is never the fastest.** It costs 0.2 s at 32K with PP 8, and up to
  1.8 s with PP 16 at 8K, almost 3 s at `v = 4`: the longer the pipeline, the
  bigger the stack each hop carries.
- **With plain 1F1B, direct beats chain everywhere.** The hops sit on the
  critical path ([[step.1]]), so taking the blocks off them pays every
  microbatch: 0.5 s at PP 8, 1.8 s at PP 16 (8K).
- **With interleaving, cache and direct tie at PP 8,** within 0.1 s: they move
  the same bytes, cache spread evenly over the hops, direct piled on the first
  rank. At PP 16, direct pulls ahead (0.3 s at `v = 2`, 0.8 s at `v = 4`): with
  twice the stages, the blocks cache carries for a lap make hops too big to
  hide completely.

TorchTitan has chain for plain 1F1B and cache for interleaving. Of the two,
cache is the one that matters: the leading configs interleave, and there it
is as good as direct at PP 8.

**At 8K, interleaved PP 8 leads, with what TorchTitan has today.** PP 8,
interleaved (`v = 4`) with cache takes 17.0 s, with CP 2 or TP 2 alike: TP 2
pays 1.1 s of exposed TP, CP 2 pays 0.5 s of exposed CP but runs its compute
slower. Interleaving helps twice. It shrinks the bubble, from 1.7 s with plain
1F1B to 0.7 s with `v = 2` and 0.3 s with `v = 4`. And it hides the hops
([[step]]): they add 0.1-0.2 s, against 0.7 s with plain 1F1B and chain. So
direct distribution matters only for plain 1F1B (18.7 s with chain, 18.1 s
direct). With interleaving it does not help: the hops are already hidden, and
direct moves the same bytes as cache but piles them on the first rank, which
sends the embedding to every other rank, so its compute slows the most. Chain
under interleaving, which TorchTitan does not run and the chart leaves out,
only loses: 17.4 s at
`v = 2` and 17.6 s at `v = 4`, since every lap resends the whole stack; at PP 16
with `v = 4` its hops grow too big to hide (21.9 s, against 19.8 s with cache).

**At 32K, the top configs tie.** PP 8, 1F1B with CP 4, with direct
distribution (22.5 s) or chain (22.7 s); PP 8, interleaved (`v = 4`) with cache
and TP 4 (22.9 s); and PP 16, interleaved (`v = 2`) with CP 4 (22.9 s) sit
within 0.4 s, and chain and cache run on TorchTitan today. Memory forces
every config at 32K into CP 4 or TP 4, and each pays for it differently. PP 8
with CP 4 cannot interleave (it does not fit), so its 32-microbatch pipeline
keeps a 3.8 s bubble; its hops cost little either way, because its stages are
uneven: the last one, which also runs the LM head, sets the pace, and the
others have more spare time per microbatch than a hop takes. PP 8 interleaves
only with TP 4, which adds 2.7 s of exposed TP. PP 16 interleaves with CP 4, but not well beyond `v = 2`: 93
layers over 64 virtual stages leaves chunks of one or two layers, and at 32K an
MLA layer weighs much more than a KDA one, so the uneven chunks stall the
schedule (a 4.5 s bubble at `v = 4`, 2.2 s at `v = 2`).

**The margins are small.** The top configs sit within 2% of each other at both
contexts, so the remaining assumptions matter: bandwidth and overlap inside a
64-GPU NVL domain (measured up to 32 GPUs), TP and CP charged as fully exposed
(a chunked, overlapped version of either would move the TP and CP configs up),
the schedule simulation (each chunk's forward and backward from the model, a
hop moving as soon as both sides are ready), the uneven split of 93 layers over
the stages, and the optimizer step, which is not included. FSDP 512 could also
keep each NVL domain's experts replicated and reduce their gradients across
NVL domains once per step (HSDP), which we have not modelled.

## Notes
