# The Memory Problem

The blueprint is simple. First, find the parallelism configs that fit the
weight-related memory: the sharded master weights, gradients and optimizer state.
Whatever is left of the GPU's memory is that config's budget for activations.
Then, for each config, find the best speed-memory trade-off within that budget
using remat and offloading. We set step time aside until the next section. A
tool could search memory and speed together, but separating them keeps the
reasoning small enough to follow (at least for us humans!).

The section follows the same order. We first look at how large the persistent
state is, and the discussion grows from there into the sharding needed to fit
it on our cluster; then we turn to activations:

- What is the memory footprint we are dealing with?
- Shrinking the weight footprint: ZeRO/FSDP, EP and TP
- Shrinking the activations
- Spending the leftover memory on activations
- Which parallelism configs fit

## What is the memory footprint we are dealing with?

Let's start with some napkin math. Kimi K3 has 2.78T parameters. Stored in
fp32, the weights alone are 11.1 TB, about 39 times one GPU's 288 GB. And the
weights are only a fraction of what a training run has to store.

**Persistent state.** For the whole run, training keeps three things per
parameter: the fp32 master weight, its fp32 gradient, and the optimizer state
(Muon keeps one fp32 momentum buffer). That is 12 bytes per parameter:

|  | Bytes per parameter | Kimi K3 |
|---|---|---|
| Master weights (fp32) | 4 | 11.1 TB |
| Gradients (fp32) | 4 | 11.1 TB |
| Optimizer state (Muon, fp32) | 4 | 11.1 TB |
| **Total** | **12** | **33.4 TB** |

*33.4 TB is about 116 times one GPU's 288 GB, before a single activation is
stored.*

A small simplification: we ignore AdamW's extra optimizer state. Muon handles
the matrix projections, and AdamW covers the rest (the embedding, the LM head,
the norms and a few small projections), which is only 0.09% of the parameters.

**Stashed activations.** These are the tensors the forward pass keeps for the
backward pass (through `save_for_backward` in an `autograd.Function`, and the
same mechanism inside PyTorch's own ops). We look at them in detail later. Without
any rematerialization or offloading, one sequence stashes 746 GB at 8K and
3.0 TB at 32K (4x the footprint at 8K). We leave DistMoE out for now and bring
it in when we get to shrinking activations. We measure them with a simple pair of
[saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

All of this is deliberately simple: no parallelism, no remat, no offloading yet.
As we bring those in, these two categories shrink, but new ones creep in, such
as FSDP's transient buffers. Here is all of it on a log2 scale, where each
gridline is a factor of 2.

::: figure memory-by-kind

*Log2 axis: each gridline is a factor of 2, so halving something (say, sharding
it over 2 GPUs) shortens its bar by exactly one gridline. The dashed line is one
GPU. Both charts in this section share the axis.*

The key takeaway is the scale. The whole cluster has 512 x 288 GB = 147 TB, and
the persistent state alone fills a quarter of it. The stashed activations of a
single 8K sequence, 746 GB, are 2.6 times one GPU. It is an understatement to
call this model memory hungry!

## Shrinking the weight footprint: ZeRO/FSDP, EP and TP

Alright, time to roll up our sleeves! Let's shard the persistent state and fit
it on our cluster.

::: tree-step dp

We start with plain data parallelism over all 512 GPUs. On its own, data
parallelism only multiplies the problem: every GPU holds the whole model. What shards the
persistent state across those 512 copies is [ZeRO](https://arxiv.org/abs/1910.02054).

### Sharding the weights with ZeRO / FSDP

ZeRO comes in three stages, each sharding one more thing:

- **ZeRO-1** shards the master weights and the optimizer state.
- **ZeRO-2** also shards the gradients.
- **ZeRO-3** also shards the *compute weights*, the copy the matmuls run on and
  part of the *working set* (more on that below). This is what PyTorch's FSDP (https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) does by default, as
  long as you are not using pipeline parallelism (why? we will see later), so I
  use "ZeRO-3" and "FSDP" interchangeably.

Let's apply them one at a time and watch what happens to the memory per GPU
(green: less than the row above, red: more).

::: figure zero-table

**No sharding.** Our problem statement: every GPU holds all 33.4 TB of
persistent state.

::: tree-step zero1

**ZeRO-1.** Sharding the master weights and the optimizer state divides both by
512: the two green cells under persistent state. A big improvement! But what is
going on with the working set? This is where working from first principles pays
off. Once the master weights are sharded, how does a GPU run the model?
Specifically, where does it get the weights from? Each GPU all-gathers the full
compute weights from the shards, once per step, and keeps them for the whole
step: *that copy is the working set*. We compute in MXFP8, so most compute
weights take 1 byte per parameter, which is why the it is close to 2.78T, number
of parameters. Keep an eye on the working set; it matters a lot in a roofline
study. Overall, ZeRO-1 takes us from 33.4 TB per GPU to 14.0 TB.

::: tree-step zero2

**ZeRO-2.** Now the gradients are sharded too, so the persistent gradients
divide by 512. All good! But again the working set takes a hit. Backward
produces each layer's full gradient before reduce-scattering it to the shards,
and to hide that communication, the reduce-scatter of the previous layer runs
while the current layer computes. So at the peak a GPU holds two layers' full
gradients, in fp32: 242 GB. Still, we keep going down, to 3.15 TB per GPU.

::: tree-step zero3 snapshot

**ZeRO-3.** Finally, our very own PyTorch FSDP shards the compute weights as
well. The trick mirrors the reduce-scatter: FSDP all-gathers the next layer's
weights while the current layer computes, so it holds two layers' worth of
weights. In MXFP8 that is 61.8 GB, about a quarter of the fp32 gradients. We
get very close, 378 GB, but that is still over the 288 GB of one GPU.

Now for something interesting. Suppose we had 1,024 GPUs instead of 512. What
would the ZeRO-3 row look like? The persistent state halves, sure. But what
about the working set? Does it budge? Not one byte: each GPU still holds two
whole layers in flight, however many GPUs share the shards. Time for our first
nugget:

> NUGGET: FSDP's working set sets a floor that more GPUs cannot lower. With infinitely many GPUs the persistent state per GPU goes to zero, but Kimi K3 would still need 304 GB per GPU for the two layers in flight (242 GB of gradients plus 61.8 GB of compute weights), more than a GPU holds, before counting a single activation.

So FSDP is not enough; we need to shrink the working set itself. Almost all of
it is experts, and I like to think the people who first scaled MoE models hit
this same wall. Their answer was expert parallelism.

### Shrinking the expert weights with EP

Almost all of the working set is routed experts: 296 of the 304 GB. Expert
parallelism (EP) goes after exactly that part.

**How EP shards.** EP does not need GPUs of its own; it is carved out of the
same 512 GPUs that FSDP shards over. Let's take EP 64 as an example. Each
layer's 896 experts are divided into 64 groups of 14, and each group is then
FSDP-sharded over the remaining 512 / 64 = 8 GPUs. The dense layers do not
take part in EP: they stay FSDP-sharded over all 512 GPUs.

::: tree-step ep snapshot

So for the persistent state, nothing changes compared to FSDP (ZeRO-3): each
GPU still holds a 1/512 shard. What changes is what each GPU **gathers** for a
layer, its working set: only its group's 14 experts, instead of all 896. The
working set shrinks dramatically, from 304 GB to just 12.2 GB, bringing the
persistent state plus working set down to 86.5 GB, way under our 288 GB budget
and leaving about 200 GB for activations (we tackle those later).

::: figure zero-ep-table

However, **EP is not free**. It adds all-to-alls on the critical path, while
FSDP's communication can be overlapped with compute. These all-to-alls are so
expensive that serious training runs use custom kernels like DistMoE
(introduced later) to overlap them with the routed experts' own compute.

**Wait, why pair EP with FSDP and not with TP?** That is what my colleague
Tianyu asked me at this point, and I did not have a good answer. If you are
stumped like I was, look at how EP changes the makeup of the working set.
Without EP, the routed experts are 97% of it. With EP, each GPU gathers only
1/64 of them, while the dense layers stay the same, so their share grows to 62%:

::: figure working-set-flip

*EP shrinks the experts' part of the working set 64x, from 296 GB to 4.6 GB,
while the dense layers stay at 7.6 GB, so the dense layers become the larger
part. Percentages are each part's share of that config's working set.*

This flip is what Tianyu was pointing at. TP shards the dense layers. Before
EP they were 3% of the working set, so TP had little to gain; after EP they are
the larger part, and TP can go after them. So, lets take a step back from FSDP and explore TP.

> NOTE: The TP direction is mostly an exercise in building the mental model. TP puts communication on the critical path in every one of the 93 layers, on top of EP's all-to-alls, so this config will likely be slow. We work it out anyway and add it to the configs that fit in memory; the rankings will show whether it is indeed slow.

### Shrinking the dense weights with TP

Spoiler alert: TP shows up again later, to shrink the activations. For now,
let's carry on shrinking the weights.

**How does TP sharding work?** EP was carved out of the FSDP axis. TP instead
gets an axis of its own, because it splits the model in a different way.
Conceptually, though, its weight sharding is much like EP's: TP first splits
the dense weights, and each TP shard is then FSDP-sharded over the FSDP axis.
With TP 2, the mesh is 256 (FSDP) x 2 (TP), and a TP-sharded parameter is still
split 2 x 256 = 512 ways, so the persistent state per GPU does not move.

The interesting question is how TP works inside the MoE layers. torchtitan has
no expert TP. Instead, the TP ranks become EP ranks: EP is carved out of the
FSDP and TP axes together. With EP 64 and TP 2, an EP group of 64 GPUs spans 32
positions on the FSDP axis times the 2 TP ranks, and each group of 14 experts
is still sharded over 8 GPUs. So TP changes nothing for the experts; it only
shrinks the dense part.

::: tree-step tp snapshot

**What TP buys.** Under FSDP + EP, not much: the dense part of the working set
is already small. So let's go back to a clean slate and start from ZeRO-1,
which we discarded earlier.

::: figure ep-tp-table

Let's get our bearings with two rows: plain ZeRO-1, and ZeRO-1 + EP 64 without
TP. Here is the flip again. Under plain ZeRO-1, the experts are 98% of the
gradients. Add EP 64 and their gradients drop to 170 GB, while the dense
gradients stay at 227 GB. The compute weights flip the same way. Now TP has
something to go after: the dense part.

But with EP 64 there is little room for it. The boxed parts are what TP cannot
touch, and they add up to 262 GB, leaving only 25.7 GB of margin. That is not
enough for the dense part at any TP degree, not even TP 8.

So we raise EP to 128. The part TP cannot touch drops to 156 GB, and now TP has
room: TP 4 and TP 8 both fit.

That wraps up the weights. The tree that has been growing alongside the text
tracks which parallelism configs fit so far, and how much of each GPU their
weights take. Whatever is left is each config's budget for activations, and
that is where we go next.

::: tree-end

## How big are the activations?

The forward pass keeps some tensors around for the backward pass. We call them
*stashed activations*, to avoid the word "saved", which is easily confused with
saving memory. Unlike the persistent state, they scale with tokens, not
parameters.

To keep things simple, a microbatch is one sequence: 8,192 tokens at 8K, or
32,768 at 32K. Measured with a fake-tensor run (see [[methodology.1]] for how), one sequence stashes
**746 GB at 8K** and **3.0 TB at 32K** (4x the footprint at 8K). The toggle at the
top right switches the charts between the two.

Where do they go? At 8K:

| Stashed by | GB | Share |
|---|---|---|
| Routed experts | 420 | 56% |
| Attention (KDA and MLA) | 244 | 33% |
| Shared experts, norms, head, loss, layer 0 | 57 | 8% |
| Attention residuals | 25 | 3% |
| **Total** | **746** |  |

More than half of it is the routed experts. Each token goes to 16 experts, so each of its
expert intermediates is stashed 16 times over.

In practice the routed experts run in DistMoE, which manages their activations
itself, in preallocated slot buffers, and has its own rematerialization knob:
how big those buffers are. Big enough for every layer's expert intermediates,
nothing is recomputed. With no buffers, DistMoE keeps only the expert inputs
and recomputes the rest in backward (today's default). Per 8K sequence on
FSDP + EP 64, with nothing else rematerialized:

| DistMoE | Stashed | DistMoE's backward |
|---|---|---|
| Keeps its intermediates (buffers sized for balanced routing) | about 660 GB | no recompute |
| Recomputes them (default) | about 346 GB | 1.5 ms more per MoE layer |

*Measured on 8 layers (and one MoE layer for the buffers), scaled to all 93.*

Either way, one sequence is more than a whole GPU. That is what we are dealing
with; the next section shrinks it. DistMoE's buffer size is one of the knobs:
if a config has memory to spare, growing the buffers so DistMoE stops
recomputing is one way to spend it.

This is deliberately simple again. Tensor parallelism (with sequence
parallelism) and context parallelism split each sequence across GPUs and shrink
what each one stashes; we bring them in when we get there.

The attention residuals run on TorchTitan's fused Triton kernel (an opt-in
override adapted from Flash Linear Attention). It reads the block stack and the
current partial sum directly, so it stashes no concatenated copies of them.

## Use the leftover memory for activations effectively

Each config that fits its weights leaves some memory over, and that is its
budget for activations. Here is the recipe for spending it:

1. **Start simple:** no rematerialization, no offloading, one sequence per
   microbatch. Check whether it fits the budget.
2. **If it fits** (rare for LLMs), grow the microbatch as long as it still fits
   and stays within the GPU's share of the batch. Under FSDP this pays: FSDP
   gathers the weights and reduce-scatters the gradients for every microbatch,
   so fewer, larger microbatches mean less communication. (ZeRO-1 communicates
   once per step either way.)
3. **If it does not fit** (our case: a single 8K sequence is already more than
   a GPU), find the fastest mix of rematerialization and offloading that fits.
   Measure the memory-time Pareto frontier of the two together and take the
   fastest point within the budget, counting DistMoE's buffer size as one more
   knob. The memory floor is offloading everything,
   not full rematerialization, which turns out to be neither the smallest nor
   the fastest option. As in step 2, a point that frees enough memory for a
   larger microbatch is worth keeping for FSDP configs.
4. **Context and tensor parallelism are part of the config, not a last
   resort.** Each splits every sequence across its GPUs and shrinks the
   activations roughly by its degree, at the cost of communication on the
   critical path (and, with fewer data-parallel replicas, more microbatches
   per GPU). A high enough degree can remove the need for rematerialization
   altogether, so whether CP 4 without remat beats CP 1 with full remat is for
   the ranking to decide, not a fixed order.
5. **Pipeline parallelism** is the last lever: it shrinks the weights to one
   stage, but multiplies the activations by the microbatches in flight.

The rest of this section follows the recipe: the remat and offload frontier,
then context and tensor parallelism, then pipeline parallelism.


<!--
## Rematerialization

Rematerialization (torch_remat, commonly known as activation checkpointing) stashes fewer tensors in the forward pass, and recomputes them during backward. With full remat, each layer keeps only its input (plus the
routing decisions), and the activations of one 8K sequence drop from 746 GB to
**15.6 GB**, about 48x less. The price is one extra forward pass: 8 FLOPs per parameter per
token instead of 6, about 33% more compute.

**Recompute's contribution to the peak.** Before each layer's backward, remat
reruns its forward, recreating that layer's intermediate activations: about
6 GB per 8K tokens, one layer at a time. The peak comes at the first layer of
the backward pass, when the whole stash is still live, so it is the stash plus
one layer's working set.

The chart below compares no remat, TorchTitan's default remat (which keeps
every region the model declares except the routed experts' first projection,
w13, recomputing only that) and full remat, drawn as its stash plus the
recompute peak. From here on, full remat means both: 21 GB per 8K sequence.

::: figure memory-remat

## DistMoE

[DistMoE](https://github.com/meta-pytorch/dist_moe) is a recent library from
the PyTorch team for running MoE layers efficiently: its kernels overlap the
token dispatch and combine with the expert GEMMs, drop no tokens, and work with
CUDA graphs. Any production training system has something like it to efficiently manage
routing imbalance, and its implementation changes both memory
and speed. So even though it pulls us away from the idealized picture, we bring
it in now.

Memory is tight, so we configure DistMoE to recompute too: per MoE layer it
stashes only the layer's input and recomputes the expert intermediates in
backward. On top of full remat, that adds four things:

- **Its stashed input**, held in DistMoE's own preallocated buffer.
- **Its output**, which full remat keeps instead of recomputing (why: see
  [[methodology.2]]).
- **Scratch** for one layer's temporaries, reused by every layer and sized with
  a capacity factor of 4 so that experts receiving more than their share of
  tokens still fit.
- **Routing indices**: three int64 index arrays per MoE layer, stashed so the
  backward can send gradients back to the right tokens. They are negligible
  without expert parallelism (0.3 GB at 8K), but they are sized for the worst
  case across the expert-parallel group, so with EP 64 they grow 56x, to 16 GB.

For one sequence:

|  | 8K | 32K |
|---|---|---|
| Full remat (stash + recompute peak) | 21.2 GB | 85.0 GB |
| DistMoE stashed inputs | 5.4 GB | 21.6 GB |
| DistMoE outputs | 5.4 GB | 21.6 GB |
| DistMoE scratch (capacity factor 4) | 23.6 GB | 94.5 GB |
| **Activations** | **56 GB** | **223 GB** |

::: figure memory-distmoe

> NOTE: DistMoE's scratch gets large at 32K: 94.5 GB, about half of one GPU's 192 GB on its own. It scales with the capacity factor, so it is a knob we can turn down once we know how imbalanced the routing really is. For now we keep the factor at 4.

From here on, "activations" means full remat (stash plus recompute peak) plus
DistMoE's stashed inputs, outputs and scratch -- 56 GB per 8K sequence and
223 GB per 32K sequence. The routing indices come in with expert
parallelism.

## MXFP8

MXFP8 runs the matmuls on 8-bit operands with a shared scale per 32 elements,
about 1 byte per element instead of 2. We use it for the routed experts only,
through DistMoE's own MXFP8 kernels; the dense linears stay in bf16. Memory
barely changes: **with full remat, a layer stashes only
its bf16 input, so there is little left for MXFP8 to shrink.** DistMoE still
stashes its input in bf16, and its scratch shrinks by only about 6% (23.6 to
22.2 GB at 8K), since most of its temporaries stay in bf16. The rest of the
post keeps the bf16 figure, 23.6 GB, a slightly conservative bound.

## Where We Stand

::: figure memory-summary

Rematerialization brought the activations down to 56 GB per 8K sequence, well
under one GPU. **At 32K the problem is not solved**: 223 GB is still more than a
whole GPU, before anything else is stored. We come back to it with context and
tensor parallelism ([[memory.4.1]], [[memory.4.2]]). The persistent state has
not moved: 33.4 TB, 174 times one GPU. No single-GPU trick fixes that; it has
to be split across many GPUs. -->

### Context Parallelism

Context parallelism (CP) splits each sequence across GPUs: with CP 2, each GPU
holds half of a sequence's tokens. Everything that works token by token -- the
projections, the norms, the MoE layers -- runs on the local tokens with no extra
communication. Only attention needs the other GPUs' tokens: MLA uses Ulysses,
and KDA, being a recurrence, passes a small summary of each chunk's state along.
What that costs in time comes later; here we only care about memory.

For memory, CP is exactly the lever the previous step was missing. Activations
scale with tokens per GPU, so CP divides them, DistMoE's stash, output and
scratch included. Nothing else changes. **The CP ranks join the FSDP group**, so
each GPU still stores 1/512 of the persistent state, and the layers in flight
are the same weights and gradients whatever the token count. The work per GPU
does not change either: each GPU now runs CP times as many microbatches, each
1/CP of a sequence. In the table, "shards" are the 1/512 slices of the master
weights, optimizer state and gradients.

::: figure cp-table

At 32K, CP 2 is not enough (230 GB), but CP 4 fits: 158 GB, the same as 8K
without CP, since each GPU again holds 8,192 tokens per microbatch. DistMoE's
index arrays shrink with CP too, because they scale with tokens per GPU.

### Tensor Parallelism

Tensor parallelism (TP), which we always pair with sequence parallelism, is the
other lever for activations: like CP, it splits the tokens across GPUs. It also
splits the dense weights (not the experts). Under FSDP over 512 GPUs, those are
already tiny per GPU, so the extra saving is small.

::: figure tp-table

So under FSDP, TP and CP are interchangeable for memory: TP 2 and CP 2 differ by
4 GB. TP's weight splitting pays off with pipeline parallelism, which keeps a
stage's weights gathered all step.

On memory, TP looks like a free win; on speed, it may not be. CP splits along
the sequence, so it scales with attention, which grows with context length. TP
splits the dense projections themselves, which are already small, and narrower
matmuls run less efficiently. The performance roofline will have to weigh this.

Putting it together, FSDP + EP 64 with each sequence split by CP or TP:

::: figure split-steps

> INSIGHT: For memory alone, when activations are what overflow, reach for CP or TP before pipeline parallelism. Both split the tokens of a sequence, and activations are the only part of GPU memory that grows with them, so they cut exactly the part that is too big. The sharding of weights, gradients and optimizer state stays as it was. For speed it is another matter: without PP, CP and TP are the slowest configs of all ([[ranking.1]]).

### Pipeline Parallelism

So far every parallelism has been SPMD: every GPU runs the same program on its
own slice of the data, weights or tokens. FSDP and EP, with CP 4 or TP 4 at 32K,
already fit the model this way. So why look further? Because memory is not the
only thing that matters:

1. Without PP, FSDP gathers and reduce-scatters every layer for every
   microbatch, across all 512 GPUs. CP and TP multiply the microbatches, and
   with them that traffic and the weight copies and re-quantization that come
   with each gather.
2. Those all-gathers and reduce-scatters run on the GPU's SMs and slow down the
   compute they overlap with.

We model both later. For now, take them as a reason to look, and see how the
model shards with PP.

Pipeline parallelism (PP) cuts the model into stages of consecutive layers and
gives each stage its own group of GPUs. With PP 8, for example, there are eight
stages of about 12 layers (93 does not divide evenly), each on 64 GPUs (one NVL
domain). Inside a stage we use FSDP and EP as before, over the stage's GPUs, so
EP stays 64 (it drops to 32 with PP 16, which the table below also shows).

What PP does to memory:

- **Each GPU only stores its stage.** But 1/P of the layers, sharded over 512/P
  GPUs, is still about 1/512 of the model: *same memory as before*.
- **Microbatches pile up on the first stage.** With the 1F1B schedule, the
  first stage holds P microbatches in flight, each stashing activations for its
  93/P layers. P microbatches x 93/P layers is about one whole-model
  microbatch: *same memory as before*. This depends on the schedule, though:
  interleaving, in the next section, keeps more in flight.
- **ZeRO-1 becomes affordable.** Without PP, ZeRO-1 meant every GPU holding the
  whole model's compute weights and gradients, 14 TB. With PP it only has to
  hold one stage's, and that fits. This is FSDP's default under PP: a stage's
  compute weights stay gathered and its gradients unsharded for the whole step,
  instead of being regathered and reduce-scattered for every microbatch. That
  is where the performance win can come from, at the cost of some memory.

> INSIGHT: A ZeRO stage is really a choice of how much stays unsharded. PP shrinks that from the whole model to one stage, which turns ZeRO-1 from impossible (14 TB per GPU) into affordable, and with it, keeping weights gathered and gradients unsharded across microbatches instead of paying FSDP's communication on every one.

Here are PP 8 and PP 16 under ZeRO-1, with each sequence split by
CP or TP (master weights and optimizer state are 45 GB per GPU in every row):

**PP 8** (eight stages of 11-12 layers, 64 GPUs each, so FSDP 64 and EP 64 within a stage):

::: figure pp8-table

**PP 16** (sixteen stages of 5-6 layers, 32 GPUs each, so FSDP 32 and EP 32 within a stage):

::: figure pp16-table

> NOTE: PP 16's activations are 7 GB smaller than PP 8's only because DistMoE's routing indices shrink with EP (32 instead of 64). Everything else in flight is the same: the first stage holds 8 microbatches x 12 layers with PP 8, and 16 x 6 with PP 16, 96 layer-microbatches either way.

At 8K everything fits but PP 8 without a split, which lands right on the
192 GB line (192.03 GB; we count it as not fitting). At 32K, every config needs
CP 4 or TP 4, and PP 8 with CP 4 sits on the same line: ZeRO-1's unsharded
stage (73 GB with PP 8: bf16 dense weights and fp32 gradients) comes on top of
the activations and of DistMoE's index arrays, which with 1F1B pile up for
every microbatch in flight. And TP finally pays off: TP 2 beats CP 2 by about
22 GB with PP 8, 133 against 155 GB at 8K.

> INSIGHT: Once EP shrinks the routed experts on each GPU, and ZeRO-1 (common with PP) keeps the stage's weights and gradients unsharded all step, the dense weights become a large share of what each GPU holds, and only TP splits them. Under FSDP alone, only two layers are ever gathered, so TP gains little over CP for memory. Details in [[methodology.3]].

### Interleaved Pipeline Parallelism

Interleaved 1F1B gives each rank `v` smaller stages (virtual stages) instead of
one: with PP 8 and `v = 2`, rank 0 runs layers 0-5 and 48-53, rank 1 layers
6-11 and 54-59, and so on, so each microbatch loops through the ranks `v`
times. What it buys is a smaller pipeline bubble, by a factor of `v`, which we
come back to with the step time. What it costs is memory:

- **Each GPU stores the same weights.** A rank still holds 1/P of the layers,
  only in `v` pieces: the shards, compute weights and gradients do not change.
- **More microbatches pile up on the first rank.** Before its first backward,
  rank 0 runs `(v - 1) x P + 2 x (P - 1)` forward chunks (PyTorch's
  `ScheduleInterleaved1F1B`), so it holds about `(v + 1) P` chunk-microbatches
  in flight instead of P. Each chunk has only `93 / (P v)` layers, so in layers
  that is about `(v + 1) / v` times 1F1B's: 1.5x at `v = 2`, 1.25x at `v = 4`.
  Everything stashed per layer and microbatch grows with it: the activations,
  DistMoE's routing indices, and its stashed inputs and outputs.

Rank 0 of each schedule. At 32K the table uses CP 4: without a split, 32K has
only 8 microbatches per pipeline, and every schedule already holds all 8. With
CP 4 each GPU holds the same 8,192 tokens per microbatch as 8K without a
split, so the two contexts give the same numbers:

::: figure pp-vpp-table

Interleaving costs 19 GB with PP 8 at `v = 2`, and less at `v = 4`, whose
smaller chunks hold fewer layers in flight. Every config that fits with 1F1B
still fits interleaved: at 32K, PP 8 with TP 4 and PP 16 with CP 4 or TP 4 fit
at every `v` ([[memory.4.5]]).

> NOTE: The attention residuals' blocks do not add to this. With interleaving, TorchTitan's cache transport ([[step.2]]) keeps on each rank the blocks a microbatch has already brought there, so a hop carries only the new ones. Those cached blocks are the same tensors the stash already counts.

### What Fits

Putting it all together, every config in this section (green: fits in 192 GB,
red: does not):

::: figure fit-map

32 of the 35 fit at 8K, and 11 of the 35 at 32K. Interleaving never changes
the answer: what fits with 1F1B fits interleaved, and what does not, does not.

### How Close Are We to a Real Run?

Every number so far comes from the model. Checking it on 512 GPUs would be
expensive, but PyTorch has a neat trick: a **fake process group**. One process
plays a single rank of the 512-GPU job. Collectives do no communication, but
every tensor that rank would allocate is allocated for real, so the step's peak
memory can be measured on a single GPU. For PP, a hybrid mode runs one real
process per stage, with real send/recv between them, and fakes everything
else. TorchTitan exposes both modes (`--comm-backend fake` and
`real_pp_fake_spmd`).

Peak allocated memory per GPU, model vs fake run:

| Config | Model | Fake run |
|---|---|---|
| FSDP 512, EP 64, 8K | 158 GB | 157 GB |
| FSDP 512, EP 64, CP 2, 32K | 230 GB | 228 GB |
| PP 4, ZeRO-1, 8K (first stage) | 237 GB | 228 GB |

The model lands within about 4% of the measured peak. Two things it leaves out:

- **Fragmentation.** The caching allocator reserves more than it hands out.
  In the PP 4 run it reserved 279 GB to serve a 228 GB peak.
- **Communication buffers.** NCCL's own buffers are not allocated in a
  fake run. They add a few GB per GPU in a real job.

So the model is accurate for what the program allocates. A config near
192 GB still needs headroom for both.

Memory alone does not pick a winner: next we build a performance model to rank
the configs that fit.

## Notes

### Parked from the old "Using Parallelism to Fit the Model" section

Not published; move into the new text if still wanted. Their [[memory...]] links are stale.

**Compute weights.** Sharding the master weights has a catch. A GPU holding 1/512
of the fp32 master weights cannot run a forward pass on them, so it needs a
separate copy of the weights in the format the matmuls use: the **compute
weights**: MXFP8 for the routed experts, about 1 byte per parameter, and bf16
for everything else. (FSDP gathers the experts in bf16 and frees that buffer
once their MXFP8 copy is built; the fake runs in [ [memory.7.8]] agree.) Under
ZeRO-1 and ZeRO-2 every GPU holds the whole model's compute weights: 2.84 TB,
which after ZeRO-2 is the largest thing left.

**Padding.** FSDP splits each tensor along its first dimension, and pads it when
that does not divide evenly. For the expert weights that dimension is the expert
count, 896 over 512 GPUs, which pads to 1,024: 14% more for the expert part of
every shard.

**Caveat: DistMoE's routing indices.** For backward, DistMoE stashes three
int64 index arrays per MoE layer, sized for the worst case: EP x tokens x 14
entries. With EP 64 at 8K that is 16.2 GB over 92 layers, which is why the EP
row's activations jump from 56 to 72 GB in [ [memory.7.1]]'s table. (We are
working with the DistMoE authors to shrink this.)
