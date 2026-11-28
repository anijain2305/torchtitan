# The Memory Problem

The blueprint is simple. First, find the parallelism configs that fit the
weight-related memory: the sharded master weights, gradients and optimizer
state. Whatever is left of the GPU's memory is that config's budget for
activations. Then, for each config, find the best speed-memory trade-off within
that budget using remat and offloading. The section is organized as follows:

- [What is the memory footprint we are dealing with?](#sec-memory-1)
- [Shrinking the weight footprint: ZeRO/FSDP, EP, TP and PP](#sec-memory-2)
- [How big are the activations?](#sec-memory-3)
- [Use the leftover memory for activations effectively](#sec-memory-4)
- [What parallelism configs fit](#sec-memory-5)

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
same mechanism inside PyTorch's own ops). We look at them in detail later.
Without any rematerialization, offloading or DistMoE, one sequence stashes
746 GB at 8K and 3.0 TB at 32K. We measure them with a simple pair of
[saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

All of this is deliberately simple: no parallelism, no remat, no offloading yet.
As we bring those in, these two categories shrink, but new ones creep in, such
as FSDP's transient buffers. Here is all of it:

::: figure memory-by-kind

*Log2 axis: each gridline is a factor of 2, so halving something (say, sharding
it over 2 GPUs) shortens its bar by exactly one gridline. The dashed line is one
GPU.*

The key takeaway is the scale. The whole cluster has 512 x 288 GB = 147 TB, and
the persistent state alone fills a quarter of it. The stashed activations of a
single 8K sequence, 746 GB, are 2.6 times one GPU. It is an understatement to
call this model memory hungry!

## Shrinking the weight footprint: ZeRO/FSDP, EP, TP and PP

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
  part of the *working set* (more on that below). This is what PyTorch's [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) does by default, as
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
weights take 1 byte per parameter, which is why it is close to 2.78 TB, about
one byte per parameter. Keep an eye on the working set; it matters a lot in a roofline
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

**How EP shards.** EP does not need GPUs of its own; it is carved out of the
same 512 GPUs that FSDP shards over. Let's take EP 64 as an example. Each
layer's 896 experts are divided into 64 groups of 14, and each group is then
FSDP-sharded over the remaining 512 / 64 = 8 GPUs. The dense layers do not
take part in EP; they stay FSDP-sharded over all 512 GPUs.

::: tree-step ep snapshot

So for the persistent state, nothing changes compared to FSDP (ZeRO-3): each
GPU still holds a 1/512 shard. What changes is what each GPU **gathers** for a
layer, its working set: only its group's 14 experts, instead of all 896. The
working set shrinks dramatically, from 304 GB to just 12.2 GB, bringing the
persistent state plus working set down to 86.5 GB, way under our 288 GB budget
and leaving about 200 GB for activations.

::: figure zero-ep-table

EP's cost is its all-to-alls, which sit on the critical path. They can be so
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
the larger part, and TP can go after them. So, let's take a step back from FSDP and explore TP.

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

The interesting question is how TP works inside the MoE layers. TorchTitan has
no expert TP. Instead, the TP ranks become EP ranks: EP is carved out of the
FSDP and TP axes together. With EP 64 and TP 2, an EP group of 64 GPUs spans 32
positions on the FSDP axis times the 2 TP ranks, and each group of 14 experts
is still sharded over 8 GPUs. So TP changes nothing for the experts; it only
shrinks the dense part.

::: tree-step tp snapshot

**What TP buys.** For memory under FSDP + EP, not much: the dense part of the
working set is already small. So let's go back to a clean slate and start from ZeRO-1,
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

### Shrinking the weights with PP

Before we turn to activations, let's look at pipeline parallelism (PP). PP is
known to be complex, because the training system is no longer SPMD: you have to
send data between pipeline stages, balance the stages, pick a pipeline schedule,
and more. So why take on all that complexity when an FSDP config already fits in
memory? The reason is FSDP's communication. FSDP gathers and reduce-scatters
every layer for every microbatch. If there is not enough compute to hide it,
that communication is exposed; and even when it is hidden, it slows down the
compute it overlaps with (as we will measure later). If that is the case in your
pretraining run, PP becomes an attractive option.

**How PP shards.** PP splits the layers into stages, and each stage owns a group
of GPUs, within which it is SPMD. With PP 8, for example, each stage holds 12 of
the 93 layers on 64 GPUs. Everything we have seen so far then happens *inside* a
stage: ZeRO-1 shards the stage's master weights and optimizer state over its
GPUs, EP spreads the stage's experts over them, and the same goes for TP and CP
(discussed later).

The interesting part is that PP makes ZeRO-1 affordable. Let's see how, by
comparing ZeRO-1 + EP 64 from 3.2.3 with ZeRO-1 + PP on its first stage. PP 8
runs the same EP 64 inside each stage, so the difference is PP's alone. As we
saw, ZeRO-1 + EP 64 keeps 156 GB of compute weights and 397 GB of unsharded
gradients alive for the whole step, over the GPU in total. Now look at the PP
rows:

- The master weights and optimizer state stay roughly the same, 1/512 of the
  model each.
- But a GPU now keeps compute weights and gradients only for its stage's
  layers. At PP 8, that is 12 of the 93 layers: 21 GB of compute weights and 53
  GB of gradients, about 7.5x less.

::: tree-step pp snapshot

::: figure pp-weight-table

Time for a nugget:

> NUGGET: PP makes ZeRO-1 affordable. The persistent shards stay at 1/512, but everything ZeRO-1 keeps unsharded for the whole step, the compute weights and the gradients, shrinks to one stage's worth. That is how PP avoids FSDP's per-microbatch communication, and why TorchTitan's FSDP defaults to ZeRO-1 under PP.

That wraps up the weights. We have a few candidates that fit in memory; now our
job is to fit the activations in the budget each one leaves.

::: tree-end

## How big are the activations?

Let's set up the problem for activations. First, a reminder of what they are:
the tensors the forward pass stashes for the backward pass to use later. With a microbatch of one sequence, PyTorch
stashes **746 GB at 8K** (and 3.0 TB at 32K), far more than a GPU's 288 GB.
Where does it go, at 8K?

| Stashed by | GB | Share |
|---|---|---|
| Routed experts | 420 | 56% |
| Attention (KDA and MLA) | 244 | 33% |
| Shared experts, norms, head, loss, layer 0 | 57 | 8% |
| Attention residuals | 25 | 3% |
| **Total** | **746** |  |

More than half of it is the routed experts. Each token goes to 16 experts, so
each of its expert intermediates is stashed 16 times over.

But this assumes no special kernels for the MoE layers. In practice, MoE
layers run on custom kernels, in our case
[DistMoE](https://github.com/meta-pytorch/dist_moe). DistMoE stashes the MoE
intermediates in one activation buffer shared by the whole model, whose size we
set at model init. At runtime, each MoE layer checks how much room is left in
the buffer. If there is enough, it stashes its inputs and its intermediates. If
not, it stashes only its inputs and recomputes the rest in backward. So with the
smallest buffer every MoE layer recomputes, and with a big enough one every
layer stashes. In between it is a dial: size the buffer at 30% of that range,
and about 30% of the MoE layers stash while the rest recompute. Conceptually
this is rematerialization, built into the kernel and decided at runtime,
depending on how unevenly the tokens are routed.

Besides the intermediates, three DistMoE allocations always count toward the
peak:

- **Its inputs**, because that is what DistMoE recomputes from.
- **Its output.** DistMoE manages its own recompute, so when the rest of a
  layer is rematerialized in backward (full remat, next section), DistMoE is
  not run again. The ops after it need its output to carry on the recompute,
  so the output stays stashed.
- **Its scratch**, a workspace for temporary tensors, reused by every layer.
  With uneven routing these temporaries can get big, so the scratch is sized
  for an imbalance of 4.

Let's look at the total activations for three buffer sizes, written as the
share of that range. We use this notation throughout the post:

- **+100%** (big enough for balanced routing): every MoE layer stashes, nothing
  is recomputed. 683 GB.
- **+30%**: 30% of the MoE layers stash, the rest recompute. 463 GB.
- **+0%** (the smallest buffer, today's default): every MoE layer recomputes,
  our counterpart to full remat for the routed experts, at the cost of a
  slightly longer backward. 368 GB.

::: figure activations-distmoe

Even at +100%, DistMoE stashes less than PyTorch's 746 GB, scratch included;
at +0% the intermediates are gone entirely.

> NOTE: The meta point here is that a roofline must account for the implementation details of the components that dominate it. DistMoE changes both the memory footprint and the latency of the MoE layers, the most expensive and most dynamic part of the model. Leave it out early on, and your roofline can be way off.

That is our problem statement: at 8K, fit up to 683 GB of activations into
whatever budget each parallelism config leaves.

## Use the leftover memory for activations effectively

Here is the general recipe I followed:

1. **Start simple:** no remat, no offload, one sequence per microbatch.
2. **If it fits** (rare at this size), grow the microbatch while it still fits:
   under FSDP, fewer, larger microbatches mean fewer rounds of gathers and
   reduce-scatters.
3. **If it does not fit** (our case: one 8K sequence is already more than a
   GPU), use two levers: rematerialization and offloading, applied greedily
   ([[memory.4.1]]), and context or tensor parallelism, which split each
   sequence ([[memory.4.2]]). For each CP or TP degree, find its best remat and
   offload setting, and let the ranking pick.

Let's look at each lever in more detail.

### Rematerialization and offloading

There are two ways to stash less, and both trade something for memory:

- **Rematerialization** (with [torch_remat](https://github.com/meta-pytorch/remat))
  does not stash a region's tensors; it recomputes them in backward, at the
  cost of the recompute time. DistMoE's buffer size from 3.3 also sits here.
- **Offloading** stashes them in CPU memory instead: forward copies them out,
  backward copies them back, over the link between the GPU and its CPU (on a
  GB300, the C2C link to the Grace CPU). It costs bandwidth, and it is free as
  long as the copies hide behind compute.

This is a large search space, so we use a greedy approach. We did not put
serious effort into it, so it can very likely be improved. The idea is to rank
the steps by their overhead (recompute or offload time) per byte freed, and keep
applying the cheapest one until the activations fit under the budget set by the
parallelism config. The steps are discrete and can jump a lot, so the last one
may overshoot and free far more memory than needed. We can then undo some
earlier steps and win back some of their recompute or offload cost. Again, this
is a very simple recipe, and you can certainly come up with better ones.

Let's look at three different paths that fit FSDP + EP 64 at 8K into its
budget. The blue one is our greedy approach, and the other two show alternatives
that end up slower. To read the chart, start at the bottom right, with no remat
and no offload. Each line then applies its own steps in order.

::: figure remat-offload-frontier

- **Remat only.** Let DistMoE recompute, then recompute attention. It lands
  right on the budget line.
- **Greedy.** Take the step that frees the most memory for the least time:
  offload the attention projections and the feed-forward, as the copies hide
  behind compute; then let DistMoE recompute, and finally recompute attention.
  This overshoots, leaving a lot of memory unused, so we buy back some speed:
  we grow DistMoE's activation buffer, so that some layers keep their
  intermediates instead of recomputing them.
- **Full remat.** One step. It gets under the budget, but it is very slow.

A general insight that was new to me from this experiment: offloading is very
powerful. If you are not using the C2C link for anything else, it removes a
good part of the memory footprint for free.

> NUGGET: Offloading is free as long as it hides behind compute, and that free budget is best spent on the bytes most expensive to recompute.

Now look at the same config at 32K context (use the toggle at the top right).
To fit the budget, we have to go all the way to full remat, and offload the
block inputs on top, and full remat is slow. Is there another lever? Yes:
context and tensor parallelism.

### Shrinking activations with Context and Tensor Parallelism

We saw earlier how TP shards the dense weights. Let's now look at what it does
to the activations. In pretraining, TP usually comes with *sequence
parallelism* (SP): the TP ranks share one microbatch, each keeps a shard of it
for the ops outside the TP GEMMs (norms, residual adds), and an all-gather or
reduce-scatter sits around each TP GEMM. So TP also cuts the activation memory,
roughly by the same degree.

*Context parallelism* (CP), on the other hand, targets long sequences. It
shards each sequence across the CP ranks, and adds a mechanism so that
attention still sees the whole sequence, such as an all-to-all (Ulysses) or a
ring of all-gathers for dense attention. We use Ulysses for MLA, and for KDA we
pass only its small recurrent state between ranks. So CP also cuts the
activation memory by its degree, inputs included, but it leaves the weights
untouched.

I admit this is a very short explanation of CP and TP. Here, the one takeaway
is that both cut the activations by about their degree. For the details, read
the [Ultra-Scale
Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) and the
background on linear attention.

At this point, it is also worth building a mental model of which one should be
faster. The answer: it depends.

- **What favors TP.** If FSDP's communication holds your config back, and EP
  has made the dense weights a big part of what FSDP moves (the flip we saw in
  [[memory.2.2]]), TP shrinks every all-gather and reduce-scatter to its shard, and the memory it
  frees can buy back some remat.
- **What it costs.** TP splits every GEMM, which runs less efficiently when
  split too far, and it communicates around every dense layer. CP communicates
  only around attention, and in the KDA layers just a small recurrent state.

So CP is an attractive choice for long contexts with linear attention, unless
FSDP's communication is what holds you back. And if you have been following
along, you guessed it: nugget time.

> NUGGET: With linear attention, CP communicates far less than TP. A linear-attention layer like KDA hands only a small recurrent state to the other CP ranks, however long the sequence, while TP communicates activations around every dense layer.

We follow the same approach for every config: go to reasonable CP and TP
degrees, and run the greedy search to fit each one into its activation budget.

### Pipeline Schedules and the Activation Footprint

As a reminder, we brought in PP because it makes ZeRO-1 affordable, which
saves FSDP's per-microbatch communication. The question now is how it affects
the activation footprint. It depends on the pipeline schedule.

Of the many PP schedules, this study keeps two, 1F1B and Interleaved 1F1B, and
for each we care about its bubble and its activation footprint.

**1F1B.** Each stage holds 1/P of the layers, but up to P microbatches are
alive on the first stage, so it holds about one whole sequence's activations,
roughly the same as without PP. But 1F1B suffers from pipeline bubbles: while
the pipeline fills at the start of a step and drains at the end, ranks sit
idle.

**Interleaved 1F1B.** To shrink 1F1B's bubble, Interleaved 1F1B cuts the
model into `v` times more, smaller chunks and deals them to the ranks round
robin. With PP 8 and `v = 2`, rank 0 runs layers 0-5 and 48-53, rank 1 layers
6-11 and 54-59, and so on. Each rank stores the same weights, just in `v`
pieces, and the bubble shrinks by about `v` (we come back to this with the step
time). The price is memory: to keep every rank busy, the first rank starts more
microbatches before its first backward, so more activations are alive at once.
A larger `v` gives some of that back, because with smaller chunks, the
microbatches in flight are released sooner.

On the first rank, with attention recomputed at 8K:

::: figure pp-schedule-acts

> NOTE: There is a fascinating world of pipeline schedules out there, and I hope to learn it someday and extend this study to more of them. For now, this post is complex enough, so we stick with these two.

## What Parallelism Configs Fit

With all the pieces in place, here are the parallelism configs that fit our
budget. Each config walks the greedy path ([[memory.4.1]]) from left to right and
stops at the first state that fits (outlined). If that leaves memory to spare,
the spare memory buys back part of DistMoE's buffer, shown by the arrow in the
green cell. The buy-back sets how many MoE layers stop recomputing: "+30%"
means that instead of all of them recomputing, 30% now stash their
intermediates and 70% still recompute. "&ndash;": not needed after the pick. The
tables show 8K; use the toggle at the top right for 32K.

::: tree-step acts snapshot

**FSDP + EP 64.** The table is mostly self-explanatory. The one row to point
out is the shaded one, with 2 sequences per microbatch: half as many
microbatches means half as many rounds of FSDP's gathers and reduce-scatters, as
discussed earlier.

::: figure fit-configs

*"~": estimated; "host full": the offload would not fit in the node's CPU
memory, so the path skips it; "OOM": that setting's measured run ran out of
memory.*

Note that CP and TP let a config get away with less remat and offload. Whether
that pays off, or the extra FSDP communication from more microbatches outweighs
it, is for the rankings to tell.

**ZeRO-1 + EP 128 + TP**, the 3.2.3 configs without FSDP. TorchTitan cannot
run ZeRO-1 without PP, so these rows are modeled. TP 4 leaves so little memory
for activations that it does not fit at all.

::: figure fit-configs-zero1

**ZeRO-1 + PP with 1F1B**, grouped by PP degree. PP 4 runs with real pipeline stages on four GPUs;
PP 8 and 16 use a PP model that matches the measured PP 4 to 0.2 GiB.

::: figure fit-configs-pp

**ZeRO-1 + PP with Interleaved 1F1B**, for the PP configs that fit with 1F1B:
the same weights per GPU, but the first rank holds more of each microbatch in
flight ([[memory.4.3]]), modeled the same way.

::: figure fit-configs-interleaved

That brings us to the end of this section. We now have every config that fits
in memory, each with the remat and offload setting its greedy path picked to
trade memory for speed. The [[ranking]] puts them in order.

::: tree-end


## Notes
