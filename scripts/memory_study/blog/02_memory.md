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
comparing plain ZeRO-1 from 3.2.1 with ZeRO-1 + PP on its first stage. As we
saw, plain ZeRO-1 keeps 2.84 TB of compute weights and 11.1 TB of unsharded
gradients alive for the whole step. Now look at the PP rows:

- The master weights and optimizer state stay roughly the same, 1/512 of the
  model each.
- But a GPU now keeps compute weights and gradients only for its stage. PP
  divides them by the number of stages, and EP inside the stage divides the
  experts further. At PP 8, that is 21 GB of compute weights and 53 GB of
  gradients.

::: tree-step pp snapshot

::: figure pp-weight-table

Time for a nugget:

> NUGGET: PP makes ZeRO-1 affordable. The persistent shards stay at 1/512, but everything ZeRO-1 keeps unsharded for the whole step, the compute weights and the gradients, shrinks to one stage's worth. That is how PP avoids FSDP's per-microbatch communication, and why TorchTitan's FSDP defaults to ZeRO-1 under PP.

Every PP degree fits. A higher degree, though, means more stages, and so a
bigger pipeline bubble and more hops between them; we will see what that costs
in the rankings. An attentive reader might wonder why we have not discussed
pipeline schedules. For the widely used schedules, the weight footprint is the
same; only the activation footprint changes. So PP comes back when we count the
activations.

That wraps up the weights. Three families fit: FSDP + EP, and two ways of affording ZeRO-1: ZeRO-1 + EP + TP and ZeRO-1 + PP. The tree that has been growing
alongside the text tracks which configs fit so far, and how much of each GPU
their weights take. Whatever is left is each config's budget for activations, and
that is where we go next.

::: tree-end

## How big are the activations?

Let's set up the problem for activations, as we did for the persistent state.
First, a reminder of what they are: the tensors the forward pass stashes for
the backward pass to use later. With a microbatch of one sequence, PyTorch
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
[DistMoE](https://github.com/meta-pytorch/dist_moe), and DistMoE manages their
activations itself. Kernels like it handle MoE activations separately because
the number of tokens each GPU receives varies from step to step. DistMoE
stashes the expert intermediates in its own activation buffer, whose size we
choose when we set up the model. At runtime it checks how many tokens arrived,
and the rows that do not fit in the buffer are not stashed but recomputed in
backward. Conceptually this is rematerialization, but built into the kernels
and decided at runtime rather than ahead of time.

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

To see its impact, take its two extremes, with nothing else in the model
rematerialized:

- **DistMoE keeps everything** (buffers big enough for balanced routing): no
  recompute.
- **DistMoE recomputes everything** (no buffers, today's default): our
  counterpart to full remat, for the routed experts. Its backward takes a bit
  longer.

Let's look at the numbers:

::: figure activations-distmoe

The naive PyTorch stash is 746 GB at 8K. DistMoE keeping everything already
brings it down to 683 GB, scratch included. DistMoE recomputing everything
removes the intermediates entirely, leaving 368 GB.

Its inputs and output are small, and so is the scratch at 8K. At 32K, though,
the scratch grows to about a third of a GPU. Once you know how uneven the
routing really is in a training run, you can size the scratch for a smaller
imbalance than 4; for now we keep 4.

> NOTE: The meta point here is that a roofline must account for the implementation details of the components that dominate it. DistMoE changes both the memory footprint and the latency of the MoE layers, the most expensive and most dynamic part of the model. Leave it out early on, and your roofline can be way off.

That sets up the problem. With no recompute at all, one 8K sequence needs
683 GB of activations, far more than what is left of a GPU's 288 GB once the
weights' persistent state and working set are in place. Next, let's see how to
shrink the activations to fit.

## Use the leftover memory for activations effectively

Here is the general recipe I followed:

1. **Start simple:** no rematerialization, no offloading, one sequence per
   microbatch. Check whether it fits the budget.
2. **If it fits** (rare for LLMs of this size), increase the microbatch size as
   long as it still fits, up to the GPU's share of the local batch size. This may seem
   confusing now, but it pays off under FSDP: FSDP
   gathers the weights and reduce-scatters the gradients for every microbatch,
   so fewer, larger microbatches mean less communication.
3. **If it does not fit** (our case: a single 8K sequence is already more than a
   GPU), there are two levers:
    - **Rematerialization and offloading.** There are many ways to get under
      the budget ([[memory.4.1]] shows a few). We apply a greedy approach for now:
      incrementally apply these optimizations in the order that frees the most memory for the least time,
      starting with offloads the GPU can hide behind compute, until the config
      fits. If the last step overshoots by a lot, undo some of the steps you
      took, buying back recompute or offload time with the memory you do not
      need.
    - **Context and tensor parallelism.** They split
      every sequence across their GPUs and shrink the activations roughly by
      their degree, at the cost of communication on the critical path (and,
      under FSDP, more microbatches per GPU, so more FSDP communication). A
      high enough degree can remove the need for rematerialization altogether:
      CP 1 with full remat versus CP 4 with none is a real trade. So for each
      CP or TP degree, find its best remat and offload setting as above, and
      let the ranking pick.


The rest of this section follows the recipe: the remat and offload frontier,
then context and tensor parallelism. At the last, we also look at how pipeline
parallelism schedules affects activation footprint.

### Rematerialization and offloading

There are two ways to stash less, and both trade something for memory:

- **Rematerialization** (with [torch_remat](https://github.com/meta-pytorch/remat))
  does not stash a region's tensors; backward recomputes them. We pay recompute overhead. DistMoE's own config from 3.3 also sits here: the size of its buffers.
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

*Filled points are measured at full depth, hollow ones estimated. 8K runs use
CUDA graphs; 32K runs (the toggle at the top right) are eager.*

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

> NUGGET: Offloading is free until the CPU link saturates, and that free budget is best spent on the bytes most expensive to recompute.

This experiment covers just one parallelism config. Every config needs its own
run of these steps, because each leaves a different budget for activations. But
part of the analysis carries over: the price of each step per byte freed. TP and
CP shrink the memory footprint and the forward time together, so these prices
stay roughly the same. We run this greedy approach for every config.


Now look at the same config at 32K context (use the toggle at the top right).
To fit the budget, we have to go all the way to full remat, and offload the
block inputs on top. Is there another lever? Yes: context and tensor
parallelism. They cut the activation memory, so the remat can be less
aggressive, but they put communication on the critical path. The question
becomes which costs more: the remat or the collectives. Let's look Let's do 3 and then 1.at CP and TP
next.

### Shrinking activations with Context and Tensor Parallelism

We saw earlier how TP shards the dense weights. In pretraining, TP usually
comes with *sequence parallelism* (SP): the TP ranks share one microbatch, each
keeps a shard of it for the ops outside the TP GEMMs (norms, residual adds),
and an all-gather or reduce-scatter sits around each TP GEMM. So TP also cuts
the activation memory, roughly by its degree.

*Context parallelism* (CP) targets long sequences instead: it shards each
sequence across the CP ranks, and adds a mechanism so that attention still sees
the whole sequence, such as an all-to-all (Ulysses) or a ring of all-gathers
for dense attention. We use Ulysses for MLA, and for KDA we pass only its small
recurrent state between ranks. CP leaves the weights untouched; it only cuts
the activations.

If this was too quick, the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
covers TP, SP and CP in depth.

Both add collectives, and CP's are cheaper here. TP shrinks the GEMMs, which
lowers their throughput as the degree grows, and it communicates in every
layer. CP only communicates in the attention layers, and for KDA that is a tiny
state. So, to set expectations: CP will beat TP, even though both cut the
activations by about the same amount. Confirming it takes the rankings, but it
is worth keeping in mind now.



> NUGGET: With linear attention, CP scales more easily than TP. A linear-attention layer like KDA hands only a small recurrent state to the next CP rank, however long the sequence, while TP communicates activations in every layer and shrinks every GEMM.

We follow the same approach for every config: go to reasonable CP and TP
degrees, and run the greedy search to fit each one into its activation budget.

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
at every `v` ([[memory.5]]).

> NOTE: The attention residuals' blocks do not add to this. With interleaving, TorchTitan's cache transport ([[step.2]]) keeps on each rank the blocks a microbatch has already brought there, so a hop carries only the new ones. Those cached blocks are the same tensors the stash already counts.

## What Parallelism Configs Fit

With all of that in place, here are the parallelism configs that fit our
budget:

Each config's greedy path ([[memory.4.1]]) runs left to right: the peak per GPU
of every state up to the first that fits one GPU's 288 GB, which is the config's
pick (outlined), and "not needed" after it. The pick shows its peak, then its
final peak once the memory left over buys back DistMoE's buffer ("+45% DistMoE
buffer": 45% of the layers keep their DistMoE intermediates). A row with no
outlined cell does not fit. The ranking uses each config's pick.

::: tree-step acts snapshot

**FSDP + EP 64.** The shaded row takes a larger microbatch, as the recipe's step
2 suggests: 2 sequences per microbatch halve the microbatches per step, and with
them the rounds of FSDP's gathers and reduce-scatters. It holds 16K tokens per
GPU with FSDP + EP 64's weights, as 32K with CP 2 does, so it borrows that
config's memory and greedy path. We only grow the microbatch where there is no
split to lower: under CP or TP, a microbatch of 2 sequences holds as many tokens
per GPU as half the degree, while still paying the split's communication.

::: figure fit-configs

**ZeRO-1 + PP with 1F1B.** PP 4 runs with real pipeline stages on four GPUs;
PP 8 and 16 use a PP model that matches the measured PP 4 to 0.2 GiB.

::: figure fit-configs-pp

**ZeRO-1 + PP with Interleaved 1F1B**, for the PP configs that fit with 1F1B:
the same weights per GPU, but the first rank holds more of each microbatch in
flight ([[memory.4.3]]), modeled the same way.

::: figure fit-configs-interleaved

*In all three tables: offload moves the attention projections and feed-forward
to CPU memory; "host full": it would not fit in a GB300 node's CPU memory, so
the path continues without it; "OOM": that setting's run ran out of memory.
Peaks are measured at full depth with a fake process group (CP runs report their
first step's peak); "~" marks estimates, anchored on the config's own measured
settings.*

What the table says:

- **Splitting the sequence buys you out of remat.** At 8K, CP 4 needs no remat
  at all and TP 4 only the free offload, while without a split the greedy path
  has to recompute attention as well.
- **At 32K, CP beats TP for memory too.** CP 4 fits by recomputing attention
  alone; TP 4 also needs the offload. Without a split, or with a split of 2,
  only full remat fits, and offload runs out of host memory.
- **PP does not save activations.** Its first stage holds about one whole
  sequence's activations in flight, so PP needs the same remat as FSDP without
  a split, on top of its stage's weights. At 32K, PP fits only with CP 4.

::: tree-end

<!--
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

-->

## Notes
