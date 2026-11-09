# Using Parallelism to Fit the Model

Let's now spread the model across the cluster. The goal of this section is to
find the parallelism configs that fit Kimi K3 in memory; how fast each one
trains (its step time) comes later. Along the way we build an intuition for how
the different kinds of parallelism interact: what each one shards, what it
leaves untouched, and what it costs in extra memory. We also bring in the
implementation details of PyTorch's sharding techniques that affect memory, so
the numbers get closer to what a real run would use.

## ZeRO / FSDP

We start with plain data parallelism over all 512 GPUs. 2,048 sequences per step
(at 8K) gives each GPU 4 sequences, run one at a time as 4 microbatches with
gradient accumulation. Every GPU runs the whole model, so on its own data
parallelism only multiplies the problem; ZeRO is what shards the persistent
state across those 512 copies, in three stages:

- **ZeRO-1** shards the master weights and optimizer state: each GPU keeps
  1/512 of them, 22.2 TB down to 49.5 GB (including some padding, below).
- **ZeRO-2** also shards the gradients: 11.1 TB down to 24.8 GB.
- **ZeRO-3** also shards the weights used for compute. This is what PyTorch's
  FSDP does by default without pipeline parallelism, so we use "ZeRO-3" and
  "FSDP" interchangeably.

**Compute weights.** Sharding the master weights has a catch. A GPU holding 1/512
of the fp32 master weights cannot run a forward pass on them, so it needs a
separate copy of the weights in the format the matmuls use: the **compute
weights**: MXFP8 for the routed experts, about 1 byte per parameter, and bf16
for everything else. (FSDP gathers the experts in bf16 and frees that buffer
once their MXFP8 copy is built; the fake runs in [[sharding.8]] agree.) Under
ZeRO-1 and ZeRO-2 every GPU holds the whole model's compute weights: 2.84 TB,
which after ZeRO-2 is the largest thing left.

Let's see what each step does to the memory per GPU (green: less than the row
above, red: more), then walk through the rows.

::: figure zero-table

**The ZeRO-1 and ZeRO-2 rows** are straightforward: the master weights and
optimizer state (ZeRO-1), then the gradients (ZeRO-2), drop to their 1/512
shards, but every GPU now has to hold the whole model's compute weights.

**The ZeRO-3 row.** ZeRO-3 shards the compute weights too: FSDP gathers a
layer's weights just before it runs and frees them afterwards. Yet the
gradients column jumps, which is counterintuitive: ZeRO-3 only adds sharding of
the compute weights, so the gradients should stay where they were. The reason is
a second catch. To hide the communication, FSDP works one layer ahead: while one
layer computes, the next layer's weights are already being gathered, and in
backward the previous layer's full gradient is still being reduce-scattered. So
at the peak, in backward, a GPU holds **two layers' compute weights and two
layers' full gradients** on top of its shards, plus remat's recompute working
set for one layer ([[memory.3]]). Because the model is so large, two layers'
worth of gradients is 242 GB on its own.

**Padding.** FSDP splits each tensor along its first dimension, and pads it when
that does not divide evenly. For the expert weights that dimension is the expert
count, 896 over 512 GPUs, which pads to 1,024: 14% more for the expert part of
every shard.

## Expert Parallelism

Under ZeRO-3, holding two unsharded layers is a lot: 242 GB of gradients and
61.8 GB of compute weights, more than a whole GPU on their own, and almost all
of it is experts. We need a way to shard what a GPU gathers, not just what it
stores. Enter **expert parallelism** (EP).

EP is counterintuitive at first. With EP 64, the 896 experts of every layer
split into 64 groups of 14, and each group lives on 8 GPUs: FSDP shards the
group's 14 experts over those 8 along the expert dimension, so each GPU stores
two whole experts (14 pads to 16; the eighth GPU stores none). So each GPU
still stores about 1/512 of the weights, as before. **The difference is
in what it gathers.** Without EP, the two layers in flight hold the unsharded
gradients and compute weights of all 896 experts; with EP, only those of the
GPU's group of 14. The expert part
of those per-layer buffers shrinks 64x, taking the in-flight gradients from
242 GB to 8.8 GB and the compute weights from 61.8 GB to 3.5 GB (the dense part
of each layer, in bf16, does not shrink). The padding stays: each group's 14
experts are sharded over 8 GPUs, which pads to 16.

**Caveat: DistMoE's routing indices.** For backward, DistMoE stashes three
int64 index arrays per MoE layer, sized for the worst case: EP x tokens x 14
entries. With EP 64 at 8K that is 16.2 GB over 92 layers, which is why the EP
row's activations jump from 56 to 72 GB in [[sharding.1]]'s table. (We are
working with the DistMoE authors to shrink this.)

> INSIGHT: Sharding what a GPU stores does not shard what it holds at the peak. FSDP stores 1/512 of everything, yet the layers in flight are unsharded: two layers' compute weights and gradients, hundreds of GB for a large MoE layer. EP does not change what each GPU stores; it shrinks what each GPU gathers, from every expert of a layer to its own.

::: figure sharding-steps

**Finally, with FSDP over 512 GPUs plus EP 64, the model fits at 8K: 158 GB per
GPU.** At 32K the activations are 223 GB even before DistMoE's routing indices,
which grow to 65 GB with EP 64, so it is far from fitting. Spoiler: we will need context or tensor
parallelism.

## Context Parallelism

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

## Tensor Parallelism

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

## Pipeline Parallelism

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

## Interleaved Pipeline Parallelism

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
at every `v` ([[sharding.7]]).

> NOTE: The attention residuals' blocks do not add to this. With interleaving, TorchTitan's cache transport ([[step.2]]) keeps on each rank the blocks a microbatch has already brought there, so a hop carries only the new ones. Those cached blocks are the same tensors the stash already counts.

## What Fits

Putting it all together, every config in this section (green: fits in 192 GB,
red: does not):

::: figure fit-map

32 of the 35 fit at 8K, and 11 of the 35 at 32K. Interleaving never changes
the answer: what fits with 1F1B fits interleaved, and what does not, does not.

## How Close Are We to a Real Run?

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
