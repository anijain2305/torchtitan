# Using parallelism to fit the model

The goal of this section is to find the parallelism configs that fit Kimi K3 in
memory. How fast each of them trains comes later. Along the way we want to build
an intuition for how the different kinds of parallelism interact: what each one
shards, what it leaves untouched, and what it costs in extra memory.

## ZeRO / FSDP and expert parallelism

We start with plain data parallelism over all 512 GPUs. 2,048 sequences per step
(at 8K) gives each GPU 4 sequences, run one at a time as 4 microbatches with
gradient accumulation. Every GPU runs the whole model, so on its own data
parallelism only multiplies the problem; ZeRO is what shards the persistent
state across those 512 copies, in three stages:

- **ZeRO-1** shards the master weights and optimizer state: each GPU keeps
  1/512 of them, 22.2 TB down to 43.4 GB.
- **ZeRO-2** also shards the gradients: 11.1 TB down to 21.7 GB.
- **ZeRO-3** also shards the weights used for compute. This is what PyTorch's
  FSDP does by default without pipeline parallelism, so we use "ZeRO-3" and
  "FSDP" interchangeably.

Sharding the master weights has a catch. A GPU holding 1/512 of the fp32 master
weights cannot run a forward pass on them, so it needs a separate copy of the
weights in the format the matmuls use: the **compute weights**: MXFP8 for the routed experts, about 1 byte per parameter, and bf16
for everything else. Under ZeRO-1 and ZeRO-2 every GPU holds the whole model's
compute weights: 2.84 TB, now the largest thing left.

ZeRO-3 shards the compute weights too. Before a layer runs, FSDP all-gathers its
weights (in bf16), converts them to MXFP8 and releases the bf16 copy; after the
layer, it frees them. In backward it also needs that layer's full fp32 gradient
before reduce-scattering it. To hide the communication, FSDP works one layer
ahead: while one layer computes, the next layer's weights are already being
gathered, and in backward the previous layer's gradient is still being
reduce-scattered. So at the peak, in backward, a GPU holds **two layers' compute
weights and two layers' gradients** instead of the whole model's.

For Kimi K3 even one layer is a lot: it holds all 896 experts, 30.2B parameters,
so one layer's fp32 gradient alone is 121 GB, and two are 242 GB. **Expert parallelism** (EP) fixes it. With
EP 64, each GPU owns 14 of the 896 experts of every layer, and the remaining
8-way replication of those experts is sharded by FSDP. The shard each GPU stores
is the same 1/512, but it only ever gathers its own 14 experts: 1.1B parameters
per layer, so two layers in flight need 2.2 GB of compute weights and 8.8 GB of
gradients.

::: figure zero-table

::: figure sharding-steps

At 8K, FSDP + EP 64 needs 125 GB per GPU and fits. At 32K the activations alone
are 204 GB, so it does not: we need to split each sequence too, which is where
context and tensor parallelism come in.

## Context parallelism

Context parallelism (CP) splits each sequence across GPUs: with CP 2, each GPU
holds half of a sequence's tokens. Everything that works token by token -- the
projections, the norms, the MoE layers -- runs on the local tokens with no extra
communication. Only attention needs the other GPUs' tokens: MLA uses Ulysses,
and KDA, being a recurrence, passes a small summary of each chunk's state along.
What that costs in time comes later; here we only care about memory.

For memory, CP is exactly the lever the previous step was missing. Activations
scale with tokens per GPU, so CP divides them, DistMoE's stash, output and
scratch included. Nothing else changes. The CP ranks join the FSDP group, so
each GPU still stores 1/512 of the persistent state, and the layers in flight
are the same weights and gradients whatever the token count. The work per GPU
does not change either: each GPU now runs CP times as many microbatches, each
1/CP of a sequence.

::: figure cp-table

At 32K, CP 2 halves the activations to 102 GB and brings FSDP + EP 64 to
178 GB: it fits, though without much room to spare. CP 4 gives a comfortable
127 GB, the same as 8K without CP, since each GPU again holds 8,192 tokens per
microbatch.

> INSIGHT: CP shrinks only the activations, because they are the only part of
> GPU memory that grows with sequence length. Weights, gradients and optimizer
> state stay the same, and so does each GPU's work per step: CP just cuts it
> into more, smaller microbatches.

## Tensor parallelism

Tensor parallelism (TP), which we always pair with sequence parallelism, is the
other lever for activations: like CP, it splits the tokens across GPUs. It also
splits the dense weights (not the experts). Under FSDP over 512 GPUs, those are
already tiny per GPU, so the extra saving is small.

::: figure tp-table

So under FSDP, TP and CP are interchangeable for memory: TP 2 and CP 2 differ by
3 GB. TP's weight splitting pays off with pipeline parallelism, which keeps a
stage's weights gathered all step.

Memory makes TP look like a free win; speed may not. CP splits along the
sequence, so it scales with attention, which grows with context length. TP
splits the dense projections themselves, which are already small, and narrower
matmuls run less efficiently.

## Pipeline parallelism

FSDP + EP fits, so why look further? Because memory is not the only thing that
matters. Without PP, FSDP gathers and reduce-scatters every layer, every
microbatch, across all 512 GPUs, and those rings cross the slow InfiniBand
links between NVLink domains. That traffic may not hide behind compute. Even
when it does, the all-gathers and reduce-scatters run on the GPU's SMs and slow
down the compute they overlap with. We model both in the performance roofline
later. For now, let's see how the model shards with PP, and whether it fits.

Pipeline parallelism (PP) cuts the model into stages of consecutive layers and
gives each stage its own group of GPUs: with PP 8, eight stages of 12 layers,
each on 64 GPUs (one NVLink domain); with PP 16, sixteen stages of 6 layers on
32 GPUs. Inside a stage we use FSDP and EP as before, over the stage's GPUs,
so EP becomes 64 or 32. Microbatches flow from stage to stage.

What PP does to memory:

- **Each GPU only stores its stage.** But 1/P of the layers, sharded over 512/P
  GPUs, is still about 1/512 of the model: *same memory as before*.
- **Microbatches pile up on the first stage.** With the 1F1B schedule, the
  first stage holds P microbatches in flight, each stashing activations for its
  93/P layers. P microbatches x 93/P layers is about one whole-model
  microbatch: *same memory as before*. This depends on the schedule, though;
  other schedules keep different numbers of microbatches in flight. We stick to
  1F1B for now.
- **ZeRO-1 becomes affordable.** Without PP, ZeRO-1 meant every GPU holding the
  whole model's compute weights and gradients, 14 TB. With PP it only has to
  hold one stage's, and that fits. This is FSDP's default under PP: a stage's
  compute weights stay gathered and its gradients unsharded for the whole step,
  instead of being regathered and reduce-scattered for every microbatch. That
  is where the performance win can come from, at the cost of some memory.

Here are PP 8 and PP 16 under ZeRO-1, with each sequence split by
CP or TP (master weights and optimizer state are 45 GB per GPU in every row):

::: figure pp-table

At 8K everything fits. At 32K, ZeRO-1's unsharded stage (66 GB with PP 8)
leaves less room for activations: PP 8 needs CP 4 or TP 4, while PP 16 fits
with TP 2. And TP finally pays off: the stage's dense weights and gradients now
stay unsharded, 38 of the 66 GB ZeRO-1 adds with PP 8 (30 GB of gradients,
7.6 GB of compute weights). TP splits exactly that part, so TP 2 halves it and
beats CP 2 by 19 GB: 117 against 137 GB at 8K.

## What fits

Putting it all together, every config in this section (green: fits in 192 GB,
red: does not):

::: figure fit-map

All 15 fit at 8K, and 9 of the 15 at 32K. Memory alone does not pick a winner:
next we build a performance model to rank them.


## Notes
