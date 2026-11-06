# The memory problem

<!-- Order: persistent state (unsharded) -> saved activations -> remat ->
DistMoE -> MXFP8 -> (next) sharding. All numbers per GPU, bf16 compute, one
microbatch of one sequence, before any sharding. Detailed version:
kimi_k3_params.html. -->

Kimi K3 has 2.78T parameters. Stored in bf16, the weights alone are 5.6 TB,
29 times one GPU's 192 GB. But weights are only part of what lives in GPU memory
during training, and not the largest part. Let's look at what else is there.

## Persistent state

Training needs room for three things per parameter for the whole run: the fp32
master weight, its fp32 gradient, and the optimizer state (Muon keeps one fp32
momentum buffer). That is 12 bytes per parameter:

| | Bytes per parameter | Kimi K3 |
|---|---|---|
| Master weights (fp32) | 4 | 11.1 TB |
| Gradients (fp32) | 4 | 11.1 TB |
| Optimizer state (Muon, fp32) | 4 | 11.1 TB |
| **Total** | **12** | **33.4 TB** |

33.4 TB is 174 times one GPU's 192 GB, before a single activation is stored.
We ignore the AdamW optimizer state, since the parameters it covers are only
0.09% of the model.

This is a deliberately simple mental model: no parallelism yet, and none of its
implementation details. Once FSDP comes in, for example, each GPU also keeps
temporary bf16 copies of the weights it gathers for compute. We add those
details as each parallelism is introduced.

> RULE: Persistent state is about 12-16 bytes per parameter with fp32 master
> weights (12 with Muon, 16 with AdamW). Dividing it by one GPU's memory gives
> the minimum number of GPUs the model needs just to exist: 174 for Kimi K3.

## Saved activations

The forward pass saves tensors for the backward pass. Unlike the persistent
state, this scales with tokens, not parameters. Again we keep it simple: a
microbatch is one sequence (8,192 tokens at 8K, 32,768 at 32K), held entirely on
one GPU. One 8K sequence through all 93 layers saves **839 GB**, and a 32K
sequence four times that, 3.4 TB. Later, context parallelism and tensor
parallelism split each sequence across GPUs and shrink this per GPU; we add
them when we get there.

::: figure memory-by-kind

*Log2 axis: each gridline is a factor of 2, so halving something (say, sharding
it over 2 GPUs) shortens its bar by exactly one gridline. The dashed line is one
GPU. Both charts in this section share the axis.*

> RULE: Persistent state scales with parameters; activations scale with tokens
> per GPU. Sharding the parameters does nothing for activations -- only
> rematerialization, or splitting each sequence across GPUs, does.

## Rematerialization

Rematerialization (remat) saves less in the forward pass and recomputes the
rest during backward. With full remat each layer keeps only its input (plus the
routing decisions), and the activations of one 8K sequence drop from 839 GB to
**16.5 GB**, about 50x less. The price is one extra forward pass: 8 FLOPs per parameter per
token instead of 6, about 33% more compute.

## DistMoE

[DistMoE](https://github.com/meta-pytorch/dist_moe) is a recent library from
the PyTorch team for running MoE layers efficiently: its kernels overlap the
token dispatch and combine with the expert GEMMs, drop no tokens, and work with
CUDA graphs. Any production training system has something like it to manage
expert memory and routing imbalance, and its implementation changes both memory
and speed. So even though it pulls us away from the idealized picture, we bring
it in now.

Memory is tight, so we configure DistMoE to recompute too: per MoE layer it
stashes only the layer's input and recomputes the expert intermediates in
backward. On top of full remat, that adds four things:

- **Its stashed input**, held in DistMoE's own preallocated buffer.
- **Its output**, which autograd keeps because full remat never replays the
  DistMoE call (replaying it would repeat the token dispatch).
- **Scratch** for one layer's temporaries, reused by every layer and sized with
  a capacity factor of 4 so that experts receiving more than their share of
  tokens still fit.
- **Routing indices**: three int64 index arrays per MoE layer, stashed so the
  backward can send gradients back to the right tokens. They are sized for the
  worst case, every GPU in the expert-parallel group sending all its tokens to
  all of this GPU's experts, so they grow with expert parallelism. Without it
  they are small; with EP 64 they reach 16 GB at 8K ([[sharding]]).

For one sequence:

| | 8K | 32K |
|---|---|---|
| Full remat (chunked loss) | 16.5 GB | 66 GB |
| DistMoE stashed inputs | 5.4 GB | 21.6 GB |
| DistMoE outputs | 5.4 GB | 21.6 GB |
| DistMoE scratch (capacity factor 4) | 23.6 GB | 94.5 GB |
| DistMoE routing indices (no expert parallelism) | 0.3 GB | 1.2 GB |
| **Activations** | **51 GB** | **205 GB** |

> NOTE: DistMoE's scratch gets large at 32K: 94.5 GB, about half of one GPU's
> 192 GB on its own. It scales with the capacity factor, so it is a knob we can
> turn down once we know how imbalanced the routing really is. For now we keep
> the factor at 4.

The chart below puts these next to no remat and TorchTitan's default remat, which keeps every region the
model declares except the routed experts' first projection (w13), recomputing
only that.

::: figure memory-remat

> RULE: Fused MoE kernels hold memory autograd does not see: stashed inputs,
> outputs, and scratch sized by a capacity factor. The scratch can be the
> largest piece, so the capacity factor is a memory knob, not just a
> load-balancing one.

From here on, "activations" means full remat plus DistMoE: full remat's stashed
tensors, and DistMoE's stashed inputs, outputs, scratch and routing indices --
51 GB per 8K sequence and 205 GB per 32K sequence, before expert parallelism
grows the indices.

## MXFP8

MXFP8 runs the matmuls on 8-bit operands with a shared scale per 32 elements,
about 1 byte per element instead of 2. We use it for the routed experts only,
through DistMoE's own MXFP8 kernels; the dense linears stay in bf16. Memory
barely changes: with full remat, a layer stashes only
its bf16 input, so there is little left for MXFP8 to shrink. DistMoE still
stashes its input in bf16, and its scratch shrinks by only about 6% (23.6 to
22.2 GB at 8K), since most of its temporaries stay in bf16. The rest of the
post keeps the bf16 figure, 23.6 GB, a slightly conservative bound.

> RULE: Low-precision compute is a speed optimization, not a memory one, when
> master weights and optimizer state stay fp32 and activations are already
> rematerialized.

## Where we stand

::: figure memory-summary

Rematerialization brought the activations down to 58 GB per 8K sequence, well
under one GPU. **At 32K the problem is not solved**: 232 GB is still more than a
whole GPU, before anything else is stored. We come back to it with context and
tensor parallelism ([[sharding.2]], [[sharding.3]]). The persistent state has
not moved: 33.4 TB, 174 times one GPU. No single-GPU trick fixes that; it has
to be split across many GPUs.

## Notes
