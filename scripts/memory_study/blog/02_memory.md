# The Memory Problem

Kimi K3 has 2.78T parameters. *Stored in fp32, the weights alone are 11.1 TB,
58 times one GPU's 192 GB.* But weights are only part of what lives in GPU memory
during training, and not the largest part. Let's look at what else is there.

## Persistent State

Training needs room for three things per parameter for the whole run: the fp32
master weight, its fp32 gradient, and the optimizer state (Muon keeps one fp32
momentum buffer). That is 12 bytes per parameter:

|  | Bytes per parameter | Kimi K3 |
|---|---|---|
| Master weights (fp32) | 4 | 11.1 TB |
| Gradients (fp32) | 4 | 11.1 TB |
| Optimizer state (Muon, fp32) | 4 | 11.1 TB |
| **Total** | **12** | **33.4 TB** |

*33.4 TB is 174 times one GPU's 192 GB, before a single activation is stored.*

We ignore the AdamW optimizer state, since the parameters it covers are only
0.09% of the model.

This is a deliberately simple mental model: no parallelism yet, and none of its
implementation details. Once FSDP comes in, for example, each GPU also keeps
temporary bf16 copies of the weights it gathers for compute. We add those
details as each parallelism is introduced later.

## Stashed Activations

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
expert intermediates is stashed 16 times over. That is what rematerialization
goes after next.

This is deliberately simple again. Tensor parallelism (with sequence
parallelism) and context parallelism split each sequence across GPUs and shrink
what each one stashes; we bring them in when we get there.

The attention residuals run on TorchTitan's fused Triton kernel (an opt-in
override adapted from Flash Linear Attention). It reads the block stack and the
current partial sum directly, so it stashes no concatenated copies of them.

::: figure memory-by-kind

*Log2 axis: each gridline is a factor of 2, so halving something (say, sharding
it over 2 GPUs) shortens its bar by exactly one gridline. The dashed line is one
GPU. Both charts in this section share the axis.*

The bars are far past the one-GPU line. Even with nothing else in memory, one 8K
sequence's stashed activations, 746 GB, are 3.9 times one GPU. Before tackling
the persistent state, the next three subsections deal with activations:
rematerialization, which shrinks them; DistMoE, which does not shrink them but
is how we run the routed experts, so its own memory has to be part of the
picture; and MXFP8, which shrinks matmul operands to 8 bits.

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
  case across the expert-parallel group, so with EP 64 they grow 56x, to 16 GB
  ([[sharding]]).

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
tensor parallelism ([[sharding.3]], [[sharding.4]]). The persistent state has
not moved: 33.4 TB, 174 times one GPU. No single-GPU trick fixes that; it has
to be split across many GPUs.

## Notes
