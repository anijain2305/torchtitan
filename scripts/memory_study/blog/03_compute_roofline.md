# The compute roofline

Memory told us which configs fit. Now we want to know how fast each one trains.
We start with compute alone: how long one step takes with no communication and
nothing else in the way.

We build that estimate in two passes:

- **The ideal roofline** ([[compute_roofline.1]]) runs every op at a fixed
  fraction of the GPU's peak: one assumed efficiency for all GEMMs, full HBM
  bandwidth for everything else. It is quick to compute, the same for every
  parallelism config, and shows where the time goes.
- **The practical roofline** ([[compute_roofline.2]]) keeps the same ops but
  times each one at the efficiency we measure for the shapes each config
  produces. It sees what the ideal one cannot: how splitting the model shrinks
  the GEMMs and changes the work around them. This is the model the rest of the
  post builds on.

## The Ideal Roofline

Every GPU processes the same number of tokens per step, whatever the
parallelism: 16M tokens / 512 GPUs = **32,768 tokens**. With full remat, a
linear layer costs 8 FLOPs per parameter per token: 2 in forward, 2 when remat
reruns it, and 4 in backward.

| Symbol | Meaning | Kimi K3 |
|---|---|---|
| `T_gpu` | tokens per GPU per step | 32,768 |
| `N_dense` | dense linear parameters, summed over all 93 layers | 54.4B |
| `N_expert` | parameters of one routed expert (w13 + w2) | 33.0M |
| `V`, `D` | vocabulary size, model width | 163,840, 7,168 |
| `seq_len` | context length | 8K or 32K |
| `H` | MLA attention heads | 96 |
| `d_qk`, `d_v` | MLA query/key and value head dimensions | 192, 128 |
| `B_hbm` | HBM bandwidth | 8 TB/s |

Per GPU per step:

- **Dense linears**: `8 x T_gpu x N_dense`.
- **Routed experts**: `8 x T_gpu x 16 x N_expert x 92 layers` (16 experts per
  token).
- **LM head**: `6 x T_gpu x V x D` (not rematerialized).
- **MLA attention**: `4.5 x T_gpu x seq_len x H x (d_qk + d_v)` per layer
  (each token attends to half the sequence on average; forward + remat +
  backward is 4.5 forwards). The one term that grows with context.
- **KDA attention**: linear in tokens; its chunked kernels are not plain
  matmuls, so we use their measured time per token.
- **Everything else** (the attention-residual mix, MXFP8 quantization, FSDP's
  buffer copies, norms, residual adds): memory-bandwidth bound,
  `bytes moved / B_hbm`. Without PP, FSDP re-gathers, copies out and
  re-quantizes every layer's weights each microbatch; under PP's ZeRO-1 the
  stage stays gathered, so this happens once per step. [[methodology.4]]
  counts the bytes class by class.

The matmuls run at an assumed fraction of peak: **60% of 5 PFLOPS** in MXFP8
(linears, routed experts) and **70% of 2.5 PFLOPS** in bf16 (LM head,
attention).

::: figure roofline-table

**14.2 s per step at 8K and 14.9 s at 32K.** Two things stand out. The dense
linears cost more than the routed experts: experts are 98% of the parameters,
but a token touches about 54B dense parameters and only 49B expert ones. And
every GPU does the same FLOPs per step, so **the compute part is the same for
every config**: the ideal roofline cannot rank them.

## A Practical Roofline

To rank the configs, we refine the model with one practical choice: **we assume
the compute kernels are already well optimized**, and charge each op what such
a kernel actually takes at the config's own shapes, instead of one assumed
efficiency for everything. This study does not change the kernels, so we can
simply measure them. The model still leaves out communication and pipeline
bubbles, which come later.

All measurements run on one GB300, which has the same bf16 (2.5 PFLOPS) and FP8
(5 PFLOPS) peaks as GB200, so the percentages carry over. Kernels are timed as
CUDA-graph replays: kernel time, no CPU launch overhead.

### Compute ops at each config's shapes

Parallelism changes the shapes:

- **CP** cuts the tokens per microbatch, so the dense GEMMs get fewer rows.
- **TP** splits the dense weights: column-parallel projections lose output
  columns, row-parallel ones lose input columns. MLA's small low-rank
  projections, the router and the routed experts' down and up projections stay
  whole.
- **EP** sets how many experts each GPU owns (14 with EP 64, 28 with EP 32).
  With CP and TP setting the local tokens, that fixes the rows per expert.
- **PP** changes no shape itself, only EP (EP 32 with PP 16).

**Dense GEMMs** in MXFP8: forward, input-gradient and weight-gradient GEMMs of
every linear in a layer (green: at or above 60%; red: more than 5 points
below):

::: figure dense-shapes

**Routed experts**: we time each layer's experts as one MXFP8 grouped GEMM
(torchao's, which TorchTitan's MXFP8 experts use), w13 and w2, forward and
backward, counting only the GEMM kernels. This isolates the GEMMs from DistMoE,
which fuses them with token dispatch and combine. With balanced routing, each
expert's `M` is (rounded down to a multiple of 128, the MXFP8 scale tile):

```
M per expert = local_tokens x top_k / local_experts
             = local_tokens x top_k x EP / num_experts
```


::: figure expert-shapes

**MLA** on FlashAttention 4 in bf16. As for the GEMMs, the model uses the
measured time. FA4 fuses the whole attention into one kernel, so that time
already includes the softmax (exponentials, running max and sum) and the causal
masking. To compare it with the ideal roofline's 70%, the table expresses the
time as an efficiency: the two matmuls' FLOPs (what the ideal roofline counts)
divided by the measured time. The softmax adds time but no matmul FLOPs, so it
shows up as a lower efficiency, not as missing time. Ulysses CP and TP both
split the heads and keep the whole sequence on each GPU:

::: figure mla-shapes

The dense and expert GEMMs hold 60% down to about 4,600 rows per expert; TP 4
costs a few points, and experts drop to 52% and 42% at 2,300 and 1,150 rows (CP
4 or TP 4 at 8K). MLA does not reach 70%: its forward gets 59-66%, but
forward + backward only 42-52%. The softmax runs inside the kernel between the
two matmuls, its exponentials run on slower special-function units, and the
backward recomputes it and runs five matmuls instead of two. **KDA** runs at
0.40 us per token per layer; under CP a local token costs 1.3-2x more (the
recurrent state is passed between ranks), and TP 2 or TP 4 speeds it up only
1.7x at 8K.

### Memory-bound work

Here our knowledge points the other way: these kernels are not tuned yet. We
keep the bytes from [[compute_roofline.1]], but instead of the full 8 TB/s, we
charge each class the bandwidth its kernels reach in a measured step (FSDP 512
+ EP 64 at 8K, CUDA graphs):

::: figure nonmatmul-table

- **This is where the ideal roofline is furthest off**: 3.5 s at full
  bandwidth, 8.3 s measured, 43% of the bandwidth on average.
- **MXFP8 quantization and norms** come close: 61-71%.
- **The attention residuals and FSDP's copies** reach only about 43%.
- **The attention residuals alone take 3.7 s**, almost as much as all the dense
  MXFP8 GEMMs (3.8 s). Their region upcasts every source to fp32 and runs the
  weighted sum as a batched matrix multiply over at most 9 sources, which lands
  on slow GEMV kernels.

> CAVEAT: These memory-bound kernels likely have room to improve, and the
> practical roofline **deliberately** takes them as they are. So it is not a
> target for optimizing them: speed them up, and the step can run faster than
> the practical roofline predicts. That may sound odd for a roofline, but it is
> our choice: the goal here is not to optimize compute, but to rank the
> parallelism configs that fit in memory. To optimize compute, the ideal
> roofline is the better target.

### Step Time per Config, Sans Communication

Per GPU per step (green / red: faster / slower than the same family without a
split; greyed out: does not fit in memory):

::: figure compute-table

- **Without PP, CP and TP are expensive.** They multiply the microbatches, and
  FSDP re-gathers, copies and re-quantizes every layer's weights each
  microbatch: at 8K, CP 4 grows the memory-bound work from 8.3 to 15 s.
- **PP's ZeRO-1 removes that cost**: PP 8 at 8K needs 16.8 s against 19.0 s for
  FSDP 512, before counting its bubble.
- **Small shapes cost a few seconds**: PP 16 with CP 4 or TP 4 runs its experts
  at 42%, and at 8K TP 4 more than doubles KDA's time.

### Checking against a measured step

A fake-process-group step measures exactly this part: its collectives do no
work, so the GPU time is compute and memory-bound work only. One GPU holds a
single rank of every FSDP 512 + EP 64 config except 32K without a split, so we
ran them all (PP needs one GPU per stage, so its configs are not here):

::: figure sweep-table

- **Close at the base configs**: 5% under at 8K without a split, 1% at 32K with
  CP 2.
- **Further off as the microbatches multiply** (CP and TP at 8K, TP 4 at 32K),
  up to 17%, always on the optimistic side: those configs redo FSDP's weight
  work more often, and those kernels run a little slower than calibrated.
- **The order holds**, which is what the ranking needs, with one exception
  within a second: at 32K the model puts TP 4 ahead of CP 4, the measurement
  the other way round.

> NOTE: Why build the practical roofline at all, if a fake-process-group run
> measures the step directly? Because the step is not one number once
> communication enters. Each collective overlaps a different part of the
> compute: FSDP's gathers the next layer's GEMMs, DistMoE's dispatch the expert
> GEMMs, and TP's and CP's sit between the ops that wait for them. To tell whether a collective is really hidden, and
> which compute it slows down, we need the time of each op, not one total.

**From here on, "the roofline" means this practical roofline**, not the ideal
one from [[compute_roofline.1]].

## Notes
