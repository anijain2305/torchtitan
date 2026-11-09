# Pipelines: Hops and Schedules

With compute ([[compute_roofline]]) and communication ([[communication]])
modelled, one piece of the step time is left: the pipeline itself. A PP step
is more than each stage's work: microbatches move between stages in hops, and
stages wait on each other. This section builds the step time of a PP config
from two choices: the pipeline schedule (plain or interleaved 1F1B), and how
a hop carries Kimi K3's attention residuals (the transport).

## Schedules

Every pipeline schedule trades peak memory, bubble and step time differently,
and each would need its own model. We keep to two, both in TorchTitan:

- **Plain 1F1B** alternates one forward and one backward per stage, so a stage
  never holds more than `P` microbatches at a time: that bounds its activation
  memory.
- **Interleaved 1F1B** splits each stage into `v` chunks to shrink the pipeline
  bubble, at the cost of holding more microbatches at a time.

A pipeline's **bubble** is the time a stage sits idle. It comes from three
places: the **warmup**, while the first microbatches travel down the pipeline
and the later stages wait for work; the **cooldown**, while the last ones
drain and the early stages have nothing left; and the **hops** in between,
when a stage has to wait for its input to arrive from another one. We build up
the step time piece by piece, with `P` equal stages, `M` microbatches (not the
GEMM rows `M` of [[compute_roofline]]), `t` one
stage's forward and backward for one microbatch, and `h` one hop. (In the real
pipeline both change from stage to stage: the stages are uneven, and a hop's
bytes depend on the transport ([[step.2]]) and where it sits in the pipeline.
Our model represents that: for every config it simulates the schedule itself,
each chunk's forward and backward with its compute, communication and
penalties, each hop with its own bytes. Equal stages and hops here just keep
the explanation simple.)

**Warmup and cooldown.** Each stage has `M t` of work. But the last stage
cannot start until the first microbatch has gone through the `P - 1` stages
before it, and the first stage cannot finish until the last microbatch's
backward has come back through them. Every stage idles `(P - 1) t` in total,
split differently: the last stage waits `(P - 1)` forwards at the start and
`(P - 1)` backwards at the end, while the first stage does all its waiting at
the start, until the first backward comes back. Interleaving splits each stage into `v` chunks of `t / v`, so the
pipeline fills and drains in chunk-sized steps: `(P - 1) t / v`.

**Hops in plain 1F1B.** Every microbatch's round trip, down the pipeline and
back, crosses `2 (P - 1)` hops. Plain 1F1B lets each stage hold only as many
microbatches as keep it busy while that trip runs, if the trip were compute
only. So the stages end up waiting for every hop: `2 (P - 1)` per microbatch,
shared by the `P` stages that work in parallel, or `2 (P - 1) / P` hops per
microbatch, almost two, for the whole step (plus `2 (P - 2)` more while the
pipeline fills and drains).

::: figure pp-hop-timeline

**Hops in interleaved 1F1B.** This schedule keeps more microbatches in flight on
each stage, and that extra work covers the hops in the steady state: by the
time a stage runs out of other work, the data it waits for has arrived. Only
while the pipeline fills and drains is there nothing to cover them, and then
they cost one microbatch's round trip through all `P v` chunks: `2 (P v - 1)`
hops, once. The same work and hops, with two chunks per stage (the subscript;
scroll sideways):

::: figure pp-hop-timeline-interleaved

**Adding it up**, the step is the sum of four parts (they match the schedule
simulation exactly for equal stages; [[methodology.7]] shows why the hops
behave this way):

::: figure bubble-formulas

Interleaving shrinks the warmup and cooldown by `v`, and turns the hops from a
cost every microbatch pays into one paid once.

## What a Hop Carries

Besides the hidden state, a stage needs the attention-residual blocks that
earlier stages finished ([[setup]]). How they get there is the hop's
**transport**. TorchTitan uses one for each schedule:

- **chain** (plain 1F1B): every hop carries the hidden state and the whole
  block stack so far, so the hops grow along the pipeline: at PP 8 the hop
  into the last stage carries nearly three times the bytes of the hop into the
  second. The backward mirrors it, with the blocks' gradients.
- **cache** (interleaved 1F1B): a rank comes back for later chunks of the same
  microbatch, so it keeps the blocks it received for its earlier chunks, and a
  hop carries only the hidden state and the blocks the receiving rank has not
  seen yet.

::: figure transport-diagram-titan

The transport sets the hop time `h` of [[step.1]]: chain's grows along the
pipeline, cache's stays small after the first lap. [[ranking.4]] asks whether
a transport TorchTitan does not have would do better.
