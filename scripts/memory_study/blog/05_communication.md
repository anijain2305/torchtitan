# Communication

So far the step was compute only. Every parallelism adds communication, and
whether it costs time depends on one thing: whether it can run while the GPU
computes something else. Here is which communication each parallelism adds,
which link it uses, and what it hides behind:

| Parallelism | Communication | Link | Hides behind |
|---|---|---|---|
| FSDP | all-gather of each layer's weights (forward, backward); reduce-scatter of its gradients (backward) | NVLink within a PP stage; without PP it spans all 8 domains: NVLink inside each, InfiniBand between them | the neighbouring layer's compute (one layer ahead) |
| EP | token dispatch and combine (all-to-all), per MoE layer and pass | NVLink (EP 64 fits in one domain) | the routed-expert GEMMs (fused in DistMoE) |
| TP | all-gather before column-parallel linears, reduce-scatter after row-parallel ones | NVLink | nothing today: async TP cannot fuse MXFP8's block-scaled GEMMs, so they run in sequence |
| CP | MLA: all-to-alls around attention (Ulysses); KDA: recurrent state between ranks | NVLink | nothing today: attention waits for its all-to-all, and the next op for the one after it |
| PP | activations and the AttnRes block stack between stages | InfiniBand (stages sit in different domains) | nothing: on the critical path |
| Optimizer | DistMuon gathers gradient shards into whole matrices and scatters the update | NVLink / InfiniBand | nothing: once per step |

TP and CP get the same treatment: in TorchTitan with MXFP8 both are exposed
today. Both could hide behind compute with the same trick, splitting the work
into chunks and overlapping one chunk's collective with another chunk's compute
(async TP does this for bf16 and row-wise FP8 GEMMs; a head-chunked Ulysses
could do it around attention), but the model charges them as they run now.

## Overlap Is Not Free

"Hides behind" does not mean free. An NCCL collective is a kernel: each of its
channels is one thread block pinned to one SM, and every byte it moves is read
from HBM on one GPU and written on another. So while it runs, the compute it
overlaps gets fewer SMs and less memory bandwidth, and the collective itself
slows down too. Every rank runs the kernel, so every GPU pays.

We measured it on 16 GPUs connected by NVLink, as the GPUs of one 64-GPU domain
are ([[methodology.5]]): each collective running back to back on a side stream
while a GEMM, or a memory-bound kernel, runs on the main stream:

::: figure overlap-table

- **The slowdown follows the collective's share of the SMs.** NCCL runs 32
  channels (21% of the GPU's 152 SMs), and both the GEMM and the memory-bound
  kernel run about 1.3x slower while a collective is active: close to the
  1 / (1 - 0.21) = 1.27 that losing those SMs alone predicts.
- **The collective slows down as well**, 1.4-2.2x when it shares the GPU with
  compute. A collective that looks hidden by its solo time may not be.
- **The size of the domain barely matters.** 32 GPUs give the same bandwidth
  and slowdowns, so the model uses these numbers for 64 GPUs as well.

So an overlapped collective costs two penalties. With <i>t</i><sub>comm</sub>
the collective's time alone, <i>t</i><sub>compute</sub> the time alone of the
compute it hides behind, and the measured slowdowns (compute 1.3x; collective
1.6x, the middle of the all-gather's and reduce-scatter's, the collectives
FSDP runs):

::: figure overlap-formulas

- **Compute penalty**: the compute runs 1.3x slower for as long as the
  collective runs alongside it.
- **Collective penalty**: the collective runs 1.6x slower while it overlaps
  (and the compute it hides behind lasts 1.3x longer). That costs nothing as
  long as it still finishes within the compute; when it does not, the part left
  over runs afterwards, alone and at full speed. The last term,
  max(0, <i>t</i><sub>comm</sub> &minus; <i>t</i><sub>compute</sub>), is what
  would be exposed even without any slowdown; the model counts it as plain
  exposed time.

Two examples:

::: figure overlap-examples

In the first, the collective ends after 1.6 ms, while the compute has done
1.6 / 1.3 = 1.23 ms of its work: 0.37 ms of penalty, and the rest of the
compute runs at full speed. In the second, the compute ends first, after 13 ms;
the collective has done 13 / 1.6 = 8.13 ms of its work by then, and its other
1.88 ms runs alone afterwards.

## Crossing NVLink Domains

Only two kinds of traffic leave a 64-GPU NVLink domain: **FSDP's** all-gathers
and reduce-scatters without PP, which span all 512 GPUs, and **PP's** hops
between stages, which sit in different domains. Both go over InfiniBand, where
each GPU has its own NIC, about 13x slower than its NVLink.

**One rule covers both.** NCCL runs an all-gather or reduce-scatter as a ring
through every GPU of the group: over NVLink inside a domain, through NICs
between domains. A group with <i>k</i> GPUs in each domain has <i>k</i> NICs
there, and NCCL spreads the ring's traffic over all of them, so:

::: figure ib-rule-formula

We checked it on a GB300 cluster, on islands of 4 or 8 GPUs connected by NVLink,
joined by InfiniBand ([[methodology.5]]):

::: figure ib-rule-table

> NOTE: The measurements run on GB300, whose NICs are 800 Gb/s (100 GB/s per
> GPU per direction), twice as fast as the 400 Gb/s (50 GB/s) NICs of the GB200
> cluster we model. So the measured bandwidths here are twice what GB200 would
> get: what carries over is the fraction of a NIC (one NIC's worth, 90% of line
> rate, and so on), which the model applies to GB200's 50 GB/s.

<i>k</i> GPUs per island give about <i>k</i> NICs' worth, whatever the number
of islands.

**What it means on GB200** (50 GB/s NICs):

::: figure gb200-ib-table

- **FSDP's dense weights are fine.** Every NVL domain holds 64 / TP GPUs of
  their FSDP group, so its NICs together outrun NVLink.
- **FSDP's experts are not.** EP 64 spreads the experts over an NVL domain's
  64 GPUs, so each expert shard's FSDP group is just its 8 replicas, one per
  NVL domain: its ring runs through a single NIC per NVL domain.

The difference is not how much data each moves but how many NICs carry it.
EP keeps the experts' share of a GPU's gathers small: per layer and
microbatch (FSDP 512, 8K), a GPU receives 1.26 GB of dense weights and 0.81 GB
of its expert shard, about 60 / 40. But the dense weights come from an FSDP
group of 512 GPUs, 64 in every NVL domain, so 64 NICs per NVL domain carry
them; the expert shard comes from 8 GPUs, one per NVL domain, so one NIC does:

| Per layer and microbatch | Received per GPU | NICs per NVL domain | All-gather |
|---|---|---|---|
| Dense weights | 1.26 GB | 64: NVLink speed | 4.0 ms |
| Expert shard | 0.81 GB | 1: about 41 GB/s | 19.8 ms |

Half of the dense gather's 4.0 ms is not transfer at all but **setup cost**: a
ring over 512 GPUs takes 511 sequential steps of a few microseconds each (we
measured 2.2 us over NVLink, 6.5 us across InfiniBand), about 2 ms per
collective however little data it moves. Even so, against
a 6.5 ms layer forward the dense gather hides; the expert gather, with less
data, takes five times longer and does not. This is what sinks FSDP
512 in [[ranking]].

So the model times FSDP's collectives across domains as:

::: figure fsdp-ib-formula

While they overlap compute, they slow it down like collectives inside a domain
(measured on islands of 8: about 1.3x on the compute, 1.7x on the collective),
with the same formulas as in [[communication.1]].

**PP's hops** go from one GPU to one GPU, so through one NIC: 90% of its line
rate in our measurements, at any number of islands. The NIC moves the data
itself, so the hop costs the compute little: NCCL takes one SM per side, and
GEMMs on both the sending and the receiving GPU slow down about 1.10x,
memory-bound work barely at all. A hop sits on the critical path ([[step]]), so
its whole time counts:

::: figure hop-formulas

> TODO: measure 64 GPUs in one NVLink domain; the model uses the 32-GPU
> measurements for 64.
