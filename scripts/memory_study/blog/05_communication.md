# Communication

We have modelled memory and compute; now we add communication. At best, a
collective costs nothing: it is hidden, overlapped with compute that does not
depend on it. So first, which collectives each parallelism adds, which link
they use, and what compute they can hide behind:

| Parallelism | Communication | Link | Hides behind |
|---|---|---|---|
| FSDP | all-gather of each layer's weights (forward, backward); reduce-scatter of its gradients (backward) | NVLink within a PP stage; without PP it spans both NVLink and InfiniBand | the neighbouring layer's compute (one layer ahead) |
| EP | token dispatch and combine (all-to-all), per MoE layer and pass | NVLink (EP 64 fits in one NVL domain) | the routed-expert GEMMs (fused in DistMoE) |
| TP | all-gather before column-parallel linears, reduce-scatter after row-parallel ones | NVLink | nothing today: TP's collectives and GEMMs run in sequence (no async TP) |
| CP | MLA: all-to-alls around attention (Ulysses); KDA: recurrent state between ranks | NVLink | nothing today: attention waits for its all-to-all, and the next op for the one after it |
| PP | activations and the AttnRes block stack between stages | InfiniBand (stages sit in different NVL domains) | nothing: on the critical path |

TP and CP get the same treatment: both are exposed in TorchTitan today. Both
could be overlapped (async TP, overlapped Ulysses), but the model charges them
as TorchTitan runs them; [[ranking.5]] asks what overlapping them would change.

## Overlap Is Not Free

But in reality, "hides behind" does not mean free. An NCCL collective is a
kernel: each of its channels is one thread block pinned to one SM, and every
byte it moves is read from HBM on one GPU and written on another. So the
overlapped compute gets fewer SMs and less memory bandwidth, and the collective
itself slows down too. For SPMD collectives, every rank runs the kernel, so every GPU pays.

We measured it on 16 GPUs connected by NVLink, as the GPUs of one 64-GPU NVL domain
are ([[methodology.5]]): each collective running back to back on a side stream
while a GEMM, or a memory-bound kernel, runs on the main stream:

::: figure overlap-table

- **The slowdown follows the collective's share of the SMs.** NCCL runs 32
  channels (21% of the GPU's 152 SMs), and both the GEMM and the memory-bound
  kernel run about 1.3x slower while a collective is active: close to the
  1 / (1 - 0.21) = 1.27 that losing those SMs alone predicts.
- **The collective slows down as well**, 1.4-2.2x when it shares the GPU with
  compute. A collective that looks hidden by its solo time may not be.

So an overlapped collective costs two penalties. With <i>t</i><sub>comm</sub>
the collective's time alone, <i>t</i><sub>compute</sub> the time alone of the
compute it hides behind, and the measured slowdowns (compute 1.3x; collective
1.6x, the middle of the all-gather's and reduce-scatter's, the collectives
FSDP runs):

::: figure overlap-formulas

- **Compute penalty**: the compute runs 1.3x slower while the collective runs
  alongside it.
- **Collective penalty**: the collective runs 1.6x slower while it overlaps.
  That is free as long as it still finishes within the compute (itself
  stretched 1.3x); otherwise its remainder runs afterwards, alone and at full
  speed. The last term, max(0, <i>t</i><sub>comm</sub> &minus;
  <i>t</i><sub>compute</sub>), would be exposed even without any slowdown, so
  the model counts it as plain exposed time.

Two examples:

::: figure overlap-examples

In the first, the collective finishes at 1.6 ms, when the compute has done
1.23 ms of its work: 0.37 ms of penalty. In the second, the compute finishes
first, at 13 ms; the collective has done 8.13 ms of its work by then, and its
last 1.88 ms run alone.

> INSIGHT: Hiding a collective is not free. While it overlaps compute, both run slower, so a model that only checks whether communication fits behind compute calls configs free that are not.

## Crossing NVL Domains

Only two kinds of traffic leave a 64-GPU NVL domain: **FSDP's** all-gathers
and reduce-scatters without PP, which span all 512 GPUs, and **PP's** hops
between stages, which sit in different NVL domains. Both go over InfiniBand.
Each GPU of our cluster has two 400 Gb/s NICs, 100 GB/s per direction
together, about 6x slower than NVLink collectives (about 640 GB/s); below, "a
GPU's NIC" means both.

**One rule covers both.** FSDP gathers each weight from, and reduce-scatters
its gradient over, the GPUs that shard it: its **FSDP group**. NCCL runs these
collectives as a ring through every GPU of the FSDP group: over NVLink inside
an NVL domain, through NICs between NVL domains. An FSDP group with <i>k</i>
GPUs in each NVL domain has <i>k</i> NICs there, and NCCL spreads the ring's
traffic over all of them, so:

::: figure ib-rule-formula

We checked it on a GB300 cluster with NVL domains of 4 or 8 GPUs, joined by
InfiniBand at the same 100 GB/s per GPU (one 800 Gb/s NIC each)
([[methodology.5]]):

::: figure ib-rule-table

What sets the bandwidth is <i>k</i>, how many of each NVL domain's GPUs (so
NICs) the FSDP group uses, not how many GPUs the NVL domain has: <i>k</i> GPUs per
NVL domain give about <i>k</i> NICs' worth, whatever the number of NVL domains.

**What it means on GB200:**

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
| Expert shard | 0.81 GB | 1: about 82 GB/s | 9.9 ms |

Half of the dense gather's 4.0 ms is not transfer at all but **setup cost**: a
ring over 512 GPUs takes 511 sequential steps of a few microseconds each (we
measured 2.2 us over NVLink, 6.5 us across InfiniBand), about 2 ms per
collective however little data it moves. The expert gather, with less data,
takes 2.5 times longer. Together the two take 13.9 ms, about as long as the
layer's forward (13.9 ms with its memory-bound work), so they barely hide, and
they slow that compute down the whole time. This is what holds FSDP 512 back
in [[ranking]].

> INSIGHT: Across NVL domains, bandwidth is set by how many NICs carry the traffic, not by how much data there is: k GPUs of a group in each NVL domain give it k NICs there. Placement that leaves a group one GPU per domain puts its whole ring on a single NIC.

So the model times FSDP's collectives across NVL domains as:

::: figure fsdp-ib-formula

While they overlap compute, they slow it down like collectives inside an NVL
domain (measured with NVL domains of 8: about 1.3x on the compute, 1.7x on the collective),
with the same formulas as in [[communication.1]].

**PP's hops** go from one GPU to one GPU, so through one NIC: 90% of its line
rate in our measurements, at any number of NVL domains. The NIC moves the data
itself, so the hop costs the compute little: NCCL takes one SM per side, and
GEMMs on both the sending and the receiving GPU slow down about 1.1x
(memory-bound work barely at all; we charge 1.1x on all of it). Whether a hop's time
ends up on the critical path depends on the pipeline schedule: with plain
1F1B it mostly does, with interleaving it mostly does not ([[step]]). Either
way, the compute that runs meanwhile pays the penalty of [[communication.1]]
at 1.1x (the hop's own slowdown, at most 1.2x, we leave out):

::: figure hop-formulas
