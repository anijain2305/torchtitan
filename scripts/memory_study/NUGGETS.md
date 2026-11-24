# Nugget candidates for the Kimi K3 post rewrite

A nugget is a non-obvious fact you can derive from a few numbers, stated as a
fact (not advice), and confirmed by a measurement. Candidates collected while
discussing the rewrite (target: 512 GB300, 288 GB). Numbers are from the study
unless marked "to measure".

## Memory: who gets the memory

1. **Every parallelism choice decides who gets the memory: weights (saves
   communication) or activations (saves recompute).** Per GPU, total = weight
   state + activations + temporaries. FSDP -> ZeRO-1/2 spends memory on
   unsharded weights/gradients to skip per-microbatch gathers; PP shrinks
   weights to one stage but multiplies activations by the microbatches in
   flight; TP/CP shrink activations; remat/offload trade activations for
   compute/bandwidth. With the remat ladder, the parallelism choice decides how
   much remat a config pays. Candidate unifying nugget; show a split bar
   (weights / activations / temporaries) and the remat point per fit-map cell.
2. **Sharding storage does not shard the peak.** FSDP stores 1/512 of
   everything, but the gathered layers in flight are unsharded: ZeRO-3 alone
   needs 434 GB at 8K, over 288 GB. (Existing; holds at 288 GB.)
3. **ZeRO-1 alone is impossible; PP makes it affordable** (14 TB per GPU without
   PP). (Existing; holds.)
4. **Gradients, not weights, are what keep ZeRO-1 from fitting.** ZeRO-1 + EP 128
   + TP 8 at 8K: master+optimizer 43 GB, MXFP8 expert weights 21, bf16 dense
   weights 24, unsharded fp32 gradients 134 (85 of it the experts'). ZeRO-2
   (sharded gradients) brings it to 121 GB. Answers the "why not EP at ZeRO-1"
   feedback.
5. **TP can only shrink dense weights to 10%.** TP shards 90% of the 56.7B
   non-expert params; the latent MoE projections (routed_down/up, 4.7B), router
   and MLA low-rank projections stay whole on every TP rank. So ZeRO-1 + TP has
   a floor.
6. **CP/TP can buy you out of remat.** At 8K, no remat needs 322 GiB per
   microbatch (FSDP + EP 64), but CP 4 or TP 4 cuts it ~4x, so a high-CP/TP
   config can drop remat entirely; CP 1 + full remat vs CP 4 + no remat is a
   real trade (to measure in the ranking).
7. **When activations overflow, reach for CP/TP before PP** (32K: FSDP + EP needs
   374 GB; CP 2 or TP 2 fits at 288 GB). (Existing; holds.)

## Remat

8. **A few regions hold most of the memory.** Recomputing the attention kernels
   only (KDA + MLA inner_attention) takes activations from 257 GiB (save every
   region) to 149 GiB per microbatch at 8K; the KDA kernel alone keeps 2.26 GiB
   per layer for ~1.2 ms of forward.
9. **Selective remat is worth ~9% of GPU time at 8K.** Full depth, FSDP + EP 64:
   GPU busy 19.67 s (full remat) vs 17.90 s (selective), all of it GEMM replay;
   the attention kernels are recomputed under both.
10. **"Full remat" doesn't replay the whole forward.** DistMoE is always saved
    (about half of each block's forward), so full remat replays ~4.2 s per step,
    not the full forward.
11. **DistMoE's intermediates are about as cheap to recompute per byte as the
    KDA kernel** (slot factor 1.0 vs None: 3.2 GiB per layer for 1.5 ms net at
    8K; 12.7 GiB for 16.7 ms at 32K), so recomputing them (today's default)
    belongs near the top of the ladder; saving all 92 layers would need ~290 GiB.
12. **Offload can stand in for recompute** (to measure): selective remat saves
    ~1.6 GiB per layer per ~13 ms forward, ~125 GB/s D2H vs GB300 C2C ~190-210
    GB/s per GPU (shared by two GPUs per Grace). DistMoE slot buffers are not
    offloadable.

## Communication

13. **Hiding a collective is not free.** Overlap slows both compute and the
    collective. (Existing, measured.)
14. **Across NVL domains, bandwidth is set by how many NICs carry the traffic.**
    (Existing, measured.)
15. **In Kimi K3, TP moves several times more data than CP.** TP + SP
    communicates in all 93 layers (~235 MB per layer forward at 8K, TP 2); CP
    only in the 24 MLA layers (Ulysses, ~250 MB per layer) because KDA passes a
    ~3 MB recurrent state. Model-specific, derivable from the architecture.

## Pipelines

16. **More PP stages shrink everything inside a stage** (smaller EP, smaller
    expert GEMMs). (Existing.)
17. **Interleaving shrinks the bubble ~v times until chunks get too small.**
    (Existing; close to textbook.)

## Not nuggets (keep as prose or caveats)

- DistMoE's worst-case routing-pointer sizing (16 GB at 8K, 65 GB at 32K):
  an implementation overhead with a fix in progress, not a principle.
- "A roofline isn't one-size-fits-all": true but not derivable.
- GB300 reports 276.6 GiB usable; selective remat at CP 1 fits only with
  expandable_segments. Budget definition, not a nugget.
