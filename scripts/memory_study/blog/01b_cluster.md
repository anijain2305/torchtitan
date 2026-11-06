# The cluster and the training run

The intro gave the headline numbers; here are the details the rest of the post
leans on.

## The cluster

- 512 GB200 GPUs, each with 192 GB of HBM, 2.5 PFLOPS of dense bf16 compute and
  5 PFLOPS in FP8.
- The GPUs sit in 8 **NVL domains** of 64 (64 of each NVL72 rack's 72 GPUs):
  an NVL domain is a group of GPUs connected to each other by NVLink. Inside
  an NVL domain every GPU reaches every other at 900 GB/s per direction, of
  which collectives over 64 GPUs reach about 640 GB/s. Between NVL domains
  traffic goes over InfiniBand: on this cluster each GPU has two 400 Gb/s
  NICs, 100 GB/s per direction, about 6x slower than NVLink collectives.
  Which parallelism lands on which link turns out to matter a lot
  ([[communication]]).

## The training run

- A global batch of 16M tokens per step: 2,048 sequences at 8K context, or 512
  at 32K.
- Mixed precision: master weights, gradients and optimizer state (Muon) stay in
  fp32; the dense linears (attention projections, shared experts) run their
  matmuls in bf16, and the routed experts in MXFP8, 8-bit operands with a
  shared scale per 32 elements, to use GB200's FP8 throughput. The Kimi K3
  report also limits MXFP8 to the routed experts. In TorchTitan's config:
    - `training.dtype = "float32"`: parameters and optimizer state are fp32.
    - `training.mixed_precision_param = "bfloat16"`: FSDP all-gathers
      parameters in bf16.
    - `training.mixed_precision_reduce = "float32"`: gradients are reduced in
      fp32.
    - DistMoE's `expert_precision="mxfp8"`: quantizes the gathered routed
      expert weights to MXFP8 for the matmuls.
- Routed experts run on DistMoE, which fuses token dispatch, the grouped expert
  GEMMs and the combine into a few kernels.

## Notes
