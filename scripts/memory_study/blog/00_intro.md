# Roofline Analysis for Kimi K3

*"What is the fastest way to pretrain Kimi K3 on 512 GB300s?"*

The TorchTitan team looked into this question recently. GPU clusters are
exorbitantly expensive, so before a hero run, how do you gain confidence
that the parallelism config you picked (how you shard the model across GPUs) is
actually good? And how do you know what you left on the table? The answer:
roofline analysis.

## Why I wrote this

Everyone knows the prerequisite reads for distributed training, the two
classics: the [Ultra-Scale
Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) and [How
to Scale Your Model](https://jax-ml.github.io/scaling-book/). They taught me the
parallelisms. Yet I found myself grasping at straws in our team meetings about
training Kimi K3. I started working on distributed training very recently, and I
struggled to follow the conversations: when to reach for context parallelism
instead of tensor parallelism, or for pipeline parallelism instead of FSDP. My
colleagues had internalized some rules of thumb that I could not pick up from
the two books alone. This post was born out of curiosity (or FOMO, if you
prefer). I wanted to be able to reason about these questions myself.

So I would not call this post novel. These parallelisms have been around for
years, and many of the lessons here are already known. But rooflining a hard
model from first principles can give you the epiphanies that others earned over
years of working on distributed training. For example, I knew what expert
parallelism was, but working from first principles I reached a point where it
was evident that FSDP would not be enough even with infinitely many GPUs, and I
arrived at expert parallelism. These **nuggets** (a word I borrowed from the
very talented Driss Guessous's [blog](https://drisspg.github.io/nuggets/)) were
the real wins: they accelerated my learning so that I can reason about
parallelism with more confidence.

Besides teaching you, working from first principles also finds bugs. When a
measurement disagrees with what the first-principles model predicts, one of
them is wrong, and sometimes it is the code. For example, I found that
[DistMoE](https://github.com/meta-pytorch/dist_moe) sized the routing indices it
saves for backward for the worst case, costing Kimi K3 tens of GB per GPU.
Similarly, the kernels for attention residuals (new in Kimi K3) reached only
about 20% of the GPU's memory bandwidth, so they needed a custom kernel.

I am writing this with two goals in mind:

- **To encourage you to do a roofline analysis yourself.** Understanding the
  parallelisms from first principles is very satisfying!
- **To share what I learned along the way:** the nuggets, and a framework for
  building a roofline of an LLM training step end to end. I haven't come across
  a framework that carries a roofline through a whole model, so I think it is
  worth sharing. I also don't know whether mine is the best way to do it, so
  feedback is very welcome: email me at jainanimesh2305@gmail.com with the
  subject "Kimi K3 Roofline Blog".

I also want to give a shout-out to Edward Yang's excellent [walkthrough of
DeepSeek-V3](https://deepseek-v3.ezyang.com/index.html) and its roofline. It was
a large part of what inspired this study, and this post is a similar exercise,
with two aims of its own: finding these nuggets, and working out what a mental
model for rooflining should look like.

## The recipe

The post turned out to be long, so here is the tldr of the approach

1. **Find every parallelism config that fits in memory.** As you will see later, parallelism
   configs trade persistent memory (weights, gradients, optimizer state) against
   activation memory, so for each config the persistent memory sets the budget
   left for activations.
2. **Spend the leftover memory well.** For each config, pick the activation
   checkpointing ([torch_remat](https://github.com/meta-pytorch/remat)) and
   [activation offloading](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/ACTIVATION_OFFLOAD.md)
   setting that is fastest while still fitting that budget. All of this can be
   measured on a single GPU with
   [fake-process-group runs](https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md#fake-backend-debugging),
   which run one rank of the full cluster.
3. **Build a latency model for compute and communication.** For compute, measure
   the kernels at each config's real shapes (fake-process-group runs again),
   and keep the ideal roofline as a sanity check that those kernels are
   reasonable. For communication, measure your cluster's bandwidths, and measure
   the overlap penalty too: *hiding communication behind compute is not free*.
4. **Rank the configs** by their predicted step time.
5. **Validate the top few** with real runs, as precisely as you need.

Note that this is a paper study: we could not keep the 512-GPU cluster, so the final
rankings were never validated end to end on it. Fortunately, earlier runs on
GB200s let us check part of the ranking. Beyond that, every number that could
be measured was measured on real GB300s: mostly through fake-process-group runs
on a single node, plus short runs on up to 32 GPUs across hosts to measure
communication.

## How this post is organized

- **Kimi K3 primer:** the model, the cluster and the training run.
- **The memory problem:** what each parallelism shards, which configs fit, and
  how to spend the memory that is left on remat and offload.
- **The latency model:** compute at each config's shapes, and what
  communication really costs once it overlaps with compute.
- **Rankings:** which config wins, why, and how far to trust the ranking.


This post is the result of many people's help; see the
[thank-yous at the end](#sec-thanks).

## Notes
