# Roofline Analysis for Kimi K3

<!-- One paragraph: the question and the answer. -->

If you haven't read it yet, start with Edward Yang's excellent
[walkthrough of DeepSeek-V3](https://deepseek-v3.ezyang.com/index.html), which
works out from first principles how the model is sharded across 2,048 H800 GPUs.

This post tries the same exercise for Kimi K3, a 2.78T-parameter model. The
setup: 512 GB200 GPUs with 192 GB of memory each, a global batch of 16M tokens
per step, and two context lengths, 8K and 32K. First we find every parallelism
config that fits the model on this cluster. Then we build a performance
roofline to see which of them trains fastest.

## Notes
