# Roofline Analysis for Kimi K3

If you haven't read it yet, start with Edward Yang's excellent
[walkthrough of DeepSeek-V3](https://deepseek-v3.ezyang.com/index.html), which
works out from first principles how the model is sharded across 2,048 H800 GPUs.

This post extends that exercise to Kimi K3. Kimi K3 brings constraints the
DeepSeek-V3 analysis did not need to explore:

- **A 4x bigger model on fewer GPUs.** 2.78T parameters (vs 671B) on 512
  GB200s (vs 2,048 H800s), so memory is tight from the start.
- **Deeper, with longer context.** 93 layers (vs 61) at up to 32K tokens:
  activations outgrow what FSDP and EP can fit, so we also investigate context
  and tensor parallelism.
- **Attention residuals.** Each layer reads from many earlier layers, which
  complicates pipeline parallelism.
- **Built to actually run.** The goal is to train this model in TorchTitan, so
  we trade paper idealizations for practical choices: DistMoE for the experts,
  more precise memory and latency models, and memory and compute checked
  against fake-process-group runs in PyTorch.

## Notes
