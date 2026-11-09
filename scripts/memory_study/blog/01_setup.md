# Kimi K3 at a Glance

Kimi K3 is a 93-layer decoder. Two kinds of attention alternate through it:

- **KDA** (Kimi Delta Attention), a linear-attention layer whose cost grows
  linearly with sequence length. 69 of the 93 layers use it.
- **MLA** (Multi-head Latent Attention), DeepSeek's full attention with
  compressed keys and values, whose cost grows quadratically. Every fourth layer
  is MLA, and so is the last one: 24 in all.

::: figure layer-strip

Every layer except the first is a mixture-of-experts (MoE) layer: 896 routed
experts, of which each token picks 16, plus one shared expert that every token
uses.

**Attention residuals** are new in Kimi K3. Instead of taking only the previous
layer's output, each layer's input is a learned, softmax-weighted mix of many
earlier layers' outputs. Keeping every layer's output around would be
expensive, so the layers are grouped into blocks of 12 (the outlined rows
above). Inside a block, outputs add up into a running sum, the partial; when
the block ends, the partial becomes a single entry on a stack. Each layer's
input mixes the stack entries (the embedding and every finished block) with
the current partial, and one last mix over all of them feeds the LM head.
With 93 layers that is at most 9 sources instead of 93, which
keeps memory down and, as we will see later, keeps the data passed between
pipeline stages manageable.

::: figure attention-residuals

## Almost All of It Is Experts

| Component | Parameters | Share |
|---|---|---|
| Routed experts | 2,723B | 98.0% |
| Attention | 36B | 1.3% |
| Shared experts | 12B | 0.4% |
| Everything else | 8B | 0.3% |
| **Total** | **2,779B** |  |

Routed experts are 98% of the model: 92 layers x 896 experts of 33M parameters
each. Yet each token only visits 16 of the 896, so a token touches about
**105B parameters**, under 4% of the model. That gap sets up everything that
follows: memory has to hold all 2.78T parameters, but compute only pays for the
105B each token uses.

## Notes
