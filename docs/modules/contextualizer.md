# Contextualizer

The contextualizer (paper module **C**) lets the signal content of each patch flow across the other channels and
over time, into the past of its own channel. A patch on its own is a short window of one channel; after the
contextualizer, it carries the state of the other channels at that moment and the history of its channel, which is what
captures the spatio-temporal structure of the recording. Its output, the **contextualized patches** `z`, is the input of
the encoder and of the TACO decoder.

## What it does

The contextualizer **factorises** the attention over the two dimensions of the patch grid `(channel, patch)`. Each of
its `num_layers` layers (3 in the paper) runs:

- an **attention over channels** at each patch time: all the channels attend to each other, at a given time;
- a **attention over time** for each channel: the patches of a channel attend along time, and with `is_causal=True` a
  patch only sees the patches before it;

followed by a feed-forward block. Factorising the two passes, instead of attending over every pair of patches, reduces
the cost of the attention from `O(N_c^2 N_t^2)` to `O(N_t N_c^2 + N_c N_t^2)`: with the paper's sizes (1,024 channels,
15 patches) it is about a 15-fold speed-up. The two passes use the same attention parameters by default
(`share_attention_weights=True`) and run in the order `attention_order` (`"channel_time"`: channels first, as in the paper).

## The inputs

Each patch enters the contextualizer as two projections that are summed: a projection of the **signal** of the patch
(`patch_length` samples to `hidden_dim`) and a projection of its **position**. The position is the channel embedding,
and, when a time embedding is configured, the time embedding too (see [Tokenizer](tokenizer.md)). The two projections
are summed rather than concatenated, so that masked patches can drop their whole signal term, bias included.

Whether the position includes time depends on the objective:

- for **TACO**, as in the paper, the position is the channel embedding only: the temporal order is inferred from the
  causal attention, and no time-position embedding is given;
- for the **vanilla reconstruction** objective, the paper adds sinusoidal time-position embeddings, and removes the
  causal mask, since the time order is then given explicitly.

## Options that matter

| Option | Default | Experiments | Effect |
|---|---|---|---|
| `hidden_dim` | 128 | 128 | Width of the layers (`D`). |
| `num_layers` | 3 | 3 | Number of channel and time layers (`L_pre`); `0` removes the attention. |
| `num_heads` | 4 | 4 | Attention heads per pass. |
| `ffn_ratio` | 2 | 2 | Widening of the feed-forward blocks. |
| `is_causal` | `True` | `True` | Causal attention along time: required by TACO, see [Decoders](decoder.md). |
| `attention_order` | `"channel_time"` | `"channel_time"` | Which pass comes first in each layer. |
| `share_attention_weights` | `True` | `True` | One attention module for both passes. |
| `use_gradient_checkpointing` | `False` | `False` | Recompute the layers in the backward pass to save memory. |

## Reference

::: timber.modules.contextualizer
    options:
      members:
        - Contextualizer
        - ContextualizerLayer
