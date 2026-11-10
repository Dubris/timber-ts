# Tokenizer

The tokenizer turns a raw multichannel recording into **patch embeddings**, one per channel and patch. It is the only
place where the input samples enter the model, and where the model learns what each channel is.

## What it does

1. **Patching.** Each channel is cut into non-overlapping patches of `patch_length` samples (60 in the paper, so 15
   patches for a window of 900 samples). `patch_stride` gives overlapping patches; `None` means non-overlapping.
2. **Channel embedding.** Each channel has a vector of size `channel_embedding_dim` (64 in the paper), which tells the
   model which node a patch comes from. In the paper it is learned. The fMRI experiments start it from a spectral
   embedding of the cortical mesh (`channel_embedding_init`), which training then refines.
3. **Output.** The tokenizer returns a `TokenizerOutput` with the patches, the channel embeddings and, if enabled, the
   time embeddings. The next modules read it (see [Contextualizer](contextualizer.md)).
4. **Time embedding (optional).** `time_embedding_type` adds a Fourier embedding of the time. The paper's pretraining
   uses **none**: the time order is inferred from the causal attention. The option exists for the time-encoding
   ablation, with `"patch"` (the patch index) or `"timestep"` (the absolute timestep of the patch start).

## Options that matter

| Option | Default | Experiments | Effect |
|---|---|---|---|
| `patch_length` | 16 | 60 | Samples per patch. |
| `patch_stride` | `None` | `None` | Hop between patches; `None` means non-overlapping. |
| `num_channels` | 1024 | 59412 (fMRI) | Size of the channel embedding table: the largest channel index you may use. |
| `channel_embedding_dim` | 64 | 64 | Size of each channel embedding (`D_p`). |
| `trainable_channel_embedding` | `True` | yes | Whether the channel embeddings are learned. |
| `time_embedding_type` | `None` | `None` | `None`, `"patch"` or `"timestep"`. |

## Reference

::: timber.modules.tokenizer
    options:
      members:
        - PatchTokenizer
        - TokenizerOutput
        - fourier_features
        - linear_positions
        - num_patches
        - forecast_offset
