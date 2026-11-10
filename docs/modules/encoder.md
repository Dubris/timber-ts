# Encoder

The encoder (paper module **E**) compresses the contextualised patches of an entity into a fixed number of
**latent embedding** `e`: 512 latent vectors of dimension 256 in the paper, whatever the number of channels. The design
follows Perceiver IO (Jaegle et al., 2022): a small set of learned **latent queries** `e_queries` reads the (large) input
through cross-attention, then refines itself.

## What it does

1. **Cross-attention.** The latent queries attend to the contextualized patches (keys and values), so that each query
   gathers the information it needs from the whole recording.
2. **Latent self-attention.** `num_self_attention_layers` layers let the latents exchange information.
3. **Blocks.** The cross-attention and self-attention are repeated `num_blocks` times. Each block re-attends the refined
   latents to the same patches. The latents of every block are kept, because the paper's analyses read several of them.

An optional **CLS token** is appended to the latents (`use_cls_token`); the head uses it when pooling is `"cls_token"`.

## Patch masking

During pretraining, a fraction `patch_mask_ratio` (0.7 in the paper) of the **contextualized patches** is hidden from
the encoder. The masking takes place after the contextualizer and before the encoder: the contextualizer sees the whole
recording, and the encoder only reads the patches that are not masked. The model must then infer the hidden content from
the rest instead of copying its input.

The paper draws each patch independently with a Bernoulli(`patch_mask_ratio`) trial. The implementation hides **exactly
`floor(patch_mask_ratio * number of patches)` patches in every row of the batch**, so that the kept patches form a
tensor of static shape (efficient on MPS and with `torch.compile`). The expected fraction is the same. The encoder
selects the kept patches before the cross-attention, so keys and values are only computed for them.

## Options that matter

| Option | Default | Experiments | Effect |
|---|---|---|---|
| `num_latents` | 512 | 512 | Number of latent queries, which gives the size of the latent embedding. |
| `latent_dim` | 256 | 256 | Width of the latents. |
| `num_blocks` | 2 | 2 | Cross-attention and self-attention blocks (`L_blocks`). |
| `num_self_attention_layers` | 4 | 4 | Latent self-attention layers per block (`L_enc`). |
| `num_cross_attention_heads` | 1 | 1 | Heads of the cross-attention. |
| `use_cls_token` | `True` | | Appends a CLS latent. |
| `latent_dropout_ratio` | 0.0 | | Drops whole latents during training. |

## References

Jaegle, A. et al. (2022). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR.

## Reference API

::: timber.modules.encoder
    options:
      members:
        - PerceiverEncoder
        - PerceiverEncoderBlock
        - EncoderOutput
