# Attention, masking and devices

These helpers are used by the modules above. They are documented here because the choices they make (fused
attention, static shapes, device defaults) affect speed and memory.

## Attention

All attention goes through `torch.nn.functional.scaled_dot_product_attention`, so the fused kernels of CUDA and
Apple MPS are used when they apply. The three shared blocks are:

- `MultiHeadAttention`: the attention itself, with boolean masks (`True` means "may attend");
- `FeedForward`: the position-wise MLP after each attention;
- `PerceiverLayer`: cross-attention from queries to a context, then a feed-forward block. The encoder and decoders
  are built from it.

On MPS the explicit attention path is used whenever gradients or dropout are active: the fused kernel rejects dropout,
and the explicit path has faster backward passes there.

::: timber.modules.attention
    options:
      members:
        - MultiHeadAttention
        - FeedForward
        - PerceiverLayer

## Patch masking

Pretraining hides a fraction `patch_mask_ratio` of the patches. Every row of the batch hides the same number of patches,
so the kept and hidden patches can be picked with `gather` into tensors of fixed shape. The shapes do not depend on the
random draw, which keeps MPS and `torch.compile` efficient.

::: timber.modules.masking
    options:
      members:
        - PatchMask
        - num_masked_patches
        - random_patch_mask
        - gather_patches
        - token_dropout

## Devices and precision

`get_accelerator()` returns `"cuda"`, `"mps"` or `"cpu"`, in that order of preference. `default_precision()` returns
`"bf16-mixed"` on CUDA and MPS and `"32-true"` on CPU; an explicit precision in the trainer always wins.

::: timber.device
    options:
      members:
        - get_accelerator
        - default_precision
