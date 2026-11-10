"""Perceiver IO encoder (paper module E): latents cross-attend to the patches, then self-attend.

The encoder is a stack of blocks. Each block has its own cross-attention from the
latents to the (kept) patches, followed by ``num_self_attention_layers`` latent
self-attention layers. An optional CLS token is appended after the latents.

Patch masking is applied by *selecting* the kept patches (``kept_indices``)
before the cross-attention, so keys and values are only computed for visible
patches; this is equivalent to masking them out of the attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import PerceiverLayer
from .masking import gather_patches, token_dropout


@dataclass
class EncoderOutput:
    """Output of `PerceiverEncoder`.

    Attributes:
        latents: ``(batch, num_tokens, latent_dim)`` after the last block, with
            ``num_tokens = num_latents (+ 1 with the CLS token, last position)``.
        block_latents: Latents after every block (``block_latents[-1] is latents``).
        cross_attentions: Per block ``(batch, heads, num_tokens, num_kept_patches)`` when requested.
        self_attentions: Per block, the last latent self-attention layer ``(batch, heads, num_tokens, num_tokens)``
            when requested.

    """

    latents: torch.Tensor
    block_latents: tuple[torch.Tensor, ...]
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    self_attentions: tuple[torch.Tensor | None, ...] | None = None


class PerceiverEncoderBlock(nn.Module):
    """Cross-attention from the latents to the patches, then latent self-attention layers."""

    def __init__(
        self,
        patch_dim: int,
        latent_dim: int,
        num_self_attention_layers: int,
        num_self_attention_heads: int,
        num_cross_attention_heads: int,
        self_attention_widening_factor: int,
        cross_attention_widening_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attention = PerceiverLayer(
            latent_dim,
            key_value_dim=patch_dim,
            num_heads=num_cross_attention_heads,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout,
        )
        self.self_attention_layers = nn.ModuleList(
            PerceiverLayer(
                latent_dim,
                num_heads=num_self_attention_heads,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
            )
            for _ in range(num_self_attention_layers)
        )

    def forward(
        self,
        latents: torch.Tensor,
        patches: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``latents (batch, tokens, latent_dim)``, ``patches (batch, kept, patch_dim)`` ->
        latents, cross-attention and last self-attention probabilities (``None`` unless ``need_weights``)."""
        latents, cross_attention = self.cross_attention(latents, patches, need_weights=need_weights)
        self_attention = None
        for i, layer in enumerate(self.self_attention_layers):
            is_last = i == len(self.self_attention_layers) - 1
            latents, weights = layer(latents, need_weights=need_weights and is_last)
            self_attention = weights if is_last else self_attention
        return latents, cross_attention, self_attention


class PerceiverEncoder(nn.Module):
    """Perceiver IO encoder over contextualised patches.

    Args:
        patch_dim: Channels of the input patches.
        num_latents: Number of latent tokens.
        latent_dim: Channels of the latents.
        num_blocks: Number of encoder blocks (each with its own cross-attention).
        num_self_attention_layers: Self-attention layers per block.
        num_self_attention_heads: Heads of the latent self-attention.
        num_cross_attention_heads: Heads of the cross-attention.
        self_attention_widening_factor: MLP widening of the self-attention layers.
        cross_attention_widening_factor: MLP widening of the cross-attention layers.
        dropout: Dropout on the attention probabilities.
        use_cls_token: Append a trainable CLS token after the latents.
        latent_dropout_ratio: Probability to drop each latent token (training only, before every block).
        use_gradient_checkpointing: Recompute each block in the backward pass (training only).
        initializer_range: Standard deviation of the CLS token initialisation.

    """

    def __init__(
        self,
        patch_dim: int,
        num_latents: int,
        latent_dim: int,
        num_blocks: int = 1,
        num_self_attention_layers: int = 4,
        num_self_attention_heads: int = 1,
        num_cross_attention_heads: int = 1,
        self_attention_widening_factor: int = 1,
        cross_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        use_cls_token: bool = False,
        latent_dropout_ratio: float = 0.0,
        use_gradient_checkpointing: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.latent_dropout_ratio = latent_dropout_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cls_token = None
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
            nn.init.trunc_normal_(self.cls_token, std=initializer_range)
        self.blocks = nn.ModuleList(
            PerceiverEncoderBlock(
                patch_dim=patch_dim,
                latent_dim=latent_dim,
                num_self_attention_layers=num_self_attention_layers,
                num_self_attention_heads=num_self_attention_heads,
                num_cross_attention_heads=num_cross_attention_heads,
                self_attention_widening_factor=self_attention_widening_factor,
                cross_attention_widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )

    @property
    def use_cls_token(self) -> bool:
        """Whether a CLS latent is appended to the latents."""
        return self.cls_token is not None

    @property
    def num_tokens(self) -> int:
        """Latent tokens in the output (latents plus the CLS token)."""
        return self.num_latents + int(self.use_cls_token)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        kept_indices: torch.Tensor | None = None,
        num_blocks: int | None = None,
        need_weights: bool = False,
    ) -> EncoderOutput:
        """Encode the patches into latents.

        Args:
            patch_embeddings: ``(batch, num_patches, patch_dim)``.
            kept_indices: Optional ``(batch, num_kept)`` indices of the visible patches
                (see `PatchMask`); all patches are used when ``None``.
            num_blocks: Run only the first ``num_blocks`` blocks (default: all).
            need_weights: Return the cross-attention probabilities of every block and the probabilities of
                its last self-attention layer.

        Returns:
            `EncoderOutput`.

        """
        # Masked patches are removed here, before the cross-attention: keys and values are only computed for the kept
        # ones.
        patches = patch_embeddings if kept_indices is None else gather_patches(patch_embeddings, kept_indices)
        batch_size = patches.shape[0]
        # The latents are shared by all entities: expand them along the batch axis (no copy).
        latents = self.latents.expand(batch_size, -1, -1)
        if self.cls_token is not None:
            # The CLS token is one more latent, appended after the learned latents.
            latents = torch.cat((latents, self.cls_token.expand(batch_size, -1, -1)), dim=1)

        block_latents, cross_attentions, self_attentions = [], [], []
        for block in self.blocks[:num_blocks]:
            # Optionally drop whole latents before each block (training only).
            latents = token_dropout(latents, self.latent_dropout_ratio, self.training)
            # Gradient checkpointing trades compute for memory: the block is recomputed in the backward pass.
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled() and not need_weights:
                latents, cross_attention, self_attention = checkpoint(block, latents, patches, use_reentrant=False)
            else:
                latents, cross_attention, self_attention = block(latents, patches, need_weights=need_weights)
            # Keep the latents of every block: the analyses read several of them.
            block_latents.append(latents)
            cross_attentions.append(cross_attention)
            self_attentions.append(self_attention)

        return EncoderOutput(
            latents=latents,
            block_latents=tuple(block_latents),
            cross_attentions=tuple(cross_attentions) if need_weights else None,
            self_attentions=tuple(self_attentions) if need_weights else None,
        )
