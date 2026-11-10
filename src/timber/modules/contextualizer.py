"""Contextualizer (paper module C): factorised spatial / temporal attention over patches.

Each patch is embedded as ``signal_projection(signal) + position_projection(position)``
(position = channel and time embeddings concatenated), then every layer attends over the channel axis at each
patch time and over the patch (time) axis at each channel, followed by a
feed-forward block. The pass over time can be causal, so that a contextualised
patch only depends on the past (needed for the TACO forecasting objective).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import FeedForward, MultiHeadAttention

AttentionOrder = Literal["channel_time", "time_channel"]


class ContextualizerLayer(nn.Module):
    """One factorised layer: attention over one axis, then over the other, then a feed-forward block.

    Each sub-layer is a residual block ``x + dropout(f(x))`` normalised after the sum
    (post-norm, default) or before ``f`` (``use_pre_norm``).

    Args:
        hidden_dim: Channels of the patch embeddings.
        num_heads: Attention heads (both passes).
        ffn_ratio: Hidden size of the feed-forward block as a multiple of ``hidden_dim``.
        dropout: Dropout on attention probabilities, feed-forward activations and residual branches.
        attention_order: ``"channel_time"`` (channels first) or ``"time_channel"`` (patches first).
        is_causal: Causal mask on the pass over time.
        share_attention_weights: Use one attention module for both passes.
        use_pre_norm: Pre-norm instead of post-norm residual blocks.

    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        ffn_ratio: int = 2,
        dropout: float = 0.0,
        attention_order: AttentionOrder = "channel_time",
        is_causal: bool = True,
        share_attention_weights: bool = True,
        use_pre_norm: bool = False,
    ) -> None:
        super().__init__()
        if attention_order not in ("channel_time", "time_channel"):
            msg = f"attention_order must be 'channel_time' or 'time_channel', got {attention_order!r}."
            raise ValueError(msg)
        self.attention_order = attention_order
        self.is_causal = is_causal
        self.share_attention_weights = share_attention_weights
        self.use_pre_norm = use_pre_norm

        def attention() -> MultiHeadAttention:
            return MultiHeadAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        # One module called twice when the weights are shared (no aliased state-dict keys)
        if share_attention_weights:
            self.attention = attention()
        else:
            self.channel_attention = attention()
            self.time_attention = attention()
        # Post-norm by default (``use_pre_norm=False``): normalise after each residual sum.
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_ratio * hidden_dim, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _attention(self, axis: str) -> MultiHeadAttention:
        if self.share_attention_weights:
            return self.attention
        return self.channel_attention if axis == "channel" else self.time_attention

    # Residual connection around a sublayer, with the norm before (pre-norm) or after (post-norm) the sum.
    def _residual(self, x: torch.Tensor, norm: nn.LayerNorm, sublayer) -> torch.Tensor:
        if self.use_pre_norm:
            return x + self.dropout(sublayer(norm(x)))
        return norm(x + self.dropout(sublayer(x)))

    def _attend_channels(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, channel, patch, dim) -> sequences over channels: (batch * patch, channel, dim)
        batch, channels, patches, dim = x.shape
        # Attention over channels: one sequence per (entity, patch time), running along the channel axis.
        sequences = x.transpose(1, 2).reshape(batch * patches, channels, dim)
        attention = self._attention("channel")
        sequences = self._residual(sequences, self.channel_norm, lambda h: attention(h)[0])
        return sequences.view(batch, patches, channels, dim).transpose(1, 2)

    def _attend_time(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, channel, patch, dim) -> sequences over patches: (batch * channel, patch, dim)
        batch, channels, patches, dim = x.shape
        # Attention over time: one sequence per (entity, channel), running along the patch axis.
        sequences = x.reshape(batch * channels, patches, dim)
        attention = self._attention("time")
        # The causal mask only applies to the time pass: a patch cannot attend to the patches after it.
        sequences = self._residual(sequences, self.time_norm, lambda h: attention(h, is_causal=self.is_causal)[0])
        return sequences.view(batch, channels, patches, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (batch, channel, patch, hidden_dim)`` -> same shape."""
        # The two passes run in the order of ``attention_order`` (the paper: channels first, then time).
        if self.attention_order == "channel_time":
            x = self._attend_time(self._attend_channels(x))
        else:
            x = self._attend_channels(self._attend_time(x))
        return self._residual(x, self.ffn_norm, self.ffn)


class Contextualizer(nn.Module):
    """Embed patches with their positions and contextualise them with factorised attention.

    Args:
        patch_length: Samples per patch.
        position_dim: Channels of the position embeddings (channel + time).
        hidden_dim: Channels of the layers.
        output_dim: Channels of the output projection, or ``None`` for no projection
            (output has ``hidden_dim`` channels).
        num_layers: Number of `ContextualizerLayer`. ``0`` keeps only the input
            embedding (the "no contextualizer" ablation).
        num_heads: See `ContextualizerLayer`.
        ffn_ratio: See `ContextualizerLayer`.
        dropout: See `ContextualizerLayer`.
        attention_order: See `ContextualizerLayer`.
        is_causal: See `ContextualizerLayer`.
        share_attention_weights: See `ContextualizerLayer`.
        use_pre_norm: See `ContextualizerLayer`.
        use_gradient_checkpointing: Recompute each layer in the backward pass instead of storing
            its activations (training only). Saves memory at the cost of a second forward pass.

    """

    def __init__(
        self,
        patch_length: int,
        position_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        num_heads: int = 1,
        ffn_ratio: int = 2,
        dropout: float = 0.0,
        attention_order: AttentionOrder = "channel_time",
        is_causal: bool = True,
        share_attention_weights: bool = True,
        use_pre_norm: bool = False,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Two projections summed (not one on the concatenation):
        # masked patches drop their whole signal term, bias included
        self.signal_projection = nn.Linear(patch_length, hidden_dim)
        self.position_projection = nn.Linear(position_dim, hidden_dim)
        self.layers = nn.ModuleList(
            ContextualizerLayer(
                hidden_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                attention_order=attention_order,
                is_causal=is_causal,
                share_attention_weights=share_attention_weights,
                use_pre_norm=use_pre_norm,
            )
            for _ in range(num_layers)
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim) if output_dim is not None else nn.Identity()

    @property
    def output_dim(self) -> int:
        """Channels of the contextualised patches (``patch_dim`` in the encoder)."""
        if isinstance(self.output_projection, nn.Linear):
            return self.output_projection.out_features
        return self.hidden_dim

    def forward(
        self,
        patches: torch.Tensor,
        position_embeddings: torch.Tensor,
        patch_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contextualise patches.

        Args:
            patches: ``(batch, channel, patch, patch_length)``.
            position_embeddings: ``(batch, channel, patch, position_dim)``.
            patch_keep: Optional ``(batch, channel, patch)`` boolean mask; the signal of masked
                (``False``) patches is hidden, only their position embedding is kept. Used when
                no encoder masks the patches downstream (no-encoder ablation).

        Returns:
            ``(batch, channel, patch, output_dim)``.

        """
        signal = self.signal_projection(patches)
        if patch_keep is not None:
            # Hide the signal of masked patches; their position embedding stays, so the model still knows where they
            # are.
            signal = signal * patch_keep.unsqueeze(-1).to(signal.dtype)
        x = signal + self.position_projection(position_embeddings)
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                # Recompute the layer in the backward pass instead of storing its activations (training only).
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.output_projection(x)
