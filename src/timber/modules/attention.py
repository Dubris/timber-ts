"""Attention building blocks shared by the contextualizer, the Perceiver encoder and the decoders.

All attention goes through `scaled_dot_product_attention`
(SDPA), so the fused CUDA/MPS kernels are used when available. Masks are boolean
(``True`` = may attend); no ``-inf`` float masks are materialised.

Shape conventions: ``batch`` is any flattened batch axis, ``query`` / ``key`` are
sequence axes, ``dim`` is the channel axis.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-head (self- or cross-) attention with separate query/key/value projections.

    Args:
        query_dim: Channels of the query input.
        key_value_dim: Channels of the key/value input (defaults to ``query_dim``).
        num_heads: Number of attention heads.
        qk_dim: Channels of the query/key projections (defaults to ``query_dim``).
        value_dim: Channels of the value projection (defaults to ``qk_dim``).
        output_dim: Channels of the output projection (defaults to ``query_dim``).
        dropout: Dropout on the attention probabilities (training only).
        bias: Whether the projections have a bias.

    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int | None = None,
        num_heads: int = 1,
        qk_dim: int | None = None,
        value_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        key_value_dim = query_dim if key_value_dim is None else key_value_dim
        qk_dim = query_dim if qk_dim is None else qk_dim
        value_dim = qk_dim if value_dim is None else value_dim
        output_dim = query_dim if output_dim is None else output_dim
        if qk_dim % num_heads or value_dim % num_heads:
            msg = f"qk_dim ({qk_dim}) and value_dim ({value_dim}) must be divisible by num_heads ({num_heads})."
            raise ValueError(msg)

        self.num_heads = num_heads
        self.qk_head_dim = qk_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(query_dim, qk_dim, bias=bias)
        self.k_proj = nn.Linear(key_value_dim, qk_dim, bias=bias)
        self.v_proj = nn.Linear(key_value_dim, value_dim, bias=bias)
        self.out_proj = nn.Linear(value_dim, output_dim, bias=bias)

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        # (batch, seq, heads * head_dim) -> (batch, heads, seq, head_dim)
        return x.unflatten(-1, (self.num_heads, head_dim)).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend from ``query`` to ``key_value`` (self-attention when ``key_value`` is ``None``).

        Args:
            query: ``(batch, query, query_dim)``.
            key_value: ``(batch, key, key_value_dim)`` or ``None``.
            attn_mask: Optional boolean mask broadcastable to ``(batch, heads, query, key)``.
            is_causal: Apply a causal mask (query ``i`` attends keys ``<= i``).
            need_weights: Also return the attention probabilities ``(batch, heads, query, key)``.
                This uses the explicit (non-fused) path, so keep it for analysis.

        Returns:
            ``(output, weights)`` with output ``(batch, query, output_dim)``; ``weights`` is
            ``None`` unless ``need_weights``.

        """
        # Self-attention when no separate key/value input is given.
        key_value = query if key_value is None else key_value
        q = self._split_heads(self.q_proj(query), self.qk_head_dim)
        k = self._split_heads(self.k_proj(key_value), self.qk_head_dim)
        v = self._split_heads(self.v_proj(key_value), self.value_head_dim)
        # Dropout only applies in training mode.
        dropout_p = self.dropout if self.training else 0.0

        weights = None
        # On MPS (torch 2.15), the fused SDPA kernel is the fastest without autograd but rejects dropout, and
        # SDPA's backward is slower than the explicit path (paper channel pass, bf16: 149 ms vs 92 ms):
        # use the explicit path whenever autograd or dropout is involved
        on_mps = q.device.type == "mps"
        # Explicit path (softmax written out): needed to return the attention weights, and on MPS when autograd or
        # dropout is on.
        explicit = need_weights or (on_mps and (dropout_p > 0 or torch.is_grad_enabled()))
        if explicit:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            if is_causal:
                # Lower-triangular mask: query ``i`` may attend to keys ``<= i`` only.
                causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            probabilities = scores.softmax(dim=-1)
            weights = probabilities if need_weights else None
            context = F.dropout(probabilities, p=dropout_p, training=self.training) @ v
        else:
            # Fused kernel (FlashAttention or memory-efficient on CUDA, fused on MPS): the same maths, without storing
            # the scores.
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        context = context.transpose(1, 2).flatten(-2)  # (batch, query, value_dim)
        return self.out_proj(context), weights


class FeedForward(nn.Module):
    """Position-wise MLP: ``fc1 -> GELU -> dropout -> fc2``.

    Args:
        dim: Input and output channels.
        hidden_dim: Hidden channels.
        dropout: Dropout after the activation (training only).

    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., dim) -> (..., dim)``."""
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class PerceiverLayer(nn.Module):
    """Pre-norm Perceiver layer: attention then MLP, each with a residual connection.

    ``output = h + mlp(norm(h))`` with ``h = attention(norm(query), norm(key_value)) [+ query]``.
    The query residual can be disabled (used by the decoders, whose queries and
    outputs have different meanings).

    Args:
        query_dim: Channels of the queries (and of the output).
        key_value_dim: Channels of the keys/values for cross-attention, or ``None`` for self-attention.
        num_heads: Number of attention heads.
        qk_dim: Channels of the query/key projections (defaults to ``query_dim``).
        widening_factor: MLP hidden size as a multiple of ``query_dim``.
        dropout: Dropout on the attention probabilities.
        use_query_residual: Add the (un-normalised) query to the attention output.

    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int | None = None,
        num_heads: int = 1,
        qk_dim: int | None = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        use_query_residual: bool = True,
    ) -> None:
        super().__init__()
        self.is_cross_attention = key_value_dim is not None
        self.use_query_residual = use_query_residual
        self.query_norm = nn.LayerNorm(query_dim)
        self.key_value_norm = nn.LayerNorm(key_value_dim) if key_value_dim is not None else None
        self.attention = MultiHeadAttention(
            query_dim=query_dim,
            key_value_dim=key_value_dim,
            num_heads=num_heads,
            qk_dim=qk_dim,
            output_dim=query_dim,
            dropout=dropout,
        )
        self.mlp_norm = nn.LayerNorm(query_dim)
        self.mlp = FeedForward(query_dim, widening_factor * query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``query (batch, query, query_dim)``, ``key_value (batch, key, key_value_dim)``.

        Returns ``(output, weights)`` with output ``(batch, query, query_dim)``.
        """
        normed_query = self.query_norm(query)
        normed_key_value = self.key_value_norm(key_value) if self.key_value_norm is not None else None
        hidden, weights = self.attention(
            normed_query,
            normed_key_value,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        if self.use_query_residual:
            # Residual on the queries, so that the output keeps the identity of each query (disabled for the decoders).
            hidden = hidden + query
        # Second residual: the feed-forward block after the attention.
        return hidden + self.mlp(self.mlp_norm(hidden)), weights
