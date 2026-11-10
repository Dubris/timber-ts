"""Cross-attention decoders (paper module D) and the queries/targets of the pretraining objectives.

A decoder query cross-attends to the encoder latents and is projected to a patch
of ``patch_length`` samples. The objectives differ only by their queries:

- **Forecasting** (``forecasting_<k>``, TACO for ``k = 1``): the query is a
  contextualised patch and the target is the patch ``k`` steps later (plus the
  overlap offset of overlapping patches). With a causal contextualizer the query
  only carries information from the past.
- **Reconstruction**: the queries are the position embeddings of the masked
  patches and the targets are their raw signal.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import PerceiverLayer
from .masking import gather_patches


class CrossAttentionDecoder(nn.Module):
    """Decode queries from the latents: cross-attention + MLP, then a linear projection.

    Args:
        query_dim: Channels of the queries.
        latent_dim: Channels of the encoder latents.
        output_dim: Channels of the prediction (``patch_length``).
        num_heads: Cross-attention heads.
        widening_factor: MLP widening of the cross-attention layer.
        dropout: Dropout on the attention probabilities.

    """

    def __init__(
        self,
        query_dim: int,
        latent_dim: int,
        output_dim: int,
        num_heads: int = 1,
        widening_factor: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attention = PerceiverLayer(
            query_dim,
            key_value_dim=latent_dim,
            num_heads=num_heads,
            qk_dim=latent_dim,
            widening_factor=widening_factor,
            dropout=dropout,
            # No query residual: the output is a patch, not a refinement of the query.
            use_query_residual=False,
        )
        self.output_projection = nn.Linear(query_dim, output_dim)

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``queries (batch, query, query_dim)``, ``latents (batch, latent, latent_dim)``.

        Returns ``(batch, query, output_dim)``.
        """
        # Each query reads the latents, then is projected to one patch of ``output_dim`` samples.
        hidden, weights = self.cross_attention(queries, latents, need_weights=need_weights)
        return self.output_projection(hidden), weights


def parse_forecast_horizon(objective: str) -> int:
    """``"forecasting_<k>"`` -> ``k`` (must be a positive integer)."""
    prefix = "forecasting_"
    if not objective.startswith(prefix) or not objective[len(prefix) :].isdigit() or int(objective[len(prefix) :]) < 1:
        msg = f"Invalid forecasting objective {objective!r}: expected 'forecasting_<k>' with an integer k >= 1."
        raise ValueError(msg)
    return int(objective[len(prefix) :])


def forecasting_queries(contextualized: torch.Tensor, shift: int) -> torch.Tensor:
    """Queries of a forecasting decoder: every patch that has a target ``shift`` patches later.

    Args:
        contextualized: ``(batch, channel, patch, dim)`` contextualised patches.
        shift: Patches between a query and its target (``horizon + forecast_offset``).

    Returns:
        ``(batch, channel * (patch - shift), dim)``, channel-major.

    """
    # The last ``shift`` patches of each channel have no target, so they are not queries.
    # Flatten (channel, patch) into one sequence axis, channel-major.
    return contextualized[:, :, :-shift].flatten(1, 2)


def forecasting_targets(patches: torch.Tensor, shift: int) -> torch.Tensor:
    """Targets matching `forecasting_queries`.

    ``(batch, channel, patch, length)`` -> ``(batch, channel * (patch - shift), length)``.
    """
    # Target of query ``p`` is the patch ``p + shift`` of the same channel.
    return patches[:, :, shift:].flatten(1, 2)


def reconstruction_queries(position_embeddings: torch.Tensor, masked_indices: torch.Tensor) -> torch.Tensor:
    """Queries of the reconstruction decoder: position embeddings of the masked patches.

    Args:
        position_embeddings: ``(batch, channel, patch, position_dim)``.
        masked_indices: ``(batch, num_masked)`` indices into the flattened ``channel * patch`` axis.

    Returns:
        ``(batch, num_masked, position_dim)``.

    """
    # Queries are the position embeddings of the masked patches only: the decoder must infer their content.
    return gather_patches(position_embeddings.flatten(1, 2), masked_indices)


def reconstruction_targets(patches: torch.Tensor, masked_indices: torch.Tensor) -> torch.Tensor:
    """Raw signal of the masked patches: ``(batch, channel, patch, length)`` -> ``(batch, num_masked, length)``."""
    return gather_patches(patches.flatten(1, 2), masked_indices)
