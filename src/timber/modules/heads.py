"""Finetuning heads (paper module H): latent pooling, view fusion and an MLP.

An entity contributes several *views* (recordings, channel chunks, time chunks) to a
batch. ``fusion_stage`` sets where the views of an entity are combined:

- ``"early"``: average the pooled latents of the views, then apply the MLP once per entity;
- ``"late"``: apply the MLP per view, then average the predictions (`fuse_views`);
- ``"loss"``: keep one prediction per view and repeat the entity label for each view.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

PoolingType = Literal["mean", "weighted_mean", "cls_token", "attention"]
FusionStage = Literal["early", "late", "loss"]


class MLPHead(nn.Module):
    """MLP with ``num_hidden_layers`` hidden layers of size ``widening_factor * input_dim``.

    Layout (as in the original implementation): ``[LayerNorm] -> (Linear -> act -> Dropout) x n -> Linear``;
    without hidden layers the dropout is applied to the input: ``Dropout -> Linear``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        widening_factor: float = 1.0,
        dropout: float = 0.1,
        use_input_norm: bool = False,
    ) -> None:
        super().__init__()
        # Without hidden layers the MLP is a single linear map from the input to the outputs.
        hidden_dim = int(widening_factor * input_dim) if num_hidden_layers > 0 else input_dim
        layers: list[tuple[str, nn.Module]] = []
        if use_input_norm:
            layers.append(("input_norm", nn.LayerNorm(input_dim)))
        for i in range(num_hidden_layers):
            layers += [
                (f"linear{i}", nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)),
                (f"activation{i}", nn.GELU()),
                (f"dropout{i}", nn.Dropout(dropout)),
            ]
        # With no hidden layer, dropout is applied directly to the input (the paper's linear head).
        if num_hidden_layers == 0:
            layers.append(("dropout", nn.Dropout(dropout)))
        layers.append(("output", nn.Linear(hidden_dim, output_dim)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., input_dim) -> (..., output_dim)``."""
        return self.layers(x)


class LatentPooling(nn.Module):
    """Pool latent tokens ``(batch, tokens, dim)`` into ``(batch, dim)``.

    Args:
        pooling: ``"mean"``, ``"weighted_mean"`` (softmax of learned per-token weights),
            ``"cls_token"`` (last token) or ``"attention"`` (softmax of a learned linear score).
        latent_dim: Channels of the tokens (for ``"attention"``).
        num_tokens: Number of tokens (for ``"weighted_mean"``).
        attention_dropout: Dropout on the attention-pooling weights (training only).

    """

    def __init__(
        self,
        pooling: PoolingType,
        latent_dim: int,
        num_tokens: int,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "weighted_mean", "cls_token", "attention"):
            msg = f"pooling must be 'mean', 'weighted_mean', 'cls_token' or 'attention', got {pooling!r}."
            raise ValueError(msg)
        self.pooling = pooling
        self.attention_dropout = attention_dropout
        if pooling == "weighted_mean":
            self.token_weights = nn.Parameter(torch.zeros(num_tokens))
        if pooling == "attention":
            self.attention_score = nn.Linear(latent_dim, 1)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the pooled latents ``(batch, dim)`` and, for attention pooling, the weights ``(batch, tokens)``."""
        if self.pooling == "mean":
            return latents.mean(dim=1), None
        if self.pooling == "cls_token":
            return latents[:, -1], None
        if self.pooling == "weighted_mean":
            # Learned weight per token, normalised over the tokens.
            weights = self.token_weights.softmax(dim=0)
            return (latents * weights[None, :, None]).sum(dim=1), None
        # Attention pooling: one score per token, softmax over the tokens (the paper's head).
        weights = self.attention_score(latents).squeeze(-1).softmax(dim=1)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        return (latents * weights.unsqueeze(-1)).sum(dim=1), weights


def fuse_views(x: torch.Tensor, num_entities: int) -> torch.Tensor:
    """Average the views of each entity: ``(entity * view, dim) -> (entity, dim)`` (entity-major)."""
    return x.reshape(num_entities, -1, x.shape[-1]).mean(dim=1)


class PredictionHead(nn.Module):
    """Latents -> prediction: optional projection, dropout, pooling, early view fusion, MLP.

    Args:
        latent_dim: Channels of the encoder latents.
        num_tokens: Latent tokens (including the CLS token, if any).
        output_dim: Outputs (regression targets, or classes).
        pooling: See `LatentPooling`.
        fusion_stage: ``"early"`` averages the pooled views before the MLP; ``"late"`` and
            ``"loss"`` return one prediction per view (fused by the loss).
        projection_dim: Optional ``Linear + GELU`` projection of every token before pooling.
        pooling_input_dropout: Dropout on the tokens before pooling.
        num_hidden_layers: See `MLPHead`.
        widening_factor: See `MLPHead`.
        dropout: See `MLPHead`.
        use_input_norm: See `MLPHead`.

    """

    def __init__(
        self,
        latent_dim: int,
        num_tokens: int,
        output_dim: int,
        pooling: PoolingType = "mean",
        fusion_stage: FusionStage = "late",
        projection_dim: int | None = None,
        pooling_input_dropout: float = 0.0,
        num_hidden_layers: int = 0,
        widening_factor: float = 1.0,
        dropout: float = 0.5,
        use_input_norm: bool = False,
    ) -> None:
        super().__init__()
        if fusion_stage not in ("early", "late", "loss"):
            msg = f"fusion_stage must be 'early', 'late' or 'loss', got {fusion_stage!r}."
            raise ValueError(msg)
        self.fusion_stage: FusionStage = fusion_stage
        self.pooling_input_dropout = pooling_input_dropout
        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Sequential(nn.Linear(latent_dim, projection_dim), nn.GELU())
            latent_dim = projection_dim
        self.pooling = LatentPooling(pooling, latent_dim=latent_dim, num_tokens=num_tokens)
        self.mlp = MLPHead(
            latent_dim,
            output_dim,
            num_hidden_layers=num_hidden_layers,
            widening_factor=widening_factor,
            dropout=dropout,
            use_input_norm=use_input_norm,
        )

    def forward(
        self,
        latents: torch.Tensor,
        num_entities: int,
        fusion_stage: FusionStage | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Predict from the latents of every view.

        Args:
            latents: ``(entity * view, tokens, latent_dim)``, entity-major.
            num_entities: Entities in the batch.
            fusion_stage: Override of the configured fusion stage (e.g. ``"late"`` for validation).

        Returns:
            ``prediction`` (``(entity, output_dim)`` for early fusion, else ``(entity * view, output_dim)``),
            ``pooled`` (``(entity * view, dim)``) and ``pooling_weights`` (attention pooling only).

        """
        # Optional projection of every token to a smaller width before pooling.
        if self.projection is not None:
            latents = self.projection(latents)
        latents = F.dropout(latents, p=self.pooling_input_dropout, training=self.training)
        pooled, pooling_weights = self.pooling(latents)
        fusion_stage = fusion_stage or self.fusion_stage
        # Early fusion: average the views before the MLP, so one prediction per entity.
        # Late and loss fusion keep one prediction per view.
        features = fuse_views(pooled, num_entities) if fusion_stage == "early" else pooled
        return {"prediction": self.mlp(features), "pooled": pooled, "pooling_weights": pooling_weights}
