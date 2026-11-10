"""TIMBER models as Hugging Face `PreTrainedModel`.

- `TimberModel`: tokenizer -> contextualizer -> Perceiver encoder; returns the latents
  (the embeddings) and the intermediate tensors.
- `TimberForPretraining`: `TimberModel` + the decoders of ``config.decoders``
  (TACO forecasting, reconstruction); computes the pretraining loss.
- `TimberForPrediction`: `TimberModel` + a regression or classification head
  (``config.head``), optionally with pretraining decoders trained jointly (e.g. TACO finetuning).

Inputs follow the data loader layout: ``signal`` is ``(*batch, channel, time)``. The leading batch
axis is the entity axis: ``signal.shape[0]`` entities, each with ``prod(batch[1:])`` views (recordings,
channel chunks, time chunks), flattened entity-major.

Example::

    model = TimberModel.from_pretrained("path/or/hub-id")
    embeddings = model(signal, channel_indices=channels, mask_patches=False).last_hidden_state
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_timber import DecoderConfig, TimberConfig
from .modules.attention import PerceiverLayer
from .modules.contextualizer import Contextualizer, ContextualizerLayer
from .modules.decoder import (
    CrossAttentionDecoder,
    forecasting_queries,
    forecasting_targets,
    reconstruction_queries,
    reconstruction_targets,
)
from .modules.encoder import PerceiverEncoder
from .modules.heads import LatentPooling, PredictionHead, fuse_views
from .modules.masking import PatchMask, gather_patches, random_patch_mask
from .modules.tokenizer import PatchTokenizer


def _require[T](value: T | None, name: str) -> T:
    """``value``, which the caller needs: raise a clear error instead of failing on ``None`` later."""
    if value is None:
        msg = f"The model output has no {name}."
        raise ValueError(msg)
    return value


# --- Outputs ---------------------------------------------------------------------------------------
@dataclass
class TimberModelOutput(ModelOutput):
    """Output of `TimberModel`.

    Attributes:
        last_hidden_state: Latents of the last computed encoder block ``(batch, tokens, latent_dim)``
            (CLS token last). Without an encoder: the contextualised patches ``(batch, channel * patch, patch_dim)``.
        block_latents: Latents after every computed encoder block.
        contextualized: ``(batch, channel, patch, patch_dim)`` contextualised patches.
        patches: ``(batch, channel, patch, patch_length)`` raw signal patches (targets of the objectives).
        position_embeddings: ``(batch, channel, patch, position_dim)``.
        patch_keep: ``(batch, channel * patch)`` boolean mask of the patches visible to the encoder, if masked.
        masked_indices: ``(batch, num_masked)`` indices of the masked patches, if masked.
        cross_attentions: Encoder cross-attention probabilities per block (``output_attentions=True``).
        self_attentions: Probabilities of the last latent self-attention layer of every block
            (``output_attentions=True``).

    """

    last_hidden_state: torch.Tensor | None = None
    block_latents: tuple[torch.Tensor, ...] | None = None
    contextualized: torch.Tensor | None = None
    patches: torch.Tensor | None = None
    position_embeddings: torch.Tensor | None = None
    patch_keep: torch.Tensor | None = None
    masked_indices: torch.Tensor | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    self_attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class TimberPretrainingOutput(ModelOutput):
    """Output of `TimberForPretraining`.

    Attributes:
        loss: Weighted sum of the objective losses.
        loss_terms: Unweighted loss of every objective with a non-zero weight.
        predictions: Decoder outputs per objective ``(batch, queries, patch_length)``.
        targets: Matching targets per objective.
        last_hidden_state, patch_keep: See `TimberModelOutput`.

    """

    loss: torch.Tensor | None = None
    loss_terms: dict[str, torch.Tensor] | None = None
    predictions: dict[str, torch.Tensor] | None = None
    targets: dict[str, torch.Tensor] | None = None
    last_hidden_state: torch.Tensor | None = None
    patch_keep: torch.Tensor | None = None


@dataclass
class TimberPredictionOutput(ModelOutput):
    """Output of `TimberForPrediction`.

    Attributes:
        loss: Weighted sum of the head loss and of the decoder losses (when ``labels`` are given).
        loss_terms: Unweighted loss terms (``"regression"`` / ``"classification"`` and objectives).
        logits: Predictions used by the loss: ``(entity, outputs)`` for early and late fusion,
            ``(entity * view, outputs)`` for loss-level fusion.
        view_logits: Predictions per view ``(entity * view, outputs)`` (``None`` for early fusion).
        pooled: Pooled latents per view ``(entity * view, dim)``.
        pooling_weights: Attention-pooling weights ``(entity * view, tokens)``.
        last_hidden_state: Latents read by the head.

    """

    loss: torch.Tensor | None = None
    loss_terms: dict[str, torch.Tensor] | None = None
    logits: torch.Tensor | None = None
    view_logits: torch.Tensor | None = None
    pooled: torch.Tensor | None = None
    pooling_weights: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None


# --- Base class ------------------------------------------------------------------------------------
def _default_linear_init(module: nn.Linear) -> None:
    # Same distribution as nn.Linear.reset_parameters, with tensor methods (unaffected by HF's no_init_weights)
    bound = 1.0 / math.sqrt(module.in_features) if module.in_features > 0 else 0.0
    module.weight.data.uniform_(-bound, bound)
    if module.bias is not None:
        module.bias.data.uniform_(-bound, bound)


class TimberPreTrainedModel(PreTrainedModel):
    """Weight initialisation, gradient checkpointing and the Hugging Face plumbing shared by the TIMBER models.

    The initialisation follows the original implementation: contextualizer linears ``N(0, 0.02)`` with zero
    bias (PatchTST), every other linear the PyTorch default, LayerNorm ``(1, 0)``, channel embedding
    and latents ``N(0, 1)``, CLS token ``N(0, initializer_range)``.
    """

    config_class = TimberConfig
    base_model_prefix = "timber"
    main_input_name = "signal"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.encoder.initializer_range if self.config.encoder is not None else 0.02
        if isinstance(module, Contextualizer):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    submodule.weight.data.normal_(0.0, std)
                    submodule.bias.data.zero_()
                    submodule._is_hf_initialized = True  # pyright: ignore[reportArgumentType]
        elif isinstance(module, nn.Linear) and not getattr(module, "_is_hf_initialized", False):
            _default_linear_init(module)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)
        elif isinstance(module, PerceiverEncoder):
            module.latents.data.normal_(0.0, 1.0)
            if module.cls_token is not None:
                module.cls_token.data.normal_(0.0, std)
        elif isinstance(module, LatentPooling) and module.pooling == "weighted_mean":
            module.token_weights.data.zero_()

    def _gradient_checkpointing_modules(self) -> list[Contextualizer | PerceiverEncoder]:
        return [m for m in self.modules() if isinstance(m, (Contextualizer, PerceiverEncoder))]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None) -> None:
        """Recompute the contextualizer layers and encoder blocks in the backward pass."""
        for module in self._gradient_checkpointing_modules():
            module.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Turn gradient checkpointing off in every layer."""
        for module in self._gradient_checkpointing_modules():
            module.use_gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is enabled."""
        return any(m.use_gradient_checkpointing for m in self._gradient_checkpointing_modules())

    def compile_layers(self, **compile_kwargs: Any) -> int:
        """``torch.compile`` every attention layer in place (regional compilation); returns their number.

        The contextualizer layers and the Perceiver layers (encoder and decoders) are compiled one by
        one: compilation is quick, the parameter names and checkpoints are unchanged, and it composes with
        gradient checkpointing. About 2x faster training for the paper model on Apple MPS.
        ``compile_kwargs`` go to `Module.compile`.
        """
        layers = [m for m in self.modules() if isinstance(m, (ContextualizerLayer, PerceiverLayer))]
        for layer in layers:
            layer.compile(**compile_kwargs)
        return len(layers)


# --- Backbone --------------------------------------------------------------------------------------
class TimberModel(TimberPreTrainedModel):
    """Tokenizer, contextualizer and Perceiver encoder: multichannel signal -> latent embeddings.

    Args:
        config: Model configuration.
        channel_embedding_init: Optional ``(num_channels, channel_embedding_dim)`` array, or path to a
            ``.npy`` file, to initialise the channel embedding.
            Only used when training from scratch: ``from_pretrained`` loads the trained embedding.

    """

    # Loading the backbone of a pretraining / prediction checkpoint drops its decoders and head
    _keys_to_ignore_on_load_unexpected = [r"^objectives\.", r"^head\."]  # noqa: RUF012 (transformers API)

    def __init__(
        self, config: TimberConfig, channel_embedding_init: np.ndarray | torch.Tensor | str | None = None
    ) -> None:
        super().__init__(config)
        self.tokenizer = PatchTokenizer(**config.tokenizer.module_kwargs())
        contextualizer = config.contextualizer
        self.contextualizer = Contextualizer(
            patch_length=config.tokenizer.patch_length,
            position_dim=config.tokenizer.position_dim,
            hidden_dim=contextualizer.hidden_dim,
            output_dim=contextualizer.output_dim,
            num_layers=contextualizer.num_layers,
            num_heads=contextualizer.num_heads,
            ffn_ratio=contextualizer.ffn_ratio,
            dropout=contextualizer.dropout,
            attention_order=contextualizer.attention_order,
            is_causal=contextualizer.is_causal,
            share_attention_weights=contextualizer.share_attention_weights,
            use_pre_norm=contextualizer.use_pre_norm,
            use_gradient_checkpointing=contextualizer.use_gradient_checkpointing,
        )
        self.encoder = None
        if config.encoder is not None:
            self.encoder = PerceiverEncoder(patch_dim=config.patch_dim, **_encoder_kwargs(config))
        self.post_init()
        if channel_embedding_init is not None:
            self.load_channel_embedding(channel_embedding_init)

    def load_channel_embedding(self, values: np.ndarray | torch.Tensor | str) -> None:
        """Overwrite the channel embedding with ``values`` (array or ``.npy`` path)."""
        values = np.load(values) if isinstance(values, str) else values
        values = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        weight = self.tokenizer.channel_embedding.weight
        if values.shape != weight.shape:
            msg = f"The channel embedding init must have shape {tuple(weight.shape)}, got {tuple(values.shape)}."
            raise ValueError(msg)
        with torch.no_grad():
            weight.copy_(values)

    def make_patch_mask(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> PatchMask:
        """Random mask of ``config.patch_mask_ratio`` of the ``num_patches`` (channel * patch) patches per row."""
        return random_patch_mask(
            batch_size, num_patches, self.config.patch_mask_ratio, device=device, generator=generator
        )

    def forward(
        self,
        signal: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        patch_mask: PatchMask | torch.Tensor | None = None,
        mask_patches: bool | None = None,
        num_encoder_blocks: int | None = None,
        output_attentions: bool = False,
        generator: torch.Generator | None = None,
    ) -> TimberModelOutput:
        """Embed multichannel signal windows.

        Args:
            signal: ``(*batch, channel, time)`` multichannel time series.
            channel_indices: ``(*batch, channel)`` channel index of every row (``None``: ``0 .. channel - 1``).
            timesteps: ``(*batch, time)`` absolute timesteps (needed for ``time_embedding_type="timestep"``).
            patch_mask: Explicit mask (a `PatchMask` or a ``(batch, channel * patch)`` keep mask).
            mask_patches: Draw a random mask (default: when ``config.patch_mask_ratio > 0``). Ignored
                when ``patch_mask`` is given. Use ``False`` to embed every patch at inference.
            num_encoder_blocks: Run only the first blocks of the encoder.
            output_attentions: Return the encoder cross-attention and last self-attention probabilities.
            generator: Random generator of the patch mask.

        """
        # Cut the signal into patches and build their position embeddings (channel, and time if enabled).
        tokens = self.tokenizer(signal, channel_indices, timesteps)
        position_embeddings = tokens.position_embeddings()
        batch_size, num_channels, num_patches, _ = tokens.patches.shape

        if patch_mask is not None and not isinstance(patch_mask, PatchMask):
            patch_mask = PatchMask.from_keep(patch_mask)
        # Draw the random mask of the pretraining (only when the ratio is positive, unless asked otherwise).
        if patch_mask is None and (self.config.patch_mask_ratio > 0 if mask_patches is None else mask_patches):
            patch_mask = self.make_patch_mask(batch_size, num_channels * num_patches, signal.device, generator)

        # Without an encoder, the contextualizer itself must hide the masked patches
        keep = None
        if self.encoder is None and patch_mask is not None:
            # The contextualizer hides the signal of masked patches here, because there is no encoder to drop them.
            keep = patch_mask.keep.view(batch_size, num_channels, num_patches)
        # A frozen front end (tokenizer and contextualizer, in finetuning) runs without autograd: no stored activations.
        front_is_frozen = not any(
            p.requires_grad for m in (self.tokenizer, self.contextualizer) for p in m.parameters()
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and not front_is_frozen):
            # The contextualised patches: the input of the encoder and of the TACO decoder.
            contextualized = self.contextualizer(tokens.patches, position_embeddings, patch_keep=keep)

        # Without an encoder, the contextualised patches are the representation (flattened over channel and patch).
        latents, block_latents, cross_attentions, self_attentions = contextualized.flatten(1, 2), None, None, None
        if self.encoder is not None:
            encoded = self.encoder(
                latents,
                # The encoder only reads the patches that are not masked.
                kept_indices=patch_mask.kept_indices if patch_mask is not None else None,
                num_blocks=num_encoder_blocks,
                need_weights=output_attentions,
            )
            latents, block_latents = encoded.latents, encoded.block_latents
            cross_attentions, self_attentions = encoded.cross_attentions, encoded.self_attentions

        return TimberModelOutput(
            last_hidden_state=latents,
            block_latents=block_latents,
            contextualized=contextualized,
            patches=tokens.patches,
            position_embeddings=position_embeddings,
            patch_keep=patch_mask.keep if patch_mask is not None else None,
            masked_indices=patch_mask.masked_indices if patch_mask is not None else None,
            cross_attentions=cross_attentions,
            self_attentions=self_attentions,
        )


def _encoder_kwargs(config: TimberConfig) -> dict:
    encoder = config.encoder
    return {
        "num_latents": encoder.num_latents,
        "latent_dim": encoder.latent_dim,
        "num_blocks": encoder.num_blocks,
        "num_self_attention_layers": encoder.num_self_attention_layers,
        "num_self_attention_heads": encoder.num_self_attention_heads,
        "num_cross_attention_heads": encoder.num_cross_attention_heads,
        "self_attention_widening_factor": encoder.self_attention_widening_factor,
        "cross_attention_widening_factor": encoder.cross_attention_widening_factor,
        "dropout": encoder.dropout,
        "use_cls_token": encoder.use_cls_token,
        "latent_dropout_ratio": encoder.latent_dropout_ratio,
        "use_gradient_checkpointing": encoder.use_gradient_checkpointing,
        "initializer_range": encoder.initializer_range,
    }


# --- Pretraining objectives ------------------------------------------------------------------------
class PatchProjectionDecoder(nn.Module):
    """No-encoder ablation: reconstruct every patch from its contextualised embedding (dropout + linear)."""

    def __init__(self, patch_dim: int, patch_length: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(patch_dim, patch_length)

    def forward(self, contextualized: torch.Tensor) -> torch.Tensor:
        """``(batch, channel, patch, patch_dim) -> (batch, channel * patch, patch_length)``."""
        return self.projection(self.dropout(contextualized.flatten(1, 2)))


class PretrainingObjectives(nn.Module):
    """The decoders of ``config.decoders`` and their losses."""

    def __init__(self, config: TimberConfig) -> None:
        super().__init__()
        self.config = config
        decoders = {}
        for decoder in config.decoders:
            if config.encoder is None:
                # No-encoder ablation: the decoder predicts the masked patches straight from the contextualised patches.
                decoders[decoder.objective] = PatchProjectionDecoder(
                    config.patch_dim,
                    config.tokenizer.patch_length,
                    decoder.dropout,
                )
                continue
            decoders[decoder.objective] = CrossAttentionDecoder(
                query_dim=config.tokenizer.position_dim if decoder.objective == "reconstruction" else config.patch_dim,
                latent_dim=config.encoder.latent_dim,
                output_dim=config.tokenizer.patch_length,
                num_heads=decoder.num_heads,
                widening_factor=decoder.widening_factor,
                dropout=decoder.dropout,
            )
        self.decoders = nn.ModuleDict(decoders)

    def _decode(self, decoder_config: DecoderConfig, outputs: TimberModelOutput) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self.decoders[decoder_config.objective]
        patches = _require(outputs.patches, "patches")
        contextualized = _require(outputs.contextualized, "contextualized patches")
        if self.config.encoder is None:
            masked_indices = _require(outputs.masked_indices, "masked patches")
            # Predict every patch, then keep only the masked ones for the loss.
            prediction = decoder(contextualized)
            return gather_patches(prediction, masked_indices), reconstruction_targets(patches, masked_indices)
        last_hidden_state = _require(outputs.last_hidden_state, "latents")
        latents = last_hidden_state[:, : self.config.encoder.num_latents]  # drop the CLS token
        # Reconstruction (ablation): the queries are the position embeddings of the masked patches.
        if decoder_config.objective == "reconstruction":
            if outputs.masked_indices is None:
                msg = "The reconstruction objective needs masked patches (patch_mask_ratio > 0 or a patch_mask)."
                raise ValueError(msg)
            position_embeddings = _require(outputs.position_embeddings, "position embeddings")
            queries = reconstruction_queries(position_embeddings, outputs.masked_indices)
            targets = reconstruction_targets(patches, outputs.masked_indices)
        else:
            # Distance in patches between a query and its target: the horizon, plus the offset of overlapping patches.
            shift = decoder_config.forecast_horizon + self.config.forecast_offset
            if shift >= patches.shape[2]:
                msg = f"{decoder_config.objective} needs more than {shift} patches per window."
                raise ValueError(msg)
            queries = forecasting_queries(contextualized, shift)
            targets = forecasting_targets(patches, shift)
        prediction, _ = decoder(queries, latents)
        return prediction, targets

    def forward(self, outputs: TimberModelOutput) -> tuple[dict, dict, dict]:
        """Return ``(loss_terms, predictions, targets)`` for every objective with a non-zero weight."""
        loss_terms, predictions, targets = {}, {}, {}
        for decoder_config in self.config.decoders:
            if decoder_config.loss_weight == 0:
                continue
            prediction, target = self._decode(decoder_config, outputs)
            name = decoder_config.objective
            predictions[name], targets[name] = prediction, target
            # Losses are computed in float32, whatever the mixed-precision dtype of the model.
            loss_terms[name] = F.mse_loss(prediction.float(), target.float())
        return loss_terms, predictions, targets

    def weighted_loss(self, loss_terms: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Sum of the objective losses, each weighted by its ``loss_weight``."""
        weights = {d.objective: d.loss_weight for d in self.config.decoders}
        terms = [weights[name] * value for name, value in loss_terms.items()]
        return reduce(operator.add, terms) if terms else None


class TimberForPretraining(TimberPreTrainedModel):
    """TIMBER with its pretraining decoders (TACO forecasting and/or masked reconstruction)."""

    def __init__(
        self, config: TimberConfig, channel_embedding_init: np.ndarray | torch.Tensor | str | None = None
    ) -> None:
        super().__init__(config)
        if not config.decoders:
            msg = "TimberForPretraining needs at least one decoder (config.decoders)."
            raise ValueError(msg)
        self.timber = TimberModel(config)
        self.objectives = PretrainingObjectives(config)
        self.post_init()
        if channel_embedding_init is not None:
            self.timber.load_channel_embedding(channel_embedding_init)

    def forward(
        self,
        signal: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        patch_mask: PatchMask | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TimberPretrainingOutput:
        """Compute the pretraining loss. Arguments as in `TimberModel.forward`."""
        outputs = self.timber(signal, channel_indices, timesteps, patch_mask=patch_mask, generator=generator)
        loss_terms, predictions, targets = self.objectives(outputs)
        return TimberPretrainingOutput(
            loss=self.objectives.weighted_loss(loss_terms),
            loss_terms=loss_terms,
            predictions=predictions,
            targets=targets,
            last_hidden_state=outputs.last_hidden_state,
            patch_keep=outputs.patch_keep,
        )


# --- Finetuning ------------------------------------------------------------------------------------
class TimberForPrediction(TimberPreTrainedModel):
    """TIMBER with a regression or classification head (``config.head``).

    The decoders of ``config.decoders`` (if any) are trained jointly, e.g. TACO as an auxiliary
    objective during finetuning.
    """

    def __init__(
        self, config: TimberConfig, channel_embedding_init: np.ndarray | torch.Tensor | str | None = None
    ) -> None:
        super().__init__(config)
        if config.head is None:
            msg = "TimberForPrediction needs a head (config.head)."
            raise ValueError(msg)
        self.timber = TimberModel(config)
        head = config.head
        self.head = PredictionHead(
            latent_dim=config.encoder.latent_dim,
            num_tokens=config.encoder.num_tokens,
            output_dim=head.num_outputs,
            pooling=head.pooling,
            fusion_stage=head.fusion_stage,
            projection_dim=head.projection_dim,
            pooling_input_dropout=head.pooling_input_dropout,
            num_hidden_layers=head.num_hidden_layers,
            widening_factor=head.widening_factor,
            dropout=head.dropout,
            use_input_norm=head.use_input_norm,
        )
        self.objectives = PretrainingObjectives(config) if config.decoders else None
        self.post_init()
        if channel_embedding_init is not None:
            self.timber.load_channel_embedding(channel_embedding_init)

    @property
    def latent_block(self) -> int:
        """Encoder block (1-based) whose latents feed the head; the last one by default."""
        return self.config.head.latent_block or self.config.encoder.num_blocks

    def head_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Loss of the head.

        ``labels`` is ``(entity, outputs)`` (regression) or ``(entity,)`` / ``(entity, 1)`` classes.
        """
        head = self.config.head
        if head.task == "classification":
            labels = labels.reshape(labels.shape[0], -1)[:, 0].long()
        labels = labels.repeat_interleave(logits.shape[0] // labels.shape[0], dim=0)  # loss-level fusion
        if head.task == "classification":
            return F.cross_entropy(logits.float(), labels, label_smoothing=head.label_smoothing)
        return F.huber_loss(logits.float(), labels.to(torch.float32))

    def forward(
        self,
        signal: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        num_entities: int | None = None,
        patch_mask: PatchMask | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        fusion_stage: str | None = None,
    ) -> TimberPredictionOutput:
        """Predict an entity-level target from multichannel windows.

        Args:
            signal: See `TimberModel.forward`.
            channel_indices: See `TimberModel.forward`.
            timesteps: See `TimberModel.forward`.
            patch_mask: See `TimberModel.forward`.
            generator: See `TimberModel.forward`.
            labels: ``(entity, outputs)`` regression targets, or class indices ``(entity,)`` / ``(entity, 1)``.
            num_entities: Entities in the batch (default: ``signal.shape[0]``; the remaining batch axes are views).
            fusion_stage: Override of ``config.head.fusion_stage`` for this call (the original validation used
                ``"late"`` to score entities).

        """
        num_entities = num_entities or (signal.shape[0] if signal.dim() > 2 else 1)
        num_blocks = None if self.objectives is not None else self.latent_block
        outputs = self.timber(
            signal,
            channel_indices,
            timesteps,
            patch_mask=patch_mask,
            num_encoder_blocks=num_blocks,
            generator=generator,
        )
        latents = outputs.block_latents[self.latent_block - 1]
        fusion = fusion_stage or self.config.head.fusion_stage
        head = self.head(latents, num_entities=num_entities, fusion_stage=fusion)

        view_logits = None if fusion == "early" else head["prediction"]
        logits = fuse_views(head["prediction"], num_entities) if fusion == "late" else head["prediction"]

        loss, loss_terms = None, {}
        if labels is not None:
            loss_terms[self.config.head.task] = self.head_loss(logits, labels)
            loss = self.config.head.loss_weight * loss_terms[self.config.head.task]
        if self.objectives is not None:
            objective_terms, _, _ = self.objectives(outputs)
            loss_terms.update(objective_terms)
            objective_loss = self.objectives.weighted_loss(objective_terms)
            if objective_loss is not None:
                loss = objective_loss if loss is None else loss + objective_loss

        return TimberPredictionOutput(
            loss=loss,
            loss_terms=loss_terms,
            logits=logits,
            view_logits=view_logits,
            pooled=head["pooled"],
            pooling_weights=head["pooling_weights"],
            last_hidden_state=latents,
        )
