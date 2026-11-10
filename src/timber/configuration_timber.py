"""Configuration of the TIMBER models (architecture only).

`TimberConfig` is a `PretrainedConfig`: it is what
``config.json`` stores and what ``from_pretrained`` rebuilds. It is made of
nested dataclasses, one per module:

- `TokenizerConfig` (patching, channel and time embeddings),
- `ContextualizerConfig` (paper module C),
- `EncoderConfig` (paper module E; ``None`` for the no-encoder ablation),
- a list of `DecoderConfig` (pretraining objectives, paper module D),
- `HeadConfig` (finetuning head, paper module H; ``None`` for pretraining).

The architecture defaults are those of the TIMBER paper (TACO pretraining, ``forecasting_1``);
the data-shaped defaults (``num_channels``, ``patch_length``) are placeholders to set for your data.
Optimisation settings (learning rates, epochs, ...) are not part of this
configuration; they belong to the training code.

Every sub-configuration validates itself in ``__post_init__``; cross-module
rules are checked by `TimberConfig.validate`. Unknown keys raise an error.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from dataclasses import asdict, dataclass
from typing import Any, Literal

from transformers import PretrainedConfig

from .modules.decoder import parse_forecast_horizon
from .modules.tokenizer import forecast_offset

CONFIG_VERSION = 1

TimeEmbeddingType = Literal["patch", "timestep"]
AttentionOrder = Literal["channel_time", "time_channel"]
Task = Literal["regression", "classification"]
Pooling = Literal["mean", "weighted_mean", "cls_token", "attention"]
FusionStage = Literal["early", "late", "loss"]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_choice(name: str, value: Any, choices: tuple) -> None:
    _check(value in choices, f"{name} must be one of {choices}, got {value!r}.")


@dataclass
class TokenizerConfig:
    """Patching and position embeddings.

    Attributes:
        patch_length: Samples per patch.
        patch_stride: Step between patch starts (``None``: ``patch_length``, no overlap).
        num_channels: Number of channels with a channel embedding (channel indices are in ``[0, num_channels)``).
        channel_embedding_dim: Channels of the channel embedding.
        trainable_channel_embedding: Train the channel embedding.
        time_embedding_type: ``None``, ``"patch"`` (index of the patch in the window) or
            ``"timestep"`` (absolute timestep of the patch start).
        num_time_frequencies: Fourier bands of the time embedding (it has ``2 * k + 1`` channels).
        time_resolution: Fourier resolution; with ``"timestep"`` also the number of timesteps covered.

    """

    patch_length: int = 16
    patch_stride: int | None = None
    num_channels: int = 1024
    channel_embedding_dim: int = 64
    trainable_channel_embedding: bool = True
    time_embedding_type: TimeEmbeddingType | None = None
    num_time_frequencies: int = 0
    time_resolution: int = 100

    def __post_init__(self) -> None:
        _check(self.patch_length >= 1, "patch_length must be >= 1.")
        _check(self.patch_stride is None or self.patch_stride >= 1, "patch_stride must be >= 1.")
        _check(
            self.num_channels >= 1 and self.channel_embedding_dim >= 1,
            "num_channels and channel_embedding_dim must be >= 1.",
        )
        _check_choice("time_embedding_type", self.time_embedding_type, (None, "patch", "timestep"))
        if self.time_embedding_type is not None:
            _check(self.num_time_frequencies >= 1, "A time embedding needs num_time_frequencies >= 1.")
            _check(self.time_resolution >= 2, "time_resolution must be >= 2.")

    @property
    def stride(self) -> int:
        """Hop between patches: ``patch_stride``, or ``patch_length`` when it is ``None``."""
        return self.patch_length if self.patch_stride is None else self.patch_stride

    @property
    def time_embedding_dim(self) -> int:
        """Width of the time embedding: ``2 * num_time_frequencies + 1``, or 0 without it."""
        return 2 * self.num_time_frequencies + 1 if self.time_embedding_type is not None else 0

    @property
    def position_dim(self) -> int:
        """Channels of a patch position embedding (channel + time)."""
        return self.channel_embedding_dim + self.time_embedding_dim

    @property
    def forecast_offset(self) -> int:
        """Extra shift of forecasting targets for overlapping patches."""
        return forecast_offset(self.patch_length, self.stride)

    def module_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of `PatchTokenizer`."""
        kwargs = asdict(self)
        if self.time_embedding_type is None:
            kwargs["num_time_frequencies"] = 0
        return kwargs


@dataclass
class ContextualizerConfig:
    """Factorised spatial / temporal attention over patches (paper module C).

    Attributes:
        hidden_dim: Channels of the layers.
        output_dim: Channels of the output projection (``None``: no projection, ``hidden_dim`` channels).
        num_layers: Number of layers (``0``: patch embedding only, the "no contextualizer" ablation).
        num_heads: Attention heads.
        ffn_ratio: Feed-forward hidden size as a multiple of ``hidden_dim``.
        dropout: Dropout on attention probabilities, feed-forward and residual branches.
        attention_order: ``"channel_time"`` (channels first) or ``"time_channel"`` (patches first).
        is_causal: Causal attention over time (required by the forecasting objectives).
        share_attention_weights: One attention module for both passes.
        use_pre_norm: Pre-norm residual blocks (default: post-norm).
        use_gradient_checkpointing: Recompute layers in the backward pass to save memory.

    """

    hidden_dim: int = 128
    output_dim: int | None = 128
    num_layers: int = 3
    num_heads: int = 4
    ffn_ratio: int = 2
    dropout: float = 0.1
    attention_order: AttentionOrder = "channel_time"
    is_causal: bool = True
    share_attention_weights: bool = True
    use_pre_norm: bool = False
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        _check(self.num_layers >= 0, "num_layers must be >= 0.")
        _check(self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads.")
        _check_choice("attention_order", self.attention_order, ("channel_time", "time_channel"))
        _check(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1).")

    @property
    def patch_dim(self) -> int:
        """Channels of the contextualised patches."""
        return self.hidden_dim if self.output_dim is None else self.output_dim


@dataclass
class EncoderConfig:
    """Perceiver IO encoder (paper module E).

    Attributes:
        num_latents: Latent tokens.
        latent_dim: Channels of the latents.
        num_blocks: Encoder blocks (each with its own cross-attention to the patches).
        num_self_attention_layers: Latent self-attention layers per block.
        num_self_attention_heads, num_cross_attention_heads: Attention heads.
        self_attention_widening_factor, cross_attention_widening_factor: MLP widening factors.
        dropout: Dropout on the attention probabilities.
        use_cls_token: Append a trainable CLS token after the latents.
        latent_dropout_ratio: Probability to drop each latent token before every block (training only).
        initializer_range: Standard deviation of the CLS token initialisation.
        use_gradient_checkpointing: Recompute blocks in the backward pass to save memory.

    """

    num_latents: int = 512
    latent_dim: int = 256
    num_blocks: int = 2
    num_self_attention_layers: int = 4
    num_self_attention_heads: int = 1
    num_cross_attention_heads: int = 1
    self_attention_widening_factor: int = 1
    cross_attention_widening_factor: int = 1
    dropout: float = 0.1
    use_cls_token: bool = True
    latent_dropout_ratio: float = 0.0
    initializer_range: float = 0.02
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        _check(self.num_latents >= 1 and self.num_blocks >= 1, "num_latents and num_blocks must be >= 1.")
        for heads in (self.num_self_attention_heads, self.num_cross_attention_heads):
            _check(self.latent_dim % heads == 0, "latent_dim must be divisible by the number of heads.")
        _check(0.0 <= self.latent_dropout_ratio < 1.0, "latent_dropout_ratio must be in [0, 1).")

    @property
    def num_tokens(self) -> int:
        """Output tokens: latents plus the CLS token."""
        return self.num_latents + int(self.use_cls_token)


@dataclass
class DecoderConfig:
    """A pretraining objective and its cross-attention decoder (paper module D).

    Attributes:
        objective: ``"reconstruction"`` (masked patches from their positions) or ``"forecasting_<k>"``
            (the patch ``k`` steps ahead from a contextualised patch; ``forecasting_1`` is TACO).
        loss_weight: Weight of this objective in the total loss (``0`` skips it).
        num_heads: Cross-attention heads.
        widening_factor: MLP widening of the cross-attention layer.
        dropout: Dropout on the attention probabilities (for the no-encoder ablation: dropout
            before its linear projection).

    """

    objective: str = "forecasting_1"
    loss_weight: float = 1.0
    num_heads: int = 1
    widening_factor: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.objective != "reconstruction":
            parse_forecast_horizon(self.objective)
        _check(self.loss_weight >= 0.0, "loss_weight must be >= 0.")

    @property
    def is_forecasting(self) -> bool:
        """Whether the objective is a forecasting objective (``forecasting_<k>``)."""
        return self.objective != "reconstruction"

    @property
    def forecast_horizon(self) -> int:
        """The horizon ``k`` of a forecasting objective, or ``None`` for reconstruction."""
        return parse_forecast_horizon(self.objective) if self.is_forecasting else 0


@dataclass
class HeadConfig:
    """Finetuning head (paper module H).

    Attributes:
        task: ``"regression"`` (Huber loss) or ``"classification"`` (cross-entropy).
        num_outputs: Regression targets, or number of classes.
        pooling: ``"mean"``, ``"weighted_mean"``, ``"cls_token"`` or ``"attention"`` pooling of the latents.
        fusion_stage: Where the views of an entity are combined: ``"early"`` (pooled latents),
            ``"late"`` (predictions) or ``"loss"`` (one loss term per view).
        projection_dim: Optional ``Linear + GELU`` projection of the latents before pooling.
        pooling_input_dropout: Dropout on the latents before pooling.
        num_hidden_layers, widening_factor, dropout, use_input_norm: MLP layout.
        label_smoothing: Label smoothing of the classification loss.
        loss_weight: Weight of the head loss (relevant when decoders are also trained).
        latent_block: Read the latents of this encoder block (1-based; ``None``: last block).

    """

    task: Task = "regression"
    num_outputs: int = 1
    pooling: Pooling = "mean"
    fusion_stage: FusionStage = "late"
    projection_dim: int | None = None
    pooling_input_dropout: float = 0.0
    num_hidden_layers: int = 0
    widening_factor: float = 1.0
    dropout: float = 0.5
    use_input_norm: bool = False
    label_smoothing: float = 0.0
    loss_weight: float = 1.0
    latent_block: int | None = None

    def __post_init__(self) -> None:
        _check_choice("task", self.task, ("regression", "classification"))
        _check_choice("pooling", self.pooling, ("mean", "weighted_mean", "cls_token", "attention"))
        _check_choice("fusion_stage", self.fusion_stage, ("early", "late", "loss"))
        _check(self.num_outputs >= 1, "num_outputs must be >= 1.")
        if self.task == "classification":
            _check(self.num_outputs >= 2, "Classification needs num_outputs (classes) >= 2.")
        _check(self.latent_block is None or self.latent_block >= 1, "latent_block is 1-based.")


_SUB_CONFIGS = {
    "tokenizer": TokenizerConfig,
    "contextualizer": ContextualizerConfig,
    "encoder": EncoderConfig,
    "head": HeadConfig,
}


class _Default(enum.Enum):
    """Distinguishes "not given" from an explicit ``None`` (no encoder)."""

    TOKEN = enum.auto()


_DEFAULT = _Default.TOKEN


@functools.cache
def _base_config_keys() -> frozenset[str]:
    """Keys a plain PretrainedConfig accepts or writes (``config.json`` metadata, generation defaults, ...)."""
    extra = {
        "architectures",
        "auto_map",
        "transformers_version",
        "_name_or_path",
        "model_type",
        "torch_dtype",
        "_attn_implementation",
        "_attn_implementation_autoset",
        "_commit_hash",
        "attn_implementation",
        "name_or_path",
        "return_unused_kwargs",
        "trust_remote_code",
    }
    return frozenset(PretrainedConfig().to_dict()) | frozenset(extra)


def _build(name: str, cls: type, value: Any) -> Any:
    if value is None or isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        msg = f"{name} must be a {cls.__name__}, a dict or None, got {type(value).__name__}."
        raise TypeError(msg)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(value) - known)
    if unknown:
        msg = f"Unknown {name} option(s) {unknown}. Valid options: {sorted(known)}."
        raise ValueError(msg)
    return cls(**value)


class TimberConfig(PretrainedConfig):
    """Configuration of the TIMBER models.

    Args:
        tokenizer: `TokenizerConfig` (or dict).
        contextualizer: `ContextualizerConfig` (or dict).
        encoder: `EncoderConfig` (or dict), or ``None`` for the no-encoder ablation.
        decoders: Pretraining objectives, a list of `DecoderConfig` (or dicts). Empty for
            a plain finetuning model.
        head: `HeadConfig` (or dict), or ``None`` for pretraining.
        patch_mask_ratio: Fraction of patches hidden from the encoder (``0``: no masking).
        config_version: Version of this configuration schema (for later migrations).
        **kwargs: Standard `PretrainedConfig` arguments.

    Example::

        config = TimberConfig(tokenizer={"num_channels": 64, "patch_length": 16})  # TACO pretraining
        finetuning = TimberConfig(decoders=[], head={"task": "regression", "num_outputs": 3}, patch_mask_ratio=0.0)

    """

    model_type = "timber"
    _architecture_keys = ("tokenizer", "contextualizer", "encoder", "decoders", "head", "patch_mask_ratio")
    _sub_config_keys = (*_architecture_keys, "config_version")

    def __init__(
        self,
        tokenizer: TokenizerConfig | dict | None = None,
        contextualizer: ContextualizerConfig | dict | None = None,
        encoder: EncoderConfig | dict | None | Literal[_Default.TOKEN] = _DEFAULT,
        decoders: list[DecoderConfig | dict] | None = None,
        head: HeadConfig | dict | None = None,
        patch_mask_ratio: float = 0.7,
        config_version: int = CONFIG_VERSION,
        **kwargs: Any,
    ) -> None:
        unknown = sorted(set(kwargs) - _base_config_keys())
        if unknown:
            msg = f"Unknown TimberConfig option(s) {unknown}. Model options: {list(self._sub_config_keys)}."
            raise ValueError(msg)
        super().__init__(**kwargs)
        self.tokenizer = _build("tokenizer", TokenizerConfig, tokenizer) or TokenizerConfig()
        self.contextualizer = _build("contextualizer", ContextualizerConfig, contextualizer) or ContextualizerConfig()
        self.encoder = EncoderConfig() if encoder is _DEFAULT else _build("encoder", EncoderConfig, encoder)
        if decoders is None:
            decoders = [DecoderConfig()]
        self.decoders = [d if isinstance(d, DecoderConfig) else _build("decoder", DecoderConfig, d) for d in decoders]
        self.head = _build("head", HeadConfig, head)
        self.patch_mask_ratio = patch_mask_ratio
        self.config_version = config_version
        self.validate()

    def __setattr__(self, name: str, value: Any) -> None:
        # Overrides such as from_pretrained(..., head={...}) set attributes directly: convert dicts
        if name in _SUB_CONFIGS and isinstance(value, dict):
            value = _build(name, _SUB_CONFIGS[name], value)
        elif name == "decoders" and value is not None:
            value = [d if isinstance(d, DecoderConfig) else _build("decoder", DecoderConfig, d) for d in value]
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        """Build from a dict (``config.json``) and apply overrides, then validate the result."""
        result = super().from_dict(config_dict, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.validate()
        return result

    # --- Derived values ---------------------------------------------------------------------------
    @property
    def patch_dim(self) -> int:
        """Channels of the contextualised patches (input of the encoder)."""
        return self.contextualizer.patch_dim

    @property
    def forecast_offset(self) -> int:
        """Sample offset of the forecasting targets (see `forecast_offset`)."""
        return self.tokenizer.forecast_offset

    @property
    def objectives(self) -> list[str]:
        """The pretraining objectives built from ``decoders``."""
        return [decoder.objective for decoder in self.decoders]

    # --- Validation -------------------------------------------------------------------------------
    def validate(self) -> None:
        """Check the rules that involve several modules."""
        _check(0.0 <= self.patch_mask_ratio < 1.0, "patch_mask_ratio must be in [0, 1).")
        _check(len(set(self.objectives)) == len(self.objectives), f"Duplicate objectives: {self.objectives}.")
        for decoder in self.decoders:
            if decoder.objective == "reconstruction":
                _check(self.patch_mask_ratio > 0, "The reconstruction objective needs patch_mask_ratio > 0.")
                _check(
                    self.tokenizer.time_embedding_type is not None,
                    "The reconstruction objective needs a time embedding (tokenizer.time_embedding_type).",
                )
        if self.encoder is None:
            _check(
                self.objectives == ["reconstruction"] and self.head is None,
                "Without an encoder, only the reconstruction objective is supported (no-encoder ablation).",
            )
        if self.head is not None:
            if self.encoder is None:
                msg = "A head needs an encoder."
                raise ValueError(msg)
            if self.head.pooling == "cls_token":
                _check(self.encoder.use_cls_token, "pooling='cls_token' needs encoder.use_cls_token=True.")
            if self.head.latent_block is not None:
                _check(
                    self.head.latent_block <= self.encoder.num_blocks, "head.latent_block exceeds encoder.num_blocks."
                )
        _check(bool(self.decoders) or self.head is not None, "The model needs at least one decoder or a head.")

    # --- Serialisation ----------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The configuration as a nested dictionary (the content of ``config.json``)."""
        output = super().to_dict()
        for name in ("tokenizer", "contextualizer", "encoder", "head"):
            value = getattr(self, name)
            output[name] = asdict(value) if value is not None else None
        output["decoders"] = [asdict(decoder) for decoder in self.decoders]
        return output

    def architecture_dict(self) -> dict[str, Any]:
        """The model options only (nested dicts), without the transformers bookkeeping fields.

        ``TimberConfig(**config.architecture_dict())`` rebuilds an equivalent configuration.
        """
        full = self.to_dict()
        return {key: full[key] for key in self._architecture_keys}

    def to_diff_dict(self) -> dict[str, Any]:
        """Like the base class, but always write the full model configuration (readable ``config.json``)."""
        output = super().to_diff_dict()
        full = self.to_dict()
        output.update({key: full[key] for key in self._sub_config_keys})
        return output
