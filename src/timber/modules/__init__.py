"""Building blocks of TIMBER (pure ``torch.nn`` modules, no dependency on transformers internals).

Pipeline (paper notation): tokenizer -> contextualizer C -> Perceiver encoder E ->
decoders D (pretraining) or head H (finetuning).
"""

from .attention import FeedForward, MultiHeadAttention, PerceiverLayer
from .contextualizer import Contextualizer, ContextualizerLayer
from .decoder import (
    CrossAttentionDecoder,
    forecasting_queries,
    forecasting_targets,
    parse_forecast_horizon,
    reconstruction_queries,
    reconstruction_targets,
)
from .encoder import EncoderOutput, PerceiverEncoder, PerceiverEncoderBlock
from .heads import LatentPooling, MLPHead, PredictionHead, fuse_views
from .masking import PatchMask, gather_patches, num_masked_patches, random_patch_mask, token_dropout
from .tokenizer import PatchTokenizer, TokenizerOutput, forecast_offset, fourier_features

__all__ = [
    "Contextualizer",
    "ContextualizerLayer",
    "CrossAttentionDecoder",
    "EncoderOutput",
    "FeedForward",
    "LatentPooling",
    "MLPHead",
    "MultiHeadAttention",
    "PatchMask",
    "PatchTokenizer",
    "PerceiverEncoder",
    "PerceiverEncoderBlock",
    "PerceiverLayer",
    "PredictionHead",
    "TokenizerOutput",
    "forecast_offset",
    "forecasting_queries",
    "forecasting_targets",
    "fourier_features",
    "fuse_views",
    "gather_patches",
    "num_masked_patches",
    "parse_forecast_horizon",
    "random_patch_mask",
    "reconstruction_queries",
    "reconstruction_targets",
    "token_dropout",
]
