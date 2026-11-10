"""TIMBER configurations of the fMRI application (cortical resting-state fMRI, fsLR 32k)."""

from __future__ import annotations

from pathlib import Path

from timber import TimberConfig

from .data import NUM_CORTICAL_VERTICES
from .paths import PATHS


def spectral_embeddings_path(dim: int = 64) -> Path:
    """Spectral embeddings of the cortical mesh, used to initialise the channel embedding."""
    return PATHS.assets / f"spectral_embeddings_{dim}.npy"


def paper_config(**overrides) -> TimberConfig:
    """The paper's pretraining setting: TACO on 59412 cortical vertices, 60-sample patches, no time embedding.

    Use ``TimberForPretraining(paper_config(), channel_embedding_init=spectral_embeddings_path())``
    to start from the spectral initialisation of the paper.
    """
    config = {
        "tokenizer": {"patch_length": 60, "num_channels": NUM_CORTICAL_VERTICES, "channel_embedding_dim": 64},
        "contextualizer": {
            "hidden_dim": 128,
            "output_dim": 128,
            "num_layers": 3,
            "num_heads": 4,
            "ffn_ratio": 2,
            "dropout": 0.1,
            "attention_order": "channel_time",
            "is_causal": True,
        },
        "encoder": {
            "num_latents": 512,
            "latent_dim": 256,
            "num_blocks": 2,
            "num_self_attention_layers": 4,
            "use_cls_token": True,
        },
        "decoders": [{"objective": "forecasting_1"}],
        "patch_mask_ratio": 0.7,
    }
    config.update(overrides)
    return TimberConfig(**config)
