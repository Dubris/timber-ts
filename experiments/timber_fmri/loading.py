"""Load TIMBER configurations, weights and models from any of the sources the experiments use.

A *source* is a Hugging Face folder or Hub id, or a checkpoint of the ``timber.lightning`` modules.
Checkpoints of the former implementation are not loaded.
"""

from __future__ import annotations

from pathlib import Path

import torch

from timber import TimberConfig, TimberForPrediction, TimberForPretraining, TimberModel
from timber.lightning import TimberFinetuningModule, TimberLightningModule, TimberPretrainingModule


def _raw_checkpoint(source: str | Path) -> dict | None:
    """The Lightning checkpoint of a ``.ckpt`` source (``None`` for other sources)."""
    path = Path(source)
    if path.suffix != ".ckpt":
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint(source: str | Path) -> dict | None:
    """Like `_raw_checkpoint`, refusing checkpoints of the former implementation."""
    checkpoint = _raw_checkpoint(source)
    if checkpoint is not None and any(key.startswith("backbone.") for key in checkpoint["state_dict"]):
        msg = f"{source} is a checkpoint of the former implementation, which this version does not load."
        raise ValueError(msg)
    return checkpoint


def load_config(source: str | Path) -> TimberConfig:
    """The model configuration of a source."""
    checkpoint = _checkpoint(source)
    if checkpoint is not None:
        return TimberConfig(**checkpoint["hyper_parameters"]["model_config"])
    return TimberConfig.from_pretrained(str(source))


def model_class_for(config: TimberConfig) -> type:
    if config.head is not None:
        return TimberForPrediction
    return TimberForPretraining if config.decoders else TimberModel


def load_model(source: str | Path):
    """The Hugging Face model of a source (eval mode, on CPU)."""
    checkpoint = _checkpoint(source)
    if checkpoint is not None:
        config = TimberConfig(**checkpoint["hyper_parameters"]["model_config"])
        module_class = TimberFinetuningModule if config.head is not None else TimberPretrainingModule
        return module_class.load_from_checkpoint(source, map_location="cpu").model.eval()
    config = TimberConfig.from_pretrained(str(source))
    return model_class_for(config).from_pretrained(str(source)).eval()


def load_pretrained_into(module: TimberLightningModule, source: str | Path, discard=()) -> dict[str, list[str]]:
    """Load pretrained weights into a Lightning module."""
    return module.load_pretrained(source, discard=discard)


def pretraining_subjects(source: str | Path) -> list[str]:
    """Train + validation subjects of the pretraining run that produced a checkpoint (empty if unknown)."""
    checkpoint = _raw_checkpoint(source)
    if checkpoint is None:
        return []
    hp = checkpoint.get("datamodule_hyper_parameters", {})
    train = hp.get("train_entities", hp.get("train_subjects", []))  # generic names of the saved data module
    val = hp.get("val_entities", hp.get("val_subjects", []))
    return list(train) + list(val)
