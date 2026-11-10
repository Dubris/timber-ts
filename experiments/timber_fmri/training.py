"""Shared training entry point of the fMRI experiments (Hydra config -> Lightning run tracked in MLflow).

Experiment config (``experiments/<n>/config/config.yaml``)::

    name_experiment: 3_finetuning_hcp_intelligence
    name_run: split_1                     # default: number of training subjects
    seed: 42
    pretrained: ${path:results}/run/checkpoint.ckpt  # optional HF folder or .ckpt (converted if old format)
    resume_from: null                     # Lightning checkpoint to resume (MLflow run id from its path)
    discard: null                         # parameter prefixes not loaded from `pretrained`
    freeze: [timber.tokenizer, timber.contextualizer]
    channel_embedding_init: ${path:assets}/spectral_embeddings_64.npy   # training from scratch only
    model: {...}       # TimberConfig; with `pretrained`, overrides of the pretrained configuration
    training: {...}    # TimberTrainingConfig
    datamodule: {...}  # FmriDataModule options
    trainer: {...}     # lightning.Trainer arguments
    callbacks:
      checkpoint: {monitor: val/loss, mode: min, save_top_k: 1, save_last: true}
      early_stopping: null              # e.g. {monitor: val_auc, mode: max, patience: 3}
      plateau_early_stopping: null      # e.g. {monitor: val/loss, window: 3, min_rel_improvement: 0.005}
      periodic_checkpoint_every_n_epochs: null
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from timber import TimberConfig
from timber.device import default_precision
from timber.lightning import TimberFinetuningModule, TimberPretrainingModule

from .callbacks import PlateauRelativeEarlyStopping
from .configs import deep_merge
from .data import FmriDataModule
from .loading import load_config, load_pretrained_into
from .paths import PATHS


def _container(cfg: DictConfig, key: str, default: Any = None) -> Any:
    value = cfg.get(key, default)
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


logger = logging.getLogger(__name__)


def build_model_config(cfg: DictConfig) -> TimberConfig:
    """``cfg.model`` as a TimberConfig.

    With ``cfg.pretrained``, ``cfg.model`` holds overrides of the pretrained configuration.
    """
    overrides = _container(cfg, "model", {}) or {}
    if cfg.get("pretrained"):
        base = load_config(cfg.pretrained).architecture_dict()
        overrides = deep_merge(base, overrides)
    return TimberConfig(**overrides)


def build_callbacks(cfg: DictConfig, progress_bar: bool = True) -> list:
    callbacks_cfg = _container(cfg, "callbacks", {}) or {}
    checkpoint = {"monitor": "val/loss", "mode": "min", "save_top_k": 1, "save_last": True}
    checkpoint.update(callbacks_cfg.get("checkpoint") or {})
    callbacks = [ModelCheckpoint(**checkpoint), LearningRateMonitor(logging_interval="epoch")]
    if progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=10))
    if callbacks_cfg.get("early_stopping"):
        callbacks.append(EarlyStopping(**callbacks_cfg["early_stopping"]))
    if callbacks_cfg.get("plateau_early_stopping"):
        callbacks.append(PlateauRelativeEarlyStopping(**callbacks_cfg["plateau_early_stopping"]))
    every = callbacks_cfg.get("periodic_checkpoint_every_n_epochs")
    if every:
        callbacks.append(
            ModelCheckpoint(every_n_epochs=every, save_top_k=-1, save_last=False, save_on_train_epoch_end=True)
        )
    return callbacks


def train(cfg: DictConfig, tracking_uri: str | None = None) -> Trainer:
    """Train one experiment run described by ``cfg`` (see the module docstring).

    ``tracking_uri``: MLflow tracking URI (default: the ``mlruns`` folder of the paths file).
    """
    torch.set_float32_matmul_precision("medium")
    seed_everything(cfg.seed, workers=True)

    config = build_model_config(cfg)
    module_class = TimberFinetuningModule if config.head is not None else TimberPretrainingModule
    init = cfg.get("channel_embedding_init") if not cfg.get("pretrained") else None
    module = module_class(config, training=_container(cfg, "training", {}), channel_embedding_init=init)

    if cfg.get("pretrained"):
        report = load_pretrained_into(module, cfg.pretrained, discard=_container(cfg, "discard") or ())
        logger.info("Loaded %s: missing %s, unexpected %s", cfg.pretrained, report["missing"], report["unexpected"])
    if cfg.get("freeze"):
        frozen = module.freeze_parameters(_container(cfg, "freeze"))
        logger.info("Frozen %d parameter tensors.", len(frozen))

    datamodule = FmriDataModule(_container(cfg, "datamodule"))

    run_id = None
    if cfg.get("resume_from"):
        run_id = Path(cfg.resume_from).parts[-3]  # mlruns/<experiment>/<run>/checkpoints/<file>.ckpt
    mlflow_logger = MLFlowLogger(
        experiment_name=f"timber/{cfg.name_experiment}",
        run_name=cfg.get("name_run") or str(len(datamodule.train_subjects)),
        tracking_uri=tracking_uri or f"file:{PATHS.mlruns}",
        log_model=False,
        run_id=run_id,
    )
    config_file = Path(f"config_{mlflow_logger.run_id}.yaml")
    config_file.write_text(OmegaConf.to_yaml(cfg, resolve=True))  # the paths of this machine
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_file), artifact_path="config")
    config_file.unlink()

    trainer_kwargs = _container(cfg, "trainer", {}) or {}
    trainer_kwargs.setdefault("precision", default_precision(trainer_kwargs.get("accelerator")))
    callbacks = build_callbacks(cfg, progress_bar=trainer_kwargs.get("enable_progress_bar", True))
    trainer = Trainer(
        logger=mlflow_logger, callbacks=callbacks, max_epochs=module.training_config.max_epochs, **trainer_kwargs
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))
    return trainer
