"""Shared Lightning logic: training/validation steps, optimiser groups, schedule, pretrained loading.

The Lightning modules are thin wrappers around the Hugging Face models of
`modeling_timber` (``self.model``). Their checkpoints store the model
configuration and the `TimberTrainingConfig` as hyper-parameters, so
``load_from_checkpoint`` rebuilds them, and `TimberLightningModule.export_pretrained`
writes a Hugging Face folder.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch.optim.lr_scheduler import LambdaLR

from ..configuration_timber import TimberConfig
from ..data.datamodule import SamplePositions


@dataclass
class TimberTrainingConfig:
    """Optimisation settings (saved in the Lightning checkpoint, not in ``config.json``).

    Attributes:
        learning_rate: Learning rate of the backbone and the decoders.
        weight_decay: AdamW weight decay (biases and norm weights are not decayed).
        head_learning_rate: Learning rate of the finetuning head (``None``: ``5 * learning_rate``).
        head_weight_decay: Weight decay of the head (``None``: ``weight_decay``).
        latents_learning_rate: Learning rate of the encoder latents and CLS token, which then get no
            weight decay. ``None``: ``learning_rate / 5`` when finetuning; when pretraining the latents
            are ordinary backbone parameters.
        warmup_ratio: Fraction of the steps with a linear warm-up, followed by a cosine decay.
        max_epochs: Number of epochs (passed to the Lightning ``Trainer``).
        grad_norm_log_interval: Log the gradient norms every N optimiser steps (``0``: never).
        compile: ``torch.compile`` the attention layers (`TimberPreTrainedModel.compile_layers`).

    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    head_learning_rate: float | None = None
    head_weight_decay: float | None = None
    latents_learning_rate: float | None = None
    warmup_ratio: float = 0.1
    max_epochs: int = 30
    grad_norm_log_interval: int = 50
    compile: bool = False

    @classmethod
    def from_dict(cls, values: dict | TimberTrainingConfig | None) -> TimberTrainingConfig:
        """Build a config from a mapping; unknown keys raise."""
        if values is None:
            return cls()
        if isinstance(values, TimberTrainingConfig):
            return values
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            msg = f"Unknown training option(s) {unknown}. Valid options: {sorted(known)}."
            raise ValueError(msg)
        return cls(**values)


def cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warm-up then half-cosine decay to 0 (same as ``transformers.get_cosine_schedule_with_warmup``)."""

    # The factor multiplies the base learning rate: a linear ramp up, then a cosine decay to zero at the last step.
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, factor)


def is_no_decay(name: str) -> bool:
    """Biases and normalisation weights are not decayed."""
    # The module name is the second-to-last part of the parameter name (e.g. ``query_norm`` in
    # ``...query_norm.weight``).
    return name.endswith(".bias") or "norm" in name.split(".")[-2]


def matches(name: str, patterns: Iterable[str]) -> bool:
    """Whether a parameter name starts with one of the patterns (module prefixes)."""
    # A pattern matches a module and everything below it: ``timber.encoder.blocks.0`` does not match ``blocks.01``.
    return any(name == p or name.startswith(p if p.endswith(".") else p + ".") for p in patterns)


class TimberLightningModule(LightningModule):
    """Base class of the TIMBER Lightning modules.

    Args:
        model_config: `TimberConfig` (or its dict).
        training: `TimberTrainingConfig` (or dict).
        channel_embedding_init: Optional initial channel embedding (array or ``.npy`` path), not saved.

    """

    model_class: type
    latents_in_separate_group = False

    def __init__(
        self,
        model_config: TimberConfig | dict,
        training: TimberTrainingConfig | dict | None = None,
        channel_embedding_init: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(model_config, dict):
            model_config = TimberConfig(**{k: v for k, v in model_config.items() if k in TimberConfig._sub_config_keys})
        self.training_config = TimberTrainingConfig.from_dict(training)
        self.frozen_patterns: list[str] = []
        self.model = self.model_class(model_config, channel_embedding_init=channel_embedding_init)
        if self.training_config.compile:
            self.model.compile_layers()
        self.save_hyperparameters(
            {
                "model_config": {k: model_config.to_dict()[k] for k in TimberConfig._sub_config_keys},
                "training": dataclasses.asdict(self.training_config),
            }
        )

    @property
    def config(self) -> TimberConfig:
        """The model configuration (`TimberConfig`)."""
        return self.model.config

    # --- Steps --------------------------------------------------------------------------------------
    def forward_batch(self, batch: tuple, stage: str):
        """Run the model on a batch and return its output."""
        raise NotImplementedError

    def loss_weights(self) -> dict[str, float]:
        """Loss weight of every objective, by decoder name."""
        weights = {d.objective: d.loss_weight for d in self.config.decoders}
        if self.config.head is not None:
            weights[self.config.head.task] = self.config.head.loss_weight
        return weights

    @staticmethod
    def log_name(term: str) -> str:
        """Name of a logged metric: ``{stage}/{name}``."""
        return "prediction" if term == "regression" else term  # metric name of the first logs

    def _step(self, batch: tuple, stage: str) -> torch.Tensor:
        output = self.forward_batch(batch, stage)
        weights = self.loss_weights()
        on_step = stage == "train"
        batch_size = batch[0].shape[0]
        for name, value in output.loss_terms.items():
            self.log(
                # Log every objective with its weight, so that the logged terms add up to the total loss.
                f"{stage}/loss_{self.log_name(name)}",
                weights[name] * value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        self.log(f"{stage}/loss", output.loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return output.loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One optimisation step: logs the losses and returns the total loss."""
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One validation step: logs the losses and the metrics."""
        return self._step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One test step: logs the losses and the metrics."""
        return self._step(batch, "test")

    @staticmethod
    def unpack(batch: tuple) -> tuple[torch.Tensor, SamplePositions, torch.Tensor | None]:
        """Split a batch into the signal, the positions and the labels."""
        signal, positions, labels = batch
        if not isinstance(positions, SamplePositions):  # plain tuple of positions
            positions = SamplePositions(*positions)
        return signal, positions, labels

    # --- Epochs -------------------------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        """Give the epoch to the sampler (also when it is wrapped in a BatchSampler)."""
        loader = self.trainer.train_dataloader
        for sampler in (
            getattr(loader, "sampler", None),
            getattr(getattr(loader, "batch_sampler", None), "sampler", None),
        ):
            set_epoch = getattr(sampler, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(self.current_epoch)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log the gradient norm every ``grad_norm_log_interval`` steps."""
        interval = self.training_config.grad_norm_log_interval
        if interval and self.global_step % interval == 0:
            self.log_dict(grad_norm(self, norm_type=2))

    # --- Optimisation -------------------------------------------------------------------------------
    def parameter_roles(self) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
        """Trainable parameters by role: ``backbone`` (incl. decoders), ``latents``, ``head``."""
        roles: dict[str, list] = {"backbone": [], "latents": [], "head": []}
        latents = {"timber.encoder.latents", "timber.encoder.cls_token"}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # The latent queries get their own group when they have their own learning rate (finetuning).
            if name in latents and (self.latents_in_separate_group or self.training_config.latents_learning_rate):
                roles["latents"].append((name, param))
            # Everything else is backbone (tokenizer, contextualizer, encoder, decoders).
            elif name.startswith("head."):
                roles["head"].append((name, param))
            else:
                roles["backbone"].append((name, param))
        return roles

    def parameter_groups(self) -> list[dict]:
        """Optimiser parameter groups: backbone, head and latents, with their learning rates."""
        training = self.training_config
        lr = training.learning_rate
        settings = {
            "backbone": (lr, training.weight_decay),
            # Defaults follow the paper's finetuning: the head learns 5 times faster than the backbone, the latents 5
            # times slower.
            "head": (
                training.head_learning_rate or 5 * lr,
                training.weight_decay if training.head_weight_decay is None else training.head_weight_decay,
            ),
            "latents": (training.latents_learning_rate or lr / 5, 0.0),
        }
        groups = []
        for role, named in self.parameter_roles().items():
            group_lr, decay = settings[role]
            # Each role is split in two groups: parameters with weight decay, and biases, norms and latents without it.
            decayed = [p for n, p in named if role != "latents" and not is_no_decay(n)]
            not_decayed = [p for n, p in named if role == "latents" or is_no_decay(n)]
            for params, weight_decay, suffix in ((decayed, decay, "decay"), (not_decayed, 0.0, "no_decay")):
                if params:
                    groups.append(
                        {"params": params, "lr": group_lr, "weight_decay": weight_decay, "name": f"{role}_{suffix}"}
                    )
        grouped = {id(p) for g in groups for p in g["params"]}
        missing = [n for n, p in self.named_parameters() if p.requires_grad and id(p) not in grouped]
        if missing:
            # Guard: a trainable parameter outside every group would never be updated, so fail loudly.
            msg = f"Trainable parameters without an optimiser group: {missing}"
            raise RuntimeError(msg)
        return groups

    def configure_optimizers(self) -> dict:
        # Fused AdamW on accelerators, except with gradient clipping: Lightning refuses to clip the gradients
        # of a fused optimizer under mixed precision (the default foreach implementation is used instead)
        """AdamW with a linear warm-up then a cosine decay (``warmup_ratio``)."""
        clipping = bool(self.trainer.gradient_clip_val)
        fused = self.device.type in ("cuda", "mps") and not clipping
        optimizer = torch.optim.AdamW(self.parameter_groups(), fused=fused or None)
        # The schedule covers the whole run: the warm-up is a fraction of all the optimiser steps.
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(self.training_config.warmup_ratio * total_steps)
        scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    # --- Pretrained weights, freezing and export ----------------------------------------------------
    def load_pretrained(
        self,
        source: str | Path,
        discard: Sequence[str] = (),
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Load pretrained weights into ``self.model``.

        Args:
            source: A Hugging Face folder or Hub id, or a checkpoint of these Lightning modules.
            discard: Parameter-name prefixes (of ``self.model``) not to load.
            strict: Fail on missing or unexpected keys.

        Returns:
            ``{"missing": [...], "unexpected": [...]}``.

        """
        state_dict = load_pretrained_state_dict(source)
        own = self.model.state_dict()
        # Drop the discarded prefixes (e.g. the pretraining decoders) before loading.
        state_dict = {k: v for k, v in state_dict.items() if not matches(k, discard)}
        mismatched = [k for k, v in state_dict.items() if k in own and own[k].shape != v.shape]
        if mismatched:
            msg = f"Shape mismatch for {mismatched[:5]}: discard these keys (discard=...)."
            raise ValueError(msg)
        report = self.model.load_state_dict(state_dict, strict=strict)
        return {"missing": list(report.missing_keys), "unexpected": list(report.unexpected_keys)}

    def freeze_parameters(self, patterns: Sequence[str]) -> list[str]:
        """Freeze the parameters of ``self.model`` whose names start with one of ``patterns``.

        Patterns are module prefixes, e.g. ``timber.tokenizer``, ``timber.contextualizer``,
        ``timber.encoder.blocks.0``. The frozen modules also stay in eval mode during training (no
        dropout), see `train`. Returns the names of the frozen parameters.
        """
        frozen = []
        for name, param in self.model.named_parameters():
            if matches(name, patterns):
                param.requires_grad_(False)
                frozen.append(name)
        self.frozen_patterns = sorted({*self.frozen_patterns, *patterns})
        self.train(self.training)
        return frozen

    def frozen_modules(self) -> list[torch.nn.Module]:
        """The modules of ``self.model`` frozen by `freeze_parameters`."""
        return [module for name, module in self.model.named_modules() if name and matches(name, self.frozen_patterns)]

    def train(self, mode: bool = True):
        """Set the training mode, keeping the frozen modules in eval mode (fixed features, no dropout)."""
        super().train(mode)
        if mode:
            for module in self.frozen_modules():
                module.eval()
        return self

    def export_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        """Save ``self.model`` as a Hugging Face folder (``config.json`` + ``model.safetensors``)."""
        self.model.save_pretrained(output_dir, **kwargs)


def load_pretrained_state_dict(source: str | Path) -> dict[str, torch.Tensor]:
    """State dict in the layout of the new models (``timber.*``, ``objectives.*``, ``head.*``)."""
    path = Path(source)
    if path.suffix == ".ckpt":
        state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        if not any(key.startswith("model.") for key in state_dict):
            msg = f"{path} is not a checkpoint of the timber Lightning modules."
            raise ValueError(msg)
        return {key.removeprefix("model."): value for key, value in state_dict.items()}

    from safetensors.torch import load_file  # noqa: PLC0415
    from transformers.utils import cached_file  # noqa: PLC0415

    if path.is_dir():
        weights = path / "model.safetensors"
    else:
        downloaded = cached_file(str(source), "model.safetensors")
        if downloaded is None:
            msg = f"No model.safetensors in {source}."
            raise FileNotFoundError(msg)
        weights = Path(downloaded)
    state_dict = load_file(weights)
    if not any(key.startswith("timber.") for key in state_dict):  # a TimberModel (backbone only) was saved
        state_dict = {f"timber.{key}": value for key, value in state_dict.items()}
    return state_dict
