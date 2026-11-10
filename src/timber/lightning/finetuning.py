"""Lightning module for finetuning (`TimberForPrediction`)."""

from __future__ import annotations

import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

from ..modeling_timber import TimberForPrediction, TimberPredictionOutput
from .base import TimberLightningModule


class TimberFinetuningModule(TimberLightningModule):
    """Finetuning of a regression or classification head (optionally with decoders, e.g. TACO).

    Behaviour kept from the original implementation:

    - validation and test use late fusion (entity-level predictions), whatever ``fusion_stage``;
    - default learning rates: head ``5 x``, latents and CLS token ``/ 5`` (no weight decay);
    - epoch metrics ``{train,val}_mae``, ``_mse``, ``_pearson`` (mean over outputs) for regression and
      ``{train,val}_auc``, ``_accuracy`` for classification.
    """

    model_class = TimberForPrediction
    latents_in_separate_group = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        head = self.config.head
        for stage in ("train", "val"):
            if head.task == "regression":
                metrics = {
                    "mae": MeanAbsoluteError(num_outputs=head.num_outputs),
                    "mse": MeanSquaredError(num_outputs=head.num_outputs),
                    "pearson": PearsonCorrCoef(num_outputs=head.num_outputs),
                }
            elif head.num_outputs == 2:
                metrics = {"auc": BinaryAUROC(), "accuracy": BinaryAccuracy()}
            else:
                metrics = {"auc": MulticlassAUROC(head.num_outputs), "accuracy": MulticlassAccuracy(head.num_outputs)}
            setattr(self, f"{stage}_metrics", nn.ModuleDict(metrics))

    def forward_batch(self, batch: tuple, stage: str) -> TimberPredictionOutput:
        """Run the finetuning model on a batch (see `TimberLightningModule.forward_batch`)."""
        signal, positions, labels = self.unpack(batch)
        fusion_stage = None if stage == "train" else "late"
        output = self.model(signal, positions.channel, positions.timestep, labels=labels, fusion_stage=fusion_stage)
        if labels is not None and stage in ("train", "val"):
            self._update_metrics(stage, output.logits.detach(), labels)
        return output

    def _update_metrics(self, stage: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
        labels = labels.repeat_interleave(logits.shape[0] // labels.shape[0], dim=0)  # loss-level fusion
        metrics = getattr(self, f"{stage}_metrics")
        if self.config.head.task == "regression":
            for metric in metrics.values():
                metric.update(logits.float(), labels.float())
            return
        classes = labels.reshape(labels.shape[0], -1)[:, 0].long()
        probabilities = logits.float().softmax(dim=-1)
        if self.config.head.num_outputs == 2:
            probabilities = probabilities[:, 1]
        for metric in metrics.values():
            metric.update(probabilities, classes)

    def _log_metrics(self, stage: str) -> None:
        metrics = getattr(self, f"{stage}_metrics")
        for name, metric in metrics.items():
            self.log(f"{stage}_{name}", metric.compute().mean(), prog_bar=True)
            metric.reset()

    def on_train_epoch_end(self) -> None:
        """Compute and log the training metrics of the epoch."""
        self._log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        """Compute and log the validation metrics of the epoch."""
        self._log_metrics("val")
