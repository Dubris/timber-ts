"""Lightning callbacks of the fMRI experiments (enabled from the ``callbacks`` section of the configs)."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class PlateauRelativeEarlyStopping(Callback):
    """Stop when the monitored metric improved by less than a relative threshold over a window of validations.

    The improvement is ``(old - new) / old`` between the value ``window`` validations ago and the
    latest one (a metric to minimise, e.g. a loss).

    Config::

        callbacks:
          plateau_early_stopping: {monitor: val/loss, window: 3, min_rel_improvement: 0.005}

    Args:
        monitor: Logged metric to watch.
        window: Number of validations the improvement is measured over.
        min_rel_improvement: Training stops below this relative improvement.
        verbose: Log a message when stopping.

    """

    def __init__(
        self,
        monitor: str = "val/loss",
        window: int = 3,
        min_rel_improvement: float = 0.005,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.window = window
        self.min_rel_improvement = min_rel_improvement
        self.verbose = verbose
        self.values: deque[float] = deque(maxlen=window + 1)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None or trainer.sanity_checking:
            return
        self.values.append(float(current.detach().cpu()))
        if len(self.values) < self.window + 1:
            return
        old, new = self.values[0], self.values[-1]
        improvement = (old - new) / max(abs(old), 1e-8)
        if improvement < self.min_rel_improvement:
            trainer.should_stop = True
            if self.verbose:
                logger.info(
                    "Relative improvement of %s over %d validations: %.4f%% < %.4f%%. Stopping.",
                    self.monitor,
                    self.window,
                    100 * improvement,
                    100 * self.min_rel_improvement,
                )

    # Saved in the Lightning checkpoints, so that resuming keeps the window
    def state_dict(self) -> dict[str, Any]:
        return {"values": list(self.values)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.values = deque(state_dict.get("values", state_dict.get("losses", [])), maxlen=self.window + 1)
