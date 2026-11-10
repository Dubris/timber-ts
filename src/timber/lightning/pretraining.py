"""Lightning module for pretraining (`TimberForPretraining`)."""

from __future__ import annotations

from ..modeling_timber import TimberForPretraining, TimberPretrainingOutput
from .base import TimberLightningModule


class TimberPretrainingModule(TimberLightningModule):
    """Pretraining with the objectives of ``config.decoders`` (TACO forecasting, reconstruction).

    Logs ``{train,val}/loss`` and ``{train,val}/loss_<objective>`` (weighted terms).
    """

    model_class = TimberForPretraining

    def forward_batch(self, batch: tuple, stage: str) -> TimberPretrainingOutput:
        """Run the pretraining model on a batch (see `TimberLightningModule.forward_batch`)."""
        signal, positions, _ = self.unpack(batch)
        return self.model(signal, positions.channel, positions.timestep)
