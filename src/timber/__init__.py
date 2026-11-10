"""TIMBER: self-supervised representation learning for multichannel time series.

A patch tokenizer, a causal factorised channel/time contextualizer, a Perceiver IO encoder
and the TACO forecasting objective, packaged as Hugging Face models (``TimberModel``,
``TimberForPretraining``, ``TimberForPrediction``) with Lightning wrappers (``timber.lightning``)
and a generic data pipeline for memory-mapped arrays (``timber.data``).

TIMBER stands for Time-series Embedded Representation learning: the patches of a multichannel time series are
embedded, and their representations are learned without labels.

Importing ``timber`` registers the models with the transformers Auto classes, so that
``AutoConfig`` / ``AutoModel`` / ``AutoModelForPreTraining`` can load TIMBER checkpoints.
"""

from importlib.metadata import PackageNotFoundError, version

from transformers import AutoConfig, AutoModel, AutoModelForPreTraining

from .configuration_timber import (
    ContextualizerConfig,
    DecoderConfig,
    EncoderConfig,
    HeadConfig,
    TimberConfig,
    TokenizerConfig,
)
from .modeling_timber import (
    TimberForPrediction,
    TimberForPretraining,
    TimberModel,
    TimberModelOutput,
    TimberPredictionOutput,
    TimberPreTrainedModel,
    TimberPretrainingOutput,
)

try:
    __version__ = version("timber-ts")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

AutoConfig.register(TimberConfig.model_type, TimberConfig, exist_ok=True)
AutoModel.register(TimberConfig, TimberModel, exist_ok=True)
AutoModelForPreTraining.register(TimberConfig, TimberForPretraining, exist_ok=True)

__all__ = [
    "ContextualizerConfig",
    "DecoderConfig",
    "EncoderConfig",
    "HeadConfig",
    "TimberConfig",
    "TimberForPrediction",
    "TimberForPretraining",
    "TimberModel",
    "TimberModelOutput",
    "TimberPreTrainedModel",
    "TimberPredictionOutput",
    "TimberPretrainingOutput",
    "TokenizerConfig",
    "__version__",
]
