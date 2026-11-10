"""Generic data loading for memory-mapped multichannel time series.

`PartitionSampler` (index grid) -> `NpyTimeSeriesDataset` (files) -> batches of
`TimberDataModule` (``signal``, `SamplePositions`, ``labels``).
"""

from .datamodule import (
    ChunkCollator,
    DataModuleConfig,
    SamplePositions,
    SamplerConfig,
    TimberDataModule,
    build_dataloader,
    split_entities,
)
from .dataset import FileKey, NpyTimeSeriesDataset, is_channel_major, normalise_file_keys
from .sampler import ChunkIndex, PartitionSampler, sample_with_refill

__all__ = [
    "ChunkCollator",
    "ChunkIndex",
    "DataModuleConfig",
    "FileKey",
    "NpyTimeSeriesDataset",
    "PartitionSampler",
    "SamplePositions",
    "SamplerConfig",
    "TimberDataModule",
    "build_dataloader",
    "is_channel_major",
    "normalise_file_keys",
    "sample_with_refill",
    "split_entities",
]
