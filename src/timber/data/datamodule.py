"""Batches, data loaders and the Lightning data module.

A batch is ``(signal, positions, labels)``:

- ``signal``: ``(*chunks_per_batch, channel, time)`` float32 (the leading axis is the entity axis);
- ``positions``: `SamplePositions` of ``(*chunks_per_batch, 1)`` entity indices and recordings,
  ``(*chunks_per_batch, channel)`` global channel indices and ``(*chunks_per_batch, time)`` timepoints;
- ``labels``: ``(entity, num_labels)`` (one row per entity), or ``None`` without labels.

`TimberDataModule` receives the files and labels explicitly (no file-name convention);
an application builds them from its own storage. All the settings live in
`DataModuleConfig` (single source of defaults).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader

from .dataset import NpyTimeSeriesDataset, normalise_file_keys
from .sampler import PartitionSampler


class SamplePositions(NamedTuple):
    """Where every sample of a batch comes from."""

    entity: torch.Tensor  # (*chunks, 1) index into the dataset entities
    recording: torch.Tensor  # (*chunks, 1)
    channel: torch.Tensor  # (*chunks, channel) global channel indices
    timestep: torch.Tensor  # (*chunks, time) timepoints


def merge_shard_axis(chunks_per_batch: Sequence[int]) -> list[int]:
    """5-axis ``(entity, recording, channel, time, shard)`` batch layout -> 4 axes (shards merged into channels)."""
    chunks = list(chunks_per_batch)
    if len(chunks) == 5:
        return [chunks[0], chunks[1], chunks[2] * chunks[4], chunks[3]]
    return chunks


class ChunkCollator:
    """Stack the samples of a `PartitionSampler` batch and reshape them to ``chunks_per_batch``."""

    def __init__(self, chunks_per_batch: Sequence[int]) -> None:
        self.chunks = merge_shard_axis(chunks_per_batch)

    def __call__(self, items: list) -> tuple[torch.Tensor, SamplePositions, torch.Tensor | None]:
        signals, positions, targets = zip(*items, strict=True)
        signal = torch.from_numpy(np.stack(signals)).view(*self.chunks, *signals[0].shape)
        entities, recordings, channels, timesteps = zip(*positions, strict=True)
        sample_positions = SamplePositions(
            entity=torch.as_tensor(np.asarray(entities), dtype=torch.long).view(*self.chunks, 1),
            recording=torch.as_tensor(np.asarray(recordings), dtype=torch.long).view(*self.chunks, 1),
            channel=torch.as_tensor(np.stack(channels), dtype=torch.long).view(*self.chunks, -1),
            timestep=torch.as_tensor(np.stack(timesteps), dtype=torch.long).view(*self.chunks, -1),
        )
        target = np.stack(targets)
        labels = None
        if target.size > 0:  # one row per entity
            labels = torch.from_numpy(target).view(*self.chunks, -1)[:, 0, 0, 0]
        return signal, sample_positions, labels


@dataclass
class SamplerConfig:
    """Arguments of `PartitionSampler` (see there). ``partition_shape[0]`` is set to the number of entities."""

    partition_shape: list[int]
    chunk_shape: list[int]
    chunks_per_batch: list[int]
    shuffle_before_chunking: list[bool] | bool = True
    crop_shape: list[int] | None = None
    labels: list | None = None
    shuffle_batches: bool = True
    max_batches: int | None = None
    max_batches_per_axis: list[int] | None = None
    resample_per_batch: list[bool] | None = None
    seed: int = 42


def _strict(cls: type, values: Mapping, what: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        msg = f"Unknown {what} option(s) {unknown}. Valid options: {sorted(known)}."
        raise ValueError(msg)
    return cls(**values)


@dataclass
class DataModuleConfig:
    """Settings of `TimberDataModule`.

    Attributes:
        sampler: Training sampler (`SamplerConfig` or dict).
        eval_sampler_overrides: Sampler fields overridden for validation and test (``max_batches`` is always ``None``).
        num_train_entities, num_val_entities, num_test_entities: Sizes of the random entity split.
        train_entities, val_entities, test_entities: Explicit splits (override the sizes).
        resample_time_window, resample_channels, num_resample_channels, read_threads: See
            `NpyTimeSeriesDataset`.
        drop_last: Drop the last incomplete batch.
        use_batch_sampler: Wrap the sampler in a ``BatchSampler``.
        num_workers, prefetch_factor: DataLoader workers (workers are persistent).
        seed: Seed of the entity split.

    """

    sampler: SamplerConfig  # a dict is converted in __post_init__
    eval_sampler_overrides: dict | None = None
    num_train_entities: int = 0
    num_val_entities: int = 0
    num_test_entities: int = 0
    train_entities: list[str] | None = None
    val_entities: list[str] | None = None
    test_entities: list[str] | None = None
    resample_time_window: bool = False
    resample_channels: bool = False
    num_resample_channels: int | None = None
    read_threads: int = 0
    drop_last: bool = False
    use_batch_sampler: bool = False
    num_workers: int = 0
    prefetch_factor: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.sampler, Mapping):
            self.sampler = _strict(SamplerConfig, self.sampler, "sampler")

    @classmethod
    def from_dict(cls, values: DataModuleConfig | Mapping) -> DataModuleConfig:
        """Build a config from a mapping (a config is returned unchanged); unknown keys raise."""
        return values if isinstance(values, DataModuleConfig) else _strict(cls, values, "data module")


def split_entities(config: DataModuleConfig, entities: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Train / validation / test entities (explicit lists, or a seeded random split of the sorted ``entities``)."""
    # Seeded split: the same seed gives the same entities, whatever the order of the files.
    rng = np.random.default_rng(config.seed)
    train = config.train_entities
    if train is None:
        train = rng.choice(entities, size=config.num_train_entities, replace=False).tolist()
    val = config.val_entities
    # Validation and test entities are drawn from the entities that are not in the training set.
    if val is None:
        val = rng.choice(sorted(set(entities) - set(train)), size=config.num_val_entities, replace=False).tolist()
    test = config.test_entities
    # Test entities are the remaining entities (the first ``num_test_entities`` of them).
    if test is None:
        test = sorted(set(entities) - set(train) - set(val))[: config.num_test_entities]
    for name, given, size in (
        ("train", config.train_entities, config.num_train_entities),
        ("val", config.val_entities, config.num_val_entities),
        ("test", config.test_entities, config.num_test_entities),
    ):
        # An explicit list overrides the size: warn when both are given and disagree.
        if given is not None and size and len(given) != size:
            warnings.warn(f"num_{name}_entities ignored: {len(given)} {name} entities were given.", stacklevel=3)
    return list(train), list(val), list(test)


def build_dataloader(
    dataset: NpyTimeSeriesDataset,
    sampler_config: SamplerConfig,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    drop_last: bool = False,
    use_batch_sampler: bool = False,
) -> DataLoader:
    """DataLoader over ``dataset`` with a `PartitionSampler` and `ChunkCollator`."""
    kwargs = dataclasses.asdict(sampler_config)
    kwargs["partition_shape"] = list(kwargs["partition_shape"])
    # The entity axis of the sampler is the number of entities of this dataset (the files define it).
    kwargs["partition_shape"][0] = len(dataset.entities)
    sampler = PartitionSampler(**kwargs)
    loader_kwargs: dict[str, Any] = {
        "collate_fn": ChunkCollator(sampler_config.chunks_per_batch),
        "num_workers": num_workers,
        # Pinned memory only helps CUDA; workers stay alive between epochs when there are any.
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }
    # Optionally, a BatchSampler groups the indices into batches itself (the same batches, another code path).
    if use_batch_sampler:
        batch_sampler = BatchSampler(sampler, batch_size=sampler.batch_size, drop_last=drop_last)
        return DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
    return DataLoader(dataset, sampler=sampler, batch_size=sampler.batch_size, drop_last=drop_last, **loader_kwargs)


class TimberDataModule(LightningDataModule):
    """Entity-level train / validation / test split and the matching data loaders.

    Args:
        config: `DataModuleConfig` or the same fields as a dict (e.g. from Hydra).
        files: ``{(entity, recording[, shard]): path}`` of ``(time, channel)`` ``.npy`` arrays.
        labels: Optional ``{entity: values}``; entities without labels are excluded.
        channel_index: Optional ``{shard: global_channel_indices}`` (see `NpyTimeSeriesDataset`).

    Subclasses can set `dataset_class` to adapt how files are read.
    """

    dataset_class: type[NpyTimeSeriesDataset] = NpyTimeSeriesDataset

    def __init__(
        self,
        config: DataModuleConfig | Mapping,
        files: Mapping[tuple, str | Path],
        labels: Mapping[str, Sequence[float]] | None = None,
        channel_index: Mapping[int, Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.config = DataModuleConfig.from_dict(config)
        self.files = normalise_file_keys(files)
        self.labels = labels
        self.channel_index = channel_index
        entities = sorted({key.entity for key in self.files})
        if labels is not None:
            entities = [entity for entity in entities if entity in labels]
        self.train_entities, self.val_entities, self.test_entities = split_entities(self.config, entities)
        self.save_hyperparameters(
            {
                "datamodule": dataclasses.asdict(self.config),
                "train_entities": self.train_entities,
                "val_entities": self.val_entities,
                "test_entities": self.test_entities,
            }
        )
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def make_dataset(self, entities: Sequence[str]) -> NpyTimeSeriesDataset | None:
        """Dataset over the files of ``entities`` (``None`` when there is none)."""
        wanted = set(entities)
        files = {key: path for key, path in self.files.items() if key.entity in wanted}
        if not files:
            return None
        return self.dataset_class(
            files,
            labels=self.labels,
            channel_index=self.channel_index,
            resample_time_window=self.config.resample_time_window,
            resample_channels=self.config.resample_channels,
            num_resample_channels=self.config.num_resample_channels,
            read_threads=self.config.read_threads,
        )

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets of the train, validation and test splits."""
        self.train_dataset = self.make_dataset(self.train_entities)
        self.val_dataset = self.make_dataset(self.val_entities)
        self.test_dataset = self.make_dataset(self.test_entities)

    def sampler_config(self, stage: str) -> SamplerConfig:
        """The training `SamplerConfig`."""
        # Validation and test use a copy of the sampler, with their overrides (e.g. no shuffling, no max batches).
        config = deepcopy(self.config.sampler)
        if stage != "train":
            config.max_batches = None
            for key, value in (self.config.eval_sampler_overrides or {}).items():
                setattr(config, key, value)
        return config

    def _loader(self, dataset: NpyTimeSeriesDataset | None, stage: str) -> DataLoader | None:
        if dataset is None:
            return None
        return build_dataloader(
            dataset,
            self.sampler_config(stage),
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=self.config.drop_last,
            use_batch_sampler=self.config.use_batch_sampler,
        )

    def train_dataloader(self) -> DataLoader:
        """Data loader of the training split."""
        loader = self._loader(self.train_dataset, "train")
        if loader is None:
            msg = "No training entities: call setup() and give train entities."
            raise ValueError(msg)
        return loader

    def val_dataloader(self) -> DataLoader | None:
        """Data loader of the validation split (``eval_sampler_overrides`` applied)."""
        return self._loader(self.val_dataset, "val")

    def test_dataloader(self) -> DataLoader | None:
        """Data loader of the test split (``eval_sampler_overrides`` applied)."""
        return self._loader(self.test_dataset, "test")
