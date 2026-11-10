"""Sampler that cuts an (entity, recording, channel, time) data grid into chunks and batches.

The data is a grid with one axis per *dimension*: entities (what carries the labels: a subject, a patient, a machine),
recordings of an entity, channels, timepoints, and optionally shards (a recording stored as
several channel-subset files). Along each axis the indices are (optionally) shuffled and cut
into chunks of ``chunk_shape[axis]`` consecutive elements. A batch takes
``chunks_per_batch[axis]`` chunks along every axis, and the sampler yields one index tuple per
combination (the Cartesian product of the chunks), so a batch has ``prod(chunks_per_batch)``
samples. `ChunkCollator` reshapes it back to
``(*chunks_per_batch, channel, time)``.

Example: ``partition_shape = (entities, 4 recordings, 10000 channels, 1200 timepoints)``,
``chunk_shape = (1, 1, 1024, 1200)``, ``chunks_per_batch = (2, 2, 2, 1)``: each batch has
2 entities x 2 recordings x 2 sets of 1024 channels x 1 window of 1200 timepoints (cropped to 900
with ``crop_shape = (0, 0, 0, 300)``).

The batches of an epoch depend only on ``seed + epoch`` (call `set_epoch`).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Sampler


class ChunkIndex(tuple):
    """Index of one sample, ``(entity, recording, channels, timepoints[, shard])``, with the epoch it was drawn in.

    It compares equal to the plain tuple. The epoch reaches the dataset (also in persistent workers) and seeds
    its random resampling, so resampled windows and channels change from one epoch to the next.
    """

    epoch: int

    def __new__(cls, values, epoch: int = 0) -> ChunkIndex:
        index = super().__new__(cls, values)
        index.epoch = epoch
        return index


def sample_with_refill(values: ArrayLike, size: int, seed: int | None = None) -> np.ndarray:
    """Draw ``size`` values without replacement, refilling the pool whenever it is exhausted."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)  # pyright: ignore[reportAssignmentType]
    if len(values) == 0:
        msg = "Cannot sample from an empty array."
        raise ValueError(msg)
    sampled: list = []
    while len(sampled) < size:
        sampled.extend(rng.choice(values, size=min(len(values), size - len(sampled)), replace=False))
    return np.array(sampled)


class PartitionSampler(Sampler):
    """Yield index tuples ``(entity, recording, channels, timepoints[, shard])`` batch by batch.

    Args:
        partition_shape: Size of every axis of the data grid.
        chunk_shape: Elements per chunk along every axis (a chunk of size 1 yields an ``int``,
            larger chunks yield sorted index arrays).
        chunks_per_batch: Chunks per batch along every axis.
        shuffle_before_chunking: Shuffle the indices of each axis (or all axes) before cutting them into chunks.
        crop_shape: Elements randomly cropped from every chunk (``chunk - crop`` are kept), per axis.
        labels: Optional per-axis labels; chunks along that axis then contain one element of every
            label (stratified), e.g. balanced classes per batch.
        shuffle_batches: Shuffle the order of the batches.
        max_batches: Keep only the first batches of every epoch.
        max_batches_per_axis: Keep only the first chunk groups along every axis.
        resample_per_batch: Per axis, redraw the indices of every batch (for shuffled axes).
        seed: Base seed; the epoch seed is ``seed + epoch``.

    """

    def __init__(
        self,
        partition_shape: Sequence[int],
        chunk_shape: Sequence[int],
        chunks_per_batch: Sequence[int],
        shuffle_before_chunking: Sequence[bool] | bool = True,
        crop_shape: Sequence[int] | None = None,
        labels: Sequence[Sequence[int] | None] | None = None,
        shuffle_batches: bool = True,
        max_batches: int | None = None,
        max_batches_per_axis: Sequence[int] | None = None,
        resample_per_batch: Sequence[bool] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        num_axes = len(partition_shape)

        def check_length(name: str, value: Sequence | None) -> None:
            if value is not None and len(value) != num_axes:
                msg = f"{name} must have one element per axis ({num_axes}), got {len(value)}."
                raise ValueError(msg)

        check_length("chunk_shape", chunk_shape)
        check_length("chunks_per_batch", chunks_per_batch)
        if isinstance(shuffle_before_chunking, bool):
            shuffle_before_chunking = [shuffle_before_chunking] * num_axes
        check_length("shuffle_before_chunking", shuffle_before_chunking)
        crop_shape = list(crop_shape) if crop_shape is not None else [0] * num_axes
        check_length("crop_shape", crop_shape)
        # A crop must leave at least one element of every chunk (the chunk may be one longer than its crop).
        if any(np.array(crop_shape) > np.array(chunk_shape) + 1):
            msg = "Every element of crop_shape must be smaller than the matching chunk_shape."
            raise ValueError(msg)
        check_length("max_batches_per_axis", max_batches_per_axis)
        check_length("resample_per_batch", resample_per_batch)
        labels = list(labels) if labels is not None else [None] * num_axes
        check_length("labels", labels)
        for axis, axis_labels in enumerate(labels):
            if axis_labels is None:
                continue
            if len(axis_labels) != partition_shape[axis]:
                msg = f"labels[{axis}] must have partition_shape[{axis}] elements."
                raise ValueError(msg)
            # A stratified chunk holds one element of every label, so the labels must match the chunk size.
            if len(np.unique(axis_labels)) != chunk_shape[axis]:
                msg = f"The number of distinct labels[{axis}] must equal chunk_shape[{axis}]."
                raise ValueError(msg)

        self.partition_shape = np.array(partition_shape)
        self.chunk_shape = np.array(chunk_shape)
        self.chunks_per_batch = np.array(chunks_per_batch)
        self.shuffle_before_chunking = list(shuffle_before_chunking)
        self.crop_shape = crop_shape
        self.labels = labels
        self.shuffle_batches = shuffle_batches
        self.max_batches = max_batches
        self.max_batches_per_axis = max_batches_per_axis
        self.resample_per_batch = resample_per_batch
        self.seed = seed
        self.epoch = 0
        self._cache: tuple[int, list] | None = None

    def set_epoch(self, epoch: int) -> None:
        """Select the batches of an epoch (they only depend on ``seed + epoch``)."""
        self.epoch = epoch

    @property
    def batch_size(self) -> int:
        """Samples per batch: ``prod(chunks_per_batch)``."""
        return int(np.prod(self.chunks_per_batch))

    @property
    def groups_per_axis(self) -> list[int]:
        """Number of chunk groups (batch positions) along every axis in the current epoch."""
        self._batches()
        return list(self._groups_per_axis)

    def __len__(self) -> int:
        num_batches = len(self._batches())
        if self.max_batches is not None:
            num_batches = min(num_batches, self.max_batches)
        return num_batches * self.batch_size

    def _batches(self) -> list:
        """The batches of the current epoch (computed once per epoch)."""
        if self._cache is None or self._cache[0] != self.epoch:
            self._cache = (self.epoch, self._generate_batches())
        return self._cache[1]

    def _generate_batches(self) -> list:
        epoch_seed = self.seed + self.epoch
        rng = np.random.default_rng(epoch_seed)

        indices = [np.arange(size) for size in self.partition_shape]
        # Shuffle the axes that ask for it, before cutting them into chunks.
        for axis_indices, shuffle in zip(indices, self.shuffle_before_chunking, strict=True):
            if shuffle:
                rng.shuffle(axis_indices)

        # Cut every axis into chunks (sorted inside a chunk for I/O locality)
        # Cut each axis into consecutive chunks of ``chunk_shape`` elements; a size of 1 gives a plain index.
        chunks: list[Any] = [
            [
                np.sort(axis_indices[i : i + size]) if size > 1 else int(axis_indices[i])
                for i in range(0, len(axis_indices), size)
                if i + size <= len(axis_indices)
            ]
            for axis_indices, size in zip(indices, self.chunk_shape, strict=True)
        ]

        # Stratified chunks: one element of every label per chunk
        for axis, axis_labels in enumerate(self.labels):
            if axis_labels is not None:
                num_chunks = len(chunks[axis])
                axis_labels = np.asarray(axis_labels)
                per_label = [
                    sample_with_refill(np.flatnonzero(axis_labels == label), num_chunks, epoch_seed)
                    for label in np.unique(axis_labels)  # sorted labels, positions in order
                ]
                # Column ``k`` of the result is the chunk ``k``: one element of each label.
                chunks[axis] = np.stack(per_label).T

        # Pad every axis to a multiple of chunks_per_batch
        for axis, (axis_chunks, per_batch) in enumerate(zip(chunks, self.chunks_per_batch, strict=True)):
            num_chunks = len(axis_chunks)
            # Pad the axis with extra chunks so that every batch is full (``chunks_per_batch`` chunks per batch).
            if num_chunks % per_batch:
                target = (num_chunks // per_batch + 1) * per_batch
                if self.shuffle_before_chunking[axis]:
                    extra = sample_with_refill(axis_chunks, target % num_chunks, epoch_seed)
                else:
                    extra = axis_chunks[: target % num_chunks]
                chunks[axis] = (target // num_chunks) * list(axis_chunks) + list(extra)

        # Group the chunks of every axis by batch, then combine the axes
        groups = [
            [axis_chunks[i : i + per_batch] for i in range(0, len(axis_chunks), per_batch)]
            # Group consecutive chunks into batches, along every axis.
            for axis_chunks, per_batch in zip(chunks, self.chunks_per_batch, strict=True)
        ]
        max_batches_per_axis = self.max_batches_per_axis
        if max_batches_per_axis is not None:
            groups = [g[:limit] for g, limit in zip(groups, max_batches_per_axis, strict=True)]
        self._groups_per_axis = [len(g) for g in groups]
        # A batch is one group per axis: their Cartesian product gives the entities, recordings, channel sets and
        # windows.
        batches: list[Any] = list(product(*groups))
        if self.shuffle_batches:
            rng.shuffle(batches)

        # Random crop inside the chunks
        if any(self.crop_shape):
            cropped = []
            for batch in batches:
                # Draw a random start inside every chunk, leaving room for the crop: a random window, per chunk.
                starts: list[Any] = [
                    [rng.choice(crop, 1).item() for _ in range(per_batch)] if crop > 0 else None
                    for crop, per_batch in zip(self.crop_shape, self.chunks_per_batch, strict=True)
                ]
                cropped.append(
                    [
                        [
                            chunk[start : start + len(chunk) - crop]
                            for start, chunk in zip(axis_starts, axis_chunks, strict=True)
                        ]
                        if crop > 0
                        else axis_chunks
                        for axis_chunks, crop, axis_starts in zip(batch, self.crop_shape, starts, strict=True)
                    ]
                )
            batches = cropped
        return batches

    def __iter__(self) -> Iterator[ChunkIndex]:
        batches = self._batches()
        if self.max_batches is not None:
            batches = batches[: self.max_batches]
        for i, batch in enumerate(batches):
            if self.resample_per_batch is not None:
                batch = [
                    sample_with_refill(np.arange(size), len(axis_chunks), self.seed + self.epoch + i)
                    if resample and shuffled
                    else axis_chunks
                    for size, axis_chunks, resample, shuffled in zip(
                        self.partition_shape,
                        batch,
                        self.resample_per_batch,
                        self.shuffle_before_chunking,
                        strict=True,
                    )
                ]
            # Each index is one sample: one chunk per axis, tagged with the epoch (it seeds the dataset's random draws).
            for index in product(*batch):
                yield ChunkIndex(index, self.epoch)
