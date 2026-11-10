"""Dataset of memory-mapped ``(time, channel)`` arrays, one ``.npy`` file per (entity, recording, shard).

The dataset is given the files explicitly, so it makes no assumption about file names or
folders: an application maps its own storage to ``{(entity, recording, shard): path}``. A
recording may be split into several *shards* (channel-subset files); ``channel_index`` then
maps the local channel indices of every shard to global channel indices.

Files are memory-mapped: the OS page cache is shared by the DataLoader workers, so the data
is not duplicated per worker. Override `NpyTimeSeriesDataset.load` to read another
storage layout.

Storage order: a random subset of channels is read much faster from a file stored channel by
channel. Save such files in Fortran order (``np.save(path, np.asfortranarray(series))``; the
array keeps its ``(time, channel)`` shape, the ``.npy`` header records the order): a chunk then
reads one contiguous run per channel instead of scattered bytes of every time row (about 50x
less I/O for 1024 of 60k channels). Both orders are read transparently.

Threaded reads: on drives with a high latency (USB SSDs, network file systems), a memory map
reads one page at a time per process. With ``read_threads > 0``, the channels of channel-major
files are read with concurrent ``os.pread`` calls instead (they release the GIL), which keeps
many requests in flight (about 7x faster per process on a USB SSD). The items are identical.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from torch.utils.data import Dataset


class FileKey(NamedTuple):
    """Key of a time-series file."""

    entity: str
    recording: int
    shard: int = 0


def normalise_file_keys(files: Mapping[tuple, str | Path]) -> dict[FileKey, Path]:
    """``{(entity, recording[, shard]): path}`` -> ``{FileKey: Path}`` (shard ``0`` when omitted)."""
    normalised = {}
    for key, path in files.items():
        if len(key) not in (2, 3):
            msg = f"File keys are (entity, recording[, shard]), got {key!r}."
            raise ValueError(msg)
        normalised[FileKey(str(key[0]), int(key[1]), int(key[2]) if len(key) == 3 else 0)] = Path(path)
    return normalised


def is_channel_major(series: np.ndarray) -> bool:
    """Whether a ``(time, channel)`` array is stored channel by channel (Fortran order)."""
    return series.ndim == 2 and series.flags.f_contiguous and not series.flags.c_contiguous


def _read_channels(series: np.ndarray, channels: np.ndarray, time_slice: slice) -> np.ndarray:
    """``(channel, time)`` block of a ``(time, channel)`` array, reading along the storage order."""
    if is_channel_major(series):
        return series.T[channels, time_slice]  # one contiguous read per channel
    return series[time_slice, channels].T


def _whole_channel_major_file(series: np.ndarray) -> tuple[str, int] | None:
    """``(path, data offset)`` when ``series`` memory-maps a whole channel-major file, else ``None``.

    Covers Fortran-order ``(time, channel)`` files and transposed C-order ``(channel, time)`` files
    (same bytes on disk). Views of part of a file are rejected.
    """
    if not isinstance(series, np.memmap) or series.ndim != 2 or series.filename is None:
        return None
    item = series.dtype.itemsize
    if series.strides != (item, item * series.shape[0]):
        return None
    if os.path.getsize(series.filename) != series.offset + series.nbytes:
        return None
    return str(series.filename), series.offset


def _pread_channels(
    path: str,
    offset: int,
    series: np.ndarray,
    channels: np.ndarray,
    time_slice: slice,
    pool: ThreadPoolExecutor,
) -> np.ndarray:
    """``(channel, time)`` float32 block read with one concurrent ``pread`` per channel."""
    num_timepoints = series.shape[0]
    start, stop, _ = time_slice.indices(num_timepoints)
    dtype, item = series.dtype, series.dtype.itemsize
    length = (stop - start) * item
    out = np.empty((len(channels), stop - start), dtype=np.float32)
    fd = os.open(path, os.O_RDONLY)
    try:

        def read(row: int) -> None:
            position = offset + (int(channels[row]) * num_timepoints + start) * item
            data = os.pread(fd, length, position)
            if len(data) != length:
                msg = f"Short read in {path} at byte {position}."
                raise OSError(msg)
            out[row] = np.frombuffer(data, dtype=dtype)

        for _ in pool.map(read, range(len(channels))):
            pass  # re-raises the first error
    finally:
        os.close(fd)
    return out


def _rng(seed: int, epoch: int) -> np.random.Generator:
    """Generator of a chunk's resampling (epoch 0 uses the plain seed)."""
    return np.random.default_rng(int(seed) if epoch == 0 else [int(seed), epoch])


class NpyTimeSeriesDataset(Dataset):
    """Map-style dataset indexed by the tuples of `PartitionSampler`.

    An index is ``(entity_index, recording, channels, timepoints[, shard])`` where ``entity_index``
    indexes `entities` (sorted entity ids), ``channels`` are local channel indices in the file
    and ``timepoints`` a contiguous window.

    Args:
        files: ``{(entity, recording[, shard]): path}`` of ``(time, channel)`` ``.npy`` arrays.
        labels: Optional ``{entity: values}`` targets (one row per entity).
        channel_index: Optional ``{shard: global_channel_indices}`` mapping the local channels of
            every shard to global channel indices (identity when ``None``).
        resample_time_window: Replace the requested window by a random window of the same length
            (deterministic per index and epoch, see `ChunkIndex`).
        resample_channels: Replace the requested channels by random channels (deterministic per index and epoch).
        num_resample_channels: Channels to draw from when resampling (default: all the channels of the file).
        read_threads: Read channel-major files with this many concurrent ``pread`` calls per process
            (``0``: through the memory map). Faster on high-latency drives; the items are identical.

    Items are ``(signal, positions, target)``: ``signal`` ``(channel, time)`` float32, ``positions``
    ``(entity_index, recording, global_channel_indices, timepoints)`` and ``target`` ``(num_labels,)``.

    """

    def __init__(
        self,
        files: Mapping[Any, str | Path],
        labels: Mapping[str, Sequence[float]] | None = None,
        channel_index: Mapping[int, Sequence[int]] | None = None,
        resample_time_window: bool = False,
        resample_channels: bool = False,
        num_resample_channels: int | None = None,
        read_threads: int = 0,
    ) -> None:
        if len(files) == 0:
            msg = "The dataset needs at least one file."
            raise ValueError(msg)
        self.files = normalise_file_keys(files)
        self.entities = sorted({key.entity for key in self.files})
        self.labels = None
        if labels is not None:
            missing = [entity for entity in self.entities if entity not in labels]
            if missing:
                msg = f"Missing labels for entities {missing[:5]}."
                raise ValueError(msg)
            self.labels = {entity: np.asarray(labels[entity], dtype=np.float32).reshape(-1) for entity in self.entities}
        self.channel_index = (
            {int(k): np.asarray(v) for k, v in channel_index.items()} if channel_index is not None else None
        )
        self.resample_time_window = resample_time_window
        self.resample_channels = resample_channels
        self.num_resample_channels = num_resample_channels
        self.read_threads = read_threads
        self._pool: ThreadPoolExecutor | None = None  # created in each process (DataLoader workers)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pool"] = None  # threads are not picklable
        return state

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.read_threads, thread_name_prefix="timber-read")
        return self._pool

    def __len__(self) -> int:
        """Number of files."""
        return len(self.files)

    def load(self, path: Path) -> np.ndarray:
        """Read a file as a ``(time, channel)`` array (memory-mapped)."""
        return np.load(path, mmap_mode="r")

    def global_channels(self, series: np.ndarray, shard: int, local_channels: np.ndarray) -> np.ndarray:
        """Global indices of the ``local_channels`` of a shard (see ``channel_index``)."""
        if self.channel_index is None:
            return local_channels
        return self.channel_index[shard][local_channels]

    def __getitem__(self, index: tuple) -> tuple[np.ndarray, tuple, np.ndarray]:
        entity_index, recording, channels, timepoints = index[:4]
        shard = index[4] if len(index) == 5 else 0
        # The epoch seeds the random draws below (epoch 0 uses the plain seed).
        epoch = getattr(index, "epoch", 0)  # set by the sampler: new random draws every epoch
        entity = self.entities[entity_index]
        # Memory-mapped: the bytes are read only when the window is copied below.
        series = self.load(self.files[FileKey(entity, recording, shard)])

        if self.resample_time_window and timepoints is not None:
            window = timepoints[-1] - timepoints[0] + 1
            seed = entity_index + recording + np.sum(channels) + timepoints[0] + shard
            # Replace the requested window by a random window of the same length, inside the recording.
            start = _rng(seed, epoch).integers(0, series.shape[0] - window + 1).item()
            timepoints = np.arange(start, start + window)
        time_slice = slice(timepoints[0], timepoints[-1] + 1) if timepoints is not None else slice(None)

        if self.resample_channels and channels is not None:
            num_channels = min(self.num_resample_channels or series.shape[1], series.shape[1])
            seed = entity_index + recording + np.sum(channels) + (time_slice.start or 0) + shard
            # Replace the channels by random ones of the file; sorted, so that the reads go forward through the file.
            channels = np.sort(_rng(seed, epoch).choice(num_channels, size=len(channels), replace=False))

        # Indices of the channels inside this file (the global indices are computed from them below).
        local_channels = np.array(channels, copy=True)
        # Threaded reads only apply to whole channel-major files; everything else goes through the memory map.
        file = _whole_channel_major_file(series) if self.read_threads > 0 and channels is not None else None
        if file is not None:
            signal = _pread_channels(*file, series, local_channels, time_slice, self._executor())
        else:
            signal = np.ascontiguousarray(_read_channels(series, local_channels, time_slice), dtype=np.float32)
        if timepoints is None:
            timepoints = np.arange(series.shape[0])
        # The targets of the entity, or an empty vector when the dataset has no labels.
        target = self.labels[entity] if self.labels is not None else np.zeros(0, dtype=np.float32)
        positions = (
            entity_index,
            recording,
            self.global_channels(series, shard, local_channels),
            np.asarray(timepoints),
        )
        return signal, positions, target
