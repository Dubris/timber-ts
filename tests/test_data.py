"""Tests of the generic data pipeline (:mod:`timber.data`) on synthetic ``.npy`` files."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from timber.data import (
    ChunkIndex,
    DataModuleConfig,
    NpyTimeSeriesDataset,
    PartitionSampler,
    SamplePositions,
    TimberDataModule,
    is_channel_major,
    normalise_file_keys,
)

NUM_ENTITIES, NUM_RECORDINGS, NUM_TIMEPOINTS, NUM_CHANNELS = 6, 2, 40, 50


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict:
    folder = tmp_path_factory.mktemp("arrays")
    rng = np.random.default_rng(0)
    files = {}
    for entity in range(NUM_ENTITIES):
        for recording in range(NUM_RECORDINGS):
            path = folder / f"entity{entity}_rec{recording}.npy"
            np.save(path, rng.standard_normal((NUM_TIMEPOINTS, NUM_CHANNELS)).astype(np.float32))
            files[(f"u{entity}", recording)] = path
    return files


def sampler_kwargs(**overrides) -> dict:
    kwargs = {
        "partition_shape": [NUM_ENTITIES, NUM_RECORDINGS, NUM_CHANNELS, NUM_TIMEPOINTS],
        "chunk_shape": [1, 1, 8, 32],
        "chunks_per_batch": [2, 2, 1, 1],
        "shuffle_before_chunking": [True, True, True, False],
        "crop_shape": [0, 0, 0, 8],
        "seed": 3,
    }
    kwargs.update(overrides)
    return kwargs


def as_lists(indices) -> list:
    return [[np.asarray(i).tolist() for i in index] for index in indices]


# --- Sampler -------------------------------------------------------------------------------------
def test_sampler_batches():
    sampler = PartitionSampler(**sampler_kwargs())
    indices = list(sampler)
    assert len(indices) == len(sampler) and sampler.batch_size == 4
    assert sampler.groups_per_axis == [3, 1, 6, 1]  # 6 entities / 2, 2 recordings / 2, 50 // 8 channels, 1 window
    for entity, recording, channels, timepoints in indices:
        assert 0 <= entity < NUM_ENTITIES and 0 <= recording < NUM_RECORDINGS
        assert len(channels) == 8 and (np.diff(channels) > 0).all()
        assert len(timepoints) == 24 and (np.diff(timepoints) == 1).all()


def test_sampler_epochs_differ_and_batches_are_cached(monkeypatch):
    sampler = PartitionSampler(**sampler_kwargs())
    first = as_lists(sampler)
    calls = []
    original = sampler._generate_batches
    monkeypatch.setattr(sampler, "_generate_batches", lambda: calls.append(1) or original())
    len(sampler), list(sampler), len(sampler)
    assert not calls  # same epoch: cached
    sampler.set_epoch(1)
    assert as_lists(sampler) != first and len(calls) == 1
    sampler.set_epoch(0)
    assert as_lists(sampler) == first


def test_stratified_chunks_have_one_element_per_label():
    labels = [0, 1, 0, 1, 0, 1]
    sampler = PartitionSampler(
        **sampler_kwargs(labels=[labels, None, None, None], chunk_shape=[2, 1, 8, 32], chunks_per_batch=[1, 2, 1, 1])
    )
    for entity_chunk, *_ in sampler:
        assert sorted(labels[i] for i in entity_chunk) == [0, 1]


def test_sampler_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="one element per axis"):
        PartitionSampler(**sampler_kwargs(chunk_shape=[1, 1, 8]))


# --- Dataset -------------------------------------------------------------------------------------
def test_dataset_items(files):
    labels = {f"u{u}": [float(u), -float(u)] for u in range(NUM_ENTITIES)}
    dataset = NpyTimeSeriesDataset(files, labels=labels)
    assert dataset.entities == [f"u{u}" for u in range(NUM_ENTITIES)] and len(dataset) == NUM_ENTITIES * NUM_RECORDINGS
    signal, positions, target = dataset[(2, 1, np.array([3, 5, 9]), np.arange(4, 20))]
    expected = np.load(files[("u2", 1)])[4:20, [3, 5, 9]].T
    np.testing.assert_array_equal(signal, expected)
    assert signal.dtype == np.float32 and signal.shape == (3, 16)
    np.testing.assert_array_equal(positions[2], [3, 5, 9])
    np.testing.assert_array_equal(target, [2.0, -2.0])


def test_dataset_shards_and_channel_index(files):
    sharded = {("u0", 0, 0): files[("u0", 0)], ("u0", 0, 1): files[("u0", 1)]}
    dataset = NpyTimeSeriesDataset(sharded, channel_index={0: np.arange(50), 1: np.arange(50) + 50})
    _, positions, _ = dataset[(0, 0, np.array([1, 2]), np.arange(10), 1)]
    np.testing.assert_array_equal(positions[2], [51, 52])


def test_dataset_resampling_is_deterministic(files):
    dataset = NpyTimeSeriesDataset(files, resample_time_window=True, resample_channels=True, num_resample_channels=20)
    index = (1, 0, np.array([0, 1, 2, 3]), np.arange(0, 10))
    first, second = dataset[index], dataset[index]
    np.testing.assert_array_equal(first[0], second[0])
    assert (first[1][2] < 20).all() and len(first[1][3]) == 10


@pytest.mark.parametrize("resample", [False, True])
def test_channel_major_files_give_the_same_items(files, tmp_path, resample):
    """Files stored channel by channel (Fortran order) are read along their storage order, with the same result."""
    fortran = {}
    for key, path in files.items():
        fortran[key] = tmp_path / path.name
        np.save(fortran[key], np.asfortranarray(np.load(path)))
    assert is_channel_major(np.load(next(iter(fortran.values())), mmap_mode="r"))
    assert not is_channel_major(np.load(next(iter(files.values())), mmap_mode="r"))
    options = {"resample_time_window": resample, "resample_channels": resample}
    c_order, f_order = NpyTimeSeriesDataset(files, **options), NpyTimeSeriesDataset(fortran, **options)
    for index in [(1, 0, np.array([3, 7, 8, 40]), np.arange(5, 25)), (4, 1, np.array([0, 49]), None)]:
        (signal, positions, _), (expected, expected_positions, _) = f_order[index], c_order[index]
        np.testing.assert_array_equal(signal, expected)
        assert signal.flags.c_contiguous and signal.dtype == np.float32
        for got, want in zip(positions, expected_positions, strict=True):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("resample", [False, True])
def test_threaded_reads_give_the_same_items(files, tmp_path, resample):
    """``read_threads`` reads channel-major files with concurrent preads, with the same items as the memory map."""
    fortran = {}
    for key, path in files.items():
        fortran[key] = tmp_path / path.name
        np.save(fortran[key], np.asfortranarray(np.load(path)))
    options = {"resample_time_window": resample, "resample_channels": resample}
    indices = [(1, 0, np.array([3, 7, 8, 40]), np.arange(5, 25)), (4, 1, np.array([0, 49]), None)]
    for source in (fortran, files):  # C-order files fall back to the memory map
        plain = NpyTimeSeriesDataset(source, **options)
        threaded = NpyTimeSeriesDataset(source, read_threads=3, **options)
        for index in indices:
            (signal, positions, _), (expected, expected_positions, _) = threaded[index], plain[index]
            np.testing.assert_array_equal(signal, expected)
            assert signal.flags.c_contiguous and signal.dtype == np.float32
            for got, want in zip(positions, expected_positions, strict=True):
                np.testing.assert_array_equal(got, want)
    copy = pickle.loads(pickle.dumps(threaded))  # DataLoader workers: the pool is rebuilt per process
    assert copy._pool is None
    np.testing.assert_array_equal(copy[indices[0]][0], plain[indices[0]][0])


def test_threaded_reads_only_use_whole_channel_major_files(tmp_path):
    """Partial views and in-memory arrays are read through the memory map path."""
    from timber.data.dataset import _whole_channel_major_file  # noqa: PLC0415

    series = np.arange(60, dtype=np.float32).reshape(10, 6)
    np.save(tmp_path / "f.npy", np.asfortranarray(series))
    np.save(tmp_path / "c.npy", series.T.copy())  # (channel, time) C order: the same bytes
    mapped = np.load(tmp_path / "f.npy", mmap_mode="r")
    assert _whole_channel_major_file(mapped) == (str(tmp_path / "f.npy"), mapped.offset)
    assert _whole_channel_major_file(np.load(tmp_path / "c.npy", mmap_mode="r").T) is not None
    assert _whole_channel_major_file(mapped[:, 2:]) is None
    assert _whole_channel_major_file(mapped[3:]) is None
    assert _whole_channel_major_file(np.asfortranarray(series)) is None
    assert _whole_channel_major_file(np.load(tmp_path / "c.npy", mmap_mode="r")) is None


def test_resampling_changes_with_the_epoch(files):
    """The sampler tags its indices with the epoch, which seeds the dataset's resampling."""
    dataset = NpyTimeSeriesDataset(files, resample_time_window=True, resample_channels=True, num_resample_channels=20)
    values = (1, 0, np.array([0, 1, 2, 3]), np.arange(0, 10))
    epoch_0, epoch_1 = ChunkIndex(values, 0), ChunkIndex(values, 1)
    assert epoch_0 == values and pickle.loads(pickle.dumps(epoch_1)).epoch == 1  # reaches the workers
    np.testing.assert_array_equal(dataset[epoch_0][0], dataset[values][0])  # epoch 0: the plain seed
    draws = [dataset[ChunkIndex(values, epoch)][1] for epoch in (1, 1, 2)]
    for got, expected in zip(draws[0], draws[1], strict=True):
        np.testing.assert_array_equal(got, expected)  # reproducible within an epoch
    assert not np.array_equal(draws[0][2], draws[2][2]) or not np.array_equal(draws[0][3], draws[2][3])

    sampler = PartitionSampler(**sampler_kwargs())
    sampler.set_epoch(3)
    assert all(isinstance(index, ChunkIndex) and index.epoch == 3 for index in sampler)


def test_dataset_checks(files):
    with pytest.raises(ValueError, match="Missing labels"):
        NpyTimeSeriesDataset(files, labels={"u0": [1.0]})
    with pytest.raises(ValueError, match="File keys"):
        normalise_file_keys({("u0",): "x.npy"})


# --- Data module ---------------------------------------------------------------------------------
def config(**overrides) -> dict:
    values = {
        "sampler": sampler_kwargs(),
        "train_entities": ["u0", "u1", "u2", "u3"],
        "val_entities": ["u4"],
        "test_entities": ["u5"],
    }
    values.update(overrides)
    return values


def test_batches_and_labels(files):
    labels = {f"u{u}": [float(u)] for u in range(NUM_ENTITIES)}
    datamodule = TimberDataModule(config(), files, labels=labels)
    datamodule.setup()
    signal, positions, batch_labels = next(iter(datamodule.train_dataloader()))
    assert isinstance(positions, SamplePositions)
    assert signal.shape == (2, 2, 1, 1, 8, 24) and positions.channel.shape == (2, 2, 1, 1, 8)
    entities = [datamodule.train_dataset.entities[i] for i in positions.entity[:, 0, 0, 0, 0].tolist()]
    torch.testing.assert_close(batch_labels, torch.tensor([labels[u] for u in entities]))


def test_batch_sampler_gives_the_same_batches(files):
    plain, batched = TimberDataModule(config(), files), TimberDataModule(config(use_batch_sampler=True), files)
    plain.setup()
    batched.setup()
    for a, b in zip(plain.train_dataloader(), batched.train_dataloader(), strict=True):
        torch.testing.assert_close(a[0], b[0])


def test_threaded_reads_in_dataloader_workers(files, tmp_path):
    """``read_threads`` reaches the datasets and works in (spawned) DataLoader workers."""
    fortran = {}
    for key, path in files.items():
        fortran[key] = tmp_path / path.name
        np.save(fortran[key], np.asfortranarray(np.load(path)))
    plain = TimberDataModule(config(), fortran)
    threaded = TimberDataModule(config(read_threads=2, num_workers=2), fortran)
    plain.setup()
    threaded.setup()
    assert threaded.train_dataset is not None and threaded.train_dataset.read_threads == 2
    for a, b in zip(plain.train_dataloader(), threaded.train_dataloader(), strict=True):
        torch.testing.assert_close(a[0], b[0])


def test_no_labels(files):
    datamodule = TimberDataModule(config(), files)
    datamodule.setup()
    assert next(iter(datamodule.train_dataloader()))[2] is None


def test_entities_without_labels_are_excluded(files):
    datamodule = TimberDataModule(
        config(train_entities=None, val_entities=None, test_entities=None, num_train_entities=2),
        files,
        labels={"u0": [0.0], "u1": [1.0], "u2": [2.0]},
    )
    assert set(datamodule.train_entities) <= {"u0", "u1", "u2"}


def test_random_split_is_reproducible(files):
    values = config(
        train_entities=None,
        val_entities=None,
        test_entities=None,
        num_train_entities=3,
        num_val_entities=2,
        num_test_entities=1,
    )
    a, b = TimberDataModule(values, files), TimberDataModule(values, files)
    assert (a.train_entities, a.val_entities, a.test_entities) == (b.train_entities, b.val_entities, b.test_entities)
    assert not set(a.train_entities) & set(a.val_entities) and len(a.test_entities) == 1


def test_eval_sampler_overrides(files):
    datamodule = TimberDataModule(
        config(eval_sampler_overrides={"crop_shape": [0, 0, 0, 0]}, sampler=sampler_kwargs(max_batches=1)), files
    )
    assert datamodule.sampler_config("val").max_batches is None
    assert datamodule.sampler_config("val").crop_shape == [0, 0, 0, 0]
    assert datamodule.sampler_config("train").max_batches == 1


def test_unknown_keys_raise(files):
    with pytest.raises(ValueError, match="Unknown data module option"):
        TimberDataModule(config(size_training_set=3), files)
    with pytest.raises(ValueError, match="Unknown sampler option"):
        DataModuleConfig(sampler={**sampler_kwargs(), "size_chunk": [1]})


def test_workers(files):
    datamodule = TimberDataModule(config(num_workers=2, prefetch_factor=2), files)
    datamodule.setup()
    assert len(list(datamodule.train_dataloader())) == 12  # (4 entities / 2) x (2 recordings / 2) x (50 // 8)
