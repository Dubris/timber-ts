# Data pipeline

`timber.data` turns time-series recordings into the batches the model is trained on. The examples of this section,
like the rest of the Training part, follow the fMRI application of the paper: each **entity** is a subject, each
**recording** a run, and each **channel** a cortical vertex.

```text
files  ->  NpyTimeSeriesDataset  ->  PartitionSampler  ->  TimberDataModule  ->  batches
(arrays)   (reads a window)         (draws the grid)      (splits the entities)  (signal, positions, labels)
```

## Input files

Each file is one **recording** of one **entity**, stored as a NumPy `.npy` array of shape `(time, channel)`, typically
`float32`. You describe the files with a mapping from keys to paths:

```python
files = {
    ("subject-01", 0): "data/subject-01_run-0.npy",   # (entity, recording)
    ("subject-01", 1): "data/subject-01_run-1.npy",
    ("subject-02", 0): "data/subject-02_run-0.npy",
}
```

**Storage order.** Random channel subsets read much faster from files stored channel by channel, that is in Fortran
order. Save such files with `np.save(path, np.asfortranarray(array))`: the shape stays `(time, channel)`, and both
orders are read correctly. The fMRI experiments include a converter for existing folders
(`experiments/data/convert_to_channel_major.py`).

## Sampling: a random grid of chunks

The paper never feeds a whole recording to the model. Each recording has about 60,000 channels and 1,200 time points,
and attention over all of them is out of reach. Instead, every training sample is a **random sub-grid** of the data:

- **Channels:** a random subset of 1,024 of the 59,412 cortical vertices (node sampling), so that the attention runs
  over 1,024 channels;
- **Time:** a random window of 900 of the 1,200 time points (the window is contiguous, so that the temporal attention
  sees a real time course).

Sampling is a computational necessity, and it also helps learning. The paper argues that nearby and functionally related
vertices are highly correlated, so a subset preserves the dominant spatiotemporal patterns: each draw is a noisy view of
the same recording. The model sees a different view at every step, which reduces overfitting and makes it more robust
to spatial and temporal perturbations (see the paper's sampling and patching section).

The sampler works on a grid with one axis per dimension: **entities**, **recordings**, **channels**, **time** and,
optionally, **shards**. Along each axis it cuts the indices into **chunks** of `chunk_shape` consecutive elements, and a
batch takes `chunks_per_batch` chunks along every axis. Each combination of chunks is one sample:

- a **channel chunk** is a set of channels of one recording (for example 32 of the 1,024 sampled channels);
- a **time chunk** is a window of consecutive time points (for example 128 samples of a 900-sample window).

The steps, in order: shuffle each axis if asked (`shuffle_before_chunking`), cut it into chunks, optionally stratify
the chunks by label (`labels`: one element of every label per chunk), pad the axis to a multiple of the batch size,
group the chunks into batches, and crop each chunk randomly (`crop_shape`, for example to take 900 of 1,200 samples).

| Setting | Meaning |
|---|---|
| `partition_shape` | Size of every axis: `(entities, recordings, channels, time)`. |
| `chunk_shape` | Elements per chunk along each axis. The paper: `(1, 1, 1024, 900)` for one entity, one recording, 1,024 channels and a 900-sample window. |
| `chunks_per_batch` | Chunks per batch along each axis; the batch holds the product of these. |
| `shuffle_before_chunking` | Shuffle each axis before cutting it into chunks. **Set it to `False` for time**, otherwise a chunk of time points is spread over the recording instead of forming a window. |
| `crop_shape` | Elements randomly cropped from each chunk. The paper crops 1,200 samples to 900 with `(0, 0, 0, 300)`. |
| `labels` | Stratify an axis by label, so that each chunk holds every class. |
| `resample_per_batch` | Redraw the indices of every batch along the chosen axes. |
| `seed` | Base seed; the batches of an epoch depend only on `seed` and the epoch number. |

The paper's batch of 8 runs is `chunks_per_batch = (2, 2, 2, 1)`: 2 subjects, 2 runs, 2 channel chunks of 1,024
vertices, one time window. `ChunkCollator` reshapes the samples back to `(*chunks_per_batch, channel, time)`.

```python
sampler = {
    "partition_shape": [100, 2, 59412, 1200],
    "chunk_shape": [1, 1, 1024, 900],                       # 1,024 random vertices, a 900-sample window
    "chunks_per_batch": [2, 2, 2, 1],
    "shuffle_before_chunking": [True, True, True, False],   # time windows stay contiguous
    "crop_shape": [0, 0, 0, 300],                           # 1,200 samples -> random 900-sample window
}
```

## Shards

A **shard** is a file that holds a subset of the channels of a recording. A recording stored as several shards
is one recording split by channel: the files `subject-01_run-0_shard-0.npy` and `subject-01_run-0_shard-1.npy` could
hold the first and the second half of the vertices. The key of a shard is `(entity, recording, shard)`, and
`channel_index` maps the local channels of each shard to the global channel indices, so that the model still knows
which vertex is which.

In the fMRI application, a shard is a set of cortical vertices of a recording. The paper draws 1,024 random vertices out
of about 60,000 for every sample; storing the recording split by vertex subsets gives the same channel split on disk.
The package reads any number of shards per recording. To sample across shards, add the shard count as a fifth
axis of `partition_shape` (see the example in [Usage](usage.md)); with four axes the sampler always reads shard 0.

```python
files = {
    ("subject-01", 0, 0): "data/subject-01_run-0_shard-0.npy",  # vertices 0 to 31 of the cortex
    ("subject-01", 0, 1): "data/subject-01_run-0_shard-1.npy",  # vertices 32 to 63
    ("subject-02", 0, 0): "data/subject-02_run-0_shard-0.npy",
    ("subject-02", 0, 1): "data/subject-02_run-0_shard-1.npy",
}
channel_index = {0: np.arange(0, 32), 1: np.arange(32, 64)}   # local channel -> global vertex, per shard
```

## Preprocessing the paper applies

The package does not preprocess the signals; the model expects them preprocessed as in the paper:

- **HCP** (S1200, resting state): the HCP minimal preprocessing pipeline (motion correction, spatial normalisation to
  MNI space, ICA-FIX denoising), a band-pass filter of **0.01–0.1 Hz**, and a **z-score per vertex**. The paper uses
  **948 subjects with complete data** (four runs of 1,200 time points, TR = 0.72 s) and pretrains on 200 of them.
  The cortex has 59,412 vertices without the medial wall (fsLR 32k).
- **ABIDE I**: 871 subjects (403 ASD, 468 controls), processed with C-PAC (motion correction, normalisation, nuisance
  regression). Each run is resampled to TR = 0.72 s and brought to 1,200 time points by repeating it, then band-pass
  filtered and z-scored per vertex, as for HCP.

## Datasets and data modules

- **`NpyTimeSeriesDataset`** reads one window per index. Its `load(path)` method can be overridden to read another
  storage layout. It can also resample the time window and the channels at random, deterministically for each index and
  epoch.
- **`TimberDataModule`** splits the entities into train, validation and test sets (by number, or by explicit lists) and
  builds one data loader per split. It is configured with `DataModuleConfig`.
- **Batches** are tuples `(signal, positions, labels)`: `signal` is the float32 `(view..., channel, time)` tensor,
  `positions` a `SamplePositions` with the entity, recording, channel and timestep of every sample, and `labels` the
  targets of the entities, or `None`.

Every option and its default is in the reference below.

## Reference

::: timber.data.sampler
    options:
      members:
        - PartitionSampler
        - ChunkIndex

::: timber.data.dataset
    options:
      members:
        - NpyTimeSeriesDataset
        - is_channel_major

::: timber.data.datamodule
    options:
      members:
        - SamplerConfig
        - DataModuleConfig
        - TimberDataModule
        - ChunkCollator
        - SamplePositions
