# Paper experiments (fMRI application)

This folder holds the experiments of the TIMBER paper, which apply the package to cortical resting-state fMRI from the
Human Connectome Project (HCP) and ABIDE. The package itself is generic: everything specific to fMRI (file names,
measures, paths, configurations) lives in `experiments/timber_fmri`, and this folder only uses it.

## The five experiments

| Folder | Task | Start from |
|---|---|---|
| `1_learning_curve` | TACO pretraining on HCP, with 50 to 500 training subjects (one config per size) | scratch |
| `2_ablation_study` | Pretraining ablations: without TACO, causal masking, contextualizer, encoder, patch masking; one or two encoder blocks | scratch |
| `3_finetuning_hcp_intelligence` | Regression of the HCP cognitive scores; the tokenizer, contextualizer and first encoder block are frozen | pretrained (experiment 1) |
| `4_finetuning_hcp_gender` | Classification of gender on HCP | pretrained |
| `5_finetuning_abide_asd` | Classification of the diagnosis (`DX_GROUP`) on ABIDE | pretrained |

Each folder has the same layout:

```text
<n>_<name>/
  train.py          # training entry point (a few lines, calls experiments.timber_fmri.training.train)
  postprocess.py    # samples the trained model over the whole dataset (embeddings, attention, predictions)
  config/
    config.yaml     # the configuration of the run (Hydra), overrides the generated groups below
    model/          # architecture (experiment 1 and 2)
    trainer/        # Lightning Trainer settings
    datamodule/     # subject splits and sampler, one file per split or size
    make_config.py  # generates the datamodule and trainer files from the cluster data
```

## Input data

### HCP (experiments 1 to 4)

- **One file per subject and run**, stored as NumPy `.npy` arrays. The file name is
  `sub-<subject>_ses-<session>_run-<run>_*.npy`. The run index of the four runs of a subject is
  `2 * (session - 1) + (run - 1)`.
- **Shape:** `(1200, 59412)`, time by vertex, `float32`. 59412 is the fsLR 32k cortex without the medial wall. Files with
  the subcortex have 91282 columns (`keep_subcortex: true`, the default of the data module); the paper experiments use
  the cortex only.
- **Storage order:** files stored channel by channel (Fortran order) are read much faster when training samples random
  channel subsets. The converter `experiments/data/convert_to_channel_major.py` rewrites a folder in place, safely and
  resumably: `uv run python -m experiments.data.convert_to_channel_major --dry-run`, then without `--dry-run`.
- **Measures:** a CSV with a `Subject` column (the subject ids as text). Numeric measures are z-scored over the file; text
  measures are encoded as category codes. Subjects missing one of the used measures are dropped.
- **Complete subjects:** a subject enters an experiment only if all its runs are present.

### ABIDE (experiment 5)

- One file per subject and run, named `sub-*.npy` and read as `(time, vertex)`, with the phenotypic CSV of ABIDE. The
  measure used is `DX_GROUP`.

### Atlas and assets

- `experiments/data/` holds the assets: the spectral embedding of the cortical mesh (`spectral_embeddings_64.npy`), which
  initialises the channel embeddings, and the random parcellations (`random_parcellation_*.npy`) used to pool the
  samples. See `experiments/data/description.md`.
- The Schaefer atlas (used by the analysis to report atlas regions) is read from `schaefer_atlas` in the paths file.

## Machine paths: `paths.yaml`

Machine-specific paths are never written in the code or in the committed configurations. They are in
`experiments/paths.yaml`, which is git-ignored:

1. copy `experiments/paths.example.yaml` to `experiments/paths.yaml`;
2. set the keys for this machine (`hcp_data`, `hcp_people`, `hcp_measures`, `abide_data`, `abide_measures`,
   `schaefer_atlas`, `results`). `mlruns` and `assets` default to folders of the repository;
3. or point the environment variable `TIMBER_PATHS` to another file.

In Python, use `from experiments.timber_fmri.paths import PATHS` and then `PATHS.hcp_data`. In the YAML configurations,
use `${path:hcp_data}`, also inside a string (`${path:mlruns}/<experiment>/<run>/checkpoints/last.ckpt`). Relative paths
are relative to the repository root.

## Configuration

The configuration of a run is `config/config.yaml`. It lists its groups (`model`, `trainer`, `datamodule`) and then
its own values, which override the groups. The sections are:

| Section | Content |
|---|---|
| `name_experiment`, `name_run`, `seed` | MLflow experiment and run names, random seed |
| `pretrained` | a Hugging Face folder, or a Lightning checkpoint of these modules, to start from |
| `resume_from` | a Lightning checkpoint to resume (the MLflow run id is read from its path) |
| `freeze` / `discard` | parameter prefixes to freeze, and pretrained parameters not to load |
| `channel_embedding_init` | initial channel embeddings, when training from scratch |
| `model` | the `TimberConfig`; with `pretrained`, only the changes from the pretrained configuration |
| `training` | the `TimberTrainingConfig` (learning rates, epochs, ...) |
| `datamodule` | the subjects, the sampler and the loader (see `docs/data.md`) |
| `trainer` | arguments of the Lightning `Trainer` |
| `callbacks` | checkpoints, early stopping, periodic checkpoints |

The generated files of `datamodule/` and `trainer/` are created by `config/make_config.py`, which needs the cluster data.
They are git-ignored, so generate them on each machine.

## Training

From the repository root:

```bash
uv run python -m experiments.1_learning_curve.train
uv run python -m experiments.1_learning_curve.train training.max_epochs=2 datamodule.num_workers=4   # overrides
```

- The run is logged to MLflow in `mlruns` (`${path:mlruns}`). List the runs with
  `uv run python -m experiments.timber_fmri.tracking`, or open `mlflow ui --backend-store-uri experiments/mlruns`.
- Checkpoints are written to `<run>/checkpoints/`. The `callbacks.checkpoint` section chooses what is kept (for example the
  best one by `val/loss`, and optionally the last one).
- The data loader can read the files with several threads per worker (`datamodule.read_threads`), which helps on slow
  drives; `0` uses the memory map.

Experiment 2 has seven ablation configurations, `config/config_1.yaml` to `config_7.yaml`. Each is a complete
configuration; select one with `--config-name`:

```bash
uv run python -m experiments.2_ablation_study.train --config-name config_4
```

## Post-processing

`postprocess.py` loads a checkpoint, runs it over every subject of the dataset without masking, and saves the results
as an `xarray` dataset (a pickle):

```bash
uv run python -m experiments.1_learning_curve.postprocess --path_ckpt <checkpoint.ckpt> [--path_save <file.pkl>]
```

The results have one entry per subject and run, and per latent or channel (optionally averaged over the regions of an
atlas). They are read by the notebooks of the paper. `docs/data.md` describes how the batches are built. `analysis.py` documents what is sampled: the contextualised
patches, the latents of each block with their attention, and the predictions and pooled latents of a finetuned head.

## Modules of `experiments/timber_fmri`

| Module | Purpose |
|---|---|
| `data.py` | The fMRI data: file names, measures, `FmriDataset` and `FmriDataModule` (subjects are mapped to entities, runs to recordings, vertices to channels). |
| `paths.py` | `PATHS`: the machine paths of `paths.yaml`, and the `${path:...}` resolver of the YAML files. |
| `presets.py` | `paper_config()`: the architecture and sampler of the paper; `spectral_embeddings_path()`. |
| `training.py` | `train(cfg)`: builds the model, the data module, the logger and the callbacks, then fits. Used by every `train.py`. |
| `loading.py` | Loads a configuration, a model or pretrained weights from a Hugging Face folder or a Lightning checkpoint; lists the subjects of a pretraining run. |
| `analysis.py` | `model_outputs` (one forward pass returning an `Activations` object) and `sample_model` / `sample_model_finetuning` (the post-processing). |
| `callbacks.py` | `PlateauRelativeEarlyStopping`: stops when the relative improvement over a window of validations is small. |
| `configs.py` | `deep_merge` and `deep_diff` for configurations, and `split_subjects` for the subject lists. |
| `tracking.py` | Lists the MLflow runs of a folder without a server. |
| `tests/` | Tests of the application: file names, measures, paths, callbacks, and smoke runs of pretraining, finetuning and analysis on synthetic data. |

## Settings of the experiments

Experiment 1 uses the pretraining setting of the paper. The values are those of its configuration:

| Setting | Value |
|---|---|
| Patch length, channels per chunk, window | 60 samples, 1024 channels, 900 samples (the 1200-sample runs are cropped by 300) |
| Contextualizer: width, layers, heads, feed-forward ratio | 128, 3, 4, 2 |
| Encoder: latents, latent width, blocks, self-attention layers | 512, 256, 2, 4 |
| Decoder | TACO (`forecasting_1`), widening factor 1 |
| Patch masking ratio | 0.7 |
| Learning rate, weight decay, epochs | 2e-4, 0.01, 30 (no gradient clipping) |
| Batch | `chunks_per_batch` 2, 2, 2, 1: 8 runs (2 subjects, 2 runs, 2 channel sets) |

Experiments 3 to 5 start from the pretrained model and train a head. They freeze the tokenizer, the contextualizer, the
first encoder block and the latent queries (`freeze`), and they use mean pooling. The exact values of each experiment are
in its `config/config.yaml`.

Points worth knowing when reading the results:

- **Masking.** Each row of a batch hides exactly `floor(patch_mask_ratio * number of patches)` patches, which keeps the
  tensors of a fixed shape (see `docs/modules/encoder.md`).
- **Channel embedding.** The fMRI experiments start the channel embedding from a spectral embedding of the cortical mesh
  (`channel_embedding_init`), and training refines it.
- **Vertices.** The fsLR 32k cortex has 59,412 vertices without the medial wall (`NUM_CORTICAL_VERTICES`).
- **Probes and saliency.** The ridge probes are in `experiments/postprocessing.ipynb`; the integrated-gradient maps are in
  `experiments/exploration.ipynb`. The package provides the embeddings, the attention weights and the pooled latents
  (`experiments/timber_fmri/analysis.py`) that these analyses use.

## Tests

```bash
uv run --no-sync pytest experiments/timber_fmri
```

The tests use synthetic data. One of them checks that every path used in the configurations and scripts is documented in
`paths.example.yaml`.
