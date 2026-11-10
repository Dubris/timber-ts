"""Tests of the fMRI application layer (file conventions, measures, presets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.timber_fmri import (
    NUM_CORTICAL_VERTICES,
    NUM_VERTICES_WITH_SUBCORTEX,
    FmriDataModule,
    FmriDataset,
    find_files,
    load_measures,
    paper_config,
    parse_file_name,
    spectral_embeddings_path,
)
from timber import TimberModel


def test_file_names():
    assert parse_file_name("sub-100307_ses-2_run-1_cortex.npy") == ("100307", 2, 0)  # run index 2 * (s - 1) + (r - 1)
    assert parse_file_name("sub-007_bold.npy") == ("007", 0, 0)  # one run per subject
    with pytest.raises(ValueError, match="Unexpected file name"):
        parse_file_name("subject7.npy")


def test_measures(tmp_path):
    pd.DataFrame(
        {"Subject": ["1", "2", "3", "4"], "age": [20.0, 30.0, 40.0, None], "sex": ["F", "M", "F", "M"]}
    ).to_csv(tmp_path / "m.csv", index=False)
    labels, categories = load_measures(tmp_path / "m.csv", ["age", "sex"])
    assert set(labels) == {"1", "2", "3"}  # subject 4 misses a measure
    np.testing.assert_allclose(labels["2"], [0.0, 1.0])  # age z-scored, sex category code
    assert list(categories["sex"]) == ["F", "M"]


def test_files_are_read_as_stored(tmp_path):
    array = np.arange(1200 * 3, dtype=np.float32).reshape(1200, 3)  # (time, vertex)
    np.save(tmp_path / "sub-1_ses-1_run-1_x.npy", array)
    dataset = FmriDataset({("1", 0): tmp_path / "sub-1_ses-1_run-1_x.npy"})
    signal, positions, _ = dataset[(0, 0, np.array([0, 2]), np.arange(5))]
    np.testing.assert_array_equal(signal, array[:5, [0, 2]].T)
    np.testing.assert_array_equal(positions[2], [0, 2])  # channels are the vertices of the file


def test_datamodule_vertex_range(tmp_path):
    for subject in range(3):
        np.save(tmp_path / f"sub-{subject}_ses-1_run-1_bold.npy", np.zeros((20, 10), dtype=np.float32))
    sampler = {"partition_shape": [3, 1, 10, 20], "chunk_shape": [1, 1, 4, 20], "chunks_per_batch": [1, 1, 1, 1]}
    datamodule = FmriDataModule(
        {
            "sampler": sampler,
            "path_data": str(tmp_path),
            "train_subjects": ["0", "1"],
            "val_subjects": ["2"],
            "test_subjects": [],
            "keep_subcortex": False,
        }
    )
    assert datamodule.train_subjects == ["0", "1"]
    assert datamodule.config.num_resample_channels == NUM_CORTICAL_VERTICES
    default = FmriDataModule(
        {
            "sampler": sampler,
            "path_data": str(tmp_path),
            "train_subjects": ["0"],
            "val_subjects": [],
            "test_subjects": [],
        }
    )
    assert default.config.num_resample_channels == NUM_VERTICES_WITH_SUBCORTEX


def test_paper_preset():
    config = paper_config()
    assert config.tokenizer.num_channels == NUM_CORTICAL_VERTICES and config.tokenizer.patch_length == 60
    assert config.objectives == ["forecasting_1"] and config.patch_mask_ratio == 0.7
    assert config.contextualizer.is_causal and config.encoder.use_cls_token
    assert config.patch_dim == 128 and config.encoder.num_tokens == 513
    path = spectral_embeddings_path(64)
    assert path.is_file()
    small = paper_config(encoder={"num_latents": 4, "latent_dim": 8, "num_blocks": 1, "num_self_attention_layers": 1})
    model = TimberModel(small, channel_embedding_init=str(path))
    np.testing.assert_allclose(model.tokenizer.channel_embedding.weight.detach().numpy(), np.load(path), rtol=1e-6)


def test_subject_keys_and_properties(tmp_path):
    for subject in range(3):
        np.save(tmp_path / f"sub-{subject}_ses-1_run-1_bold.npy", np.zeros((20, 10), dtype=np.float32))
    sampler = {"partition_shape": [3, 1, 10, 20], "chunk_shape": [1, 1, 4, 20], "chunks_per_batch": [1, 1, 1, 1]}
    options = {"sampler": sampler, "path_data": str(tmp_path), "keep_subcortex": False}
    datamodule = FmriDataModule({**options, "num_train_subjects": 2, "num_val_subjects": 1, "num_test_subjects": 0})
    datamodule.setup()
    assert len(datamodule.train_subjects) == 2 and datamodule.train_subjects == datamodule.train_entities
    assert datamodule.train_dataset.subjects == sorted(datamodule.train_subjects)
    with pytest.raises(ValueError, match="Use subject names"):
        FmriDataModule({**options, "train_entities": ["0"]})


def test_paths_file_and_resolver(tmp_path, monkeypatch):
    from omegaconf import OmegaConf  # noqa: PLC0415

    from experiments.timber_fmri.paths import PATHS, REPO_ROOT, Paths  # noqa: PLC0415

    file = tmp_path / "paths.yaml"
    file.write_text(f"hcp_data: {tmp_path}/hcp\nresults: relative/results\n")
    paths = Paths(file)
    assert paths.hcp_data == tmp_path / "hcp" and paths["results"] == REPO_ROOT / "relative/results"
    assert paths.assets == REPO_ROOT / "experiments/data"  # default
    with pytest.raises(AttributeError, match="No path 'abide_data'"):
        _ = paths.abide_data

    monkeypatch.setenv("TIMBER_PATHS", str(file))
    PATHS.reload()
    try:
        cfg = OmegaConf.create(
            {"path_data": "${path:hcp_data}", "ckpt": "${path:results}/run/last.ckpt", "emb": "${path:assets,emb.npy}"}
        )
        assert cfg.path_data == str(tmp_path / "hcp")
        assert cfg.ckpt == f"{REPO_ROOT}/relative/results/run/last.ckpt"
        assert cfg.emb == str(REPO_ROOT / "experiments/data/emb.npy")
    finally:
        monkeypatch.delenv("TIMBER_PATHS")
        PATHS.reload()


def test_plateau_early_stopping():
    import torch  # noqa: PLC0415

    from experiments.timber_fmri.callbacks import PlateauRelativeEarlyStopping  # noqa: PLC0415

    class FakeTrainer:
        def __init__(self) -> None:
            self.sanity_checking, self.should_stop, self.callback_metrics = False, False, {}

    trainer, callback = FakeTrainer(), PlateauRelativeEarlyStopping(window=2, min_rel_improvement=0.1)
    for value, stop in ((1.0, False), (0.8, False), (0.75, False), (0.74, True)):  # 1.0 -> 0.75: 25%; 0.8 -> 0.74: 7.5%
        trainer.callback_metrics = {"val/loss": torch.tensor(value)}
        callback.on_validation_end(trainer, None)  # type: ignore[arg-type]
        assert trainer.should_stop == stop
    restored = PlateauRelativeEarlyStopping(window=2)
    restored.load_state_dict(callback.state_dict())
    assert list(restored.values) == list(callback.values)


def test_list_mlruns(tmp_path, capsys):
    from experiments.timber_fmri.tracking import print_mlruns, read_mlruns  # noqa: PLC0415

    run = tmp_path / "12" / "abc"
    run.mkdir(parents=True)
    (tmp_path / "12" / "meta.yaml").write_text("experiment_id: '12'\nname: timber/test\ncreation_time: 0\n")
    (run / "meta.yaml").write_text("run_id: abc\nrun_name: split_1\nstatus: 3\nstart_time: 1000\nend_time: null\n")
    (tmp_path / ".trash").mkdir()
    runs = read_mlruns(tmp_path)
    assert runs["12"]["name"] == "timber/test" and runs["12"]["runs"][0]["run_name"] == "split_1"
    print_mlruns(tmp_path)
    assert "split_1" in capsys.readouterr().out


def test_convert_to_channel_major(tmp_path):
    from experiments.data.convert_to_channel_major import TEMPORARY_SUFFIX, convert_folder  # noqa: PLC0415
    from timber.data import is_channel_major  # noqa: PLC0415

    rng = np.random.default_rng(0)
    arrays = {f"sub-{i}_ses-1_run-1_bold.npy": rng.standard_normal((30, 12)).astype(np.float32) for i in range(3)}
    for name, array in arrays.items():
        np.save(tmp_path / name, array)
    (tmp_path / f"sub-0_ses-1_run-1_bold.npy{TEMPORARY_SUFFIX}").write_bytes(b"interrupted")

    assert convert_folder(tmp_path, dry_run=True)["to_convert"] == 3
    assert convert_folder(tmp_path) == {"converted": 3, "skipped": 0}
    for name, array in arrays.items():
        converted = np.load(tmp_path / name, mmap_mode="r")
        assert is_channel_major(converted) and converted.shape == array.shape
        np.testing.assert_array_equal(converted, array)
    assert not list(tmp_path.glob(f"*{TEMPORARY_SUFFIX}"))
    assert convert_folder(tmp_path) == {"converted": 0, "skipped": 3}  # resumable: done files are skipped

    # The fMRI dataset reads converted files like the original ones
    dataset = FmriDataset(find_files(tmp_path))
    signal, _, _ = dataset[(1, 0, np.array([2, 5]), np.arange(4, 10))]
    np.testing.assert_array_equal(signal, arrays["sub-1_ses-1_run-1_bold.npy"][4:10, [2, 5]].T)
