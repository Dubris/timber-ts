"""The experiment configs compose and validate, and the training / analysis code runs end to end.

Synthetic fMRI files replace the cluster data; the experiment configs are composed with Hydra
and shrunk (tiny model, 2 training batches) before calling the real training entry point.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from experiments.timber_fmri import FmriDataset, find_files, load_measures
from experiments.timber_fmri.analysis import sample_model, sample_model_finetuning
from experiments.timber_fmri.loading import load_config, load_model
from experiments.timber_fmri.training import build_model_config, train
from timber import TimberForPrediction
from timber.data import SamplerConfig, build_dataloader

ROOT = Path(__file__).parents[2]
NUM_VERTICES = 40
TINY_MODEL = {
    "tokenizer": {"num_channels": NUM_VERTICES, "patch_length": 8, "channel_embedding_dim": 8},
    "contextualizer": {"hidden_dim": 16, "output_dim": 16, "num_heads": 2, "num_layers": 1},
    "encoder": {"num_latents": 8, "latent_dim": 16, "num_blocks": 2, "num_self_attention_layers": 1},
}
TINY_TRAINER = {
    "accelerator": "cpu",
    "devices": 1,
    "precision": "32-true",
    "limit_train_batches": 2,
    "limit_val_batches": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


def compose_experiment(name: str, config_name: str = "config", overrides: list[str] | None = None):
    """Compose an experiment config; defaults generated on the cluster (make_config.py) are stubbed empty."""
    with tempfile.TemporaryDirectory() as folder:
        config_dir = Path(folder) / "config"
        shutil.copytree(ROOT / name / "config", config_dir, ignore=shutil.ignore_patterns("*.py"))
        defaults = OmegaConf.load(config_dir / f"{config_name}.yaml").get("defaults", [])
        for entry in defaults:
            if isinstance(entry, str) or not entry:
                continue
            ((group, option),) = entry.items()
            stub = config_dir / group / f"{option}.yaml"
            if not stub.exists():
                stub.parent.mkdir(exist_ok=True)
                stub.write_text("{}\n")
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
            return compose(config_name=config_name, overrides=overrides or [])


@pytest.fixture(scope="module")
def fmri_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("fmri")
    rng = np.random.default_rng(0)
    for subject in range(6):
        for run in range(2):
            np.save(
                folder / f"sub-{subject:03d}_ses-1_run-{run + 1}_bold.npy",
                rng.standard_normal((40, NUM_VERTICES)).astype(np.float32),
            )
    pd.DataFrame(
        {
            "Subject": [f"{s:03d}" for s in range(6)],
            "score_a": rng.standard_normal(6),
            "score_b": rng.standard_normal(6),
            "score_c": rng.standard_normal(6),
            "Gender": ["F", "M"] * 3,
        }
    ).to_csv(folder / "measures.csv", index=False)
    return folder


def tiny_datamodule(fmri_dir: Path, measures=None) -> dict:
    return {
        "sampler": {
            "partition_shape": [6, 2, NUM_VERTICES, 40],
            "chunk_shape": [1, 1, 8, 32],
            "chunks_per_batch": [2, 2, 1, 1],
            "crop_shape": [0, 0, 0, 8],
        },
        "train_subjects": ["000", "001", "002", "003"],
        "val_subjects": ["004", "005"],
        "test_subjects": [],
        "path_data": str(fmri_dir),
        "keep_subcortex": False,
        "path_measures": str(fmri_dir / "measures.csv") if measures else None,
        "measures": measures,
    }


# --- Every committed config composes into a valid model configuration ---------------------------
@pytest.mark.parametrize(
    ("name", "config_name"),
    [
        ("1_learning_curve", "config"),
        *[("2_ablation_study", f"config_{k}") for k in range(1, 8)],
    ],
)
def test_pretraining_configs(name, config_name):
    cfg = compose_experiment(name, config_name)
    config = build_model_config(cfg)
    assert config.decoders and config.head is None
    from experiments.timber_fmri.data import FmriDataModule  # noqa: PLC0415

    options = OmegaConf.to_container(cfg.datamodule)
    assert {"sampler", "path_data", "train_subjects"} <= set(options)
    assert FmriDataModule  # the options are checked in the smoke runs below


@pytest.mark.parametrize("name", ["3_finetuning_hcp_intelligence", "4_finetuning_hcp_gender", "5_finetuning_abide_asd"])
def test_finetuning_configs(name, tmp_path):
    from experiments.timber_fmri.presets import paper_config  # noqa: PLC0415

    paper_config().save_pretrained(tmp_path)  # stands for the pretrained checkpoint
    cfg = compose_experiment(name, overrides=[f"pretrained={tmp_path}"])
    config = build_model_config(cfg)
    assert config.head is not None and not config.decoders and config.patch_mask_ratio == 0.0
    assert not config.encoder.use_cls_token and config.tokenizer.num_channels == 59412


# --- Smoke runs: pretraining, finetuning from its checkpoint, analysis ---------------------------
def shrink(cfg, fmri_dir: Path, measures=None, pretrained: str | None = None):
    with open_dict(cfg):
        if pretrained is None:
            cfg.model = OmegaConf.merge(cfg.model, TINY_MODEL)
        cfg.pretrained = pretrained
        cfg.channel_embedding_init = None
        cfg.datamodule = tiny_datamodule(fmri_dir, measures)
        cfg.training.max_epochs = 1
        cfg.training.compile = False  # inductor does not build on every CPU (covered in tests/test_lightning.py)
        cfg.trainer = TINY_TRAINER
    return cfg


@pytest.fixture(scope="module")
def pretrained_checkpoint(fmri_dir, tmp_path_factory) -> Path:
    workdir = tmp_path_factory.mktemp("run_pretraining")
    cfg = shrink(compose_experiment("1_learning_curve"), fmri_dir)
    trainer = train(cfg, tracking_uri=f"file:{workdir / 'mlruns'}")
    assert torch.isfinite(trainer.callback_metrics["val/loss"])
    return Path(trainer.checkpoint_callback.best_model_path)


def test_pretraining_checkpoint_layout(pretrained_checkpoint):
    assert pretrained_checkpoint.parent.name == "checkpoints"  # resume_from reads the MLflow run id from the path
    assert load_config(pretrained_checkpoint).tokenizer.num_channels == NUM_VERTICES


@pytest.fixture(scope="module")
def finetuned_checkpoint(fmri_dir, pretrained_checkpoint, tmp_path_factory) -> Path:
    workdir = tmp_path_factory.mktemp("run_finetuning")
    cfg = compose_experiment("3_finetuning_hcp_intelligence")
    cfg = shrink(cfg, fmri_dir, measures=["score_a", "score_b", "score_c"], pretrained=str(pretrained_checkpoint))
    trainer = train(cfg, tracking_uri=f"file:{workdir / 'mlruns'}")
    module = trainer.lightning_module
    assert {"val_mae", "val_pearson", "val/loss"} <= set(trainer.callback_metrics)
    assert not module.model.timber.contextualizer.signal_projection.weight.requires_grad  # frozen
    assert module.model.head.mlp.layers.output.weight.requires_grad
    return Path(trainer.checkpoint_callback.last_model_path)


def analysis_loader(fmri_dir: Path, shuffle_batches: bool = True):
    files = find_files(fmri_dir)
    labels, _ = load_measures(fmri_dir / "measures.csv", ["score_a", "Gender"], drop_incomplete=False)
    dataset = FmriDataset(files, labels=labels, measure_names=["score_a", "Gender"])
    sampler = SamplerConfig(
        partition_shape=[6, 2, NUM_VERTICES, 40],
        chunk_shape=[1, 1, 8, 40],
        chunks_per_batch=[1, 1, 5, 1],
        shuffle_before_chunking=[True, True, False, False],
        shuffle_batches=shuffle_batches,
    )
    return build_dataloader(dataset, sampler)


def test_analysis_of_the_finetuned_model(fmri_dir, finetuned_checkpoint, tmp_path):
    model = load_model(finetuned_checkpoint)
    assert isinstance(model, TimberForPrediction)
    results = sample_model(
        model,
        analysis_loader(fmri_dir),
        save_path=tmp_path,
        include=[
            "signal_space",
            "inputs",
            "latent_embeddings_2",
            "self_attentions_2",
            "cross_attentions_1",
            "head_prediction",
            "cls_token",
        ],
        dims_to_aggregate={"timepatch": "mean"},
    )
    assert (tmp_path / "sampled_data.nc").is_file()
    assert results["latent_embeddings_2"].shape == (6, 2, 8, 16)  # subjects, runs, latents (no CLS), dim
    assert results["inputs"].shape == (6, 2, NUM_VERTICES, 16)
    assert results["head_prediction"].shape == (6, 3)
    assert np.isfinite(results["self_attentions_2"].values).all()
    assert list(results.coords["subject"].values) == [f"{s:03d}" for s in range(6)]
    assert "score_a" in results.coords

    finetuning = sample_model_finetuning(
        model, analysis_loader(fmri_dir, shuffle_batches=False), include=["latent_embeddings_2", "cls_token"]
    )
    assert finetuning["cls_token"].shape == (6, 2, 5, 16)  # subjects, runs, views (5 channel chunks), dim


def test_path_references_are_documented():
    """Every ${path:...} of the committed configs and scripts is a key of paths.example.yaml (or a default)."""
    import re  # noqa: PLC0415

    from experiments.timber_fmri.paths import DEFAULTS, EXAMPLE_PATHS_FILE  # noqa: PLC0415

    documented = set(OmegaConf.load(EXAMPLE_PATHS_FILE)) | set(DEFAULTS)
    files = [*ROOT.glob("*/config/**/*.yaml"), *ROOT.glob("*/config/make_config.py"), *ROOT.glob("*/postprocess.py")]
    used = {name for file in files for name in re.findall(r"\$\{path:(\w+)", file.read_text())}
    used |= {name for file in files for name in re.findall(r"PATHS\.(\w+)", file.read_text())}
    assert used and used <= documented, used - documented
