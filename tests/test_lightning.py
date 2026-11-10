"""Tests of the Lightning wrappers (:mod:`timber.lightning`) with synthetic data."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from transformers import get_cosine_schedule_with_warmup

from timber import TimberConfig, TimberForPrediction, TimberForPretraining
from timber.data import TimberDataModule
from timber.device import default_precision, get_accelerator
from timber.lightning import (
    TimberFinetuningModule,
    TimberPretrainingModule,
    TimberTrainingConfig,
    cosine_schedule_with_warmup,
)

NUM_CHANNELS = 32
ACCELERATORS = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict:
    """Synthetic arrays (6 entities x 2 recordings) with a regression score and a binary class per entity."""
    folder = tmp_path_factory.mktemp("arrays")
    rng = np.random.default_rng(0)
    files = {}
    for entity in range(6):
        for recording in range(2):
            path = folder / f"u{entity}_r{recording}.npy"
            np.save(path, rng.standard_normal((40, NUM_CHANNELS)).astype(np.float32))
            files[(f"u{entity}", recording)] = path
    labels = {
        "score": {f"u{u}": [rng.standard_normal()] for u in range(6)},
        "group": {f"u{u}": [float(u % 2)] for u in range(6)},
    }
    return {"files": files, "labels": labels}


def datamodule(data, labels=None) -> TimberDataModule:
    config = {
        "sampler": {
            "partition_shape": [6, 2, NUM_CHANNELS, 40],
            "chunk_shape": [1, 1, 8, 32],
            "chunks_per_batch": [2, 2, 1, 1],
            "crop_shape": [0, 0, 0, 8],
            "max_batches": 3,
        },
        "train_entities": ["u0", "u1", "u2", "u3"],
        "val_entities": ["u4", "u5"],
        "test_entities": [],
    }
    return TimberDataModule(config, data["files"], labels=data["labels"][labels] if labels else None)


def model_config(**overrides) -> dict:
    config = {
        "tokenizer": {"patch_length": 8, "num_channels": NUM_CHANNELS, "channel_embedding_dim": 8},
        "contextualizer": {"hidden_dim": 16, "output_dim": 16, "num_heads": 2, "num_layers": 1},
        "encoder": {"num_latents": 8, "latent_dim": 16, "num_blocks": 2, "num_self_attention_layers": 1},
        "decoders": [{"objective": "forecasting_1"}],
    }
    config.update(overrides)
    return config


def finetuning_config(task: str = "regression", **head) -> dict:
    head = {"task": task, "num_outputs": 1 if task == "regression" else 2, **head}
    return model_config(decoders=[], patch_mask_ratio=0.0, head=head)


def fit(module, data, tmp_path, accelerator="cpu", steps=2, **trainer_kwargs) -> Trainer:
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision=default_precision(accelerator),
        max_epochs=1,
        limit_train_batches=steps,
        limit_val_batches=1,
        logger=CSVLogger(tmp_path),
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        **trainer_kwargs,
    )
    trainer.fit(module, datamodule=data)
    return trainer


# --- Training runs -------------------------------------------------------------------------------
@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_pretraining_run(data, tmp_path, accelerator):
    module = TimberPretrainingModule(model_config(), {"learning_rate": 1e-3, "grad_norm_log_interval": 1})
    trainer = fit(module, datamodule(data), tmp_path, accelerator)
    logged = set(trainer.logged_metrics)
    assert {"train/loss_step", "train/loss_forecasting_1_step", "val/loss", "val/loss_forecasting_1"} <= logged
    assert any(name.startswith("grad_2.0_norm") for name in logged)
    assert torch.isfinite(trainer.logged_metrics["val/loss"])


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_gradient_clipping(data, tmp_path, accelerator):
    """Lightning cannot clip the gradients of a fused optimizer under mixed precision: AdamW is then not fused."""
    module = TimberPretrainingModule(model_config())
    trainer = fit(module, datamodule(data), tmp_path, accelerator, gradient_clip_val=5.0)
    assert not trainer.optimizers[0].defaults["fused"]
    assert torch.isfinite(trainer.logged_metrics["val/loss"])


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize(
    ("task", "labels", "metrics"),
    [
        ("regression", "score", {"val_mae", "val_mse", "val_pearson", "val/loss_prediction"}),
        ("classification", "group", {"val_auc", "val_accuracy", "val/loss_classification"}),
    ],
)
def test_finetuning_run(data, tmp_path, accelerator, task, labels, metrics):
    module = TimberFinetuningModule(finetuning_config(task, fusion_stage="early"))
    trainer = fit(module, datamodule(data, labels), tmp_path, accelerator)
    assert metrics <= set(trainer.logged_metrics)


def test_validation_uses_late_fusion(data):
    module = TimberFinetuningModule(finetuning_config(fusion_stage="loss")).eval()
    loaders = datamodule(data, "score")
    loaders.setup()
    batch = next(iter(loaders.train_dataloader()))
    assert module.forward_batch(batch, "train").logits.shape == (4, 1)  # 2 subjects x 2 views
    assert module.forward_batch(batch, "val").logits.shape == (2, 1)  # one per subject


# --- Optimisation --------------------------------------------------------------------------------
def test_parameter_groups_cover_every_trainable_parameter():
    module = TimberFinetuningModule(finetuning_config(pooling="weighted_mean"), {"learning_rate": 1e-4})
    frozen = module.freeze_parameters(["timber.tokenizer", "timber.contextualizer", "timber.encoder.blocks.0"])
    assert frozen and all(
        n.startswith(("timber.tokenizer", "timber.contextualizer", "timber.encoder.blocks.0")) for n in frozen
    )
    groups = {g["name"]: g for g in module.parameter_groups()}
    assert groups["head_decay"]["lr"] == pytest.approx(5e-4)
    assert groups["latents_no_decay"]["lr"] == pytest.approx(2e-5) and groups["latents_no_decay"]["weight_decay"] == 0
    grouped = {id(p) for g in groups.values() for p in g["params"]}
    for name, param in module.model.named_parameters():
        assert (id(param) in grouped) == param.requires_grad, name
    assert id(module.model.head.pooling.token_weights) in grouped
    no_decay = {id(p) for g in groups.values() if g["weight_decay"] == 0 for p in g["params"]}
    for name, param in module.model.named_parameters():
        if param.requires_grad and (name.endswith(".bias") or "_norm." in name):
            assert id(param) in no_decay, name


def test_pretraining_keeps_latents_in_the_backbone_group():
    groups = {g["name"] for g in TimberPretrainingModule(model_config()).parameter_groups()}
    assert groups == {"backbone_decay", "backbone_no_decay"}


def test_schedule_matches_transformers():
    def factors(make):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = make(optimizer)
        values = []
        for _ in range(100):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return values

    np.testing.assert_allclose(
        factors(lambda o: cosine_schedule_with_warmup(o, 10, 100)),
        factors(lambda o: get_cosine_schedule_with_warmup(o, 10, 100)),
        atol=1e-12,
    )


def test_unknown_training_option_raises():
    with pytest.raises(ValueError, match="Unknown training option"):
        TimberTrainingConfig.from_dict({"lr": 1e-3})


# --- Checkpoints, export, pretrained weights -----------------------------------------------------
def test_checkpoint_round_trip_and_export(data, tmp_path):
    module = TimberFinetuningModule(finetuning_config(), {"learning_rate": 3e-4})
    trainer = fit(module, datamodule(data, "score"), tmp_path)
    path = tmp_path / "module.ckpt"
    trainer.save_checkpoint(path)
    reloaded = TimberFinetuningModule.load_from_checkpoint(path)
    assert reloaded.training_config.learning_rate == 3e-4
    assert reloaded.config.to_dict()["head"] == module.config.to_dict()["head"]
    for key, value in module.model.state_dict().items():
        assert torch.equal(reloaded.model.state_dict()[key].cpu(), value.cpu()), key

    module.export_pretrained(tmp_path / "hf")
    exported = TimberForPrediction.from_pretrained(tmp_path / "hf")
    assert torch.equal(exported.head.mlp.layers.output.weight, module.model.head.mlp.layers.output.weight.cpu())

    # A Lightning checkpoint is also a valid pretrained source
    other = TimberFinetuningModule(finetuning_config())
    report = other.load_pretrained(path)
    assert not report["missing"] and not report["unexpected"]


def test_load_pretrained_from_hf_folder(tmp_path):
    pretrained = TimberForPretraining(TimberConfig(**model_config()))
    pretrained.timber.save_pretrained(tmp_path / "backbone")  # a bare TimberModel
    module = TimberFinetuningModule(finetuning_config())
    report = module.load_pretrained(tmp_path / "backbone")
    assert report["missing"] == ["head.mlp.layers.output.weight", "head.mlp.layers.output.bias"]
    assert torch.equal(module.model.timber.encoder.latents, pretrained.timber.encoder.latents)


def test_frozen_front_end_runs_without_autograd():
    module = TimberFinetuningModule(finetuning_config())
    module.freeze_parameters(["timber.tokenizer", "timber.contextualizer"])
    signal = torch.randn(2, 1, 8, 32)
    out = module.model.timber(signal, torch.randint(0, NUM_CHANNELS, (2, 1, 8)))
    assert not out.contextualized.requires_grad and out.last_hidden_state.requires_grad


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_frozen_modules_stay_in_eval_mode_during_fit(data, tmp_path, accelerator):
    module = TimberFinetuningModule(finetuning_config())
    module.freeze_parameters(["timber.tokenizer", "timber.contextualizer"])
    modes = []

    class RecordModes(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            timber = pl_module.model.timber
            modes.append((timber.contextualizer.training, timber.tokenizer.training, timber.encoder.training))

    trainer = fit(module, datamodule(data, "score"), tmp_path, accelerator)
    trainer.fit_loop.max_epochs = 2  # a second epoch: Lightning calls train() again after validation
    trainer.callbacks.append(RecordModes())
    trainer.fit(module, datamodule=datamodule(data, "score"))
    assert modes and all(mode == (False, False, True) for mode in modes)


def compile_works(device: str) -> bool:
    """Whether torch.compile can build kernels on ``device`` (the CPU backend needs a working C++ toolchain)."""
    try:
        torch.compile(lambda x: x * 2 + 1)(torch.ones(4, device=device))
    except Exception:
        return False
    return True


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_compile_layers_matches_eager(accelerator):
    if not compile_works(accelerator):
        pytest.skip(f"torch.compile does not work on {accelerator} here")
    torch.manual_seed(0)
    config = model_config()  # no dropout: compiled kernels draw other random numbers
    config["contextualizer"]["dropout"] = config["encoder"]["dropout"] = 0.0
    config["decoders"] = [{"objective": "forecasting_1", "dropout": 0.0}]
    eager = TimberForPretraining(TimberConfig(**config)).to(accelerator)
    compiled = copy.deepcopy(eager)
    assert compiled.compile_layers() == 1 + 2 * 2 + 1  # contextualizer layer, encoder blocks, decoder
    assert compiled.state_dict().keys() == eager.state_dict().keys()
    signal = torch.randn(2, NUM_CHANNELS, 32, device=accelerator)
    channels = torch.arange(NUM_CHANNELS, device=accelerator).expand(2, -1)
    losses = []
    for model in (eager, compiled):
        output = model(signal, channels, generator=torch.Generator(accelerator).manual_seed(0))
        output.loss.backward()
        losses.append(output.loss)
    torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=1e-4)
    for (name, a), b in zip(eager.named_parameters(), compiled.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)


def test_device_defaults():
    assert get_accelerator() in ("cuda", "mps", "cpu")
    assert default_precision("cpu") == "32-true"
    assert default_precision("mps") == default_precision("cuda") == "bf16-mixed"
    assert default_precision("auto") == default_precision(None) == default_precision(get_accelerator())
