"""Tests of the Hugging Face models (:mod:`timber.modeling_timber`) and their configuration.

``save_pretrained`` / ``from_pretrained`` round trip, readable nested ``config.json``, Auto
classes, loading a pretrained backbone into a finetuning model, strict and validated
configuration.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForPreTraining

from timber import (
    DecoderConfig,
    HeadConfig,
    TimberConfig,
    TimberForPrediction,
    TimberForPretraining,
    TimberModel,
)


def tiny_config(**overrides) -> TimberConfig:
    kwargs = {
        "tokenizer": {
            "patch_length": 8,
            "num_channels": 32,
            "channel_embedding_dim": 16,
            "time_embedding_type": "patch",
            "num_time_frequencies": 4,
            "time_resolution": 128,
        },
        "contextualizer": {"hidden_dim": 24, "output_dim": 24, "num_heads": 2},
        "encoder": {"num_latents": 16, "latent_dim": 32, "num_self_attention_layers": 2, "num_self_attention_heads": 2},
        "decoders": [{"objective": "forecasting_1"}, {"objective": "reconstruction", "loss_weight": 0.5}],
    }
    kwargs.update(overrides)
    return TimberConfig(**kwargs)


def tiny_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    signal = torch.randn(2, 2, 8, 48, generator=generator)  # 2 subjects x 2 views, 8 vertices, 48 samples
    vertices = torch.randint(0, 32, (2, 2, 8), generator=generator)
    return signal, vertices


# --- Hugging Face behaviour ------------------------------------------------------------------------
def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = TimberForPretraining(tiny_config()).eval()
    signal, vertices = tiny_batch()
    mask = model.timber.make_patch_mask(4, 8 * 6, signal.device, torch.Generator().manual_seed(0))
    expected = model(signal, vertices, patch_mask=mask)

    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").is_file()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "timber"
    assert config["encoder"]["num_latents"] == 16 and config["decoders"][1]["objective"] == "reconstruction"

    reloaded = TimberForPretraining.from_pretrained(tmp_path).eval()
    for key, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value), key
    torch.testing.assert_close(reloaded(signal, vertices, patch_mask=mask).loss, expected.loss)

    auto = AutoModelForPreTraining.from_pretrained(tmp_path)
    assert isinstance(auto, TimberForPretraining)
    assert isinstance(AutoModel.from_pretrained(tmp_path), TimberModel)
    assert isinstance(AutoConfig.from_pretrained(tmp_path), TimberConfig)


def test_pretrained_backbone_into_finetuning_model(tmp_path):
    torch.manual_seed(0)
    pretrained = TimberForPretraining(tiny_config())
    pretrained.save_pretrained(tmp_path)
    config = TimberConfig.from_pretrained(tmp_path, decoders=[], head={"num_outputs": 3}, patch_mask_ratio=0.0)
    model = TimberForPrediction.from_pretrained(tmp_path, config=config)
    for key, value in pretrained.timber.state_dict().items():
        assert torch.equal(model.timber.state_dict()[key], value), key
    output_layer = model.head.mlp.layers.output
    assert output_layer.weight.abs().sum() > 0 and torch.isfinite(output_layer.weight).all()

    signal, vertices = tiny_batch()
    out = model(signal, vertices, labels=torch.randn(2, 3))
    assert out.logits.shape == (2, 3) and out.view_logits.shape == (4, 3)
    out.loss.backward()


@pytest.mark.parametrize("fusion_stage", ["early", "late", "loss"])
def test_classification_and_fusion_shapes(fusion_stage):
    config = tiny_config(
        decoders=[],
        patch_mask_ratio=0.0,
        head={
            "task": "classification",
            "num_outputs": 2,
            "fusion_stage": fusion_stage,
            "pooling": "attention",
            "label_smoothing": 0.1,
        },
    )
    model = TimberForPrediction(config)
    signal, vertices = tiny_batch()
    out = model(signal, vertices, labels=torch.tensor([[0], [1]]))
    assert out.logits.shape == ((4, 2) if fusion_stage == "loss" else (2, 2))
    assert torch.isfinite(out.loss)


def test_finetuning_with_taco_and_first_block_latents():
    config = tiny_config(
        decoders=[{"objective": "forecasting_1", "loss_weight": 0.5}],
        patch_mask_ratio=0.0,
        head={"num_outputs": 3, "latent_block": 1},
    )
    model = TimberForPrediction(config)
    signal, vertices = tiny_batch()
    out = model(signal, vertices, labels=torch.randn(2, 3))
    assert set(out.loss_terms) == {"regression", "forecasting_1"}
    torch.testing.assert_close(out.loss, out.loss_terms["regression"] + 0.5 * out.loss_terms["forecasting_1"])


def test_embedding_extraction_without_masking():
    model = TimberModel(tiny_config()).eval()
    signal, vertices = tiny_batch()
    with torch.inference_mode():
        out = model(signal, vertices, mask_patches=False, output_attentions=True)
    assert out.last_hidden_state.shape == (4, 17, 32)  # 16 latents + CLS
    assert out.patch_keep is None
    assert out.cross_attentions[0].shape == (4, 1, 17, 8 * 6)
    assert out.self_attentions[1].shape == (4, 2, 17, 17)
    torch.testing.assert_close(out.self_attentions[0].sum(-1), torch.ones(4, 2, 17))


def test_no_encoder_ablation():
    config = tiny_config(encoder=None, decoders=[{"objective": "reconstruction", "dropout": 0.0}])
    model = TimberForPretraining(config)
    signal, vertices = tiny_batch()
    out = model(signal, vertices)
    assert out.last_hidden_state.shape == (4, 8 * 6, 24) and torch.isfinite(out.loss)


def test_gradient_checkpointing_toggle():
    model = TimberForPretraining(tiny_config())
    assert not model.is_gradient_checkpointing
    model.gradient_checkpointing_enable()
    assert model.timber.contextualizer.use_gradient_checkpointing and model.timber.encoder.use_gradient_checkpointing
    model.train()
    model(*tiny_batch()).loss.backward()
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


def test_channel_embedding_init():
    values = torch.randn(32, 16)
    model = TimberModel(tiny_config(), channel_embedding_init=values.numpy())
    assert torch.equal(model.tokenizer.channel_embedding.weight.detach(), values)


# --- Configuration ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"encoder": {"num_latent": 8}}, "Unknown encoder option"),
        ({"enocder": {}}, "Unknown TimberConfig option"),
        ({"decoders": [{"objective": "forecasting_0"}]}, "Invalid forecasting objective"),
        ({"decoders": [{"objective": "reconstruction"}]}, "needs a time embedding"),
        ({"decoders": [], "head": {"pooling": "cls_token"}, "encoder": {"use_cls_token": False}}, "use_cls_token"),
        ({"decoders": [], "head": {"task": "classification", "num_outputs": 1}}, "num_outputs"),
        ({"decoders": [], "head": None}, "at least one decoder or a head"),
        ({"encoder": None}, "Without an encoder"),
        ({"patch_mask_ratio": 1.0}, "patch_mask_ratio"),
    ],
)
def test_invalid_configs_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TimberConfig(**kwargs)


def test_sub_configs_accept_dataclasses_and_dicts():
    a = TimberConfig(decoders=[DecoderConfig("forecasting_2")], head=HeadConfig(num_outputs=2))
    b = TimberConfig(decoders=[{"objective": "forecasting_2"}], head={"num_outputs": 2})
    assert a.to_dict() == b.to_dict()


def test_architecture_dict_rebuilds_the_config():
    config = TimberConfig(decoders=[{"objective": "forecasting_2"}], head={"task": "classification", "num_outputs": 2})
    options = config.architecture_dict()
    assert set(options) == {"tokenizer", "contextualizer", "encoder", "decoders", "head", "patch_mask_ratio"}
    assert TimberConfig(**options).to_dict() == config.to_dict()


def test_default_config_is_the_taco_setting():
    config = TimberConfig()
    assert config.objectives == ["forecasting_1"] and config.patch_mask_ratio == 0.7
    assert config.contextualizer.is_causal and config.encoder.use_cls_token
    assert config.patch_dim == 128 and config.encoder.num_tokens == 513
