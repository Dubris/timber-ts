"""Unit tests of :mod:`timber.modules` (synthetic data, no fixture)."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from timber.modules import (
    Contextualizer,
    CrossAttentionDecoder,
    MultiHeadAttention,
    PatchMask,
    PatchTokenizer,
    PerceiverEncoder,
    PredictionHead,
    forecast_offset,
    forecasting_queries,
    gather_patches,
    random_patch_mask,
    token_dropout,
)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def small_contextualizer(**overrides) -> Contextualizer:
    kwargs = {"patch_length": 4, "position_dim": 6, "hidden_dim": 16, "output_dim": 12, "num_layers": 2, "num_heads": 2}
    kwargs.update(overrides)
    return Contextualizer(**kwargs).eval()


def small_encoder(**overrides) -> PerceiverEncoder:
    kwargs = {"patch_dim": 12, "num_latents": 8, "latent_dim": 16, "num_blocks": 2, "num_self_attention_layers": 1}
    kwargs.update(overrides)
    return PerceiverEncoder(**kwargs).eval()


# --- Masking -------------------------------------------------------------------------------------
def test_random_patch_mask_counts_and_indices():
    generator = torch.Generator().manual_seed(0)
    mask = random_patch_mask(5, 20, 0.7, generator=generator)
    assert mask.keep.shape == (5, 20)
    assert (mask.keep.sum(dim=1) == 6).all()
    assert mask.masked_indices.shape == (5, 14) and mask.kept_indices.shape == (5, 6)
    assert (mask.kept_indices.diff(dim=1) > 0).all() and (mask.masked_indices.diff(dim=1) > 0).all()
    assert mask.keep.gather(1, mask.kept_indices).all() and not mask.keep.gather(1, mask.masked_indices).any()


def test_random_patch_mask_is_deterministic_with_generator():
    a = random_patch_mask(3, 10, 0.5, generator=torch.Generator().manual_seed(1))
    b = random_patch_mask(3, 10, 0.5, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a.keep, b.keep)


def test_patch_mask_from_keep_round_trip():
    mask = random_patch_mask(4, 12, 0.25, generator=torch.Generator().manual_seed(2))
    rebuilt = PatchMask.from_keep(mask.keep.float())
    assert torch.equal(rebuilt.kept_indices, mask.kept_indices)
    assert torch.equal(rebuilt.masked_indices, mask.masked_indices)
    with pytest.raises(ValueError, match="same number"):
        PatchMask.from_keep(torch.tensor([[True, False], [True, True]]))


def test_gather_patches():
    x = torch.arange(24.0).view(2, 4, 3)
    out = gather_patches(x, torch.tensor([[0, 2], [1, 3]]))
    assert torch.equal(out, torch.stack([x[0, [0, 2]], x[1, [1, 3]]]))


def test_token_dropout_drops_whole_tokens():
    x = torch.ones(64, 10, 4)
    out = token_dropout(x, 0.5, training=True, generator=torch.Generator().manual_seed(0))
    per_token = out[..., 0]
    assert ((out == 0).all(-1) | (out == 2).all(-1)).all()  # whole token dropped or rescaled
    assert 0.3 < (per_token == 0).float().mean() < 0.7
    assert torch.equal(token_dropout(x, 0.5, training=False), x)


# --- Attention -----------------------------------------------------------------------------------
@pytest.mark.parametrize("is_causal", [False, True])
def test_attention_explicit_path_matches_sdpa(is_causal):
    torch.manual_seed(0)
    attention = MultiHeadAttention(16, key_value_dim=16, num_heads=4).eval()
    x = torch.randn(3, 7, 16)
    fused, none = attention(x, is_causal=is_causal)
    explicit, weights = attention(x, is_causal=is_causal, need_weights=True)
    assert none is None and weights.shape == (3, 4, 7, 7)
    torch.testing.assert_close(fused, explicit, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 4, 7))
    if is_causal:
        assert (weights.triu(diagonal=1) == 0).all()


def test_encoder_gather_equals_masked_attention():
    """Selecting the kept patches is the same as masking the others out of the cross-attention."""
    torch.manual_seed(0)
    encoder = small_encoder(num_blocks=1)
    patches = torch.randn(2, 10, 12)
    mask = random_patch_mask(2, 10, 0.6, generator=torch.Generator().manual_seed(0))
    gathered = encoder(patches, kept_indices=mask.kept_indices).latents

    layer = encoder.blocks[0].cross_attention
    latents = encoder.latents.expand(2, -1, -1)
    hidden, _ = layer.attention(
        layer.query_norm(latents),
        layer.key_value_norm(patches),
        attn_mask=mask.keep[:, None, None, :],
    )
    hidden = hidden + latents
    hidden = hidden + layer.mlp(layer.mlp_norm(hidden))
    for self_attention in encoder.blocks[0].self_attention_layers:
        hidden, _ = self_attention(hidden)
    torch.testing.assert_close(gathered, hidden, atol=1e-5, rtol=1e-5)


# --- Contextualizer --------------------------------------------------------------------------------
@pytest.mark.parametrize("attention_order", ["channel_time", "time_channel"])
def test_contextualizer_is_causal_in_time(attention_order):
    """Changing a future patch must not change the contextualised past patches."""
    torch.manual_seed(0)
    model = small_contextualizer(attention_order=attention_order, is_causal=True)
    patches, position = torch.randn(2, 5, 6, 4), torch.randn(2, 5, 6, 6)
    perturbed = patches.clone()
    perturbed[:, :, 4:] += 10.0  # patches 4 and 5, every vertex
    out, out_perturbed = model(patches, position), model(perturbed, position)
    torch.testing.assert_close(out[:, :, :4], out_perturbed[:, :, :4], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out[:, :, 4:], out_perturbed[:, :, 4:])

    non_causal = small_contextualizer(attention_order=attention_order, is_causal=False)
    assert not torch.allclose(non_causal(patches, position)[:, :, :4], non_causal(perturbed, position)[:, :, :4])


def test_contextualizer_shared_weights_have_no_aliased_keys(tmp_path):
    shared, unshared = (
        small_contextualizer(share_attention_weights=True),
        small_contextualizer(share_attention_weights=False),
    )
    shared_keys = set(shared.state_dict())
    assert not any("channel_attention" in k or "time_attention" in k for k in shared_keys)
    assert len(unshared.state_dict()) > len(shared_keys)
    path = tmp_path / "contextualizer.safetensors"
    save_file(shared.state_dict(), path)  # would fail on shared tensors
    assert set(load_file(path)) == shared_keys


def test_contextualizer_without_layers_and_mask():
    model = small_contextualizer(num_layers=0, output_dim=None)
    patches, position = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 6)
    keep = torch.ones(1, 3, 4, dtype=torch.bool)
    keep[0, 0, 1] = False
    out = model(patches, position, patch_keep=keep)
    torch.testing.assert_close(out[0, 0, 1], model.position_projection(position[0, 0, 1]))
    torch.testing.assert_close(out[0, 1], model(patches, position)[0, 1])


@pytest.mark.parametrize("module", ["contextualizer", "encoder"])
def test_gradient_checkpointing_gives_the_same_gradients(module):
    torch.manual_seed(0)
    if module == "contextualizer":
        reference, inputs = small_contextualizer(), (torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 6))
    else:
        reference, inputs = small_encoder(), (torch.randn(2, 10, 12),)
    checkpointed = small_contextualizer() if module == "contextualizer" else small_encoder()
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.use_gradient_checkpointing = True
    grads = []
    for model in (reference, checkpointed):
        model.train()  # dropout is 0, so train mode is deterministic
        out = model(*inputs)
        out = out if isinstance(out, torch.Tensor) else out.latents
        out.square().mean().backward()
        grads.append({name: p.grad.clone() for name, p in model.named_parameters()})
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], atol=1e-6, rtol=1e-5, msg=name)


# --- Tokenizer, decoder, heads ---------------------------------------------------------------------
def test_tokenizer_shapes_and_overlap_offset():
    tokenizer = PatchTokenizer(
        patch_length=8,
        num_channels=32,
        channel_embedding_dim=6,
        patch_stride=4,
        time_embedding_type="patch",
        num_time_frequencies=3,
        time_resolution=64,
    )
    signal = torch.randn(2, 3, 5, 50)  # (*batch, vertex, time)
    vertices = torch.randint(0, 32, (2, 3, 5))
    out = tokenizer(signal, vertices)
    assert out.patches.shape == (6, 5, 11, 8)  # (50 - 8) // 4 + 1 = 11 patches
    assert out.position_embeddings().shape == (6, 5, 11, 6 + 7)
    torch.testing.assert_close(out.patches[0, 0, 1], signal[0, 0, 0, 4:12])
    torch.testing.assert_close(out.channel_embeddings[4], tokenizer.channel_embedding.weight[vertices[1, 1]])
    assert tokenizer.forecast_offset == forecast_offset(8, 4) == 1
    assert forecast_offset(8, 8) == 0 and forecast_offset(8, 3) == 2


def test_tokenizer_timestep_embedding_uses_patch_starts():
    tokenizer = PatchTokenizer(
        patch_length=4,
        num_channels=3,
        channel_embedding_dim=2,
        patch_stride=2,
        time_embedding_type="timestep",
        num_time_frequencies=2,
        time_resolution=100,
    )
    timesteps = torch.arange(10, 22)[None, :]  # one window of 12 samples starting at t = 10
    out = tokenizer(torch.randn(1, 3, 12), timesteps=timesteps)
    torch.testing.assert_close(out.time_embeddings[0], tokenizer.timestep_table[torch.tensor([10, 12, 14, 16, 18])])


def test_decoder_and_forecasting_queries_shapes():
    decoder = CrossAttentionDecoder(query_dim=12, latent_dim=16, output_dim=4)
    contextualized = torch.randn(2, 3, 6, 12)
    queries = forecasting_queries(contextualized, shift=1)
    assert queries.shape == (2, 15, 12)
    prediction, _ = decoder(queries, torch.randn(2, 8, 16))
    assert prediction.shape == (2, 15, 4)


@pytest.mark.parametrize("pooling", ["mean", "weighted_mean", "cls_token", "attention"])
@pytest.mark.parametrize("fusion_stage", ["early", "late", "loss"])
def test_prediction_head_shapes(pooling, fusion_stage):
    head = PredictionHead(16, num_tokens=9, output_dim=3, pooling=pooling, fusion_stage=fusion_stage).eval()
    out = head(torch.randn(2 * 4, 9, 16), num_entities=2)
    assert out["prediction"].shape == ((2, 3) if fusion_stage == "early" else (8, 3))
    assert out["pooled"].shape == (8, 16)


# --- Devices -------------------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_pipeline_forward_backward_on_device(device):
    """Tokenizer -> contextualizer -> encoder -> decoder on each available device (bf16 autocast on MPS)."""
    torch.manual_seed(0)
    tokenizer = PatchTokenizer(
        patch_length=4,
        num_channels=20,
        channel_embedding_dim=6,
        time_embedding_type="patch",
        num_time_frequencies=2,
        time_resolution=32,
    ).to(device)
    contextualizer = small_contextualizer(position_dim=6 + 5).to(device).train()
    encoder = small_encoder(use_cls_token=True, latent_dropout_ratio=0.1).to(device).train()
    decoder = CrossAttentionDecoder(query_dim=12, latent_dim=16, output_dim=4).to(device)

    signal = torch.randn(2, 5, 24, device=device)
    vertices = torch.randint(0, 20, (2, 5), device=device)
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "mps")
    with autocast:
        tokens = tokenizer(signal, vertices)
        contextualized = contextualizer(tokens.patches, tokens.position_embeddings())
        mask = random_patch_mask(2, 5 * 6, 0.7, device=device)
        latents = encoder(contextualized.flatten(1, 2), kept_indices=mask.kept_indices).latents
        prediction, _ = decoder(forecasting_queries(contextualized, 1), latents[:, :8])
        loss = (prediction.float() - tokens.patches[:, :, 1:].flatten(1, 2)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in encoder.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_attention_dropout_without_autograd(device):
    """Train-mode dropout under no_grad (a frozen layer during finetuning) works on every device."""
    attention = MultiHeadAttention(16, num_heads=2, dropout=0.5).to(device).train()
    x = torch.randn(3, 7, 16, device=device)
    with torch.no_grad():
        out, weights = attention(x, is_causal=True)
    assert out.shape == (3, 7, 16) and weights is None and torch.isfinite(out).all()
