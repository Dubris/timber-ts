"""Patch tokenizer: cuts multichannel time series into patches and builds their position embeddings.

A sample is ``C`` channels x ``T`` timepoints. It is cut into ``P`` patches of
``patch_length`` samples (stride ``patch_stride``). Each patch gets a channel
embedding (per channel, trainable, optionally initialised from a given array, e.g.
a geometric embedding of the sensors) and optionally a Fourier time embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

TimeEmbeddingType = Literal["patch", "timestep"]


def fourier_features(positions: torch.Tensor, num_frequencies: int, max_resolution: int) -> torch.Tensor:
    """Fourier features of 1-D positions in ``[-1, 1]``.

    Frequencies are ``linspace(1, max_resolution / 2, num_frequencies)``. The
    channel order is ``[position, sin(pi f_1 x) ... sin(pi f_K x), cos(pi f_1 x) ... cos(pi f_K x)]``
    (same as the Perceiver Fourier encoding).

    Args:
        positions: ``(num_positions,)`` float positions.
        num_frequencies: Number of frequency bands ``K``.
        max_resolution: Resolution that sets the highest (Nyquist) frequency.

    Returns:
        ``(num_positions, 2 * num_frequencies + 1)``.

    """
    frequencies = torch.linspace(1.0, max_resolution / 2, num_frequencies, device=positions.device)
    angles = math.pi * (positions[:, None] * frequencies[None, :])
    return torch.cat([positions[:, None], torch.sin(angles), torch.cos(angles)], dim=-1)


def linear_positions(num_positions: int, device: torch.device | str | None = None) -> torch.Tensor:
    """``num_positions`` positions evenly spaced in ``[-1, 1]``."""
    return torch.linspace(-1.0, 1.0, num_positions, dtype=torch.float32, device=device)


def num_patches(num_timepoints: int, patch_length: int, patch_stride: int) -> int:
    """Number of whole patches in a window of ``num_timepoints`` samples."""
    return (num_timepoints - patch_length) // patch_stride + 1


def forecast_offset(patch_length: int, patch_stride: int) -> int:
    """Extra patch shift so that a forecast target does not overlap its query patch.

    ``0`` for non-overlapping patches (``patch_stride >= patch_length``); otherwise
    the number of following patches that still overlap the query patch.
    """
    if patch_stride >= patch_length:
        return 0
    return patch_length // patch_stride - 1 + int(patch_length % patch_stride > 0)


@dataclass
class TokenizerOutput:
    """Patches and position embeddings (batch axes flattened into one).

    Attributes:
        patches: ``(batch, channel, patch, patch_length)`` raw signal patches.
        channel_embeddings: ``(batch, channel, channel_dim)``.
        time_embeddings: ``(batch or 1, patch, time_dim)`` or ``None`` without time embedding.

    """

    patches: torch.Tensor
    channel_embeddings: torch.Tensor
    time_embeddings: torch.Tensor | None

    @property
    def num_channels(self) -> int:
        """Number of channels of the batch."""
        return self.patches.shape[1]

    @property
    def num_patches(self) -> int:
        """Number of patches per channel."""
        return self.patches.shape[2]

    def position_embeddings(self) -> torch.Tensor:
        """Channel and time embeddings of every patch, concatenated.

        Returns ``(batch, channel, patch, channel_dim + time_dim)``.
        """
        batch, channels, patches, _ = self.patches.shape
        channel = self.channel_embeddings[:, :, None, :].expand(batch, channels, patches, -1)
        if self.time_embeddings is None:
            return channel
        time = self.time_embeddings[:, None, :, :].expand(batch, channels, patches, -1)
        return torch.cat((channel, time), dim=-1)


class PatchTokenizer(nn.Module):
    """Patchify multichannel time series and build their position embeddings.

    Args:
        patch_length: Samples per patch.
        num_channels: Number of channels with a channel embedding (channel indices are in ``[0, num_channels)``).
        channel_embedding_dim: Channels of the channel embedding.
        patch_stride: Step between patch starts (defaults to ``patch_length``: no overlap).
        time_embedding_type: ``None`` (no time embedding), ``"patch"`` (Fourier features of the
            patch index within the window) or ``"timestep"`` (Fourier features of the absolute
            timestep of the patch start, out of ``time_resolution`` timesteps).
        num_time_frequencies: Number of Fourier frequency bands of the time embedding.
        time_resolution: Fourier resolution (highest frequency ``time_resolution / 2``); with
            ``"timestep"`` it is also the number of timesteps covered.
        channel_embedding_init: Optional ``(num_channels, channel_embedding_dim)`` array (or a path to a
            ``.npy`` file) to initialise the channel embedding.
        trainable_channel_embedding: Whether the channel embedding is trained.

    """

    timestep_table: torch.Tensor  # buffer, registered when time_embedding_type='timestep'

    def __init__(
        self,
        patch_length: int,
        num_channels: int,
        channel_embedding_dim: int,
        patch_stride: int | None = None,
        time_embedding_type: TimeEmbeddingType | None = None,
        num_time_frequencies: int = 0,
        time_resolution: int = 1,
        channel_embedding_init: np.ndarray | torch.Tensor | str | None = None,
        trainable_channel_embedding: bool = True,
    ) -> None:
        super().__init__()
        if time_embedding_type not in (None, "patch", "timestep"):
            msg = f"time_embedding_type must be None, 'patch' or 'timestep', got {time_embedding_type!r}."
            raise ValueError(msg)
        if time_embedding_type is not None and num_time_frequencies < 1:
            msg = "A time embedding needs num_time_frequencies >= 1."
            raise ValueError(msg)

        self.patch_length = patch_length
        # Non-overlapping patches unless a stride is given.
        self.patch_stride = patch_length if patch_stride is None else patch_stride
        self.time_embedding_type = time_embedding_type
        self.num_time_frequencies = num_time_frequencies if time_embedding_type is not None else 0
        self.time_resolution = time_resolution

        # Channel embedding (nn.Embedding default init: N(0, 1), as in the original implementation)
        self.channel_embedding = nn.Embedding(num_channels, channel_embedding_dim)
        if channel_embedding_init is not None:
            init = channel_embedding_init
            if isinstance(init, str):
                init = np.load(init)
            init = torch.as_tensor(np.asarray(init), dtype=torch.float32)
            if init.shape != (num_channels, channel_embedding_dim):
                expected = (num_channels, channel_embedding_dim)
                msg = f"channel_embedding_init must have shape {expected}, got {tuple(init.shape)}."
                raise ValueError(msg)
            # Start from the given embeddings (e.g. a spectral embedding of the cortex), then train them as the rest of
            # the model.
            with torch.no_grad():
                self.channel_embedding.weight.copy_(init)
        # Freeze the channel embeddings if asked (e.g. when they are fixed by a geometric embedding).
        self.channel_embedding.weight.requires_grad_(trainable_channel_embedding)

        # Fourier table over absolute timesteps (not saved: it is a function of the config)
        if time_embedding_type == "timestep":
            table = fourier_features(linear_positions(time_resolution), self.num_time_frequencies, time_resolution)
            self.register_buffer("timestep_table", table, persistent=False)

    @property
    def channel_embedding_dim(self) -> int:
        """Size of the channel embedding."""
        return self.channel_embedding.embedding_dim

    @property
    def time_embedding_dim(self) -> int:
        """``2 * num_time_frequencies + 1`` with a time embedding, else ``0``."""
        return 2 * self.num_time_frequencies + 1 if self.time_embedding_type is not None else 0

    @property
    def position_dim(self) -> int:
        """Channels of `TokenizerOutput.position_embeddings`."""
        return self.channel_embedding_dim + self.time_embedding_dim

    @property
    def forecast_offset(self) -> int:
        """Sample offset of the forecasting targets (see `forecast_offset`)."""
        return forecast_offset(self.patch_length, self.patch_stride)

    def forward(
        self,
        signal: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> TokenizerOutput:
        """Cut the signal into patches and embed their positions.

        Args:
            signal: ``(*batch, channel, time)``. Trailing timepoints that do not fill a whole patch are dropped.
            channel_indices: ``(*batch, channel)`` or ``(channel,)`` indices into the channel embedding.
                ``None`` means channels ``0 .. channel - 1``.
            timesteps: ``(*batch, time)`` absolute timestep of every sample; required for
                ``time_embedding_type="timestep"``.

        Returns:
            `TokenizerOutput` with the batch axes flattened.

        """
        *batch_shape, num_channels, num_timepoints = signal.shape
        patches_per_window = num_patches(num_timepoints, self.patch_length, self.patch_stride)
        used_timepoints = self.patch_length + (patches_per_window - 1) * self.patch_stride

        # Flatten the batch axes so that every entity and view is one row; drop samples that do not fill a patch.
        signal = signal[..., :used_timepoints].reshape(-1, num_channels, used_timepoints)
        # Cut every channel into windows: a view with no copy.
        patches = signal.unfold(-1, self.patch_length, self.patch_stride)  # (batch, channel, patch, patch_length)
        batch_size = patches.shape[0]

        if channel_indices is None:
            channel_indices = torch.arange(num_channels, device=signal.device)
        channel_indices = channel_indices.expand(*batch_shape, num_channels).reshape(batch_size, num_channels)
        # Look up the embedding of each channel; the same embedding is used for all its patches.
        channel_embeddings = self.channel_embedding(channel_indices.long())

        time_embeddings = None
        if self.time_embedding_type == "patch":
            # Position of each patch in the window, as Fourier features (only if the time embedding is on).
            positions = linear_positions(patches_per_window, device=signal.device)
            time_embeddings = fourier_features(positions, self.num_time_frequencies, self.time_resolution)[None]
        elif self.time_embedding_type == "timestep":
            if timesteps is None:
                msg = "timesteps are required when time_embedding_type='timestep'."
                raise ValueError(msg)
            patch_starts = torch.arange(patches_per_window, device=signal.device) * self.patch_stride
            # Absolute time of the start of each patch, used to look up the Fourier features in the table.
            start_timesteps = timesteps.reshape(batch_size, -1)[:, patch_starts].long()
            time_embeddings = self.timestep_table[start_timesteps]  # (batch, patch, time_dim)

        if time_embeddings is not None:
            time_embeddings = time_embeddings.to(dtype=patches.dtype)
        return TokenizerOutput(patches=patches, channel_embeddings=channel_embeddings, time_embeddings=time_embeddings)
