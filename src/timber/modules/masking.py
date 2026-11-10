"""Vectorised patch masking and token dropout.

Every row masks the same number of patches, so the kept and masked patches can
be selected with `gather` into tensors of static shape (no boolean
indexing): this keeps shapes fixed for MPS and ``torch.compile``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatchMask:
    """A patch mask with the same number of masked patches in every row.

    Attributes:
        keep: ``(batch, num_patches)`` boolean, ``True`` for kept (visible) patches.
        kept_indices: ``(batch, num_kept)`` indices of the kept patches, increasing.
        masked_indices: ``(batch, num_masked)`` indices of the masked patches, increasing.

    """

    keep: torch.Tensor
    kept_indices: torch.Tensor
    masked_indices: torch.Tensor

    @classmethod
    def from_keep(cls, keep: torch.Tensor) -> PatchMask:
        """Build from a ``(batch, num_patches)`` keep mask (bool or 0/1) with a constant count per row."""
        keep = keep.bool()
        num_kept = int(keep[0].sum())
        if keep.numel() and not bool((keep.sum(dim=1) == num_kept).all()):
            msg = "Every row of the patch mask must keep the same number of patches."
            raise ValueError(msg)
        # A stable sort on (masked first / kept first) keeps the original order inside each group
        # Kept patches sort first (key 0), then masked ones (key 1); the stable sort keeps their order.
        order = torch.argsort((~keep).to(torch.uint8), dim=1, stable=True)
        return cls(keep=keep, kept_indices=order[:, :num_kept], masked_indices=order[:, num_kept:])


def num_masked_patches(num_patches: int, mask_ratio: float) -> int:
    """Number of patches masked per row (rounding down: ``int(num_patches * mask_ratio)``)."""
    if not 0.0 <= mask_ratio <= 1.0:
        msg = f"mask_ratio must be in [0, 1], got {mask_ratio}."
        raise ValueError(msg)
    return int(num_patches * mask_ratio)


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> PatchMask:
    """Mask ``int(num_patches * mask_ratio)`` random patches in each row, independently per row."""
    num_masked = num_masked_patches(num_patches, mask_ratio)
    # Draw independent noise per row: sorting it picks a random subset of patches for each row.
    noise = torch.rand(batch_size, num_patches, device=device, generator=generator)
    # The first ``num_masked`` indices of the sorted noise are the masked patches; the rest are kept.
    order = noise.argsort(dim=1)
    masked_indices = order[:, :num_masked].sort(dim=1).values
    kept_indices = order[:, num_masked:].sort(dim=1).values
    keep = torch.ones(batch_size, num_patches, dtype=torch.bool, device=noise.device)
    # Mark the masked patches as not kept (the boolean mask is the one returned to callers).
    keep.scatter_(1, masked_indices, False)
    return PatchMask(keep=keep, kept_indices=kept_indices, masked_indices=masked_indices)


def gather_patches(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select patches along axis 1: ``x (batch, num_patches, dim)``, ``indices (batch, k)`` -> ``(batch, k, dim)``."""
    # Gather along the patch axis, broadcasting the indices over the feature axis.
    return torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def token_dropout(x: torch.Tensor, p: float, training: bool, generator: torch.Generator | None = None) -> torch.Tensor:
    """Zero whole tokens with probability ``p`` and rescale the others by ``1 / (1 - p)``.

    Same behaviour as ``nn.Dropout1d`` on ``(batch, tokens, dim)``: ``x (batch, tokens, dim)``.
    """
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return torch.zeros_like(x)
    # Drop each token independently, then rescale so that the expected sum is unchanged.
    keep = torch.rand(x.shape[:2], device=x.device, generator=generator) >= p
    return x * keep.unsqueeze(-1).to(x.dtype) / (1.0 - p)
