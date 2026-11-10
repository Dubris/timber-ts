"""Device and precision defaults (CUDA > MPS > CPU)."""

from __future__ import annotations

import torch


def get_accelerator() -> str:
    """Best available accelerator: ``"cuda"``, ``"mps"`` or ``"cpu"``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_precision(accelerator: str | None = None) -> str:
    """Lightning precision for an accelerator: ``bf16-mixed`` on CUDA and MPS, ``32-true`` on CPU.

    ``None`` or ``"auto"`` use the best available accelerator (`get_accelerator`).
    """
    if accelerator in (None, "auto"):
        accelerator = get_accelerator()
    return "32-true" if accelerator == "cpu" else "bf16-mixed"
