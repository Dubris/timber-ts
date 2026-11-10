"""Rewrite ``(time, channel)`` ``.npy`` time series channel by channel (Fortran order), in place.

Why: the training samples random subsets of ~1024 of ~60k channels. In the default C order, each
time row is ~240 KB and such a subset touches almost every page of every row: a chunk reads ~200 MB
to use ~3.5 MB. Stored channel by channel, it reads one contiguous run per channel. Measured on the
HCP files on an external USB SSD: 0.4 -> ~3.5 batches/s with 4 workers.

The arrays are unchanged (same shape, dtype and values; only the ``.npy`` header records
``fortran_order: True``), so every reader of the files keeps working and
:class:`timber.data.NpyTimeSeriesDataset` picks the fast path by itself.

Safe and resumable: each file is written to a temporary file next to it, read back and compared
with the original, and only then renamed over it. Files already in Fortran order are skipped,
leftover temporary files of an interrupted run are removed. Free space: one file.

Usage (default folder: ``hcp_data`` of ``experiments/paths.yaml``)::

    uv run python -m experiments.data.convert_to_channel_major --dry-run
    uv run python -m experiments.data.convert_to_channel_major [folder ...] [--pattern "sub-*.npy"]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

from timber.data import is_channel_major

logger = logging.getLogger(__name__)
TEMPORARY_SUFFIX = ".channel-major.tmp"


def needs_conversion(path: Path) -> bool:
    """Whether a file holds a 2D numeric array stored in C order."""
    series = np.load(path, mmap_mode="r")
    return series.ndim == 2 and series.dtype != object and series.shape[1] > 1 and not is_channel_major(series)


def convert_file(path: Path) -> None:
    """Rewrite one file in Fortran order (checked before it replaces the original)."""
    series = np.load(path)  # one sequential read
    temporary = path.with_name(path.name + TEMPORARY_SUFFIX)
    with temporary.open("wb") as file:
        np.save(file, np.asfortranarray(series))
        file.flush()
        os.fsync(file.fileno())
    written = np.load(temporary, mmap_mode="r")
    if not (is_channel_major(written) and written.dtype == series.dtype and np.array_equal(written, series)):
        temporary.unlink()
        msg = f"The channel-major copy of {path} differs from the original; the original is kept."
        raise RuntimeError(msg)
    del written
    os.replace(temporary, path)


def convert_folder(folder: Path, pattern: str = "sub-*.npy", dry_run: bool = False) -> dict[str, int]:
    """Convert every matching file of ``folder``; returns ``{"converted", "skipped"}`` counts."""
    for leftover in folder.glob(f"*{TEMPORARY_SUFFIX}"):
        logger.info("Removing %s (interrupted run)", leftover.name)
        if not dry_run:
            leftover.unlink()
    files = sorted(path for path in folder.glob(pattern) if not path.name.endswith(TEMPORARY_SUFFIX))
    todo = [path for path in files if needs_conversion(path)]
    logger.info("%s: %d files, %d to convert", folder, len(files), len(todo))
    if dry_run:
        return {"converted": 0, "skipped": len(files) - len(todo), "to_convert": len(todo)}
    start, done_bytes = time.perf_counter(), 0
    total_bytes = sum(path.stat().st_size for path in todo)
    for i, path in enumerate(todo, start=1):
        convert_file(path)
        done_bytes += path.stat().st_size
        elapsed = time.perf_counter() - start
        remaining = elapsed / done_bytes * (total_bytes - done_bytes)
        logger.info(
            "[%d/%d] %s (%.0f MiB/s, %.0f min left)",
            i,
            len(todo),
            path.name,
            done_bytes / 2**20 / elapsed,
            remaining / 60,
        )
    return {"converted": len(todo), "skipped": len(files) - len(todo)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path, help="Folders of .npy files (default: PATHS.hcp_data).")
    parser.add_argument("--pattern", default="sub-*.npy", help="Files to convert (glob, default: sub-*.npy).")
    parser.add_argument("--dry-run", action="store_true", help="Only count the files to convert.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.folders:
        from experiments.timber_fmri.paths import PATHS  # noqa: PLC0415

        args.folders = [PATHS.hcp_data]
    for folder in args.folders:
        logger.info("%s", convert_folder(folder, args.pattern, args.dry_run))


if __name__ == "__main__":
    main()
