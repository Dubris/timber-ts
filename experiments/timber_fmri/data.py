"""fMRI data for TIMBER: maps preprocessed fMRI files to the generic `data` pipeline.

Conventions of this application:

- one ``.npy`` file per subject and run, ``(time, vertex)``, named ``sub-<subject>_ses-<session>_run-<run>_*.npy``
  for HCP (recording index ``2 * (session - 1) + (run - 1)``), or ``sub-<subject>_*.npy`` otherwise (one run);
- subject measures in a CSV with a ``Subject`` column (numeric measures z-scored, text measures
  encoded as category codes).

This application speaks of subjects: the fMRI configs use ``train_subjects``, ``num_train_subjects``,
... and `FmriDataModule` / `FmriDataset` expose ``train_subjects`` / ``subjects``. They map
to the vocabulary of the generic package as follows: subject -> entity, run -> recording,
vertex -> channel.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timber.data import FileKey, NpyTimeSeriesDataset, TimberDataModule

NUM_VERTICES_WITH_SUBCORTEX = 91282  # fsLR 32k grayordinates (cortex + subcortex)
NUM_CORTICAL_VERTICES = 59412  # fsLR 32k cortex without the medial wall

_PATTERN_HCP = re.compile(r"sub-(\d+)_ses-(\d+)_run-(\d+)_.+npy")
_PATTERN_SINGLE_RUN = re.compile(r"sub-(\w+?)_.*npy")


def parse_file_name(path: str | Path) -> FileKey:
    """``(subject, recording, 0)`` of a time-series file (see the module docstring for the names)."""
    name = Path(path).name
    if "_ses-" in name:
        match = _PATTERN_HCP.match(name)
        if match is None:
            msg = f"Unexpected file name {name!r} (expected sub-<id>_ses-<s>_run-<r>_*.npy)."
            raise ValueError(msg)
        subject, session, run = match.groups()
        return FileKey(subject, 2 * (int(session) - 1) + (int(run) - 1))
    match = _PATTERN_SINGLE_RUN.match(name)
    if match is None:
        msg = f"Unexpected file name {name!r} (expected sub-<id>_*.npy)."
        raise ValueError(msg)
    return FileKey(match.group(1), 0)


def subject_of(path: str | Path) -> str:
    """Subject id of a file (``sub-<id>_...``)."""
    return Path(path).name.split("_")[0][len("sub-") :]


def find_files(path_data: str | Path | None = None, files: Sequence[str | Path] | None = None) -> dict[FileKey, Path]:
    """``{(subject, recording, 0): path}`` of the ``sub-*.npy`` files of a folder (or of the given files)."""
    if files is None:
        if path_data is None:
            msg = "Give path_data or files."
            raise ValueError(msg)
        files = sorted(Path(path_data).glob("sub-*.npy"))
    return {parse_file_name(path): Path(path) for path in files}


def load_measures(
    path_measures: str | Path,
    measures: Sequence[str],
    drop_incomplete: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Subject measures from a CSV with a ``Subject`` column.

    Numeric measures are z-scored over the whole file and text measures encoded as category codes.
    Subjects missing any of the ``measures`` are left out, unless ``drop_incomplete=False``
    (their missing values are then NaN, or the code ``-1`` for text measures).

    Returns:
        ``({subject: values}, {text measure: categories})``.

    """
    raw = pd.read_csv(path_measures, dtype={"Subject": str}).set_index("Subject")[list(measures)]
    frame = raw.copy()
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = (frame[numeric] - frame[numeric].mean()) / frame[numeric].std()
    categories = {}
    for column in frame.select_dtypes(include=["object"]).columns:
        codes = frame[column].astype("category").cat
        frame[column] = codes.codes
        categories[column] = codes.categories
    complete = raw.notna().all(axis=1) if drop_incomplete else raw.index == raw.index
    labels = {subject: frame.loc[subject].to_numpy(dtype=np.float32) for subject in frame.index[complete]}
    return labels, categories


class FmriDataset(NpyTimeSeriesDataset):
    """`NpyTimeSeriesDataset` of the fMRI files.

    ``subjects`` are the subject ids (the generic ``entities``); ``measure_names`` names the label
    columns (used by the analysis code).
    """

    def __init__(self, *args: Any, measure_names: Sequence[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.measure_names = list(measure_names) if measure_names is not None else None

    @property
    def subjects(self) -> list[str]:
        return self.entities


_FMRI_OPTIONS = ("path_data", "files", "path_measures", "measures", "keep_subcortex")
_SPLITS = ("train", "val", "test")


def to_generic_split_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    """fMRI split options (``train_subjects``, ``num_train_subjects``, ...) -> generic ``*_entities`` options."""
    config = dict(config)
    generic = [key for split in _SPLITS for key in (f"{split}_entities", f"num_{split}_entities") if key in config]
    if generic:
        msg = f"Use subject names in the fMRI configs ({', '.join(generic)} -> *_subjects)."
        raise ValueError(msg)
    for split in _SPLITS:
        for prefix in ("", "num_"):
            if f"{prefix}{split}_subjects" in config:
                config[f"{prefix}{split}_entities"] = config.pop(f"{prefix}{split}_subjects")
    return config


class FmriDataModule(TimberDataModule):
    """Data module of the fMRI experiments.

    Args:
        config: The `DataModuleConfig` fields, with subjects for entities
            (``train_subjects`` / ``val_subjects`` / ``test_subjects`` and ``num_train_subjects``, ...;
            ``resample_channels`` resamples vertices), plus the fMRI options:

            - ``path_data`` / ``files``: folder of ``sub-*.npy`` files, or the files;
            - ``path_measures`` / ``measures``: subject measures used as labels;
            - ``keep_subcortex``: the files include the subcortex (91282 grayordinates) or only the
              cortex (59412 vertices), the range of vertices drawn by ``resample_channels``.

    """

    dataset_class = FmriDataset

    def __init__(self, config: Mapping[str, Any]) -> None:
        config = to_generic_split_keys(config)
        fmri = {key: config.pop(key) for key in _FMRI_OPTIONS if key in config}
        keep_subcortex = fmri.get("keep_subcortex", True)
        config.setdefault(
            "num_resample_channels",
            NUM_VERTICES_WITH_SUBCORTEX if keep_subcortex else NUM_CORTICAL_VERTICES,
        )
        files = find_files(fmri.get("path_data"), fmri.get("files"))
        labels = None
        if fmri.get("path_measures") is not None and fmri.get("measures") is not None:
            labels, self.measure_categories = load_measures(fmri["path_measures"], fmri["measures"])
        super().__init__(config, files, labels=labels)
        self.fmri_options = fmri

    def make_dataset(self, subjects: Sequence[str]) -> FmriDataset | None:
        dataset = super().make_dataset(subjects)
        if dataset is not None and self.labels is not None:
            dataset.measure_names = list(self.fmri_options.get("measures") or [])
        return dataset

    @property
    def train_subjects(self) -> list[str]:
        return self.train_entities

    @property
    def val_subjects(self) -> list[str]:
        return self.val_entities

    @property
    def test_subjects(self) -> list[str]:
        return self.test_entities
