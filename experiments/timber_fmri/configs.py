"""Helpers for the experiment configurations."""

from __future__ import annotations

import copy
from collections.abc import Mapping


def deep_merge(base: Mapping, overrides: Mapping) -> dict:
    """Recursive dict merge (``overrides`` wins; lists are replaced)."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def deep_diff(full: Mapping, base: Mapping) -> dict:
    """The entries of ``full`` that differ from ``base`` (recursively), so that ``deep_merge(base, diff) == full``."""
    diff = {}
    for key, value in full.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = deep_diff(value, base[key])
            if nested:
                diff[key] = nested
        elif key not in base or base[key] != value:
            diff[key] = copy.deepcopy(value)
    return diff


_EARLIER_SPLIT_SIZES = {"train": "size_training_set", "val": "size_validation_set", "test": "size_test_set"}


def split_subjects(datamodule: Mapping, split: str) -> list[str]:
    """The ``split`` (``train``/``val``/``test``) subjects of a logged datamodule config (any version of the keys)."""
    subjects = datamodule.get(f"{split}_subjects", datamodule.get(f"{split}_entities"))
    return list(subjects) if subjects is not None else []


def num_split_subjects(datamodule: Mapping, split: str) -> int:
    """The number of ``split`` subjects of a logged datamodule config (any version of the keys)."""
    for key in (f"num_{split}_subjects", f"num_{split}_entities", _EARLIER_SPLIT_SIZES[split]):
        if datamodule.get(key) is not None:
            return int(datamodule[key])
    return len(split_subjects(datamodule, split))
