"""Machine-specific paths of the fMRI experiments, gathered in one YAML file.

Changing machine only means editing ``experiments/paths.yaml`` (git-ignored; copy
``experiments/paths.example.yaml``). The environment variable ``TIMBER_PATHS`` selects another file.

- Python: ``PATHS.hcp_data`` (a `Path`), ``PATHS["hcp_data"]``.
- YAML configs (Hydra / OmegaConf): ``${path:hcp_data}``, also inside a string
  (``${path:mlruns}/<experiment id>/<run id>/checkpoints/last.ckpt``) or with sub-paths
  (``${path:assets,spectral_embeddings_64.npy}``). The resolver is registered when
  `timber_fmri` is imported.

Relative paths in the file are relative to the repository root. ``assets`` (the files of
``experiments/data``) and ``mlruns`` (MLflow tracking folder) have defaults inside the repository.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPO_ROOT / "experiments" / "paths.yaml"
EXAMPLE_PATHS_FILE = REPO_ROOT / "experiments" / "paths.example.yaml"
DEFAULTS = {"assets": "experiments/data", "mlruns": "experiments/mlruns"}


class Paths(Mapping[str, Path]):
    """The named paths of a paths file (loaded on first access).

    Args:
        file: The YAML file (default: ``$TIMBER_PATHS``, else ``experiments/paths.yaml``).

    """

    def __init__(self, file: str | Path | None = None) -> None:
        self._file = Path(file) if file is not None else None
        self._paths: dict[str, Path] | None = None

    @property
    def file(self) -> Path:
        if self._file is not None:
            return self._file
        return Path(os.environ.get("TIMBER_PATHS", DEFAULT_PATHS_FILE))

    def _load(self) -> dict[str, Path]:
        if self._paths is None:
            values = dict(DEFAULTS)
            if self.file.is_file():
                values.update(OmegaConf.to_container(OmegaConf.load(self.file), resolve=True) or {})  # type: ignore[arg-type]
            self._paths = {name: self._absolute(value) for name, value in values.items() if value is not None}
        return self._paths

    @staticmethod
    def _absolute(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    def reload(self) -> None:
        """Read the file again on the next access (e.g. after editing it in a notebook)."""
        self._paths = None

    def __getitem__(self, name: str) -> Path:
        paths = self._load()
        if name not in paths:
            msg = (
                f"No path {name!r} in {self.file}. Add it there (see {EXAMPLE_PATHS_FILE.name}); "
                f"known paths: {sorted(paths)}."
            )
            raise KeyError(msg)
        return paths[name]

    def __getattr__(self, name: str) -> Path:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(str(error)) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


PATHS = Paths()


def resolve_path(name: str, *parts: str) -> str:
    """``${path:name[,sub,path]}``: the named path, optionally joined with sub-paths."""
    return str(PATHS[name].joinpath(*(str(part).strip() for part in parts)))


def register_resolver() -> None:
    """Register ``${path:...}`` with OmegaConf (idempotent)."""
    if not OmegaConf.has_resolver("path"):
        OmegaConf.register_new_resolver("path", resolve_path)


register_resolver()
