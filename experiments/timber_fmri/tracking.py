"""List the MLflow experiments and runs of a file-based tracking folder (no MLflow server needed).

Command line (default folder: the ``mlruns`` path of ``experiments/paths.yaml``)::

    uv run python -m experiments.timber_fmri.tracking [path/to/mlruns]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .paths import PATHS


def format_timestamp(milliseconds: int | None) -> str:
    """MLflow timestamp (milliseconds since the epoch) -> ``YYYY-mm-dd HH:MM:SS UTC``."""
    if milliseconds is None:
        return "N/A"
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S %Z")


def read_mlruns(mlruns: str | Path | None = None) -> dict[str, dict]:
    """``{experiment_id: {"name", "created_at", "runs": [{"run_id", "run_name", "status", ...}]}}``."""
    mlruns = Path(mlruns) if mlruns is not None else PATHS.mlruns
    results = {}
    for experiment_dir in sorted(mlruns.iterdir()):
        meta_path = experiment_dir / "meta.yaml"
        if not experiment_dir.is_dir() or experiment_dir.name.startswith(".") or not meta_path.exists():
            continue  # .trash, models, files
        experiment = yaml.safe_load(meta_path.read_text())
        runs = []
        for run_dir in sorted(experiment_dir.iterdir()):
            run_meta_path = run_dir / "meta.yaml"
            if not run_meta_path.exists():
                continue
            run = yaml.safe_load(run_meta_path.read_text())
            runs.append(
                {
                    "run_id": run.get("run_id", run_dir.name),
                    "run_name": run.get("run_name", "unknown"),
                    "status": run.get("status", "unknown"),
                    "created_at": format_timestamp(run.get("start_time")),
                    "ended_at": format_timestamp(run.get("end_time")),
                }
            )
        results[experiment.get("experiment_id", experiment_dir.name)] = {
            "name": experiment.get("name", "unknown"),
            "created_at": format_timestamp(experiment.get("creation_time")),
            "runs": runs,
        }
    return results


def print_mlruns(mlruns: str | Path | None = None) -> None:
    """Print every experiment and its runs (id, name, status, start and end times)."""
    for experiment_id, experiment in read_mlruns(mlruns).items():
        print(f"\n{'=' * 60}")
        print(f"Experiment: {experiment['name']}  (id={experiment_id})")
        print(f"Created at: {experiment['created_at']}")
        print(f"  {'Run ID':<32} {'Name':<20} {'Status':<12} {'Created at':<25} {'Ended at'}")
        print(f"  {'-' * 32} {'-' * 20} {'-' * 12} {'-' * 25} {'-' * 25}")
        for run in experiment["runs"]:
            print(
                f"  {run['run_id']:<32} {run['run_name']:<20} {run['status']:<12} "
                f"{run['created_at']:<25} {run['ended_at']}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mlruns", nargs="?", default=None, help="MLflow tracking folder (default: PATHS.mlruns).")
    print_mlruns(parser.parse_args(argv).mlruns)


if __name__ == "__main__":
    main()
