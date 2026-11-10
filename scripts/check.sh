#!/usr/bin/env bash
# The local checks, defined once. Modes:
#
#   scripts/check.sh --staged    pre-commit hook: ruff (with fixes) on the staged Python files, pyright if src/timber changed
#   scripts/check.sh             full check: lint, types, tests, documentation (leaves the experiment environment installed)
#   scripts/check.sh --docs      documentation only, strict build (also run by .github/workflows/docs.yml)
#   scripts/check.sh --release   full check, then the wheel built and tested in a fresh venv
#
# Set UV_LOCKED=1 in CI so that a stale uv.lock fails the run instead of being updated.
set -euo pipefail
cd "$(dirname "$0")/.."

LINT_PATHS=(src tests scripts experiments)
mode="${1:-full}"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

if [[ "$mode" == "--staged" ]]; then
    staged=()
    while IFS= read -r file; do staged+=("$file"); done < <(git diff --cached --name-only --diff-filter=ACM -- '*.py')
    if [[ ${#staged[@]} -gt 0 ]]; then
        uv run --no-sync ruff check --fix "${staged[@]}"
        uv run --no-sync ruff format "${staged[@]}"
    fi
    if git diff --cached --name-only -- src/timber | grep -q .; then
        uv run --no-sync pyright
    fi
    exit 0
fi

if [[ "$mode" == "--docs" ]]; then
    step "documentation (strict build)"
    uv run --group docs mkdocs build --strict --site-dir "$(mktemp -d)"
    step "documentation checks passed"
    exit 0
fi

step "lint and format"
uv run --no-sync ruff check "${LINT_PATHS[@]}"
uv run --no-sync ruff format --check "${LINT_PATHS[@]}"

step "package and fMRI tests"
uv sync --quiet --group dev --group processing --group experiment --group docs
uv run --no-sync pyright
uv run --no-sync pytest -q

step "documentation (strict build)"
uv run --no-sync mkdocs build --strict --site-dir "$(mktemp -d)"

if [[ "$mode" == "--release" ]]; then
    step "wheel in a fresh venv (core dependencies only)"
    rm -rf dist
    uv build --wheel --quiet
    uvx twine check dist/*
    venv="$(mktemp -d)/venv"
    uv venv --quiet --python 3.12 "$venv"
    uv pip install --quiet --python "$venv/bin/python" dist/*.whl pytest

    # The application libraries are the experiment-only dependencies of pyproject.toml
    forbidden=()
    while IFS= read -r name; do forbidden+=("$name"); done < <(uv run --no-sync python - <<'PYEOF'
import re, tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text())
def names(specs):
    return {re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower().replace("_", "-") for spec in specs}

core = names(project["project"]["dependencies"])
experiment = [spec for spec in project["dependency-groups"]["experiment"] if isinstance(spec, str)]
for name in sorted(names(experiment) - core):
    print(name)
PYEOF
)
    if ! "$venv/bin/python" - "${forbidden[@]}" <<'PYEOF'
import importlib.metadata as metadata
import sys

present = []
for name in sys.argv[1:]:
    try:
        metadata.distribution(name)
        present.append(name)
    except metadata.PackageNotFoundError:
        pass
if present:
    print("The wheel pulls in: " + ", ".join(present), file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        exit 1
    fi
    (cd "$(mktemp -d)" && cp -r "$OLDPWD/tests" . && "$venv/bin/python" -m pytest -q -p no:cacheprovider tests)
fi

step "all checks passed"
