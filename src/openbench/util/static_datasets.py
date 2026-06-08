"""Resolution helpers for OpenBench static classification datasets."""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

STATIC_DATASET_ENV = "OPENBENCH_DATASET_DIR"


def static_dataset_candidates(filename: str) -> list[Path]:
    """Return filesystem candidates checked for a static dataset."""
    candidates: list[Path] = []
    env_root = os.environ.get(STATIC_DATASET_ENV)
    if env_root:
        candidates.append(Path(env_root) / filename)
    candidates.append(Path("./dataset") / filename)
    return candidates


def static_dataset_exists(filename: str) -> bool:
    """Return whether a static dataset can be resolved from env, legacy CWD, or package resources."""
    if any(path.exists() and path.is_file() for path in static_dataset_candidates(filename)):
        return True
    try:
        return (files("openbench") / "dataset" / filename).is_file()
    except (FileNotFoundError, ModuleNotFoundError):
        return False


@contextmanager
def static_dataset_path(filename: str) -> Iterator[str]:
    """Yield a concrete path for a static dataset.

    Resolution order:
      1. ``$OPENBENCH_DATASET_DIR/<filename>``
      2. ``./dataset/<filename>`` for legacy source-tree runs
      3. packaged ``openbench/dataset/<filename>`` via ``importlib.resources``

    The packaged-resource branch uses ``as_file`` so it also works for
    zipped wheels.  On a complete miss, yield the legacy path so callers
    that open the dataset still raise a clear file-not-found error naming
    the expected legacy location.
    """
    candidates = static_dataset_candidates(filename)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            yield str(candidate)
            return

    try:
        traversable = files("openbench") / "dataset" / filename
        if traversable.is_file():
            with as_file(traversable) as concrete:
                yield str(concrete)
                return
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    yield str(candidates[-1])
