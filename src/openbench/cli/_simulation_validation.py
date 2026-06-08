"""Shared validation helpers for simulation inputs used by CLI commands."""

from __future__ import annotations

import os
import re
from pathlib import Path

from openbench.config.schema import OpenBenchConfig

_UNRESOLVED_ENV_RE = re.compile(r"(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%)")


def simulation_root_errors(cfg: OpenBenchConfig) -> list[tuple[str, str]]:
    """Return validation errors for local simulation root directories."""
    errors: list[tuple[str, str]] = []
    for label, entry in cfg.simulation.items():
        raw = str(entry.root_dir).strip()
        if not raw:
            errors.append((label, "Simulation root is empty"))
            continue

        expanded = os.path.expandvars(os.path.expanduser(raw))
        if _UNRESOLVED_ENV_RE.search(expanded):
            errors.append(
                (
                    label,
                    f"Simulation root contains unresolved environment variable: {raw}",
                )
            )
            continue

        path = Path(expanded)
        if not path.exists() or not path.is_dir():
            errors.append((label, f"Simulation root does not exist: {path}"))

    return errors
