"""Validation helpers for plot inputs."""

from __future__ import annotations

import numpy as np


def finite_values(values, *, label: str) -> np.ndarray:
    """Return finite numeric values or fail before a renderer creates a misleading empty plot."""
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{label}: no finite data to plot")
    return finite


def finite_min_max(values, *, label: str, percentile: tuple[float, float] | None = None) -> tuple[float, float]:
    """Return robust finite min/max, widening constants for color normalization."""
    finite = finite_values(values, label=label)
    if percentile is None:
        min_value = float(np.nanmin(finite))
        max_value = float(np.nanmax(finite))
    else:
        min_value, max_value = (float(v) for v in np.nanpercentile(finite, percentile))
    if min_value == max_value:
        max_value = min_value + 1.0
    return min_value, max_value


def require_finite_columns(values, columns, *, label: str) -> None:
    """Require every 2-D plotted column/axis to contain finite data."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{label}: expected a 2-D array, got shape {array.shape}")
    if array.shape[1] != len(columns):
        raise ValueError(f"{label}: data columns {array.shape[1]} do not match labels {len(columns)}")
    for index, column in enumerate(columns):
        finite_values(array[:, index], label=f"{label}/{column}")
