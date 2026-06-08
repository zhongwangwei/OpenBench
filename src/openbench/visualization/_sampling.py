"""Plot-only sampling helpers for expensive distribution figures."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SAMPLES = 50_000
_DEFAULT_KDE_MAX_SAMPLES = 10_000


def _plot_options(option: dict[str, Any] | None) -> dict[str, Any]:
    if not option:
        return {}
    nested = option.get("visualization")
    if isinstance(nested, dict):
        merged = dict(option)
        merged.update(nested)
        return merged
    return option


def _limit_for(option: dict[str, Any], purpose: str) -> int:
    default = _DEFAULT_KDE_MAX_SAMPLES if purpose == "kde" else _DEFAULT_MAX_SAMPLES
    key = "kde_max_samples" if purpose == "kde" else "max_samples_per_series"
    try:
        return int(option.get(key, option.get("max_samples_per_series", default)))
    except (TypeError, ValueError):
        return default


def sample_series_for_plot(
    data: Any,
    option: dict[str, Any] | None = None,
    *,
    purpose: str = "distribution",
) -> np.ndarray:
    """Return a deterministic sample for plotting a distribution series."""
    opts = _plot_options(option)
    if str(opts.get("plotting_mode", "balanced")).lower() == "full":
        return np.asarray(data)

    values = np.asarray(data)
    if values.size == 0:
        return values

    limit = _limit_for(opts, purpose)
    if limit <= 0 or values.size <= limit:
        return values

    method = str(opts.get("sample_method", "stride")).lower()
    if method != "stride":
        logger.warning("Unknown plot sample method %r; using stride", method)
    step = max(1, values.size // limit)
    sampled = values[::step][:limit]
    logger.info("Sampled plot series from %s to %s values using stride", values.size, sampled.size)
    return sampled


def sample_distribution_series(
    series: Iterable[Any],
    option: dict[str, Any] | None = None,
    *,
    purpose: str = "distribution",
) -> list[np.ndarray]:
    """Apply ``sample_series_for_plot`` to each distribution series."""
    return [sample_series_for_plot(data, option, purpose=purpose) for data in series]
