"""Plot-only point limiting helpers for dense diagram figures."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MAX_DIAGRAM_POINTS = 200
_FAST_MAX_DIAGRAM_POINTS = 75


def _plot_options(option: dict[str, Any] | None) -> dict[str, Any]:
    if not option:
        return {}
    nested = option.get("visualization")
    if isinstance(nested, dict):
        merged = dict(option)
        merged.update(nested)
        return merged
    return option


def _limit_for(option: dict[str, Any]) -> int | None:
    mode = str(option.get("plotting_mode", "balanced")).lower()
    if mode == "full":
        return None
    default = _FAST_MAX_DIAGRAM_POINTS if mode == "fast" else _DEFAULT_MAX_DIAGRAM_POINTS
    try:
        limit = int(option.get("max_diagram_points", default))
    except (TypeError, ValueError):
        return default
    if limit <= 0:
        return None
    return limit


def _sample_indices(count: int, limit: int, option: dict[str, Any]) -> np.ndarray:
    method = str(option.get("diagram_sample_method", "stride")).lower()
    if method != "stride":
        logger.warning("Unknown diagram sample method %r; using stride", method)
    step = max(1, count // limit)
    return np.arange(count)[::step][:limit]


def limit_diagram_points(
    series: Iterable[Any],
    labels: Iterable[Any],
    option: dict[str, Any] | None = None,
    *,
    has_reference: bool = False,
    context: str = "diagram",
) -> tuple[list[np.ndarray], list[Any]]:
    """Limit dense diagram point series while keeping labels aligned.

    Taylor diagrams pass statistics with the first element reserved for the
    reference field; set ``has_reference=True`` to preserve that element and
    sample only the simulation points.
    """
    arrays = [np.asarray(values) for values in series]
    label_list = list(labels)
    if not arrays:
        return [], label_list

    opts = _plot_options(option)
    limit = _limit_for(opts)
    if limit is None:
        return arrays, label_list

    offset = 1 if has_reference else 0
    point_count = max(0, len(arrays[0]) - offset)
    if label_list:
        point_count = min(point_count, len(label_list))
    if point_count <= limit:
        return arrays, label_list

    sampled = _sample_indices(point_count, limit, opts)
    limited_arrays = []
    for values in arrays:
        data = values[offset:]
        sampled_values = data[sampled]
        if has_reference:
            sampled_values = np.concatenate([values[:1], sampled_values])
        limited_arrays.append(sampled_values)

    limited_labels = [label_list[int(index)] for index in sampled] if label_list else []
    logger.info("Sampled %s points from %s to %s using stride", context, point_count, len(sampled))
    return limited_arrays, limited_labels
