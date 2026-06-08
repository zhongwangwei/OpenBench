"""Helpers for bounding expensive plot combination expansion."""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MAX_COMBINATIONS = 100
_FAST_MAX_COMBINATIONS = 25


def _plot_options(option: dict[str, Any] | None) -> dict[str, Any]:
    if not option:
        return {}
    nested = option.get("visualization")
    if isinstance(nested, dict):
        merged = dict(option)
        merged.update(nested)
        return merged
    return option


def _default_limit(plotting_mode: str) -> int | None:
    if plotting_mode == "full":
        return None
    if plotting_mode == "fast":
        return _FAST_MAX_COMBINATIONS
    return _DEFAULT_MAX_COMBINATIONS


def _max_combinations(option: dict[str, Any]) -> int | None:
    plotting_mode = str(option.get("plotting_mode", "balanced")).lower()
    if "max_combinations" not in option:
        return _default_limit(plotting_mode)
    try:
        limit = int(option["max_combinations"])
    except (TypeError, ValueError):
        return _default_limit(plotting_mode)
    if limit <= 0:
        return None
    return limit


def _total_combinations(groups: Sequence[Sequence[Any]]) -> int:
    return math.prod(len(group) for group in groups)


def limited_product(
    groups: Iterable[Iterable[Any]],
    option: dict[str, Any] | None = None,
    *,
    context: str = "plot combinations",
) -> Iterator[tuple[Any, ...]]:
    """Yield a deterministic, bounded cartesian product for expensive figures.

    ``plotting_mode=full`` keeps all combinations unless ``max_combinations`` is
    explicitly set. Other modes default to a cap to avoid accidental
    combinatorial explosions when each evaluation item has many references.
    """
    materialized_groups = [list(group) for group in groups]
    combinations = itertools.product(*materialized_groups)

    opts = _plot_options(option)
    limit = _max_combinations(opts)
    if limit is None:
        yield from combinations
        return

    total = _total_combinations(materialized_groups)
    if total > limit:
        logger.warning("Limiting %s from %s to %s combinations", context, total, limit)
        yield from itertools.islice(combinations, limit)
        return
    yield from combinations
