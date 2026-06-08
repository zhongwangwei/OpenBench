"""Cache-hit validation helpers for local runner tasks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from openbench.runner.cache import EvaluationCache

logger = logging.getLogger(__name__)


def _local_attr(name: str, default: Any = None) -> Any:
    local_runner = sys.modules.get("openbench.runner.local")
    if local_runner is not None and hasattr(local_runner, name):
        return getattr(local_runner, name)
    return default


def cached_task_result(task: dict[str, Any]) -> dict[str, Any] | None:
    """Return a success result when a task is already cached and outputs exist."""
    if not task.get("use_cache") or task.get("cache_dir") is None:
        return None

    cache_key = task["cache_key"]
    config_hash = task["config_hash"]
    cache_dir = Path(task["cache_dir"])
    cache = EvaluationCache(cache_dir)
    if not cache.is_cached(cache_key, config_hash):
        return None

    has_complete_outputs = _local_attr("_has_complete_outputs")
    if has_complete_outputs(cache_dir, task):
        logger.info(
            "Cached, skipping %s: sim=%s ref=%s",
            task["var_name"],
            task["sim_source"],
            task["ref_source"],
        )
        return {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "cache_key": cache_key,
            "config_hash": config_hash,
            "skipped": True,
        }

    logger.warning(
        "Cache stale (output missing or unreadable), re-evaluating %s: sim=%s ref=%s",
        task["var_name"],
        task["sim_source"],
        task["ref_source"],
    )
    try:
        cache.invalidate(cache_key)
    except Exception as inv_err:
        logger.warning(
            "Cache invalidate failed for %s (sim=%s ref=%s): %s — proceeding with re-evaluation",
            task["var_name"],
            task["sim_source"],
            task["ref_source"],
            inv_err,
        )
    return None
