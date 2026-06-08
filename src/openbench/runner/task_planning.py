"""Task planning and cache-skip grouping for the local runner."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

TaskHashPayload = Callable[..., dict[str, Any]]
CachedTaskResult = Callable[[dict[str, Any]], dict[str, Any] | None]


def build_evaluation_tasks(
    *,
    cfg: Any,
    bindings: Any,
    output_dir: Path,
    metric_vars: list[str],
    score_vars: list[str],
    comparison_vars: list[str],
    statistic_vars: list[str],
    use_cache: bool,
    only_drawing: bool,
    task_hash_payload_fn: TaskHashPayload,
) -> list[dict[str, Any]]:
    """Build one runner task per variable/reference/simulation source triple."""
    from openbench.runner.cache import EvaluationCache, make_cache_key

    tasks: list[dict[str, Any]] = []
    for source in bindings.iter_task_sources(cfg.evaluation.variables):
        var_name = source.var_name
        sim_source = source.sim_source
        ref_source = source.ref_source
        cache_key = make_cache_key(var_name, sim_source, ref_source)
        config_hash = EvaluationCache.hash_config(
            task_hash_payload_fn(
                cfg=cfg,
                bindings=bindings,
                var_name=var_name,
                sim_source=sim_source,
                ref_source=ref_source,
                metric_vars=metric_vars,
                score_vars=score_vars,
                comparison_vars=comparison_vars,
                statistic_vars=statistic_vars,
            )
        )
        tasks.append(
            {
                "var_name": var_name,
                "sim_source": sim_source,
                "ref_source": ref_source,
                "bindings": bindings,
                "cache_key": cache_key,
                "config_hash": config_hash,
                "use_cache": use_cache,
                "update_cache": not only_drawing,
                "cache_dir": str(output_dir),
                "output_requirements": {
                    "metrics": metric_vars,
                    "scores": score_vars,
                },
            }
        )
        logger.info("Queued %s: sim=%s ref=%s", var_name, sim_source, ref_source)
    return tasks


def group_tasks_by_variable(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group planned tasks by evaluation variable."""
    var_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        var_tasks[task["var_name"]].append(task)
    return var_tasks


def collect_cached_results(
    var_tasks: dict[str, list[dict[str, Any]]],
    *,
    comparison_only: bool,
    only_drawing: bool,
    unified_mask: bool,
    cached_task_result_fn: CachedTaskResult,
) -> list[dict[str, Any]]:
    """Mark cache-skipped tasks and return reusable cached result records."""
    cached_results: list[dict[str, Any]] = []
    if comparison_only or only_drawing:
        return cached_results

    for vtasks in var_tasks.values():
        cached_for_var = [cached_task_result_fn(task) for task in vtasks]
        if cached_for_var and all(result is not None for result in cached_for_var):
            for task, result in zip(vtasks, cached_for_var):
                task["cache_skipped"] = True
                cached_results.append(result)
        elif not unified_mask:
            # Without unified_mask, tasks are independent. A cached sibling
            # should not touch source data merely because another sibling
            # must rerun; this keeps partial cache hits usable when source
            # files for the cached task are unavailable but complete
            # outputs already exist.
            for task, result in zip(vtasks, cached_for_var):
                if result is not None:
                    task["cache_skipped"] = True
                    cached_results.append(result)
    return cached_results
