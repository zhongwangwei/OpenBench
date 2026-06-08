"""Task-level evaluation dispatch helpers for the local runner."""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


def _local_attr(name: str, default: Any = None) -> Any:
    """Return monkeypatch-aware attributes from ``openbench.runner.local``."""
    local_runner = sys.modules.get("openbench.runner.local")
    if local_runner is not None and hasattr(local_runner, name):
        return getattr(local_runner, name)
    return default


def evaluation_task_worker_count(num_cores: Any, task_count: int) -> int:
    """Return task-level evaluation workers, bounded by user cores, CPUs, and work size."""
    if task_count <= 1:
        return 1
    try:
        requested = int(num_cores)
    except (TypeError, ValueError):
        requested = 1
    if requested <= 1:
        return 1
    os_module = _local_attr("os", os)
    cpu_limit = max(1, os_module.cpu_count() or 1)
    return min(requested, task_count, cpu_limit)


def task_level_parallel_safe(ready_tasks: list[dict[str, Any]]) -> bool:
    """Return whether task-level process parallelism is safe for these tasks."""
    try:
        from openbench.config.adapter import RunnerBindings
    except ImportError:  # pragma: no cover
        return False

    return all(
        isinstance(task.get("bindings"), RunnerBindings)
        and task.get("ref_data_type", "grid") == "grid"
        and task.get("sim_data_type", "grid") == "grid"
        for task in ready_tasks
    )


def evaluate_task_group(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate a serial group inside one process."""
    evaluate_single = _local_attr("_evaluate_single")
    if evaluate_single is None:  # pragma: no cover - defensive fallback for direct imports
        from openbench.runner.task_execution import evaluate_single
    return [evaluate_single(task) for task in tasks]


def unified_mask_parallel_groups(ready_tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group preprocessed unified-mask tasks by shared flat-ref mutation domain."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in ready_tasks:
        groups.setdefault((str(task.get("var_name", "")), str(task.get("ref_source", ""))), []).append(task)
    return list(groups.values())


def evaluate_ready_tasks(
    ready_tasks: list[dict[str, Any]],
    *,
    num_cores: Any,
    unified_mask: bool,
    only_drawing: bool,
    dask_distributed: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate ready tasks, parallelizing only when task independence is explicit."""
    if not ready_tasks:
        return []

    worker_count = _local_attr("_evaluation_task_worker_count", evaluation_task_worker_count)
    evaluate_single = _local_attr("_evaluate_single")
    parallel_safe = _local_attr("_task_level_parallel_safe", task_level_parallel_safe)
    group_tasks = _local_attr("_unified_mask_parallel_groups", unified_mask_parallel_groups)
    evaluate_group = _local_attr("_evaluate_task_group", evaluate_task_group)
    executor_cls = _local_attr("ProcessPoolExecutor", ProcessPoolExecutor)

    workers = worker_count(num_cores, len(ready_tasks))
    if dask_distributed:
        logger.info("Task-level process parallelism disabled while dask.distributed is active")
        return [evaluate_single(task) for task in ready_tasks]
    if only_drawing or workers <= 1 or not parallel_safe(ready_tasks):
        return [evaluate_single(task) for task in ready_tasks]

    if unified_mask:
        # Preprocessing already applied the shared mask. Keep tasks that share
        # a flat ref serial, but allow different refs to evaluate in parallel.
        if not all(task.get("ref_preprocessed") for task in ready_tasks):
            return [evaluate_single(task) for task in ready_tasks]
        groups = group_tasks(ready_tasks)
        group_workers = worker_count(num_cores, len(groups))
        if group_workers <= 1 or len(groups) <= 1:
            return [evaluate_single(task) for task in ready_tasks]
        logger.info(
            "Evaluating %d unified-mask ref group(s) in parallel with %d worker(s)",
            len(groups),
            group_workers,
        )
        with executor_cls(max_workers=group_workers) as executor:
            grouped_results = executor.map(evaluate_group, groups)
        return [result for group in grouped_results for result in group]

    logger.info("Evaluating %d independent grid task(s) in parallel with %d worker(s)", len(ready_tasks), workers)
    with executor_cls(max_workers=workers) as executor:
        return list(executor.map(evaluate_single, ready_tasks))
