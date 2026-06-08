"""Monkeypatch-compatible preflight facade for legacy runner.local helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from openbench.runner import preflight


def _local_attr(name: str) -> Any:
    return getattr(sys.modules["openbench.runner.local"], name)


def make_phase_error(phase: str, message: str, **details: Any) -> dict[str, Any]:
    return preflight.make_phase_error(phase, message, **details)


def find_existing_outputs(output_dir: Path, task: dict[str, Any]) -> list[Path]:
    return preflight.find_existing_outputs(output_dir, task)


def task_output_requirement(task: dict[str, Any], key: str) -> list[str]:
    return preflight.task_output_requirement(task, key)


def task_output_data_types(task: dict[str, Any]) -> tuple[str, str]:
    return preflight.task_output_data_types(task, build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"))


def expected_output_paths(output_dir: Path, task: dict[str, Any]) -> list[Path]:
    return preflight.expected_output_paths(
        output_dir,
        task,
        build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"),
    )


def output_file_is_readable(path: Path) -> bool:
    return preflight.output_file_is_readable(path)


def missing_expected_outputs(output_dir: Path, task: dict[str, Any]) -> list[Path]:
    return preflight.missing_expected_outputs(
        output_dir,
        task,
        build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"),
    )


def station_outputs_missing_required_columns(output_dir: Path, task: dict[str, Any]) -> list[Path]:
    return preflight.station_outputs_missing_required_columns(
        output_dir,
        task,
        build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"),
    )


def has_complete_outputs(output_dir: Path, task: dict[str, Any]) -> bool:
    return preflight.has_complete_outputs(
        output_dir,
        task,
        build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"),
    )


def validate_comparison_only_inputs(output_dir: Path, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return preflight.validate_comparison_only_inputs(
        output_dir,
        tasks,
        build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"),
    )


def filter_evaluation_items_with_outputs(output_dir: Path, evaluation_items: list[str]) -> list[str]:
    return preflight.filter_evaluation_items_with_outputs(output_dir, evaluation_items)


def task_sources_from_bindings(bindings: Any, variables: list[str]) -> list[Any]:
    return preflight.task_sources_from_bindings(bindings, variables)


def as_list(value: Any) -> list[Any]:
    return preflight.as_list(value)


def post_phase_tasks_from_namelists(
    namelists: Any,
    evaluation_items: list[str],
    metric_vars: list[str],
    score_vars: list[str],
) -> list[dict[str, Any]]:
    return preflight.post_phase_tasks_from_namelists(namelists, evaluation_items, metric_vars, score_vars)


def post_phase_preflight_errors(
    bindings: Any,
    output_dir: Path,
    evaluation_items: list[str],
    metric_vars: list[str],
    score_vars: list[str],
    namelists: Any | None = None,
) -> list[dict[str, Any]]:
    return preflight.post_phase_preflight_errors(
        bindings,
        output_dir,
        evaluation_items,
        metric_vars,
        score_vars,
        namelists,
        build_runtime_info_fn=_local_attr("_build_bridge_runtime_info"),
    )
