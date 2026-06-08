"""Runner preflight and output-completeness helpers.

This module owns the rules that decide whether an evaluation task already has
usable metric/score outputs and whether post-processing/comparison-only phases
can safely run against existing files.  ``runner.local`` keeps compatibility
wrappers for older tests and imports, but the logic lives here to keep the runner
orchestrator focused on orchestration.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

RuntimeInfoBuilder = Callable[[dict[str, Any]], dict[str, Any]]


def make_phase_error(phase: str, message: str, **details: Any) -> dict[str, Any]:
    """Create a structured error entry for runner results."""
    error = {"phase": phase, "status": "error", "message": message}
    error.update(details)
    return error


def find_existing_outputs(output_dir: Path, task: dict[str, Any]) -> list[Path]:
    """Find existing evaluation outputs for comparison-only mode."""
    var_name = glob.escape(str(task["var_name"]))
    ref_source = glob.escape(str(task["ref_source"]))
    sim_source = glob.escape(str(task["sim_source"]))
    patterns = [
        f"{var_name}_ref_{ref_source}_sim_{sim_source}_*",
        f"{var_name}_stn_{ref_source}_{sim_source}_evaluations*",
    ]
    matches: list[Path] = []
    for subdir in ("metrics", "scores"):
        output_subdir = output_dir / subdir
        if not output_subdir.is_dir():
            continue
        for pattern in patterns:
            matches.extend(output_subdir.glob(pattern))
    return matches


def task_output_requirement(task: dict[str, Any], key: str) -> list[str]:
    """Return requested output names from the task's runner-owned contract."""
    requirements = task.get("output_requirements")
    if isinstance(requirements, dict) and requirements.get(key) is not None:
        return list(requirements.get(key) or [])
    legacy_key = "metric_vars" if key == "metrics" else "score_vars"
    return list(task.get(legacy_key) or [])


def task_output_data_types(
    task: dict[str, Any],
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> tuple[str, str]:
    """Return ref/sim data types for output naming, defaulting to grid."""
    var_name = str(task.get("var_name", ""))
    ref_source = str(task.get("ref_source", ""))
    sim_source = str(task.get("sim_source", ""))

    requirements = task.get("output_requirements")
    if isinstance(requirements, dict):
        ref_dtype = requirements.get("ref_data_type")
        sim_dtype = requirements.get("sim_data_type")
        if ref_dtype or sim_dtype:
            return str(ref_dtype or "grid"), str(sim_dtype or "grid")

    ref_dtype = task.get("ref_data_type")
    sim_dtype = task.get("sim_data_type")
    if ref_dtype or sim_dtype:
        return str(ref_dtype or "grid"), str(sim_dtype or "grid")

    bindings = task.get("bindings")
    namelists = getattr(bindings, "namelists", None)
    if namelists is not None:
        ref_section = getattr(namelists, "reference", {}).get(var_name, {})
        sim_section = getattr(namelists, "simulation", {}).get(var_name, {})
        ref_dtype = ref_section.get(f"{ref_source}_data_type")
        sim_dtype = sim_section.get(f"{sim_source}_data_type")
        if ref_dtype or sim_dtype:
            return str(ref_dtype or "grid"), str(sim_dtype or "grid")

    if bindings is not None and build_runtime_info_fn is not None:
        try:
            info = build_runtime_info_fn(task)
            return str(info.get("ref_data_type", "grid")), str(info.get("sim_data_type", "grid"))
        except Exception as exc:
            logger.debug(
                "Could not derive output data types for %s/%s/%s: %s",
                var_name,
                ref_source,
                sim_source,
                exc,
            )

    return "grid", "grid"


def expected_output_paths(
    output_dir: Path,
    task: dict[str, Any],
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> list[Path]:
    """Return the exact evaluation outputs needed to reuse this task."""
    metric_vars = task_output_requirement(task, "metrics")
    score_vars = task_output_requirement(task, "scores")
    if not metric_vars and not score_vars:
        return []

    var_name = str(task["var_name"])
    ref_source = str(task["ref_source"])
    sim_source = str(task["sim_source"])
    ref_dtype, sim_dtype = task_output_data_types(task, build_runtime_info_fn=build_runtime_info_fn)
    uses_station_outputs = ref_dtype == "stn" or sim_dtype == "stn"

    expected: list[Path] = []
    if uses_station_outputs:
        filename = f"{var_name}_stn_{ref_source}_{sim_source}_evaluations.csv"
        if metric_vars:
            expected.append(output_dir / "metrics" / filename)
        if score_vars:
            expected.append(output_dir / "scores" / filename)
        return expected

    stem = f"{var_name}_ref_{ref_source}_sim_{sim_source}"
    expected.extend(output_dir / "metrics" / f"{stem}_{metric}.nc" for metric in metric_vars)
    expected.extend(output_dir / "scores" / f"{stem}_{score}.nc" for score in score_vars)
    return expected


def output_file_is_readable(path: Path) -> bool:
    """Return True when an output file exists and is minimally parseable."""
    if not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            logger.warning("Evaluation output is empty: %s", path)
            return False
    except OSError as exc:
        logger.warning("Could not stat evaluation output %s: %s", path, exc)
        return False

    suffix = path.suffix.lower()
    try:
        if suffix in {".nc", ".nc4", ".cdf"}:
            import xarray as xr

            with xr.open_dataset(path):
                return True
        if suffix == ".csv":
            import pandas as pd

            pd.read_csv(path, nrows=1)
            return True
    except Exception as exc:
        logger.warning("Evaluation output is unreadable: %s (%s)", path, exc)
        return False

    return True


def station_outputs_missing_required_columns(
    output_dir: Path,
    task: dict[str, Any],
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> list[Path]:
    """Return station CSV outputs that lack requested metric/score columns."""
    ref_dtype, sim_dtype = task_output_data_types(task, build_runtime_info_fn=build_runtime_info_fn)
    if ref_dtype != "stn" and sim_dtype != "stn":
        return []

    var_name = str(task["var_name"])
    ref_source = str(task["ref_source"])
    sim_source = str(task["sim_source"])
    filename = f"{var_name}_stn_{ref_source}_{sim_source}_evaluations.csv"
    checks = (
        (output_dir / "metrics" / filename, task_output_requirement(task, "metrics")),
        (output_dir / "scores" / filename, task_output_requirement(task, "scores")),
    )

    missing: list[Path] = []
    for path, required_columns in checks:
        if not required_columns or not path.is_file():
            continue
        try:
            import pandas as pd

            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        absent = [column for column in required_columns if column not in columns]
        if absent:
            logger.warning("Station output %s is missing requested columns: %s", path, absent)
            missing.append(path)
    return missing


def missing_expected_outputs(
    output_dir: Path,
    task: dict[str, Any],
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> list[Path]:
    """Return requested outputs that are absent or unreadable."""
    expected = expected_output_paths(output_dir, task, build_runtime_info_fn=build_runtime_info_fn)
    if not expected:
        existing_outputs = find_existing_outputs(output_dir, task)
        return [] if any(output_file_is_readable(path) for path in existing_outputs) else [output_dir]

    missing = [path for path in expected if not output_file_is_readable(path)]
    missing.extend(
        station_outputs_missing_required_columns(output_dir, task, build_runtime_info_fn=build_runtime_info_fn)
    )
    # Preserve order while avoiding duplicates when an unreadable CSV also has
    # missing columns by construction.
    return list(dict.fromkeys(missing))


def has_complete_outputs(
    output_dir: Path,
    task: dict[str, Any],
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> bool:
    """Return True only when all requested task outputs exist."""
    return not missing_expected_outputs(output_dir, task, build_runtime_info_fn=build_runtime_info_fn)


def validate_comparison_only_inputs(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> list[dict[str, Any]]:
    """Ensure comparison-only mode has pre-existing evaluation outputs."""
    errors = []
    for task in tasks:
        missing = missing_expected_outputs(output_dir, task, build_runtime_info_fn=build_runtime_info_fn)
        if not missing:
            continue
        errors.append(
            make_phase_error(
                "preflight",
                "missing prerequisite outputs for comparison-only mode",
                variable=task["var_name"],
                sim=task["sim_source"],
                ref=task["ref_source"],
                missing_outputs=[str(path) for path in missing if path != output_dir],
            )
        )
    return errors


def filter_evaluation_items_with_outputs(output_dir: Path, evaluation_items: list[str]) -> list[str]:
    """Return evaluation items that have at least one readable metrics/scores output file."""
    scores_dir = output_dir / "scores"
    metrics_dir = output_dir / "metrics"
    filtered = []
    for item in evaluation_items:
        item_escaped = glob.escape(item)
        patterns = (f"{item_escaped}_ref_*", f"{item_escaped}_stn_*")
        has_scores = scores_dir.exists() and any(
            any(output_file_is_readable(path) for path in scores_dir.glob(pattern)) for pattern in patterns
        )
        has_metrics = metrics_dir.exists() and any(
            any(output_file_is_readable(path) for path in metrics_dir.glob(pattern)) for pattern in patterns
        )
        if has_scores or has_metrics:
            filtered.append(item)
        else:
            logger.info("Skipping post-processing for '%s': no evaluation outputs found", item)
    return filtered


def task_sources_from_bindings(bindings: Any, variables: list[str]) -> list[Any]:
    """Return task sources from bindings when available."""
    iter_task_sources = getattr(bindings, "iter_task_sources", None)
    if not callable(iter_task_sources):
        return []
    try:
        return list(iter_task_sources(variables))
    except Exception as exc:
        logger.debug("Could not derive task sources for post-phase completeness check: %s", exc)
        return []


def as_list(value: Any) -> list[Any]:
    """Return value as a list while treating strings as scalar values."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def post_phase_tasks_from_namelists(
    namelists: Any,
    evaluation_items: list[str],
    metric_vars: list[str],
    score_vars: list[str],
) -> list[dict[str, Any]]:
    """Build post-phase output checks from legacy namelist source lists."""
    if namelists is None:
        return []

    reference = getattr(namelists, "reference", {}) or {}
    simulation = getattr(namelists, "simulation", {}) or {}
    ref_general = reference.get("general", {})
    sim_general = simulation.get("general", {})

    tasks: list[dict[str, Any]] = []
    for var_name in evaluation_items:
        ref_sources = as_list(ref_general.get(f"{var_name}_ref_source"))
        sim_sources = as_list(sim_general.get(f"{var_name}_sim_source"))
        if not ref_sources or not sim_sources:
            continue

        ref_section = reference.get(var_name, {})
        sim_section = simulation.get(var_name, {})
        for ref_source in ref_sources:
            for sim_source in sim_sources:
                tasks.append(
                    {
                        "var_name": var_name,
                        "sim_source": sim_source,
                        "ref_source": ref_source,
                        "ref_data_type": ref_section.get(f"{ref_source}_data_type", "grid"),
                        "sim_data_type": sim_section.get(f"{sim_source}_data_type", "grid"),
                        "output_requirements": {
                            "metrics": list(metric_vars),
                            "scores": list(score_vars),
                        },
                    }
                )
    return tasks


def post_phase_preflight_errors(
    bindings: Any,
    output_dir: Path,
    evaluation_items: list[str],
    metric_vars: list[str],
    score_vars: list[str],
    namelists: Any | None = None,
    *,
    build_runtime_info_fn: RuntimeInfoBuilder | None = None,
) -> list[dict[str, Any]]:
    """Validate complete task outputs before post-processing helpers run."""
    sources = task_sources_from_bindings(bindings, evaluation_items)
    if sources:
        tasks = [
            {
                "var_name": source.var_name,
                "sim_source": source.sim_source,
                "ref_source": source.ref_source,
                "bindings": bindings,
                "output_requirements": {
                    "metrics": list(metric_vars),
                    "scores": list(score_vars),
                },
            }
            for source in sources
        ]
    else:
        tasks = post_phase_tasks_from_namelists(
            namelists or getattr(bindings, "namelists", None),
            evaluation_items,
            metric_vars,
            score_vars,
        )

    if not tasks:
        return []
    return validate_comparison_only_inputs(output_dir, tasks, build_runtime_info_fn=build_runtime_info_fn)
