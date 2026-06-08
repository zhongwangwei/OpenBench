"""Local evaluation runner.

Orchestrates the evaluation pipeline using the new config system
and the migrated core engine.
"""

from __future__ import annotations

import logging
import os  # noqa: F401 - compatibility re-export for runner tests/monkeypatching
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor  # noqa: F401 - compatibility re-export
from pathlib import Path
from typing import Any

from openbench.config.schema import OpenBenchConfig
from openbench.runner import cache_state as _runner_cache_state
from openbench.runner import config_preflight as _runner_config_preflight
from openbench.runner import context as _runner_context
from openbench.runner import dask_runtime as _runner_dask_runtime
from openbench.runner import evaluation_dispatch as _runner_evaluation_dispatch
from openbench.runner import hashing as _runner_hashing
from openbench.runner import masking as _runner_masking
from openbench.runner import orchestration as _runner_orchestration
from openbench.runner import pair_ref as _runner_pair_ref
from openbench.runner import postprocessing as _runner_postprocessing
from openbench.runner import preflight_facade as _runner_preflight_facade
from openbench.runner import preprocessing as _runner_preprocessing
from openbench.runner import task_execution as _runner_task_execution
from openbench.runner import task_planning as _runner_task_planning
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic

logger = logging.getLogger(__name__)

OPENBENCH_ALGORITHM_VERSION = _runner_hashing.OPENBENCH_ALGORITHM_VERSION


BRIDGE_RUNTIME_FIELDS = _runner_context.BRIDGE_RUNTIME_FIELDS
RuntimeContext = _runner_context.RuntimeContext


_FACADE_ATTRS = {
    # Hashing helpers retained as import-compatible lazy facade attributes.
    "_source_specific_section": (_runner_hashing, "source_specific_section"),
    "_legacy_source_value": (_runner_hashing, "legacy_source_value"),
    "_file_sample_sha256": (_runner_hashing, "file_sample_sha256"),
    "_package_version": (_runner_hashing, "package_version"),
    "_openbench_version": (_runner_hashing, "openbench_version"),
    "_regrid_backend_signature": (_runner_hashing, "regrid_backend_signature"),
    "_configured_regrid_backend": (_runner_hashing, "configured_regrid_backend"),
    "_input_file_signature": (_runner_hashing, "input_file_signature"),
    "_stable_hash_data": (_runner_hashing, "stable_hash_data"),
    # Pair-ref filesystem helpers.
    "_remove_partial_pair_ref": (_runner_pair_ref, "remove_partial_pair_ref"),
    "_try_clonefile": (_runner_pair_ref, "try_clonefile"),
    "_try_reflink": (_runner_pair_ref, "try_reflink"),
    "_try_hardlink": (_runner_pair_ref, "try_hardlink"),
    "_try_symlink": (_runner_pair_ref, "try_symlink"),
    # Dask/runtime helpers.
    "_env_flag": (_runner_dask_runtime, "env_flag"),
    "_env_positive_int": (_runner_dask_runtime, "env_positive_int"),
    "_dask_distributed_requested": (_runner_dask_runtime, "dask_distributed_requested"),
    "_task_uses_station_data": (_runner_dask_runtime, "task_uses_station_data"),
    "_tasks_use_station_data": (_runner_dask_runtime, "tasks_use_station_data"),
    "_config_uses_station_data": (_runner_dask_runtime, "config_uses_station_data"),
    "_dask_station_guard_blocks": (_runner_dask_runtime, "dask_station_guard_blocks"),
    "_project_num_cores": (_runner_dask_runtime, "project_num_cores"),
    "_project_dask_config": (_runner_dask_runtime, "project_dask_config"),
    "_project_dask_local_directory": (_runner_dask_runtime, "project_dask_local_directory"),
    "_project_io_config": (_runner_dask_runtime, "project_io_config"),
    "_io_env_defaults": (_runner_dask_runtime, "io_env_defaults"),
    "_temporary_env_defaults": (_runner_dask_runtime, "temporary_env_defaults"),
    "_dask_option": (_runner_dask_runtime, "dask_option"),
    "_start_optional_dask_client": (_runner_dask_runtime, "start_optional_dask_client"),
    "_close_optional_dask_client": (_runner_dask_runtime, "close_optional_dask_client"),
    # Preflight helpers that need runner.local bridge callbacks.
    "_make_phase_error": (_runner_preflight_facade, "make_phase_error"),
    "_find_existing_outputs": (_runner_preflight_facade, "find_existing_outputs"),
    "_task_output_requirement": (_runner_preflight_facade, "task_output_requirement"),
    "_task_output_data_types": (_runner_preflight_facade, "task_output_data_types"),
    "_expected_output_paths": (_runner_preflight_facade, "expected_output_paths"),
    "_output_file_is_readable": (_runner_preflight_facade, "output_file_is_readable"),
    "_missing_expected_outputs": (_runner_preflight_facade, "missing_expected_outputs"),
    "_station_outputs_missing_required_columns": (_runner_preflight_facade, "station_outputs_missing_required_columns"),
    "_has_complete_outputs": (_runner_preflight_facade, "has_complete_outputs"),
    "_validate_comparison_only_inputs": (_runner_preflight_facade, "validate_comparison_only_inputs"),
    "_filter_evaluation_items_with_outputs": (_runner_preflight_facade, "filter_evaluation_items_with_outputs"),
    "_task_sources_from_bindings": (_runner_preflight_facade, "task_sources_from_bindings"),
    "_as_list": (_runner_preflight_facade, "as_list"),
    "_post_phase_tasks_from_namelists": (_runner_preflight_facade, "post_phase_tasks_from_namelists"),
    "_post_phase_preflight_errors": (_runner_preflight_facade, "post_phase_preflight_errors"),
}


def __getattr__(name: str) -> Any:
    """Expose legacy runner-local helper names through a thin lazy facade."""
    target = _FACADE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    return getattr(module, attr)


def _local_attr(name: str) -> Any:
    """Return monkeypatch-aware local facade attributes."""
    return getattr(sys.modules[__name__], name)


def _coerce_bridge_runtime_info(bridge_info):
    """Compatibility wrapper for runner context helpers."""
    return _runner_context.coerce_bridge_runtime_info(bridge_info)


def _build_runtime_context(task: dict[str, Any]) -> RuntimeContext:
    return _runner_context.build_runtime_context(task)


def _build_bridge_runtime_info(task: dict[str, Any]) -> dict[str, Any]:
    return _runner_context.build_bridge_runtime_info(task)


def _preprocess_variable_tasks(
    var_name: str,
    vtasks: list[dict],
    *,
    unified_mask: bool,
    time_alignment: str,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for variable-level preprocessing."""
    return _runner_preprocessing.preprocess_variable(
        var_name,
        vtasks,
        unified_mask=unified_mask,
        time_alignment=time_alignment,
        build_bridge_runtime_info_fn=_build_bridge_runtime_info,
        make_phase_error_fn=_local_attr("_make_phase_error"),
        clone_or_link_ref_for_pair_fn=_clone_or_link_ref_for_pair,
        apply_unified_mask_fn=_apply_unified_mask,
    )


def _bindings_general(bindings: Any) -> dict[str, Any]:
    """Return runner general options from a bindings-like object."""
    runner_cfg = getattr(bindings, "runner_cfg", None)
    return getattr(runner_cfg, "general", {}) or {}


def _bindings_only_drawing(bindings: Any) -> bool:
    """Return True only when bindings explicitly request only_drawing."""
    return _bindings_general(bindings).get("only_drawing") is True


def _build_evaluation_tasks(
    *,
    cfg: OpenBenchConfig,
    bindings: Any,
    output_dir: Path,
    metric_vars: list[str],
    score_vars: list[str],
    comparison_vars: list[str],
    statistic_vars: list[str],
    use_cache: bool,
    only_drawing: bool,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for task planning."""
    return _runner_task_planning.build_evaluation_tasks(
        cfg=cfg,
        bindings=bindings,
        output_dir=output_dir,
        metric_vars=metric_vars,
        score_vars=score_vars,
        comparison_vars=comparison_vars,
        statistic_vars=statistic_vars,
        use_cache=use_cache,
        only_drawing=only_drawing,
        task_hash_payload_fn=_task_hash_payload,
    )


def _group_tasks_by_variable(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Compatibility wrapper for variable task grouping."""
    return _runner_task_planning.group_tasks_by_variable(tasks)


def _collect_cached_results(
    var_tasks: dict[str, list[dict[str, Any]]],
    *,
    comparison_only: bool,
    only_drawing: bool,
    unified_mask: bool,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for cache skip marking."""
    return _runner_task_planning.collect_cached_results(
        var_tasks,
        comparison_only=comparison_only,
        only_drawing=only_drawing,
        unified_mask=unified_mask,
        cached_task_result_fn=_cached_task_result,
    )


def _evaluate_single(task: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for single-task evaluation."""
    return _runner_task_execution.evaluate_single(
        task,
        build_bridge_runtime_info_fn=_build_bridge_runtime_info,
        bindings_only_drawing_fn=_local_attr("_bindings_only_drawing"),
        task_output_requirement_fn=_local_attr("_task_output_requirement"),
        missing_expected_outputs_fn=_local_attr("_missing_expected_outputs"),
    )


def _evaluation_task_worker_count(num_cores: Any, task_count: int) -> int:
    """Compatibility wrapper for task-level worker sizing."""
    return _runner_evaluation_dispatch.evaluation_task_worker_count(num_cores, task_count)


def _task_level_parallel_safe(ready_tasks: list[dict[str, Any]]) -> bool:
    """Compatibility wrapper for task-level parallel safety checks."""
    return _runner_evaluation_dispatch.task_level_parallel_safe(ready_tasks)


def _evaluate_task_group(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility wrapper for serial group evaluation."""
    return _runner_evaluation_dispatch.evaluate_task_group(tasks)


def _unified_mask_parallel_groups(ready_tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Compatibility wrapper for unified-mask task grouping."""
    return _runner_evaluation_dispatch.unified_mask_parallel_groups(ready_tasks)


def _evaluate_ready_tasks(
    ready_tasks: list[dict[str, Any]],
    *,
    num_cores: Any,
    unified_mask: bool,
    only_drawing: bool,
    dask_distributed: bool = False,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for task-level evaluation dispatch."""
    return _runner_evaluation_dispatch.evaluate_ready_tasks(
        ready_tasks,
        num_cores=num_cores,
        unified_mask=unified_mask,
        only_drawing=only_drawing,
        dask_distributed=dask_distributed,
    )


def _apply_unified_mask(
    info: dict,
    var_name: str,
    ref_source: str,
    sim_source: str,
    ref_override: str | None = None,
) -> None:
    """Compatibility wrapper for runner unified-mask preprocessing."""
    return _runner_masking.apply_unified_mask(
        info,
        var_name,
        ref_source,
        sim_source,
        ref_override,
        write_netcdf_atomic_fn=_write_netcdf_atomic,
    )


def _shared_mask_peer_payload(
    *,
    cfg: OpenBenchConfig,
    bindings: Any,
    var_name: str,
    ref_source: str,
) -> dict[str, Any] | None:
    return _runner_hashing.shared_mask_peer_payload(
        cfg=cfg,
        bindings=bindings,
        var_name=var_name,
        ref_source=ref_source,
    )


def _task_hash_payload(
    *,
    cfg: OpenBenchConfig,
    bindings: Any,
    var_name: str,
    sim_source: str,
    ref_source: str,
    metric_vars: list[str],
    score_vars: list[str],
    comparison_vars: list[str],
    statistic_vars: list[str],
) -> dict[str, Any]:
    return _runner_hashing.task_hash_payload(
        cfg=cfg,
        bindings=bindings,
        var_name=var_name,
        sim_source=sim_source,
        ref_source=ref_source,
        metric_vars=metric_vars,
        score_vars=score_vars,
        comparison_vars=comparison_vars,
        statistic_vars=statistic_vars,
        openbench_version_fn=_local_attr("_openbench_version"),
        regrid_backend_signature_fn=_local_attr("_regrid_backend_signature"),
    )


def _cached_task_result(task: dict[str, Any]) -> dict[str, Any] | None:
    """Compatibility wrapper for cache-hit task results."""
    return _runner_cache_state.cached_task_result(task)


def existing_output_preflight_errors(cfg: OpenBenchConfig) -> list[dict[str, Any]]:
    """Compatibility wrapper for existing-output preflight checks."""
    return _runner_config_preflight.existing_output_preflight_errors(cfg)


def comparison_only_preflight_errors(cfg: OpenBenchConfig) -> list[dict[str, Any]]:
    """Compatibility wrapper for comparison-only preflight checks."""
    return _runner_config_preflight.comparison_only_preflight_errors(cfg)


def _cleanup_pair_ref_overrides(tasks: list[dict[str, Any]]) -> None:
    """Compatibility wrapper for per-pair reference cleanup."""
    return _runner_pair_ref.cleanup_pair_ref_overrides(tasks)


def _clone_or_link_ref_for_pair(src: str, dst: str) -> str:
    return _runner_pair_ref.clone_or_link_ref_for_pair(
        src,
        dst,
        creators=(
            ("clonefile", _local_attr("_try_clonefile")),
            ("reflink", _local_attr("_try_reflink")),
            ("hardlink", _local_attr("_try_hardlink")),
            ("symlink", _local_attr("_try_symlink")),
        ),
        copy2_fn=shutil.copy2,
    )


def run_evaluation(cfg: OpenBenchConfig, force: bool = False, comparison_only: bool = False) -> dict[str, Any]:
    """Run evaluation with optional runner-level dask.distributed scheduling."""
    project = getattr(cfg, "project", None)
    io_env_defaults = _local_attr("_io_env_defaults")(_local_attr("_project_io_config")(cfg))
    dask_config = _local_attr("_project_dask_config")(cfg)
    station_heavy = _local_attr("_config_uses_station_data")(cfg)
    with _local_attr("_temporary_env_defaults")(io_env_defaults):
        dask_handle = _local_attr("_start_optional_dask_client")(
            _local_attr("_project_num_cores")(cfg),
            only_drawing=bool(getattr(project, "only_drawing", False)),
            comparison_only=comparison_only,
            local_directory=_local_attr("_project_dask_local_directory")(cfg),
            dask_config=dask_config,
            station_heavy=station_heavy,
        )
        try:
            return _run_evaluation_impl(
                cfg,
                force=force,
                comparison_only=comparison_only,
                dask_distributed_active=dask_handle is not None,
            )
        finally:
            _local_attr("_close_optional_dask_client")(dask_handle)


def _run_evaluation_impl(
    cfg: OpenBenchConfig,
    force: bool = False,
    comparison_only: bool = False,
    dask_distributed_active: bool | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the split runner orchestration."""
    return _runner_orchestration.run_evaluation_impl(
        cfg,
        force=force,
        comparison_only=comparison_only,
        dask_distributed_active=dask_distributed_active,
    )


# ─── Post-evaluation phases ───


def _run_comparison(bindings, comparison_vars, output_dir):
    """Run comparison visualizations (Taylor diagrams, heat maps, etc.)."""
    return _runner_postprocessing.run_comparison(
        bindings,
        comparison_vars,
        output_dir,
        make_phase_error_fn=_local_attr("_make_phase_error"),
        bindings_only_drawing_fn=_local_attr("_bindings_only_drawing"),
        post_phase_preflight_errors_fn=_local_attr("_post_phase_preflight_errors"),
        filter_evaluation_items_with_outputs_fn=_local_attr("_filter_evaluation_items_with_outputs"),
    )


def _run_groupby(cfg, bindings, output_dir):
    """Run land cover and climate zone groupby analysis."""
    return _runner_postprocessing.run_groupby(
        cfg,
        bindings,
        output_dir,
        make_phase_error_fn=_local_attr("_make_phase_error"),
        bindings_only_drawing_fn=_local_attr("_bindings_only_drawing"),
        post_phase_preflight_errors_fn=_local_attr("_post_phase_preflight_errors"),
        filter_evaluation_items_with_outputs_fn=_local_attr("_filter_evaluation_items_with_outputs"),
    )


def _run_statistics(bindings, statistic_vars, output_dir: Path | None = None):
    """Run statistical analysis."""
    return _runner_postprocessing.run_statistics(
        bindings,
        statistic_vars,
        output_dir,
        make_phase_error_fn=_local_attr("_make_phase_error"),
        post_phase_preflight_errors_fn=_local_attr("_post_phase_preflight_errors"),
        filter_evaluation_items_with_outputs_fn=_local_attr("_filter_evaluation_items_with_outputs"),
    )


def _run_report(bindings, output_dir):
    """Generate evaluation report."""
    return _runner_postprocessing.run_report(
        bindings,
        output_dir,
        make_phase_error_fn=_local_attr("_make_phase_error"),
    )
