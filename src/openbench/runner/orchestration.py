"""High-level local runner orchestration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


def run_evaluation_impl(
    cfg: OpenBenchConfig,
    force: bool = False,
    comparison_only: bool = False,
    dask_distributed_active: bool | None = None,
) -> dict[str, Any]:
    """Run evaluation from a validated config.

    This is the main entry point that replaces the old openbench.py script.
    It uses the config adapter to bridge between new and legacy formats,
    builds legacy namelists from the registry, and drives the evaluation
    engine for each variable / reference / simulation combination.

    Dataset preprocessing and statistics may use joblib internally when
    ``cfg.project.num_cores`` is greater than 1. The runner also maintains an
    incremental cache that skips re-computation when a task's runtime-sensitive
    config has not changed (pass ``force=True`` to bypass).

    Args:
        cfg: Validated OpenBenchConfig instance.
        force: If True, bypass the incremental cache and re-run all evaluations.

    Returns:
        Summary dict with results.
    """
    # Disable HDF5 file locking for parallel reads.
    # HDF5 ≥ 1.14 locks files even for read-only access, which causes
    # "Resource temporarily unavailable" when multiple workers open the
    # same reference NC file.  All writes go to distinct output files,
    # so disabling the lock is safe.

    # Resolve through openbench.runner.local at call time so existing tests and
    # downstream integrations that monkeypatch local helper wrappers still affect
    # orchestration after the god-module split.
    from openbench.runner import local as _local_runner

    _apply_unified_mask = _local_runner._apply_unified_mask
    _build_evaluation_tasks = _local_runner._build_evaluation_tasks
    _cleanup_pair_ref_overrides = _local_runner._cleanup_pair_ref_overrides
    _collect_cached_results = _local_runner._collect_cached_results
    _dask_distributed_requested = _local_runner._dask_distributed_requested
    _evaluate_ready_tasks = _local_runner._evaluate_ready_tasks
    _group_tasks_by_variable = _local_runner._group_tasks_by_variable
    _has_complete_outputs = _local_runner._has_complete_outputs
    _make_phase_error = _local_runner._make_phase_error
    _missing_expected_outputs = _local_runner._missing_expected_outputs
    _preprocess_variable_tasks = _local_runner._preprocess_variable_tasks
    _project_dask_config = _local_runner._project_dask_config
    _run_comparison = _local_runner._run_comparison
    _run_groupby = _local_runner._run_groupby
    _run_report = _local_runner._run_report
    _run_statistics = _local_runner._run_statistics
    _validate_comparison_only_inputs = _local_runner._validate_comparison_only_inputs

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    from openbench.config.adapter import build_runner_bindings

    bindings = build_runner_bindings(cfg)
    runner_cfg = bindings.runner_cfg
    general = runner_cfg.general

    # Setup output directories
    basedir = Path(runner_cfg.basedir)
    basename = runner_cfg.basename
    output_dir = basedir / basename
    for sub in ["data", "metrics", "scores", "figures", "comparisons", "reports", "scratch", "tmp"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation: %s", basename)
    logger.info("Output directory: %s", output_dir)
    logger.info("Variables: %s", list(runner_cfg.evaluation_items.keys()))
    logger.info("Simulations: %s", list(cfg.simulation.keys()))

    # Derive list keys from the legacy config
    metric_vars = list(runner_cfg.metrics)
    score_vars = list(runner_cfg.scores)
    comparison_vars = list(runner_cfg.comparisons)
    statistic_vars = list(runner_cfg.statistics)

    # Determine parallelism level (auto-detect when num_cores is None/0).
    # Currently unused at variable level — parallelism lives inside
    # DatasetProcessing (station processing, yearly combination).
    # Reserved for future variable-level parallel dispatch.
    num_cores: int = max(1, os.cpu_count() or 1)
    if hasattr(cfg, "project") and cfg.project is not None:
        num_cores = getattr(cfg.project, "num_cores", 0) or num_cores

    # Also honour force flag from cfg.project if not passed directly
    if not force and hasattr(cfg, "project") and cfg.project is not None:
        force = bool(getattr(cfg.project, "force", False))

    only_drawing = bool(general.get("only_drawing", False))
    use_cache = not force and not only_drawing

    # Build task list
    tasks = _build_evaluation_tasks(
        cfg=cfg,
        bindings=bindings,
        output_dir=output_dir,
        metric_vars=metric_vars,
        score_vars=score_vars,
        comparison_vars=comparison_vars,
        statistic_vars=statistic_vars,
        use_cache=use_cache,
        only_drawing=only_drawing,
    )

    if not tasks:
        errors = [
            _make_phase_error(
                "preflight",
                "no evaluation tasks were queued; check evaluation.variables and reference mappings",
            )
        ]
        return {
            "status": "error",
            "basename": basename,
            "output_dir": str(output_dir),
            "variables": list(runner_cfg.evaluation_items.keys()),
            "simulations": list(cfg.simulation.keys()),
            "metrics": metric_vars,
            "evaluated": [],
            "errors": errors,
        }

    if comparison_only and not cfg.comparison.enabled:
        return {
            "status": "error",
            "basename": basename,
            "output_dir": str(output_dir),
            "variables": list(runner_cfg.evaluation_items.keys()),
            "simulations": list(cfg.simulation.keys()),
            "metrics": metric_vars,
            "evaluated": [],
            "errors": [
                _make_phase_error(
                    "preflight",
                    "comparison-only mode requires comparison.enabled: true",
                )
            ],
        }

    # ─── Phase 1: Evaluation ───
    if comparison_only:
        logger.info("Comparison-only mode: skipping evaluation phase")
        # Validate up front: every requested task must have the complete
        # metric/score outputs. Running comparison from a partial task set can
        # generate figures that look complete while silently excluding a model
        # or variable.
        errors = _validate_comparison_only_inputs(output_dir, tasks)
        if errors:
            return {
                "status": "error",
                "basename": basename,
                "output_dir": str(output_dir),
                "variables": list(runner_cfg.evaluation_items.keys()),
                "simulations": list(cfg.simulation.keys()),
                "metrics": metric_vars,
                "evaluated": [],
                "errors": errors,
            }

        evaluated = []
        skipped = 0
        for t in tasks:
            if _has_complete_outputs(output_dir, t):
                evaluated.append({"variable": t["var_name"], "sim": t["sim_source"], "ref": t["ref_source"]})
            else:
                skipped += 1
                logger.info(
                    "Comparison-only: skipping %s/%s (no pre-existing outputs)",
                    t["var_name"],
                    t["sim_source"],
                )
        if skipped:
            logger.info("Comparison-only: %d task(s) skipped, %d available", skipped, len(evaluated))

    elif only_drawing:
        logger.info("only_drawing mode: validating existing evaluation outputs before rendering")
        # only_drawing reuses previously generated metrics/scores to render
        # figures. Validate the same exact task-level output contract as
        # comparison-only before invoking visualizers; otherwise a missing
        # output can either fail deep inside plotting code or be hidden by a
        # visualizer that does not touch every requested metric/score.
        errors = _validate_comparison_only_inputs(output_dir, tasks)
        if errors:
            return {
                "status": "error",
                "basename": basename,
                "output_dir": str(output_dir),
                "variables": list(runner_cfg.evaluation_items.keys()),
                "simulations": list(cfg.simulation.keys()),
                "metrics": metric_vars,
                "evaluated": [],
                "errors": errors,
            }

    # ─── Pre-process data: parallel across variables, serial within each variable ───
    #
    # Strategy: group tasks by variable, preprocess each variable's tasks in parallel.
    # Within each variable: ref once → for each sim: sim + unified_mask (serial, mask accumulates).
    # Different variables write to different files → safe to parallelize.
    unified_mask = general.get("unified_mask", True)

    # Group tasks by variable and mark safe cache skips.
    var_tasks = _group_tasks_by_variable(tasks)
    cached_results = _collect_cached_results(
        var_tasks,
        comparison_only=comparison_only,
        only_drawing=only_drawing,
        unified_mask=bool(unified_mask),
    )

    time_alignment = cfg.project.time_alignment  # "intersection", "per_pair", "strict"

    # Dispatch preprocessing + evaluation (skip preprocessing in comparison-only
    # and only_drawing modes).
    if not comparison_only:
        var_names = [
            var_name for var_name, vtasks in var_tasks.items() if not all(task.get("cache_skipped") for task in vtasks)
        ]
        # Preprocess + evaluate: serial across variables (like old openbench.py).
        # Parallelism lives *inside* station processing (Parallel n_jobs=num_cores)
        # and inside yearly file combination, not at the variable/task level.
        # This avoids nested-parallel deadlocks and I/O contention on shared
        # reference files that plagued the previous variable-level parallel dispatch.
        preprocess_errors: list[dict[str, Any]] = []
        if only_drawing:
            logger.info("only_drawing mode: skipping preprocessing and metric recomputation")
        else:
            for vn in var_names:
                preprocess_errors.extend(
                    _preprocess_variable_tasks(
                        vn,
                        var_tasks[vn],
                        unified_mask=bool(unified_mask),
                        time_alignment=time_alignment,
                    )
                )

        ready_tasks = [task for task in tasks if not task.get("preprocess_failed") and not task.get("cache_skipped")]

        raw_results: list[dict[str, Any]] = list(cached_results)
        task_level_num_cores = getattr(cfg.project, "num_cores", 1) if getattr(cfg, "project", None) else 1
        if dask_distributed_active is None:
            dask_active_for_tasks = _dask_distributed_requested(_project_dask_config(cfg)) and not comparison_only
        else:
            dask_active_for_tasks = bool(dask_distributed_active) and not comparison_only
        try:
            raw_results.extend(
                _evaluate_ready_tasks(
                    ready_tasks,
                    num_cores=task_level_num_cores or 1,
                    unified_mask=bool(unified_mask),
                    only_drawing=bool(only_drawing),
                    dask_distributed=dask_active_for_tasks,
                )
            )
        except Exception:
            _cleanup_pair_ref_overrides(tasks)
            raise

        evaluated = []
        errors = list(preprocess_errors)
        for res in raw_results:
            if res["status"] == "success":
                evaluated.append(
                    {
                        "variable": res["variable"],
                        "sim": res["sim"],
                        "ref": res["ref"],
                        "status": "success",
                        "skipped": res.get("skipped", False),
                    }
                )
            else:
                errors.append(
                    _make_phase_error(
                        "evaluation",
                        res.get("error", "evaluation failed"),
                        variable=res["variable"],
                        sim=res["sim"],
                        ref=res["ref"],
                    )
                )

    logger.info("Evaluation phase: %d succeeded, %d failed", len(evaluated), len(errors))
    post_phases_allowed = bool(evaluated) and (comparison_only or not errors)
    if evaluated and errors and not comparison_only:
        logger.warning(
            "Skipping post-evaluation phases because %d evaluation/preprocessing error(s) occurred",
            len(errors),
        )

    # ─── Phase 2: Comparison ───
    if cfg.comparison.enabled and comparison_vars and post_phases_allowed:
        logger.info("Starting comparison phase: %s", comparison_vars)
        try:
            errors.extend(_run_comparison(bindings, comparison_vars, output_dir) or [])
        except Exception as exc:
            logger.exception("Comparison phase failed")
            errors.append(_make_phase_error("comparison", f"comparison phase failed: {exc}"))

    # ─── Phase 2b: Groupby (IGBP / PFT / Climate Zone) ───
    # Skipped under --comparison-only: the CLI flag advertises "only run
    # comparisons", so groupby/statistics/report belong to the full
    # pipeline only.
    if cfg.comparison.enabled and post_phases_allowed and not comparison_only:
        try:
            errors.extend(_run_groupby(cfg, bindings, output_dir) or [])
        except Exception as exc:
            logger.exception("Groupby phase failed")
            errors.append(_make_phase_error("groupby", f"groupby phase failed: {exc}"))

    # ─── Phase 3: Statistics ───
    # Statistics module operates on gridded NC files (spatial remap + aggregation).
    # Skip for purely station-based evaluations where metrics are CSV-only.
    # Also skip under --comparison-only (see note above).
    grid_evidence = bindings.has_grid_evaluation(cfg.evaluation.variables)
    if (
        cfg.statistics.enabled
        and statistic_vars
        and grid_evidence.has_grid
        and post_phases_allowed
        and not comparison_only
        and not only_drawing
    ):
        logger.info("Starting statistics phase: %s", statistic_vars)
        try:
            errors.extend(_run_statistics(bindings, statistic_vars, output_dir) or [])
        except Exception as exc:
            logger.exception("Statistics phase failed")
            errors.append(
                _make_phase_error("statistics", f"statistics phase failed: {exc}", source="StatisticsProcessing")
            )
    elif cfg.statistics.enabled and statistic_vars and post_phases_allowed and only_drawing and not comparison_only:
        logger.info("Skipping statistics phase in only_drawing mode: statistics recomputation is disabled")
    elif (
        cfg.statistics.enabled
        and statistic_vars
        and not grid_evidence.has_grid
        and post_phases_allowed
        and not comparison_only
    ):
        logger.info("Skipping statistics phase: not applicable for station-only evaluations")

    # ─── Phase 4: Report ───
    # Skipped under --comparison-only — the report aggregates evaluation
    # outputs across all phases and would partially regenerate reports
    # users may have already curated.
    if cfg.project.generate_report and post_phases_allowed and not comparison_only:
        try:
            errors.extend(_run_report(bindings, output_dir) or [])
        except Exception as exc:
            logger.exception("Report generation failed")
            errors.append(_make_phase_error("report", f"report generation failed: {exc}"))

    if not errors and not evaluated:
        errors.append(
            _make_phase_error(
                "evaluation",
                "no evaluation tasks completed successfully; check preprocessing, cache, and output prerequisites",
            )
        )

    if errors and evaluated:
        status = "partial"
    elif errors:
        status = "error"
    else:
        status = "success"

    results: dict[str, Any] = {
        "status": status,
        "basename": basename,
        "output_dir": str(output_dir),
        "variables": list(runner_cfg.evaluation_items.keys()),
        "simulations": list(cfg.simulation.keys()),
        "metrics": metric_vars,
        "evaluated": evaluated,
        "errors": errors,
    }

    _cleanup_pair_ref_overrides(tasks)

    logger.info("All phases complete: %d evaluated, %d errors", len(evaluated), len(errors))
    return results
