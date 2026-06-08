"""Single-task execution helpers for the local runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

BridgeInfoBuilder = Callable[[dict[str, Any]], dict[str, Any]]
BindingsOnlyDrawing = Callable[[Any], bool]
OutputRequirement = Callable[[dict[str, Any], str], list[str]]
MissingOutputs = Callable[[Path, dict[str, Any]], list[Path]]


def evaluate_single(
    task: dict[str, Any],
    *,
    build_bridge_runtime_info_fn: BridgeInfoBuilder,
    bindings_only_drawing_fn: BindingsOnlyDrawing,
    task_output_requirement_fn: OutputRequirement,
    missing_expected_outputs_fn: MissingOutputs,
) -> dict[str, Any]:
    """Evaluate a single variable+sim+ref pair.

    Args:
        task: Dict with keys: var_name, sim_source, ref_source, bindings,
              cache_key, config_hash, use_cache, cache_dir.

    Returns:
        Result dict with keys: variable, sim, ref, status, cache_key,
        config_hash, skipped.
    """
    var_name = task["var_name"]
    sim_source = task["sim_source"]
    ref_source = task["ref_source"]
    cache_key = task["cache_key"]
    config_hash = task["config_hash"]
    use_cache = task["use_cache"]
    update_cache = bool(task.get("update_cache", use_cache))
    cache_dir = task.get("cache_dir")
    bindings = task["bindings"]
    only_drawing = bindings_only_drawing_fn(bindings)

    # Each worker process needs its own cache instance (for parallel safety)
    cache = None
    if (use_cache or update_cache) and cache_dir is not None:
        from openbench.runner.cache import EvaluationCache

        cache = EvaluationCache(Path(cache_dir))
        if use_cache and cache.is_cached(cache_key, config_hash):
            # Verify that output files actually exist before trusting cache.
            # Pattern MUST include ref_source — multi-ref configs have several
            # tasks per (var, sim) and an earlier ref's outputs would otherwise
            # let a later ref's cache check falsely pass with skipped=True even
            # though that ref had never been evaluated. Also require the full
            # requested metric/score set; a single leftover output is stale.
            output_dir = Path(cache_dir)
            missing = missing_expected_outputs_fn(output_dir, task)
            if not missing:
                logger.info("Cached, skipping %s: sim=%s ref=%s", var_name, sim_source, ref_source)
                return {
                    "variable": var_name,
                    "sim": sim_source,
                    "ref": ref_source,
                    "status": "success",
                    "cache_key": cache_key,
                    "config_hash": config_hash,
                    "skipped": True,
                }
            logger.warning(
                "Cache stale (output missing or unreadable), re-evaluating %s: sim=%s ref=%s",
                var_name,
                sim_source,
                ref_source,
            )
            # invalidate() now takes an fcntl.flock; on NFS / locked
            # filesystems that can OSError or EPERM. Don't let cache
            # bookkeeping errors crash the worker — the evaluation
            # itself will simply re-run, which is the safe default.
            try:
                cache.invalidate(cache_key)
            except Exception as inv_err:
                logger.warning(
                    "Cache invalidate failed for %s (sim=%s ref=%s): %s — proceeding with re-evaluation",
                    var_name,
                    sim_source,
                    ref_source,
                    inv_err,
                )

    try:
        info = build_bridge_runtime_info_fn(task)
        evaluation_fig_nml = bindings.build_evaluation_fig_nml().to_fig_nml()

        # Step 1: Preprocess data (skip if already done by _preprocess_variable)
        if not only_drawing and not task.get("ref_preprocessed"):
            from openbench.data.processing import DatasetProcessing

            dataset_processor = DatasetProcessing(info)
            dataset_processor.prepare_source("ref")
            dataset_processor.prepare_source("sim")

        # Step 2: Run evaluation
        ref_dtype = info.get("ref_data_type", "grid")
        sim_dtype = info.get("sim_data_type", "grid")

        if ref_dtype == "stn" or sim_dtype == "stn":
            if only_drawing:
                from openbench.visualization.Mod_Only_Drawing import Evaluation_stn_only_drawing as Evaluation_stn
            else:
                from openbench.core.evaluation import Evaluation_stn

            evaluator = Evaluation_stn(info, evaluation_fig_nml)
            evaluator.make_evaluation_P()
        else:
            if only_drawing:
                from openbench.visualization.Mod_Only_Drawing import Evaluation_grid_only_drawing as Evaluation_grid
            else:
                from openbench.core.evaluation import Evaluation_grid

            evaluator = Evaluation_grid(info, evaluation_fig_nml)
            evaluator.make_Evaluation()

        output_dir = Path(cache_dir) if cache_dir is not None else Path(info.get("casedir", "."))
        has_output_requirements = bool(task_output_requirement_fn(task, "metrics")) or bool(
            task_output_requirement_fn(task, "scores")
        )
        if not only_drawing and has_output_requirements:
            output_task = {
                **task,
                "ref_data_type": info.get("ref_data_type", "grid"),
                "sim_data_type": info.get("sim_data_type", "grid"),
            }
            missing_outputs = missing_expected_outputs_fn(output_dir, output_task)
            if missing_outputs:
                missing_text = ", ".join(str(path) for path in missing_outputs)
                logger.warning(
                    "Evaluation completed for %s (sim=%s ref=%s) but requested outputs are missing or unreadable: %s",
                    var_name,
                    sim_source,
                    ref_source,
                    missing_text,
                )
                return {
                    "variable": var_name,
                    "sim": sim_source,
                    "ref": ref_source,
                    "status": "error",
                    "error": f"requested outputs are missing or unreadable: {missing_text}",
                    "cache_key": cache_key,
                    "config_hash": config_hash,
                    "skipped": False,
                }

        if cache is not None and update_cache:
            # The evaluation already succeeded — its output files are on
            # disk. A failure to update the cache index (e.g. fcntl.flock
            # rejected on NFS) MUST NOT downgrade success to error,
            # otherwise GUI / CLI reports a false negative and the user
            # re-runs an already-completed evaluation. Log and continue.
            try:
                cache.mark_done(cache_key, config_hash)
            except Exception as mark_err:
                logger.warning(
                    "mark_done failed for %s (sim=%s ref=%s): %s — evaluation succeeded, cache index not updated",
                    var_name,
                    sim_source,
                    ref_source,
                    mark_err,
                )

        logger.info("Completed %s: sim=%s ref=%s", var_name, sim_source, ref_source)
        return {
            "variable": var_name,
            "sim": sim_source,
            "ref": ref_source,
            "status": "success",
            "cache_key": cache_key,
            "config_hash": config_hash,
            "skipped": False,
        }

    except Exception as exc:
        logger.exception("Evaluation failed for %s (sim=%s, ref=%s)", var_name, sim_source, ref_source)
        return {
            "variable": var_name,
            "sim": sim_source,
            "ref": ref_source,
            "status": "error",
            "error": str(exc),
            "cache_key": cache_key,
            "config_hash": config_hash,
            "skipped": False,
        }
