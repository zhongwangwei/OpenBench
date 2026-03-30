"""Local evaluation runner.

Orchestrates the evaluation pipeline using the new config system
and the migrated core engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


def _evaluate_single(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single variable+sim+ref pair.

    Args:
        task: Dict with keys: var_name, sim_source, ref_source, main_nl,
              sim_nml, ref_nml, metric_vars, score_vars, comparison_vars,
              statistic_vars, fig_nml, cache_key, config_hash, use_cache,
              cache_dir.

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
    cache_dir = task.get("cache_dir")

    # Each worker process needs its own cache instance (for parallel safety)
    cache = None
    if use_cache and cache_dir is not None:
        from openbench.runner.cache import EvaluationCache

        cache = EvaluationCache(Path(cache_dir))
        if cache.is_cached(cache_key, config_hash):
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

    try:
        from openbench.config.legacy_processors import GeneralInfoReader

        info_reader = GeneralInfoReader(
            main_nl=task["main_nl"],
            sim_nml=task["sim_nml"],
            ref_nml=task["ref_nml"],
            metric_vars=task["metric_vars"],
            score_vars=task["score_vars"],
            comparison_vars=task["comparison_vars"],
            statistic_vars=task["statistic_vars"],
            item=var_name,
            sim_source=sim_source,
            ref_source=ref_source,
        )

        info = info_reader.to_dict()
        info["ref_source"] = ref_source
        info["sim_source"] = sim_source

        # Step 1: Preprocess data (read raw NetCDF, align, save to casedir/data/)
        from openbench.data.processing import DatasetProcessing

        dataset_processor = DatasetProcessing(info)
        dataset_processor.process("ref")
        dataset_processor.process("sim")

        # Step 2: Run evaluation
        ref_dtype = info.get("ref_data_type", "grid")
        sim_dtype = info.get("sim_data_type", "grid")

        if ref_dtype == "stn" or sim_dtype == "stn":
            from openbench.core.evaluation import Evaluation_stn

            evaluator = Evaluation_stn(info, task["fig_nml"])
            try:
                evaluator.make_evaluation_P()
            except (KeyError, TypeError) as viz_err:
                logger.warning("Metrics computed but visualization skipped: %s", viz_err)
        else:
            from openbench.core.evaluation import Evaluation_grid

            evaluator = Evaluation_grid(info, task["fig_nml"])
            try:
                evaluator.make_Evaluation()
            except (KeyError, TypeError) as viz_err:
                logger.warning("Metrics computed but visualization skipped: %s", viz_err)

        if cache is not None:
            cache.mark_done(cache_key, config_hash)

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


def run_evaluation(cfg: OpenBenchConfig, force: bool = False) -> dict[str, Any]:
    """Run evaluation from a validated config.

    This is the main entry point that replaces the old openbench.py script.
    It uses the config adapter to bridge between new and legacy formats,
    builds legacy namelists from the registry, and drives the evaluation
    engine for each variable / reference / simulation combination.

    Supports variable-level parallelism via joblib when ``cfg.options.num_cores``
    is greater than 1, and an incremental cache that skips re-computation when
    the config for a variable hasn't changed (pass ``force=True`` to bypass).

    Args:
        cfg: Validated OpenBenchConfig instance.
        force: If True, bypass the incremental cache and re-run all evaluations.

    Returns:
        Summary dict with results.
    """
    from openbench.config.adapter import build_legacy_namelists, to_legacy_config
    from openbench.runner.cache import EvaluationCache, make_cache_key

    legacy = to_legacy_config(cfg)
    general = legacy["general"]

    # Setup output directories
    basedir = Path(general["basedir"])
    basename = general["basename"]
    output_dir = basedir / basename
    for sub in ["data", "metrics", "scores", "figures", "comparisons", "reports", "scratch", "tmp"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation: %s", basename)
    logger.info("Output directory: %s", output_dir)
    logger.info("Variables: %s", list(legacy["evaluation_items"].keys()))
    logger.info("Simulations: %s", list(cfg.simulation.keys()))

    # Build the three legacy namelists from new config + registry
    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    # Derive list keys from the legacy config
    metric_vars = list(legacy["metrics"].keys())
    score_vars = list(legacy["scores"].keys())
    comparison_vars = list(legacy["comparisons"].keys())
    statistic_vars = list(legacy["statistics"].keys())

    # Build figure configuration from bundled figure config files
    from openbench.config.adapter import build_fig_nml

    fig_nml = build_fig_nml()

    # Determine parallelism level
    num_cores: int = 1
    if hasattr(cfg, "options") and cfg.options is not None:
        num_cores = getattr(cfg.options, "num_cores", 1) or 1

    # Also honour force flag from cfg.options if not passed directly
    if not force and hasattr(cfg, "options") and cfg.options is not None:
        force = bool(getattr(cfg.options, "force", False))

    use_cache = not force

    # Build task list
    tasks: list[dict[str, Any]] = []
    for var_name in cfg.evaluation.variables:
        ref_source = ref_nml["general"].get(f"{var_name}_ref_source")
        sim_sources = sim_nml["general"].get(f"{var_name}_sim_source", [])

        if not ref_source:
            logger.warning("Skipping %s: no reference source", var_name)
            continue

        for sim_source in sim_sources:
            cache_key = make_cache_key(var_name, sim_source, ref_source)
            config_hash = EvaluationCache.hash_config(
                {
                    "variable": var_name,
                    "sim_source": sim_source,
                    "ref_source": ref_source,
                    "metrics": metric_vars,
                    "scores": score_vars,
                    "comparisons": comparison_vars,
                    "statistics": statistic_vars,
                }
            )
            tasks.append(
                {
                    "var_name": var_name,
                    "sim_source": sim_source,
                    "ref_source": ref_source,
                    "main_nl": main_nl,
                    "sim_nml": sim_nml,
                    "ref_nml": ref_nml,
                    "metric_vars": metric_vars,
                    "score_vars": score_vars,
                    "comparison_vars": comparison_vars,
                    "statistic_vars": statistic_vars,
                    "fig_nml": fig_nml,
                    "cache_key": cache_key,
                    "config_hash": config_hash,
                    "use_cache": use_cache,
                    "cache_dir": str(output_dir),
                }
            )
            logger.info("Queued %s: sim=%s ref=%s", var_name, sim_source, ref_source)

    # Execute tasks: parallel when num_cores > 1, else sequential
    raw_results: list[dict[str, Any]]
    if num_cores > 1 and len(tasks) > 1:
        try:
            from joblib import Parallel, delayed

            logger.info("Running %d tasks in parallel (n_jobs=%d)", len(tasks), num_cores)
            raw_results = Parallel(n_jobs=num_cores)(delayed(_evaluate_single)(t) for t in tasks)
        except Exception:
            logger.warning("Parallel execution failed, falling back to sequential", exc_info=True)
            raw_results = [_evaluate_single(t) for t in tasks]
    else:
        raw_results = [_evaluate_single(t) for t in tasks]

    evaluated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for res in raw_results:
        if res["status"] == "success":
            evaluated.append({
                "variable": res["variable"], "sim": res["sim"], "ref": res["ref"],
                "status": "success", "skipped": res.get("skipped", False),
            })
        else:
            errors.append({
                "variable": res["variable"], "sim": res["sim"], "ref": res["ref"],
                "status": "error",
            })

    results: dict[str, Any] = {
        "status": "success" if not errors else "partial",
        "basename": basename,
        "output_dir": str(output_dir),
        "variables": list(legacy["evaluation_items"].keys()),
        "simulations": list(cfg.simulation.keys()),
        "metrics": metric_vars,
        "evaluated": evaluated,
        "errors": errors,
    }

    logger.info("Evaluation complete: %d succeeded, %d failed", len(evaluated), len(errors))
    return results
