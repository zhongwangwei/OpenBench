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

    logger.info("Evaluation phase: %d succeeded, %d failed", len(evaluated), len(errors))

    # ─── Phase 2: Comparison ───
    if cfg.comparison.enabled and comparison_vars and evaluated:
        logger.info("Starting comparison phase: %s", comparison_vars)
        _run_comparison(main_nl, sim_nml, ref_nml, legacy, comparison_vars, fig_nml, output_dir)

    # ─── Phase 2b: Groupby (IGBP / PFT / Climate Zone) ───
    if evaluated:
        _run_groupby(cfg, main_nl, sim_nml, ref_nml, legacy, fig_nml, output_dir)

    # ─── Phase 3: Statistics ───
    if cfg.statistics.enabled and statistic_vars:
        logger.info("Starting statistics phase: %s", statistic_vars)
        _run_statistics(main_nl, statistic_vars, fig_nml)

    # ─── Phase 4: Report ───
    if cfg.options.generate_report:
        _run_report(main_nl, legacy, ref_nml, sim_nml, output_dir)

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

    logger.info("All phases complete: %d evaluated, %d errors", len(evaluated), len(errors))
    return results


# ─── Post-evaluation phases ───


def _run_comparison(main_nl, sim_nml, ref_nml, legacy, comparison_vars, fig_nml, output_dir):
    """Run comparison visualizations (Taylor diagrams, heat maps, etc.)."""
    import gc

    try:
        from openbench.core.comparison import ComparisonProcessing

        basedir = str(output_dir)
        evaluation_items = list(legacy["evaluation_items"].keys())
        score_vars = list(legacy["scores"].keys())
        metric_vars = list(legacy["metrics"].keys())

        ch = ComparisonProcessing(main_nl, score_vars, metric_vars)

        comparison_fig = fig_nml.get("Comparison", {})

        for cvar in comparison_vars:
            logger.info("Running %s comparison...", cvar)
            if cvar in ("Mean", "Median", "Max", "Min", "Sum"):
                method_name = "scenarios_Basic_comparison"
            else:
                method_name = f"scenarios_{cvar}_comparison"

            fig_opts = comparison_fig.get(cvar, {})

            if hasattr(ch, method_name):
                try:
                    getattr(ch, method_name)(
                        basedir, sim_nml, ref_nml, evaluation_items,
                        score_vars, metric_vars, fig_opts,
                    )
                    logger.info("Completed %s comparison", cvar)
                except Exception:
                    logger.exception("Failed %s comparison", cvar)
            else:
                logger.warning("Comparison method %s not found, skipping", method_name)

            gc.collect()

    except ImportError:
        logger.warning("ComparisonProcessing not available, skipping comparison phase")
    except Exception:
        logger.exception("Comparison phase failed")


def _run_groupby(cfg, main_nl, sim_nml, ref_nml, legacy, fig_nml, output_dir):
    """Run land cover and climate zone groupby analysis."""
    import gc

    basedir = str(output_dir)
    evaluation_items = list(legacy["evaluation_items"].keys())
    score_vars = list(legacy["scores"].keys())
    metric_vars = list(legacy["metrics"].keys())
    validation_fig = fig_nml.get("IGBP_groupby", fig_nml.get("Validation", {}))

    if cfg.options.IGBP_groupby:
        try:
            from openbench.core.landcover_groupby import LC_groupby

            logger.info("Running IGBP land cover groupby...")
            lc = LC_groupby(main_nl, score_vars, metric_vars)
            lc.scenarios_IGBP_groupby_comparison(
                basedir, sim_nml, ref_nml, evaluation_items,
                score_vars, metric_vars, validation_fig,
            )
            gc.collect()
            logger.info("IGBP groupby complete")
        except Exception:
            logger.exception("IGBP groupby failed")

    if cfg.options.PFT_groupby:
        try:
            from openbench.core.landcover_groupby import LC_groupby

            logger.info("Running PFT groupby...")
            lc = LC_groupby(main_nl, score_vars, metric_vars)
            lc.scenarios_PFT_groupby_comparison(
                basedir, sim_nml, ref_nml, evaluation_items,
                score_vars, metric_vars, validation_fig,
            )
            gc.collect()
            logger.info("PFT groupby complete")
        except Exception:
            logger.exception("PFT groupby failed")

    if cfg.options.climate_zone_groupby:
        try:
            from openbench.core.climatezone_groupby import CZ_groupby

            logger.info("Running climate zone groupby...")
            cz = CZ_groupby(main_nl, score_vars, metric_vars)
            cz_fig = fig_nml.get("Climate_zone_groupby", validation_fig)
            cz.scenarios_CZ_groupby_comparison(
                basedir, sim_nml, ref_nml, evaluation_items,
                score_vars, metric_vars, cz_fig,
            )
            gc.collect()
            logger.info("Climate zone groupby complete")
        except Exception:
            logger.exception("Climate zone groupby failed")


def _run_statistics(main_nl, statistic_vars, fig_nml):
    """Run statistical analysis."""
    import gc
    import os

    try:
        from openbench.core.statistics.Mod_Statistics import StatisticsProcessing

        basedir = os.path.join(main_nl["general"]["basedir"], main_nl["general"]["basename"])
        stats_dir = os.path.join(basedir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        stats_handler = StatisticsProcessing(
            main_nl, {},  # stats_nml placeholder
            stats_dir,
            num_cores=main_nl["general"].get("num_cores", 1),
        )

        statistic_fig = fig_nml.get("Statistic", {})

        for statistic in statistic_vars:
            logger.info("Running %s analysis...", statistic)
            if statistic in ("Mean", "Median", "Max", "Min", "Sum"):
                method_name = "scenarios_Basic_analysis"
            else:
                method_name = f"scenarios_{statistic}_analysis"

            if hasattr(stats_handler, method_name):
                try:
                    stat_fig = statistic_fig.get(statistic, {})
                    getattr(stats_handler, method_name)(statistic, {}, stat_fig)
                    logger.info("Completed %s analysis", statistic)
                except Exception:
                    logger.exception("Failed %s analysis", statistic)
            else:
                logger.warning("Statistics method %s not found, skipping", method_name)

            gc.collect()

    except ImportError:
        logger.warning("StatisticsProcessing not available, skipping statistics phase")
    except Exception:
        logger.exception("Statistics phase failed")


def _run_report(main_nl, legacy, ref_nml, sim_nml, output_dir):
    """Generate evaluation report."""
    try:
        from openbench.util.report import ReportGenerator

        report_config = {
            "evaluation_items": list(legacy["evaluation_items"].keys()),
            "metrics": legacy.get("metrics", {}),
            "scores": legacy.get("scores", {}),
            "comparisons": legacy.get("comparisons", {}),
            "statistics": legacy.get("statistics", {}),
            "general": legacy.get("general", {}),
            "ref_nml": dict(ref_nml) if ref_nml else {},
            "sim_nml": dict(sim_nml) if sim_nml else {},
        }

        report_gen = ReportGenerator(report_config, str(output_dir))
        report_paths = report_gen.generate_report()

        if report_paths:
            for fmt, path in report_paths.items():
                logger.info("Report generated: %s → %s", fmt, path)
        else:
            logger.info("Report generation completed")

    except ImportError:
        logger.warning("ReportGenerator not available, skipping report generation")
    except Exception:
        logger.exception("Report generation failed")
