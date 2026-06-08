"""Post-evaluation runner phases: comparison, groupby, statistics, reports."""

from __future__ import annotations

import gc
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

MakePhaseError = Callable[..., dict[str, Any]]
PostPhasePreflight = Callable[..., list[dict[str, Any]]]
ItemFilter = Callable[[Path, list[str]], list[str]]
BindingsOnlyDrawing = Callable[[Any], bool]


def run_comparison(
    bindings: Any,
    comparison_vars: list[str],
    output_dir: Path,
    *,
    make_phase_error_fn: MakePhaseError,
    bindings_only_drawing_fn: BindingsOnlyDrawing,
    post_phase_preflight_errors_fn: PostPhasePreflight,
    filter_evaluation_items_with_outputs_fn: ItemFilter,
) -> list[dict[str, Any]]:
    """Run comparison visualizations (Taylor diagrams, heat maps, etc.)."""
    phase_errors: list[dict[str, Any]] = []
    basic_methods = ("Mean", "Median", "Max", "Min", "Sum")

    try:
        basedir = str(output_dir)
        context = bindings.build_comparison_context()
        if bindings_only_drawing_fn(bindings):
            from openbench.visualization.Mod_Only_Drawing import (
                ComparisonProcessing_only_drawing as ComparisonProcessing,
            )
        else:
            from openbench.core.comparison import ComparisonProcessing

        namelists = context.namelists
        score_vars = context.score_vars
        metric_vars = context.metric_vars

        preflight_errors = post_phase_preflight_errors_fn(
            bindings,
            output_dir,
            list(context.evaluation_items),
            metric_vars,
            score_vars,
            namelists=namelists,
        )
        if preflight_errors:
            logger.warning("Skipping comparison phase: incomplete evaluation outputs")
            return preflight_errors

        evaluation_items = filter_evaluation_items_with_outputs_fn(
            output_dir,
            list(context.evaluation_items),
        )
        if not evaluation_items:
            logger.warning("No evaluation items with data files, skipping comparison phase")
            return phase_errors

        ch = ComparisonProcessing(namelists.main, score_vars, metric_vars)

        comparison_fig = context.comparison_fig

        for cvar in comparison_vars:
            logger.info("Running %s comparison...", cvar)
            if cvar in basic_methods:
                method_name = "scenarios_Basic_comparison"
                fig_opts = dict(comparison_fig.get(cvar) or comparison_fig.get("Basic", {}))
                fig_opts.setdefault("key", cvar)
            else:
                method_name = f"scenarios_{cvar}_comparison"
                fig_opts = comparison_fig.get(cvar, {})

            if hasattr(ch, method_name):
                try:
                    getattr(ch, method_name)(
                        basedir,
                        namelists.simulation,
                        namelists.reference,
                        evaluation_items,
                        score_vars,
                        metric_vars,
                        fig_opts,
                    )
                    logger.info("Completed %s comparison", cvar)
                except Exception as exc:
                    logger.exception("Failed %s comparison", cvar)
                    phase_errors.append(
                        make_phase_error_fn(
                            "comparison",
                            f"{cvar} comparison failed: {exc}",
                            item=cvar,
                            source=method_name,
                        )
                    )
            else:
                logger.warning("Comparison method %s not found, skipping", method_name)
                phase_errors.append(
                    make_phase_error_fn(
                        "comparison",
                        f"{cvar} comparison method not found",
                        item=cvar,
                        source=method_name,
                    )
                )

            gc.collect()

    except ImportError:
        logger.warning("ComparisonProcessing not available, skipping comparison phase")
        phase_errors.append(
            make_phase_error_fn("comparison", "comparison processing is not available", source="ComparisonProcessing")
        )
    except Exception as exc:
        logger.exception("Comparison phase failed")
        phase_errors.append(
            make_phase_error_fn("comparison", f"comparison phase failed: {exc}", source="ComparisonProcessing")
        )

    return phase_errors


def run_groupby(
    cfg: Any,
    bindings: Any,
    output_dir: Path,
    *,
    make_phase_error_fn: MakePhaseError,
    bindings_only_drawing_fn: BindingsOnlyDrawing,
    post_phase_preflight_errors_fn: PostPhasePreflight,
    filter_evaluation_items_with_outputs_fn: ItemFilter,
) -> list[dict[str, Any]]:
    """Run land cover and climate zone groupby analysis."""
    phase_errors: list[dict[str, Any]] = []

    try:
        basedir = str(output_dir)
        context = bindings.build_groupby_context()
        namelists = context.namelists
        score_vars = context.score_vars
        metric_vars = context.metric_vars

        raw_evaluation_items = list(context.evaluation_items)
        preflight_errors = post_phase_preflight_errors_fn(
            bindings,
            output_dir,
            raw_evaluation_items,
            metric_vars,
            score_vars,
            namelists=namelists,
        )
        if preflight_errors:
            logger.warning("Skipping groupby phase: incomplete evaluation outputs")
            return preflight_errors

        evaluation_items = filter_evaluation_items_with_outputs_fn(
            output_dir,
            raw_evaluation_items,
        )
        validation_fig = context.validation_fig
    except Exception as exc:
        logger.exception("Groupby phase failed before dispatch")
        return [make_phase_error_fn("groupby", f"groupby phase failed: {exc}")]

    if not evaluation_items:
        logger.warning("No evaluation items with data files, skipping groupby phase")
        return phase_errors

    if cfg.project.IGBP_groupby:
        try:
            if bindings_only_drawing_fn(bindings):
                from openbench.visualization.Mod_Only_Drawing import LC_groupby_only_drawing as LC_groupby
            else:
                from openbench.core.landcover_groupby import LC_groupby

            logger.info("Running IGBP land cover groupby...")
            lc = LC_groupby(namelists.main, score_vars, metric_vars)
            lc.scenarios_IGBP_groupby_comparison(
                basedir,
                namelists.simulation,
                namelists.reference,
                evaluation_items,
                score_vars,
                metric_vars,
                validation_fig,
            )
            gc.collect()
            logger.info("IGBP groupby complete")
        except Exception as exc:
            logger.exception("IGBP groupby failed")
            phase_errors.append(
                make_phase_error_fn(
                    "groupby",
                    f"IGBP groupby failed: {exc}",
                    item="IGBP_groupby",
                    source="LC_groupby",
                )
            )

    if cfg.project.PFT_groupby:
        try:
            if bindings_only_drawing_fn(bindings):
                from openbench.visualization.Mod_Only_Drawing import LC_groupby_only_drawing as LC_groupby
            else:
                from openbench.core.landcover_groupby import LC_groupby

            logger.info("Running PFT groupby...")
            lc = LC_groupby(namelists.main, score_vars, metric_vars)
            lc.scenarios_PFT_groupby_comparison(
                basedir,
                namelists.simulation,
                namelists.reference,
                evaluation_items,
                score_vars,
                metric_vars,
                validation_fig,
            )
            gc.collect()
            logger.info("PFT groupby complete")
        except Exception as exc:
            logger.exception("PFT groupby failed")
            phase_errors.append(
                make_phase_error_fn(
                    "groupby",
                    f"PFT groupby failed: {exc}",
                    item="PFT_groupby",
                    source="LC_groupby",
                )
            )

    if cfg.project.climate_zone_groupby:
        try:
            if bindings_only_drawing_fn(bindings):
                from openbench.visualization.Mod_Only_Drawing import CZ_groupby_only_drawing as CZ_groupby
            else:
                from openbench.core.climatezone_groupby import CZ_groupby

            logger.info("Running climate zone groupby...")
            cz = CZ_groupby(namelists.main, score_vars, metric_vars)
            cz_fig = context.climate_zone_fig
            cz.scenarios_CZ_groupby_comparison(
                basedir,
                namelists.simulation,
                namelists.reference,
                evaluation_items,
                score_vars,
                metric_vars,
                cz_fig,
            )
            gc.collect()
            logger.info("Climate zone groupby complete")
        except Exception as exc:
            logger.exception("Climate zone groupby failed")
            phase_errors.append(
                make_phase_error_fn(
                    "groupby",
                    f"climate zone groupby failed: {exc}",
                    item="CZ_groupby",
                    source="CZ_groupby",
                )
            )

    return phase_errors


def run_statistics(
    bindings: Any,
    statistic_vars: list[str],
    output_dir: Path | None = None,
    *,
    make_phase_error_fn: MakePhaseError,
    post_phase_preflight_errors_fn: PostPhasePreflight,
    filter_evaluation_items_with_outputs_fn: ItemFilter,
) -> list[dict[str, Any]]:
    """Run statistical analysis."""
    phase_errors: list[dict[str, Any]] = []
    basic_methods = ("Mean", "Median", "Max", "Min", "Sum")

    try:
        from openbench.core.statistics.Mod_Statistics import StatisticsProcessing

        evaluation_items = None
        if output_dir is not None:
            runner_cfg = getattr(bindings, "runner_cfg", None)
            all_items = list(getattr(runner_cfg, "evaluation_items", {}).keys())
            preflight_errors = post_phase_preflight_errors_fn(
                bindings,
                output_dir,
                all_items,
                list(getattr(runner_cfg, "metrics", []) or []),
                list(getattr(runner_cfg, "scores", []) or []),
            )
            if preflight_errors:
                logger.warning("Skipping statistics phase: incomplete evaluation outputs")
                return preflight_errors

            evaluation_items = filter_evaluation_items_with_outputs_fn(
                output_dir,
                all_items,
            )
            if not evaluation_items:
                logger.warning("No evaluation items with data files, skipping statistics phase")
                return phase_errors

        build_statistics_context = bindings.build_statistics_context
        if "evaluation_items" in inspect.signature(build_statistics_context).parameters:
            context = build_statistics_context(
                statistic_vars,
                evaluation_items=evaluation_items,
            )
        else:
            context = bindings.build_statistics_context(statistic_vars)
        main_nl = context.namelists.main
        stats_dir = context.stats_dir
        stats_nml = context.stats_nml
        os.makedirs(stats_dir, exist_ok=True)

        stats_handler = StatisticsProcessing(
            main_nl,
            stats_nml,
            stats_dir,
            num_cores=context.num_cores,
        )

        statistic_fig = context.statistic_fig

        for statistic in statistic_vars:
            logger.info("Running %s analysis...", statistic)
            if statistic in basic_methods:
                method_name = "scenarios_Basic_analysis"
            else:
                method_name = f"scenarios_{statistic}_analysis"

            if hasattr(stats_handler, method_name):
                try:
                    if statistic in basic_methods:
                        stat_fig = dict(statistic_fig.get(statistic) or statistic_fig.get("Basic", {}))
                    else:
                        stat_fig = statistic_fig.get(statistic, {})
                    getattr(stats_handler, method_name)(statistic, stats_nml.get(statistic, {}), stat_fig)
                    logger.info("Completed %s analysis", statistic)
                except Exception as exc:
                    logger.exception("Failed %s analysis", statistic)
                    phase_errors.append(
                        make_phase_error_fn(
                            "statistics",
                            f"{statistic} analysis failed: {exc}",
                            item=statistic,
                            source=method_name,
                        )
                    )
            else:
                logger.warning("Statistics method %s not found, skipping", method_name)
                phase_errors.append(
                    make_phase_error_fn(
                        "statistics",
                        f"{statistic} analysis method not found",
                        item=statistic,
                        source=method_name,
                    )
                )

            gc.collect()

    except ImportError:
        logger.warning("StatisticsProcessing not available, skipping statistics phase")
        phase_errors.append(
            make_phase_error_fn("statistics", "statistics processing is not available", source="StatisticsProcessing")
        )
    except Exception as exc:
        logger.exception("Statistics phase failed")
        phase_errors.append(
            make_phase_error_fn("statistics", f"statistics phase failed: {exc}", source="StatisticsProcessing")
        )

    return phase_errors


def run_report(
    bindings: Any,
    output_dir: Path,
    *,
    make_phase_error_fn: MakePhaseError,
) -> list[dict[str, Any]]:
    """Generate evaluation report."""
    phase_errors: list[dict[str, Any]] = []
    try:
        from openbench.util.report import ReportGenerator

        report_config = bindings.build_report_config().to_report_config()

        report_gen = ReportGenerator(report_config, str(output_dir))
        report_paths = report_gen.generate_report()

        if report_paths:
            for fmt, path in report_paths.items():
                logger.info("Report generated: %s → %s", fmt, path)
        else:
            logger.info("Report generation completed")

    except ImportError:
        logger.warning("ReportGenerator not available, skipping report generation")
        phase_errors.append(make_phase_error_fn("report", "report generation is not available"))
    except Exception as exc:
        logger.exception("Report generation failed")
        phase_errors.append(make_phase_error_fn("report", f"report generation failed: {exc}"))

    return phase_errors
