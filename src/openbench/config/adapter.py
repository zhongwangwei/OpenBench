"""Adapter to convert new OpenBenchConfig to legacy dict format.

The evaluation engine expects config as nested dicts with specific key
patterns. This adapter translates the new dataclass-based config.
"""

from __future__ import annotations

import os
from typing import Any

from openbench.config.schema import OpenBenchConfig


def to_legacy_config(cfg: OpenBenchConfig) -> dict[str, Any]:
    """Convert OpenBenchConfig to the legacy dict format."""
    general = {
        "basename": cfg.project.name,
        "basedir": cfg.project.output_dir,
        "syear": cfg.project.years[0],
        "eyear": cfg.project.years[1],
        "min_year": cfg.project.min_year_threshold,
        "min_lat": cfg.project.lat_range[0],
        "max_lat": cfg.project.lat_range[1],
        "min_lon": cfg.project.lon_range[0],
        "max_lon": cfg.project.lon_range[1],
        "num_cores": cfg.options.num_cores or max(1, os.cpu_count() or 1),
        "evaluation": True,
        "comparison": cfg.comparison.enabled,
        "statistics": cfg.statistics.enabled,
        "debug_mode": cfg.options.debug_mode,
        "only_drawing": cfg.options.only_drawing,
        "IGBP_groupby": cfg.options.IGBP_groupby,
        "PFT_groupby": cfg.options.PFT_groupby,
        "Climate_zone_groupby": cfg.options.climate_zone_groupby,
        "unified_mask": cfg.options.unified_mask,
        "generate_report": cfg.options.generate_report,
        "weight": cfg.comparison.weight or "area",
        "compare_tim_res": cfg.comparison.tim_res or "Month",
        "compare_tzone": cfg.comparison.timezone or 0,
        "compare_grid_res": cfg.comparison.grid_res or 0.5,
    }

    evaluation_items = {var: True for var in cfg.evaluation.variables}

    if cfg.metrics:
        metrics_dict = {m: True for m in cfg.metrics}
    else:
        metrics_dict = {"bias": True, "RMSE": True, "correlation": True}

    if cfg.scores:
        scores_dict = {s: True for s in cfg.scores}
    else:
        scores_dict = {"Overall_Score": True}

    if cfg.comparison.items:
        comparisons_dict = {c: True for c in cfg.comparison.items}
    else:
        comparisons_dict = {"Taylor_Diagram": True, "HeatMap": True}

    if cfg.statistics.items:
        statistics_dict = {s: True for s in cfg.statistics.items}
    else:
        statistics_dict = {}

    return {
        "general": general,
        "evaluation_items": evaluation_items,
        "metrics": metrics_dict,
        "scores": scores_dict,
        "comparisons": comparisons_dict,
        "statistics": statistics_dict,
    }
