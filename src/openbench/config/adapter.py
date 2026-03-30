"""Adapter to convert new OpenBenchConfig to legacy dict format.

The evaluation engine expects config as nested dicts with specific key
patterns. This adapter translates the new dataclass-based config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


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


def build_legacy_namelists(cfg: OpenBenchConfig) -> tuple[dict, dict, dict]:
    """Build legacy ref_nml, sim_nml, and main_nl from new config.

    Uses the registry to resolve reference dataset variable mappings
    and model profiles to resolve simulation variable mappings.

    Returns:
        (main_nl, ref_nml, sim_nml) tuple of legacy-format dicts.
    """
    from openbench.data.registry.manager import RegistryManager

    registry = RegistryManager()
    legacy = to_legacy_config(cfg)

    # --- main_nl: general settings + evaluation_items ---
    main_nl = {
        "general": legacy["general"],
        "evaluation_items": legacy["evaluation_items"],
        "metrics": legacy["metrics"],
        "scores": legacy["scores"],
        "comparisons": legacy["comparisons"],
        "statistics": legacy["statistics"],
    }

    # --- ref_nml ---
    ref_general: dict[str, Any] = {}
    ref_sections: dict[str, dict[str, Any]] = {}

    for var_name in cfg.evaluation.variables:
        ref_source_name = cfg.reference.get(var_name)
        if not ref_source_name:
            logger.warning("No reference source configured for variable %s, skipping", var_name)
            continue

        ref_general[f"{var_name}_ref_source"] = ref_source_name

        ref_ds = registry.get_reference(ref_source_name)
        section: dict[str, Any] = {}
        prefix = ref_source_name

        if ref_ds is not None:
            var_map = ref_ds.variables.get(var_name)
            if var_map is None:
                logger.warning(
                    "Variable %s not found in reference %s registry entry", var_name, ref_source_name
                )
                continue

            # Construct directory: root_dir / sub_dir (if both present)
            ref_dir = ref_ds.root_dir or ""
            if var_map.sub_dir:
                ref_dir = os.path.join(ref_dir, var_map.sub_dir) if ref_dir else var_map.sub_dir

            section[f"{prefix}_data_type"] = ref_ds.data_type
            section[f"{prefix}_varname"] = var_map.varname
            section[f"{prefix}_varunit"] = var_map.varunit
            section[f"{prefix}_data_groupby"] = ref_ds.data_groupby
            section[f"{prefix}_tim_res"] = ref_ds.tim_res
            section[f"{prefix}_grid_res"] = ref_ds.grid_res
            section[f"{prefix}_syear"] = ref_ds.years[0] if ref_ds.years else cfg.project.years[0]
            section[f"{prefix}_eyear"] = ref_ds.years[1] if len(ref_ds.years) > 1 else cfg.project.years[1]
            section[f"{prefix}_dir"] = ref_dir
            section[f"{prefix}_prefix"] = var_map.prefix
            section[f"{prefix}_suffix"] = var_map.suffix
            section[f"{prefix}_timezone"] = ref_ds.timezone

            # Optional station-related fields
            if var_map.fulllist:
                section[f"{prefix}_fulllist"] = var_map.fulllist
            elif ref_ds.fulllist:
                section[f"{prefix}_fulllist"] = ref_ds.fulllist
            if var_map.max_uparea is not None:
                section[f"{prefix}_max_uparea"] = var_map.max_uparea
            if var_map.min_uparea is not None:
                section[f"{prefix}_min_uparea"] = var_map.min_uparea
        else:
            logger.warning(
                "Reference %s not found in registry; variable %s will need inline config",
                ref_source_name,
                var_name,
            )

        ref_sections[var_name] = section

    ref_nml = {"general": ref_general, **ref_sections}

    # --- sim_nml ---
    sim_general: dict[str, Any] = {}
    sim_sections: dict[str, dict[str, Any]] = {}

    for var_name in cfg.evaluation.variables:
        sim_sources: list[str] = []
        var_section: dict[str, Any] = {}

        for sim_label, sim_entry in cfg.simulation.items():
            model_name = sim_entry.model
            sim_sources.append(sim_label)

            model_profile = registry.get_model(model_name)

            # Determine variable mapping: inline overrides > model profile > fallback
            inline_vars = (sim_entry.variables or {}).get(var_name, {})

            if model_profile and var_name in model_profile.variables:
                profile_var = model_profile.variables[var_name]
                varname = inline_vars.get("varname", profile_var.varname)
                varunit = inline_vars.get("varunit", profile_var.varunit)
                var_prefix = inline_vars.get("prefix", profile_var.prefix)
                var_suffix = inline_vars.get("suffix", profile_var.suffix)
            elif inline_vars:
                varname = inline_vars.get("varname", var_name)
                varunit = inline_vars.get("varunit", "")
                var_prefix = inline_vars.get("prefix", "")
                var_suffix = inline_vars.get("suffix", "")
            else:
                logger.warning(
                    "No variable mapping for %s in model %s (label %s); using variable name as varname",
                    var_name,
                    model_name,
                    sim_label,
                )
                varname = var_name
                varunit = ""
                var_prefix = ""
                var_suffix = ""

            # Data type / resolution: inline override > sim_entry override > model profile > defaults
            data_type = sim_entry.data_type or (model_profile.data_type if model_profile else "grid")
            grid_res = sim_entry.grid_res or (model_profile.grid_res if model_profile else None)
            tim_res = sim_entry.tim_res or (model_profile.tim_res if model_profile else "Month")

            # Construct sim directory: root_dir from sim_entry, optionally with sub_dir from profile
            sim_dir = sim_entry.root_dir
            if model_profile and var_name in model_profile.variables:
                profile_sub = model_profile.variables[var_name].sub_dir
                if profile_sub:
                    sim_dir = os.path.join(sim_dir, profile_sub)

            prefix = sim_label
            var_section[f"{prefix}_data_type"] = data_type
            var_section[f"{prefix}_varname"] = varname
            var_section[f"{prefix}_varunit"] = varunit
            var_section[f"{prefix}_data_groupby"] = inline_vars.get("data_groupby", "Year")
            var_section[f"{prefix}_tim_res"] = tim_res
            var_section[f"{prefix}_grid_res"] = grid_res
            var_section[f"{prefix}_syear"] = cfg.project.years[0]
            var_section[f"{prefix}_eyear"] = cfg.project.years[1]
            var_section[f"{prefix}_dir"] = sim_dir
            var_section[f"{prefix}_prefix"] = var_prefix
            var_section[f"{prefix}_suffix"] = var_suffix
            var_section[f"{prefix}_timezone"] = inline_vars.get("timezone", 0)

            # Optional station-related fields from inline config
            if "fulllist" in inline_vars:
                var_section[f"{prefix}_fulllist"] = inline_vars["fulllist"]
            if "max_uparea" in inline_vars:
                var_section[f"{prefix}_max_uparea"] = inline_vars["max_uparea"]
            if "min_uparea" in inline_vars:
                var_section[f"{prefix}_min_uparea"] = inline_vars["min_uparea"]

        sim_general[f"{var_name}_sim_source"] = sim_sources
        sim_sections[var_name] = var_section

    sim_nml = {"general": sim_general, **sim_sections}

    return main_nl, ref_nml, sim_nml
