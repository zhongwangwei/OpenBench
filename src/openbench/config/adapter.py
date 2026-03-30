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


def _resolve_varname(varname, root_dir: str | None = None) -> str:
    """Resolve a variable name, handling fallback chains.

    If varname is a list like ["f_gpp", "f_assim"], check the first NC file
    in root_dir to find which variable actually exists. Returns the first match.
    If varname is a plain string, return it directly.
    """
    if isinstance(varname, str):
        return varname

    if not isinstance(varname, list) or not varname:
        return str(varname)

    # If no root_dir to check, return first option
    if not root_dir:
        return varname[0]

    # Try to find which varname exists in the data files
    import glob
    from pathlib import Path

    nc_files = sorted(glob.glob(str(Path(root_dir) / "*.nc")))
    if not nc_files:
        return varname[0]

    try:
        import xarray as xr

        ds = xr.open_dataset(nc_files[0])
        available = set(ds.data_vars)
        ds.close()

        for vn in varname:
            if vn in available:
                logger.info("Resolved varname fallback: %s → %s (from %s)", varname, vn, Path(nc_files[0]).name)
                return vn

        logger.warning("None of %s found in %s, using first: %s", varname, Path(nc_files[0]).name, varname[0])
    except Exception:
        pass

    return varname[0]


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


def build_fig_nml() -> dict[str, Any]:
    """Build the figure namelist from bundled figure config files.

    Reads all figure config YAML files from the package's data/fignml/ directory
    and organizes them into the structure expected by the evaluation code:
        fig_nml["make_geo_plot_index"] = {...}   (validation configs, flattened)
        fig_nml["Comparison"]["Taylor_Diagram"] = {...}
        fig_nml["Statistic"]["Basic"] = {...}

    Returns:
        Processed fig_nml dict.
    """
    from pathlib import Path

    import yaml

    fignml_dir = Path(__file__).parent.parent / "data" / "fignml"
    figlib_path = fignml_dir / "figlib.yaml"

    if not figlib_path.exists():
        logger.warning("figlib.yaml not found at %s, visualization will be skipped", figlib_path)
        return {}

    with open(figlib_path) as f:
        figlib = yaml.safe_load(f)

    fig_nml: dict[str, Any] = {}

    # Process validation configs — keys go directly into fig_nml (flattened)
    for key, rel_path in figlib.get("validation_nml", {}).items():
        config_name = key.replace("_source", "")
        filename = Path(rel_path).name
        config_path = fignml_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            fig_nml[config_name] = data.get("general", data)
        else:
            logger.debug("Figure config not found: %s", config_path)

    # Process comparison configs — nested under fig_nml["Comparison"]
    comparison = {}
    for key, rel_path in figlib.get("comparison_nml", {}).items():
        config_name = key.replace("_source", "")
        filename = Path(rel_path).name
        config_path = fignml_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            comparison[config_name] = data.get("general", data)
    fig_nml["Comparison"] = comparison

    # Process statistic configs — nested under fig_nml["Statistic"]
    statistic = {}
    for key, rel_path in figlib.get("statistic_nml", {}).items():
        config_name = key.replace("_source", "")
        filename = Path(rel_path).name
        config_path = fignml_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            statistic[config_name] = data.get("general", data)
    fig_nml["Statistic"] = statistic

    # Keep raw registry sections for UpdateFigNamelist compatibility
    fig_nml["validation_nml"] = figlib.get("validation_nml", {})
    fig_nml["comparison_nml"] = figlib.get("comparison_nml", {})
    fig_nml["statistic_nml"] = figlib.get("statistic_nml", {})

    return fig_nml


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

        # Derive target resolution for auto-resolve:
        # Use user-specified comparison resolution (both ref and sim will be regridded to this)
        # Fallback to simulation resolution if comparison not specified
        target_tim_res = cfg.comparison.tim_res
        target_grid_res = cfg.comparison.grid_res
        if not target_tim_res:
            for entry in cfg.simulation.values():
                target_tim_res = entry.tim_res or target_tim_res
                break
        if not target_grid_res:
            for entry in cfg.simulation.values():
                target_grid_res = entry.grid_res or target_grid_res
                break

        ref_ds = registry.get_reference(
            ref_source_name, sim_tim_res=target_tim_res, sim_grid_res=target_grid_res
        )

        # Update source name to resolved name (may differ from base name)
        resolved_name = ref_ds.name if ref_ds else ref_source_name
        ref_general[f"{var_name}_ref_source"] = resolved_name
        section: dict[str, Any] = {}
        prefix = resolved_name

        if ref_ds is not None:
            var_map = ref_ds.variables.get(var_name)
            if var_map is None:
                logger.warning("Variable %s not found in reference %s registry entry", var_name, ref_source_name)
                continue

            # Construct directory: data_root / sub_dir or root_dir / sub_dir
            data_root = cfg.options.data_root or ref_ds.root_dir or ""
            ref_dir = data_root
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

            # Entry-level prefix/suffix (shared across all variables for this sim)
            entry_prefix = sim_entry.prefix or ""
            entry_suffix = sim_entry.suffix or ""

            if model_profile and var_name in model_profile.variables:
                profile_var = model_profile.variables[var_name]
                raw_varname = inline_vars.get("varname", profile_var.varname)
                varname = _resolve_varname(raw_varname, sim_entry.root_dir)
                varunit = inline_vars.get("varunit", profile_var.varunit)
                var_prefix = inline_vars.get("prefix", entry_prefix or profile_var.prefix)
                var_suffix = inline_vars.get("suffix", entry_suffix or profile_var.suffix)
            elif inline_vars:
                varname = inline_vars.get("varname", var_name)
                varunit = inline_vars.get("varunit", "")
                var_prefix = inline_vars.get("prefix", entry_prefix)
                var_suffix = inline_vars.get("suffix", entry_suffix)
            else:
                logger.warning(
                    "No variable mapping for %s in model %s (label %s); using variable name as varname",
                    var_name,
                    model_name,
                    sim_label,
                )
                varname = var_name
                varunit = ""
                var_prefix = entry_prefix
                var_suffix = entry_suffix

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
            var_section[f"{prefix}_data_groupby"] = (
                sim_entry.data_groupby or inline_vars.get("data_groupby", "Year")
            )
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
