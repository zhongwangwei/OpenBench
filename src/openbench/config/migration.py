"""Migrate old JSON/NML configs to the new unified YAML format.

Reads the old multi-file config structure (main + ref + sim + variable defs)
and produces a single openbench.yaml in the new format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def migrate_config(main_config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convert old-format config to new unified YAML.

    Args:
        main_config_path: Path to the old main config file (JSON, YAML, or NML).
        output_path: Where to write the new openbench.yaml.

    Returns:
        Summary dict with migration statistics.

    Raises:
        FileNotFoundError: If the main config file doesn't exist.
    """
    main_config_path = Path(main_config_path)
    output_path = Path(output_path)

    if not main_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {main_config_path}")

    # Read main config
    main = _read_old_config(main_config_path)
    files_read = 1

    general = main.get("general", {})
    base_dir = main_config_path.parent

    # Read reference config
    ref_sources = {}
    ref_nml_path = general.get("reference_nml")
    if ref_nml_path:
        ref_path = _resolve_path(ref_nml_path, base_dir)
        if ref_path.exists():
            ref_config = _read_old_config(ref_path)
            files_read += 1
            ref_general = ref_config.get("general", {})

            # Extract source names per variable
            for key, value in ref_general.items():
                if key.endswith("_ref_source"):
                    var_name = key.replace("_ref_source", "")
                    ref_sources[var_name] = value

    # Read simulation config
    sim_entries = {}
    sim_nml_path = general.get("simulation_nml")
    if sim_nml_path:
        sim_path = _resolve_path(sim_nml_path, base_dir)
        if sim_path.exists():
            sim_config = _read_old_config(sim_path)
            files_read += 1
            sim_general = sim_config.get("general", {})
            sim_def_nml = sim_config.get("def_nml", {})

            # Collect all unique simulation source names
            all_sim_sources: set[str] = set()
            for key, value in sim_general.items():
                if key.endswith("_sim_source"):
                    sources = value if isinstance(value, list) else [value]
                    all_sim_sources.update(sources)

            # Read each simulation definition file
            for source_name in sorted(all_sim_sources):
                if source_name in sim_def_nml:
                    def_path = _resolve_path(sim_def_nml[source_name], base_dir)
                    if def_path.exists():
                        sim_def = _read_old_config(def_path)
                        files_read += 1

                        sim_general_def = sim_def.get("general", {})
                        variables = {}
                        for var_key, var_val in sim_def.items():
                            if var_key != "general" and isinstance(var_val, dict):
                                variables[var_key] = var_val

                        entry: dict[str, Any] = {
                            "model": source_name,
                            "root_dir": sim_general_def.get("root_dir", sim_general_def.get("dir", "")),
                        }
                        if sim_general_def.get("data_type"):
                            entry["data_type"] = sim_general_def["data_type"]
                        if sim_general_def.get("grid_res"):
                            entry["grid_res"] = sim_general_def["grid_res"]
                        if sim_general_def.get("tim_res"):
                            entry["tim_res"] = sim_general_def["tim_res"]
                        if variables:
                            entry["variables"] = variables

                        sim_entries[source_name] = entry

    # Build new config
    eval_items = main.get("evaluation_items", {})
    enabled_variables = [k for k, v in eval_items.items() if v]

    metrics_dict = main.get("metrics", {})
    enabled_metrics = [k for k, v in metrics_dict.items() if v]

    scores_dict = main.get("scores", {})
    enabled_scores = [k for k, v in scores_dict.items() if v]

    comparisons_dict = main.get("comparisons", {})
    enabled_comparisons = [k for k, v in comparisons_dict.items() if v]

    # Filter reference to only enabled variables
    filtered_ref = {var: ref_sources[var] for var in enabled_variables if var in ref_sources}

    new_config: dict[str, Any] = {
        "project": {
            "name": general.get("basename", "migrated"),
            "output_dir": general.get("basedir", "./output"),
            "years": [general.get("syear", 2000), general.get("eyear", 2020)],
        },
        "evaluation": {
            "variables": enabled_variables,
        },
        "reference": filtered_ref,
        "simulation": sim_entries if sim_entries else {"default": {"model": "unknown", "root_dir": "."}},
    }

    if enabled_metrics:
        new_config["metrics"] = enabled_metrics
    if enabled_scores:
        new_config["scores"] = enabled_scores

    comparison_enabled = general.get("comparison", False)
    if comparison_enabled or enabled_comparisons:
        comp: dict[str, Any] = {"enabled": bool(comparison_enabled)}
        if enabled_comparisons:
            comp["items"] = enabled_comparisons
        new_config["comparison"] = comp

    # Options
    options: dict[str, Any] = {}
    if general.get("num_cores"):
        options["num_cores"] = general["num_cores"]
    if general.get("unified_mask") is not None:
        options["unified_mask"] = general["unified_mask"]
    if general.get("generate_report") is not None:
        options["generate_report"] = general["generate_report"]
    if general.get("IGBP_groupby"):
        options["IGBP_groupby"] = True
    if general.get("PFT_groupby"):
        options["PFT_groupby"] = True
    if general.get("Climate_zone_groupby"):
        options["climate_zone_groupby"] = True
    if options:
        new_config["options"] = options

    # Add non-default spatial/temporal bounds
    if general.get("min_year"):
        new_config["project"]["min_year_threshold"] = general["min_year"]
    lat_range = [general.get("min_lat", -90), general.get("max_lat", 90)]
    lon_range = [general.get("min_lon", -180), general.get("max_lon", 180)]
    if lat_range != [-90, 90]:
        new_config["project"]["lat_range"] = lat_range
    if lon_range != [-180, 180]:
        new_config["project"]["lon_range"] = lon_range

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return {
        "files_read": files_read,
        "variables": enabled_variables,
        "simulations": list(sim_entries.keys()),
        "metrics": enabled_metrics,
        "scores": enabled_scores,
    }


def _read_old_config(path: Path) -> dict:
    """Read an old config file (JSON or YAML)."""
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix == ".json":
            return json.load(f)
        elif suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        elif suffix == ".nml":
            try:
                import f90nml

                nml = f90nml.read(path)
                return dict(nml)
            except ImportError:
                raise ImportError("Migrating Fortran NML files requires f90nml: pip install f90nml")
        else:
            raise ValueError(f"Unsupported config format: {suffix}")


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir if not absolute.

    First tries resolving relative to base_dir; if the result doesn't exist,
    falls back to resolving relative to the current working directory. This
    handles fixtures whose embedded paths are relative to the project root
    rather than to the config file's own directory.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p

    candidate = base_dir / p
    if candidate.exists():
        return candidate

    # Fall back to CWD-relative resolution
    return Path(path_str)
