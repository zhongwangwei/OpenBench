"""Migrate old JSON/NML configs to the new unified YAML format.

Reads the old multi-file config structure (main + ref + sim + variable defs)
and produces a single openbench.yaml in the new format.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from openbench.util.netcdf import write_file_atomic

logger = logging.getLogger(__name__)


def _write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write migrated YAML through a same-directory temp file."""

    def writer(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                data,
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    write_file_atomic(path, writer, suffix=".tmp.yaml")


def _unique_label(entries: dict[str, Any], label: str) -> str:
    """Return a non-overwriting label for migrated simulation entries."""
    if label not in entries:
        return label
    idx = 2
    while f"{label}_{idx}" in entries:
        idx += 1
    return f"{label}_{idx}"


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

    if _looks_like_modern_config(main):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_yaml_atomic(output_path, main)
        return {
            "files_read": files_read,
            "variables": list((main.get("evaluation") or {}).get("variables") or []),
            "simulations": [key for key in (main.get("simulation") or {}) if key != "_defaults"],
            "metrics": list(main.get("metrics") or []),
            "scores": list(main.get("scores") or []),
            "statistics": list((main.get("statistics") or {}).get("items") or []),
            "already_modern": True,
        }

    general = main.get("general", {})
    base_dir = main_config_path.parent

    # Read reference config
    ref_sources = {}
    # Per-source data paths from the legacy two-level nml (top-level def_nml
    # → per-source nml → general.root_dir). Without this, migrated configs
    # silently fall back to the registry's hard-coded root_dir which may
    # not match the user's actual data layout.
    ref_per_source_root: dict[str, str] = {}
    ref_nml_path = general.get("reference_nml")
    if ref_nml_path:
        ref_path = _resolve_path(ref_nml_path, base_dir)
        if ref_path.exists():
            ref_config = _read_old_config(ref_path)
            files_read += 1
            ref_general = ref_config.get("general", {})
            ref_def_nml = ref_config.get("def_nml", {})

            # Extract source names per variable. A `*_ref_source` value can
            # be a single name or a comma-separated / list form.
            for key, value in ref_general.items():
                if key.endswith("_ref_source"):
                    var_name = key.replace("_ref_source", "")
                    ref_sources[var_name] = value

            # Resolve def_nml → per-source nml → general.root_dir so we can
            # surface the user's real reference paths in the migrated config.
            # f90nml lower-cases keys, so we store with lowercase keys and
            # do case-insensitive lookups against ref_sources values.
            for source_name, source_nml_path in ref_def_nml.items():
                if not isinstance(source_nml_path, str):
                    continue
                sub_path = _resolve_path(source_nml_path, base_dir)
                if not sub_path.exists():
                    logger.warning(
                        "Migration: reference def_nml entry %s -> %s does not exist; skipping",
                        source_name,
                        sub_path,
                    )
                    continue
                try:
                    sub_config = _read_old_config(sub_path)
                    files_read += 1
                    sub_general = sub_config.get("general", {})
                    root_dir = str(sub_general.get("root_dir", "")).strip()
                    if root_dir:
                        ref_per_source_root[str(source_name).lower()] = root_dir
                except Exception as e:
                    logger.warning("Migration: failed to read per-source ref nml %s: %s", sub_path, e)

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

                        # Detect model from model_namelist path
                        model_name = _detect_model(sim_general_def.get("model_namelist", ""))

                        root_dir = sim_general_def.get("root_dir", sim_general_def.get("dir", ""))
                        prefix = sim_general_def.get("prefix", "")

                        # Derive a clean label from root_dir or prefix
                        label = _unique_label(sim_entries, _derive_case_label(source_name, root_dir, prefix))

                        entry: dict[str, Any] = {
                            "model": model_name,
                            "root_dir": root_dir,
                        }
                        for field in (
                            "data_type",
                            "grid_res",
                            "data_groupby",
                            "prefix",
                            "suffix",
                            "fulllist",
                        ):
                            if field in sim_general_def and sim_general_def[field] is not None:
                                entry[field] = sim_general_def[field]
                        if sim_general_def.get("tim_res") is not None:
                            entry["tim_res"] = _normalize_tim_res(sim_general_def["tim_res"])

                        # Preserve per-variable legacy NML fields as inline
                        # overrides even when the model has a registry profile.
                        # The v3 adapter resolves inline overrides before
                        # profile defaults, so dropping them here silently
                        # changes user-provided varname/varunit/prefix/etc.
                        variables = {}
                        for var_key, var_val in sim_def.items():
                            if var_key != "general" and isinstance(var_val, dict):
                                variables[var_key] = var_val
                        if variables:
                            entry["variables"] = variables

                        sim_entries[label] = entry

    # Build new config
    eval_items = main.get("evaluation_items") or {}
    enabled_variables = [k for k, v in eval_items.items() if v]

    metrics_dict = main.get("metrics") or {}
    enabled_metrics = [k for k, v in metrics_dict.items() if v]

    scores_dict = main.get("scores") or {}
    enabled_scores = [k for k, v in scores_dict.items() if v]

    comparisons_dict = main.get("comparisons") or {}
    enabled_comparisons = [k for k, v in comparisons_dict.items() if v]

    # Filter reference to only enabled variables
    filtered_ref: dict[str, Any] = {var: ref_sources[var] for var in enabled_variables if var in ref_sources}

    # Compute reference.data_root from per-source root_dir values discovered
    # in the legacy def_nml chain. Strategy:
    #   - Restrict to sources that ended up referenced by enabled variables.
    #   - If they share a non-trivial common path prefix, write that path
    #     as `reference.data_root` so adapter.py prefers it over the
    #     registry's hard-coded root_dir.
    #   - If paths diverge (no usable common prefix), warn and leave
    #     data_root unset; the user must pick one or per-source override.
    used_source_names: set[str] = set()
    for raw_value in filtered_ref.values():
        if isinstance(raw_value, list):
            for item in raw_value:
                if isinstance(item, str):
                    used_source_names.update(s.strip() for s in item.split(","))
        elif isinstance(raw_value, str):
            used_source_names.update(s.strip() for s in raw_value.split(","))

    # Case-insensitive lookup: ref_per_source_root keys are lower-cased
    # because f90nml lower-cases all namelist identifiers.
    used_roots = [ref_per_source_root[s.lower()] for s in used_source_names if s.lower() in ref_per_source_root]
    if used_roots:
        # commonpath raises ValueError if mixing absolute/relative or empty
        try:
            common = os.path.commonpath(used_roots)
        except ValueError:
            common = ""
        # Reject trivially-empty / root-only prefixes
        if common and common not in ("/", "."):
            filtered_ref["data_root"] = common
            if len(set(used_roots)) > 1:
                logger.info(
                    "Migration: reference paths share common prefix %s; "
                    "writing as reference.data_root. Per-source roots were: %s",
                    common,
                    ref_per_source_root,
                )
        else:
            logger.warning(
                "Migration: legacy reference root_dir paths have no usable "
                "common prefix; reference.data_root left unset. Migrated "
                "config will resolve via registry root_dir, which may not "
                "match your data layout. Legacy per-source roots: %s",
                {s: ref_per_source_root[s.lower()] for s in used_source_names if s.lower() in ref_per_source_root},
            )

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
        "simulation": _build_simulation_section(sim_entries),
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
    if general.get("num_cores") is not None:
        options["num_cores"] = general["num_cores"]
    if general.get("unified_mask") is not None:
        options["unified_mask"] = general["unified_mask"]
    if general.get("generate_report") is not None:
        options["generate_report"] = general["generate_report"]
    general_ci = {str(k).lower(): v for k, v in general.items()}
    if general_ci.get("igbp_groupby"):
        options["IGBP_groupby"] = True
    if general_ci.get("pft_groupby"):
        options["PFT_groupby"] = True
    if general_ci.get("climate_zone_groupby"):
        options["climate_zone_groupby"] = True
    if options:
        new_config["project"].update(options)

    # Add non-default spatial/temporal bounds
    if general.get("min_year") is not None:
        new_config["project"]["min_year_threshold"] = general["min_year"]
    lat_range = [general.get("min_lat", -90), general.get("max_lat", 90)]
    lon_range = [general.get("min_lon", -180), general.get("max_lon", 180)]
    if lat_range != [-90, 90]:
        new_config["project"]["lat_range"] = lat_range
    if lon_range != [-180, 180]:
        new_config["project"]["lon_range"] = lon_range

    # Migrate legacy `compare_*` resolution + `weight` fields. Without
    # these the new YAML silently falls back to adapter defaults
    # ("Month" / 0.5), which usually does NOT match what the user had.
    if general.get("compare_tim_res") is not None:
        new_config["project"]["tim_res"] = _normalize_tim_res(general["compare_tim_res"])
    if general.get("compare_grid_res") is not None:
        new_config["project"]["grid_res"] = general["compare_grid_res"]
    if general.get("compare_tzone") is not None:
        new_config["project"]["timezone"] = general["compare_tzone"]
    if general.get("weight") is not None:
        new_config["project"]["weight"] = general["weight"]

    # Migrate the statistics section. Legacy uses a separate `statistics_nml`
    # file plus a boolean `statistics` toggle in the main config; the new
    # schema collapses both into one section.
    stats_enabled = bool(general.get("statistics", False))
    stats_items: list[str] = []
    stats_nml_path = general.get("statistics_nml")
    if stats_nml_path:
        stats_path = _resolve_path(stats_nml_path, base_dir)
        if stats_path.exists():
            try:
                stats_config = _read_old_config(stats_path)
                files_read += 1
                # Top-level section names are the method identifiers used by
                # the runner; skip "general" which holds method→data_source
                # routing (not a method itself).
                stats_items = [k for k in stats_config.keys() if k != "general" and isinstance(stats_config[k], dict)]
            except Exception as exc:
                logger.warning(
                    "Migration: failed to read statistics_nml %s: %s",
                    stats_path,
                    exc,
                )
        else:
            logger.warning(
                "Migration: statistics_nml path %s does not exist; statistics.items left empty.",
                stats_path,
            )
    if stats_enabled or stats_items:
        new_config["statistics"] = {
            "enabled": stats_enabled,
            **({"items": stats_items} if stats_items else {}),
        }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml_atomic(output_path, new_config)

    return {
        "files_read": files_read,
        "variables": enabled_variables,
        "simulations": list(sim_entries.keys()),
        "metrics": enabled_metrics,
        "scores": enabled_scores,
        "statistics": stats_items,
    }


_TIM_RES_NORMALIZE = {
    "month": "Month",
    "monthly": "Month",
    "day": "Day",
    "daily": "Day",
    "hour": "Hour",
    "hourly": "Hour",
    "year": "Year",
    "yearly": "Year",
    "annual": "Year",
    "3hour": "3Hour",
    "3hr": "3Hour",
    "6hour": "6Hour",
    "6hr": "6Hour",
    "8day": "8Day",
    "8d": "8Day",
    "climatology-month": "climatology-month",
    "climatology-year": "climatology-year",
}


def _normalize_tim_res(value: Any) -> Any:
    """Map legacy lower-case tim_res strings to the schema's Title-case form.

    Pass through unrecognised values unchanged so the loader can still
    surface a friendly validation error to the user instead of silently
    rewriting it.
    """
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if re.fullmatch(r"[1-9]\d*month", lowered):
        return lowered
    return _TIM_RES_NORMALIZE.get(lowered, value)


def _looks_like_modern_config(data: dict[str, Any]) -> bool:
    """Return True for v3 unified YAML configs.

    A repeated migrate invocation should be idempotent. Treat a file that
    already has the canonical top-level sections as modern and copy it
    through unchanged instead of interpreting it as legacy JSON/NML with no
    ``general`` section, which would otherwise rewrite it to a mostly-empty
    default config.
    """
    required = ("project", "evaluation", "reference", "simulation")
    return all(isinstance(data.get(section), dict) for section in required)


def _build_simulation_section(sim_entries: dict[str, Any]) -> dict[str, Any]:
    """Build simulation section, extracting _defaults for shared fields.

    If all entries share the same model/data_type/grid_res/tim_res/data_groupby,
    extract them into _defaults to avoid repetition.
    """
    if not sim_entries:
        return {"default": {"model": "unknown", "root_dir": "."}}

    if len(sim_entries) < 2:
        return sim_entries

    # Find common fields across all entries
    common_keys = ["model", "data_type", "grid_res", "tim_res", "data_groupby"]
    defaults: dict[str, Any] = {}

    first = next(iter(sim_entries.values()))
    for key in common_keys:
        val = first.get(key)
        if val is not None and all(e.get(key) == val for e in sim_entries.values()):
            defaults[key] = val

    if not defaults:
        return sim_entries

    # Build section with _defaults
    result: dict[str, Any] = {"_defaults": defaults}
    for label, entry in sim_entries.items():
        cleaned = {}
        for k, v in entry.items():
            if k in defaults and defaults[k] == v:
                continue  # Skip fields covered by _defaults
            cleaned[k] = v
        result[label] = cleaned

    return result


def _detect_model(model_namelist_path: str) -> str:
    """Detect the model name from model_namelist path.

    Examples:
        './nml/Mod_variables_definition/CoLM.nml' → 'CoLM2024'
        './nml/nml-yaml/Mod_variables_definition/CoLM.yaml' → 'CoLM2024'
        './nml/Mod_variables_definition/LS3MIP.nml' → 'LS3MIP'
    """
    if not model_namelist_path:
        return "unknown"

    name = Path(model_namelist_path).stem.lower()

    # Map known model definition filenames to registry model names
    model_map = {
        "colm": "CoLM2024",
        "clm": "CLM5",
        "clm5": "CLM5",
        "noah": "NOAH",
        "era5land": "ERA5-Land",
        "era5-land": "ERA5-Land",
        "gldas": "GLDAS",
    }

    return model_map.get(name, name)


def _derive_case_label(source_name: str, root_dir: str, prefix: str) -> str:
    """Derive a clean case label from path/prefix info.

    Examples:
        root_dir='./dataset/Simulation/Case01/history/', prefix='Case01_hist_' → 'Case01'
        root_dir='./dataset/Simulation/trajectory2/01/history/', prefix='01_case_hist_' → 'Case01'
        source_name='01_case' → 'Case01'
    """
    import re

    # Try to extract CaseXX from root_dir
    match = re.search(r"(Case\d+)", root_dir)
    if match:
        return match.group(1)

    # Try to extract from prefix: "01_case_hist_" → Case01
    match = re.search(r"(\d+)_case", prefix)
    if match:
        return f"Case{match.group(1).zfill(2)}"

    # Try to extract from source_name: "01_case" → Case01
    match = re.search(r"(\d+)_case", source_name)
    if match:
        return f"Case{match.group(1).zfill(2)}"

    return source_name


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
                raise ImportError(
                    "Migrating Fortran NML files requires f90nml: pip install 'colm-openbench[migration]'"
                )
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
