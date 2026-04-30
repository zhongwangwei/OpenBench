"""YAML configuration loader with validation and _defaults merge.

Public API:
    load_config(path) -> OpenBenchConfig
    ConfigError - raised on validation failure
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
    StatisticsConfig,
)

VALID_TIME_ALIGNMENTS = {"intersection", "per_pair", "strict"}


class ConfigError(Exception):
    """Raised when config loading or validation fails."""


class _IncludeLoader(yaml.SafeLoader):
    """YAML loader with !include tag support."""


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    """Handle !include tags in YAML files."""
    path_str = loader.construct_scalar(node)
    # Resolve relative to the YAML file's directory
    base_dir = Path(loader.name).parent if hasattr(loader, "name") else Path(".")
    target = base_dir / path_str

    if "*" in path_str:
        # Glob pattern: !include sim/*.yaml
        import glob

        matches = sorted(glob.glob(str(target)))
        if not matches:
            raise ConfigError(f"!include pattern '{path_str}' matched no files (resolved to {target})")

        result = {}
        for match in matches:
            try:
                # Use _IncludeLoader so nested !include directives resolve;
                # plain yaml.safe_load would raise ConstructorError on
                # encountering !include in an included file.
                with open(match) as f:
                    data = yaml.load(f, Loader=_IncludeLoader)
                    if isinstance(data, dict):
                        result.update(data)
            except Exception as e:
                raise ConfigError(f"!include failed to read '{match}': {e}") from e
        return result
    else:
        # Single file: !include ref.yaml
        if not target.exists():
            raise ConfigError(f"!include file not found: '{path_str}' (resolved to {target})")
        try:
            with open(target) as f:
                # Same recursive-include-aware loader as the multi-file branch.
                return yaml.load(f, Loader=_IncludeLoader)
        except Exception as e:
            raise ConfigError(f"!include failed to read '{target}': {e}") from e


_IncludeLoader.add_constructor("!include", _include_constructor)


def _merge_variable_defaults(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge simulation.variables entries by variable name."""
    merged: dict[str, Any] = dict(defaults or {})

    for var_name, override_value in (overrides or {}).items():
        default_value = merged.get(var_name)
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            merged[var_name] = {**default_value, **override_value}
        else:
            merged[var_name] = override_value

    return merged


def load_config(path: str | Path) -> OpenBenchConfig:
    """Load and validate an openbench.yaml config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated OpenBenchConfig instance.

    Raises:
        ConfigError: If the file is invalid or validation fails.
    """
    path = Path(path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ConfigError(
            f"Only YAML config files are supported (got {path.suffix}). "
            "Use 'openbench migrate' to convert old JSON/NML configs."
        )

    try:
        with open(path) as f:
            raw = yaml.load(f, Loader=_IncludeLoader)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> OpenBenchConfig:
    """Build and validate an OpenBenchConfig from a raw dict."""
    # --- project (required) ---
    if "project" not in raw:
        raise ConfigError("Missing required section: 'project'")

    # Backward compatibility: merge old 'options' section into project
    raw_project = dict(raw["project"])
    if "options" in raw:
        migrated_keys = [k for k in raw["options"] if k not in raw_project]
        if migrated_keys:
            logger.warning(
                "Deprecated 'options' section detected — migrating keys %s into 'project'. "
                "Please move them to 'project' directly or run 'openbench migrate'.",
                migrated_keys,
            )
        for key, value in raw["options"].items():
            if key not in raw_project:
                raw_project[key] = value

    # Backward compatibility: merge old comparison resolution fields into project
    raw_comparison = raw.get("comparison", {}) or {}
    migrated_comp = [k for k in ("tim_res", "grid_res", "timezone", "weight")
                     if k in raw_comparison and k not in raw_project]
    if migrated_comp:
        logger.warning(
            "Deprecated comparison-level keys %s detected — migrating into 'project'. "
            "Please move them to 'project' directly or run 'openbench migrate'.",
            migrated_comp,
        )
    for key in ("tim_res", "grid_res", "timezone", "weight"):
        if key in raw_comparison and key not in raw_project:
            raw_project[key] = raw_comparison[key]

    project = _build_project(raw_project)

    # --- evaluation (required) ---
    if "evaluation" not in raw:
        raise ConfigError("Missing required section: 'evaluation'")
    evaluation = _build_evaluation(raw["evaluation"])

    # --- reference (required) ---
    if "reference" not in raw:
        raise ConfigError("Missing required section: 'reference'")
    raw_reference = dict(raw["reference"])
    # Backward compatibility: old options.data_root → reference.data_root
    if "data_root" not in raw_reference:
        old_data_root = (raw.get("options") or {}).get("data_root")
        if old_data_root:
            logger.warning(
                "Deprecated 'options.data_root' detected — migrating to 'reference.data_root'. "
                "Please move it to 'reference.data_root' directly or run 'openbench migrate'.",
            )
            raw_reference["data_root"] = old_data_root
    reference = _build_reference(raw_reference)

    # Check all evaluation variables have a reference
    for var in evaluation.variables:
        if var not in reference.sources:
            raise ConfigError(
                f"Variable '{var}' is in evaluation.variables but has no entry in 'reference'. "
                f"Add: reference.{var}: <source_name>"
            )

    # --- simulation (required) ---
    if "simulation" not in raw:
        raise ConfigError("Missing required section: 'simulation'")
    simulation = _build_simulation(raw["simulation"])

    # --- optional sections ---
    metrics = raw.get("metrics")
    scores = raw.get("scores")
    comparison = _build_comparison(raw.get("comparison", {}))
    statistics = _build_statistics(raw.get("statistics", {}))

    return OpenBenchConfig(
        project=project,
        evaluation=evaluation,
        reference=reference,
        simulation=simulation,
        metrics=metrics,
        scores=scores,
        comparison=comparison,
        statistics=statistics,
    )


def _build_project(raw: dict[str, Any]) -> ProjectConfig:
    """Build and validate ProjectConfig (includes former options fields)."""
    required = ["name", "output_dir", "years"]
    for key in required:
        if key not in raw:
            raise ConfigError(f"Missing required field: project.{key}")

    years = raw["years"]
    if not isinstance(years, list) or len(years) != 2:
        raise ConfigError("project.years must be a list of [start_year, end_year]")
    if not all(isinstance(y, int) for y in years):
        raise ConfigError(f"project.years must be integers, got {[type(y).__name__ for y in years]}")
    if years[0] > years[1]:
        raise ConfigError(f"project.years start year ({years[0]}) must be <= end year ({years[1]})")

    time_alignment = raw.get("time_alignment", "intersection")
    if time_alignment not in VALID_TIME_ALIGNMENTS:
        raise ConfigError(f"project.time_alignment must be one of {VALID_TIME_ALIGNMENTS}, got '{time_alignment}'")

    return ProjectConfig(
        name=raw["name"],
        output_dir=str(Path(raw["output_dir"]).expanduser()),
        years=years,
        min_year_threshold=raw.get("min_year_threshold", 3),
        lat_range=raw.get("lat_range", [-90.0, 90.0]),
        lon_range=raw.get("lon_range", [-180.0, 180.0]),
        # Target resolution
        tim_res=raw.get("tim_res"),
        grid_res=raw.get("grid_res"),
        timezone=raw.get("timezone"),
        weight=raw.get("weight"),
        # Runtime
        num_cores=raw.get("num_cores"),
        time_alignment=time_alignment,
        unified_mask=raw.get("unified_mask", True),
        generate_report=raw.get("generate_report", True),
        # Groupby
        IGBP_groupby=raw.get("IGBP_groupby", False),
        PFT_groupby=raw.get("PFT_groupby", False),
        climate_zone_groupby=raw.get("climate_zone_groupby", False),
        # Advanced
        debug_mode=raw.get("debug_mode", False),
        only_drawing=raw.get("only_drawing", False),
        force=raw.get("force", False),
        strict_reference=raw.get("strict_reference", False),
    )


def _build_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    """Build and validate EvaluationConfig."""
    if "variables" not in raw:
        raise ConfigError("Missing required field: evaluation.variables")
    variables = raw["variables"]
    if not isinstance(variables, list) or len(variables) == 0:
        raise ConfigError("evaluation.variables must be a non-empty list")
    return EvaluationConfig(variables=variables)


def _build_reference(raw: Any) -> ReferenceConfig:
    """Build ReferenceConfig: extract data_root, treat rest as variable→source mappings.

    Each variable's value may be a single source string OR a list of source
    strings (multi-reference comparison, matching v2.x behavior). Comma-
    separated strings (from old Fortran namelist migration) are auto-split.
    """
    if not isinstance(raw, dict):
        raise ConfigError("'reference' must be a mapping")

    raw_copy = dict(raw)
    data_root = raw_copy.pop("data_root", None)

    # Remaining keys are variable → source mappings (str or list[str])
    sources: dict[str, str | list[str]] = {}
    for key, value in raw_copy.items():
        if isinstance(value, str):
            # Auto-split comma-separated strings (legacy NML form)
            if "," in value:
                parts = [s.strip() for s in value.split(",") if s.strip()]
                sources[key] = parts if len(parts) > 1 else (parts[0] if parts else "")
            else:
                sources[key] = value
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise ConfigError(
                        f"reference.{key}[{i}] must be a string (source name), "
                        f"got {type(item).__name__}"
                    )
            if not value:
                raise ConfigError(f"reference.{key} must not be an empty list")
            # Single-item list collapses to plain string for downstream simplicity
            sources[key] = value[0] if len(value) == 1 else list(value)
        else:
            raise ConfigError(
                f"reference.{key} must be a string or list of strings (source name), "
                f"got {type(value).__name__}"
            )

    return ReferenceConfig(data_root=data_root, sources=sources)


def _build_simulation(raw: dict[str, Any]) -> dict[str, SimulationEntry]:
    """Build simulation entries with _defaults merge."""
    if not isinstance(raw, dict):
        raise ConfigError("'simulation' must be a mapping")

    # Extract and remove _defaults before iterating
    raw_copy = dict(raw)
    defaults = raw_copy.pop("_defaults", {})
    if not raw_copy:
        raise ConfigError("'simulation' must contain at least one simulation entry (not just _defaults)")
    result = {}

    for label, entry in raw_copy.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"simulation.{label} must be a mapping")

        # Merge defaults: defaults are base, entry overrides
        merged = {**defaults, **entry}

        # For 'variables', merge per variable so partial overrides keep defaults
        if "variables" in defaults or "variables" in entry:
            merged["variables"] = _merge_variable_defaults(
                defaults.get("variables"),
                entry.get("variables"),
            )

        if "model" not in merged:
            raise ConfigError(f"simulation.{label} must have a 'model' field")
        if "root_dir" not in merged:
            raise ConfigError(f"simulation.{label} must have a 'root_dir' field")

        result[label] = SimulationEntry(
            model=merged["model"],
            root_dir=merged["root_dir"],
            data_type=merged.get("data_type"),
            grid_res=merged.get("grid_res"),
            tim_res=merged.get("tim_res"),
            data_groupby=merged.get("data_groupby"),
            prefix=merged.get("prefix"),
            suffix=merged.get("suffix"),
            fulllist=merged.get("fulllist"),
            variables=merged.get("variables"),
        )

    return result


def _build_comparison(raw: Any) -> ComparisonConfig:
    """Build ComparisonConfig — only enabled + items."""
    if not raw or not isinstance(raw, dict):
        return ComparisonConfig()
    return ComparisonConfig(
        enabled=raw.get("enabled", False),
        items=raw.get("items"),
    )


def _build_statistics(raw: Any) -> StatisticsConfig:
    """Build StatisticsConfig from raw dict or None."""
    if not raw or not isinstance(raw, dict):
        return StatisticsConfig()
    return StatisticsConfig(
        enabled=raw.get("enabled", False),
        items=raw.get("items"),
    )
