"""YAML configuration loader with validation and _defaults merge.

Public API:
    load_config(path) -> OpenBenchConfig
    ConfigError - raised on validation failure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
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

        result = {}
        for match in sorted(glob.glob(str(target))):
            with open(match) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    result.update(data)
        return result
    else:
        # Single file: !include ref.yaml
        with open(target) as f:
            return yaml.safe_load(f)


_IncludeLoader.add_constructor("!include", _include_constructor)


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
    project = _build_project(raw["project"])

    # --- evaluation (required) ---
    if "evaluation" not in raw:
        raise ConfigError("Missing required section: 'evaluation'")
    evaluation = _build_evaluation(raw["evaluation"])

    # --- reference (required) ---
    if "reference" not in raw:
        raise ConfigError("Missing required section: 'reference'")
    reference = raw["reference"]
    if not isinstance(reference, dict):
        raise ConfigError("'reference' must be a mapping of variable -> source name")

    # Check all evaluation variables have a reference
    for var in evaluation.variables:
        if var not in reference:
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
    options = _build_options(raw.get("options", {}))

    return OpenBenchConfig(
        project=project,
        evaluation=evaluation,
        reference=reference,
        simulation=simulation,
        metrics=metrics,
        scores=scores,
        comparison=comparison,
        statistics=statistics,
        options=options,
    )


def _build_project(raw: dict[str, Any]) -> ProjectConfig:
    """Build and validate ProjectConfig."""
    required = ["name", "output_dir", "years"]
    for key in required:
        if key not in raw:
            raise ConfigError(f"Missing required field: project.{key}")

    years = raw["years"]
    if not isinstance(years, list) or len(years) != 2:
        raise ConfigError("project.years must be a list of [start_year, end_year]")
    if years[0] > years[1]:
        raise ConfigError(f"project.years start year ({years[0]}) must be <= end year ({years[1]})")

    return ProjectConfig(
        name=raw["name"],
        output_dir=raw["output_dir"],
        years=years,
        min_year_threshold=raw.get("min_year_threshold", 3),
        lat_range=raw.get("lat_range", [-90.0, 90.0]),
        lon_range=raw.get("lon_range", [-180.0, 180.0]),
    )


def _build_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    """Build and validate EvaluationConfig."""
    if "variables" not in raw:
        raise ConfigError("Missing required field: evaluation.variables")
    variables = raw["variables"]
    if not isinstance(variables, list) or len(variables) == 0:
        raise ConfigError("evaluation.variables must be a non-empty list")
    return EvaluationConfig(variables=variables)


def _build_simulation(raw: dict[str, Any]) -> dict[str, SimulationEntry]:
    """Build simulation entries with _defaults merge."""
    if not isinstance(raw, dict):
        raise ConfigError("'simulation' must be a mapping")

    # Extract and remove _defaults before iterating
    raw_copy = dict(raw)
    defaults = raw_copy.pop("_defaults", {})
    result = {}

    for label, entry in raw_copy.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"simulation.{label} must be a mapping")

        # Merge defaults: defaults are base, entry overrides
        merged = {**defaults, **entry}

        # For 'variables', do shallow-per-variable merge
        if "variables" in defaults and "variables" in entry:
            merged_vars = {**defaults.get("variables", {}), **entry["variables"]}
            merged["variables"] = merged_vars

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
            variables=merged.get("variables"),
        )

    return result


def _build_comparison(raw: Any) -> ComparisonConfig:
    """Build ComparisonConfig from raw dict or None."""
    if not raw or not isinstance(raw, dict):
        return ComparisonConfig()
    return ComparisonConfig(
        enabled=raw.get("enabled", False),
        items=raw.get("items"),
        weight=raw.get("weight"),
        tim_res=raw.get("tim_res"),
        timezone=raw.get("timezone"),
        grid_res=raw.get("grid_res"),
    )


def _build_statistics(raw: Any) -> StatisticsConfig:
    """Build StatisticsConfig from raw dict or None."""
    if not raw or not isinstance(raw, dict):
        return StatisticsConfig()
    return StatisticsConfig(
        enabled=raw.get("enabled", False),
        items=raw.get("items"),
    )


def _build_options(raw: Any) -> OptionsConfig:
    """Build and validate OptionsConfig."""
    if not raw or not isinstance(raw, dict):
        return OptionsConfig()

    time_alignment = raw.get("time_alignment", "intersection")
    if time_alignment not in VALID_TIME_ALIGNMENTS:
        raise ConfigError(f"options.time_alignment must be one of {VALID_TIME_ALIGNMENTS}, got '{time_alignment}'")

    return OptionsConfig(
        num_cores=raw.get("num_cores"),
        time_alignment=time_alignment,
        unified_mask=raw.get("unified_mask", True),
        generate_report=raw.get("generate_report", True),
        IGBP_groupby=raw.get("IGBP_groupby", False),
        PFT_groupby=raw.get("PFT_groupby", False),
        climate_zone_groupby=raw.get("climate_zone_groupby", False),
        debug_mode=raw.get("debug_mode", False),
        only_drawing=raw.get("only_drawing", False),
    )
