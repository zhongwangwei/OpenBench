"""YAML configuration loader with validation and _defaults merge.

Public API:
    load_config(path) -> OpenBenchConfig
    ConfigError - raised on validation failure
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from openbench.config.schema import (
    ComparisonConfig,
    DaskConfig,
    EvaluationConfig,
    IOConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
    StatisticsConfig,
    is_simple_project_name,
)
from openbench.util.names import (
    AmbiguousNameError,
    get_mapping_key_case_insensitive,
    normalize_name,
)

logger = logging.getLogger(__name__)

VALID_TIME_ALIGNMENTS = {"intersection", "per_pair", "strict"}
VALID_SIM_DATA_TYPES = {"grid", "stn"}
VALID_PROJECT_WEIGHTS = {"none", "area", "mass"}
VALID_REGRID_BACKENDS = {
    "openbench_conservative",
    "cdo_remapcon",
    "xesmf_conservative",
    "basic_interpolation",
}
VALID_TIM_RES_VALUES = {
    "Month",
    "Day",
    "Hour",
    "Year",
    "3Hour",
    "6Hour",
    "8Day",
    "climatology-month",
    "climatology-year",
}
_MULTI_MONTH_RE = re.compile(r"[1-9]\d*month")


class ConfigError(Exception):
    """Raised when config loading or validation fails."""


class _IncludeLoader(yaml.SafeLoader):
    """YAML loader with !include tag support."""


_INCLUDE_STACK: list[Path] = []
_INCLUDE_ROOTS_STACK: list[tuple[Path, ...]] = []
_INCLUDE_LOCK = threading.RLock()


def _path_is_relative_to(path: Path, root: Path) -> bool:
    """Return whether path is inside root, compatible with py>=3.10."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _include_roots_for_config(config_path: Path) -> tuple[Path, ...]:
    """Allowed roots for !include targets.

    Includes are resolved relative to the including YAML file, but may only
    read files under an explicit trust boundary:
      * the config file's own directory (always);
      * the current project directory when the config file is inside it, so
        ../shared.yaml works for configs in subdirectories;
      * optional extra roots from OPENBENCH_INCLUDE_ROOTS (os.pathsep list).

    This permits intentional ``..`` traversal within the project tree while
    blocking escapes such as ``../../etc/passwd``.
    """
    resolved = config_path.resolve()
    roots = [resolved.parent]

    cwd = Path.cwd().resolve()
    if cwd != cwd.parent and _path_is_relative_to(resolved, cwd):
        roots.append(cwd)

    for entry in os.environ.get("OPENBENCH_INCLUDE_ROOTS", "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry).expanduser().resolve())

    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return tuple(deduped)


def _assert_include_allowed(target: Path) -> Path:
    """Resolve and validate an include target against the active roots."""
    resolved = target.resolve()
    roots = _INCLUDE_ROOTS_STACK[-1] if _INCLUDE_ROOTS_STACK else (Path.cwd().resolve(),)
    if not any(_path_is_relative_to(resolved, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ConfigError(f"!include path is outside allowed roots: {resolved} (allowed roots: {allowed})")
    return resolved


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    """Handle !include tags in YAML files."""
    path_str = loader.construct_scalar(node)
    # Resolve relative to the YAML file currently being parsed.
    if _INCLUDE_STACK:
        base_dir = _INCLUDE_STACK[-1].parent
    elif hasattr(loader, "name"):
        base_dir = Path(loader.name).parent
    else:
        base_dir = Path(".")
    target = base_dir / path_str

    if "*" in path_str:
        # Glob pattern: !include sim/*.yaml
        import glob

        matches = sorted(glob.glob(str(target)))
        if not matches:
            raise ConfigError(f"!include pattern '{path_str}' matched no files (resolved to {target})")

        result = {}
        for match in matches:
            resolved_match = _assert_include_allowed(Path(match))
            if resolved_match in _INCLUDE_STACK:
                chain = " -> ".join(str(p) for p in [*_INCLUDE_STACK, resolved_match])
                raise ConfigError(f"Recursive !include detected: {chain}")
            try:
                # Use _IncludeLoader so nested !include directives resolve;
                # plain yaml.safe_load would raise ConstructorError on
                # encountering !include in an included file.
                _INCLUDE_STACK.append(resolved_match)
                with open(resolved_match) as f:
                    data = yaml.load(f, Loader=_IncludeLoader)
                    if isinstance(data, dict):
                        result.update(data)
            except ConfigError:
                raise
            except Exception as e:
                raise ConfigError(f"!include failed to read '{match}': {e}") from e
            finally:
                if _INCLUDE_STACK and _INCLUDE_STACK[-1] == resolved_match:
                    _INCLUDE_STACK.pop()
        return result
    else:
        # Single file: !include ref.yaml
        if not target.exists():
            raise ConfigError(f"!include file not found: '{path_str}' (resolved to {target})")
        resolved_target = _assert_include_allowed(target)
        if resolved_target in _INCLUDE_STACK:
            chain = " -> ".join(str(p) for p in [*_INCLUDE_STACK, resolved_target])
            raise ConfigError(f"Recursive !include detected: {chain}")
        try:
            _INCLUDE_STACK.append(resolved_target)
            with open(resolved_target) as f:
                # Same recursive-include-aware loader as the multi-file branch.
                return yaml.load(f, Loader=_IncludeLoader)
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"!include failed to read '{target}': {e}") from e
        finally:
            if _INCLUDE_STACK and _INCLUDE_STACK[-1] == resolved_target:
                _INCLUDE_STACK.pop()


_IncludeLoader.add_constructor("!include", _include_constructor)


def _merge_variable_defaults(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge simulation.variables entries by variable name."""
    merged: dict[str, Any] = dict(defaults or {})

    for var_name, override_value in (overrides or {}).items():
        try:
            merged_key = get_mapping_key_case_insensitive(merged, var_name)
        except AmbiguousNameError as exc:
            raise ConfigError(f"simulation variable defaults are ambiguous: {exc}") from exc
        target_key = merged_key or var_name
        default_value = merged.get(target_key)
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            merged[target_key] = {**default_value, **override_value}
        else:
            merged[target_key] = override_value

    return merged


def _evaluation_variable_lookup(variables: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for var in variables:
        if not isinstance(var, str) or not var.strip():
            raise ConfigError("evaluation.variables entries must be non-empty strings")
        key = normalize_name(var)
        if key in lookup and lookup[key] != var:
            raise ConfigError(f"evaluation.variables contains duplicate names ignoring case: {lookup[key]}, {var}")
        lookup[key] = var
    return lookup


def _canonical_evaluation_variable(name: str, lookup: dict[str, str]) -> str | None:
    return lookup.get(normalize_name(name))


def _canonicalize_variable_mapping_keys(
    raw: dict[str, Any] | None,
    lookup: dict[str, str],
    *,
    path: str,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    result: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _canonical_evaluation_variable(str(key), lookup) or key
        existing_key = get_mapping_key_case_insensitive(result, canonical)
        if existing_key is not None:
            raise ConfigError(f"{path} contains duplicate variable keys ignoring case: {existing_key}, {key}")
        result[canonical] = value
    return result


def _validated_numeric_pair(
    raw: Any,
    field_name: str,
    default: list[float],
    *,
    lower: float,
    upper: float,
) -> list[float]:
    """Validate a two-number inclusive range from project config."""
    value = default if raw is None else raw
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"project.{field_name} must be a list of [min, max]")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ConfigError(f"project.{field_name} values must be numeric")
    if value[0] > value[1]:
        raise ConfigError(f"project.{field_name} minimum ({value[0]}) must be <= maximum ({value[1]})")
    if value[0] < lower or value[1] > upper:
        raise ConfigError(f"project.{field_name} must be within [{lower}, {upper}]")
    return list(value)


def _validated_optional_bool(raw: Any, path: str, *, default: bool | None = None) -> bool | None:
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise ConfigError(f"{path} must be a boolean")
    return raw


def _validated_optional_nonnegative_int(raw: Any, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{path} must be an integer >= 0 or null")
    if raw < 0:
        raise ConfigError(f"{path} must be >= 0, got {raw}")
    return raw


def _validated_optional_positive_int(raw: Any, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{path} must be an integer > 0 or null")
    if raw <= 0:
        raise ConfigError(f"{path} must be > 0, got {raw}")
    return raw


def _validated_optional_float_range(raw: Any, path: str, *, lower: float, upper: float) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{path} must be a number between {lower} and {upper}")
    value = float(raw)
    if value < lower or value > upper:
        raise ConfigError(f"{path} must be between {lower} and {upper}, got {raw}")
    return value


def _validated_compression_level(raw: Any, path: str) -> int:
    if raw is None:
        return 1
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{path} must be an integer between 0 and 9")
    if raw < 0 or raw > 9:
        raise ConfigError(f"{path} must be between 0 and 9, got {raw}")
    return raw


def _validated_mfdataset_batch_size(raw: Any, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{path} must be an integer >= 0, 'auto', or null")
    if raw < 0:
        raise ConfigError(f"{path} must be >= 0, got {raw}")
    return raw


def _build_io(raw: Any) -> IOConfig:
    if raw is None or raw == {}:
        return IOConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"project.io must be a mapping, got {type(raw).__name__}")

    netcdf_compression = _validated_optional_bool(
        raw.get("netcdf_compression", raw.get("compression")),
        "project.io.netcdf_compression",
        default=False,
    )
    return IOConfig(
        netcdf_compression=bool(netcdf_compression),
        netcdf_compression_level=_validated_compression_level(
            raw.get("netcdf_compression_level", raw.get("compression_level")),
            "project.io.netcdf_compression_level",
        ),
        mfdataset_batch_size=_validated_mfdataset_batch_size(
            raw.get("mfdataset_batch_size"),
            "project.io.mfdataset_batch_size",
        ),
        mfdataset_auto_batch_min_files=_validated_optional_positive_int(
            raw.get("mfdataset_auto_batch_min_files"),
            "project.io.mfdataset_auto_batch_min_files",
        ),
        mfdataset_auto_batch_min_size_mb=_validated_optional_nonnegative_int(
            raw.get("mfdataset_auto_batch_min_size_mb"),
            "project.io.mfdataset_auto_batch_min_size_mb",
        ),
        mfdataset_auto_batch_min_size=_validated_optional_positive_int(
            raw.get("mfdataset_auto_batch_min_size"),
            "project.io.mfdataset_auto_batch_min_size",
        ),
        mfdataset_auto_batch_max_size=_validated_optional_positive_int(
            raw.get("mfdataset_auto_batch_max_size"),
            "project.io.mfdataset_auto_batch_max_size",
        ),
        mfdataset_auto_batch_memory_fraction=_validated_optional_float_range(
            raw.get("mfdataset_auto_batch_memory_fraction"),
            "project.io.mfdataset_auto_batch_memory_fraction",
            lower=0.01,
            upper=1.0,
        ),
    )


def _build_dask(raw: Any) -> DaskConfig:
    if raw is None or raw == {}:
        return DaskConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"project.dask must be a mapping, got {type(raw).__name__}")

    enabled = _validated_optional_bool(raw.get("enabled"), "project.dask.enabled", default=False)
    scheduler = _validated_optional_string(raw.get("scheduler"), "project.dask.scheduler")
    n_workers = _validated_optional_nonnegative_int(
        raw.get("n_workers", raw.get("workers")),
        "project.dask.n_workers",
    )
    threads_per_worker = _validated_optional_nonnegative_int(
        raw.get("threads_per_worker"),
        "project.dask.threads_per_worker",
    )
    processes = _validated_optional_bool(raw.get("processes"), "project.dask.processes", default=True)
    memory_limit = _validated_optional_string(raw.get("memory_limit", "auto"), "project.dask.memory_limit") or "auto"
    dashboard_address = _validated_optional_string(
        raw.get("dashboard_address"),
        "project.dask.dashboard_address",
    )
    local_directory = _validated_optional_string(raw.get("local_directory"), "project.dask.local_directory")

    return DaskConfig(
        enabled=bool(enabled),
        scheduler=scheduler,
        n_workers=n_workers,
        threads_per_worker=1 if threads_per_worker is None else threads_per_worker,
        processes=True if processes is None else processes,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        local_directory=local_directory,
    )


def _validated_variables_mapping(raw: Any, path: str) -> dict[str, dict[str, Any]] | None:
    """Validate simulation variable override mappings."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping of variable name to override mapping")
    for var_name, override in raw.items():
        if not isinstance(var_name, str):
            raise ConfigError(f"{path} keys must be variable names (strings)")
        if not isinstance(override, dict):
            raise ConfigError(f"{path}.{var_name} must be a mapping")
    return raw


def _validated_required_string(raw: Any, path: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return raw


def _validated_optional_string(raw: Any, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ConfigError(f"{path} must be a string")
    return raw


def _validated_optional_string_list(raw: Any, path: str) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ConfigError(f"{path} must be a list of strings")
    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            raise ConfigError(f"{path}[{idx}] must be a string")
    return list(raw)


def _validated_optional_choice(raw: Any, path: str, choices: set[str]) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ConfigError(f"{path} must be a string")
    if raw.lower() not in choices:
        valid = ", ".join(sorted(choices))
        raise ConfigError(f"{path} must be one of: {valid}")
    return raw.lower()


def _normalized_optional_lower_choice(raw: Any, path: str, choices: set[str]) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.lower() in choices:
        return raw.lower()
    return raw


_TIM_RES_CANONICAL = {v.lower(): v for v in VALID_TIM_RES_VALUES}


def _validated_optional_tim_res(raw: Any, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ConfigError(f"{path} must be a string")
    # Match `_validated_optional_choice` behaviour: accept any case but
    # return the canonical Title-case form. Otherwise users would have
    # `data_type: GRID` accepted but `tim_res: month` rejected, which is
    # arbitrary and surprising.
    canonical = _TIM_RES_CANONICAL.get(raw.lower())
    if canonical is None and not _MULTI_MONTH_RE.fullmatch(raw):
        valid = ", ".join([*sorted(VALID_TIM_RES_VALUES), "Nmonth"])
        raise ConfigError(f"{path} must be a supported time resolution: {valid}")
    return canonical or raw


def _validated_optional_positive_number(raw: Any, path: str) -> float | int | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ConfigError(f"{path} must be a positive number")
    if raw <= 0:
        raise ConfigError(f"{path} must be a positive number")
    return raw


def _validated_optional_number(raw: Any, path: str) -> float | int | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ConfigError(f"{path} must be a number")
    return raw


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

    resolved_path = path.resolve()
    include_roots = _include_roots_for_config(resolved_path)
    _INCLUDE_LOCK.acquire()
    try:
        _INCLUDE_STACK.append(resolved_path)
        _INCLUDE_ROOTS_STACK.append(include_roots)
        with open(resolved_path) as f:
            raw = yaml.load(f, Loader=_IncludeLoader)
    except OSError as e:
        raise ConfigError(f"Failed to read config file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e
    finally:
        if _INCLUDE_STACK and _INCLUDE_STACK[-1] == resolved_path:
            _INCLUDE_STACK.pop()
        if _INCLUDE_ROOTS_STACK and _INCLUDE_ROOTS_STACK[-1] == include_roots:
            _INCLUDE_ROOTS_STACK.pop()
        _INCLUDE_LOCK.release()

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> OpenBenchConfig:
    """Build and validate an OpenBenchConfig from a raw dict."""
    # --- project (required) ---
    if "project" not in raw:
        raise ConfigError("Missing required section: 'project'")
    if not isinstance(raw["project"], dict):
        raise ConfigError("'project' must be a mapping")

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
    migrated_comp = [
        k for k in ("tim_res", "grid_res", "timezone", "weight") if k in raw_comparison and k not in raw_project
    ]
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
    if not isinstance(raw["evaluation"], dict):
        raise ConfigError("'evaluation' must be a mapping")
    evaluation = _build_evaluation(raw["evaluation"])
    evaluation_lookup = _evaluation_variable_lookup(evaluation.variables)

    # --- reference (required) ---
    if "reference" not in raw:
        raise ConfigError("Missing required section: 'reference'")
    if not isinstance(raw["reference"], dict):
        raise ConfigError("'reference' must be a mapping")
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
    canonical_reference = (
        _canonicalize_variable_mapping_keys(
            {k: v for k, v in raw_reference.items() if k != "data_root"},
            evaluation_lookup,
            path="reference",
        )
        or {}
    )
    raw_reference = {"data_root": raw_reference.get("data_root"), **canonical_reference}
    if raw_reference.get("data_root") is None:
        raw_reference.pop("data_root", None)
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
    if not isinstance(raw["simulation"], dict):
        raise ConfigError("'simulation' must be a mapping")
    simulation = _build_simulation(raw["simulation"])
    for label, entry in simulation.items():
        entry.variables = _canonicalize_variable_mapping_keys(
            entry.variables,
            evaluation_lookup,
            path=f"simulation.{label}.variables",
        )

    # --- optional sections ---
    metrics = _validated_optional_string_list(raw.get("metrics"), "metrics")
    scores = _validated_optional_string_list(raw.get("scores"), "scores")
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

    name = raw["name"]
    if not is_simple_project_name(name):
        raise ConfigError("project.name must be a simple directory name, not a path.")

    years = raw["years"]
    if not isinstance(years, list) or len(years) != 2:
        raise ConfigError("project.years must be a list of [start_year, end_year]")
    if not all(isinstance(y, int) and not isinstance(y, bool) for y in years):
        raise ConfigError(f"project.years must be integers, got {[type(y).__name__ for y in years]}")
    if years[0] > years[1]:
        raise ConfigError(f"project.years start year ({years[0]}) must be <= end year ({years[1]})")

    time_alignment = raw.get("time_alignment", "intersection")
    if time_alignment not in VALID_TIME_ALIGNMENTS:
        raise ConfigError(f"project.time_alignment must be one of {VALID_TIME_ALIGNMENTS}, got '{time_alignment}'")

    num_cores = raw.get("num_cores")
    if num_cores is not None:
        if isinstance(num_cores, bool) or not isinstance(num_cores, int):
            raise ConfigError(f"project.num_cores must be an integer >= 0 or null, got {type(num_cores).__name__}")
        if num_cores < 0:
            raise ConfigError(f"project.num_cores must be >= 0, got {num_cores}")

    lat_range = _validated_numeric_pair(
        raw.get("lat_range"),
        "lat_range",
        [-90.0, 90.0],
        lower=-90.0,
        upper=90.0,
    )
    lon_range = _validated_numeric_pair(
        raw.get("lon_range"),
        "lon_range",
        [-180.0, 180.0],
        lower=-180.0,
        upper=180.0,
    )

    return ProjectConfig(
        name=str(name),
        # Apply $VAR + ~ expansion (matches adapter._resolve_root_relative_path).
        # Without expandvars, `output_dir: $SCRATCH/results` would be taken
        # literally and fail at directory creation time on HPC.
        output_dir=str(Path(os.path.expandvars(str(raw["output_dir"]))).expanduser()),
        years=years,
        min_year_threshold=raw.get("min_year_threshold", 3),
        lat_range=lat_range,
        lon_range=lon_range,
        # Target resolution
        tim_res=_validated_optional_tim_res(raw.get("tim_res"), "project.tim_res"),
        grid_res=_validated_optional_positive_number(raw.get("grid_res"), "project.grid_res"),
        timezone=_validated_optional_number(raw.get("timezone"), "project.timezone"),
        weight=_validated_optional_choice(raw.get("weight"), "project.weight", VALID_PROJECT_WEIGHTS),
        # Runtime
        num_cores=num_cores,
        time_alignment=time_alignment,
        regrid_backend=_validated_optional_choice(
            raw.get("regrid_backend"),
            "project.regrid_backend",
            VALID_REGRID_BACKENDS,
        )
        or "openbench_conservative",
        unified_mask=_validated_optional_bool(raw.get("unified_mask"), "project.unified_mask", default=True),
        generate_report=_validated_optional_bool(raw.get("generate_report"), "project.generate_report", default=True),
        # Groupby
        IGBP_groupby=_validated_optional_bool(raw.get("IGBP_groupby"), "project.IGBP_groupby", default=False),
        PFT_groupby=_validated_optional_bool(raw.get("PFT_groupby"), "project.PFT_groupby", default=False),
        climate_zone_groupby=_validated_optional_bool(
            raw.get("climate_zone_groupby"), "project.climate_zone_groupby", default=False
        ),
        # Advanced
        debug_mode=_validated_optional_bool(raw.get("debug_mode"), "project.debug_mode", default=False),
        only_drawing=_validated_optional_bool(raw.get("only_drawing"), "project.only_drawing", default=False),
        force=_validated_optional_bool(raw.get("force"), "project.force", default=False),
        strict_reference=_validated_optional_bool(
            raw.get("strict_reference"), "project.strict_reference", default=False
        ),
        dask=_build_dask(raw.get("dask")),
        io=_build_io(raw.get("io")),
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
    if data_root is not None:
        if not isinstance(data_root, str):
            raise ConfigError(f"reference.data_root must be a string, got {type(data_root).__name__}")
        data_root = data_root.strip()
        if not data_root:
            raise ConfigError("reference.data_root must not be empty")

    # Remaining keys are variable → source mappings (str or list[str])
    sources: dict[str, str | list[str]] = {}
    for key, value in raw_copy.items():
        if isinstance(value, str):
            # Auto-split comma-separated strings (legacy NML form)
            if "," in value:
                parts = [s.strip() for s in value.split(",") if s.strip()]
                if not parts:
                    raise ConfigError(f"reference.{key} must include at least one source name")
                sources[key] = parts if len(parts) > 1 else parts[0]
            else:
                source = value.strip()
                if not source:
                    raise ConfigError(f"reference.{key} must not be empty")
                sources[key] = source
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise ConfigError(f"reference.{key}[{i}] must be a string (source name), got {type(item).__name__}")
                if not item.strip():
                    raise ConfigError(f"reference.{key}[{i}] must not be empty")
            if not value:
                raise ConfigError(f"reference.{key} must not be an empty list")
            # Single-item list collapses to plain string for downstream simplicity
            sources[key] = value[0].strip() if len(value) == 1 else [item.strip() for item in value]
        else:
            raise ConfigError(
                f"reference.{key} must be a string or list of strings (source name), got {type(value).__name__}"
            )

    return ReferenceConfig(data_root=data_root, sources=sources)


def _build_simulation(raw: dict[str, Any]) -> dict[str, SimulationEntry]:
    """Build simulation entries with _defaults merge."""
    if not isinstance(raw, dict):
        raise ConfigError("'simulation' must be a mapping")

    # Extract and remove _defaults before iterating
    raw_copy = dict(raw)
    defaults = raw_copy.pop("_defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ConfigError(f"simulation._defaults must be a mapping, got {type(defaults).__name__}")
    if not raw_copy:
        raise ConfigError("'simulation' must contain at least one simulation entry (not just _defaults)")
    result = {}

    for label, entry in raw_copy.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"simulation.{label} must be a mapping")

        default_variables = _validated_variables_mapping(
            defaults.get("variables"),
            "simulation._defaults.variables",
        )
        entry_variables = _validated_variables_mapping(
            entry.get("variables"),
            f"simulation.{label}.variables",
        )

        # Merge defaults: defaults are base, entry overrides
        merged = {**defaults, **entry}

        # For 'variables', merge per variable so partial overrides keep defaults
        if "variables" in defaults or "variables" in entry:
            merged["variables"] = _merge_variable_defaults(
                default_variables,
                entry_variables,
            )

        if "model" not in merged:
            raise ConfigError(f"simulation.{label} must have a 'model' field")
        if "root_dir" not in merged:
            raise ConfigError(f"simulation.{label} must have a 'root_dir' field")

        model = _validated_required_string(merged["model"], f"simulation.{label}.model")
        root_dir = _validated_required_string(merged["root_dir"], f"simulation.{label}.root_dir")
        data_type = _validated_optional_choice(
            merged.get("data_type"),
            f"simulation.{label}.data_type",
            VALID_SIM_DATA_TYPES,
        )
        grid_res = _validated_optional_positive_number(
            merged.get("grid_res"),
            f"simulation.{label}.grid_res",
        )
        tim_res = _validated_optional_tim_res(
            merged.get("tim_res"),
            f"simulation.{label}.tim_res",
        )
        # data_groupby is validated by `cli/check.py:_data_groupby_error`
        # so check can report bad values alongside other config errors in
        # one pass. We keep the loader-level type check here but defer
        # value-set validation so the multi-error preflight UX works.
        data_groupby = _validated_optional_string(
            merged.get("data_groupby"),
            f"simulation.{label}.data_groupby",
        )
        prefix = _validated_optional_string(merged.get("prefix"), f"simulation.{label}.prefix")
        suffix = _validated_optional_string(merged.get("suffix"), f"simulation.{label}.suffix")
        fulllist = _validated_optional_string(merged.get("fulllist"), f"simulation.{label}.fulllist")

        result[label] = SimulationEntry(
            model=model,
            root_dir=root_dir,
            data_type=data_type,
            grid_res=grid_res,
            tim_res=tim_res,
            data_groupby=data_groupby,
            prefix=prefix,
            suffix=suffix,
            fulllist=fulllist,
            variables=merged.get("variables"),
        )

    return result


def _build_comparison(raw: Any) -> ComparisonConfig:
    """Build ComparisonConfig — only enabled + items."""
    if raw is None or raw == {}:
        return ComparisonConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"'comparison' must be a mapping, got {type(raw).__name__}")
    return ComparisonConfig(
        enabled=_validated_optional_bool(raw.get("enabled"), "comparison.enabled", default=False),
        items=_validated_optional_string_list(raw.get("items"), "comparison.items"),
    )


def _build_statistics(raw: Any) -> StatisticsConfig:
    """Build StatisticsConfig from raw dict or None."""
    if raw is None or raw == {}:
        return StatisticsConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"'statistics' must be a mapping, got {type(raw).__name__}")
    return StatisticsConfig(
        enabled=_validated_optional_bool(raw.get("enabled"), "statistics.enabled", default=False),
        items=_validated_optional_string_list(raw.get("items"), "statistics.items"),
    )
