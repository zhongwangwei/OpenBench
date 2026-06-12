"""OpenBench configuration schema defined as dataclasses.

Each section of openbench.yaml maps to a dataclass with typed fields
and sensible defaults. The top-level OpenBenchConfig holds everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Optional


def is_simple_project_name(name: object) -> bool:
    """Return True when *name* is a directory name, not a path."""
    raw = str(name)
    text = raw.strip()
    if not text or text in {".", ".."}:
        return False
    if raw != text or any(ch.isspace() for ch in text):
        return False
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and len(posix.parts) == 1
        and len(windows.parts) == 1
        and posix.name == text
        and windows.name == text
    )


@dataclass
class DaskConfig:
    """Optional dask.distributed runtime configuration."""

    enabled: bool = False
    scheduler: Optional[str] = None
    n_workers: Optional[int] = None
    threads_per_worker: int = 1
    processes: bool = True
    memory_limit: str = "auto"
    dashboard_address: Optional[str] = None
    local_directory: Optional[str] = None


@dataclass
class IOConfig:
    """Optional high-volume IO performance configuration."""

    netcdf_compression: bool = False
    netcdf_compression_level: int = 1
    mfdataset_batch_size: Optional[int] = None  # None means auto planner; 0 disables batching
    mfdataset_auto_batch_min_files: Optional[int] = None
    mfdataset_auto_batch_min_size_mb: Optional[int] = None
    mfdataset_auto_batch_min_size: Optional[int] = None
    mfdataset_auto_batch_max_size: Optional[int] = None
    mfdataset_auto_batch_memory_fraction: Optional[float] = None


@dataclass
class ProjectConfig:
    """Project identification, evaluation parameters, and runtime options."""

    # --- Identification ---
    name: str  # Project name; output subdirectory uses it
    output_dir: str  # Output root directory (relative to cwd or absolute)
    years: list[int]  # [start_year, end_year], inclusive — exactly 2 ints

    # --- Spatial-temporal bounds ---
    min_year_threshold: int = 1  # Per-station filter: drop stations whose valid span < N years (stn ref only)
    lat_range: list[float] = field(default_factory=lambda: [-90.0, 90.0])  # Spatial bounds, default global
    lon_range: list[float] = field(default_factory=lambda: [-180.0, 180.0])  # Spatial bounds, default global

    # --- Target resolution (used in ALL phases: evaluation, comparison, statistics) ---
    tim_res: Optional[str] = None  # Target time resolution: Month | Day | Hour | Year (also "3hr" etc.)
    grid_res: Optional[float] = None  # Target spatial resolution in degrees (e.g., 0.5, 0.25)
    timezone: Optional[float] = None  # Timezone offset in hours (only matters for hourly/daily eval)
    weight: Optional[str] = None  # Spatial weighting: area | mass | none (omitted → adapter substitutes "area")

    # --- Runtime ---
    num_cores: Optional[int] = None  # Parallel cores (None/0 → os.cpu_count())
    time_alignment: str = "intersection"  # intersection | per_pair | strict
    # openbench_conservative | cdo_remapcon | xesmf_conservative | basic_interpolation
    regrid_backend: str = "openbench_conservative"
    unified_mask: bool = True  # Cumulative NaN mask across sims (cross-sim fairness)
    generate_report: bool = True  # Generate HTML/PDF summary report

    # --- Groupby analysis ---
    IGBP_groupby: bool = False  # Per-IGBP-class aggregation (loads dataset/IGBP.nc)
    PFT_groupby: bool = False  # Per-PFT-class aggregation (loads dataset/PFT.nc)
    climate_zone_groupby: bool = False  # Per-Köppen-zone aggregation

    # --- Advanced ---
    debug_mode: bool = False  # Diagnostic prints in DatasetProcessing (not a global log level)
    only_drawing: bool = False  # Skip evaluation, only re-render plots from existing outputs
    force: bool = False  # Bypass incremental cache, full re-evaluation
    strict_reference: bool = False  # Treat LOW-provenance refs as errors instead of warnings
    dask: DaskConfig = field(default_factory=DaskConfig)  # Optional dask.distributed runtime options
    io: IOConfig = field(default_factory=IOConfig)  # Optional high-volume IO performance options


@dataclass
class EvaluationConfig:
    """Which variables to evaluate."""

    variables: list[str]  # Variable names to evaluate (must be non-empty)


@dataclass
class SimulationEntry:
    """A single simulation model entry.

    When 'model' matches a known model profile, variable mappings are
    resolved from the registry. Fields here override the profile.
    """

    model: str  # Model profile name (e.g., CoLM2024); resolves variable mappings
    root_dir: str  # Model output root directory
    data_type: Optional[str] = None  # grid | stn (overrides profile)
    grid_res: Optional[float] = None  # Spatial resolution; overrides profile
    tim_res: Optional[str] = None  # Time resolution; overrides profile
    data_groupby: Optional[str] = None  # Year | Month | Day | single — file split granularity
    prefix: Optional[str] = None  # File name prefix (e.g., "Case01_hist_")
    suffix: Optional[str] = None  # File name suffix
    fulllist: Optional[str] = None  # Station list CSV path (optional for stn sim; auto-scan may populate)
    variables: Optional[dict[str, dict[str, Any]]] = None  # Per-variable overrides (varname, units, ...)


@dataclass
class ReferenceConfig:
    """Reference data configuration.

    Holds both the data_root directory and variable→source mappings.
    Each variable can map to a single source or a list of sources;
    when multiple sources are given, they are evaluated independently
    (sim × ref Cartesian product, matching v2.x behavior).
    """

    data_root: Optional[str] = None  # Root directory for reference datasets
    # Variable -> source name (str) or list (multi-ref).
    sources: dict[str, "str | list[str]"] = field(default_factory=dict)


@dataclass
class ComparisonConfig:
    """Multi-model comparison settings."""

    enabled: bool = False  # Enable cross-simulation comparison phase
    items: Optional[list[str]] = None  # Figure types; omitted → ["Taylor_Diagram", "HeatMap"] (NOT all 14)


@dataclass
class StatisticsConfig:
    """Statistical analysis settings."""

    enabled: bool = False  # Enable independent statistical analysis phase
    items: Optional[list[str]] = None  # Method names; omitted → empty (enabled has no effect without items)


@dataclass
class OpenBenchConfig:
    """Top-level configuration container.

    Maps directly to openbench.yaml structure.
    """

    project: ProjectConfig  # Project metadata + runtime options
    evaluation: EvaluationConfig  # Variables list
    reference: ReferenceConfig  # Reference data mapping
    simulation: dict[str, SimulationEntry]  # Simulation entries (label → entry)

    metrics: Optional[list[str]] = None  # Metric names; omitted → ["bias", "RMSE", "correlation"]
    scores: Optional[list[str]] = None  # Score names; omitted → ["Overall_Score"]
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)  # Comparison phase config
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)  # Statistics phase config
