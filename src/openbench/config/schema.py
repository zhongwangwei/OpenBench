"""OpenBench configuration schema defined as dataclasses.

Each section of openbench.yaml maps to a dataclass with typed fields
and sensible defaults. The top-level OpenBenchConfig holds everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectConfig:
    """Project identification, evaluation parameters, and runtime options."""

    # --- Identification ---
    name: str                           # Project name; output subdirectory uses it
    output_dir: str                     # Output root directory (relative to cwd or absolute)
    years: list[int]                    # [start_year, end_year], inclusive — exactly 2 ints

    # --- Spatial-temporal bounds ---
    min_year_threshold: int = 3         # Per-station filter: drop stations whose valid span < N years (stn ref only)
    lat_range: list[float] = field(default_factory=lambda: [-90.0, 90.0])    # Spatial bounds, default global
    lon_range: list[float] = field(default_factory=lambda: [-180.0, 180.0])  # Spatial bounds, default global

    # --- Target resolution (used in ALL phases: evaluation, comparison, statistics) ---
    tim_res: Optional[str] = None       # Target time resolution: Month | Day | Hour | Year (also "3hr" etc.)
    grid_res: Optional[float] = None    # Target spatial resolution in degrees (e.g., 0.5, 0.25)
    timezone: Optional[float] = None    # Timezone offset in hours (only matters for hourly/daily eval)
    weight: Optional[str] = None        # Spatial weighting: area | mass | none (omitted → adapter substitutes "area")

    # --- Runtime ---
    num_cores: Optional[int] = None     # Parallel cores (None → os.cpu_count())
    time_alignment: str = "intersection"  # intersection | per_pair | strict
    unified_mask: bool = True           # Cumulative NaN mask across sims (cross-sim fairness)
    generate_report: bool = True        # Generate HTML/PDF summary report

    # --- Groupby analysis ---
    IGBP_groupby: bool = False          # Per-IGBP-class aggregation (loads dataset/IGBP.nc)
    PFT_groupby: bool = False           # Per-PFT-class aggregation (loads dataset/PFT.nc)
    climate_zone_groupby: bool = False  # Per-Köppen-zone aggregation

    # --- Advanced ---
    debug_mode: bool = False            # Diagnostic prints in DatasetProcessing (not a global log level)
    only_drawing: bool = False          # Skip evaluation, only re-render plots from existing outputs
    force: bool = False                 # Bypass incremental cache, full re-evaluation
    strict_reference: bool = False      # Treat LOW-provenance refs as errors instead of warnings


@dataclass
class EvaluationConfig:
    """Which variables to evaluate."""

    variables: list[str]                # Variable names to evaluate (must be non-empty)


@dataclass
class SimulationEntry:
    """A single simulation model entry.

    When 'model' matches a known model profile, variable mappings are
    resolved from the registry. Fields here override the profile.
    """

    model: str                          # Model profile name (e.g., CoLM2024); resolves variable mappings
    root_dir: str                       # Model output root directory
    data_type: Optional[str] = None     # grid | stn (overrides profile)
    grid_res: Optional[float] = None    # Spatial resolution; overrides profile
    tim_res: Optional[str] = None       # Time resolution; overrides profile
    data_groupby: Optional[str] = None  # Year | Month | Day | single — file split granularity
    prefix: Optional[str] = None        # File name prefix (e.g., "Case01_hist_")
    suffix: Optional[str] = None        # File name suffix
    fulllist: Optional[str] = None      # Station list CSV path (required for stn data)
    variables: Optional[dict[str, dict[str, Any]]] = None  # Per-variable overrides (varname, units, ...)


@dataclass
class ReferenceConfig:
    """Reference data configuration.

    Holds both the data_root directory and variable→source mappings.
    Each variable can map to a single source or a list of sources;
    when multiple sources are given, they are evaluated independently
    (sim × ref Cartesian product, matching v2.x behavior).
    """

    data_root: Optional[str] = None     # Root directory for reference datasets
    sources: dict[str, "str | list[str]"] = field(default_factory=dict)  # Variable → source name (str) or list (multi-ref)


@dataclass
class ComparisonConfig:
    """Multi-model comparison settings."""

    enabled: bool = False               # Enable cross-simulation comparison phase
    items: Optional[list[str]] = None   # Figure types; omitted → ["Taylor_Diagram", "HeatMap"] (NOT all 14)


@dataclass
class StatisticsConfig:
    """Statistical analysis settings."""

    enabled: bool = False               # Enable independent statistical analysis phase
    items: Optional[list[str]] = None   # Method names; omitted → empty (enabled has no effect without items)


@dataclass
class OpenBenchConfig:
    """Top-level configuration container.

    Maps directly to openbench.yaml structure.
    """

    project: ProjectConfig              # Project metadata + runtime options
    evaluation: EvaluationConfig        # Variables list
    reference: ReferenceConfig          # Reference data mapping
    simulation: dict[str, SimulationEntry]  # Simulation entries (label → entry)

    metrics: Optional[list[str]] = None  # Metric names; omitted → ["bias", "RMSE", "correlation"]
    scores: Optional[list[str]] = None   # Score names; omitted → ["Overall_Score"]
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)   # Comparison phase config
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)   # Statistics phase config
