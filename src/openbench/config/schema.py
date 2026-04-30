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
    name: str
    output_dir: str
    years: list[int]  # [start_year, end_year]

    # --- Spatial-temporal bounds ---
    min_year_threshold: int = 3
    lat_range: list[float] = field(default_factory=lambda: [-90.0, 90.0])
    lon_range: list[float] = field(default_factory=lambda: [-180.0, 180.0])

    # --- Target resolution (used in ALL phases: evaluation, comparison, statistics) ---
    tim_res: Optional[str] = None       # Target time resolution: Month | Day | Hour | Year
    grid_res: Optional[float] = None    # Target spatial resolution in degrees
    timezone: Optional[float] = None    # Timezone offset in hours
    weight: Optional[str] = None        # Spatial weighting: area | equal

    # --- Runtime ---
    num_cores: Optional[int] = None     # None = auto-detect
    time_alignment: str = "intersection"  # intersection | per_pair | strict
    unified_mask: bool = True
    generate_report: bool = True

    # --- Groupby analysis ---
    IGBP_groupby: bool = False
    PFT_groupby: bool = False
    climate_zone_groupby: bool = False

    # --- Advanced ---
    debug_mode: bool = False
    only_drawing: bool = False
    force: bool = False               # Bypass incremental cache
    strict_reference: bool = False    # Unresolved references are errors vs warnings


@dataclass
class EvaluationConfig:
    """Which variables to evaluate."""

    variables: list[str]


@dataclass
class SimulationEntry:
    """A single simulation model entry.

    When 'model' matches a known model profile, variable mappings are
    resolved from the registry. Fields here override the profile.
    """

    model: str
    root_dir: str
    data_type: Optional[str] = None
    grid_res: Optional[float] = None
    tim_res: Optional[str] = None
    data_groupby: Optional[str] = None  # Year, month, day, single
    prefix: Optional[str] = None  # File name prefix (e.g., "Case01_hist_")
    suffix: Optional[str] = None  # File name suffix
    fulllist: Optional[str] = None  # Station list CSV path (for stn data)
    variables: Optional[dict[str, dict[str, Any]]] = None


@dataclass
class ReferenceConfig:
    """Reference data configuration.

    Holds both the data_root directory and variable→source mappings.
    Each variable can map to a single source or a list of sources;
    when multiple sources are given, they are evaluated independently
    (sim × ref Cartesian product, matching v2.x behavior).
    """

    data_root: Optional[str] = None   # Root directory for reference datasets
    # variable_name -> source_name (single) OR list of source_names
    sources: dict[str, "str | list[str]"] = field(default_factory=dict)


@dataclass
class ComparisonConfig:
    """Multi-model comparison settings."""

    enabled: bool = False
    items: Optional[list[str]] = None


@dataclass
class StatisticsConfig:
    """Statistical analysis settings."""

    enabled: bool = False
    items: Optional[list[str]] = None


@dataclass
class OpenBenchConfig:
    """Top-level configuration container.

    Maps directly to openbench.yaml structure.
    """

    project: ProjectConfig
    evaluation: EvaluationConfig
    reference: ReferenceConfig
    simulation: dict[str, SimulationEntry]  # label -> entry

    metrics: Optional[list[str]] = None  # None = all available
    scores: Optional[list[str]] = None  # None = all available
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
