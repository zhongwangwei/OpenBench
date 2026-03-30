"""OpenBench configuration schema defined as dataclasses.

Each section of openbench.yaml maps to a dataclass with typed fields
and sensible defaults. The top-level OpenBenchConfig holds everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectConfig:
    """Project identification and spatial-temporal bounds."""

    name: str
    output_dir: str
    years: list[int]  # [start_year, end_year]
    min_year_threshold: int = 3
    lat_range: list[float] = field(default_factory=lambda: [-90.0, 90.0])
    lon_range: list[float] = field(default_factory=lambda: [-180.0, 180.0])


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
    variables: Optional[dict[str, dict[str, Any]]] = None


@dataclass
class ComparisonConfig:
    """Multi-model comparison settings."""

    enabled: bool = False
    items: Optional[list[str]] = None
    weight: Optional[str] = None
    tim_res: Optional[str] = None
    timezone: Optional[float] = None
    grid_res: Optional[float] = None


@dataclass
class StatisticsConfig:
    """Statistical analysis settings."""

    enabled: bool = False
    items: Optional[list[str]] = None


@dataclass
class OptionsConfig:
    """Global options with sensible defaults."""

    num_cores: Optional[int] = None  # None = auto-detect
    time_alignment: str = "intersection"  # intersection | per_pair | strict
    unified_mask: bool = True
    generate_report: bool = True
    IGBP_groupby: bool = False
    PFT_groupby: bool = False
    climate_zone_groupby: bool = False
    debug_mode: bool = False
    only_drawing: bool = False
    data_root: Optional[str] = None  # Root directory for reference datasets


@dataclass
class OpenBenchConfig:
    """Top-level configuration container.

    Maps directly to openbench.yaml structure.
    """

    project: ProjectConfig
    evaluation: EvaluationConfig
    reference: dict[str, str]  # variable_name -> registry source name
    simulation: dict[str, SimulationEntry]  # label -> entry

    metrics: Optional[list[str]] = None  # None = all available
    scores: Optional[list[str]] = None  # None = all available
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
