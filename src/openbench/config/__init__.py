"""Configuration loading, validation, and schema.

Public API:
    load_config(path) -> OpenBenchConfig
    ConfigError - validation error
    OpenBenchConfig, ProjectConfig, ... - schema dataclasses
"""

from openbench.config.loader import ConfigError, load_config
from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
    SimulationEntry,
    StatisticsConfig,
)

__all__ = [
    "load_config",
    "ConfigError",
    "OpenBenchConfig",
    "ProjectConfig",
    "EvaluationConfig",
    "SimulationEntry",
    "ComparisonConfig",
    "StatisticsConfig",
    "OptionsConfig",
]
