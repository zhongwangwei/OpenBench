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
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
    StatisticsConfig,
)

__all__ = [
    "load_config",
    "ConfigError",
    "OpenBenchConfig",
    "ProjectConfig",
    "EvaluationConfig",
    "ReferenceConfig",
    "SimulationEntry",
    "ComparisonConfig",
    "StatisticsConfig",
]
