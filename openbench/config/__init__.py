"""
OpenBench configuration management.

This module provides classes and functions for managing OpenBench configuration,
including reading, updating, and processing configuration data.
"""

from .manager import ConfigManager, config_manager, load_config, validate_config, create_config_template
from .readers import NamelistReader, FortranNMLReader
from .updaters import UpdateNamelist, UpdateFigNamelist
from .processors import GeneralInfoReader

__all__ = [
    # Configuration management
    "ConfigManager",
    "config_manager",
    "load_config", 
    "validate_config",
    "create_config_template",
    # Configuration readers
    "NamelistReader",
    "FortranNMLReader",
    # Configuration updaters
    "UpdateNamelist",
    "UpdateFigNamelist", 
    # Configuration processors
    "GeneralInfoReader",
]