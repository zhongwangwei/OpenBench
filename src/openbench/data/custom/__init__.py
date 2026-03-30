"""Custom data filters for model/dataset-specific processing.

Filters are loaded from two locations (user overrides built-in):
1. Built-in: openbench/data/custom/<model>_filter.py  (shipped with package)
2. User: ~/.openbench/custom/<model>_filter.py  or  OPENBENCH_CUSTOM_DIR env var

To add a custom filter for your model, create a file like:
    ~/.openbench/custom/MyModel_filter.py

with functions:
    def adjust_time_MyModel(info, ds, syear, eyear, tim_res):
        ...
        return ds

    def filter_MyModel(info, ds):
        ...
        return info, ds
"""

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

_user_custom_dir: Optional[Path] = None


def _get_user_custom_dir() -> Path:
    """Get the user custom filter directory."""
    global _user_custom_dir
    if _user_custom_dir is not None:
        return _user_custom_dir

    env_dir = os.environ.get("OPENBENCH_CUSTOM_DIR")
    if env_dir:
        _user_custom_dir = Path(env_dir)
    else:
        try:
            from platformdirs import user_config_dir

            _user_custom_dir = Path(user_config_dir("openbench")) / "custom"
        except ImportError:
            _user_custom_dir = Path.home() / ".openbench" / "custom"

    return _user_custom_dir


def load_custom_module(model_name: str) -> Optional[ModuleType]:
    """Load a custom filter module for a model.

    Searches:
    1. User custom directory (overrides built-in)
    2. Built-in openbench.data.custom package

    Returns:
        The loaded module, or None if not found.
    """
    # 1. Try user custom directory first
    user_dir = _get_user_custom_dir()
    user_file = user_dir / f"{model_name}_filter.py"
    if user_file.exists():
        try:
            module_name = f"openbench_user_custom.{model_name}_filter"
            spec = importlib.util.spec_from_file_location(module_name, user_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                logger.debug("Loaded user custom filter: %s", user_file)
                return mod
        except Exception as e:
            logger.warning("Failed to load user custom filter %s: %s", user_file, e)

    # 2. Try built-in package
    try:
        return importlib.import_module(f"openbench.data.custom.{model_name}_filter")
    except ImportError:
        return None
