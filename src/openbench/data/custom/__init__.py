"""Custom data filters for complex variable processing.

Three variable resolution mechanisms (priority high → low):
1. compute: YAML expression in model_catalog.yaml (pip/conda safe)
2. filter:  Python file in user directory (pip/conda safe)
3. direct:  ds[varname] extraction (pip/conda safe)

Filter search order:
1. User: ~/.openbench/custom/<name>_filter.py  (or OPENBENCH_CUSTOM_DIR)
2. Built-in: openbench/data/custom/<name>_filter.py (station filters shipped with package)
"""

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)


def _get_user_custom_dir() -> Path:
    """Get the user custom filter directory."""
    env_dir = os.environ.get("OPENBENCH_CUSTOM_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        from platformdirs import user_config_dir

        return Path(user_config_dir("openbench")) / "custom"
    except ImportError:
        return Path.home() / ".openbench" / "custom"


def load_filter(name: str) -> Optional[ModuleType]:
    """Load a filter module by name.

    Searches user directory first, then built-in package.

    Args:
        name: Filter name (e.g., 'CoLM', 'GRDC_Monthly')

    Returns:
        Loaded module, or None if not found.
    """
    # 1. User directory first (overrides built-in)
    user_dir = _get_user_custom_dir()
    user_file = user_dir / f"{name}_filter.py"
    if user_file.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                f"openbench_user_filter.{name}_filter", user_file
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                logger.debug("Loaded user filter: %s", user_file)
                return mod
        except Exception as e:
            logger.warning("Failed to load user filter %s: %s", user_file, e)

    # 2. Built-in package
    try:
        return importlib.import_module(f"openbench.data.custom.{name}_filter")
    except ImportError:
        return None
