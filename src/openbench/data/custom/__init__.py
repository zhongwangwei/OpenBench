"""Custom data filters for complex variable processing.

Three variable resolution mechanisms (priority high → low):
1. compute: YAML expression in model_catalog.yaml (pip/conda safe)
2. filter:  Python file in user directory (pip/conda safe)
3. direct:  ds[varname] extraction (pip/conda safe)

Filter search order:
1. User: ~/.openbench/custom/<name>_filter.py  (or OPENBENCH_CUSTOM_DIR)
2. Built-in: openbench/data/custom/<name>_filter.py (remaining package filters; most station matching is registry-driven)
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
    from openbench.config.user_settings import get_user_config_dir

    return get_user_config_dir() / "custom"


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
        spec = importlib.util.spec_from_file_location(f"openbench_user_filter.{name}_filter", user_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load user filter: {user_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        logger.debug("Loaded user filter: %s", user_file)
        return mod

    # 2. Built-in package
    try:
        return importlib.import_module(f"openbench.data.custom.{name}_filter")
    except ModuleNotFoundError as exc:
        module_name = f"openbench.data.custom.{name}_filter"
        if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
            raise

    # 3. Fallback: strip trailing version digits (CoLM2024 → CoLM, BCC_AVIM2 → BCC_AVIM)
    import re

    base_name = re.sub(r"[\d.]+$", "", name)
    if base_name and base_name != name:
        user_file_base = user_dir / f"{base_name}_filter.py"
        if user_file_base.exists():
            spec = importlib.util.spec_from_file_location(f"openbench_user_filter.{base_name}_filter", user_file_base)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load user filter: {user_file_base}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.debug("Loaded user filter (base name): %s", user_file_base)
            return mod
        try:
            return importlib.import_module(f"openbench.data.custom.{base_name}_filter")
        except ModuleNotFoundError as exc:
            module_name = f"openbench.data.custom.{base_name}_filter"
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise

    return None
