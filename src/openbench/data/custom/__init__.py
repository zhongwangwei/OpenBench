"""Reference dataset station filters.

Station/reference filters handle complex pre-processing like:
- CaMA-Flood allocation for streamflow stations
- CSV station list parsing and spatial matching
- Per-station data extraction with parallel processing

These remain as Python files because they involve I/O operations and
complex logic that can't be expressed as simple compute expressions.

Model variable computations (e.g., CoLM GPP, CLM5 Net_Radiation)
are handled via YAML compute expressions in model_catalog.yaml instead.
"""

import importlib
import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)


def load_station_filter(dataset_name: str) -> Optional[ModuleType]:
    """Load a station/reference filter module.

    Args:
        dataset_name: Reference dataset name (e.g., 'GRDC_Monthly')

    Returns:
        The loaded module, or None if not found.
    """
    try:
        return importlib.import_module(f"openbench.data.custom.{dataset_name}_filter")
    except ImportError:
        return None
