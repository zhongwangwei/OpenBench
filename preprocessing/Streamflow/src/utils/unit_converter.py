"""Unit conversion utilities for streamflow pipeline."""

import numpy as np
from typing import Union

from ..constants import DISCHARGE_CONVERSIONS, AREA_CONVERSIONS

Numeric = Union[float, np.ndarray]


def convert_discharge(value: Numeric, from_unit: str) -> Numeric:
    """Convert discharge to m3/s. For mm/d use convert_mmd_to_m3s instead."""
    if from_unit not in DISCHARGE_CONVERSIONS:
        raise ValueError(f"Unknown discharge unit: {from_unit}. Use convert_mmd_to_m3s for mm/d.")
    return value * DISCHARGE_CONVERSIONS[from_unit]


def convert_mmd_to_m3s(value: Numeric, area_km2: float) -> Numeric:
    """Convert mm/d to m3/s: Q = value * area_km2 * 1000 / 86400."""
    return value * area_km2 * 1000.0 / 86400.0


def convert_area(value: Numeric, from_unit: str) -> Numeric:
    """Convert area to km2."""
    if from_unit not in AREA_CONVERSIONS:
        raise ValueError(f"Unknown area unit: {from_unit}")
    return value * AREA_CONVERSIONS[from_unit]
