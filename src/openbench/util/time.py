"""Shared time-coordinate normalization helpers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def normalize_time_coordinate(data_array: Any, resolution: str | None) -> Any:
    """Normalize timestamps to the midpoint convention used for comparisons."""
    if not hasattr(data_array, "coords") or "time" not in data_array.coords:
        return data_array
    compare_res = str(resolution or "").strip().lower()
    if not compare_res:
        return data_array
    try:
        times = pd.to_datetime(data_array["time"].values)
    except Exception as exc:
        logger.debug("Time normalization skipped: %s", exc)
        return data_array
    if times.size == 0:
        return data_array

    if compare_res in {"day", "d", "1d", "daily"}:
        normalized = (times.floor("D") + pd.Timedelta(hours=12)).values
    elif compare_res in {"hour", "h", "1h", "hourly"}:
        normalized = (times.floor("h") + pd.Timedelta(minutes=30)).values
    elif compare_res in {"month", "mon", "m", "1m", "monthly"}:
        normalized = (times.to_period("M").to_timestamp(how="start") + pd.Timedelta(days=14, hours=12)).values
    elif compare_res in {"year", "yr", "y", "1y", "annual", "yearly"}:
        normalized = (times.to_period("Y").to_timestamp(how="start") + pd.Timedelta(days=182, hours=12)).values
    else:
        return data_array
    try:
        return data_array.assign_coords(time=("time", normalized))
    except Exception as exc:
        logger.debug("Failed to assign normalized time coordinates: %s", exc)
        return data_array


def align_time_coordinates(
    first: Any,
    second: Any,
    resolution: str | None,
) -> tuple[Any, Any, bool]:
    """Use normalized timestamps when they preserve more paired time steps."""
    exact_first, exact_second = xr.align(first, second, join="inner")
    normalized_first = normalize_time_coordinate(first, resolution)
    normalized_second = normalize_time_coordinate(second, resolution)
    try:
        normalized_first, normalized_second = xr.align(normalized_first, normalized_second, join="inner")
    except ValueError:
        return exact_first, exact_second, False
    if normalized_first.sizes.get("time", 0) > exact_first.sizes.get("time", 0):
        return normalized_first.sortby("time"), normalized_second.sortby("time"), True
    return exact_first.sortby("time"), exact_second.sortby("time"), False
