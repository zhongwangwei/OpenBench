"""Shared geographic extent, tick, and gridline configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping

import cartopy.crs as ccrs
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


_CANONICAL_DEGREE_STEPS = (15.0, 30.0, 45.0, 60.0, 90.0, 180.0)


def adaptive_degree_ticks(minimum: float, maximum: float, *, target_count: int = 5) -> np.ndarray:
    """Return readable interior degree ticks for a geographic interval."""
    minimum = float(minimum)
    maximum = float(maximum)
    if not (math.isfinite(minimum) and math.isfinite(maximum) and minimum < maximum):
        raise ValueError(f"Invalid geographic interval: [{minimum}, {maximum}]")

    span = maximum - minimum
    exponent = math.floor(math.log10(span))
    steps = set(_CANONICAL_DEGREE_STEPS)
    for power in range(exponent - 3, exponent + 2):
        scale = 10.0**power
        steps.update(base * scale for base in (1.0, 2.0, 2.5, 5.0, 10.0))

    candidates = []
    tolerance = max(span * 1e-12, 1e-12)
    for step in sorted(step for step in steps if step > 0):
        first = math.ceil((minimum + tolerance) / step) * step
        ticks = np.arange(first, maximum - tolerance, step, dtype=float)
        ticks = ticks[(ticks > minimum + tolerance) & (ticks < maximum - tolerance)]
        count = ticks.size
        if count:
            preferred_density = 0 if 4 <= count <= 7 else 1
            candidates.append(((preferred_density, abs(count - target_count), -step), ticks))

    if not candidates:
        return np.array([(minimum + maximum) / 2.0])

    ticks = min(candidates, key=lambda candidate: candidate[0])[1]
    ticks[np.isclose(ticks, 0.0, atol=tolerance)] = 0.0
    return ticks


def _ticks_or_empty(minimum: float, maximum: float) -> np.ndarray:
    # Single-point or dateline-crossing extents (min >= max) still render;
    # they just get no ticks, as before the adaptive ticks were introduced.
    try:
        return adaptive_degree_ticks(minimum, maximum)
    except ValueError:
        return np.array([], dtype=float)


def configure_geo_axis(
    ax,
    option: Mapping,
    default_extent: tuple[float, float, float, float],
    *,
    gridline_kwargs: Mapping | None = None,
) -> tuple[tuple[float, float, float, float], np.ndarray, np.ndarray]:
    """Apply one extent and one matching tick set to a Cartopy map axis."""
    if option["set_lat_lon"]:
        extent = (
            float(option["min_lon"]),
            float(option["max_lon"]),
            float(option["min_lat"]),
            float(option["max_lat"]),
        )
    else:
        extent = tuple(float(value) for value in default_extent)

    min_lon, max_lon, min_lat, max_lat = extent
    lon_ticks = _ticks_or_empty(min_lon, max_lon)
    lat_ticks = _ticks_or_empty(min_lat, max_lat)
    projection = ccrs.PlateCarree()

    ax.set_extent(extent, crs=projection)
    ax.set_xticks(lon_ticks, crs=projection)
    ax.set_yticks(lat_ticks, crs=projection)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False, bottom=True, left=True)

    if gridline_kwargs is not None:
        kwargs = dict(gridline_kwargs)
        kwargs.update(draw_labels=False, xlocs=lon_ticks, ylocs=lat_ticks)
        ax.gridlines(**kwargs)

    return extent, lon_ticks, lat_ticks
