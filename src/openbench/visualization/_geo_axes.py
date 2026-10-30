"""Shared geographic extent, tick, and gridline configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import is_color_like

from openbench.data.basin_mask import basin_mask_extent, load_basin_mask


_CANONICAL_DEGREE_STEPS = (15.0, 30.0, 45.0, 60.0, 90.0, 180.0)
_RIVER_RESOLUTIONS = {"110m", "50m", "10m"}


def add_rivers(ax, option: Mapping):
    """Validate river settings and add the configured Natural Earth feature."""
    show_rivers = option.get("show_rivers", True)
    if not isinstance(show_rivers, bool):
        raise ValueError("show_rivers must be a boolean")

    resolution = str(option.get("river_resolution", "10m"))
    if resolution not in _RIVER_RESOLUTIONS:
        raise ValueError("river_resolution must be one of: 110m, 50m, 10m")

    color = option.get("river_color", "#6baed6")
    if not is_color_like(color):
        raise ValueError(f"river_color is not a valid Matplotlib color: {color!r}")

    try:
        linewidth = float(option.get("river_linewidth", 0.35))
    except (TypeError, ValueError) as exc:
        raise ValueError("river_linewidth must be a non-negative number") from exc
    if not math.isfinite(linewidth) or linewidth < 0:
        raise ValueError("river_linewidth must be a non-negative number")

    try:
        alpha = float(option.get("river_alpha", 0.8))
    except (TypeError, ValueError) as exc:
        raise ValueError("river_alpha must be a number between 0 and 1") from exc
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("river_alpha must be a number between 0 and 1")

    if not show_rivers:
        return None

    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", resolution, edgecolor=color, facecolor="none"
    )
    return ax.add_feature(rivers, linewidth=linewidth, alpha=alpha)


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


def _draw_basin_mask_boundary(ax, mask_path: str, projection) -> None:
    mask = load_basin_mask(mask_path)
    ax.contour(
        mask["lon"].values,
        mask["lat"].values,
        np.asarray(mask.values, dtype=float),
        levels=[0.5],
        colors=["black"],
        linewidths=[1.2],
        transform=projection,
        zorder=12,
    )


def configure_geo_axis(
    ax,
    option: Mapping,
    default_extent: tuple[float, float, float, float],
    *,
    gridline_kwargs: Mapping | None = None,
    basin_mask: str | None = None,
) -> tuple[tuple[float, float, float, float], np.ndarray, np.ndarray]:
    """Apply one extent and one matching tick set to a Cartopy map axis."""
    if basin_mask:
        extent = basin_mask_extent(basin_mask)
    elif option["set_lat_lon"]:
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
    if basin_mask:
        _draw_basin_mask_boundary(ax, basin_mask, projection)
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
