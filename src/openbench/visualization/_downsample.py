"""Plot-only downsampling helpers for high-resolution gridded data.

These helpers are intentionally scoped to visualization. They must not be
used by metrics/scores/statistics calculations because they trade visual
density for rendering speed.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_PRESET_MAX_PIXELS = {
    "high": 2_000_000,
    "medium": 1_000_000,
    "low": 300_000,
}
_DEFAULT_MAX_PIXELS = 1_000_000


def _plot_options(option: dict[str, Any] | None) -> dict[str, Any]:
    if not option:
        return {}
    nested = option.get("visualization")
    if isinstance(nested, dict):
        merged = dict(option)
        merged.update(nested)
        return merged
    return option


def _max_pixels(option: dict[str, Any], render_resolution: str) -> int:
    if render_resolution in _PRESET_MAX_PIXELS:
        return _PRESET_MAX_PIXELS[render_resolution]
    try:
        return int(option.get("max_pixels", _DEFAULT_MAX_PIXELS))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PIXELS


def _coarsen_factor(lat_size: int, lon_size: int, max_pixels: int) -> int:
    pixels = lat_size * lon_size
    if max_pixels <= 0 or pixels <= max_pixels:
        return 1
    return max(2, math.ceil(math.sqrt(pixels / max_pixels)))


def downsample_for_plot(
    data: xr.DataArray | xr.Dataset,
    option: dict[str, Any] | None = None,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Downsample gridded data for rendering only.

    ``render_resolution=native`` disables the helper. Otherwise the helper
    limits the lat/lon pixel count using ``max_pixels`` or preset thresholds.
    The default method is block mean via ``xarray.coarsen(...).mean()``.
    """
    opts = _plot_options(option)
    render_resolution = str(opts.get("render_resolution", "auto")).lower()
    if render_resolution == "native":
        return data
    if lat_dim not in data.sizes or lon_dim not in data.sizes:
        return data

    lat_size = data.sizes[lat_dim]
    lon_size = data.sizes[lon_dim]
    max_pixels = _max_pixels(opts, render_resolution)
    factor = _coarsen_factor(lat_size, lon_size, max_pixels)
    if factor <= 1:
        return data

    method = str(opts.get("downsample_method", "coarsen_mean")).lower()
    if method in {"nearest", "stride"}:
        out = data.isel({lat_dim: slice(None, None, factor), lon_dim: slice(None, None, factor)})
    elif method == "coarsen_mean":
        out = data.coarsen({lat_dim: factor, lon_dim: factor}, boundary="trim").mean(skipna=True)
    else:
        logger.warning("Unknown plot downsample method %r; using coarsen_mean", method)
        out = data.coarsen({lat_dim: factor, lon_dim: factor}, boundary="trim").mean(skipna=True)

    logger.info(
        "Downsampled plot data from %sx%s to %sx%s using %s",
        lat_size,
        lon_size,
        out.sizes.get(lat_dim, lat_size),
        out.sizes.get(lon_dim, lon_size),
        method,
    )
    return out


def lat_lon_plot_args(
    data: xr.DataArray,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[xr.DataArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float], str]:
    """Return a DataArray and map coordinates normalized for x=lon, y=lat plotting."""
    if lat_dim not in data.dims or lon_dim not in data.dims:
        raise ValueError(f"Plot data must have {lat_dim!r} and {lon_dim!r} dimensions; got {data.dims!r}")

    plot_data = data.transpose(lat_dim, lon_dim)
    if plot_data[lat_dim].size > 1 and plot_data[lat_dim].values[0] > plot_data[lat_dim].values[-1]:
        plot_data = plot_data.sortby(lat_dim)
    if plot_data[lon_dim].size > 1 and plot_data[lon_dim].values[0] > plot_data[lon_dim].values[-1]:
        plot_data = plot_data.sortby(lon_dim)

    ilat = plot_data[lat_dim].values
    ilon = plot_data[lon_dim].values
    lon, lat = np.meshgrid(ilon, ilat)
    extent = (float(ilon[0]), float(ilon[-1]), float(ilat[0]), float(ilat[-1]))
    origin = "lower"
    return plot_data, ilat, ilon, lon, lat, extent, origin
