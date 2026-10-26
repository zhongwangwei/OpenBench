"""Basin-mask helpers for station-based evaluations."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _normalized_longitudes(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


@lru_cache(maxsize=8)
def _load_basin_mask_cached(path: str, size: int, mtime_ns: int) -> xr.DataArray:
    del size, mtime_ns  # Cache-key inputs; the resolved path is the file to read.
    try:
        with xr.open_dataset(path) as dataset:
            if "basin_mask" not in dataset.data_vars:
                raise ValueError(f"Basin mask file does not contain a 'basin_mask' variable: {path}")
            mask = dataset["basin_mask"]
            if mask.ndim != 2 or set(mask.dims) != {"lat", "lon"}:
                raise ValueError("Basin mask variable must be two-dimensional with ('lat', 'lon') dimensions")
            if "lat" not in mask.coords or "lon" not in mask.coords:
                raise ValueError("Basin mask variable must provide one-dimensional lat/lon coordinates")
            if mask["lat"].ndim != 1 or mask["lon"].ndim != 1:
                raise ValueError("Basin mask lat/lon coordinates must be one-dimensional")

            mask = (mask.fillna(0) > 0).load()
    except OSError as exc:
        raise ValueError(f"Could not read basin mask file {path}: {exc}") from exc

    mask = mask.assign_coords(lon=_normalized_longitudes(mask["lon"].values))
    if np.unique(mask["lat"].values).size != mask.sizes["lat"]:
        raise ValueError("Basin mask latitude coordinates must be unique")
    if np.unique(mask["lon"].values).size != mask.sizes["lon"]:
        raise ValueError("Basin mask longitude coordinates must be unique after normalization")
    return mask.sortby("lat").sortby("lon")


def load_basin_mask(path: str | os.PathLike[str]) -> xr.DataArray:
    """Load and validate the configured binary basin footprint."""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    try:
        stat = expanded.stat()
    except OSError as exc:
        raise ValueError(f"Basin mask file is not accessible: {expanded} ({exc})") from exc
    if not expanded.is_file():
        raise ValueError(f"Basin mask path is not a file: {expanded}")
    return _load_basin_mask_cached(str(expanded.resolve()), stat.st_size, stat.st_mtime_ns)


def _coordinate_extent(values, spacing_values=None) -> tuple[float, float]:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Basin mask coordinates must contain finite values")
    spacing_values = values if spacing_values is None else np.sort(np.asarray(spacing_values, dtype=float))
    padding = float(np.median(np.diff(spacing_values))) / 2.0 if spacing_values.size > 1 else 0.0
    return float(values[0] - abs(padding)), float(values[-1] + abs(padding))


def basin_mask_extent(mask_path: str | os.PathLike[str]) -> tuple[float, float, float, float]:
    """Return the complete positive-mask extent, padded to cell edges."""
    mask = load_basin_mask(mask_path)
    active = mask.where(mask, drop=True)
    if active.size == 0:
        raise ValueError(f"Basin mask has no positive cells: {mask_path}")
    min_lon, max_lon = _coordinate_extent(active["lon"].values, mask["lon"].values)
    min_lat, max_lat = _coordinate_extent(active["lat"].values, mask["lat"].values)
    return min_lon, max_lon, min_lat, max_lat


def validate_extent_covers_basin(
    extent: tuple[float, float, float, float],
    mask_path: str | os.PathLike[str],
) -> tuple[float, float, float, float]:
    """Validate that a geographic extent contains every positive mask cell."""
    basin_extent = basin_mask_extent(mask_path)
    min_lon, max_lon, min_lat, max_lat = (float(value) for value in extent)
    req_min_lon, req_max_lon, req_min_lat, req_max_lat = basin_extent
    tolerance = 1e-6
    covered = (
        min_lon <= req_min_lon + tolerance
        and max_lon >= req_max_lon - tolerance
        and min_lat <= req_min_lat + tolerance
        and max_lat >= req_max_lat - tolerance
    )
    if not covered:
        raise ValueError(
            "Configured grid extent does not cover the complete basin mask: "
            f"configured=({min_lon}, {max_lon}, {min_lat}, {max_lat}), "
            f"required=({req_min_lon}, {req_max_lon}, {req_min_lat}, {req_max_lat})"
        )
    return basin_extent


def basin_mask_for_grid(
    mask_path: str | os.PathLike[str],
    latitudes,
    longitudes,
    *,
    require_full_coverage: bool = True,
) -> xr.DataArray:
    """Map the binary basin footprint to a rectilinear target grid."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if latitudes.ndim != 1 or longitudes.ndim != 1 or not latitudes.size or not longitudes.size:
        raise ValueError("Grid basin masking requires non-empty one-dimensional lat/lon coordinates")
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("Grid basin masking requires finite lat/lon coordinates")

    normalized_longitudes = _normalized_longitudes(longitudes)
    if np.unique(latitudes).size != latitudes.size or np.unique(normalized_longitudes).size != longitudes.size:
        raise ValueError("Grid basin masking requires unique lat/lon coordinates")

    if require_full_coverage:
        min_lon, max_lon = _coordinate_extent(normalized_longitudes)
        min_lat, max_lat = _coordinate_extent(latitudes)
        validate_extent_covers_basin((min_lon, max_lon, min_lat, max_lat), mask_path)

    mask = load_basin_mask(mask_path).astype(np.uint8)
    mapped = mask.interp(
        lat=xr.DataArray(latitudes, dims="lat"),
        lon=xr.DataArray(normalized_longitudes, dims="lon"),
        method="nearest",
    )
    mapped = mapped.fillna(0).astype(bool)
    return mapped.assign_coords(lat=latitudes, lon=longitudes).rename("basin_mask")


def apply_basin_mask_to_grid(
    data: xr.Dataset | xr.DataArray,
    mask_path: str | os.PathLike[str],
) -> xr.Dataset | xr.DataArray:
    """Set grid cells outside the configured basin to NaN."""
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("Grid basin masking requires lat/lon coordinates")
    mapped = basin_mask_for_grid(mask_path, data["lat"].values, data["lon"].values)
    return data.where(mapped)


def _station_coordinate_columns(stations: pd.DataFrame) -> tuple[str, str]:
    for lon_name, lat_name in (
        ("ref_lon", "ref_lat"),
        ("sim_lon", "sim_lat"),
        ("lon", "lat"),
        ("LON", "LAT"),
        ("longitude", "latitude"),
        ("Longitude", "Latitude"),
    ):
        if lon_name in stations.columns and lat_name in stations.columns:
            return lon_name, lat_name
    raise ValueError("Basin mask filtering requires station longitude and latitude columns")


def filter_stations_by_basin_mask(
    stations: pd.DataFrame,
    mask_path: str | os.PathLike[str],
) -> pd.DataFrame:
    """Return station rows whose coordinates fall in a positive mask cell."""
    if stations.empty:
        return stations.copy()

    lon_name, lat_name = _station_coordinate_columns(stations)
    longitudes = pd.to_numeric(stations[lon_name], errors="coerce").to_numpy(dtype=float)
    latitudes = pd.to_numeric(stations[lat_name], errors="coerce").to_numpy(dtype=float)
    valid_coordinates = np.isfinite(longitudes) & np.isfinite(latitudes)
    inside = np.zeros(len(stations), dtype=bool)

    if valid_coordinates.any():
        mask = load_basin_mask(mask_path)
        selected = mask.sel(
            lon=xr.DataArray(_normalized_longitudes(longitudes[valid_coordinates]), dims="station"),
            lat=xr.DataArray(latitudes[valid_coordinates], dims="station"),
            method="nearest",
        )
        inside[valid_coordinates] = np.asarray(selected.values, dtype=bool)

    return stations.loc[inside].copy()
