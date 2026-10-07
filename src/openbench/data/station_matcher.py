"""Generic station matching engine using CaMA-Flood allocation data.

Replaces 19 individual station filter files that share identical logic.
Configured via ``station_matching`` block in reference_catalog.yaml.

Supported matching methods:
- ``cama_allocation``: match stations to grid cells using CaMA allocation
  data, filter by area error and upstream area bounds.
- ``direct``: use raw station coordinates (no CaMA allocation), e.g. for
  coastal discharge datasets like Dai & Trenberth.

Both methods drop stations whose upstream area is below the minimum a river
needs to be resolved at the simulation resolution (``MIN_UPAREA_BY_RESOLUTION``).
``cama_allocation`` also drops stations whose allocation error is missing or
larger than ``MAX_CAMA_ALLOC_ERR``. Neither limit can be set in the catalog.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from openbench.data._system_resources import effective_cpu_count
from openbench.data.station_missing import missing_sentinels as _as_missing_sentinels
from openbench.data.station_missing import valid_station_mask as _valid_flow_mask
from openbench.util.exceptions import DataProcessingError
from openbench.util.filenames import filename_component, station_file_path
from openbench.util.names import get_xarray_key_case_insensitive
from openbench.util.netcdf import write_file_atomic
from openbench.util.netcdf import write_netcdf_atomic


def _require_dataset_field(ds: xr.Dataset, requested: str, label: str, dataset_path: Path) -> str:
    key = get_xarray_key_case_insensitive(ds, requested)
    if key is not None:
        return key
    available = [*map(str, ds.data_vars), *map(str, ds.coords)]
    logging.error(
        "Station matching field %s=%r not found in %s. Available fields: %s",
        label,
        requested,
        dataset_path,
        available,
    )
    raise DataProcessingError(
        f"Station matching field '{requested}' ({label}) not found in {dataset_path.name}",
        context={"field": label, "requested": requested, "available": available[:20]},
    )


def get_resolution_suffix(sim_grid_res: float) -> str:
    """Map simulation grid resolution (degrees) to CaMA resolution suffix."""
    res_map = {
        0.25: "15min",
        0.1: "06min",
        0.0833: "05min",
        0.05: "03min",
        0.0167: "01min",
    }
    for res, suffix in res_map.items():
        if abs(float(sim_grid_res) - res) < 0.001:
            return suffix
    raise ValueError(f"Unsupported CaMA station matching resolution: {sim_grid_res}")


# Smallest upstream area (km²) a gauged river needs before it is resolved at
# each CaMA resolution. Stations with a smaller reported area are not evaluated.
MIN_UPAREA_BY_RESOLUTION = {
    "15min": 3000.0,
    "06min": 500.0,
    "05min": 350.0,
    "03min": 150.0,
    "01min": 100.0,
}


def resolution_min_uparea(sim_grid_res) -> float:
    """Return the enforced minimum upstream area (km²) for a simulation grid resolution."""
    try:
        suffix = get_resolution_suffix(sim_grid_res)
    except (TypeError, ValueError):
        supported = ", ".join(f"{suffix} {area:g} km2" for suffix, area in MIN_UPAREA_BY_RESOLUTION.items())
        raise ValueError(
            f"Station matching has no minimum upstream area for simulation grid resolution "
            f"{sim_grid_res!r}; supported resolutions: {supported}"
        ) from None
    return MIN_UPAREA_BY_RESOLUTION[suffix]


# Largest fractional CaMA allocation error (``cama_alloc_err_<res>``) a station
# may have. A missing (NaN) error cannot be checked, so that station is dropped.
MAX_CAMA_ALLOC_ERR = 0.2


def _alloc_err_within_limit(alloc_err) -> bool:
    # Compare at the stored precision: a float32 0.2 widens to 0.20000000298 as a
    # Python float and would otherwise fail its own limit.
    alloc_err = np.asarray(alloc_err)
    if not np.issubdtype(alloc_err.dtype, np.floating):
        alloc_err = alloc_err.astype(float)
    limit = np.asarray(MAX_CAMA_ALLOC_ERR, dtype=alloc_err.dtype)
    return bool(np.isfinite(alloc_err) and np.abs(alloc_err) <= limit)


FULL_DATASET_SUFFIX = "_full.nc"
DIST_DATASET_SUFFIX = "_dist.nc"


def station_dataset_candidates(root, dataset_file: str) -> list[Path]:
    """Return the station dataset file, then its redistributable subset.

    A ``<name>_full.nc`` dataset may be shipped only as its redistributable
    ``<name>_dist.nc`` subset, which is used when the full file is absent.
    """
    primary = Path(root) / dataset_file
    candidates = [primary]
    if primary.name.endswith(FULL_DATASET_SUFFIX):
        candidates.append(primary.with_name(primary.name[: -len(FULL_DATASET_SUFFIX)] + DIST_DATASET_SUFFIX))
    return candidates


def resolve_station_dataset(root, dataset_file: str) -> Optional[Path]:
    """Return the first station dataset candidate that exists, or None."""
    candidates = station_dataset_candidates(root, dataset_file)
    for path in candidates:
        if path.is_file():
            if path != candidates[0]:
                logging.info("Station dataset %s not found; using %s", candidates[0], path.name)
            return path
    return None


# Upstream-area names tried when a dataset lacks the configured one, e.g. an
# older OpenBench_Streamflow file that still calls it "area".
AREA_VAR_FALLBACKS = ("upstream_area", "area")


def _station_area_key(ds: xr.Dataset, area_var: str, dataset_path: Path, min_uparea: float) -> Optional[str]:
    """Return the dataset's upstream-area variable, or None when it has none.

    An empty ``area_var`` means the dataset has no areas. A configured name that
    is missing falls back to ``AREA_VAR_FALLBACKS``; without any area the minimum
    upstream area cannot be applied, which is logged rather than skipped silently.
    """
    if not area_var:
        return None
    key = get_xarray_key_case_insensitive(ds, area_var)
    if key is not None:
        return key
    for name in AREA_VAR_FALLBACKS:
        key = get_xarray_key_case_insensitive(ds, name)
        if key is not None:
            logging.warning(
                "Station matching: %s has no %r; using %r as upstream area", dataset_path.name, area_var, key
            )
            return key
    logging.warning(
        "Station matching: %s has no upstream-area variable (%r or %s); the minimum upstream area of %g km2 "
        "is not applied",
        dataset_path.name,
        area_var,
        "/".join(AREA_VAR_FALLBACKS),
        min_uparea,
    )
    return None


def _station_id_to_string(value) -> str:
    """Return a stable station identifier without assuming it is numeric."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _unique_station_ids(station_ids: np.ndarray, data_source_names: np.ndarray | None = None) -> list[str]:
    """Return filename-safe station IDs that stay unique for consolidated products.

    Only duplicated IDs are qualified (by source name when available, otherwise
    by row index); unique IDs are returned unchanged.
    """
    ids = [_station_id_to_string(station_id) for station_id in station_ids]
    counts = Counter(ids)
    if len(counts) == len(ids):
        return ids

    qualified = []
    for idx, station_id in enumerate(ids):
        if counts[station_id] == 1:
            qualified.append(station_id)
        elif data_source_names is not None:
            source = filename_component(_station_id_to_string(data_source_names[idx]))
            qualified.append(f"{source}_{station_id}")
        else:
            qualified.append(f"{station_id}_idx{idx}")

    # A qualified ID can still collide with another row (same source, or an
    # existing ID that happens to match); fall back to the row index there.
    qualified_counts = Counter(qualified)
    return [
        f"{station_id}_idx{idx}" if qualified_counts[station_id] > 1 else station_id
        for idx, station_id in enumerate(qualified)
    ]


def _valid_flow_in_year_window(valid_mask: np.ndarray, times: np.ndarray, start_year: int, end_year: int) -> bool:
    years = pd.to_datetime(times).year
    window_mask = (years >= int(start_year)) & (years <= int(end_year))
    return bool((valid_mask & window_mask).any())


def _valid_flow_in_yyyymm_window(valid_mask: np.ndarray, times: np.ndarray, start_year: int, end_year: int) -> bool:
    years = np.asarray([int(str(int(t))[:4]) for t in times])
    window_mask = (years >= int(start_year)) & (years <= int(end_year))
    return bool((valid_mask & window_mask).any())


def _get_dim_case_insensitive(dims, requested: str | None) -> str | None:
    if not requested:
        return None
    requested_norm = requested.lower()
    for dim in dims:
        if str(dim).lower() == requested_norm:
            return dim
    return None


def _station_time_flow_values(
    discharge_da: xr.DataArray,
    *,
    station_dim: str,
    time_key: str,
    n_stations: int,
    station_indices: Optional[np.ndarray] = None,
    time_indices: Optional[np.ndarray] = None,
):
    """Return discharge values ordered as (station, time)."""
    time_dim = _get_dim_case_insensitive(discharge_da.dims, time_key) or _get_dim_case_insensitive(
        discharge_da.dims, "time"
    )
    station_dim_key = _get_dim_case_insensitive(discharge_da.dims, station_dim)
    if station_dim and station_dim_key is None:
        raise DataProcessingError(
            f"Station matching station_dim '{station_dim}' not found in discharge variable",
            context={"station_dim": station_dim, "dims": list(discharge_da.dims)},
        )
    if station_dim_key is None:
        candidates = [dim for dim in discharge_da.dims if dim != time_dim and discharge_da.sizes.get(dim) == n_stations]
        station_dim_key = candidates[0] if candidates else None
    if time_dim is None or station_dim_key is None or time_dim == station_dim_key:
        raise DataProcessingError(
            "Could not identify station/time dimensions for station matching discharge variable",
            context={"station_dim": station_dim, "time_var": time_key, "dims": list(discharge_da.dims)},
        )
    ordered = discharge_da.transpose(station_dim_key, time_dim)
    if station_indices is not None:
        ordered = ordered.isel({station_dim_key: station_indices})
    if time_indices is not None:
        ordered = ordered.isel({time_dim: time_indices})
    return ordered.values


def _target_year_window(info) -> tuple[int, int]:
    return max(int(info.sim_syear), int(info.syear)), min(int(info.sim_eyear), int(info.eyear))


def _target_time_indices(times: np.ndarray, info, time_format: Optional[str] = None) -> np.ndarray:
    start_year, end_year = _target_year_window(info)
    if time_format == "YYYYMM":
        years = np.asarray([int(str(int(t))[:4]) for t in times])
    else:
        years = pd.to_datetime(times).year
    return np.where((years >= start_year) & (years <= end_year))[0]


def _area_within_bounds(area: float, min_uparea: float, max_uparea: float) -> bool:
    return not ((area > 0 and area < min_uparea) or (area > 0 and area > max_uparea))


def _direct_candidate_station_indices(
    lons: np.ndarray,
    lats: np.ndarray,
    areas: np.ndarray,
    info,
    min_uparea: float,
    max_uparea: float,
) -> np.ndarray:
    keep = []
    for idx, (lon, lat, area) in enumerate(zip(lons, lats, areas)):
        lon_for_bounds = _normalize_lon_to_range(float(lon), info.min_lon, info.max_lon)
        area_value = float(area) if not np.isnan(area) else -9999.0
        if lon_for_bounds < info.min_lon or lon_for_bounds > info.max_lon:
            continue
        if float(lat) < info.min_lat or float(lat) > info.max_lat:
            continue
        if not _area_within_bounds(area_value, min_uparea, max_uparea):
            continue
        keep.append(idx)
    return np.asarray(keep, dtype=int)


def _cama_candidate_station_indices(
    lons: np.ndarray,
    lats: np.ndarray,
    areas: np.ndarray,
    cama_lons: np.ndarray,
    cama_lats: np.ndarray,
    alloc_errs: np.ndarray,
    info,
    min_uparea: float,
    max_uparea: float,
) -> np.ndarray:
    keep = []
    for idx, (lon, lat, area, cama_lon, cama_lat, alloc_err) in enumerate(
        zip(lons, lats, areas, cama_lons, cama_lats, alloc_errs)
    ):
        lon_for_bounds = _normalize_lon_to_range(float(lon), info.min_lon, info.max_lon)
        cama_lon = _normalize_lon_to_range(float(cama_lon), -180.0, 180.0)
        cama_lat = float(cama_lat)
        area_value = float(area) if not np.isnan(area) else -9999.0
        if np.isnan(cama_lon) or np.isnan(cama_lat) or cama_lon < -180 or cama_lon > 180:
            continue
        if cama_lat < -90 or cama_lat > 90:
            continue
        if not _alloc_err_within_limit(alloc_err):
            continue
        if lon_for_bounds < info.min_lon or lon_for_bounds > info.max_lon:
            continue
        if float(lat) < info.min_lat or float(lat) > info.max_lat:
            continue
        if not _area_within_bounds(area_value, min_uparea, max_uparea):
            continue
        keep.append(idx)
    return np.asarray(keep, dtype=int)


def _normalize_lon_to_range(lon: float, min_lon: float, max_lon: float) -> float:
    """Return an equivalent longitude inside the requested range when possible."""
    lon = float(lon)
    min_lon = float(min_lon)
    max_lon = float(max_lon)
    if not np.isfinite(lon) or not np.isfinite(min_lon) or not np.isfinite(max_lon):
        return lon
    if min_lon <= lon <= max_lon or max_lon - min_lon > 360:
        return lon
    for candidate in (lon - 360.0, lon + 360.0):
        if min_lon <= candidate <= max_lon:
            return candidate
    return lon


def _station_matching_jobs(n_stations: int, requested: int | None = None) -> int:
    """Choose a conservative station-matching worker count."""
    if requested is not None:
        return max(1, int(requested))
    env_value = os.environ.get("OPENBENCH_STATION_MATCHER_JOBS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logging.warning("Ignoring invalid OPENBENCH_STATION_MATCHER_JOBS=%r", env_value)
    cpu_count = effective_cpu_count(os.cpu_count() or 1)
    return max(1, min(n_stations, cpu_count, 4))


# ---------------------------------------------------------------------------
# CaMA allocation matching
# ---------------------------------------------------------------------------


def _process_site_cama(
    idx: int,
    station_ids: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    areas: np.ndarray,
    cama_lons: np.ndarray,
    cama_lats: np.ndarray,
    alloc_errs: np.ndarray,
    flow: np.ndarray,
    times: np.ndarray,
    info,
    scratch_dir: Path,
    min_uparea: float,
    max_uparea: float,
    output_station_ids: list[str],
    original_indices: Optional[np.ndarray] = None,
    duplicate_station_ids: set[str] | None = None,
    missing_sentinels: tuple[float, ...] = (),
):
    """Process one station for CaMA allocation matching.  Returns metadata row or None."""
    original_idx = int(original_indices[idx]) if original_indices is not None else idx
    station_id = _station_id_to_string(station_ids[idx])
    output_station_id = output_station_ids[idx]
    lon = float(lons[idx])
    lat = float(lats[idx])
    area = float(areas[idx]) if not np.isnan(areas[idx]) else -9999.0

    cama_lon = float(cama_lons[idx])
    cama_lat = float(cama_lats[idx])
    alloc_err = alloc_errs[idx]
    lon_for_bounds = _normalize_lon_to_range(lon, info.min_lon, info.max_lon)
    cama_lon = _normalize_lon_to_range(cama_lon, -180.0, 180.0)

    if np.isnan(cama_lon) or np.isnan(cama_lat) or cama_lon < -180 or cama_lon > 180 or cama_lat < -90 or cama_lat > 90:
        return None

    # Area error filter
    if not _alloc_err_within_limit(alloc_err):
        return None

    # Streamflow time series
    valid_mask = _valid_flow_mask(flow, missing_sentinels)
    if not valid_mask.any():
        return None

    valid_indices = np.where(valid_mask)[0]
    start_year = pd.to_datetime(times[valid_indices[0]]).year
    end_year = pd.to_datetime(times[valid_indices[-1]]).year

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    # Time / spatial / area filters
    if (use_eyear - use_syear + 1) < info.min_year:
        return None
    if not _valid_flow_in_year_window(valid_mask, times, use_syear, use_eyear):
        return None
    if lon_for_bounds < info.min_lon or lon_for_bounds > info.max_lon or lat < info.min_lat or lat > info.max_lat:
        return None
    if area > 0 and area < min_uparea:
        return None
    if area > 0 and area > max_uparea:
        return None

    file_path = station_file_path(scratch_dir, station_id, index=original_idx, duplicate_ids=duplicate_station_ids)
    clean_flow = np.where(valid_mask, np.asarray(flow, dtype=float), np.nan)
    ds_out = xr.Dataset({"discharge": (["time"], clean_flow)}, coords={"time": times})
    write_netcdf_atomic(ds_out, file_path)

    return [output_station_id, cama_lon, cama_lat, use_syear, use_eyear, str(file_path)]


# ---------------------------------------------------------------------------
# Direct matching (no CaMA)
# ---------------------------------------------------------------------------


def _process_site_direct(
    idx: int,
    station_ids: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    areas: np.ndarray,
    flow: np.ndarray,
    times: np.ndarray,
    info,
    scratch_dir: Path,
    min_uparea: float,
    max_uparea: float,
    output_station_ids: list[str],
    time_format: Optional[str] = None,
    original_indices: Optional[np.ndarray] = None,
    duplicate_station_ids: set[str] | None = None,
    missing_sentinels: tuple[float, ...] = (),
):
    """Process one station with direct coordinate matching (no CaMA)."""
    original_idx = int(original_indices[idx]) if original_indices is not None else idx
    station_id = _station_id_to_string(station_ids[idx])
    output_station_id = output_station_ids[idx]
    lon = float(lons[idx])
    lat = float(lats[idx])
    area = float(areas[idx]) if not np.isnan(areas[idx]) else -9999.0
    lon = _normalize_lon_to_range(lon, info.min_lon, info.max_lon)

    valid_mask = _valid_flow_mask(flow, missing_sentinels)
    if not valid_mask.any():
        return None

    valid_indices = np.where(valid_mask)[0]

    # Handle YYYYMM time format
    if time_format == "YYYYMM":
        time_vals = times
        start_year = int(str(int(time_vals[valid_indices[0]]))[:4])
        end_year = int(str(int(time_vals[valid_indices[-1]]))[:4])
    else:
        start_year = pd.to_datetime(times[valid_indices[0]]).year
        end_year = pd.to_datetime(times[valid_indices[-1]]).year

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    if (use_eyear - use_syear + 1) < info.min_year:
        return None
    if time_format == "YYYYMM":
        has_valid_window_flow = _valid_flow_in_yyyymm_window(valid_mask, times, use_syear, use_eyear)
    else:
        has_valid_window_flow = _valid_flow_in_year_window(valid_mask, times, use_syear, use_eyear)
    if not has_valid_window_flow:
        return None
    if lon < info.min_lon or lon > info.max_lon or lat < info.min_lat or lat > info.max_lat:
        return None
    if area > 0 and area < min_uparea:
        return None
    if area > 0 and area > max_uparea:
        return None

    file_path = station_file_path(scratch_dir, station_id, index=original_idx, duplicate_ids=duplicate_station_ids)

    if time_format == "YYYYMM":
        time_dates = pd.to_datetime([str(int(t)) for t in times], format="%Y%m")
        clean_flow = np.where(valid_mask, np.asarray(flow, dtype=float), np.nan)
        ds_out = xr.Dataset({"discharge": xr.DataArray(clean_flow, dims=["time"], coords={"time": time_dates})})
    else:
        clean_flow = np.where(valid_mask, np.asarray(flow, dtype=float), np.nan)
        ds_out = xr.Dataset({"discharge": (["time"], clean_flow)}, coords={"time": times})
    write_netcdf_atomic(ds_out, file_path)

    return [output_station_id, lon, lat, use_syear, use_eyear, str(file_path)]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_station_matching(
    info,
    dataset_path: str,
    method: str = "cama_allocation",
    station_id_var: str = "station",
    lon_var: str = "lon",
    lat_var: str = "lat",
    area_var: str = "area",
    discharge_var: str = "discharge",
    time_var: str = "time",
    station_dim: str = "",
    max_uparea: float = float("inf"),
    time_format: Optional[str] = None,
    scratch_subdir: Optional[str] = None,
    n_jobs: int | None = None,
):
    """Run station matching on a consolidated reference NC file.

    Supports two methods:
    - ``cama_allocation``: uses CaMA-Flood allocation data for grid matching
    - ``direct``: uses raw station coordinates

    The minimum upstream area comes from ``info.sim_grid_res`` via
    ``MIN_UPAREA_BY_RESOLUTION``; an unsupported resolution raises ValueError.
    The allocation error limit is ``MAX_CAMA_ALLOC_ERR``.

    Modifies ``info`` in-place: sets ``stn_list``, ``ref_fulllist``,
    ``use_syear``, ``use_eyear``.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Station dataset not found: {dataset_path}")
    min_uparea = resolution_min_uparea(getattr(info, "sim_grid_res", None))

    ref_name = scratch_subdir or dataset_path.stem
    scratch_dir = Path(info.casedir) / "scratch" / f"{ref_name}_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    info.min_uparea = min_uparea
    info.max_uparea = max_uparea

    with xr.open_dataset(dataset_path) as ds:
        station_id_key = _require_dataset_field(ds, station_id_var, "station_id_var", dataset_path)
        lon_key = _require_dataset_field(ds, lon_var, "lon_var", dataset_path)
        lat_key = _require_dataset_field(ds, lat_var, "lat_var", dataset_path)
        discharge_key = _require_dataset_field(ds, discharge_var, "discharge_var", dataset_path)
        time_key = get_xarray_key_case_insensitive(ds, time_var) or get_xarray_key_case_insensitive(ds, "time")
        if time_key is None:
            time_key = _require_dataset_field(ds, time_var, "time_var", dataset_path)
        station_ids = ds[station_id_key].values
        source_key = get_xarray_key_case_insensitive(ds, "data_source_name")
        data_source_names = ds[source_key].values if source_key else None
        lons = ds[lon_key].values
        lats = ds[lat_key].values

        # Area variable (optional — may not exist in all datasets)
        area_key = _station_area_key(ds, area_var, dataset_path, min_uparea)
        if area_key:
            areas = ds[area_key].values
        else:
            areas = np.full(len(station_ids), np.nan)

        times = ds[time_key].values

        n_stations = len(station_ids)
        discharge_da = ds[discharge_key]
        missing_sentinels = _as_missing_sentinels(discharge_da.attrs, discharge_da.encoding)
        worker_count = _station_matching_jobs(n_stations, n_jobs)
        normalized_station_ids = [_station_id_to_string(station_id) for station_id in station_ids]
        station_id_counts = Counter(normalized_station_ids)
        duplicate_station_ids = {station_id for station_id, count in station_id_counts.items() if count > 1}
        output_station_ids = _unique_station_ids(station_ids, data_source_names)
        time_indices = _target_time_indices(times, info, time_format)
        if time_indices.size == 0:
            raise ValueError(f"No {dataset_path.name} time steps overlap target years {_target_year_window(info)}")
        target_times = times[time_indices]

        if method == "cama_allocation":
            res_suffix = get_resolution_suffix(info.sim_grid_res)
            logging.info(
                "Station matching [cama]: %s (%d stations, CaMA %s, min upstream area %g km2)",
                dataset_path.name,
                n_stations,
                res_suffix,
                min_uparea,
            )

            cama_lon_var = f"cama_lon_{res_suffix}"
            cama_lat_var = f"cama_lat_{res_suffix}"
            alloc_err_var = f"cama_alloc_err_{res_suffix}"

            cama_lon_key = _require_dataset_field(ds, cama_lon_var, "cama_lon_var", dataset_path)
            cama_lat_key = _require_dataset_field(ds, cama_lat_var, "cama_lat_var", dataset_path)
            alloc_err_key = _require_dataset_field(ds, alloc_err_var, "alloc_err_var", dataset_path)

            cama_lons = ds[cama_lon_key].values
            cama_lats = ds[cama_lat_key].values
            alloc_errs = ds[alloc_err_key].values
            station_indices = _cama_candidate_station_indices(
                lons,
                lats,
                areas,
                cama_lons,
                cama_lats,
                alloc_errs,
                info,
                min_uparea,
                max_uparea,
            )
            if station_indices.size == 0:
                raise ValueError(f"No stations passed non-flow filters for {dataset_path.name}")
            logging.info(
                "Station matching candidate subset: %d/%d stations, %d/%d time steps",
                station_indices.size,
                n_stations,
                time_indices.size,
                len(times),
            )
            flow_data = _station_time_flow_values(
                discharge_da,
                station_dim=station_dim,
                time_key=time_key,
                n_stations=n_stations,
                station_indices=station_indices,
                time_indices=time_indices,
            )
            station_ids_subset = station_ids[station_indices]
            lons_subset = lons[station_indices]
            lats_subset = lats[station_indices]
            areas_subset = areas[station_indices]
            cama_lons_subset = cama_lons[station_indices]
            cama_lats_subset = cama_lats[station_indices]
            alloc_errs_subset = alloc_errs[station_indices]
            output_station_ids_subset = [output_station_ids[int(idx)] for idx in station_indices]

            rows = Parallel(n_jobs=worker_count, verbose=1, prefer="threads")(
                delayed(_process_site_cama)(
                    idx,
                    station_ids_subset,
                    lons_subset,
                    lats_subset,
                    areas_subset,
                    cama_lons_subset,
                    cama_lats_subset,
                    alloc_errs_subset,
                    flow_data[idx, :],
                    target_times,
                    info,
                    scratch_dir,
                    min_uparea,
                    max_uparea,
                    output_station_ids_subset,
                    station_indices,
                    duplicate_station_ids,
                    missing_sentinels,
                )
                for idx in range(station_indices.size)
            )

        elif method == "direct":
            logging.info(
                "Station matching [direct]: %s (%d stations, min upstream area %g km2)",
                dataset_path.name,
                n_stations,
                min_uparea,
            )
            station_indices = _direct_candidate_station_indices(lons, lats, areas, info, min_uparea, max_uparea)
            if station_indices.size == 0:
                raise ValueError(f"No stations passed non-flow filters for {dataset_path.name}")
            logging.info(
                "Station matching candidate subset: %d/%d stations, %d/%d time steps",
                station_indices.size,
                n_stations,
                time_indices.size,
                len(times),
            )
            flow_data = _station_time_flow_values(
                discharge_da,
                station_dim=station_dim,
                time_key=time_key,
                n_stations=n_stations,
                station_indices=station_indices,
                time_indices=time_indices,
            )
            station_ids_subset = station_ids[station_indices]
            lons_subset = lons[station_indices]
            lats_subset = lats[station_indices]
            areas_subset = areas[station_indices]
            output_station_ids_subset = [output_station_ids[int(idx)] for idx in station_indices]

            rows = Parallel(n_jobs=worker_count, verbose=1, prefer="threads")(
                delayed(_process_site_direct)(
                    idx,
                    station_ids_subset,
                    lons_subset,
                    lats_subset,
                    areas_subset,
                    flow_data[idx, :],
                    target_times,
                    info,
                    scratch_dir,
                    min_uparea,
                    max_uparea,
                    output_station_ids_subset,
                    time_format,
                    station_indices,
                    duplicate_station_ids,
                    missing_sentinels,
                )
                for idx in range(station_indices.size)
            )
        else:
            raise ValueError(f"Unknown station matching method: {method}")

    rows = [r for r in rows if r is not None]
    if not rows:
        raise ValueError(f"No stations passed filters for {dataset_path.name}")

    df = pd.DataFrame(rows, columns=["ID", "ref_lon", "ref_lat", "use_syear", "use_eyear", "ref_dir"])
    df["use_syear"] = df["use_syear"].astype(int)
    df["use_eyear"] = df["use_eyear"].astype(int)

    info.use_syear = int(df["use_syear"].min())
    info.use_eyear = int(df["use_eyear"].max())
    info.ref_fulllist = f"{info.casedir}/stn_{ref_name}_{info.sim_source}_list.txt"
    info.stn_list = df.copy()
    write_file_atomic(info.ref_fulllist, lambda tmp_path: df.to_csv(tmp_path, index=False), suffix=".tmp.csv")

    logging.info(
        "Station matching complete: %d stations, %d-%d",
        len(df),
        info.use_syear,
        info.use_eyear,
    )
