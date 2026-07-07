"""Generic station matching engine using CaMA-Flood allocation data.

Replaces 19 individual station filter files that share identical logic.
Configured via ``station_matching`` block in reference_catalog.yaml.

Supported matching methods:
- ``cama_allocation``: match stations to grid cells using CaMA allocation
  data, filter by area error and upstream area bounds.
- ``direct``: use raw station coordinates (no CaMA allocation), e.g. for
  coastal discharge datasets like Dai & Trenberth.
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

from openbench.util.exceptions import DataProcessingError
from openbench.util.filenames import station_file_path
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
    logging.warning("Unknown resolution %s, defaulting to 03min", sim_grid_res)
    return "03min"


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


def _get_dim_case_insensitive(dims, requested: str | None) -> str | None:
    if not requested:
        return None
    requested_norm = requested.lower()
    for dim in dims:
        if str(dim).lower() == requested_norm:
            return dim
    return None


def _station_time_flow_values(discharge_da: xr.DataArray, *, station_dim: str, time_key: str, n_stations: int):
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
    return discharge_da.transpose(station_dim_key, time_dim).values


def _as_missing_sentinels(attrs: dict | None) -> tuple[float, ...]:
    """Return numeric missing-value sentinels advertised by a data variable."""
    sentinels = {-999.0}
    for key in ("_FillValue", "missing_value"):
        value = (attrs or {}).get(key)
        values = np.ravel(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
        for item in values:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                sentinels.add(numeric)
    return tuple(sentinels)


def _valid_flow_mask(flow: np.ndarray, missing_sentinels: tuple[float, ...] = ()) -> np.ndarray:
    """Mask finite/non-sentinel flow values consistently across matching methods."""
    values = np.asarray(flow)
    try:
        mask = ~np.isnan(values.astype(float, copy=False))
    except (TypeError, ValueError):
        mask = np.ones(values.shape, dtype=bool)
    for sentinel in missing_sentinels:
        try:
            mask &= values.astype(float, copy=False) != sentinel
        except (TypeError, ValueError):
            mask &= values != sentinel
    return mask


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
    cpu_count = os.cpu_count() or 1
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
    area_err_threshold: float,
    min_uparea: float,
    max_uparea: float,
    duplicate_station_ids: set[str] | None = None,
    missing_sentinels: tuple[float, ...] = (),
):
    """Process one station for CaMA allocation matching.  Returns metadata row or None."""
    station_id = _station_id_to_string(station_ids[idx])
    lon = float(lons[idx])
    lat = float(lats[idx])
    area = float(areas[idx]) if not np.isnan(areas[idx]) else -9999.0

    cama_lon = float(cama_lons[idx])
    cama_lat = float(cama_lats[idx])
    alloc_err = float(alloc_errs[idx])
    lon_for_bounds = _normalize_lon_to_range(lon, info.min_lon, info.max_lon)
    cama_lon = _normalize_lon_to_range(cama_lon, -180.0, 180.0)

    # Skip invalid CaMA allocations
    if np.isnan(cama_lon) or np.isnan(cama_lat) or cama_lon < -180 or cama_lon > 180 or cama_lat < -90 or cama_lat > 90:
        return None

    # Area error filter
    if not np.isnan(alloc_err) and alloc_err > area_err_threshold:
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
    if lon_for_bounds < info.min_lon or lon_for_bounds > info.max_lon or lat < info.min_lat or lat > info.max_lat:
        return None
    if area > 0 and area < min_uparea:
        return None
    if area > 0 and area > max_uparea:
        return None

    file_path = station_file_path(scratch_dir, station_id, index=idx, duplicate_ids=duplicate_station_ids)
    ds_out = xr.Dataset({"discharge": (["time"], flow)}, coords={"time": times})
    write_netcdf_atomic(ds_out, file_path)

    return [station_id, cama_lon, cama_lat, use_syear, use_eyear, str(file_path)]


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
    time_format: Optional[str] = None,
    duplicate_station_ids: set[str] | None = None,
    missing_sentinels: tuple[float, ...] = (),
):
    """Process one station with direct coordinate matching (no CaMA)."""
    station_id = _station_id_to_string(station_ids[idx])
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
    if lon < info.min_lon or lon > info.max_lon or lat < info.min_lat or lat > info.max_lat:
        return None
    if area > 0 and area < min_uparea:
        return None
    if area > 0 and area > max_uparea:
        return None

    file_path = station_file_path(scratch_dir, station_id, index=idx, duplicate_ids=duplicate_station_ids)

    # Build time coordinate
    if time_format == "YYYYMM":
        time_dates = pd.to_datetime([str(int(t)) for t in times], format="%Y%m")
        ds_out = xr.Dataset({"discharge": xr.DataArray(flow, dims=["time"], coords={"time": time_dates})})
    else:
        ds_out = xr.Dataset({"discharge": (["time"], flow)}, coords={"time": times})
    write_netcdf_atomic(ds_out, file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


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
    area_error_threshold: float = 0.2,
    min_uparea: float = 1000.0,
    max_uparea: float = float("inf"),
    time_format: Optional[str] = None,
    scratch_subdir: Optional[str] = None,
    n_jobs: int | None = None,
):
    """Run station matching on a consolidated reference NC file.

    Supports two methods:
    - ``cama_allocation``: uses CaMA-Flood allocation data for grid matching
    - ``direct``: uses raw station coordinates

    Modifies ``info`` in-place: sets ``stn_list``, ``ref_fulllist``,
    ``use_syear``, ``use_eyear``.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Station dataset not found: {dataset_path}")

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
        lons = ds[lon_key].values
        lats = ds[lat_key].values

        # Area variable (optional — may not exist in all datasets)
        area_key = get_xarray_key_case_insensitive(ds, area_var) if area_var else None
        if area_key:
            areas = ds[area_key].values
        else:
            areas = np.full(len(station_ids), np.nan)

        times = ds[time_key].values

        n_stations = len(station_ids)
        discharge_da = ds[discharge_key]
        missing_sentinels = _as_missing_sentinels(discharge_da.attrs)
        flow_data = _station_time_flow_values(
            discharge_da,
            station_dim=station_dim,
            time_key=time_key,
            n_stations=n_stations,
        )
        worker_count = _station_matching_jobs(n_stations, n_jobs)
        normalized_station_ids = [_station_id_to_string(station_id) for station_id in station_ids]
        station_id_counts = Counter(normalized_station_ids)
        duplicate_station_ids = {station_id for station_id, count in station_id_counts.items() if count > 1}

        if method == "cama_allocation":
            res_suffix = get_resolution_suffix(info.sim_grid_res)
            logging.info(
                "Station matching [cama]: %s (%d stations, CaMA %s)",
                dataset_path.name,
                n_stations,
                res_suffix,
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

            rows = Parallel(n_jobs=worker_count, verbose=1, prefer="threads")(
                delayed(_process_site_cama)(
                    idx,
                    station_ids,
                    lons,
                    lats,
                    areas,
                    cama_lons,
                    cama_lats,
                    alloc_errs,
                    flow_data[idx, :],
                    times,
                    info,
                    scratch_dir,
                    area_error_threshold,
                    min_uparea,
                    max_uparea,
                    duplicate_station_ids,
                    missing_sentinels,
                )
                for idx in range(n_stations)
            )

        elif method == "direct":
            logging.info(
                "Station matching [direct]: %s (%d stations)",
                dataset_path.name,
                n_stations,
            )
            rows = Parallel(n_jobs=worker_count, verbose=1, prefer="threads")(
                delayed(_process_site_direct)(
                    idx,
                    station_ids,
                    lons,
                    lats,
                    areas,
                    flow_data[idx, :],
                    times,
                    info,
                    scratch_dir,
                    min_uparea,
                    max_uparea,
                    time_format,
                    duplicate_station_ids,
                    missing_sentinels,
                )
                for idx in range(n_stations)
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
