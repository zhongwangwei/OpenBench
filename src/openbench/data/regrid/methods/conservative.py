"""Conservative regridding implementation."""

import hashlib
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from typing import overload

import numpy as np
import pandas as pd
import xarray as xr

from .. import utils

REGRID_WEIGHT_SCHEMA_VERSION = 2
_WEIGHT_KEY = tuple[int, str, tuple[str, tuple[int, ...], str], tuple[str, tuple[int, ...], str]]

_WEIGHTS_CACHE_LOCK = threading.Lock()
_WEIGHTS_CACHE: OrderedDict[_WEIGHT_KEY, np.ndarray] = OrderedDict()
_WEIGHTS_CACHE_MAXSIZE = max(0, int(os.environ.get("OPENBENCH_REGRID_WEIGHT_CACHE_SIZE", "64")))
_WEIGHTS_DISK_CACHE_DIR = os.environ.get("OPENBENCH_REGRID_WEIGHT_CACHE_DIR")
_WEIGHTS_DISK_CACHE_MAX_MB_ENV = "OPENBENCH_REGRID_WEIGHT_CACHE_MAX_MB"
_WEIGHTS_DISK_CACHE_TTL_SECONDS_ENV = "OPENBENCH_REGRID_WEIGHT_CACHE_TTL_SECONDS"
_WEIGHTS_DISK_CACHE_TTL_DAYS_ENV = "OPENBENCH_REGRID_WEIGHT_CACHE_TTL_DAYS"
_SPHERICAL_CORRECTION_CACHE_LOCK = threading.Lock()
_SPHERICAL_CORRECTION_CACHE: OrderedDict[_WEIGHT_KEY, np.ndarray] = OrderedDict()
_SPHERICAL_CORRECTION_CACHE_MAXSIZE = max(0, int(os.environ.get("OPENBENCH_REGRID_SPHERICAL_CACHE_SIZE", "64")))
_WEIGHT_CACHE_ACTIVITY = threading.local()


def reset_weight_cache_activity() -> None:
    """Reset per-thread cache diagnostics for one regrid operation."""
    _WEIGHT_CACHE_ACTIVITY.events = []


def consume_weight_cache_activity() -> str:
    """Return cache outcomes recorded since the last reset."""
    events = list(getattr(_WEIGHT_CACHE_ACTIVITY, "events", []))
    _WEIGHT_CACHE_ACTIVITY.events = []
    return ",".join(events) if events else "unknown"


def _record_weight_cache_activity(event: str) -> None:
    events = getattr(_WEIGHT_CACHE_ACTIVITY, "events", None)
    if events is not None:
        events.append(event)


@overload
def conservative_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    time_dim: str | None = "time",
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.DataArray: ...


@overload
def conservative_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    time_dim: str | None = "time",
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.Dataset: ...


def conservative_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | Hashable | None,
    time_dim: str | None = "time",
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using conservative regridding.

    The method implementation is based on a post by Stephan Hoyer; "For the case of
    interpolation between rectilinear grids (even on the sphere), you can factorize
    regridding along each axis. This is less general but makes the entire calculation
    much simpler, because its feasible to store interpolation weights as dense matrices
    and to use dense matrix multiplication."
    https://discourse.pangeo.io/t/conservative-region-aggregation-with-xarray-geopandas-and-sparse/2715/3

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        latitude_coord: Name of the latitude coordinate. If not provided, attempt to
            infer it from the first coordinate equaling the string 'lat' or 'latitude'.
        skipna: If True, enable handling for NaN values. This adds some overhead,
            so should be disabled for optimal performance on data without NaNs.
        nan_threshold: Missing-data tolerance expressed in xESMF-style
            ``na_thres`` terms.  The default value is 1.0, which keeps output
            points containing any non-null inputs.  A value of 0.0 only keeps
            output points whose source-cell overlap is fully valid.  With
            ``skipna=True`` this computes an intensive area-weighted mean over
            valid overlap; it does not preserve extensive totals for regional
            integration workflows.
        output_chunks: Optional dictionary of explicit chunk sizes for the output data.
            If not provided, the output will be chunked the same as the input data.

    Returns:
        Regridded input dataset
    """
    # Attempt to infer the latitude coordinate
    if latitude_coord is None:
        for coord in data.coords:
            if str(coord).lower() in ["lat", "latitude"]:
                latitude_coord = coord
                break

    # Make sure the regridding coordinates are sorted
    # Exclude time dimension from regridding coordinates
    coord_names = [coord for coord in target_ds.coords if coord in data.coords and coord != time_dim]
    target_ds_sorted = xr.Dataset(coords=target_ds.coords)
    for coord_name in coord_names:
        target_ds_sorted = utils.ensure_monotonic(target_ds_sorted, coord_name)
        data = utils.ensure_monotonic(data, coord_name)
    coords = {name: target_ds_sorted[name] for name in coord_names}

    regridded_data = utils.call_on_dataset(
        conservative_regrid_dataset,
        data,
        coords,
        latitude_coord,
        skipna,
        nan_threshold,
        output_chunks,
        time_dim,
    )

    regridded_data = regridded_data.reindex_like(target_ds, copy=False)

    return regridded_data


def conservative_regrid_dataset(
    data: xr.Dataset,
    coords: dict[Hashable, xr.DataArray],
    latitude_coord: Hashable,
    skipna: bool,
    nan_threshold: float,
    output_chunks: dict[Hashable, int] | None,
    time_dim: str | None,
) -> xr.Dataset:
    """Dataset implementation of the conservative regridding method."""
    data_vars = dict(data.data_vars)
    data_coords = dict(data.coords)
    data_attrs = {v: data_vars[v].attrs for v in data_vars}
    coord_attrs = {c: data_coords[c].attrs for c in data_coords}
    ds_attrs = data.attrs

    # Create weights array and coverage mask for each regridding dim.
    # Coverage is derived from actual source/target cell overlap, not from
    # center-point min/max bounds.  That keeps regional edges with no source
    # support as NaN instead of allowing zero-filled dot-product artifacts.
    weights = {}
    covered = {}
    for coord in coords:
        target_coords = coords[coord].to_numpy()
        source_coords = data[coord].to_numpy()
        if coord == latitude_coord:
            nd_weights = get_weights(source_coords, target_coords, spherical=True)
        else:
            nd_weights = get_weights(source_coords, target_coords)

        raw_weights = utils.create_dot_dataarray(nd_weights, str(coord), target_coords, source_coords)

        target_dim = f"target_{coord}"
        coverage = raw_weights.sum(dim=str(coord)) > 0
        if target_dim in coverage.dims:
            coverage = coverage.rename({target_dim: coord})
        covered[coord] = coverage
        weights[coord] = raw_weights

    for array in data_vars.keys():
        var_weights = {}
        for coord, weight_array in weights.items():
            var_input_chunks = data_vars[array].chunksizes.get(coord)
            var_output_chunks = output_chunks.get(coord) if output_chunks else None
            var_weights[coord] = format_weights(
                weight_array,
                coord,
                data_vars[array].dtype,
                var_input_chunks,
                var_output_chunks,
            )

        data_vars[array] = apply_weights(
            da=data_vars[array],
            weights=var_weights,
            skipna=skipna,
            nan_threshold=nan_threshold,
        )
        # Mask out any regridded points outside the original domain
        # Limit to dims present on this array otherwise .where broadcasts
        var_covered = xr.DataArray(True)
        for coord in var_weights.keys():
            var_covered = var_covered & covered[coord]
        data_vars[array] = data_vars[array].where(var_covered)

    for array, attrs in data_attrs.items():
        data_vars[array].attrs = attrs

    ds_regridded = xr.Dataset(data_vars=data_vars, attrs=ds_attrs)

    for coord, attrs in coord_attrs.items():
        if coord not in ds_regridded.coords:
            ds_regridded[coord] = data_coords[coord]
        ds_regridded[coord].attrs = attrs

    return ds_regridded


def apply_weights(
    da: xr.DataArray,
    weights: dict[Hashable, xr.DataArray],
    skipna: bool,
    nan_threshold: float,
    *,
    use_sparse: bool | None = None,
) -> xr.DataArray:
    """Apply weights as an intensive area-weighted mean.

    When ``skipna`` is true, invalid source cells are excluded and the weighted
    sum is divided by valid coverage.  This preserves the mean of an intensive
    field over the valid overlap; callers that need extensive total
    conservation must multiply by cell areas before/after regridding and define
    explicit missing-data policy. ``use_sparse=True`` requests the experimental
    SciPy CSR contraction; the production default retains dense ``xr.dot``.
    """
    # SciPy CSR materially reduces zero-weight work, but benchmarks show that
    # optimized BLAS can still be faster depending on grid shape and time
    # length. Keep dense as the production default until a stable crossover is
    # established; the explicit path remains available to the benchmark/tests.
    prefer_sparse = use_sparse is True
    if prefer_sparse and da.chunks is None:
        try:
            result = _apply_weights_sparse(da, weights, skipna, nan_threshold)
            logging.debug("Conservative regrid contraction backend: scipy.sparse")
            return result
        except (ImportError, TypeError, ValueError, RuntimeError, NotImplementedError) as exc:
            logging.debug("Conservative sparse contraction unavailable; using dense xr.dot: %s", exc)
    elif prefer_sparse:
        # xr.dot cannot reliably combine pydata/sparse weights with dask-backed
        # arrays across supported xarray versions. Keep the established path.
        logging.debug("Conservative regrid contraction backend: dense xr.dot (dask input)")

    if use_sparse is None:
        logging.debug("Conservative regrid contraction backend: dense xr.dot (default)")
    else:
        logging.debug("Conservative regrid contraction backend: dense xr.dot")
    return _apply_weights_dense(da, weights, skipna, nan_threshold)


def _apply_weights_dense(
    da: xr.DataArray,
    weights: dict[Hashable, xr.DataArray],
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray:
    """Established dense contraction retained as the compatibility fallback."""
    coords = list(weights.keys())
    weight_arrays = list(weights.values())

    if skipna:
        valid_frac = xr.dot(da.notnull(), *weight_arrays, dim=list(weights.keys()), optimize=True)

    da_regrid: xr.DataArray = xr.dot(da.fillna(0), *weight_arrays, dim=list(weights.keys()), optimize=True)

    if skipna:
        da_regrid /= valid_frac
        da_regrid = da_regrid.where(valid_frac >= get_valid_threshold(nan_threshold))

    coord_map = {f"target_{coord}": coord for coord in coords}
    da_regrid = da_regrid.rename(coord_map).transpose(*da.dims)

    return da_regrid


def _sparse_contract_axis(da: xr.DataArray, weight: xr.DataArray, coord: Hashable, matrix) -> xr.DataArray:
    """Contract one rectilinear axis with a SciPy CSR weight matrix."""
    if coord not in da.dims:
        raise ValueError(f"Data variable does not contain regrid dimension {coord!r}")
    target_dim = f"target_{coord}"
    if tuple(weight.dims) != (coord, target_dim):
        raise ValueError(f"Unexpected weight dimensions for {coord!r}: {weight.dims!r}")

    if matrix.shape[0] != da.sizes[coord]:
        raise ValueError(f"Weight/source size mismatch for {coord!r}")
    logging.debug(
        "Sparse conservative weights axis=%s shape=%s nnz=%d density=%.6f",
        coord,
        matrix.shape,
        matrix.nnz,
        matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1]),
    )

    output_dtype = np.result_type(da.dtype, matrix.dtype)

    def contract(values: np.ndarray) -> np.ndarray:
        flat = values.reshape((-1, values.shape[-1]))
        contracted = matrix.transpose().dot(flat.transpose()).transpose()
        return np.asarray(contracted).reshape(values.shape[:-1] + (matrix.shape[1],))

    result = xr.apply_ufunc(
        contract,
        da,
        input_core_dims=[[coord]],
        output_core_dims=[[target_dim]],
        dask="forbidden",
        output_dtypes=[output_dtype],
    )
    return result.assign_coords({target_dim: weight[target_dim]})


def _apply_weights_sparse(
    da: xr.DataArray,
    weights: dict[Hashable, xr.DataArray],
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray:
    """Apply separable rectilinear weights without multiplying stored zeros."""
    from scipy import sparse as scipy_sparse

    coords = list(weights.keys())
    sparse_weights = {
        coord: (weight, scipy_sparse.csr_matrix(np.asarray(weight.values))) for coord, weight in weights.items()
    }

    def contract_all(value: xr.DataArray) -> xr.DataArray:
        for coord, (weight, matrix) in sparse_weights.items():
            value = _sparse_contract_axis(value, weight, coord, matrix)
        return value

    if skipna:
        valid_frac = contract_all(da.notnull())
    da_regrid = contract_all(da.fillna(0))

    if skipna:
        da_regrid /= valid_frac
        da_regrid = da_regrid.where(valid_frac >= get_valid_threshold(nan_threshold))

    coord_map = {f"target_{coord}": coord for coord in coords}
    return da_regrid.rename(coord_map).transpose(*da.dims)


def get_valid_threshold(nan_threshold: float) -> float:
    """Invert the nan_threshold and coerce it to just above zero and below
    one to handle numerical precision limitations in the weight sum."""
    # Convert xESMF-style missing tolerance to the minimum valid overlap.  A
    # zero tolerance means the target cell must be fully covered by valid input.
    valid_threshold: float = 1 - np.clip(nan_threshold, 1e-6, 1.0 - 1e-6)
    return valid_threshold


def get_weights(source_coords: np.ndarray, target_coords: np.ndarray, *, spherical: bool = False) -> np.ndarray:
    """Determine the weights to map from the old coordinates to the new coordinates.

    Args:
        source_coords: Source coordinates (center points)
        target_coords Target coordinates (center points)

    Returns:
        Weights, which can be used with a dot product to apply the conservative regrid.
    """
    source_coords = np.asarray(source_coords)
    target_coords = np.asarray(target_coords)
    if source_coords.size == 0 or target_coords.size == 0:
        raise ValueError("Conservative regridding requires non-empty source and target coordinates")
    if source_coords.size == target_coords.size == 1:
        if np.isclose(source_coords[0], target_coords[0]):
            _record_weight_cache_activity("identity")
            return np.array([[1.0]], dtype=float)
        raise ValueError(
            "Conservative regridding cannot infer finite cell bounds for a non-identical single-point coordinate"
        )
    if source_coords.size == 1 or target_coords.size == 1:
        raise ValueError("Conservative regridding requires explicit cell bounds for one-sided singleton coordinates")
    mode = "spherical" if spherical else "linear"
    key = (REGRID_WEIGHT_SCHEMA_VERSION, mode, _coord_cache_token(source_coords), _coord_cache_token(target_coords))
    if _WEIGHTS_CACHE_MAXSIZE:
        with _WEIGHTS_CACHE_LOCK:
            cached = _WEIGHTS_CACHE.get(key)
            if cached is not None:
                _WEIGHTS_CACHE.move_to_end(key)
                _record_weight_cache_activity("memory")
                return cached

    disk_cached = _load_weights_from_disk(key)
    if disk_cached is not None:
        _remember_weight(key, disk_cached)
        _record_weight_cache_activity("disk")
        return disk_cached

    target_intervals = utils.to_intervalindex(target_coords)
    source_intervals = utils.to_intervalindex(source_coords)
    if spherical:
        overlap = spherical_overlap(source_intervals, target_intervals)
    else:
        overlap = utils.overlap(source_intervals, target_intervals)
    weights = utils.normalize_overlap(overlap)
    weights.setflags(write=False)
    _store_weights_to_disk(key, weights)
    existing = _remember_weight(key, weights)
    _record_weight_cache_activity("miss")
    return existing if existing is not None else weights


def spherical_overlap(a: pd.IntervalIndex, b: pd.IntervalIndex) -> np.ndarray:
    """Spherical latitude overlap area, proportional to sin(phi_hi)-sin(phi_lo)."""
    a_left = a.left.to_numpy()
    a_right = a.right.to_numpy()
    b_left = b.left.to_numpy()
    b_right = b.right.to_numpy()
    result = np.zeros((len(a), len(b)), dtype=float)
    if (
        a.is_non_overlapping_monotonic
        and b.is_non_overlapping_monotonic
        and a.is_monotonic_increasing
        and b.is_monotonic_increasing
    ):
        source = target = 0
        while source < len(a) and target < len(b):
            lower = np.clip(max(a_left[source], b_left[target]), -90.0, 90.0)
            source_right = a_right[source]
            target_right = b_right[target]
            upper = np.clip(min(source_right, target_right), -90.0, 90.0)
            if upper > lower:
                result[source, target] = np.sin(np.radians(upper)) - np.sin(np.radians(lower))
            if source_right <= target_right:
                source += 1
            if target_right <= source_right:
                target += 1
        return result

    for column, (left, right) in enumerate(zip(b_left, b_right)):
        lower = np.clip(np.maximum(a_left, left), -90.0, 90.0)
        upper = np.clip(np.minimum(a_right, right), -90.0, 90.0)
        values = result[:, column]
        np.subtract(np.sin(np.radians(upper)), np.sin(np.radians(lower)), out=values)
        np.maximum(values, 0, out=values)
    return result


def _coord_cache_token(coords: np.ndarray) -> tuple[str, tuple[int, ...], str]:
    """Return a stable cache token for a coordinate vector."""
    contiguous = np.ascontiguousarray(coords)
    digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
    return contiguous.dtype.str, tuple(contiguous.shape), digest


def _remember_weight(key: _WEIGHT_KEY, weights: np.ndarray) -> np.ndarray | None:
    if not _WEIGHTS_CACHE_MAXSIZE:
        return None
    with _WEIGHTS_CACHE_LOCK:
        existing = _WEIGHTS_CACHE.get(key)
        if existing is not None:
            _WEIGHTS_CACHE.move_to_end(key)
            return existing
        _WEIGHTS_CACHE[key] = weights
        while len(_WEIGHTS_CACHE) > _WEIGHTS_CACHE_MAXSIZE:
            _WEIGHTS_CACHE.popitem(last=False)
    return None


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _weights_disk_cache_max_bytes() -> int | None:
    max_mb = _env_optional_float(_WEIGHTS_DISK_CACHE_MAX_MB_ENV)
    if max_mb is None:
        return None
    return int(max_mb * 1024 * 1024)


def _weights_disk_cache_ttl_seconds() -> float | None:
    seconds = _env_optional_float(_WEIGHTS_DISK_CACHE_TTL_SECONDS_ENV)
    if seconds is not None:
        return seconds
    days = _env_optional_float(_WEIGHTS_DISK_CACHE_TTL_DAYS_ENV)
    if days is None:
        return None
    return days * 86400.0


def _iter_weight_cache_files(cache_dir: str) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(cache_dir) as entries:
            return [
                entry
                for entry in entries
                if entry.is_file() and entry.name.startswith("weights-") and entry.name.endswith(".npz")
            ]
    except OSError:
        return []


def prune_weight_disk_cache(
    cache_dir: str | None = None,
    *,
    max_bytes: int | None = None,
    ttl_seconds: float | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Prune conservative regrid disk-cache files by age and total size.

    Args:
        cache_dir: Cache directory. Defaults to OPENBENCH_REGRID_WEIGHT_CACHE_DIR.
        max_bytes: Keep newest files while total size is above this limit.
            ``None`` uses OPENBENCH_REGRID_WEIGHT_CACHE_MAX_MB; ``0`` removes all.
        ttl_seconds: Remove files older than this many seconds. ``None`` uses
            OPENBENCH_REGRID_WEIGHT_CACHE_TTL_SECONDS / _TTL_DAYS.
        now: Test hook for age calculations.

    Returns:
        Counts and byte totals for CLI/status reporting.
    """
    cache_dir = cache_dir or _weights_disk_cache_dir()
    if not cache_dir or not os.path.isdir(cache_dir):
        return {"files": 0, "bytes": 0, "removed_files": 0, "removed_bytes": 0}

    max_bytes = _weights_disk_cache_max_bytes() if max_bytes is None else max_bytes
    ttl_seconds = _weights_disk_cache_ttl_seconds() if ttl_seconds is None else ttl_seconds
    now = time.time() if now is None else now

    files: list[tuple[str, float, int]] = []
    for entry in _iter_weight_cache_files(cache_dir):
        try:
            stat = entry.stat()
        except OSError:
            continue
        files.append((entry.path, stat.st_mtime, stat.st_size))

    removed_files = 0
    removed_bytes = 0

    def _remove(path: str, size: int) -> bool:
        nonlocal removed_files, removed_bytes
        try:
            os.remove(path)
        except OSError:
            return False
        removed_files += 1
        removed_bytes += size
        return True

    remaining: list[tuple[str, float, int]] = []
    for path, mtime, size in files:
        if ttl_seconds is not None and ttl_seconds >= 0 and now - mtime > ttl_seconds:
            _remove(path, size)
        else:
            remaining.append((path, mtime, size))

    total_bytes = sum(size for _path, _mtime, size in remaining)
    if max_bytes is not None and max_bytes >= 0 and total_bytes > max_bytes:
        for path, _mtime, size in sorted(remaining, key=lambda item: item[1]):
            if total_bytes <= max_bytes:
                break
            if _remove(path, size):
                total_bytes -= size

    final_files = _iter_weight_cache_files(cache_dir)
    final_bytes = 0
    for entry in final_files:
        try:
            final_bytes += entry.stat().st_size
        except OSError:
            pass
    return {
        "files": len(final_files),
        "bytes": final_bytes,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
    }


def _weights_disk_cache_dir() -> str | None:
    return os.environ.get("OPENBENCH_REGRID_WEIGHT_CACHE_DIR") or _WEIGHTS_DISK_CACHE_DIR


def _weights_disk_cache_path(key: _WEIGHT_KEY) -> str | None:
    cache_dir = _weights_disk_cache_dir()
    if not cache_dir:
        return None
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"weights-{digest}.npz")


def _load_weights_from_disk(key: _WEIGHT_KEY) -> np.ndarray | None:
    path = _weights_disk_cache_path(key)
    if path is None or not os.path.exists(path):
        return None
    ttl_seconds = _weights_disk_cache_ttl_seconds()
    if ttl_seconds is not None and ttl_seconds >= 0:
        try:
            if time.time() - os.path.getmtime(path) > ttl_seconds:
                os.remove(path)
                return None
        except OSError:
            return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (
                "schema_version" not in data.files
                or int(np.asarray(data["schema_version"])) != REGRID_WEIGHT_SCHEMA_VERSION
            ):
                return None
            weights = np.asarray(data["weights"])
        if weights.shape != (key[2][1][0], key[3][1][0]) or not np.isfinite(weights).all():
            return None
        column_sums = weights.sum(axis=0)
        if not np.all((np.isclose(column_sums, 1.0)) | (np.isclose(column_sums, 0.0))):
            return None
        weights.setflags(write=False)
        return weights
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _store_weights_to_disk(key: _WEIGHT_KEY, weights: np.ndarray) -> None:
    path = _weights_disk_cache_path(key)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".weights-", suffix=".npz", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, weights=weights, schema_version=np.array(REGRID_WEIGHT_SCHEMA_VERSION))
        os.replace(tmp_path, path)
        prune_weight_disk_cache(os.path.dirname(path))
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def clear_weight_cache(*, clear_disk: bool = False, cache_dir: str | None = None) -> None:
    """Clear conservative regrid weight caches; primarily useful for tests and CLI."""
    with _WEIGHTS_CACHE_LOCK:
        _WEIGHTS_CACHE.clear()
    with _SPHERICAL_CORRECTION_CACHE_LOCK:
        _SPHERICAL_CORRECTION_CACHE.clear()
    if clear_disk:
        target_dir = cache_dir or _weights_disk_cache_dir()
        if target_dir and os.path.isdir(target_dir):
            for name in os.listdir(target_dir):
                if name.startswith("weights-") and name.endswith(".npz"):
                    try:
                        os.remove(os.path.join(target_dir, name))
                    except OSError:
                        pass


def apply_spherical_correction(dot_array: xr.DataArray, latitude_coord: Hashable) -> xr.DataArray:
    """Apply a spherical earth correction on the prepared dot product weights."""
    da = dot_array.copy()
    key = (_coord_cache_token(dot_array[latitude_coord].to_numpy()), _coord_cache_token(dot_array.values))
    if _SPHERICAL_CORRECTION_CACHE_MAXSIZE:
        with _SPHERICAL_CORRECTION_CACHE_LOCK:
            cached = _SPHERICAL_CORRECTION_CACHE.get(key)
            if cached is not None:
                _SPHERICAL_CORRECTION_CACHE.move_to_end(key)
                da.values = cached
                return da

    latitude_values = dot_array[latitude_coord].to_numpy()
    latitude_res = np.median(np.abs(np.diff(latitude_values, 1))) if latitude_values.size > 1 else 1.0
    lat_weights = lat_weight(latitude_values, latitude_res)
    corrected = utils.normalize_overlap(dot_array.values * lat_weights[:, np.newaxis])
    corrected.setflags(write=False)
    if _SPHERICAL_CORRECTION_CACHE_MAXSIZE:
        with _SPHERICAL_CORRECTION_CACHE_LOCK:
            existing = _SPHERICAL_CORRECTION_CACHE.get(key)
            if existing is not None:
                _SPHERICAL_CORRECTION_CACHE.move_to_end(key)
                da.values = existing
                return da
            _SPHERICAL_CORRECTION_CACHE[key] = corrected
            while len(_SPHERICAL_CORRECTION_CACHE) > _SPHERICAL_CORRECTION_CACHE_MAXSIZE:
                _SPHERICAL_CORRECTION_CACHE.popitem(last=False)
    da.values = corrected
    return da


def lat_weight(latitude: np.ndarray, latitude_res: float) -> np.ndarray:
    """Return the weight of gridcells based on their latitude.

    Args:
        latitude: (Center) latitude values of the gridcells, in degrees.
        latitude_res: Resolution/width of the grid cells, in degrees.

    Returns:
        Weights, same shape as latitude input.
    """
    latitude = np.asarray(latitude, dtype=float)
    if latitude.size == 1:
        half_width = abs(float(latitude_res)) / 2.0
        bounds = np.array([latitude[0] - half_width, latitude[0] + half_width])
    else:
        midpoints = (latitude[:-1] + latitude[1:]) / 2.0
        bounds = np.concatenate(
            ([latitude[0] - (midpoints[0] - latitude[0])], midpoints, [latitude[-1] + (latitude[-1] - midpoints[-1])])
        )
    bounds = np.clip(bounds, -90.0, 90.0)
    lower = np.minimum(bounds[:-1], bounds[1:])
    upper = np.maximum(bounds[:-1], bounds[1:])
    width = np.radians(upper - lower)
    spherical_width = np.sin(np.radians(upper)) - np.sin(np.radians(lower))
    return np.divide(spherical_width, width, out=np.zeros_like(spherical_width), where=width > 0)


def format_weights(
    weights: xr.DataArray,
    coord: Hashable,
    input_dtype: np.dtype,
    input_chunks: tuple[int, ...] | None,
    output_chunks: tuple[int, ...] | int | None,
) -> xr.DataArray:
    """Format the raw weights array such that:

    1. Weights match the dtype of the input data
    1. Weights are chunked 1:1 with the source data
    2. Weights are chunked as requested in the target grid. If no chunks are
        provided, the same chunksize as the source grid will be used.
        See: https://github.com/dask/dask/issues/2225
    The explicit sparse apply path converts these arrays to SciPy CSR matrices
    immediately before contraction. Dask-backed inputs retain dense xr.dot
    because xarray's sparse/dask dispatch is not stable across the supported
    dependency range.
    """
    # Use single precision weights at minimum, double if input is double
    weights_dtype = np.result_type(np.float32, input_dtype)
    new_weights = weights.copy().astype(weights_dtype)

    chunks: dict[Hashable, tuple[int, ...] | int] = {}
    if input_chunks is not None:
        chunks[coord] = input_chunks
        if output_chunks is None:
            # Set output chunking to match input, but precise chunks won't match shape,
            # so take the max in case of uneven chunks
            output_chunks = max(input_chunks)

    if output_chunks is not None:
        chunks[f"target_{coord}"] = output_chunks

    if chunks:
        new_weights = new_weights.chunk(chunks)
    return new_weights
