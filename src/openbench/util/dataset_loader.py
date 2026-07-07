# -*- coding: utf-8 -*-
"""
Dataset loading utilities with automatic chunking and glob caching.

This module provides optimized dataset loading functions that use Dask chunking
to reduce memory usage and improve performance for large NetCDF files.
It also provides cached glob operations to avoid repeated directory scans.
"""

import glob as _glob
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import xarray as xr

from openbench.util.netcdf import write_netcdf_atomic

# =============================================================================
# Glob Caching
# =============================================================================

# Global cache for glob results
# Format: {pattern: (results, timestamp)}
# All read/write/clear operations are guarded by _glob_cache_lock — without
# the lock, concurrent threads could observe a half-mutated dict during
# clear_glob_cache() iteration or trigger size-changed-during-iteration.
_glob_cache: Dict[str, tuple] = {}
_glob_cache_lock = threading.Lock()

# Cache expiry time in seconds (default: 5 minutes)
GLOB_CACHE_TTL = 300


def cached_glob(pattern: str, ttl: int = GLOB_CACHE_TTL, force_refresh: bool = False) -> List[str]:
    """
    Cached version of glob.glob() to avoid repeated directory scans.

    Parameters
    ----------
    pattern : str
        Glob pattern to match files
    ttl : int
        Cache time-to-live in seconds (default: 300s = 5 minutes)
    force_refresh : bool
        Force refresh the cache for this pattern

    Returns
    -------
    list
        List of matching file paths (sorted)

    Examples
    --------
    >>> files = cached_glob("/data/2020/*.nc")
    >>> files = cached_glob("/data/2020/*.nc", force_refresh=True)
    """
    global _glob_cache

    current_time = time.time()

    # Check cache under lock
    if not force_refresh:
        with _glob_cache_lock:
            entry = _glob_cache.get(pattern)
        if entry is not None:
            results, timestamp = entry
            if current_time - timestamp < ttl:
                logging.debug(f"Glob cache hit: {pattern} ({len(results)} files)")
                return results

    # Cache miss or expired — perform actual glob outside lock (slow I/O)
    results = sorted(_glob.glob(pattern))
    with _glob_cache_lock:
        _glob_cache[pattern] = (results, current_time)
    logging.debug(f"Glob cache miss: {pattern} ({len(results)} files)")

    return results


def clear_glob_cache(pattern: Optional[str] = None):
    """
    Clear the glob cache.

    Parameters
    ----------
    pattern : str, optional
        Specific pattern to clear. If None, clears entire cache.
    """
    global _glob_cache

    with _glob_cache_lock:
        if pattern is None:
            _glob_cache.clear()
            logging.debug("Glob cache cleared entirely")
        elif pattern in _glob_cache:
            del _glob_cache[pattern]
            logging.debug(f"Glob cache cleared for: {pattern}")


def get_glob_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the glob cache.

    Returns
    -------
    dict
        Cache statistics including size and patterns
    """
    with _glob_cache_lock:
        return {
            "size": len(_glob_cache),
            "patterns": list(_glob_cache.keys()),
            "total_files_cached": sum(len(r[0]) for r in _glob_cache.values()),
        }


# =============================================================================
# Dataset Loading with Chunking
# =============================================================================


# Default chunk sizes optimized for typical climate/earth science data
DEFAULT_CHUNKS = {
    "time": 12,  # Monthly data: 1 year per chunk
    "lat": 500,  # Spatial chunks
    "lon": 500,
    "latitude": 500,
    "longitude": 500,
    "y": 500,
    "x": 500,
}

# Size threshold for enabling chunking (bytes)
# Files smaller than this are loaded directly into memory
CHUNK_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

# Explicit batch size for large multi-file combines. If unset, OpenBench uses a
# lightweight resource plan to batch only very large file lists.
MFDATASET_BATCH_SIZE_ENV = "OPENBENCH_MFDATASET_BATCH_SIZE"
MFDATASET_AUTO_BATCH_MIN_FILES_ENV = "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES"
MFDATASET_AUTO_BATCH_MIN_SIZE_MB_ENV = "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE_MB"
MFDATASET_AUTO_BATCH_MAX_SIZE_ENV = "OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE"
MFDATASET_AUTO_BATCH_MIN_SIZE_ENV = "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE"
MFDATASET_AUTO_BATCH_MEMORY_FRACTION_ENV = "OPENBENCH_MFDATASET_AUTO_BATCH_MEMORY_FRACTION"

DEFAULT_MFDATASET_AUTO_BATCH_MIN_FILES = 200
DEFAULT_MFDATASET_AUTO_BATCH_MIN_SIZE_MB = 1024
DEFAULT_MFDATASET_AUTO_BATCH_MAX_SIZE = 100
DEFAULT_MFDATASET_AUTO_BATCH_MIN_SIZE = 10
DEFAULT_MFDATASET_AUTO_BATCH_MEMORY_FRACTION = 0.25


@dataclass(frozen=True)
class ResourcePlan:
    """Lightweight plan for memory-sensitive multi-file dataset operations."""

    file_count: int
    total_size_bytes: int
    available_memory_bytes: int | None
    mfdataset_batch_size: int
    reason: str


def _open_dataset_with_fallback(path: str, **kwargs) -> xr.Dataset:
    """
    Open a NetCDF dataset with fallback to decode_times=False if initial open fails.

    Parameters
    ----------
    path : str
        Path to the NetCDF file
    **kwargs
        Additional arguments passed to xr.open_dataset()

    Returns
    -------
    xr.Dataset
        The opened dataset
    """
    try:
        return xr.open_dataset(path, **kwargs)
    except Exception as e:
        # Retry with decode_times=False unless caller explicitly disabled it.
        # Pop decode_times from kwargs before re-passing so we don't trigger
        # "got multiple values for keyword argument 'decode_times'" when
        # the caller had decode_times=True.
        if kwargs.get("decode_times", True) is not False:
            logging.warning(f"Failed to open {path} with default time decoding: {e}. Retrying with decode_times=False")
            retry_kwargs = {k: v for k, v in kwargs.items() if k != "decode_times"}
            try:
                ds = xr.open_dataset(path, decode_times=False, **retry_kwargs)
                from openbench.data.time_utils import decode_nonstandard_time

                return decode_nonstandard_time(ds, source_path=str(path))
            except Exception as e2:
                logging.error(f"Failed to open {path} even with decode_times=False: {e2}")
                raise e2
        else:
            raise


def open_dataset_safe(path: str, **kwargs) -> xr.Dataset:
    """
    Open a NetCDF dataset with automatic fallback to decode_times=False if initial open fails.

    This is a simple wrapper that can be used as a drop-in replacement for xr.open_dataset()
    when you want automatic handling of time decoding issues.

    Parameters
    ----------
    path : str
        Path to the NetCDF file
    **kwargs
        Additional arguments passed to xr.open_dataset()

    Returns
    -------
    xr.Dataset
        The opened dataset

    Examples
    --------
    >>> ds = open_dataset_safe("file.nc")  # Will try decode_times=False on failure
    >>> ds = open_dataset_safe("file.nc", chunks={"time": 12})  # With chunking
    """
    return _open_dataset_with_fallback(path, **kwargs)


def open_dataset(
    path: str,
    chunks: Optional[Union[Dict[str, int], str]] = "auto",
    use_chunking: bool = True,
    size_threshold: int = CHUNK_SIZE_THRESHOLD,
    **kwargs,
) -> xr.Dataset:
    """
    Open a NetCDF dataset with automatic chunking for memory efficiency.

    Parameters
    ----------
    path : str
        Path to the NetCDF file
    chunks : dict, str, or None
        Chunk sizes for each dimension. Options:
        - "auto": Use smart defaults based on file size and dimensions
        - dict: Explicit chunk sizes, e.g., {"time": 12, "lat": 500, "lon": 500}
        - None: No chunking (load entirely into memory)
    use_chunking : bool
        Whether to use chunking at all. Set False for small files.
    size_threshold : int
        File size threshold (bytes) above which chunking is enabled.
        Files smaller than this are loaded directly into memory.
    **kwargs
        Additional arguments passed to xr.open_dataset()

    Returns
    -------
    xr.Dataset
        The opened dataset, potentially with Dask arrays if chunked.

    Examples
    --------
    >>> ds = open_dataset("large_file.nc")  # Auto chunking
    >>> ds = open_dataset("small_file.nc", chunks=None)  # No chunking
    >>> ds = open_dataset("file.nc", chunks={"time": 24, "lat": 1000})  # Custom
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Determine whether to use chunking based on file size
    file_size = os.path.getsize(path)

    if not use_chunking or file_size < size_threshold:
        # Small file: load directly into memory
        logging.debug(f"Loading small file directly: {path} ({file_size / 1024 / 1024:.1f} MB)")
        return _open_dataset_with_fallback(path, **kwargs)

    # Determine chunk sizes
    if chunks == "auto":
        chunks = _get_auto_chunks(path)

    logging.debug(f"Loading with chunks: {path} ({file_size / 1024 / 1024:.1f} MB), chunks={chunks}")
    return _open_dataset_with_fallback(path, chunks=chunks, **kwargs)


def _get_auto_chunks(path: str) -> Dict[str, int]:
    """
    Determine optimal chunk sizes based on file metadata.

    Parameters
    ----------
    path : str
        Path to the NetCDF file

    Returns
    -------
    dict
        Chunk sizes for each dimension found in the file
    """
    # Quick peek at the file to get dimensions
    try:
        with xr.open_dataset(path) as ds:
            dims = ds.sizes
    except Exception:
        # Try with decode_times=False
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                dims = ds.sizes
        except Exception as e:
            logging.warning(f"Could not inspect file for auto-chunking: {e}")
            return DEFAULT_CHUNKS.copy()

    chunks = {}
    for dim_name, dim_size in dims.items():
        dim_lower = dim_name.lower()

        # Match dimension names to default chunks
        if dim_lower in ("time", "t"):
            # For time: chunk by year (12 months) or less
            chunks[dim_name] = min(DEFAULT_CHUNKS.get("time", 12), dim_size)
        elif dim_lower in ("lat", "latitude", "y"):
            chunks[dim_name] = min(DEFAULT_CHUNKS.get("lat", 500), dim_size)
        elif dim_lower in ("lon", "longitude", "x"):
            chunks[dim_name] = min(DEFAULT_CHUNKS.get("lon", 500), dim_size)
        else:
            # For other dimensions, use a reasonable default
            chunks[dim_name] = min(100, dim_size)

    return chunks


def open_mfdataset(
    paths: list, chunks: Optional[Union[Dict[str, int], str]] = "auto", combine: str = "by_coords", **kwargs
) -> xr.Dataset:
    """
    Open multiple NetCDF files as a single dataset with chunking.

    Parameters
    ----------
    paths : list
        List of file paths or glob pattern
    chunks : dict, str, or None
        Chunk sizes (same as open_dataset)
    combine : str
        How to combine datasets ("by_coords" or "nested")
    **kwargs
        Additional arguments passed to xr.open_mfdataset()

    Returns
    -------
    xr.Dataset
        Combined dataset with Dask arrays
    """
    if chunks == "auto":
        # Get chunks from first file
        if isinstance(paths, list) and len(paths) > 0:
            first_file = paths[0]
        else:
            first_file = paths

        if os.path.exists(first_file):
            chunks = _get_auto_chunks(first_file)
        else:
            chunks = DEFAULT_CHUNKS.copy()

    logging.debug(f"Loading {len(paths) if isinstance(paths, list) else 'multiple'} files with chunks={chunks}")
    return xr.open_mfdataset(paths, chunks=chunks, combine=combine, **kwargs)


def _parse_int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Ignoring invalid %s=%r", name, raw)
        return default
    if minimum is not None:
        return max(minimum, value)
    return value


def _parse_float_env(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.warning("Ignoring invalid %s=%r", name, raw)
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _explicit_mfdataset_batch_size(batch_size: int | None = None) -> int | None:
    """Return explicit batch size if configured; None means use auto plan."""
    if batch_size is not None:
        try:
            return max(0, int(batch_size))
        except (TypeError, ValueError):
            logging.warning("Ignoring invalid write_mfdataset_atomic batch_size=%r", batch_size)
            return None
    raw = os.environ.get(MFDATASET_BATCH_SIZE_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        logging.warning("Ignoring invalid %s=%r", MFDATASET_BATCH_SIZE_ENV, raw)
        return None


def _available_memory_bytes() -> int | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _total_existing_size_bytes(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except OSError:
            continue
    return total


def _clamp(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def build_resource_plan(paths: Sequence[str], *, explicit_batch_size: int | None = None) -> ResourcePlan:
    """Build a cheap plan for combining many NetCDF files.

    The planner never opens NetCDF files. It only counts paths, stats known file
    sizes, and optionally reads available system memory. Explicit batch sizes
    always win: ``0`` disables batching, positive values force that batch size.
    """
    path_list = [str(path) for path in paths]
    file_count = len(path_list)
    total_size_bytes = _total_existing_size_bytes(path_list)
    available_memory_bytes = _available_memory_bytes()

    if explicit_batch_size is not None:
        return ResourcePlan(
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            available_memory_bytes=available_memory_bytes,
            mfdataset_batch_size=max(0, explicit_batch_size),
            reason="explicit",
        )

    min_files = _parse_int_env(
        MFDATASET_AUTO_BATCH_MIN_FILES_ENV,
        DEFAULT_MFDATASET_AUTO_BATCH_MIN_FILES,
        minimum=1,
    )
    min_size_bytes = (
        _parse_int_env(
            MFDATASET_AUTO_BATCH_MIN_SIZE_MB_ENV,
            DEFAULT_MFDATASET_AUTO_BATCH_MIN_SIZE_MB,
            minimum=0,
        )
        * 1024
        * 1024
    )

    if file_count <= min_files and (total_size_bytes <= 0 or total_size_bytes < min_size_bytes):
        return ResourcePlan(
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            available_memory_bytes=available_memory_bytes,
            mfdataset_batch_size=0,
            reason="below-auto-threshold",
        )

    min_batch = _parse_int_env(
        MFDATASET_AUTO_BATCH_MIN_SIZE_ENV,
        DEFAULT_MFDATASET_AUTO_BATCH_MIN_SIZE,
        minimum=1,
    )
    max_batch = _parse_int_env(
        MFDATASET_AUTO_BATCH_MAX_SIZE_ENV,
        DEFAULT_MFDATASET_AUTO_BATCH_MAX_SIZE,
        minimum=1,
    )
    if max_batch < min_batch:
        max_batch = min_batch

    batch_size = max_batch
    reason = "auto-file-count"
    if available_memory_bytes and total_size_bytes > 0 and file_count > 0:
        memory_fraction = _parse_float_env(
            MFDATASET_AUTO_BATCH_MEMORY_FRACTION_ENV,
            DEFAULT_MFDATASET_AUTO_BATCH_MEMORY_FRACTION,
            minimum=0.01,
            maximum=1.0,
        )
        average_file_size = max(1, total_size_bytes // file_count)
        target_bytes = max(1, int(available_memory_bytes * memory_fraction))
        size_based_batch = max(1, target_bytes // average_file_size)
        batch_size = _clamp(size_based_batch, min_batch, max_batch)
        reason = "auto-memory"

    # If the auto batch would cover the entire input, keep the simple one-shot
    # path; it creates less temporary I/O and preserves historical behavior.
    if batch_size >= file_count:
        batch_size = 0
        reason = "auto-single-batch"

    return ResourcePlan(
        file_count=file_count,
        total_size_bytes=total_size_bytes,
        available_memory_bytes=available_memory_bytes,
        mfdataset_batch_size=batch_size,
        reason=reason,
    )


def _batched_paths(paths: Sequence[str], batch_size: int) -> list[list[str]]:
    return [list(paths[i : i + batch_size]) for i in range(0, len(paths), batch_size)]


def _sorted_if_requested(ds: xr.Dataset, sortby: str | None) -> xr.Dataset:
    if sortby is None:
        return ds
    return ds.sortby(sortby)


def _write_netcdf_with_compression_policy(
    data: xr.Dataset,
    output_path: str | os.PathLike[str],
    *,
    compression: bool | None = None,
) -> None:
    """Call ``write_netcdf_atomic`` without a policy kw unless one is explicit.

    Several tests and downstream monkeypatches replace ``write_netcdf_atomic``
    with the historical two-argument callable. Keeping ``None`` as "omit kw"
    preserves that shape while still allowing batch shards to force
    ``compression=False``.
    """
    if compression is None:
        write_netcdf_atomic(data, output_path)
    else:
        write_netcdf_atomic(data, output_path, compression=compression)


def write_mfdataset_atomic(
    paths: Sequence[str],
    output_path: str | os.PathLike[str],
    *,
    chunks: Optional[Union[Dict[str, int], str]] = "auto",
    combine: str = "by_coords",
    batch_size: int | None = None,
    batch_dir: str | os.PathLike[str] | None = None,
    sortby: str | None = None,
    compression: bool | None = None,
    **kwargs,
) -> None:
    """Open many NetCDF files and write one combined NetCDF atomically.

    Explicit ``OPENBENCH_MFDATASET_BATCH_SIZE`` (or ``batch_size``) values win:
    ``0`` disables batching and positive values force that batch size. If no
    explicit value is set, a lightweight resource plan automatically batches
    very large file lists. Batch mode writes temporary NetCDF shards first, then
    combines those shards into the final output. Batch shards are always written
    with ``compression=False`` so final-output compression does not slow the
    intermediate combine path. ``compression`` applies only to the final target;
    ``None`` keeps the environment default. ``batch_dir`` defaults to
    ``target.parent``; passing a cross-device path only affects intermediate
    I/O speed because the final atomic replace still happens beside the target.
    """
    path_list = [str(path) for path in paths]
    if not path_list:
        raise ValueError("write_mfdataset_atomic requires at least one input file")

    target = Path(output_path)
    explicit_batch_size = _explicit_mfdataset_batch_size(batch_size)
    resource_plan = build_resource_plan(path_list, explicit_batch_size=explicit_batch_size)
    resolved_batch_size = resource_plan.mfdataset_batch_size
    if resolved_batch_size <= 0 or len(path_list) <= resolved_batch_size:
        with open_mfdataset(path_list, chunks=chunks, combine=combine, **kwargs) as ds:
            _write_netcdf_with_compression_policy(
                _sorted_if_requested(ds, sortby),
                target,
                compression=compression,
            )
        return

    root = Path(batch_dir) if batch_dir is not None else target.parent
    root.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Combining %d NetCDF files in %d-file batches for %s (reason=%s, total_size=%.1f MB)",
        len(path_list),
        resolved_batch_size,
        target,
        resource_plan.reason,
        resource_plan.total_size_bytes / 1024 / 1024,
    )
    logging.debug("mfdataset resource plan: %s", resource_plan)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.mfbatch-", dir=str(root)) as tmp_dir:
        batch_files: list[str] = []
        for index, batch in enumerate(_batched_paths(path_list, resolved_batch_size), start=1):
            batch_file = Path(tmp_dir) / f"batch-{index:05d}.nc"
            with open_mfdataset(batch, chunks=chunks, combine=combine, **kwargs) as ds:
                _write_netcdf_with_compression_policy(
                    _sorted_if_requested(ds, sortby),
                    batch_file,
                    compression=False,
                )
            batch_files.append(str(batch_file))

        with open_mfdataset(batch_files, chunks=chunks, combine=combine, **kwargs) as ds:
            _write_netcdf_with_compression_policy(
                _sorted_if_requested(ds, sortby),
                target,
                compression=compression,
            )


def write_mfdataset_zarr(
    paths: Sequence[str],
    zarr_path: str | os.PathLike[str],
    *,
    chunks: Optional[Union[Dict[str, int], str]] = "auto",
    combine: str = "by_coords",
    batch_size: int | None = None,
    batch_dir: str | os.PathLike[str] | None = None,
    sortby: str | None = None,
    to_zarr_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    """Open many NetCDF files and write one combined Zarr store.

    Uses the same explicit/automatic mfdataset batching policy as
    ``write_mfdataset_atomic``. Batch mode materializes temporary NetCDF shards
    before the final Zarr write, so CLI optimization does not have to feed
    thousands of source files into one ``xr.open_mfdataset`` call. ``batch_dir``
    defaults to ``target.parent``; passing a cross-device path only affects
    intermediate I/O speed.
    """
    path_list = [str(path) for path in paths]
    if not path_list:
        raise ValueError("write_mfdataset_zarr requires at least one input file")

    target = Path(zarr_path)
    zarr_kwargs = {"mode": "w"}
    if to_zarr_kwargs:
        zarr_kwargs.update(to_zarr_kwargs)

    explicit_batch_size = _explicit_mfdataset_batch_size(batch_size)
    resource_plan = build_resource_plan(path_list, explicit_batch_size=explicit_batch_size)
    resolved_batch_size = resource_plan.mfdataset_batch_size
    if resolved_batch_size <= 0 or len(path_list) <= resolved_batch_size:
        with open_mfdataset(path_list, chunks=chunks, combine=combine, **kwargs) as ds:
            _sorted_if_requested(ds, sortby).to_zarr(str(target), **zarr_kwargs)
        return

    root = Path(batch_dir) if batch_dir is not None else target.parent
    root.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Combining %d NetCDF files in %d-file batches before Zarr write for %s (reason=%s, total_size=%.1f MB)",
        len(path_list),
        resolved_batch_size,
        target,
        resource_plan.reason,
        resource_plan.total_size_bytes / 1024 / 1024,
    )
    logging.debug("mfdataset Zarr resource plan: %s", resource_plan)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.mfbatch-", dir=str(root)) as tmp_dir:
        batch_files: list[str] = []
        for index, batch in enumerate(_batched_paths(path_list, resolved_batch_size), start=1):
            batch_file = Path(tmp_dir) / f"batch-{index:05d}.nc"
            with open_mfdataset(batch, chunks=chunks, combine=combine, **kwargs) as ds:
                _write_netcdf_with_compression_policy(
                    _sorted_if_requested(ds, sortby),
                    batch_file,
                    compression=False,
                )
            batch_files.append(str(batch_file))

        with open_mfdataset(batch_files, chunks=chunks, combine=combine, **kwargs) as ds:
            _sorted_if_requested(ds, sortby).to_zarr(str(target), **zarr_kwargs)


def load_and_compute(
    path: str, variables: Optional[list] = None, chunks: Optional[Union[Dict[str, int], str]] = "auto", **kwargs
) -> xr.Dataset:
    """
    Load a dataset with chunking, then compute to load into memory.

    Useful when you need the data in memory but want to benefit from
    chunked loading for memory efficiency during the load process.

    Parameters
    ----------
    path : str
        Path to the NetCDF file
    variables : list, optional
        Specific variables to load (reduces memory usage)
    chunks : dict, str, or None
        Chunk sizes for loading
    **kwargs
        Additional arguments passed to open_dataset()

    Returns
    -------
    xr.Dataset
        Dataset with data loaded into memory (numpy arrays)
    """
    ds = open_dataset(path, chunks=chunks, **kwargs)

    if variables:
        ds = ds[variables]

    # Compute to load into memory
    return ds.compute()
