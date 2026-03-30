# -*- coding: utf-8 -*-
"""
Dataset loading utilities with automatic chunking and glob caching.

This module provides optimized dataset loading functions that use Dask chunking
to reduce memory usage and improve performance for large NetCDF files.
It also provides cached glob operations to avoid repeated directory scans.
"""

import glob as _glob
import os
import logging
import time
from typing import Dict, List, Optional, Union, Any

import xarray as xr


# =============================================================================
# Glob Caching
# =============================================================================

# Global cache for glob results
# Format: {pattern: (results, timestamp)}
_glob_cache: Dict[str, tuple] = {}

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

    # Check cache
    if not force_refresh and pattern in _glob_cache:
        results, timestamp = _glob_cache[pattern]
        if current_time - timestamp < ttl:
            logging.debug(f"Glob cache hit: {pattern} ({len(results)} files)")
            return results

    # Cache miss or expired - perform actual glob
    results = sorted(_glob.glob(pattern))
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
    "time": 12,      # Monthly data: 1 year per chunk
    "lat": 500,      # Spatial chunks
    "lon": 500,
    "latitude": 500,
    "longitude": 500,
    "y": 500,
    "x": 500,
}

# Size threshold for enabling chunking (bytes)
# Files smaller than this are loaded directly into memory
CHUNK_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB


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
        # If decode_times is not already set to False, try with decode_times=False
        if kwargs.get('decode_times', True) is not False:
            logging.warning(f"Failed to open {path} with default time decoding: {e}. Retrying with decode_times=False")
            try:
                return xr.open_dataset(path, decode_times=False, **kwargs)
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
    **kwargs
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
            dims = ds.dims
    except Exception:
        # Try with decode_times=False
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                dims = ds.dims
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
    paths: list,
    chunks: Optional[Union[Dict[str, int], str]] = "auto",
    combine: str = "by_coords",
    **kwargs
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


def load_and_compute(
    path: str,
    variables: Optional[list] = None,
    chunks: Optional[Union[Dict[str, int], str]] = "auto",
    **kwargs
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
