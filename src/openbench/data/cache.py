# -*- coding: utf-8 -*-
"""
Caching System for OpenBench

This module provides a comprehensive caching system with support for
memory and disk caching, intelligent eviction policies, and performance monitoring.

Author: Zhongwang Wei
Version: 1.0
Date: July 2025
"""

import hashlib
import hmac
import json
import logging
import os
import pickle
import secrets
import threading
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Import dependencies
try:
    import numpy as np
    import pandas as pd
    import xarray as xr

    _HAS_DATA_LIBS = True
except ImportError:
    _HAS_DATA_LIBS = False
    xr = None
    pd = None
    np = None

try:
    from diskcache import Cache as DiskCache

    _HAS_DISKCACHE = True
except ImportError:
    _HAS_DISKCACHE = False
    DiskCache = None

try:
    from openbench.util.exceptions import CacheError, error_handler
    from openbench.util.logging_system import (  # noqa: F401  feature detection
        get_logging_manager,
        performance_logged,
    )

    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    CacheError = Exception

    def error_handler(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def performance_logged(operation=None):
        def decorator(func):
            return func

        return decorator


class CacheKey:
    """Helper class for generating cache keys."""

    @staticmethod
    def generate(obj: Any, prefix: str = "") -> str:
        """
        Generate a unique cache key for an object.

        Args:
            obj: Object to generate key for
            prefix: Optional prefix for the key

        Returns:
            Unique cache key string
        """
        if isinstance(obj, str):
            content = obj
        elif isinstance(obj, (dict, list, tuple)):
            content = json.dumps(obj, sort_keys=True, default=str)
        elif _HAS_DATA_LIBS and isinstance(obj, (xr.Dataset, xr.DataArray)):
            content = CacheKey._fingerprint_xarray(obj)
        elif _HAS_DATA_LIBS and isinstance(obj, pd.DataFrame):
            content = CacheKey._fingerprint_dataframe(obj)
        else:
            content = str(obj)

        # Generate hash. Use full SHA-256 (64 hex chars) — the prior 16-char
        # truncation combined with shape-only xarray fingerprints caused
        # collisions across datasets with identical structure but different
        # values (e.g., ref vs sim with same grid).
        hash_obj = hashlib.sha256(content.encode())
        key = hash_obj.hexdigest()

        if prefix:
            key = f"{prefix}_{key}"

        return key

    @staticmethod
    def from_function_call(func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        func_name = f"{func.__module__}.{func.__name__}"
        args_key = CacheKey.generate(args)
        kwargs_key = CacheKey.generate(kwargs)
        return f"{func_name}_{args_key}_{kwargs_key}"

    @staticmethod
    def _fingerprint_xarray(obj) -> str:
        """Build a content fingerprint for xr.Dataset / xr.DataArray.

        Uses dims, shape, dtypes, variable names, and coord-value hashes —
        never materialises data variable values (which may be lazy/huge).
        Coordinate arrays are always small enough to read.
        """
        parts = []
        # Dims as ordered tuple of (name, size)
        if isinstance(obj, xr.Dataset):
            dim_pairs = sorted(dict(obj.sizes).items())
        else:
            dim_pairs = list(zip(obj.dims, obj.shape))
        parts.append(f"dims={dim_pairs}")

        # Coordinates: name + dtype + size + first/last/hash-of-edges
        for cname in sorted(obj.coords.keys()):
            try:
                cv = obj.coords[cname].values
                if cv.size == 0:
                    parts.append(f"c:{cname}=empty")
                    continue
                # Hash all coord values — coordinates are small; avoids
                # collisions between e.g. 2000-2010 vs 2010-2020 monthly grids.
                cv_bytes = np.ascontiguousarray(cv).tobytes()
                cv_hash = hashlib.sha256(cv_bytes).hexdigest()[:16]
                parts.append(f"c:{cname}={cv.dtype}|n={cv.size}|h={cv_hash}")
            except Exception:
                parts.append(f"c:{cname}=unhashable")

        # Variable structure (names + dtypes + shapes), no data read
        if isinstance(obj, xr.Dataset):
            for vn in sorted(obj.data_vars.keys()):
                v = obj.data_vars[vn]
                parts.append(f"v:{vn}|{v.dtype}|{v.shape}")
        else:
            parts.append(f"v:{obj.name}|{obj.dtype}|{obj.shape}")

        # Top-level attrs (often carry units / source) — keep deterministic
        if obj.attrs:
            parts.append(f"attrs={json.dumps(obj.attrs, sort_keys=True, default=str)}")

        return "|".join(parts)

    @staticmethod
    def _fingerprint_xarray_content(obj) -> str:
        """Build a content fingerprint for xr.Dataset / xr.DataArray.

        This is intentionally stronger than the structural fingerprint used by
        generic function-call caching. `DataCache.cache_dataset()` is explicitly
        persisting the dataset object, so including data bytes in the key avoids
        same-name/same-shape datasets returning stale content.
        """
        parts = [CacheKey._fingerprint_xarray(obj)]
        variables = obj.data_vars.items() if isinstance(obj, xr.Dataset) else [(obj.name or "data", obj)]
        for name, var in variables:
            try:
                values = np.ascontiguousarray(var.values)
                parts.append(f"data:{name}={hashlib.sha256(values.tobytes()).hexdigest()}")
            except Exception as exc:
                parts.append(f"data:{name}=unhashable:{type(exc).__name__}")
        return "|".join(parts)

    @staticmethod
    def _fingerprint_dataframe(obj) -> str:
        """Build a content fingerprint for pd.DataFrame.

        Hashes column names+dtypes, shape, and the index values. Does not
        hash full data (potentially large) but does include a digest of the
        first and last rows so that two frames with the same schema but
        different content do not collide.
        """
        parts = [
            f"shape={obj.shape}",
            f"cols={[(c, str(obj[c].dtype)) for c in obj.columns]}",
        ]
        try:
            idx_vals = np.ascontiguousarray(obj.index.values).tobytes()
            parts.append(f"idx_h={hashlib.sha256(idx_vals).hexdigest()[:16]}")
        except Exception:
            parts.append(f"idx_n={len(obj.index)}")
        # Sample first and last row to break ties between same-schema frames
        if len(obj) > 0:
            try:
                head_bytes = obj.head(1).to_csv(index=False).encode()
                tail_bytes = obj.tail(1).to_csv(index=False).encode()
                parts.append(f"head_h={hashlib.sha256(head_bytes).hexdigest()[:16]}")
                parts.append(f"tail_h={hashlib.sha256(tail_bytes).hexdigest()[:16]}")
            except Exception:
                pass
        return "|".join(parts)


class CacheStats:
    """Track cache statistics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_hit(self):
        with self._lock:
            self.hits += 1

    def record_miss(self):
        with self._lock:
            self.misses += 1

    def record_eviction(self):
        with self._lock:
            self.evictions += 1

    def record_error(self):
        with self._lock:
            self.errors += 1

    def update_size(self, size: int):
        with self._lock:
            self.total_size = size

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            runtime = time.time() - self.start_time

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "errors": self.errors,
                "total_requests": total_requests,
                "total_size_mb": self.total_size / (1024 * 1024),
                "runtime_seconds": runtime,
            }


class MemoryCache:
    """In-memory cache with LRU eviction."""

    def __init__(self, max_size_mb: float = 1024, ttl_seconds: Optional[int] = None):
        """
        Initialize memory cache.

        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for entries (None = no expiry)
        """
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.sizes = {}
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._current_size = 0

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if _HAS_DATA_LIBS and isinstance(obj, (xr.Dataset, xr.DataArray)):
            return obj.nbytes
        elif _HAS_DATA_LIBS and isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif _HAS_DATA_LIBS and isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            # Rough estimate for other objects
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _is_expired(self, key: str) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False

        if key not in self.access_times:
            return True

        age = time.time() - self.access_times[key]
        return age > self.ttl

    def _evict_lru(self, required_size: int):
        """Evict least recently used entries to make space."""
        with self._lock:
            # Sort by access time
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])

            freed_size = 0
            for key, _ in sorted_keys:
                if self._current_size + required_size - freed_size <= self.max_size:
                    break

                # Evict entry
                if key in self.cache:
                    freed_size += self.sizes.get(key, 0)
                    del self.cache[key]
                    del self.access_times[key]
                    del self.sizes[key]
                    self.stats.record_eviction()

            self._current_size -= freed_size

    @error_handler(reraise=False)
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.stats.record_miss()
                return None

            # Check expiry. Decrement _current_size BEFORE removing the
            # entry; the previous code dropped sizes[key] but never
            # subtracted from _current_size, so the running total drifted
            # upward over the lifetime of the process and eventually
            # blocked all set() calls because the apparent budget was
            # exhausted while the actual cache was empty.
            if self._is_expired(key):
                self._current_size -= self.sizes.get(key, 0)
                if self._current_size < 0:
                    self._current_size = 0
                del self.cache[key]
                del self.access_times[key]
                del self.sizes[key]
                self.stats.record_miss()
                return None

            # Update access time
            self.access_times[key] = time.time()
            self.stats.record_hit()
            return self.cache[key]

    @error_handler(reraise=False)
    def set(self, key: str, value: Any) -> bool:
        """Set value in cache."""
        with self._lock:
            # Estimate size
            size = self._estimate_size(value)
            old_size = self.sizes.get(key, 0)

            if size > self.max_size:
                logging.warning("Cache item %s exceeds memory cache size limit; not caching", key)
                return False

            # Check if we need to evict
            if self._current_size - old_size + size > self.max_size:
                self._evict_lru(size)
                old_size = self.sizes.get(key, 0)

            # Store value
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.sizes[key] = size
            self._current_size = max(0, self._current_size - old_size) + size

            self.stats.update_size(self._current_size)
            return True

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.sizes.clear()
            self._current_size = 0
            self.stats.update_size(0)

    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        with self._lock:
            return {
                "type": "memory",
                "entries": len(self.cache),
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size / (1024 * 1024),
                "stats": self.stats.get_stats(),
            }


# Magic header tagging cache files written by this version of the format.
# Bumping this string invalidates older entries automatically.
_PICKLE_MAGIC = b"OBPKL\x01"
_HMAC_LEN = 32  # sha256


class FileSystemCache:
    """File system based cache."""

    def __init__(self, cache_dir: str = "./cache", max_size_mb: float = 10240, ttl_seconds: Optional[int] = None):
        """
        Initialize file system cache.

        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Restrict cache directory to owner-only on POSIX. Defense in depth
        # against another local user dropping a malicious .pkl into the
        # cache. Best-effort: chmod can fail on some shared filesystems.
        try:
            self.cache_dir.chmod(0o700)
        except OSError:
            pass
        self.max_size = max_size_mb * 1024 * 1024
        self.ttl = ttl_seconds
        self.stats = CacheStats()
        self._lock = threading.RLock()
        # Per-user HMAC key. Required for the manual pickle fallback so that
        # an attacker who can write into cache_dir cannot trigger arbitrary
        # code execution via pickle.load() of a poisoned file. Stored under
        # the cache root with mode 0600.
        self._hmac_key = self._load_or_create_hmac_key()

        # Initialize disk cache if available
        if _HAS_DISKCACHE:
            self.disk_cache = DiskCache(
                str(self.cache_dir), size_limit=self.max_size, eviction_policy="least-recently-used"
            )
        else:
            self.disk_cache = None

    def _load_or_create_hmac_key(self) -> bytes:
        """Load the cache HMAC key, creating it on first use."""
        key_path = self.cache_dir / ".cache_hmac_key"
        try:
            if key_path.exists():
                data = key_path.read_bytes()
                if len(data) == 32:
                    return data
                logging.warning(
                    "cache HMAC key at %s is the wrong length (%d); regenerating",
                    key_path,
                    len(data),
                )
            key = secrets.token_bytes(32)
            tmp = key_path.with_name(key_path.name + ".tmp")
            tmp.write_bytes(key)
            try:
                tmp.chmod(0o600)
            except OSError:
                pass
            os.replace(tmp, key_path)
            return key
        except OSError as exc:
            # If we can't persist a key (read-only fs), use an in-memory one.
            # Cache entries written this session won't be reusable later, but
            # we still get the integrity guarantee within the session.
            logging.warning("cache HMAC key unavailable on disk (%s); using ephemeral key", exc)
            return secrets.token_bytes(32)

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.pkl"

    @error_handler(reraise=False)
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.disk_cache:
            if key in self.disk_cache:
                self.stats.record_hit()
                return self.disk_cache.get(key)
            self.stats.record_miss()
            return None

        # Fallback to manual file handling
        file_path = self._get_file_path(key)

        if not file_path.exists():
            self.stats.record_miss()
            return None

        # Check expiry
        if self.ttl is not None:
            age = time.time() - file_path.stat().st_mtime
            if age > self.ttl:
                file_path.unlink()
                self.stats.record_miss()
                return None

        try:
            with open(file_path, "rb") as f:
                blob = f.read()
            # Strict verification: header + HMAC must match before any
            # pickle.loads() is invoked. A poisoned file dropped by another
            # process cannot pass HMAC verification without our per-user
            # key, so it is rejected before deserialization.
            magic_len = len(_PICKLE_MAGIC)
            if len(blob) < magic_len + _HMAC_LEN or blob[:magic_len] != _PICKLE_MAGIC:
                logging.warning(
                    "Cache entry %s missing magic header; treating as miss (stale or untrusted file).",
                    file_path,
                )
                self.stats.record_miss()
                return None
            sig = blob[magic_len : magic_len + _HMAC_LEN]
            payload = blob[magic_len + _HMAC_LEN :]
            expected = hmac.new(self._hmac_key, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(sig, expected):
                logging.warning(
                    "Cache entry %s failed HMAC verification; rejecting.",
                    file_path,
                )
                self.stats.record_miss()
                return None
            value = pickle.loads(payload)
            self.stats.record_hit()
            return value
        except Exception as e:
            logging.error(f"Cache read error: {e}")
            self.stats.record_error()
            return None

    @error_handler(reraise=False)
    def set(self, key: str, value: Any) -> bool:
        """Set value in cache."""
        if self.disk_cache:
            self.disk_cache.set(key, value, expire=self.ttl)
            return True

        # Fallback to manual file handling. Write to a unique temp path
        # in the same directory and atomically rename — without this, two
        # processes computing the same cache miss would interleave bytes
        # into the same file and produce a corrupt pickle that every
        # subsequent reader silently treats as a miss.
        file_path = self._get_file_path(key)
        tmp_path = file_path.with_name(f".{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")

        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            sig = hmac.new(self._hmac_key, payload, hashlib.sha256).digest()
            with open(tmp_path, "wb") as f:
                f.write(_PICKLE_MAGIC)
                f.write(sig)
                f.write(payload)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass  # fsync not supported on all filesystems
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, file_path)
            return True
        except Exception as e:
            logging.error(f"Cache write error: {e}")
            self.stats.record_error()
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False

    def clear(self):
        """Clear all cache entries."""
        if self.disk_cache:
            self.disk_cache.clear()
        else:
            # Manual cleanup
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()

    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        if self.disk_cache:
            size_mb = self.disk_cache.volume() / (1024 * 1024)
            entries = len(self.disk_cache)
        else:
            # Calculate manually
            files = list(self.cache_dir.glob("*.pkl"))
            entries = len(files)
            size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

        return {
            "type": "filesystem",
            "directory": str(self.cache_dir),
            "entries": entries,
            "size_mb": size_mb,
            "max_size_mb": self.max_size / (1024 * 1024),
            "stats": self.stats.get_stats(),
        }


class CacheManager:
    """Unified cache manager with multiple levels."""

    def __init__(
        self,
        memory_size_mb: float = 1024,
        disk_size_mb: float = 10240,
        cache_dir: str = "./cache",
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize cache manager.

        Args:
            memory_size_mb: Memory cache size in MB
            disk_size_mb: Disk cache size in MB
            cache_dir: Directory for disk cache
            ttl_seconds: Default TTL for entries
        """
        self.memory_cache = MemoryCache(memory_size_mb, ttl_seconds)
        self.disk_cache = FileSystemCache(cache_dir, disk_size_mb, ttl_seconds)
        self._lock = threading.RLock()

        # Configuration
        self.use_memory = True
        self.use_disk = True
        self.memory_first = True

        logging.info(f"Initialized CacheManager with {memory_size_mb}MB memory, {disk_size_mb}MB disk")

    @error_handler(reraise=False)
    def get(self, key: str, level: Optional[str] = None) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            level: Cache level ('memory', 'disk', or None for auto)

        Returns:
            Cached value or None
        """
        if level == "memory" or (level is None and self.use_memory and self.memory_first):
            value = self.memory_cache.get(key)
            if value is not None:
                return value

        if level == "disk" or (level is None and self.use_disk):
            value = self.disk_cache.get(key)
            if value is not None and self.use_memory and level is None:
                # Promote to memory cache
                self.memory_cache.set(key, value)
            return value

        return None

    @error_handler(reraise=False)
    def set(self, key: str, value: Any, level: Optional[str] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            level: Cache level ('memory', 'disk', 'both', or None for auto)

        Returns:
            Success status
        """
        success = True

        if level in ["memory", "both"] or (level is None and self.use_memory):
            success &= self.memory_cache.set(key, value)

        if level in ["disk", "both"] or (level is None and self.use_disk):
            success &= self.disk_cache.set(key, value)

        return success

    def clear(self, level: Optional[str] = None):
        """Clear cache at specified level."""
        if level in ["memory", None] and self.use_memory:
            self.memory_cache.clear()

        if level in ["disk", None] and self.use_disk:
            self.disk_cache.clear()

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        info = {
            "configuration": {
                "use_memory": self.use_memory,
                "use_disk": self.use_disk,
                "memory_first": self.memory_first,
            }
        }

        if self.use_memory:
            info["memory_cache"] = self.memory_cache.get_info()

        if self.use_disk:
            info["disk_cache"] = self.disk_cache.get_info()

        return info


# Global cache manager
_cache_manager = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)

    return _cache_manager


# Decorator for automatic caching
def cached(key_prefix: str = "", ttl: Optional[int] = None, level: str = "memory"):
    """
    Decorator for automatic function caching.

    Args:
        key_prefix: Prefix for cache keys
        ttl: Time-to-live in seconds
        level: Cache level to use
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = CacheKey.from_function_call(func, args, kwargs)
            if key_prefix:
                cache_key = f"{key_prefix}_{cache_key}"

            # Try cache
            manager = get_cache_manager()
            cached_value = manager.get(cache_key, level)

            if cached_value is not None:
                logging.debug(f"Cache hit for {func.__name__}")
                return cached_value

            # Compute value
            result = func(*args, **kwargs)

            # Store in cache
            manager.set(cache_key, result, level)

            return result

        # Add cache control methods
        wrapper.clear_cache = lambda: get_cache_manager().clear(level)
        wrapper.cache_info = lambda: get_cache_manager().get_info()

        return wrapper

    return decorator


# Specialized caching for data operations
class DataCache:
    """Specialized cache for scientific data."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize data cache."""
        self.cache_manager = cache_manager or get_cache_manager()

    @performance_logged("cache_dataset")
    def cache_dataset(self, dataset: "xr.Dataset", name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Cache xarray dataset with metadata.

        Returns:
            Cache key for retrieval
        """
        if not _HAS_DATA_LIBS:
            raise CacheError("xarray required for dataset caching")

        # Generate key from both the caller's logical name and dataset content.
        # Name-only keys collide for different slices/versions stored under the
        # same label and can return stale scientific data.
        key = CacheKey.generate(
            {"name": name, "dataset": CacheKey._fingerprint_xarray_content(dataset)},
            prefix="dataset",
        )

        # Store dataset and metadata
        cache_data = {
            "dataset": dataset,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "shape": dataset.dims,
            "variables": list(dataset.data_vars),
        }

        self.cache_manager.set(key, cache_data, level="disk")
        return key

    def get_dataset(self, key: str) -> Optional[Tuple["xr.Dataset", Dict[str, Any]]]:
        """Retrieve cached dataset and metadata."""
        cache_data = self.cache_manager.get(key)

        if cache_data and isinstance(cache_data, dict):
            return cache_data.get("dataset"), cache_data.get("metadata", {})

        return None, None

    @cached(key_prefix="computation", level="memory")
    def cached_computation(self, func: Callable, *args, **kwargs) -> Any:
        """Execute computation with automatic caching."""
        return func(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize cache manager
    cache_mgr = get_cache_manager(memory_size_mb=512, disk_size_mb=2048, ttl_seconds=3600)

    # Example 1: Basic caching
    @cached(key_prefix="example", ttl=60)
    def expensive_computation(n):
        time.sleep(1)  # Simulate expensive operation
        return n**2

    # First call (miss)
    result1 = expensive_computation(10)
    print(f"Result 1: {result1}")

    # Second call (hit)
    result2 = expensive_computation(10)
    print(f"Result 2: {result2}")

    # Cache info
    print("\nCache Info:")
    print(cache_mgr.get_info())

    # Clear cache
    cache_mgr.clear()
    print("\nCache cleared")
