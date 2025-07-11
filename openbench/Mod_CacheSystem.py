# -*- coding: utf-8 -*-
"""
Caching System for OpenBench

This module provides a comprehensive caching system with support for
memory and disk caching, intelligent eviction policies, and performance monitoring.

Author: OpenBench Contributors  
Version: 1.0
Date: July 2025
"""

import os
import json
import pickle
import hashlib
import time
import logging
import threading
from typing import Dict, Any, Optional, Union, Callable, Tuple, List
from pathlib import Path
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import shutil
import weakref

# Import dependencies
try:
    import xarray as xr
    import pandas as pd
    import numpy as np
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
    from Mod_Exceptions import CacheError, error_handler
    from Mod_LoggingSystem import get_logging_manager, performance_logged
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
            # For xarray objects, use coordinates and shape
            content = f"{obj.dims}_{obj.shape}_{list(obj.coords.keys())}"
        elif _HAS_DATA_LIBS and isinstance(obj, pd.DataFrame):
            # For pandas DataFrame, use shape and columns
            content = f"{obj.shape}_{list(obj.columns)}"
        else:
            content = str(obj)
        
        # Generate hash
        hash_obj = hashlib.sha256(content.encode())
        key = hash_obj.hexdigest()[:16]  # Use first 16 chars
        
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
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'errors': self.errors,
                'total_requests': total_requests,
                'total_size_mb': self.total_size / (1024 * 1024),
                'runtime_seconds': runtime
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
            
            # Check expiry
            if self._is_expired(key):
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
            
            # Check if we need to evict
            if self._current_size + size > self.max_size:
                self._evict_lru(size)
            
            # Store value
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.sizes[key] = size
            self._current_size += size
            
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
                'type': 'memory',
                'entries': len(self.cache),
                'size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.max_size / (1024 * 1024),
                'stats': self.stats.get_stats()
            }


class FileSystemCache:
    """File system based cache."""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: float = 10240, 
                 ttl_seconds: Optional[int] = None):
        """
        Initialize file system cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size_mb * 1024 * 1024
        self.ttl = ttl_seconds
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
        # Initialize disk cache if available
        if _HAS_DISKCACHE:
            self.disk_cache = DiskCache(
                str(self.cache_dir),
                size_limit=self.max_size,
                eviction_policy='least-recently-used'
            )
        else:
            self.disk_cache = None
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    @error_handler(reraise=False)
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                self.stats.record_hit()
            else:
                self.stats.record_miss()
            return value
        
        # Fallback to manual file handling
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            self.stats.record_miss()
            return None
        
        # Check expiry
        if self.ttl:
            age = time.time() - file_path.stat().st_mtime
            if age > self.ttl:
                file_path.unlink()
                self.stats.record_miss()
                return None
        
        try:
            with open(file_path, 'rb') as f:
                value = pickle.load(f)
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
        
        # Fallback to manual file handling
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            logging.error(f"Cache write error: {e}")
            self.stats.record_error()
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
            'type': 'filesystem',
            'directory': str(self.cache_dir),
            'entries': entries,
            'size_mb': size_mb,
            'max_size_mb': self.max_size / (1024 * 1024),
            'stats': self.stats.get_stats()
        }


class CacheManager:
    """Unified cache manager with multiple levels."""
    
    def __init__(self, memory_size_mb: float = 1024, 
                 disk_size_mb: float = 10240,
                 cache_dir: str = "./cache",
                 ttl_seconds: Optional[int] = None):
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
        if level == 'memory' or (level is None and self.use_memory and self.memory_first):
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        if level == 'disk' or (level is None and self.use_disk):
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
        
        if level in ['memory', 'both'] or (level is None and self.use_memory):
            success &= self.memory_cache.set(key, value)
        
        if level in ['disk', 'both'] or (level is None and self.use_disk and not self.memory_first):
            success &= self.disk_cache.set(key, value)
        
        return success
    
    def clear(self, level: Optional[str] = None):
        """Clear cache at specified level."""
        if level in ['memory', None] and self.use_memory:
            self.memory_cache.clear()
        
        if level in ['disk', None] and self.use_disk:
            self.disk_cache.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        info = {
            'configuration': {
                'use_memory': self.use_memory,
                'use_disk': self.use_disk,
                'memory_first': self.memory_first
            }
        }
        
        if self.use_memory:
            info['memory_cache'] = self.memory_cache.get_info()
        
        if self.use_disk:
            info['disk_cache'] = self.disk_cache.get_info()
        
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
def cached(key_prefix: str = "", ttl: Optional[int] = None, 
          level: str = 'memory'):
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
    def cache_dataset(self, dataset: 'xr.Dataset', name: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Cache xarray dataset with metadata.
        
        Returns:
            Cache key for retrieval
        """
        if not _HAS_DATA_LIBS:
            raise ImportError("xarray required for dataset caching")
        
        # Generate key
        key = CacheKey.generate(name, prefix="dataset")
        
        # Store dataset and metadata
        cache_data = {
            'dataset': dataset,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'shape': dataset.dims,
            'variables': list(dataset.data_vars)
        }
        
        self.cache_manager.set(key, cache_data, level='disk')
        return key
    
    def get_dataset(self, key: str) -> Optional[Tuple['xr.Dataset', Dict[str, Any]]]:
        """Retrieve cached dataset and metadata."""
        cache_data = self.cache_manager.get(key)
        
        if cache_data and isinstance(cache_data, dict):
            return cache_data.get('dataset'), cache_data.get('metadata', {})
        
        return None, None
    
    @cached(key_prefix="computation", level='memory')
    def cached_computation(self, func: Callable, *args, **kwargs) -> Any:
        """Execute computation with automatic caching."""
        return func(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize cache manager
    cache_mgr = get_cache_manager(
        memory_size_mb=512,
        disk_size_mb=2048,
        ttl_seconds=3600
    )
    
    # Example 1: Basic caching
    @cached(key_prefix="example", ttl=60)
    def expensive_computation(n):
        time.sleep(1)  # Simulate expensive operation
        return n ** 2
    
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