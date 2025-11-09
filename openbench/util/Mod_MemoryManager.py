# -*- coding: utf-8 -*-
"""
Memory Management and Cleanup Module

This module provides comprehensive memory management and cleanup utilities
for the OpenBench evaluation system, including memory cleanup and file cleanup.

Author: Zhongwang Wei (zhongwang007@gmail.com)
Date: Nov 2025
"""

import gc
import logging
import os
import shutil
import numpy as np
import xarray as xr

# Try to import psutil for memory monitoring, use fallback if not available
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def cleanup_memory(verbose=True):
    """Comprehensive memory cleanup function.

    This function performs extensive memory cleanup including:
    - Python caches (functools, importlib, linecache)
    - Scientific library caches (numpy, xarray, pandas, matplotlib)
    - Garbage collection (all generations)
    - Weak reference cleanup
    - Memory defragmentation

    Args:
        verbose: If True, log detailed cleanup information. If False, perform silent cleanup.

    Returns:
        dict: Cleanup statistics including memory before/after and objects collected
    """
    stats = {
        'memory_before': 0,
        'memory_after': 0,
        'memory_freed': 0,
        'objects_collected': 0
    }

    try:
        # Get memory info before cleanup
        if _HAS_PSUTIL:
            process = psutil.Process()
            stats['memory_before'] = process.memory_info().rss / 1024 / 1024  # MB
        else:
            stats['memory_before'] = 0  # Fallback when psutil is not available

        # 1. Clear Python's functools caches (lru_cache)
        import functools
        # Clear all lru_cache decorated functions
        for obj in gc.get_objects():
            try:
                if isinstance(obj, functools._lru_cache_wrapper):
                    obj.cache_clear()
            except (AttributeError, TypeError):
                pass

        # 2. Clear numpy caches
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)

                # Try new numpy structure first (numpy >= 1.20)
                try:
                    if hasattr(np, '_core') and hasattr(np._core, '_internal'):
                        if hasattr(np._core._internal, 'clear_cache'):
                            np._core._internal.clear_cache()
                        elif hasattr(np._core._internal, '_clear_cache'):
                            np._core._internal._clear_cache()
                    else:
                        # Fallback to older numpy structure
                        if hasattr(np, 'core') and hasattr(np.core, '_internal'):
                            if hasattr(np.core._internal, 'clear_cache'):
                                np.core._internal.clear_cache()
                except (AttributeError, TypeError):
                    pass
        except (AttributeError, ImportError, Exception):
            pass

        # 3. Clear xarray caches
        if hasattr(xr, 'set_options'):
            xr.set_options(keep_attrs=False)

        # Clear xarray's file manager cache
        try:
            if hasattr(xr.backends, 'file_manager'):
                if hasattr(xr.backends.file_manager, 'FILE_CACHE'):
                    xr.backends.file_manager.FILE_CACHE.clear()
        except (AttributeError, TypeError):
            pass

        # 4. Clear pandas caches
        try:
            import pandas as pd
            # Clear string cache if available
            if hasattr(pd.core.strings, 'cache_readonly'):
                pd.core.strings.cache_readonly._cache.clear()
        except (AttributeError, ImportError, Exception):
            pass

        # 5. Clear matplotlib caches
        try:
            import matplotlib
            if hasattr(matplotlib, 'font_manager'):
                if hasattr(matplotlib.font_manager, '_fmcache'):
                    matplotlib.font_manager._fmcache = None
        except (ImportError, AttributeError):
            pass

        # 6. Clear Python's import cache
        try:
            import importlib
            importlib.invalidate_caches()
        except (ImportError, AttributeError):
            pass

        # 7. Clear linecache (used by traceback)
        import linecache
        linecache.clearcache()

        # 8. Force garbage collection multiple times
        # Collect unreachable objects from all generations
        collected_total = 0
        for generation in range(3):
            collected = gc.collect(generation)
            collected_total += collected

        stats['objects_collected'] = collected_total

        # 9. Force memory defragmentation if available
        if hasattr(gc, 'set_threshold'):
            # Temporarily lower GC thresholds to be more aggressive
            old_thresholds = gc.get_threshold()
            gc.set_threshold(10, 5, 5)
            gc.collect()
            # Restore original thresholds
            gc.set_threshold(*old_thresholds)

        # 10. Clear weak references
        import weakref
        # Force cleanup of dead weak references
        for obj in gc.get_objects():
            try:
                if isinstance(obj, weakref.ref):
                    obj()  # Access to trigger cleanup of dead refs
            except (TypeError, AttributeError):
                pass

        # Final garbage collection
        gc.collect()

        # Get memory info after cleanup
        if _HAS_PSUTIL:
            stats['memory_after'] = process.memory_info().rss / 1024 / 1024  # MB
            stats['memory_freed'] = stats['memory_before'] - stats['memory_after']

            if verbose and hasattr(logging, 'info'):
                logging.info(f"Memory cleanup completed:")
                logging.info(f"  - Memory before: {stats['memory_before']:.1f} MB")
                logging.info(f"  - Memory after: {stats['memory_after']:.1f} MB")
                logging.info(f"  - Objects collected: {stats['objects_collected']}")
                if stats['memory_freed'] > 0:
                    logging.info(f"  - Memory freed: {stats['memory_freed']:.1f} MB")
                else:
                    logging.info(f"  - Memory usage: {abs(stats['memory_freed']):.1f} MB (may have increased due to logging)")
        else:
            if verbose and hasattr(logging, 'info'):
                logging.info(f"Memory cleanup completed (psutil not available for detailed monitoring)")
                logging.info(f"  - Garbage collection performed")
                logging.info(f"  - Objects collected: {stats['objects_collected']}")
                logging.info(f"  - Cache clearing attempted")

    except Exception as e:
        if hasattr(logging, 'warning'):
            logging.warning(f"Memory cleanup encountered an issue: {e}")
        # Still perform basic garbage collection
        gc.collect()

    return stats


def initialize_memory_management():
    """Initialize memory management settings for optimal performance.

    This function configures:
    - Garbage collection thresholds for aggressive cleanup
    - NumPy error handling to reduce memory overhead
    - xarray options for memory efficiency
    """
    try:
        # Configure garbage collection for better memory management
        gc.set_threshold(700, 10, 10)  # More aggressive collection

        # Enable garbage collection debugging if needed (disable in production)
        # gc.set_debug(gc.DEBUG_STATS)

        # Configure numpy for memory efficiency
        if hasattr(np, 'seterr'):
            np.seterr(all='ignore')  # Ignore numpy warnings to reduce memory overhead

        # Configure xarray for memory efficiency
        if hasattr(xr, 'set_options'):
            xr.set_options(
                keep_attrs=False,  # Don't keep attributes to save memory
                display_style='text',  # Use text display to save memory
            )

        logging.info("Memory management initialized with optimized settings")

    except Exception as e:
        logging.warning(f"Failed to initialize memory management settings: {e}")


def get_memory_usage():
    """Get current memory usage in MB.

    Returns:
        float: Current memory usage in MB, or 0 if psutil is not available
    """
    if _HAS_PSUTIL:
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return 0
    return 0


def log_memory_usage(label="Current"):
    """Log current memory usage with a label.

    Args:
        label: String label to identify this measurement
    """
    memory_mb = get_memory_usage()
    if memory_mb > 0:
        logging.info(f"{label} memory usage: {memory_mb:.1f} MB")
    else:
        logging.debug(f"{label} memory usage: (psutil not available)")


def cleanup_old_outputs(main_nl, clean_level='tmp'):
    """Clean up old outputs and temporary files before running.

    Args:
        main_nl: Main namelist configuration
        clean_level: Level of cleanup to perform
            - 'tmp': Only clean tmp and scratch directories (default)
            - 'all': Clean tmp, scratch, and all outputs (metrics, scores, data)
            - 'none': Skip cleanup

    Returns:
        int: Number of directories cleaned
    """
    if clean_level == 'none':
        return 0

    # Import here to avoid circular dependency
    from openbench.util.Mod_ConfigCheck import get_platform_colors

    base_path = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'])

    # Directories to clean based on level
    cleanup_dirs = []

    if clean_level in ['tmp', 'all']:
        cleanup_dirs.extend([
            os.path.join(base_path, 'tmp'),
            os.path.join(base_path, 'scratch')
        ])

    if clean_level == 'all':
        cleanup_dirs.extend([
            os.path.join(base_path, 'output', 'metrics'),
            os.path.join(base_path, 'output', 'scores'),
            os.path.join(base_path, 'output', 'data')
        ])

    colors = get_platform_colors()
    clean_icon = "🧹" if colors['reset'] else "[CLEAN]"

    cleaned_count = 0
    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                # Count files before cleaning
                file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                if file_count > 0:
                    print(f"{clean_icon} Cleaning {dir_path} ({file_count} files)...")
                    shutil.rmtree(dir_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"{colors['yellow']}Warning: Could not clean {dir_path}: {e}{colors['reset']}")

    if cleaned_count > 0:
        check_icon = "✅" if colors['reset'] else "[OK]"
        print(f"{check_icon} {colors['green']}Cleaned {cleaned_count} directories{colors['reset']}\n")

    return cleaned_count
