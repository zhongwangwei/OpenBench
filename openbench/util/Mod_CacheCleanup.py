#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cache Cleanup Utility for OpenBench

This module provides utilities to clean up Python cache files and directories
before program execution to ensure a clean runtime environment.

Author: OpenBench Team
Date: November 2025
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple


def cleanup_pycache(base_dir: str = None, verbose: bool = True) -> Tuple[int, int]:
    """
    Recursively remove all __pycache__ directories from the specified base directory.

    Args:
        base_dir: Base directory to start cleanup from. If None, uses the OpenBench root.
        verbose: Whether to log detailed cleanup information

    Returns:
        Tuple of (directories_removed, files_removed)
    """
    if base_dir is None:
        # Get OpenBench root directory (parent of util directory)
        base_dir = Path(__file__).parent.parent.parent.absolute()
    else:
        base_dir = Path(base_dir).absolute()

    if not base_dir.exists():
        logging.error(f"Base directory does not exist: {base_dir}")
        return 0, 0

    dirs_removed = 0
    files_removed = 0
    cache_dirs_found = []

    # Find all __pycache__ directories
    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in dirs:
            cache_dir = Path(root) / '__pycache__'
            cache_dirs_found.append(cache_dir)

    # Remove found cache directories
    for cache_dir in cache_dirs_found:
        try:
            # Count files before removal
            file_count = sum(1 for _ in cache_dir.glob('*.pyc'))
            file_count += sum(1 for _ in cache_dir.glob('*.pyo'))

            # Remove the directory
            shutil.rmtree(cache_dir)
            dirs_removed += 1
            files_removed += file_count

            if verbose:
                rel_path = cache_dir.relative_to(base_dir)
                logging.info(f"  Removed: {rel_path} ({file_count} files)")

        except Exception as e:
            logging.warning(f"Failed to remove {cache_dir}: {e}")

    return dirs_removed, files_removed


def cleanup_pyc_files(base_dir: str = None, verbose: bool = True) -> int:
    """
    Remove all .pyc and .pyo files from the specified base directory.

    Args:
        base_dir: Base directory to start cleanup from. If None, uses the OpenBench root.
        verbose: Whether to log detailed cleanup information

    Returns:
        Number of files removed
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent.absolute()
    else:
        base_dir = Path(base_dir).absolute()

    if not base_dir.exists():
        logging.error(f"Base directory does not exist: {base_dir}")
        return 0

    files_removed = 0

    # Find and remove .pyc and .pyo files
    for pattern in ['**/*.pyc', '**/*.pyo']:
        for pyc_file in base_dir.glob(pattern):
            try:
                pyc_file.unlink()
                files_removed += 1
                if verbose:
                    rel_path = pyc_file.relative_to(base_dir)
                    logging.debug(f"  Removed: {rel_path}")
            except Exception as e:
                logging.warning(f"Failed to remove {pyc_file}: {e}")

    return files_removed


def cleanup_all_cache(base_dir: str = None, verbose: bool = True) -> dict:
    """
    Comprehensive cache cleanup: removes __pycache__ directories and loose .pyc files.

    Args:
        base_dir: Base directory to start cleanup from. If None, uses the OpenBench root.
        verbose: Whether to log detailed cleanup information

    Returns:
        Dictionary with cleanup statistics
    """
    if verbose:
        logging.info("🧹 Cleaning up Python cache files...")

    # Clean __pycache__ directories
    dirs_removed, cache_files_removed = cleanup_pycache(base_dir, verbose=False)

    # Clean loose .pyc files
    loose_files_removed = cleanup_pyc_files(base_dir, verbose=False)

    stats = {
        'pycache_dirs_removed': dirs_removed,
        'cache_files_removed': cache_files_removed,
        'loose_pyc_files_removed': loose_files_removed,
        'total_files_removed': cache_files_removed + loose_files_removed
    }

    if verbose:
        if stats['pycache_dirs_removed'] > 0 or stats['loose_pyc_files_removed'] > 0:
            logging.info(f"  ✓ Removed {stats['pycache_dirs_removed']} __pycache__ directories")
            logging.info(f"  ✓ Removed {stats['total_files_removed']} cached files")
        else:
            logging.info("  ✓ No cache files found (already clean)")

    return stats


def get_cache_size(base_dir: str = None) -> Tuple[int, int]:
    """
    Calculate the total size of Python cache files.

    Args:
        base_dir: Base directory to check. If None, uses the OpenBench root.

    Returns:
        Tuple of (total_size_bytes, file_count)
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent.absolute()
    else:
        base_dir = Path(base_dir).absolute()

    if not base_dir.exists():
        return 0, 0

    total_size = 0
    file_count = 0

    # Check __pycache__ directories
    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in dirs:
            cache_dir = Path(root) / '__pycache__'
            for cache_file in cache_dir.iterdir():
                if cache_file.is_file():
                    total_size += cache_file.stat().st_size
                    file_count += 1

    # Check loose .pyc files
    for pattern in ['**/*.pyc', '**/*.pyo']:
        for pyc_file in base_dir.glob(pattern):
            if pyc_file.is_file():
                total_size += pyc_file.stat().st_size
                file_count += 1

    return total_size, file_count


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("OpenBench Cache Cleanup Utility")
    print("=" * 50)

    # Get cache statistics before cleanup
    size_before, count_before = get_cache_size()
    if size_before > 0:
        size_mb = size_before / (1024 * 1024)
        print(f"\nCache size before cleanup: {size_mb:.2f} MB ({count_before} files)")
    else:
        print("\nNo cache files found.")

    # Perform cleanup
    print("\nPerforming cleanup...")
    stats = cleanup_all_cache(verbose=True)

    # Get cache statistics after cleanup
    size_after, count_after = get_cache_size()

    print("\n" + "=" * 50)
    print("Cleanup Summary:")
    print(f"  Directories removed: {stats['pycache_dirs_removed']}")
    print(f"  Files removed: {stats['total_files_removed']}")
    if size_before > 0:
        size_freed_mb = (size_before - size_after) / (1024 * 1024)
        print(f"  Space freed: {size_freed_mb:.2f} MB")
    print("=" * 50)
