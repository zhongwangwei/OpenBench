"""Small processing utilities kept outside the DatasetProcessing classes."""

from __future__ import annotations

import functools
import logging
import os
import re
import time
from typing import Callable

from .coordinates import COORDINATE_MAP

logger = logging.getLogger(__name__)

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logging.warning("psutil not available. Performance monitoring will be limited.")


def parse_time_offset(offset_str: str):
    """Parse a time offset string like '-15 days', '-1 months', '3 hours'.

    Returns a pandas DateOffset, or None if parsing fails.
    """
    import pandas as pd

    match = re.match(r"([+-]?\d+)\s*(day|days|month|months|hour|hours)", offset_str.strip())
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2).lower()

    if unit.startswith("day"):
        return pd.DateOffset(days=value)
    if unit.startswith("month"):
        return pd.DateOffset(months=value)
    if unit.startswith("hour"):
        return pd.DateOffset(hours=value)
    return None


def get_coordinate_map() -> dict[str, str]:
    """Return a copy of the shared coordinate mapping."""
    return dict(COORDINATE_MAP)


def performance_monitor(func: Callable = None, *, silent_on_error: bool = False) -> Callable:
    """Enhanced decorator to monitor function performance with error handling.

    Args:
        func: The function to decorate
        silent_on_error: If True, don't log errors (only re-raise them)
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get initial memory usage
            start_time = time.time()
            if _HAS_PSUTIL:
                try:
                    process = psutil.Process(os.getpid())
                    start_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
                    start_cpu = process.cpu_percent()
                except Exception as e:
                    logging.warning(f"Failed to get system resources: {e}")
                    start_mem = 0
                    start_cpu = 0
            else:
                start_mem = 0
                start_cpu = 0

            try:
                # Execute the function
                result = f(*args, **kwargs)

                # Calculate execution time and memory usage
                end_time = time.time()
                execution_time = end_time - start_time

                if _HAS_PSUTIL:
                    try:
                        end_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
                        end_cpu = process.cpu_percent()
                        memory_used = end_mem - start_mem
                        cpu_used = end_cpu - start_cpu
                    except Exception as e:
                        logging.warning(f"Failed to get end system resources: {e}")
                        memory_used = 0
                        cpu_used = 0
                else:
                    memory_used = 0
                    cpu_used = 0

                # Log performance
                logging.debug(f"Performance for {f.__name__}:")
                logging.debug(f"  Execution time: {execution_time:.2f} seconds")
                if _HAS_PSUTIL:
                    logging.debug(f"  Memory usage: {memory_used:.3f} GB")
                    logging.debug(f"  CPU usage: {cpu_used:.1f}%")

                # Log warning if memory usage is high
                if _HAS_PSUTIL:
                    try:
                        total_memory = psutil.virtual_memory().total / (1024**3)
                        if memory_used > 0.8 * total_memory:  # 80% of total memory
                            logging.warning(f"High memory usage detected in {f.__name__}: {memory_used:.3f} GB")
                    except Exception as e:
                        logging.debug(f"Failed to check total memory: {e}")

                return result

            except Exception as e:
                # Log error with performance context (unless silent)
                end_time = time.time()
                execution_time = end_time - start_time

                if _HAS_PSUTIL:
                    try:
                        end_mem = process.memory_info().rss / 1024 / 1024 / 1024
                        memory_used = end_mem - start_mem
                    except Exception as e:
                        logger.debug("Failed to read end memory usage: %s", e)
                        memory_used = 0
                else:
                    memory_used = 0

                if not silent_on_error:
                    if isinstance(e, (FileNotFoundError, ValueError)):
                        # Expected errors: log at debug to avoid noisy cascaded messages
                        logging.debug(f"Error in {f.__name__} after {execution_time:.2f}s: {e}")
                    else:
                        logging.error(
                            f"Error in {f.__name__} after {execution_time:.2f}s and using {memory_used:.3f} GB:"
                        )
                        logging.error(str(e))
                raise

        return wrapper

    # Support both @performance_monitor and @performance_monitor(silent_on_error=True)
    if func is None:
        return decorator
    return decorator(func)
