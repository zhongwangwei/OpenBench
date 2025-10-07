# -*- coding: utf-8 -*-
"""
Progress Monitoring and Timeout Protection for OpenBench

This module provides progress monitoring and timeout protection to prevent
the system from hanging indefinitely.

Author: Zhongwang Wei
Version: 1.0
Date: January 2025
"""

import time
import logging
import threading
from typing import Optional, Callable
from functools import wraps


class ProgressMonitor:
    """Monitor progress and detect hanging operations."""

    def __init__(self, timeout_seconds: int = 300):
        """
        Initialize progress monitor.

        Args:
            timeout_seconds: Maximum time to wait without progress (default 5 minutes)
        """
        self.timeout_seconds = timeout_seconds
        self.last_activity = time.time()
        self._lock = threading.Lock()
        self._timer = None
        self._callback = None

    def update(self, message: Optional[str] = None):
        """
        Update last activity timestamp.

        Args:
            message: Optional progress message
        """
        with self._lock:
            self.last_activity = time.time()
            if message:
                logging.debug(f"Progress: {message}")

    def check_timeout(self) -> bool:
        """
        Check if operation has timed out.

        Returns:
            True if timed out, False otherwise
        """
        with self._lock:
            elapsed = time.time() - self.last_activity
            return elapsed > self.timeout_seconds

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since last activity.

        Returns:
            Elapsed time in seconds
        """
        with self._lock:
            return time.time() - self.last_activity

    def start_watchdog(self, callback: Optional[Callable] = None, check_interval: int = 30):
        """
        Start background watchdog timer.

        Args:
            callback: Function to call on timeout
            check_interval: How often to check (seconds)
        """
        self._callback = callback

        def watchdog():
            if self.check_timeout():
                elapsed = self.get_elapsed_time()
                logging.warning(f"Operation timeout detected! No progress for {elapsed:.1f}s")
                if self._callback:
                    self._callback()
            else:
                # Reschedule
                self._timer = threading.Timer(check_interval, watchdog)
                self._timer.daemon = True
                self._timer.start()

        self._timer = threading.Timer(check_interval, watchdog)
        self._timer.daemon = True
        self._timer.start()

    def stop_watchdog(self):
        """Stop the watchdog timer."""
        if self._timer:
            self._timer.cancel()
            self._timer = None


def with_timeout(timeout_seconds: int = 300, error_message: Optional[str] = None):
    """
    Decorator to add timeout protection to functions.

    Args:
        timeout_seconds: Maximum execution time
        error_message: Custom error message

    Example:
        @with_timeout(timeout_seconds=600)
        def long_running_task():
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                msg = error_message or f"Function {func.__name__} timed out after {timeout_seconds}s"
                logging.error(msg)
                raise TimeoutError(msg)

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


class HeartbeatMonitor:
    """Simple heartbeat monitor for long-running operations."""

    def __init__(self, name: str, heartbeat_interval: int = 60):
        """
        Initialize heartbeat monitor.

        Args:
            name: Name of the operation being monitored
            heartbeat_interval: Interval between heartbeats (seconds)
        """
        self.name = name
        self.heartbeat_interval = heartbeat_interval
        self.start_time = time.time()
        self.last_heartbeat = self.start_time
        self._timer = None
        self._stopped = False

    def start(self):
        """Start heartbeat monitoring."""
        self._stopped = False
        self._schedule_heartbeat()

    def _schedule_heartbeat(self):
        """Schedule next heartbeat."""
        if not self._stopped:
            def beat():
                if not self._stopped:
                    elapsed = time.time() - self.start_time
                    logging.info(f"⏱️  [{self.name}] Still running... ({elapsed:.1f}s elapsed)")
                    self._schedule_heartbeat()

            self._timer = threading.Timer(self.heartbeat_interval, beat)
            self._timer.daemon = True
            self._timer.start()

    def stop(self):
        """Stop heartbeat monitoring."""
        self._stopped = True
        if self._timer:
            self._timer.cancel()
            self._timer = None

        elapsed = time.time() - self.start_time
        logging.info(f"✓ [{self.name}] Completed in {elapsed:.1f}s")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def log_progress(operation_name: str, total_items: int):
    """
    Decorator to log progress for iterative operations.

    Args:
        operation_name: Name of the operation
        total_items: Total number of items to process

    Example:
        @log_progress("Processing files", total_items=100)
        def process_files(files):
            for i, file in enumerate(files):
                # Your code here
                yield i  # Yield progress
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logging.info(f"Starting {operation_name} ({total_items} items)")

            try:
                result = func(*args, **kwargs)

                # If it's a generator, wrap it with progress logging
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    def progress_wrapper():
                        for i, item in enumerate(result):
                            if i % max(1, total_items // 10) == 0:  # Log every 10%
                                percent = (i / total_items) * 100
                                elapsed = time.time() - start_time
                                logging.info(f"{operation_name}: {i}/{total_items} ({percent:.1f}%) - {elapsed:.1f}s")
                            yield item

                        # Final log
                        elapsed = time.time() - start_time
                        logging.info(f"{operation_name}: Completed {total_items} items in {elapsed:.1f}s")

                    return progress_wrapper()
                else:
                    elapsed = time.time() - start_time
                    logging.info(f"{operation_name}: Completed in {elapsed:.1f}s")
                    return result

            except Exception as e:
                elapsed = time.time() - start_time
                logging.error(f"{operation_name}: Failed after {elapsed:.1f}s - {e}")
                raise

        return wrapper
    return decorator
