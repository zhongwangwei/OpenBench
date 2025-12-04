# -*- coding: utf-8 -*-
"""
Enhanced Logging System for OpenBench

This module provides a comprehensive logging system with structured logging,
log rotation, performance monitoring, and advanced filtering capabilities.

Author: Zhongwang Wei  
Version: 1.0
Date: July 2025
"""
import io
import os
import sys
import json
import logging
import logging.handlers
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import functools
import time
import threading
import queue
import atexit

# Import dependencies
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from openbench.util.Mod_Exceptions import OpenBenchException, error_handler

    _HAS_EXCEPTIONS = True
except ImportError:
    _HAS_EXCEPTIONS = False
    OpenBenchException = Exception


    def error_handler(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def __init__(self, include_extra_fields: bool = True):
        """
        Initialize structured formatter.

        Args:
            include_extra_fields: Whether to include extra fields in output
        """
        super().__init__()
        self.include_extra_fields = include_extra_fields

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log structure
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields if enabled
        if self.include_extra_fields:
            # Standard fields to exclude
            exclude_fields = {
                'name', 'msg', 'args', 'created', 'msecs', 'relativeCreated',
                'levelname', 'levelno', 'pathname', 'filename', 'module',
                'funcName', 'lineno', 'exc_info', 'exc_text', 'stack_info',
                'thread', 'threadName', 'processName', 'process'
            }

            # Add any extra fields
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in exclude_fields
            }
            if extra_fields:
                log_data['extra'] = extra_fields

        return json.dumps(log_data, default=str)


class PerformanceFilter(logging.Filter):
    """Filter that adds performance metrics to log records."""

    def __init__(self):
        """Initialize performance filter."""
        super().__init__()
        self.start_time = time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to record."""
        # Add timing information
        record.elapsed_time = time.time() - self.start_time

        # Add memory usage if available
        if _HAS_PSUTIL:
            try:
                process = psutil.Process()
                record.memory_mb = process.memory_info().rss / 1024 / 1024
                record.cpu_percent = process.cpu_percent()
            except:
                record.memory_mb = 0
                record.cpu_percent = 0
        else:
            record.memory_mb = 0
            record.cpu_percent = 0

        return True


class AsyncHandler(logging.Handler):
    """Asynchronous logging handler for improved performance."""

    def __init__(self, handler: logging.Handler, queue_size: int = 10000):
        """
        Initialize async handler.

        Args:
            handler: The underlying handler to use
            queue_size: Maximum queue size
        """
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        # Register cleanup
        atexit.register(self.close)

    def _worker(self):
        """Worker thread for processing log records."""
        while True:
            try:
                record = self.queue.get(timeout=1)
                if record is None:  # Shutdown signal
                    break
                self.handler.emit(record)
            except queue.Empty:
                continue
            except Exception:
                # Ignore errors in worker thread
                pass

    def emit(self, record: logging.LogRecord):
        """Queue record for asynchronous processing."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # If queue is full, fall back to synchronous handling
            self.handler.emit(record)

    def close(self):
        """Close the handler and wait for queue to empty."""
        self.queue.put(None)  # Signal shutdown
        self.thread.join(timeout=5)
        self.handler.close()
        super().close()


class LoggingManager:
    """Centralized logging management system."""

    def __init__(self, base_dir: str = "./logs", app_name: str = "OpenBench"):
        """
        Initialize logging manager.

        Args:
            base_dir: Base directory for log files
            app_name: Application name for logs
        """
        self.base_dir = Path(base_dir)
        self.app_name = app_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.config = {
            'console_enabled': True,
            'file_enabled': True,
            'structured_enabled': False,
            'async_enabled': False,
            'rotation_enabled': True,
            'performance_tracking': True,
            'max_bytes': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'log_level': logging.INFO
        }

        # Handlers registry
        self.handlers = {}

        # Loggers registry
        self.loggers = {}

        # Initialize root logger
        self._setup_root_logger()

    def _setup_root_logger(self):
        """Set up the root logger with default configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config['log_level'])

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add configured handlers
        if self.config['console_enabled']:
            self.add_console_handler()

        if self.config['file_enabled']:
            self.add_file_handler()

    def add_console_handler(
            self,
            level: Optional[int] = None,
            formatter: Optional[logging.Formatter] = None
    ) -> logging.Handler:
        """Add console handler to root logger."""
        stream = sys.stdout
        try:
            enc = (getattr(stream, "encoding", "") or "").lower()
            if hasattr(stream, "buffer") and "utf-8" not in enc:
                stream = io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",  # 无法编码的字符用 ? 代替，避免 UnicodeEncodeError
                    line_buffering=True
                )
        except Exception:
            pass

        handler = logging.StreamHandler(stream)
        handler.setLevel(level or self.config['log_level'])

        if formatter is None:
            if self.config['structured_enabled']:
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

        handler.setFormatter(formatter)

        # Add performance filter if enabled
        if self.config['performance_tracking']:
            handler.addFilter(PerformanceFilter())

        # Wrap in async handler if enabled
        if self.config['async_enabled']:
            handler = AsyncHandler(handler)

        logging.getLogger().addHandler(handler)
        self.handlers['console'] = handler

        return handler

    def add_file_handler(
            self,
            filename: Optional[str] = None,
            level: Optional[int] = None,
            formatter: Optional[logging.Formatter] = None
    ) -> logging.Handler:
        """Add file handler with rotation support."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.base_dir / f"{self.app_name}_{timestamp}.log"
        else:
            filename = self.base_dir / filename

        # Create handler with rotation
        if self.config['rotation_enabled']:
            handler = logging.handlers.RotatingFileHandler(
                filename,
                maxBytes=self.config['max_bytes'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
        else:
            handler = logging.FileHandler(filename, encoding='utf-8')

        handler.setLevel(level or self.config['log_level'])

        if formatter is None:
            if self.config['structured_enabled']:
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

        handler.setFormatter(formatter)

        # Add performance filter if enabled
        if self.config['performance_tracking']:
            handler.addFilter(PerformanceFilter())

        # Wrap in async handler if enabled
        if self.config['async_enabled']:
            handler = AsyncHandler(handler)

        logging.getLogger().addHandler(handler)
        self.handlers['file'] = handler

        return handler

    def get_logger(
            self,
            name: str,
            level: Optional[int] = None
    ) -> logging.Logger:
        """
        Get or create a logger with specified configuration.

        Args:
            name: Logger name
            level: Optional log level

        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        if level is not None:
            logger.setLevel(level)

        self.loggers[name] = logger
        return logger

    def set_level(self, level: Union[int, str], logger_name: Optional[str] = None):
        """
        Set logging level.

        Args:
            level: Log level (int or string)
            logger_name: Optional logger name (None for root)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        if logger_name:
            logging.getLogger(logger_name).setLevel(level)
        else:
            logging.getLogger().setLevel(level)
            self.config['log_level'] = level

    def enable_structured_logging(self):
        """Enable structured JSON logging."""
        self.config['structured_enabled'] = True

        # Update existing handlers
        for handler in logging.getLogger().handlers:
            handler.setFormatter(StructuredFormatter())

    def enable_async_logging(self):
        """Enable asynchronous logging."""
        self.config['async_enabled'] = True

        # Wrap existing handlers
        root_logger = logging.getLogger()
        for i, handler in enumerate(root_logger.handlers[:]):
            if not isinstance(handler, AsyncHandler):
                async_handler = AsyncHandler(handler)
                root_logger.removeHandler(handler)
                root_logger.addHandler(async_handler)

    def add_context(self, **kwargs):
        """
        Add context information to all log messages.

        Args:
            **kwargs: Context key-value pairs
        """

        class ContextFilter(logging.Filter):
            def filter(self, record):
                for key, value in kwargs.items():
                    setattr(record, key, value)
                return True

        # Add to all handlers
        for handler in logging.getLogger().handlers:
            handler.addFilter(ContextFilter())

    def log_performance(
            self,
            operation: str,
            duration: float,
            success: bool = True,
            details: Optional[Dict[str, Any]] = None
    ):
        """
        Log performance metrics for an operation.

        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            details: Additional details
        """
        logger = self.get_logger('performance')

        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'success': success
        }

        if details:
            log_data.update(details)

        if success:
            logger.info(f"Performance: {operation}", extra=log_data)
        else:
            logger.warning(f"Performance (failed): {operation}", extra=log_data)

    def cleanup_old_logs(self, days: int = 30):
        """
        Clean up log files older than specified days.

        Args:
            days: Number of days to keep logs
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        for log_file in self.base_dir.glob(f"{self.app_name}_*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    logging.info(f"Deleted old log file: {log_file}")
            except Exception as e:
                logging.error(f"Error deleting log file {log_file}: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logging configuration."""
        return {
            'base_directory': str(self.base_dir),
            'app_name': self.app_name,
            'configuration': self.config.copy(),
            'active_handlers': list(self.handlers.keys()),
            'registered_loggers': list(self.loggers.keys()),
            'log_files': [str(f) for f in self.base_dir.glob(f"{self.app_name}_*.log")]
        }


# Global logging manager instance
_logging_manager = None


def get_logging_manager(base_dir: Optional[str] = None) -> LoggingManager:
    """
    Get or create global logging manager.

    Args:
        base_dir: Optional base directory for logs

    Returns:
        LoggingManager instance
    """
    global _logging_manager

    if _logging_manager is None:
        _logging_manager = LoggingManager(base_dir or "./logs")

    return _logging_manager


def performance_logged(operation: Optional[str] = None):
    """
    Decorator to automatically log performance of functions.

    Args:
        operation: Operation name (defaults to function name)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            success = True
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                manager = get_logging_manager()
                manager.log_performance(
                    op_name,
                    duration,
                    success,
                    {'args_count': len(args), 'kwargs_count': len(kwargs)}
                )

        return wrapper

    return decorator


def setup_logging(
        level: Union[int, str] = logging.INFO,
        console: bool = True,
        file: bool = True,
        structured: bool = False,
        async_mode: bool = False,
        base_dir: str = "./logs"
) -> LoggingManager:
    """
    Convenience function to set up logging with common configuration.

    Args:
        level: Log level
        console: Enable console output
        file: Enable file output
        structured: Enable structured JSON logging
        async_mode: Enable asynchronous logging
        base_dir: Base directory for log files

    Returns:
        Configured LoggingManager instance
    """
    manager = get_logging_manager(base_dir)

    # Update configuration
    manager.config['console_enabled'] = console
    manager.config['file_enabled'] = file
    manager.config['structured_enabled'] = structured
    manager.config['async_enabled'] = async_mode

    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    manager.set_level(level)

    # Reinitialize with new config
    manager._setup_root_logger()

    if structured:
        manager.enable_structured_logging()

    if async_mode:
        manager.enable_async_logging()

    return manager


# Configure library loggers
def configure_library_logging():
    """Configure logging for common libraries to reduce noise."""
    # Suppress verbose library logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('xarray').setLevel(logging.WARNING)
    logging.getLogger('dask').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('fsspec').setLevel(logging.WARNING)

    # Configure specific formatters for libraries if needed
    lib_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    for lib_logger_name in ['matplotlib', 'xarray', 'dask']:
        lib_logger = logging.getLogger(lib_logger_name)
        if lib_logger.handlers:
            for handler in lib_logger.handlers:
                handler.setFormatter(lib_formatter)
