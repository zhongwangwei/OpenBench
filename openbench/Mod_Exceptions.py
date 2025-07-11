# -*- coding: utf-8 -*-
"""
Unified Exception Handling System for OpenBench

This module provides a comprehensive exception handling framework with
custom exception classes, decorators, and utilities for error management.

Author: OpenBench Contributors
Version: 1.0
Date: July 2025
"""

import logging
import traceback
import functools
from typing import Any, Dict, Optional, Callable, Union
from datetime import datetime


class OpenBenchException(Exception):
    """Base exception class for all OpenBench-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize OpenBench exception.
        
        Args:
            message: Human-readable error message
            context: Additional context information
            original_error: Original exception that caused this error
            error_code: Unique error code for categorization
        """
        self.message = message
        self.context = context or {}
        self.original_error = original_error
        self.error_code = error_code
        self.timestamp = datetime.now()
        
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format the complete error message."""
        msg = f"OpenBench Error: {self.message}"
        
        if self.error_code:
            msg = f"[{self.error_code}] {msg}"
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            msg += f" (Context: {context_str})"
        
        if self.original_error:
            msg += f" (Original: {type(self.original_error).__name__}: {self.original_error})"
        
        return msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None
        }


class DataProcessingError(OpenBenchException):
    """Exception for data processing and manipulation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATA_PROC", **kwargs)


class ConfigurationError(OpenBenchException):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG", **kwargs)


class EvaluationError(OpenBenchException):
    """Exception for evaluation process errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="EVAL", **kwargs)


class FileSystemError(OpenBenchException):
    """Exception for file system operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="FILESYSTEM", **kwargs)


class ValidationError(OpenBenchException):
    """Exception for data validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="VALIDATION", **kwargs)


class MetricsError(OpenBenchException):
    """Exception for metrics calculation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="METRICS", **kwargs)


class VisualizationError(OpenBenchException):
    """Exception for visualization and plotting errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="VISUALIZATION", **kwargs)


class ResourceError(OpenBenchException):
    """Exception for resource allocation and management errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RESOURCE", **kwargs)


def error_handler(
    reraise: bool = True,
    log_level: int = logging.ERROR,
    return_value: Any = None,
    error_types: Optional[tuple] = None
) -> Callable:
    """
    Decorator for standardized error handling.
    
    Args:
        reraise: Whether to reraise the exception after handling
        log_level: Logging level for error messages
        return_value: Value to return if exception is caught and not reraised
        error_types: Tuple of exception types to catch (default: catch all)
    
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should catch this specific error type
                if error_types and not isinstance(e, error_types):
                    raise
                
                # Create context information
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # Log the error
                logger = logging.getLogger(func.__module__)
                
                if isinstance(e, OpenBenchException):
                    # Log OpenBench exceptions with full context
                    logger.log(log_level, f"Error in {func.__name__}: {e.format_message()}")
                else:
                    # Convert regular exceptions to OpenBench exceptions
                    wrapped_error = OpenBenchException(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        context=context,
                        original_error=e
                    )
                    logger.log(log_level, wrapped_error.format_message())
                
                if reraise:
                    if isinstance(e, OpenBenchException):
                        raise
                    else:
                        # Wrap and reraise as OpenBench exception
                        raise OpenBenchException(
                            f"Error in {func.__name__}: {str(e)}",
                            context=context,
                            original_error=e
                        ) from e
                else:
                    return return_value
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_message: Optional[str] = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if function fails
        error_message: Custom error message prefix
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or default_return if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = logging.getLogger(__name__)
            message = error_message or f"Error executing {func.__name__}"
            logger.error(f"{message}: {str(e)}")
        
        return default_return


def validate_file_exists(file_path: str, error_message: Optional[str] = None) -> None:
    """
    Validate that a file exists, raising FileSystemError if not.
    
    Args:
        file_path: Path to the file to validate
        error_message: Custom error message
    
    Raises:
        FileSystemError: If file does not exist
    """
    import os
    
    if not os.path.exists(file_path):
        message = error_message or f"File not found: {file_path}"
        raise FileSystemError(message, context={'file_path': file_path})
    
    if not os.path.isfile(file_path):
        message = error_message or f"Path exists but is not a file: {file_path}"
        raise FileSystemError(message, context={'file_path': file_path})


def validate_directory_exists(dir_path: str, create: bool = False, error_message: Optional[str] = None) -> None:
    """
    Validate that a directory exists, optionally creating it.
    
    Args:
        dir_path: Path to the directory to validate
        create: Whether to create the directory if it doesn't exist
        error_message: Custom error message
    
    Raises:
        FileSystemError: If directory validation fails
    """
    import os
    
    if not os.path.exists(dir_path):
        if create:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                raise FileSystemError(f"Failed to create directory: {dir_path}", 
                                    context={'dir_path': dir_path}, original_error=e)
        else:
            message = error_message or f"Directory not found: {dir_path}"
            raise FileSystemError(message, context={'dir_path': dir_path})
    
    if not os.path.isdir(dir_path):
        message = error_message or f"Path exists but is not a directory: {dir_path}"
        raise FileSystemError(message, context={'dir_path': dir_path})


def validate_required_keys(data: Dict[str, Any], required_keys: list, context_name: str = "data") -> None:
    """
    Validate that required keys exist in a dictionary.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        context_name: Name of the data context for error messages
    
    Raises:
        ValidationError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        raise ValidationError(
            f"Missing required keys in {context_name}: {missing_keys}",
            context={'missing_keys': missing_keys, 'available_keys': list(data.keys())}
        )


def log_performance_warning(func_name: str, execution_time: float, threshold: float = 10.0) -> None:
    """
    Log performance warning if execution time exceeds threshold.
    
    Args:
        func_name: Name of the function
        execution_time: Execution time in seconds
        threshold: Time threshold for warning
    """
    if execution_time > threshold:
        logger = logging.getLogger(__name__)
        logger.warning(f"Performance warning: {func_name} took {execution_time:.2f}s (threshold: {threshold}s)")


# Context manager for error handling
class ErrorContext:
    """Context manager for scoped error handling."""
    
    def __init__(self, operation_name: str, reraise: bool = True):
        """
        Initialize error context.
        
        Args:
            operation_name: Name of the operation for error messages
            reraise: Whether to reraise exceptions
        """
        self.operation_name = operation_name
        self.reraise = reraise
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if isinstance(exc_val, OpenBenchException):
                self.logger.error(f"Error in {self.operation_name}: {exc_val.format_message()}")
            else:
                self.logger.error(f"Unexpected error in {self.operation_name}: {str(exc_val)}")
            
            if not self.reraise:
                return True  # Suppress the exception
        
        return False  # Let the exception propagate


# Global error handler for uncaught exceptions
def setup_global_error_handler():
    """Setup global exception handler for uncaught exceptions."""
    import sys
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't handle keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = logging.getLogger('openbench.global')
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception