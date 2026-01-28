#!/usr/bin/env python3
"""
Custom Exceptions for WSE Pipeline

Provides specific exception types for better error handling and debugging.
"""


class WSEError(Exception):
    """WSE Pipeline base exception.

    All custom exceptions in the WSE pipeline inherit from this class,
    allowing callers to catch all WSE-related errors with a single except clause.
    """
    pass


class ReaderError(WSEError):
    """Data reading error.

    Raised when there's an unrecoverable error while reading data files,
    such as permission denied or corrupted files.
    """
    pass


class ValidationError(WSEError):
    """Data validation error.

    Raised when data fails validation checks, such as invalid coordinates,
    missing required fields, or out-of-range values.
    """
    pass


class ConfigurationError(WSEError):
    """Configuration error.

    Raised when there's an error in the pipeline configuration,
    such as missing required settings or invalid parameter values.
    """
    pass


class DownloadError(WSEError):
    """Data download error.

    Raised when there's an error downloading data from remote sources,
    such as network failures or authentication issues.
    """
    pass
