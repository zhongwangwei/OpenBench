"""Custom exceptions for Streamflow Pipeline."""


class StreamflowError(Exception):
    """Base exception for all streamflow pipeline errors."""
    pass


class ReaderError(StreamflowError):
    """Unrecoverable error reading data files."""
    pass


class ValidationError(StreamflowError):
    """Data validation failure."""
    pass


class ConfigurationError(StreamflowError):
    """Configuration error."""
    pass


class DownloadError(StreamflowError):
    """Data download or extraction error."""
    pass
