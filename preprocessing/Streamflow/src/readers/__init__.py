"""Reader registry and factory."""

from typing import Dict, Optional, Type
import logging

from .base_reader import BaseReader

READERS: Dict[str, Type[BaseReader]] = {}


def register_reader(name: str):
    """Decorator to register a reader class."""
    def decorator(cls):
        READERS[name] = cls
        return cls
    return decorator


def get_reader(source: str, logger: Optional[logging.Logger] = None) -> BaseReader:
    """Factory: get reader instance by source name."""
    if source not in READERS:
        available = ", ".join(sorted(READERS.keys()))
        raise ValueError(f"Unknown source: '{source}'. Available: {available}")
    return READERS[source](logger=logger)
