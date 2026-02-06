#!/usr/bin/env python3
"""
Logging System for Streamflow Pipeline
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    SYMBOLS = {
        'DEBUG': '[D]',
        'INFO': '[+]',
        'WARNING': '[!]',
        'ERROR': '[X]',
        'CRITICAL': '[!!]',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        symbol = self.SYMBOLS.get(record.levelname, '')
        record.symbol = symbol
        record.color = color
        record.reset = self.RESET
        return super().format(record)


def get_logger(name: str,
               log_file: Optional[str] = None,
               level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a configured logger.

    Args:
        name: Logger name
        log_file: Log file path (optional)
        level: Log level

    Returns:
        Configured logger
    """
    log_dir = str(Path(log_file).parent) if log_file else None
    return setup_logger(name, log_dir=log_dir, log_level=level)


def setup_logger(name: str,
                 log_dir: Optional[str] = None,
                 log_level: str = "INFO",
                 dataset_name: str = "streamflow") -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_dir: Log directory
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        dataset_name: Dataset name (used in log file name)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler (colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        '%(color)s%(symbol)s%(reset)s %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{timestamp}_{dataset_name}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Log file: {log_file}")

    return logger


class ProgressLogger:
    """Progress logger for batch operations."""

    def __init__(self, logger: logging.Logger, total: int, prefix: str = ""):
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self.current = 0

    def update(self, current: Optional[int] = None, message: str = ""):
        """Update progress."""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        # Log every 100 items or on completion
        if self.current % 100 == 0 or self.current == self.total:
            pct = self.current / self.total * 100
            self.logger.info(
                f"{self.prefix}[{self.current}/{self.total}] {pct:.1f}% {message}"
            )

    def done(self, message: str = "Complete"):
        """Mark as complete."""
        self.logger.info(f"{self.prefix}{message} ({self.total} items)")
