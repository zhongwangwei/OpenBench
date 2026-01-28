"""
WSE Pipeline - Water Surface Elevation Processing

5-stage pipeline for processing altimetry data:
0. download - Data download and completeness check
1. validate - Data validation and EGM calculation
2. cama - CaMa-Flood station allocation
3. reserved - Reserved for future extensions
4. merge - Merge and output

Usage:
    from src.pipeline import Pipeline

    config = {'data_root': '/path/to/data', 'output_dir': './output'}
    pipeline = Pipeline(config)
    results = pipeline.run(['hydroweb', 'cgls'])
"""

from .pipeline import Pipeline
from .exceptions import (
    WSEError,
    ReaderError,
    ValidationError,
    ConfigurationError,
    DownloadError,
)

__version__ = '0.1.0'
__all__ = [
    'Pipeline',
    'WSEError',
    'ReaderError',
    'ValidationError',
    'ConfigurationError',
    'DownloadError',
]
