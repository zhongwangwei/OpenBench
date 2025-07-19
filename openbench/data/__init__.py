# -*- coding: utf-8 -*-
"""
OpenBench Data Module

This module provides data processing functionality including:
- Dataset processing and validation
- Data pipeline management
- Caching system
- File processing utilities
- Time handling utilities
- Unit conversion utilities

The data module is organized into:
- DatasetProcessing: Main data processing class
- Mod_Preprocessing: Data preprocessing functions
- Mod_CacheSystem: Multi-level caching system
- Mod_DataPipeline: Data processing pipeline
- Lib_Time: Time handling utilities
- Lib_FileProcessing: File processing utilities
- Lib_Unit: Unit conversion utilities
"""

# Import main classes
from .Mod_DatasetProcessing import (
    DatasetProcessing,
    BaseDatasetProcessing,
    StationDatasetProcessing,
    GridDatasetProcessing
)

from .Mod_Preprocessing import (
    check_required_nml,
    run_files_check
)

from .Mod_CacheSystem import (
    get_cache_manager,
    cached,
    DataCache
)

from .Mod_DataPipeline import (
    create_standard_pipeline,
    process_dataset,
    DataPipelineBuilder,
    DataValidationProcessor,
    CoordinateProcessor
)

from .Lib_Time import (
    timelib
)

from .Lib_FileProcessing import (
    FileProcessing
)

from .Lib_Unit import (
    UnitProcessing
)

__all__ = [
    'DatasetProcessing',
    'BaseDatasetProcessing', 
    'StationDatasetProcessing',
    'GridDatasetProcessing',
    'check_required_nml',
    'run_files_check',
    'get_cache_manager',
    'cached',
    'DataCache',
    'create_standard_pipeline',
    'process_dataset',
    'DataPipelineBuilder',
    'DataValidationProcessor',
    'CoordinateProcessor',
    'TimeProcessor',
    'time_utils',
    'FileProcessor',
    'file_utils',
    'UnitConverter',
    'unit_utils'
]

__version__ = '0.3'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com'

# Provide convenient access to the main classes
DataProcessing = DatasetProcessing
CacheSystem = DataCache
Pipeline = DataPipelineBuilder 