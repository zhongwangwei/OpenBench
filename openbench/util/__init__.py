# -*- coding: utf-8 -*-
"""
OpenBench Utility Modules

This package contains utility modules for OpenBench including:
- Exception handling (Mod_Exceptions)
- Type conversion (Mod_Converttype)
- Interfaces and base classes (Mod_Interfaces)
- Logging system (Mod_LoggingSystem)
- Output management (Mod_OutputManager)
- Parallel processing (Mod_ParallelEngine)
- API services (Mod_APIService)
- Dataset loading with chunking (Mod_DatasetLoader)
"""

# Import key utility functions and classes
try:
    from .Mod_Exceptions import OpenBenchError, error_handler
    from .Mod_Converttype import Convert_Type
    from .Mod_LoggingSystem import get_logging_manager
    from .Mod_OutputManager import ModularOutputManager
    from .Mod_ParallelEngine import ParallelEngine
    from .Mod_CacheCleanup import cleanup_all_cache, cleanup_pycache, get_cache_size
    from .Mod_DatasetLoader import open_dataset, open_mfdataset, load_and_compute, cached_glob, clear_glob_cache
except ImportError:
    # Graceful handling if some modules are not available
    pass

__all__ = [
    'OpenBenchError',
    'error_handler',
    'Convert_Type',
    'get_logging_manager',
    'ModularOutputManager',
    'ParallelEngine',
    'cleanup_all_cache',
    'cleanup_pycache',
    'get_cache_size',
    'open_dataset',
    'open_mfdataset',
    'load_and_compute',
    'cached_glob',
    'clear_glob_cache',
]