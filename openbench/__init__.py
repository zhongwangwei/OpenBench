# -*- coding: utf-8 -*-
"""
OpenBench - Land Surface Model Benchmarking System

A comprehensive evaluation framework for land surface models with modular
architecture, parallel processing, and standardized interfaces.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

try:
    from .openbench_api import OpenBench, run_evaluation, create_openbench
    __all__ = ['OpenBench', 'run_evaluation', 'create_openbench', '__version__', '__author__']
except ImportError as e:
    # Fallback: create minimal interface if full import fails
    print(f"Warning: Full OpenBench import failed ({e}), using minimal interface")
    
    class OpenBench:
        """Minimal OpenBench interface when full import fails."""
        def __init__(self, config=None):
            self.config = config or {}
            print("OpenBench minimal mode - some features may be limited")
        
        @classmethod
        def from_config(cls, config_path):
            return cls({'config_path': config_path})
        
        @classmethod  
        def from_dict(cls, config_dict):
            return cls(config_dict)
        
        def run(self, **kwargs):
            raise ImportError("Full OpenBench functionality requires resolving import dependencies")
        
        def get_system_info(self):
            return {"version": "2.0.0", "mode": "minimal", "modules_available": False}
    
    def run_evaluation(*args, **kwargs):
        raise ImportError("run_evaluation requires full OpenBench import")
    
    def create_openbench(config=None):
        return OpenBench(config)
    
    __all__ = ['OpenBench', 'run_evaluation', 'create_openbench', '__version__', '__author__']

__version__ = "2.0.0"
__author__ = "OpenBench Contributors"
__email__ = "openbench@example.com"

# Package metadata
__title__ = "OpenBench"
__description__ = "Land Surface Model Benchmarking System"
__url__ = "https://github.com/openbench/openbench"
__license__ = "MIT"