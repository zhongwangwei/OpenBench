# -*- coding: utf-8 -*-
"""
OpenBench Package

Land Surface Model Benchmarking System with unified API.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

import sys
import os

# Add openbench directory to Python path
openbench_dir = os.path.join(os.path.dirname(__file__), 'openbench')
sys.path.insert(0, openbench_dir)

# Import main classes and functions
try:
    from openbench.openbench_api import OpenBench, run_evaluation, create_openbench
    __all__ = ['OpenBench', 'run_evaluation', 'create_openbench']
except ImportError as e:
    # Fallback - try direct import
    try:
        import openbench.openbench_api as api
        OpenBench = api.OpenBench
        run_evaluation = api.run_evaluation
        create_openbench = api.create_openbench
        __all__ = ['OpenBench', 'run_evaluation', 'create_openbench']
    except ImportError as e2:
        print(f"Warning: Could not import OpenBench API: {e}, {e2}")
        __all__ = []

__version__ = "2.0.0"
__author__ = "OpenBench Contributors"