# -*- coding: utf-8 -*-
"""
OpenBench Core Module

This module provides the core functionality for model evaluation including:
- Statistical metrics for model-observation comparison
- Performance scoring functions for standardized evaluation
- Advanced statistical analysis tools
- Evaluation engines for different data types

The core module is organized into submodules:
- metrics: Statistical metrics calculation
- scores: Performance scoring functions
- statistics: Statistical analysis functions
- evaluation: Evaluation engines and processing classes
- comparison: Multi-model comparison tools
"""

from .metrics import metrics
from .scores import scores
from openbench import __version__

from .evaluation import Evaluation_grid, Evaluation_stn
from .climatezone_groupby import CZ_groupby
from .comparison import ComparisonProcessing
from .landcover_groupby import LC_groupby

_HAS_EVALUATION = True
_HAS_COMPARISON = True

__all__ = [
    "metrics",
    "scores",
    "Evaluation_grid",
    "Evaluation_stn",
    "ComparisonProcessing",
    "LC_groupby",
    "CZ_groupby",
    "__version__",
]

__author__ = "Zhongwang Wei"
__email__ = "zhongwang007@gmail.com"

# Provide convenient access to the main classes
Metrics = metrics
Scores = scores
