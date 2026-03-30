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

# Import evaluation classes
try:
    from .evaluation import Evaluation_grid, Evaluation_stn
    from .evaluation_engine import (
        GridEvaluationEngine,
        ModularEvaluationEngine,
        StationEvaluationEngine,
        create_evaluation_engine,
        evaluate_datasets,
    )
    _HAS_EVALUATION = True
except ImportError:
    _HAS_EVALUATION = False
    Evaluation_grid = None
    Evaluation_stn = None
    ModularEvaluationEngine = None
    GridEvaluationEngine = None
    StationEvaluationEngine = None
    def create_evaluation_engine(*args, **kwargs):
        return None
    def evaluate_datasets(*args, **kwargs):
        return {}

# Import comparison classes
try:
    from .climatezone_groupby import CZ_groupby
    from .comparison import ComparisonProcessing
    from .landcover_groupby import LC_groupby
    _HAS_COMPARISON = True
except ImportError:
    _HAS_COMPARISON = False
    ComparisonProcessing = None
    LC_groupby = None
    CZ_groupby = None

__all__ = [
    'metrics', 'scores',
    'Evaluation_grid', 'Evaluation_stn',
    'ModularEvaluationEngine', 'GridEvaluationEngine', 'StationEvaluationEngine',
    'create_evaluation_engine', 'evaluate_datasets',
    'ComparisonProcessing', 'LC_groupby', 'CZ_groupby'
]

__version__ = '0.3'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com'

# Provide convenient access to the main classes
Metrics = metrics
Scores = scores
