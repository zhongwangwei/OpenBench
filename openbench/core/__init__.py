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
- scoring: Performance scoring functions
- statistic: Statistical analysis functions
- evaluation: Evaluation engines and processing classes
"""

from .metrics import metrics
from .scoring import scores
from .statistic import statistics_calculate as statistic

# Import evaluation classes
try:
    from .evaluation.Mod_Evaluation import Evaluation_grid, Evaluation_stn
    from .evaluation.Mod_EvaluationEngine import (
        ModularEvaluationEngine,
        GridEvaluationEngine,
        StationEvaluationEngine,
        create_evaluation_engine,
        evaluate_datasets
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
    from .comparison.Mod_Comparison import ComparisonProcessing
    from .comparison.Mod_Landcover_Groupby import LC_groupby
    from .comparison.Mod_ClimateZone_Groupby import CZ_groupby
    _HAS_COMPARISON = True
except ImportError:
    _HAS_COMPARISON = False
    ComparisonProcessing = None
    LC_groupby = None
    CZ_groupby = None

__all__ = [
    'metrics', 'scores', 'statistic',
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
Statistic = statistic 