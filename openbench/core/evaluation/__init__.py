# -*- coding: utf-8 -*-
"""
OpenBench Evaluation Module

This module provides evaluation engines and processing classes for model evaluation:
- Grid evaluation for spatial data analysis
- Station evaluation for point data analysis
- Modular evaluation engine with pluggable metrics
- Unified interfaces for different evaluation types

Classes:
    Evaluation_grid: Grid-based evaluation processing
    Evaluation_stn: Station-based evaluation processing
    ModularEvaluationEngine: Modular evaluation engine with pluggable metrics
    GridEvaluationEngine: Specialized engine for gridded data
    StationEvaluationEngine: Specialized engine for station data

Functions:
    create_evaluation_engine: Factory function to create evaluation engines
    evaluate_datasets: Convenience function for dataset evaluation
"""

from .Mod_Evaluation import Evaluation_grid, Evaluation_stn
from .Mod_EvaluationEngine import (
    ModularEvaluationEngine,
    GridEvaluationEngine,
    StationEvaluationEngine,
    create_evaluation_engine,
    evaluate_datasets
)

__all__ = [
    'Evaluation_grid',
    'Evaluation_stn', 
    'ModularEvaluationEngine',
    'GridEvaluationEngine',
    'StationEvaluationEngine',
    'create_evaluation_engine',
    'evaluate_datasets'
]

__version__ = '1.0'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com' 