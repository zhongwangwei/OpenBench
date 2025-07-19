# -*- coding: utf-8 -*-
"""
OpenBench Metrics Module

This module provides comprehensive statistical metrics for evaluating
model performance against observational data.

The metrics module includes:
- Correlation analysis (Pearson, Spearman, Kendall)
- Error metrics (RMSE, MAE, Bias)
- Efficiency metrics (Nash-Sutcliffe, Kling-Gupta)
- Distribution metrics (PDF, CDF comparisons)
- Spatial and temporal analysis metrics

Classes:
    metrics: Main metrics calculator class with all statistical methods

Examples:
    >>> from openbench.core.metrics import metrics
    >>> calc = metrics()
    >>> rmse = calc.rmse(simulation_data, observation_data)
"""

from .Mod_Metrics import metrics

__all__ = ['metrics']

__version__ = '0.2'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com' 