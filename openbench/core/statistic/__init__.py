# -*- coding: utf-8 -*-
"""
OpenBench Statistics Module

This module provides advanced statistical analysis tools for model evaluation
and data processing.

The statistics module includes:
- Basic statistical functions (mean, median, max, min, sum)
- Correlation analysis (Pearson, Spearman)
- Advanced statistical tests (ANOVA, Mann-Kendall)
- Distribution analysis (Hellinger distance, Z-score)
- Time series analysis (autocorrelation, rolling statistics)
- Multi-variate analysis (Partial Least Squares, Three Cornered Hat)

Classes:
    statistics_calculate: Main statistics calculator with all statistical methods

Functions:
    stat_correlation: Correlation analysis
    stat_anova: Analysis of variance
    stat_mann_kendall_trend_test: Trend analysis
    stat_hellinger_distance: Distribution comparison
    stat_partial_least_squares_regression: Multi-variate regression
    stat_three_cornered_hat: Uncertainty analysis

Examples:
    >>> from openbench.core.statistic import statistics_calculate
    >>> stats = statistics_calculate()
    >>> correlation = stats.stat_correlation(data1, data2)
"""

from .base import statistics_calculate
from .stat_correlation import stat_correlation
from .stat_standard_deviation import stat_standard_deviation
from .stat_z_score import stat_z_score
from .stat_Basic import stat_mean, stat_median, stat_max, stat_min, stat_sum
from .stat_covariance import stat_covariance
from .stat_autocorrelation import stat_autocorrelation
from .stat_diff import stat_diff
from .stat_resample import stat_resample
from .stat_rolling import stat_rolling
from .stat_functional_response import stat_functional_response
from .stat_hellinger_distance import stat_hellinger_distance
from .stat_three_cornered_hat import stat_three_cornered_hat
from .stat_partial_least_squares_regression import stat_partial_least_squares_regression
from .stat_mann_kendall_trend_test import stat_mann_kendall_trend_test
from .stat_False_Discovery_Rate import stat_False_Discovery_Rate
from .stat_anova import stat_anova

# Attach all statistical functions to the main calculator class
statistics_calculate.stat_correlation = stat_correlation
statistics_calculate.stat_standard_deviation = stat_standard_deviation
statistics_calculate.stat_z_score = stat_z_score
statistics_calculate.stat_mean = stat_mean
statistics_calculate.stat_median = stat_median
statistics_calculate.stat_max = stat_max
statistics_calculate.stat_min = stat_min
statistics_calculate.stat_sum = stat_sum
statistics_calculate.stat_covariance = stat_covariance
statistics_calculate.stat_autocorrelation = stat_autocorrelation
statistics_calculate.stat_diff = stat_diff
statistics_calculate.stat_resample = stat_resample
statistics_calculate.stat_rolling = stat_rolling
statistics_calculate.stat_functional_response = stat_functional_response
statistics_calculate.stat_hellinger_distance = stat_hellinger_distance
statistics_calculate.stat_three_cornered_hat = stat_three_cornered_hat
statistics_calculate.stat_partial_least_squares_regression = stat_partial_least_squares_regression
statistics_calculate.stat_mann_kendall_trend_test = stat_mann_kendall_trend_test
statistics_calculate.stat_False_Discovery_Rate = stat_False_Discovery_Rate
statistics_calculate.stat_anova = stat_anova

__all__ = ['statistics_calculate']

__version__ = '0.2'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com'
