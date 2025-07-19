# -*- coding: utf-8 -*-
"""
OpenBench Comparison Module

This module provides comparison analysis tools for model evaluation:
- Multi-model comparison analysis
- Land cover based grouping and comparison
- Climate zone based grouping and comparison
- Statistical comparison methods
- Visualization tools for comparison results

Classes:
    ComparisonProcessing: Main comparison processing class with comprehensive comparison methods
    LC_groupby: Land cover based grouping and comparison
    CZ_groupby: Climate zone based grouping and comparison

Functions:
    Various comparison methods for different analysis types

Examples:
    >>> from openbench.core.comparison import ComparisonProcessing
    >>> comparator = ComparisonProcessing(main_nml, scores, metrics)
    >>> comparator.scenarios_HeatMap_comparison(casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option)
"""

from .Mod_Comparison import ComparisonProcessing
from .Mod_Landcover_Groupby import LC_groupby
from .Mod_ClimateZone_Groupby import CZ_groupby

__all__ = [
    'ComparisonProcessing',
    'LC_groupby',
    'CZ_groupby'
]

__version__ = '1.0'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com' 