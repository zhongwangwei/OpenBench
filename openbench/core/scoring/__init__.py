# -*- coding: utf-8 -*-
"""
OpenBench Scoring Module

This module provides performance scoring functions that convert
various metrics into normalized scores (0-1 scale) for model
evaluation and comparison.

The scoring module includes:
- Overall performance scoring
- Individual metric scoring
- Weighted scoring systems
- Multi-criteria evaluation scores
- Standardized scoring for model comparison

Classes:
    scores: Main scoring calculator class with all scoring methods

Examples:
    >>> from openbench.core.scoring import scores
    >>> scorer = scores()
    >>> overall_score = scorer.Overall_Score(simulation_data, observation_data)
"""

from .Mod_Scores import scores

__all__ = ['scores']

__version__ = '0.1'
__author__ = 'Zhongwang Wei'
__email__ = 'zhongwang007@gmail.com' 