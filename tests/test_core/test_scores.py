"""Tests for scores module — verify normalized score calculations."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def make_da(values):
    """Helper: wrap a list of floats into a time-indexed xr.DataArray."""
    times = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    return xr.DataArray(values, coords={"time": times}, dims=["time"])


def test_scores_class_exists():
    """Verify scores class can be imported."""
    from openbench.core.scores import scores
    assert scores is not None


def test_nbias_score_perfect():
    """Perfect match should give score close to 1."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(s.nBiasScore(sim, obs))
    assert result > 0.99  # Should be very close to 1


def test_nbias_score_range():
    """Score should be between 0 and 1."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([10.0, 20.0, 30.0, 40.0, 50.0])  # Very biased
    result = float(s.nBiasScore(sim, obs))
    assert 0.0 <= result <= 1.0


def test_nrmse_score_perfect():
    """Perfect match should give score close to 1."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(s.nRMSEScore(sim, obs))
    assert result > 0.99
