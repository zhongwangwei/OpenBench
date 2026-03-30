"""Tests for metrics module — verify numerical correctness."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def make_da(values):
    """Helper: wrap a list of floats into a time-indexed xr.DataArray."""
    times = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    return xr.DataArray(values, coords={"time": times}, dims=["time"])


def test_metrics_class_exists():
    """Verify metrics class can be imported."""
    from openbench.core.metrics import metrics
    assert metrics is not None


def test_bias_calculation():
    """Test bias metric: mean(sim - obs)."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([1.5, 2.5, 3.5, 4.5, 5.5])
    result = float(m.bias(sim, obs))
    assert abs(result - 0.5) < 1e-10


def test_rmse_calculation():
    """Test RMSE metric."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(m.RMSE(sim, obs))
    assert abs(result) < 1e-10  # Perfect match = 0 RMSE


def test_rmse_with_error():
    """Test RMSE with known error."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0])
    sim = make_da([2.0, 3.0, 4.0])  # All off by 1
    result = float(m.RMSE(sim, obs))
    assert abs(result - 1.0) < 1e-10


def test_correlation_calculation():
    """Test correlation metric."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect linear
    result = float(m.correlation(sim, obs))
    assert abs(result - 1.0) < 1e-10


def test_correlation_negative():
    """Test negative correlation."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([5.0, 4.0, 3.0, 2.0, 1.0])  # Inverse
    result = float(m.correlation(sim, obs))
    assert abs(result - (-1.0)) < 1e-10
