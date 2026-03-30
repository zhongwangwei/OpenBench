"""Tests for compute expression executor."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.compute import ComputeError, execute_compute


def _make_ds():
    """Create a test dataset."""
    return xr.Dataset({
        "a": xr.DataArray(np.array([1.0, 2.0, 3.0])),
        "b": xr.DataArray(np.array([10.0, 20.0, 30.0])),
        "rain": xr.DataArray(np.array([0.5, 1.0, 0.3])),
        "snow": xr.DataArray(np.array([0.1, 0.0, 0.2])),
    })


def test_simple_expression():
    ds = _make_ds()
    result = execute_compute(ds, "ds['a'] + ds['b']", "test")
    np.testing.assert_array_equal(result.values, [11.0, 22.0, 33.0])


def test_multi_step_expression():
    ds = _make_ds()
    result = execute_compute(ds, "total = ds['a'] + ds['b']; total * 2", "test")
    np.testing.assert_array_equal(result.values, [22.0, 44.0, 66.0])


def test_division():
    ds = _make_ds()
    result = execute_compute(ds, "ds['b'] / ds['a']", "test")
    np.testing.assert_array_equal(result.values, [10.0, 10.0, 10.0])


def test_precipitation_compute():
    ds = _make_ds()
    result = execute_compute(ds, "ds['rain'] + ds['snow']", "Precipitation")
    np.testing.assert_array_almost_equal(result.values, [0.6, 1.0, 0.5])


def test_numpy_available():
    ds = _make_ds()
    result = execute_compute(ds, "(ds['a']**2 + ds['b']**2)**0.5", "magnitude")
    assert result.values[0] == pytest.approx(np.sqrt(101), rel=1e-5)


def test_missing_variable_error():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="not found"):
        execute_compute(ds, "ds['nonexistent'] + ds['a']", "test")


def test_empty_expression_error():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="Empty"):
        execute_compute(ds, "", "test")


def test_fillna():
    ds = xr.Dataset({
        "x": xr.DataArray(np.array([1.0, np.nan, 3.0])),
    })
    result = execute_compute(ds, "ds['x'].fillna(0)", "test")
    np.testing.assert_array_equal(result.values, [1.0, 0.0, 3.0])
