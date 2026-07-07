"""Tests for compute expression executor."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.compute import ComputeError, execute_compute


def _make_ds():
    """Create a test dataset."""
    return xr.Dataset(
        {
            "a": xr.DataArray(np.array([1.0, 2.0, 3.0])),
            "b": xr.DataArray(np.array([10.0, 20.0, 30.0])),
            "rain": xr.DataArray(np.array([0.5, 1.0, 0.3])),
            "snow": xr.DataArray(np.array([0.1, 0.0, 0.2])),
        }
    )


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
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.array([1.0, np.nan, 3.0])),
        }
    )
    result = execute_compute(ds, "ds['x'].fillna(0)", "test")
    np.testing.assert_array_equal(result.values, [1.0, 0.0, 3.0])


def test_compute_rejects_io_calls_from_allowed_roots(tmp_path):
    ds = _make_ds()
    out = tmp_path / "out.nc"

    with pytest.raises(ComputeError, match="xarray function 'open_dataset' is not allowed"):
        execute_compute(ds, f"xr.open_dataset('{out}')", "test")

    with pytest.raises(ComputeError, match="numpy function 'fromfile' is not allowed"):
        execute_compute(ds, f"np.fromfile('{out}', dtype=np.uint8)", "test")

    with pytest.raises(ComputeError, match="method 'to_netcdf' is not allowed"):
        execute_compute(ds, f"ds['a'].to_netcdf('{out}')", "test")

    assert not out.exists()


def test_compute_allows_catalog_method_chain():
    ds = xr.Dataset({"resp": xr.DataArray(np.array([31536000.0]), attrs={"units": "gC year-1"})})

    result = execute_compute(
        ds,
        "ds['resp'] / 31536000.0 if 'year' in ds['resp'].attrs.get('units', '').lower() else ds['resp']",
        "Respiration",
    )

    np.testing.assert_allclose(result.values, [1.0])


def test_te_total_runoff_compute_handles_zdepth_dimension():
    from openbench.data.registry.manager import get_registry

    profile = get_registry().get_model("TE")
    expression = profile.variables["Total_Runoff"].compute
    ds = xr.Dataset(
        {
            "RUNOFF": (
                ["zdepth", "time"],
                np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
            )
        },
        coords={"zdepth": [0, 1], "time": [0, 1, 2]},
    )

    result = execute_compute(ds, expression, "Total_Runoff")

    assert result.dims == ("time",)
    np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])


def test_compute_rejects_unknown_identifier():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="identifier 'os' is not allowed"):
        execute_compute(ds, "os.system('touch /tmp/openbench_pwn')", "test")


def test_compute_rejects_invalid_assignment_target():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="Invalid assignment target"):
        execute_compute(ds, "ds['a'] = ds['b']; ds['a']", "test")


def test_compute_validation_allows_explicit_extra_names():
    from openbench.data.compute import _validate_expression

    _validate_expression("value * 12.011 - f_assim", allowed_names={"value", "f_assim", "np"})
    with pytest.raises(ComputeError, match="identifier 'missing' is not allowed"):
        _validate_expression("value + missing", allowed_names={"value", "np"})


def test_compute_supports_dataset_membership_checks():
    ds = xr.Dataset(
        {
            "RUNOFF": xr.DataArray(np.array([1.0, 2.0, 3.0])),
            "fallback": xr.DataArray(np.array([10.0, 20.0, 30.0])),
        }
    )

    result = execute_compute(
        ds,
        "ds['RUNOFF'] if 'RUNOFF' in ds else ds['fallback']",
        "Runoff",
    )

    np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])


def test_compute_membership_checks_are_case_insensitive():
    ds = xr.Dataset({"runoff": xr.DataArray(np.array([4.0, 5.0]))})

    result = execute_compute(ds, "ds['RUNOFF'] if 'RUNOFF' in ds else 0", "Runoff")

    np.testing.assert_array_equal(result.values, [4.0, 5.0])
