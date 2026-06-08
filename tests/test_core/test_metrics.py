"""Tests for metrics module — verify numerical correctness."""

import numpy as np
import pandas as pd
import xarray as xr


def make_da(values):
    """Helper: wrap a list of floats into a time-indexed xr.DataArray."""
    times = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    return xr.DataArray(values, coords={"time": times}, dims=["time"])


def make_daily_da(values, start):
    times = pd.date_range(start, periods=len(values), freq="D")
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


def test_percent_change_metrics_do_not_mutate_inputs_with_nan_pairs():
    """pc_* metrics should mask invalid pairs without editing caller arrays."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 4.0])
    sim = make_da([1.0, np.nan, 3.0])
    obs_before = obs.copy(deep=True)
    sim_before = sim.copy(deep=True)

    assert abs(float(m.pc_max(sim, obs)) - (-0.25)) < 1e-10
    assert float(m.pc_min(sim, obs)) == 0.0
    assert abs(float(m.pc_ampli(sim, obs)) - (-1.0 / 3.0)) < 1e-10

    xr.testing.assert_identical(obs, obs_before)
    xr.testing.assert_identical(sim, sim_before)


def test_pc_max_negative_observed_max_uses_absolute_denominator():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([-4.0, -2.0])
    sim = make_da([-3.0, -1.0])

    assert float(m.pc_max(sim, obs)) == 0.5


def test_hydrologic_metrics_align_on_inner_time_before_calculating():
    """br2/cp/dr/APFB should align timestamps instead of requiring identical axes."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_daily_da([1.0, 2.0, 3.0, 4.0], "2000-01-01")
    sim = make_daily_da([2.0, 3.0, 4.0, 5.0], "2000-01-02")

    assert abs(float(m.br2(sim, obs)) - 1.0) < 1e-10
    assert abs(float(m.cp(sim, obs)) - 1.0) < 1e-10
    assert abs(float(m.dr(sim, obs)) - 1.0) < 1e-10
    assert abs(float(m.APFB(sim, obs))) < 1e-10


def test_efficiency_metrics_use_same_valid_pairs_for_all_terms():
    """KGE/rv must compute means/stds over the same finite sim/obs pairs."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0])
    sim = make_da([2.0, np.nan, 4.0])

    # Valid pairs are (sim, obs) = (2, 1), (4, 3):
    # corr=1, alpha=1, beta=mean(sim)/mean(obs)=3/2, so KGE=0.5.
    assert abs(float(m.KGE(sim, obs)) - 0.5) < 1e-10
    assert abs(float(m.rv(sim, obs))) < 1e-10


def test_br2_returns_nan_for_constant_observations():
    """br2 is undefined for constant observed series; it should not raise."""
    from openbench.core.metrics import metrics

    m = metrics()
    result = m.br2(make_da([1.0, 2.0, 3.0]), make_da([1.0, 1.0, 1.0]))

    assert np.isnan(float(result))


def test_smpi_uses_climatological_mean_difference_not_instantaneous_error():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([0.0, 2.0])
    sim = make_da([2.0, 0.0])

    value, lower, upper = m.smpi(sim, obs, n_bootstrap=8)

    assert abs(float(value)) < 1e-12
    assert np.isfinite(lower) or np.isnan(lower)
    assert np.isfinite(upper) or np.isnan(upper)


def test_dr_returns_nan_for_constant_observations_with_error():
    from openbench.core.metrics import metrics

    m = metrics()
    result = m.dr(make_da([2.0, 3.0, 4.0]), make_da([1.0, 1.0, 1.0]))

    assert np.isnan(float(result))


def test_dr_large_error_branch_is_negative_not_near_perfect():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([0.0, 1.0, 2.0])
    sim = make_da([100.0, 100.0, 100.0])

    result = float(m.dr(sim, obs))

    assert result < 0
    assert abs(result - ((4.0 / 297.0) - 1.0)) < 1e-12


def test_kappa_coeff_handles_multidimensional_time_series():
    from openbench.core.metrics import metrics

    m = metrics()
    times = pd.date_range("2000-01-01", periods=4)
    sim = xr.DataArray(
        np.array([[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]], [[1, 1], [1, 0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0, 1], "lon": [10, 20]},
    )
    obs = sim.copy()

    result = m.kappa_coeff(sim, obs)

    assert result.dims == ("lat", "lon")
    assert np.allclose(result, 1.0)
