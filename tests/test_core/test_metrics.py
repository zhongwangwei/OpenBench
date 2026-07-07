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


def test_absolute_percent_bias_uses_absolute_observed_sum():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([-2.0, -3.0])
    sim = make_da([-1.0, -2.0])

    assert float(m.absolute_percent_bias(sim, obs)) == 40.0


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


def test_br2_negative_slope_stays_in_zero_to_r2_range():
    """br2 must use |slope| (Krause 2005): a perfectly anti-correlated series
    has slope=-1, r²=1, so br2=1 — not a negative value (L3)."""
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 3.0, 4.0])
    sim = make_da([4.0, 3.0, 2.0, 1.0])  # slope = -1, r² = 1

    value = float(m.br2(sim, obs))
    assert 0.0 <= value <= 1.0
    assert abs(value - 1.0) < 1e-10


def test_smpi_uses_climatological_mean_difference_not_instantaneous_error():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([0.0, 2.0])
    sim = make_da([2.0, 0.0])

    value, lower, upper = m.smpi(sim, obs, n_bootstrap=8)

    assert abs(float(value)) < 1e-12
    assert np.isfinite(lower) or np.isnan(lower)
    assert np.isfinite(upper) or np.isnan(upper)


def test_taylor_grid_summary_uses_pairwise_mask_for_all_three_terms():
    from openbench.core._comparison_taylor import _taylor_grid_summary_statistics
    from openbench.core.metrics import metrics

    times = pd.date_range("2000-01-01", periods=3, freq="MS")
    sim = xr.DataArray(
        np.array(
            [
                [[1.0, 10.0]],
                [[2.0, np.nan]],
                [[3.0, 30.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0, 1.0]},
    )
    obs = xr.DataArray(
        np.array(
            [
                [[1.0, 10.0]],
                [[2.0, 20.0]],
                [[3.0, 30.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0, 1.0]},
    )

    summary = _taylor_grid_summary_statistics(
        sim,
        obs,
        metric_handler=metrics(),
        weight="none",
    )

    assert abs(summary.std_sim - summary.std_ref) < 1e-12
    assert abs(summary.cor_sim - 1.0) < 1e-12
    assert abs(summary.diagram_crmsd) < 1e-12


def test_smpi_grid_summary_respects_area_weights():
    from openbench.core._comparison_smpi import _smpi_grid_summary

    times = pd.date_range("2000-01-01", periods=3, freq="MS")
    lat = [0.0, 80.0]
    lon = [0.0]
    obs = xr.DataArray(
        np.array(
            [
                [[0.0], [0.0]],
                [[1.0], [1.0]],
                [[2.0], [2.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
    )
    sim = obs.copy(deep=True)
    sim.loc[{"lat": 0.0}] = obs.sel(lat=0.0) + 2.0
    sim.loc[{"lat": 80.0}] = obs.sel(lat=80.0)

    smpi, _lower, _upper, _grid = _smpi_grid_summary(sim, obs, weight="area", n_bootstrap=0)

    area_weights = np.cos(np.deg2rad(np.array(lat)))
    expected = 4.0 / obs.var(dim="time", ddof=1).isel(lon=0).values[0]
    expected = expected * area_weights[0] / area_weights.sum()
    assert abs(float(smpi) - expected) < 1e-12


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


def test_mfm_components_recombine_to_mfm_value():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([1.0, 2.0, 4.0, 8.0, 16.0])
    sim = make_da([1.1, 2.4, 3.6, 7.2, 15.5])

    omega = m.MFM_omega(sim, obs, phase=False)
    varphi = m.MFM_varphi(sim, obs)
    eta = m.MFM_eta(sim, obs)
    recomposed = 1 - np.sqrt(((1 - omega) ** 2 + (1 - varphi) ** 2 + (1 - eta) ** 2) / 3)

    xr.testing.assert_allclose(recomposed, m.MFM(sim, obs, phase=False))


def test_mfm_component_domains_only_require_observed_mean_for_omega():
    from openbench.core.metrics import metrics

    m = metrics()
    obs = make_da([-1.0, 0.0, 1.0])
    sim = make_da([-0.5, 0.5, 1.5])

    assert np.isnan(float(m.MFM_omega(sim, obs)))
    assert np.isfinite(float(m.MFM_varphi(sim, obs)))
    assert np.isfinite(float(m.MFM_eta(sim, obs)))
    assert np.isnan(float(m.MFM(sim, obs)))
