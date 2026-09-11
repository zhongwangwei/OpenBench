"""Appendix metric formulas not covered by the legacy metric tests."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core.metrics import metrics


def da(values):
    return xr.DataArray(
        values,
        coords={"time": pd.date_range("2001-01-01", periods=len(values), freq="D")},
        dims=["time"],
    )


def assert_close(actual, expected):
    assert np.isclose(float(actual), expected, rtol=1e-10, atol=1e-10)


def test_appendix_error_and_variability_metrics():
    m = metrics()
    obs = da([1.0, 2.0, 3.0, 4.0])
    sim = da([2.0, 2.0, 4.0, 4.0])

    mse = np.mean((sim.values - obs.values) ** 2)
    rss = np.sum((sim.values - obs.values) ** 2)
    rmse = np.sqrt(mse)
    obs_mean = obs.values.mean()
    obs_std_sum = np.sqrt(np.sum((obs.values - obs_mean) ** 2))

    assert_close(m.MSE(sim, obs), mse)
    assert_close(m.RSS(sim, obs), rss)
    assert_close(m.NRMSE(sim, obs), rmse / abs(obs_mean))
    assert_close(m.RSR(sim, obs), np.sqrt(rss) / obs_std_sum)
    assert_close(m.NMAE(sim, obs), np.sum(abs(sim.values - obs.values)) / np.sum(abs(obs.values)))
    assert_close(m.rSD(sim, obs), np.std(sim.values) / np.std(obs.values))


def test_appendix_flow_bias_metrics_use_observed_thresholds():
    m = metrics()
    obs = da([1.0, 2.0, 3.0, 4.0, 100.0])
    sim = da([1.0, 3.0, 3.0, 5.0, 120.0])

    assert_close(m.PBIAS_HF(sim, obs, quantile=0.8), 20.0)
    assert_close(m.PBIAS_LF(sim, obs, quantile=0.2), 0.0)

    qs_h, qs_l = np.quantile(sim.values, [0.66, 0.33])
    qo_h, qo_l = np.quantile(obs.values, [0.66, 0.33])
    expected = 100.0 * ((np.log(qs_h) - np.log(qs_l)) - (np.log(qo_h) - np.log(qo_l))) / (np.log(qo_h) - np.log(qo_l))
    assert_close(m.pbiasfdc(sim, obs), expected)


def test_low_flow_percent_bias_rejects_zero_in_selected_observations():
    obs = da([0.0, 1.0, 2.0, 3.0])
    sim = da([1.0, 2.0, 2.0, 3.0])

    assert np.isnan(float(metrics().PBIAS_LF(sim, obs, quantile=0.5)))


def test_appendix_agreement_and_efficiency_metrics():
    m = metrics()
    obs = da([1.0, 2.0, 4.0, 8.0])
    sim = da([1.0, 3.0, 5.0, 7.0])
    obs_mean = obs.values.mean()

    mia_den = np.sum(abs(sim.values - obs_mean) + abs(obs.values - obs_mean))
    assert_close(m.MIA(sim, obs), 1 - np.sum(abs(sim.values - obs.values)) / mia_den)

    ria_num = np.sum(abs(sim.values - obs.values) / obs.values)
    ria_den = np.sum((abs(sim.values - obs_mean) + abs(obs.values - obs_mean)) / obs_mean)
    assert_close(m.RIA(sim, obs), 1 - ria_num / ria_den)

    assert_close(m.valindex(sim, obs, epsilon=1.0), 1.0)
    assert_close(m.VE(sim, obs), 1 - np.sum(abs(sim.values - obs.values)) / np.sum(obs.values))

    log_obs = np.log(obs.values)
    log_sim = np.log(sim.values)
    assert_close(m.LNSE(sim, obs), 1 - np.sum((log_sim - log_obs) ** 2) / np.sum((log_obs - log_obs.mean()) ** 2))
    assert_close(m.mNSE(sim, obs), 1 - np.sum(abs(sim.values - obs.values)) / np.sum(abs(obs.values - obs_mean)))
    assert_close(
        m.rNSE(sim, obs),
        1 - np.sum(((sim.values - obs.values) / obs.values) ** 2) / np.sum(((obs.values - obs_mean) / obs_mean) ** 2),
    )


def test_appendix_kge_variants_and_spearman_components():
    m = metrics()
    obs = da([1.0, 2.0, 3.0, 4.0])
    sim = da([1.0, 2.0, 2.0, 8.0])

    r = np.corrcoef(sim.values, obs.values)[0, 1]
    alpha = np.std(sim.values) / np.std(obs.values)
    beta = np.mean(sim.values) / np.mean(obs.values)
    gamma = (np.std(sim.values) / np.mean(sim.values)) / (np.std(obs.values) / np.mean(obs.values))

    assert_close(m.rSpearman(sim, obs), pd.Series(sim.values).rank().corr(pd.Series(obs.values).rank()))
    assert_close(m.mKGE(sim, obs), 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))
    assert_close(m.KGEkm(sim, obs), 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))

    split = m.sKGE(sim, obs)
    assert list(split.component.values) == ["r", "beta", "gamma"]
    np.testing.assert_allclose(split.values, [r, beta, gamma])

    low = obs <= obs.quantile(0.5)
    assert_close(m.KGElf(sim, obs, quantile=0.5), m.KGE(sim.where(low), obs.where(low)))

    n = sim.size
    alpha_np = 1 - 0.5 * np.sum(
        np.abs(
            np.sort(sim.values)[::-1] / (n * np.mean(sim.values))
            - np.sort(obs.values)[::-1] / (n * np.mean(obs.values))
        )
    )
    expected_np = 1 - np.sqrt(
        (float(m.rSpearman(sim, obs)) - 1) ** 2
        + (alpha_np - 1) ** 2
        + (np.mean(sim.values) / np.mean(obs.values) - 1) ** 2
    )
    assert_close(m.KGEnp(sim, obs), expected_np)


def test_appendix_metrics_return_nan_outside_domain_and_do_not_mutate_inputs():
    m = metrics()
    obs = da([0.0, 0.0, 0.0])
    sim = da([1.0, 2.0, 3.0])
    obs_before = obs.copy(deep=True)
    sim_before = sim.copy(deep=True)

    for name in ["NRMSE", "RSR", "rSD", "PBIAS_LF", "RIA", "VE", "LNSE", "mNSE", "rNSE"]:
        assert np.isnan(float(getattr(m, name)(sim, obs)))

    xr.testing.assert_identical(obs, obs_before)
    xr.testing.assert_identical(sim, sim_before)


def test_appendix_metrics_vectorize_over_non_time_dimensions():
    m = metrics()
    obs = xr.DataArray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dims=["time", "site"])
    sim = xr.DataArray([[1.0, 3.0], [2.0, 5.0], [4.0, 7.0]], dims=["time", "site"])

    assert m.MSE(sim, obs).dims == ("site",)
    np.testing.assert_allclose(m.MSE(sim, obs), [1.0 / 3.0, 1.0])
    assert m.rSpearman(sim, obs).dims == ("site",)
    np.testing.assert_allclose(m.rSpearman(sim, obs), [1.0, 1.0])


def test_appendix_weighted_metrics_reject_negative_weights():
    m = metrics()
    obs = da([1.0, 2.0, 3.0, 4.0])
    sim = da([1.0, 3.0, 2.0, 6.0])
    weights = da([1.0, 1.0, 2.0, 2.0])

    weighted_mean = np.sum(weights.values * obs.values) / np.sum(weights.values)
    expected = 1 - np.sum(weights.values * (sim.values - obs.values) ** 2) / np.sum(
        weights.values * (obs.values - weighted_mean) ** 2
    )
    assert_close(m.wNSE(sim, obs, weights=weights), expected)

    bad_weights = da([1.0, -10.0, np.nan, 2.0])
    assert np.isnan(float(m.wNSE(sim, obs, weights=bad_weights)))


def test_appendix_weighted_seasonal_nse_matches_monthly_formula():
    m = metrics()
    times = pd.to_datetime(["2001-01-01", "2001-01-02", "2001-07-01", "2001-07-02"])
    obs = xr.DataArray([1.0, 3.0, 2.0, 6.0], coords={"time": times}, dims=["time"])
    sim = xr.DataArray([1.0, 5.0, 4.0, 6.0], coords={"time": times}, dims=["time"])
    season_weights = {"DJF": 2.0, "JJA": 1.0}
    weights = np.array([2.0, 2.0, 1.0, 1.0])
    month_means = np.array([2.0, 2.0, 4.0, 4.0])
    expected = 1 - np.sum(weights * (sim.values - obs.values) ** 2) / np.sum(weights * (obs.values - month_means) ** 2)

    assert_close(m.wsNSE(sim, obs, season_weights=season_weights), expected)


def test_domain_guards_ignore_invalid_pairs_not_entire_series():
    m = metrics()
    obs = da([1.0, 2.0, np.nan, 4.0])
    sim = da([1.0, 3.0, 9.0, 5.0])
    obs_drop = da([1.0, 2.0, 4.0])
    sim_drop = da([1.0, 3.0, 5.0])

    for name in ["RIA", "VE", "LNSE", "rNSE"]:
        assert_close(getattr(m, name)(sim, obs), getattr(m, name)(sim_drop, obs_drop))


def test_wsNSE_requires_explicit_seasons_for_numeric_time_coords():
    m = metrics()
    obs = xr.DataArray([1.0, 2.0, 3.0], coords={"time": [0, 1, 2]}, dims=["time"])
    sim = xr.DataArray([1.0, 3.0, 2.0], coords={"time": [0, 1, 2]}, dims=["time"])

    with pytest.raises(ValueError, match="explicit seasons"):
        m.wsNSE(sim, obs, season_weights={"all": 1.0})
