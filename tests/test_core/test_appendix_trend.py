import numpy as np
import pandas as pd
import xarray as xr

from openbench.core.statistics import statistics_calculate


def _stats(max_sen_pairs=2_000_000):
    return statistics_calculate(
        {
            "stats_nml": {
                "Mann_Kendall_Trend_Test": {
                    "significance_level": 0.05,
                    "max_sen_pairs": max_sen_pairs,
                }
            }
        }
    )


def _da(values):
    times = pd.date_range("2000-01-01", periods=len(values), freq="D")
    return xr.DataArray(values, coords={"time": times}, dims=["time"])


def test_mann_kendall_exports_existing_and_appendix_outputs_for_increase():
    result = _stats().stat_mann_kendall_trend_test(_da([1, 2, 3, 4, 5]))

    assert {"trend", "significance", "p_value", "tau"}.issubset(result.data_vars)
    assert {"s_statistic", "z_score", "sen_slope"}.issubset(result.data_vars)
    assert float(result.trend) == 1.0
    assert float(result.significance) == 1.0
    np.testing.assert_allclose(float(result.tau), 1.0)
    assert float(result.s_statistic) == 10.0
    np.testing.assert_allclose(float(result.z_score), 9 / np.sqrt(50 / 3))
    assert float(result.sen_slope) == 1.0


def test_mann_kendall_decrease_has_negative_s_and_sen_slope():
    result = _stats().stat_mann_kendall_trend_test(_da([5, 4, 3, 2, 1]))

    assert float(result.trend) == -1.0
    assert float(result.significance) == 1.0
    np.testing.assert_allclose(float(result.tau), -1.0)
    assert float(result.s_statistic) == -10.0
    np.testing.assert_allclose(float(result.z_score), -9 / np.sqrt(50 / 3))
    assert float(result.sen_slope) == -1.0


def test_mann_kendall_ties_use_tie_corrected_z_and_average_slope():
    result = _stats().stat_mann_kendall_trend_test(_da([1, 1, 2, 3, 4]))

    assert np.isfinite(float(result.p_value))
    assert np.isfinite(float(result.tau))
    assert float(result.s_statistic) == 9.0
    np.testing.assert_allclose(float(result.z_score), 8 / np.sqrt(47 / 3))
    assert float(result.sen_slope) == 1.0


def test_mann_kendall_constant_series_keeps_undefined_p_but_zero_slope():
    result = _stats().stat_mann_kendall_trend_test(_da([2, 2, 2, 2, 2]))

    assert float(result.trend) == 0.0
    assert np.isnan(float(result.significance))
    assert np.isnan(float(result.p_value))
    assert np.isnan(float(result.tau))
    assert float(result.s_statistic) == 0.0
    assert np.isnan(float(result.z_score))
    assert float(result.sen_slope) == 0.0


def test_mann_kendall_ignores_nans_before_statistics():
    result = _stats().stat_mann_kendall_trend_test(_da([1, np.nan, 3, 4, 5]))

    assert float(result.trend) == 1.0
    assert float(result.s_statistic) == 6.0
    assert float(result.sen_slope) == 1.0


def test_mann_kendall_keeps_linear_memory_when_exact_sen_limit_is_exceeded():
    result = _stats(max_sen_pairs=3).stat_mann_kendall_trend_test(_da([1, 2, 3, 4, 5]))

    assert float(result.s_statistic) == 10.0
    assert np.isnan(float(result.sen_slope))
    assert result.attrs["sen_slope_max_exact_pairs"] == 3
