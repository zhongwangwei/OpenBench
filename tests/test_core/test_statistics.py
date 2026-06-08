"""Tests for core statistics helper edge cases."""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from openbench.core.statistics import statistics_calculate
from openbench.core.statistics.Mod_Statistics import StatisticsProcessing
from openbench.core.statistics.stat_anova import stat_anova


def make_da(values):
    times = pd.date_range("2000-01-01", periods=len(values), freq="D")
    return xr.DataArray(values, coords={"time": times}, dims=["time"], name="var")


def test_stat_resample_returns_reduced_dataarray():
    stats = statistics_calculate({})
    data = make_da([1.0, 3.0, 5.0, 7.0])

    result = stats.stat_resample(data, "2D")

    assert isinstance(result, xr.DataArray)
    assert result.sizes["time"] == 2
    np.testing.assert_allclose(result.values, [2.0, 6.0])


def test_stat_rolling_returns_window_mean_dataarray():
    stats = statistics_calculate({})
    data = make_da([1.0, 3.0, 5.0])

    result = stats.stat_rolling(data, 2)

    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, [np.nan, 2.0, 4.0])


def test_functional_response_zero_mean_dependent_series_returns_nan():
    stats = statistics_calculate({"stats_nml": {"Functional_Response": {"nbins": 2}}})
    v = make_da([-1.0, 1.0, -1.0, 1.0])
    u = make_da([0.0, 1.0, 2.0, 3.0])

    result = stats.stat_functional_response(v, u)

    assert np.isnan(float(result["functional_response_score"]))


def test_correlation_and_covariance_reject_multi_variable_datasets():
    """Dataset inputs are only unambiguous when they contain exactly one variable."""
    stats = statistics_calculate({})
    data = xr.Dataset({"a": make_da([1.0, 2.0, 3.0]), "b": make_da([3.0, 2.0, 1.0])})

    for method_name in ["stat_correlation", "stat_covariance"]:
        method = getattr(stats, method_name)
        try:
            method(data, data)
        except ValueError as exc:
            assert "exactly one data variable" in str(exc)
        else:
            raise AssertionError(f"{method_name} accepted an ambiguous multi-variable Dataset")


def test_covariance_accepts_single_variable_datasets_like_correlation():
    """Correlation/covariance should share the same single-variable Dataset contract."""
    stats = statistics_calculate({})
    data1 = make_da([1.0, 2.0, 3.0]).to_dataset()
    data2 = make_da([2.0, 4.0, 6.0]).to_dataset()

    result = stats.stat_covariance(data1, data2)

    assert result.name == "Covariance"
    assert abs(float(result) - 2.0) < 1e-10


def test_false_discovery_rate_handles_lazy_all_false_significance():
    """FDR should not rely on unavailable dask sort or lazy-array truthiness."""
    import dask.array as da

    stats = statistics_calculate({"stats_nml": {"False_Discovery_Rate": {"alpha": 0.05}}})
    times = pd.date_range("2000-01-01", periods=4, freq="D")
    values = da.from_array(np.ones((4, 2)), chunks=(2, 2))
    data1 = xr.DataArray(values, coords={"time": times, "x": [0, 1]}, dims=["time", "x"])
    data2 = xr.DataArray(values, coords={"time": times, "x": [0, 1]}, dims=["time", "x"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = stats.stat_False_Discovery_Rate(data1, data2)
        result["significant"].any().compute()

    assert result.attrs["FDR_threshold"] == 0.0
    assert not bool(result["significant"].any().compute())
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_false_discovery_rate_welch_p_values_match_scipy_sample_variance():
    """FDR's vectorized Welch test must use ddof=1 sample variances."""
    from scipy import stats as scipy_stats

    stats = statistics_calculate({"stats_nml": {"False_Discovery_Rate": {"alpha": 1.0}}})
    times = pd.date_range("2000-01-01", periods=3, freq="D")
    data1 = xr.DataArray([1.0, 2.0, 3.0], coords={"time": times}, dims=["time"])
    data2 = xr.DataArray([2.0, 3.0, 5.0], coords={"time": times}, dims=["time"])

    result = stats.stat_False_Discovery_Rate(data1, data2)
    expected = scipy_stats.ttest_ind(data1.values, data2.values, equal_var=False).pvalue

    assert abs(float(result["p_values"].isel(combination=0)) - expected) < 1e-12


def test_false_discovery_rate_dispatcher_runs_registered_method(tmp_path, monkeypatch):
    main = {
        "general": {
            "compare_grid_res": 0.5,
            "compare_tim_res": "Day",
            "min_lon": 0,
            "max_lon": 1,
            "min_lat": 0,
            "max_lat": 1,
        }
    }
    stats_nml = {
        "general": {"False_Discovery_Rate_data_source": "Runoff"},
        "False_Discovery_Rate": {"Runoff1_dir": "", "Runoff2_dir": ""},
    }
    processor = StatisticsProcessing(main, stats_nml, str(tmp_path), num_cores=1)
    calls = []

    def fake_run_analysis(source, sources, statistic_method):
        calls.append((source, sources, statistic_method))
        return str(tmp_path / "fdr.nc")

    monkeypatch.setattr(processor, "run_analysis", fake_run_analysis)

    processor.scenarios_False_Discovery_Rate_analysis("False_Discovery_Rate", stats_nml["False_Discovery_Rate"], {})

    assert calls == [("Runoff", ["Runoff1", "Runoff2"], "False_Discovery_Rate")]
    assert hasattr(processor, "stat_false_discovery_rate")


def test_autocorrelation_uses_xarray_correlation_not_missing_pandas_api():
    stats = statistics_calculate({})
    data = make_da([1.0, 2.0, 3.0, 4.0])

    result = stats.stat_autocorrelation(data)

    assert result.name == "Autocorrelation"
    assert abs(float(result) - 1.0) < 1e-12


def test_standard_deviation_uses_sample_ddof_like_smpi():
    stats = statistics_calculate({})
    data = make_da([1.0, 2.0])

    assert abs(float(stats.stat_standard_deviation(data)) - np.sqrt(0.5)) < 1e-12


class _AnovaSelf:
    stats_nml = {"ANOVA": {"n_jobs": 1, "analysis_type": "oneway"}}


def _anova_cube(values):
    arr = np.asarray(values, dtype=float).reshape(len(values), 1, 1)
    return xr.DataArray(
        arr,
        coords={"time": np.arange(len(values)), "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )


def test_oneway_anova_nan_cell_returns_nan_dataset_instead_of_index_error():
    y = _anova_cube([1, 2, 3, 4, 5, 6, 7, 8])
    x = _anova_cube([1, 2, np.nan, 4, 5, 6, 7, 8])

    result = stat_anova(_AnovaSelf(), y, x)

    assert np.isnan(result["F_statistic"].values[0, 0])
    assert np.isnan(result["raw_p_value"].values[0, 0])
    assert np.isnan(result["p_value"].values[0, 0])


def test_oneway_anova_all_negative_predictor_is_not_treated_as_empty():
    y = _anova_cube([1, 2, 3, 4, 5, 6, 7, 8])
    x = _anova_cube([-1, -2, -3, -4, -5, -6, -7, -8])

    result = stat_anova(_AnovaSelf(), y, x)

    assert np.isfinite(result["F_statistic"].values[0, 0])
    assert np.isfinite(result["raw_p_value"].values[0, 0])
    assert np.isfinite(result["p_value"].values[0, 0])


def test_anova_respects_requested_parallelism_without_eight_core_cap(monkeypatch):
    import importlib

    stat_anova_module = importlib.import_module("openbench.core.statistics.stat_anova")

    captured_n_jobs = []

    class FakeParallel:
        def __init__(self, n_jobs):
            captured_n_jobs.append(n_jobs)

        def __call__(self, tasks):
            results = []
            for func, args, kwargs in tasks:
                results.append(func(*args, **kwargs))
            return results

    class AnovaSelf:
        stats_nml = {"ANOVA": {"n_jobs": 16, "analysis_type": "oneway"}}

    monkeypatch.setattr(stat_anova_module.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(stat_anova_module, "Parallel", FakeParallel)

    y = _anova_cube([1, 2, 3, 4, 5, 6, 7, 8])
    x = _anova_cube([-1, -2, -3, -4, -5, -6, -7, -8])

    stat_anova(AnovaSelf(), y, x)

    assert captured_n_jobs == [16]


def test_statistics_processing_freq_map_uses_non_deprecated_pandas_aliases():
    assert StatisticsProcessing.freq_map["hour"] == "h"
    assert StatisticsProcessing.freq_map["h"] == "h"
    assert StatisticsProcessing.freq_map["year"] == "YE"
    assert StatisticsProcessing.freq_map["y"] == "YE"
