"""Tests for non-standard time decoding helpers."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.time_utils import decode_nonstandard_time, normalize_cftime_axis


def test_decode_nonstandard_te_month_axis_as_monthly_year():
    ds = xr.Dataset(
        {"LTNT": (["time"], np.zeros(12))},
        coords={"time": np.arange(0, 36, 3)},
    )
    ds["time"].attrs["units"] = "calendar months since 1996-01-01 00:00:00 ; "

    decoded = decode_nonstandard_time(ds, source_path="YEE2_JRA-55_LTNT_M1996_GLB050.nc")

    assert decoded.time.values[0] == np.datetime64("1996-01-01T00:00:00")
    assert decoded.time.values[-1] == np.datetime64("1996-12-01T00:00:00")
    assert decoded.time.size == 12


def test_decode_nonstandard_calendar_month_offsets_without_year_file_context():
    ds = xr.Dataset(
        {"value": (["time"], np.zeros(12))},
        coords={"time": np.arange(0, 36, 3)},
    )
    ds["time"].attrs["units"] = "calendar months since 1996-01-01 00:00:00 ; "

    decoded = decode_nonstandard_time(ds)

    assert decoded.time.values[-1] == np.datetime64("1998-10-01T00:00:00")


def test_legacy_timelib_class_is_removed():
    import openbench.data.time_utils as time_utils

    assert not hasattr(time_utils, "timelib")


def test_normalize_cftime_axis_rejects_invalid_360_day_dates():
    cftime = pytest.importorskip("cftime")
    ds = xr.Dataset(
        {"value": ("time", [1.0, 2.0])},
        coords={
            "time": [
                cftime.Datetime360Day(2001, 2, 30),
                cftime.Datetime360Day(2001, 3, 30),
            ]
        },
    )
    ds["time"].attrs["calendar"] = "360_day"

    with pytest.raises(ValueError, match="Cannot losslessly convert CF calendar"):
        normalize_cftime_axis(ds, source_path="test_360_day.nc")


def test_decode_nonstandard_month_offsets_reject_fractional_values():
    ds = xr.Dataset({"value": (["time"], np.zeros(2))}, coords={"time": [0.0, 1.5]})
    ds["time"].attrs["units"] = "calendar months since 2000-01-01"

    with pytest.raises(ValueError, match="Non-integer month offsets"):
        decode_nonstandard_time(ds)


def test_decode_nonstandard_year_offsets_reject_fractional_values():
    ds = xr.Dataset({"value": (["time"], np.zeros(2))}, coords={"time": [0.0, 0.5]})
    ds["time"].attrs["units"] = "years since 2000-01-01"

    with pytest.raises(ValueError, match="Non-integer year offsets"):
        decode_nonstandard_time(ds)


@pytest.mark.parametrize("calendar", ["DatetimeNoLeap", "Datetime360Day"])
def test_station_alignment_preserves_exact_cftime_calendar(calendar):
    from openbench.data.time_utils import align_station_times

    date = getattr(pytest.importorskip("cftime"), calendar)
    dates = (
        [date(2000, 2, day) for day in (27, 28, 29)]
        if calendar == "Datetime360Day"
        else [date(2000, 2, 27), date(2000, 2, 28), date(2000, 3, 1)]
    )
    sim = xr.DataArray([3.0, 2.0, 1.0], dims="time", coords={"time": dates[::-1]})
    ref = xr.DataArray([10.0, 20.0], dims="time", coords={"time": dates[:2]})
    aligned_sim, aligned_ref = align_station_times(sim, ref, "A", "day")
    assert aligned_sim.time.values.tolist() == dates[:2]
    assert aligned_sim.values.tolist() == [1.0, 2.0]
    xr.testing.assert_equal(aligned_sim.time, aligned_ref.time)


def test_station_alignment_reports_nonoverlapping_cftime_as_data_gap():
    from openbench.data.station_missing import StationDataUnavailable
    from openbench.data.time_utils import align_station_times

    date = pytest.importorskip("cftime").DatetimeNoLeap
    sim = xr.DataArray([1.0], dims="time", coords={"time": [date(2000, 1, 1)]})
    ref = xr.DataArray([1.0], dims="time", coords={"time": [date(2000, 1, 2)]})
    with pytest.raises(StationDataUnavailable, match="no overlapping time"):
        align_station_times(sim, ref, "A", "day")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("Month", "1ME"),
        ("3month", "3ME"),
        ("Day", "1D"),
        ("6hr", "6h"),
        ("Hour", "1h"),
        ("Year", "1YE"),
        ("week", "1W"),
    ],
)
def test_comparison_time_freq_returns_valid_pandas_frequency(configured, expected):
    import pandas as pd

    from openbench.data.time_utils import comparison_time_freq

    assert comparison_time_freq(configured) == expected
    pd.tseries.frequencies.to_offset(expected)


@pytest.mark.parametrize("configured", ["fortnight", "3-month", ""])
def test_comparison_time_freq_rejects_unsupported_resolution(configured):
    from openbench.data.time_utils import comparison_time_freq

    with pytest.raises(ValueError, match="Unsupported time resolution"):
        comparison_time_freq(configured)


@pytest.mark.parametrize(
    ("compare_tim_res", "sim_times", "ref_times"),
    [
        ("1D", ["2000-01-01T00", "2000-01-02T00"], ["2000-01-01T12", "2000-01-02T12"]),
        ("1h", ["2000-01-01T00:00", "2000-01-01T01:00"], ["2000-01-01T00:30", "2000-01-01T01:30"]),
        ("1ME", ["2000-01-31", "2000-02-29"], ["2000-01-15", "2000-02-15"]),
        ("3ME", ["2000-03-31", "2000-06-30"], ["2000-03-01", "2000-06-01"]),
        ("1YE", ["2000-12-31", "2001-12-31"], ["2000-01-01", "2001-01-01"]),
        ("Day", ["2000-01-01T00", "2000-01-02T00"], ["2000-01-01T12", "2000-01-02T12"]),
    ],
)
def test_station_alignment_normalizes_comparison_frequencies(compare_tim_res, sim_times, ref_times):
    import pandas as pd

    from openbench.data.time_utils import align_station_times

    sim = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.to_datetime(sim_times)})
    ref = xr.DataArray([3.0, 4.0], dims="time", coords={"time": pd.to_datetime(ref_times)})

    aligned_sim, aligned_ref = align_station_times(sim, ref, "A", compare_tim_res)

    assert aligned_sim.sizes["time"] == aligned_ref.sizes["time"] == 2
    np.testing.assert_array_equal(aligned_sim.values, [1.0, 2.0])
    np.testing.assert_array_equal(aligned_ref.values, [3.0, 4.0])


@pytest.mark.parametrize(
    ("compare_tim_res", "sim_times", "ref_times"),
    [
        ("Month", ["2000-01-31", "2000-02-29"], ["2000-01-15", "2000-02-15"]),
        ("Day", ["2000-01-01T00", "2000-01-02T00"], ["2000-01-01T12", "2000-01-02T12"]),
        ("Hour", ["2000-01-01T00:00", "2000-01-01T01:00"], ["2000-01-01T00:30", "2000-01-01T01:30"]),
        ("Year", ["2000-12-31", "2001-12-31"], ["2000-01-01", "2001-01-01"]),
    ],
)
def test_station_alignment_of_differently_stamped_periods_is_quiet(caplog, compare_tim_res, sim_times, ref_times):
    import logging

    import pandas as pd

    from openbench.data.time_utils import align_station_times

    sim = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.to_datetime(sim_times)})
    ref = xr.DataArray([3.0, 4.0], dims="time", coords={"time": pd.to_datetime(ref_times)})

    with caplog.at_level(logging.WARNING):
        aligned_sim, _ = align_station_times(sim, ref, "A", compare_tim_res)

    assert aligned_sim.sizes["time"] == 2
    assert caplog.records == []


def test_station_alignment_skips_a_station_with_several_values_in_one_period():
    import pandas as pd

    from openbench.data.station_missing import StationDataUnavailable
    from openbench.data.time_utils import align_station_times

    sim = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.to_datetime(["2000-01-10", "2000-01-20"])})
    ref = xr.DataArray([3.0], dims="time", coords={"time": pd.to_datetime(["2000-01-15"])})

    with pytest.raises(StationDataUnavailable, match="several simulation values fall into one Month period"):
        align_station_times(sim, ref, "A", "Month")


@pytest.mark.parametrize("module_name", ["openbench.core.comparison", "openbench.visualization.only_drawing"])
@pytest.mark.parametrize(
    ("configured", "expected"), [("Day", "1D"), ("3month", "3ME"), ("climatology-month", "climatology-month")]
)
def test_comparison_processing_stores_pandas_frequency(tmp_path, module_name, configured, expected):
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, "ComparisonProcessing", None) or module.ComparisonProcessing_only_drawing
    general = {
        "basename": "case",
        "basedir": str(tmp_path),
        "compare_grid_res": 0.5,
        "compare_tim_res": configured,
        "weight": "none",
        "num_cores": 1,
    }

    assert cls({"general": general}, [], []).compare_tim_res == expected
