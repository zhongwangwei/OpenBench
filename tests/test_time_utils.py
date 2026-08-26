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
