"""Tests for non-standard time decoding helpers."""

import numpy as np
import xarray as xr

from openbench.data.time_utils import decode_nonstandard_time


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
