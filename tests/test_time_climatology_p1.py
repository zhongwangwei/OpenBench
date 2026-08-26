import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.data.climatology import ClimatologyProcessor
from openbench.data.processing import DatasetProcessing
from openbench.data.time_utils import normalize_cftime_axis


def test_360_day_cftime_conversion_fails_instead_of_clamping_duplicates():
    cftime = pytest.importorskip("cftime")
    times = [cftime.Datetime360Day(2001, 2, day) for day in range(1, 31)]
    ds = xr.Dataset({"v": ("time", np.arange(30.0))}, coords={"time": times})
    ds.time.encoding["calendar"] = "360_day"

    with pytest.raises(ValueError, match="Cannot losslessly convert CF calendar"):
        normalize_cftime_axis(ds, source_path="360_day.nc")


def test_strict_time_integrity_rejects_missing_month_before_reindexing():
    processor = object.__new__(DatasetProcessing)
    processor.time_alignment = "strict"
    time = [pd.Timestamp(f"2001-{month:02d}-15") for month in range(1, 13) if month != 6]
    data = xr.DataArray(np.ones(len(time)), dims="time", coords={"time": time}, name="v")

    with pytest.raises(ValueError, match="strict time alignment requires complete month coverage"):
        processor.check_dataset_time_integrity(data, 2001, 2001, "Month", "stat")


def test_monthly_reference_climatology_rejects_twelve_daily_samples():
    processor = ClimatologyProcessor()
    ds = xr.Dataset(
        {"v": ("time", np.arange(12.0))},
        coords={"time": pd.date_range("2001-01-01", periods=12, freq="D")},
    )

    with pytest.raises(ValueError, match="Missing months"):
        processor.prepare_reference_climatology(ds, processor.MONTHLY_CLIMATOLOGY, 2001)


def test_monthly_reference_climatology_accepts_one_sample_per_month():
    processor = ClimatologyProcessor()
    ds = xr.Dataset(
        {"v": ("time", np.arange(12.0))},
        coords={"time": pd.date_range("1999-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)},
    )

    out = processor.prepare_reference_climatology(ds, processor.MONTHLY_CLIMATOLOGY, 2001)

    assert out.sizes["time"] == 12
    assert list(pd.to_datetime(out.time.values).month) == list(range(1, 13))
    np.testing.assert_allclose(out["v"].values, np.arange(12.0))


def test_non_climatology_missing_time_coordinate_is_rejected():
    processor = object.__new__(DatasetProcessing)
    data = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [10.0, 20.0]},
        name="Runoff",
    )

    with pytest.raises(ValueError, match="must include a 'time' coordinate"):
        processor.check_time(data, 2000, 2000, "D")
