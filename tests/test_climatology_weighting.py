import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.climatology import ClimatologyProcessor


def test_reference_annual_climatology_weights_month_lengths():
    processor = ClimatologyProcessor()
    times = pd.date_range("2001-01-01", periods=12, freq="MS") + pd.Timedelta(days=14)
    values = np.arange(1, 13, dtype=float)
    ds = xr.Dataset({"v": ("time", values)}, coords={"time": times})

    result = processor.prepare_reference_climatology(ds, processor.ANNUAL_CLIMATOLOGY, 2001)

    expected = np.average(values, weights=times.days_in_month)
    assert np.isclose(float(result["v"].isel(time=0)), expected)


def test_simulation_monthly_climatology_weights_same_month_across_years():
    processor = ClimatologyProcessor()
    times = []
    values = []
    for year in (2000, 2001):
        for month in range(1, 13):
            times.append(pd.Timestamp(year=year, month=month, day=15))
            if month == 2:
                values.append(29.0 if year == 2000 else 28.0)
            else:
                values.append(float(month))
    times = pd.to_datetime(times)
    ds = xr.Dataset({"v": ("time", np.array(values))}, coords={"time": times})

    result = processor.prepare_simulation_climatology(ds, processor.MONTHLY_CLIMATOLOGY, 2001)

    feb = result.sel(time="2001-02-15")["v"]
    expected_feb = (29.0 * 29 + 28.0 * 28) / (29 + 28)
    assert np.isclose(float(feb), expected_feb)
