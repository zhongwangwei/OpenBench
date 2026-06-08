import numpy as np
import xarray as xr

from openbench.util.converttype import Convert_Type


def test_convert_nc_preserves_float64_time_coordinate_precision():
    data = xr.DataArray(
        np.array([1.0, 2.0], dtype="float64"),
        dims=("time",),
        coords={"time": np.array([1_700_000_000.123456, 1_700_000_001.123456], dtype="float64")},
        name="var",
    )

    out = Convert_Type.convert_nc(data)

    assert out.dtype == np.float32
    assert out.coords["time"].dtype == np.float64
    np.testing.assert_array_equal(out.coords["time"].values, data.coords["time"].values)
