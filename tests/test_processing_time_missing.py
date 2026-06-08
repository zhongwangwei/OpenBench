import numpy as np
import xarray as xr

from openbench.data.processing import DatasetProcessing


def test_check_time_expands_2d_data_without_time_to_full_time_index():
    processor = object.__new__(DatasetProcessing)
    data = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [10.0, 20.0]},
        name="Runoff",
    )

    out = processor.check_time(data, 2000, 2000, "D")

    assert out.dims == ("time", "lat", "lon")
    assert out.sizes["time"] == 366
    assert np.allclose(out.isel(time=0).values, data.values)
    assert np.allclose(out.isel(time=-1).values, data.values)


def test_check_time_accepts_pandas_ye_alias_and_reindexes_all_missing_years():
    processor = object.__new__(DatasetProcessing)
    data = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("time",),
        coords={"time": np.array(["1990-12-31", "1991-12-31"], dtype="datetime64[ns]")},
        name="annual",
    )

    out = processor.make_time_integrity(data, 2000, 2001, "YE", "sim")

    assert out.sizes["time"] == 2
    assert np.isnan(out.values).all()


def test_preprocess_restores_transient_fallback_state(monkeypatch):
    processor = object.__new__(DatasetProcessing)
    processor.sim_dir = "/tmp"
    processor.sim_data_groupby = "single"
    processor.sim_varname = ["primary"]
    processor.sim_tim_res = "Month"
    processor.sim_varunit = "kg"
    processor.sim_prefix = ""
    processor.sim_suffix = ""
    processor.sim_data_type = "grid"
    processor.sim_syear = 2000
    processor.sim_eyear = 2001
    processor.sim_convert = ""
    processor._fb_convert_sim = "value * 999"

    def fake_process_grid_data(data_params):
        processor.sim_varunit = "temporary"
        processor._fb_convert_sim = "value * 2"

    monkeypatch.setattr(processor, "process_grid_data", fake_process_grid_data)

    processor._preprocess("sim")

    assert processor.sim_varunit == "kg"
    assert processor._fb_convert_sim == "value * 999"
