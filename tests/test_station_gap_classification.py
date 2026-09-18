"""Known station data gaps are partial skips; program/schema errors still fail."""

from __future__ import annotations

from types import MethodType

import pandas as pd
import pytest
import xarray as xr

from openbench.data.compute import execute_compute
from openbench.data.processing import StationDatasetProcessing


def _processor(tmp_path):
    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.casedir = str(tmp_path)
    proc.item = "Runoff"
    proc.ref_source = "Ref"
    proc.sim_source = "Sim"
    proc.ref_varname = proc.sim_varname = ["flow"]
    proc.ref_varunit = proc.sim_varunit = "m3 s-1"
    proc.compare_tim_res = "D"
    proc.compare_grid_res = 1.0
    proc.num_cores = 1
    proc._is_climatology_mode = lambda: False
    proc._resample_to_compare_resolution = lambda ds, *args: ds
    proc.check_coordinate = lambda ds: ds
    proc.check_dataset_time_integrity = lambda ds, *args: ds
    proc.process_units = lambda ds, unit: (ds, unit)
    proc.select_timerange = lambda ds, *args: ds
    return proc


def test_missing_compute_dependency_is_station_skip_but_broken_compute_fails(tmp_path):
    proc = _processor(tmp_path)
    good = tmp_path / "good.nc"
    missing_dep = tmp_path / "missing_dep.nc"
    xr.Dataset(
        {"a": ("time", [1.0]), "b": ("time", [2.0])},
        coords={"time": pd.date_range("2000-01-01", periods=1)},
    ).to_netcdf(good)
    xr.Dataset(
        {"a": ("time", [1.0])},
        coords={"time": pd.date_range("2000-01-01", periods=1)},
    ).to_netcdf(missing_dep)
    proc.station_list = pd.DataFrame(
        [
            {"ID": "A", "use_syear": 2000, "use_eyear": 2000, "ref_dir": str(good)},
            {"ID": "B", "use_syear": 2000, "use_eyear": 2000, "ref_dir": str(missing_dep)},
        ]
    )

    def compute_flow(self, source_name, ds, datasource):
        result = execute_compute(ds, "ds['a'] + ds['b']", "Runoff")
        result.name = "flow"
        return result

    proc._try_compute_from_profile = MethodType(compute_flow, proc)
    proc.process_station_data({"datasource": "ref"})

    good_output = tmp_path / "data/stn_Ref_Sim/Runoff_ref_A_2000_2000.nc"
    missing_output = tmp_path / "data/stn_Ref_Sim/Runoff_ref_B_2000_2000.nc"
    assert good_output.exists()
    assert not missing_output.exists()
    assert "not found in dataset when computing" in missing_output.with_suffix(".skip.txt").read_text()

    proc = _processor(tmp_path / "broken")
    raw = tmp_path / "broken_raw.nc"
    xr.Dataset({"a": ("time", [1.0])}, coords={"time": pd.date_range("2000-01-01", periods=1)}).to_netcdf(raw)
    proc.station_list = pd.DataFrame([{"ID": "A", "use_syear": 2000, "use_eyear": 2000, "ref_dir": str(raw)}])
    proc._try_compute_from_profile = lambda *args: (_ for _ in ()).throw(AttributeError("implementation bug"))
    with pytest.raises(AttributeError, match="implementation bug"):
        proc.process_station_data({"datasource": "ref"})


def test_out_of_domain_grid_station_is_skip_but_missing_coordinates_fail(tmp_path):
    proc = _processor(tmp_path)
    grid = xr.Dataset(
        {"flow": (("time", "lat", "lon"), [[[1.0]]])},
        coords={"time": pd.date_range("2000-01-01", periods=1), "lat": [0.0], "lon": [0.0]},
    )
    stations = pd.DataFrame(
        [
            {"ID": "A", "use_syear": 2000, "use_eyear": 2000, "ref_lat": 0.0, "ref_lon": 0.0},
            {"ID": "B", "use_syear": 2000, "use_eyear": 2000, "ref_lat": 5.0, "ref_lon": 0.0},
        ]
    )

    proc._extract_stn_parallel("sim", grid, stations, 0)
    proc._extract_stn_parallel("sim", grid, stations, 1)

    good_output = tmp_path / "data/stn_Ref_Sim/Runoff_sim_A_2000_2000.nc"
    missing_output = tmp_path / "data/stn_Ref_Sim/Runoff_sim_B_2000_2000.nc"
    assert good_output.exists()
    assert not missing_output.exists()
    assert "outside tolerance" in missing_output.with_suffix(".skip.txt").read_text()

    bad_station = pd.DataFrame([{"ID": "C", "use_syear": 2000, "use_eyear": 2000}])
    with pytest.raises(KeyError):
        proc._extract_stn_parallel("sim", grid, bad_station, 0)


def test_compute_non_variable_keyerror_remains_fatal(tmp_path):
    from openbench.data.compute import ComputeError, MissingComputeVariable

    ds = xr.Dataset({"a": ("time", [1.0])}, coords={"time": pd.date_range("2000-01-01", periods=1)})
    # A bad timestamp selector is a compute/config error, not an absent dependency.
    with pytest.raises(ComputeError) as caught:
        execute_compute(ds, "ds['a'].sel(time='2001-01-01')", "Runoff")
    assert not isinstance(caught.value, MissingComputeVariable)


def test_compute_keeps_xarray_virtual_time_variables():
    ds = xr.Dataset({"a": ("time", [1.0])}, coords={"time": pd.date_range("2000-01-01", periods=1)})
    assert execute_compute(ds, "ds['time.month']", "month").item() == 1
