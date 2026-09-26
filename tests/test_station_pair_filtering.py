import os

import numpy as np
import pandas as pd
import xarray as xr

from openbench.config.runtime_info import GeneralInfoReader


def _write_station_file(path, times, values, var_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({var_name: ("time", values)}, coords={"time": pd.to_datetime(times)}).to_netcdf(path)


def test_station_list_filters_rows_without_finite_sim_ref_pairs(tmp_path):
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data" / "stn_Ref_Sim"
    rows = [
        {"ID": "good", "use_syear": 2001, "use_eyear": 2001},
        {"ID": "bad_ref", "use_syear": 2001, "use_eyear": 2001},
        {"ID": "bad_sim", "use_syear": 2001, "use_eyear": 2001},
    ]
    month_end = ["2001-01-31", "2001-02-28"]
    month_mid = ["2001-01-15", "2001-02-15"]

    _write_station_file(data_dir / "Streamflow_sim_good_2001_2001.nc", month_end, [1.0, 2.0], "f_discharge")
    _write_station_file(data_dir / "Streamflow_ref_good_2001_2001.nc", month_mid, [1.5, 2.5], "discharge")
    _write_station_file(data_dir / "Streamflow_sim_bad_ref_2001_2001.nc", month_end, [1.0, 2.0], "f_discharge")
    _write_station_file(data_dir / "Streamflow_ref_bad_ref_2001_2001.nc", month_mid, [np.nan, np.nan], "discharge")
    _write_station_file(data_dir / "Streamflow_sim_bad_sim_2001_2001.nc", month_end, [np.nan, np.nan], "f_discharge")
    _write_station_file(data_dir / "Streamflow_ref_bad_sim_2001_2001.nc", month_mid, [1.0, 2.0], "discharge")

    info = object.__new__(GeneralInfoReader)
    info.casedir = str(case_dir)
    info.ref_source = "Ref"
    info.sim_source = "Sim"
    info.item = "Streamflow"
    info.ref_data_type = "stn"
    info.sim_data_type = "grid"
    info.compare_tim_res = "Month"
    info.stn_list = pd.DataFrame(rows)

    info._filter_existing_station_pairs_with_valid_data()

    assert info.stn_list["ID"].tolist() == ["good"]


def test_station_pair_filtering_validates_files_older_than_regenerated_list(tmp_path):
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data" / "stn_Ref_Sim"
    station_list_path = case_dir / "stn_Ref_Sim_list.csv"
    rows = [
        {"ID": "good", "use_syear": 2001, "use_eyear": 2001},
        {"ID": "bad", "use_syear": 2001, "use_eyear": 2001},
    ]

    month_end = ["2001-01-31", "2001-02-28"]
    month_mid = ["2001-01-15", "2001-02-15"]
    _write_station_file(data_dir / "Streamflow_sim_good_2001_2001.nc", month_end, [1.0, 2.0], "f_discharge")
    _write_station_file(data_dir / "Streamflow_ref_good_2001_2001.nc", month_mid, [1.5, 2.5], "discharge")
    _write_station_file(data_dir / "Streamflow_sim_bad_2001_2001.nc", month_end, [np.nan, np.nan], "f_discharge")
    _write_station_file(
        data_dir / "Streamflow_ref_bad_2001_2001.nc",
        ["2001-01-01", "2001-01-02", "2001-02-01"],
        [2.0, 2.5, 3.0],
        "discharge",
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    station_list_path.write_text("ID,use_syear,use_eyear\ngood,2001,2001\nbad,2001,2001\n")

    newer = station_list_path.stat().st_mtime + 10
    older = station_list_path.stat().st_mtime - 10
    os.utime(station_list_path, (newer, newer))
    for path in data_dir.glob("*.nc"):
        os.utime(path, (older, older))

    info = object.__new__(GeneralInfoReader)
    info.casedir = str(case_dir)
    info.ref_source = "Ref"
    info.sim_source = "Sim"
    info.item = "Streamflow"
    info.ref_data_type = "stn"
    info.sim_data_type = "grid"
    info.compare_tim_res = "Month"
    info.ref_fulllist = str(station_list_path)
    info.stn_list = pd.DataFrame(rows)

    info._filter_existing_station_pairs_with_valid_data()

    assert info.stn_list["ID"].tolist() == ["good"]
