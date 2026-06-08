"""Regression tests for station matching output lifecycle."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr


def test_station_matching_duplicate_station_ids_do_not_overwrite_scratch_files(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    ds = xr.Dataset(
        {
            "station": ("station", np.array([101, 101])),
            "lon": ("station", np.array([10.0, 11.0])),
            "lat": ("station", np.array([20.0, 21.0])),
            "discharge": (("station", "time"), np.array([[1.0, 2.0], [3.0, 4.0]])),
        },
        coords={"time": times},
    )
    ds.to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_source="SimA",
        sim_syear=2000,
        sim_eyear=2000,
        syear=2000,
        eyear=2000,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    run_station_matching(info, str(dataset_path), method="direct", min_uparea=0.0)

    paths = list(info.stn_list["ref_dir"])
    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert all("__idx" in path for path in paths)
    for path in paths:
        with xr.open_dataset(path) as station_ds:
            assert "discharge" in station_ds


def test_station_matching_preserves_existing_station_list_when_csv_write_fails(tmp_path, monkeypatch):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array([101])),
            "lon": ("station", np.array([10.0])),
            "lat": ("station", np.array([20.0])),
            "discharge": (("station", "time"), np.array([[1.0, 2.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    casedir = tmp_path / "case"
    casedir.mkdir()
    existing_list = casedir / "stn_stations_SimA_list.txt"
    existing_list.write_text("ID,ref_lon,ref_lat,use_syear,use_eyear,ref_dir\nold,0,0,2000,2000,old.nc\n")
    info = SimpleNamespace(
        casedir=str(casedir),
        sim_source="SimA",
        sim_syear=2000,
        sim_eyear=2000,
        syear=2000,
        eyear=2000,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    def fail_to_csv(self, path, *args, **kwargs):
        path.write_text("partial")
        raise OSError("simulated station-list failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    try:
        run_station_matching(info, str(dataset_path), method="direct", min_uparea=0.0)
    except Exception as exc:
        assert "simulated station-list failure" in str(exc)
    else:
        raise AssertionError("run_station_matching unexpectedly succeeded")

    assert existing_list.read_text(encoding="utf-8").startswith("ID,ref_lon")
    assert "partial" not in existing_list.read_text(encoding="utf-8")


def test_station_matching_accepts_string_station_ids(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array(["AR_0000001"], dtype=object)),
            "lon": ("station", np.array([10.0])),
            "lat": ("station", np.array([20.0])),
            "discharge": (("station", "time"), np.array([[1.0, 2.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_source="SimA",
        sim_syear=2000,
        sim_eyear=2000,
        syear=2000,
        eyear=2000,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    run_station_matching(info, str(dataset_path), method="direct", min_uparea=0.0)

    assert info.stn_list["ID"].tolist() == ["AR_0000001"]
    assert Path(info.stn_list["ref_dir"].iloc[0]).exists()


def test_station_matching_reports_missing_cama_companion_fields(tmp_path):
    from openbench.data.station_matcher import run_station_matching
    from openbench.util.exceptions import DataProcessingError

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array([101])),
            "lon": ("station", np.array([10.0])),
            "lat": ("station", np.array([20.0])),
            "discharge": (("station", "time"), np.array([[1.0, 2.0]])),
            "cama_lon_03min": ("station", np.array([10.0])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_source="SimA",
        sim_grid_res=0.05,
        sim_syear=2000,
        sim_eyear=2000,
        syear=2000,
        eyear=2000,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    try:
        run_station_matching(info, str(dataset_path), method="cama_allocation", min_uparea=0.0)
    except DataProcessingError as exc:
        assert "cama_lat_03min" in str(exc)
    else:
        raise AssertionError("missing CaMA latitude field was not reported")


def test_station_matching_honors_station_dim_for_transposed_discharge(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=3, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array(["A", "B"], dtype=object)),
            "lon": ("station", np.array([10.0, 11.0])),
            "lat": ("station", np.array([20.0, 21.0])),
            "discharge": (("time", "station"), np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_source="SimA",
        sim_syear=2000,
        sim_eyear=2000,
        syear=2000,
        eyear=2000,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    run_station_matching(info, str(dataset_path), method="direct", station_dim="station", min_uparea=0.0)

    with xr.open_dataset(info.stn_list.loc[info.stn_list["ID"] == "B", "ref_dir"].iloc[0]) as station_ds:
        np.testing.assert_allclose(station_ds["discharge"].values, [10.0, 20.0, 30.0])


def test_cama_station_matching_treats_negative_999_as_missing(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"])
    xr.Dataset(
        {
            "station": ("station", np.array([101])),
            "lon": ("station", np.array([10.0])),
            "lat": ("station", np.array([20.0])),
            "area": ("station", np.array([10_000.0])),
            "cama_lon_03min": ("station", np.array([10.0])),
            "cama_lat_03min": ("station", np.array([20.0])),
            "cama_alloc_err_03min": ("station", np.array([0.0])),
            "discharge": (("station", "time"), np.array([[-999.0, 2.0, -999.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_source="SimA",
        sim_grid_res=0.05,
        sim_syear=1999,
        sim_eyear=2003,
        syear=1999,
        eyear=2003,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    run_station_matching(info, str(dataset_path), method="cama_allocation", min_uparea=0.0, n_jobs=1)

    assert info.stn_list["use_syear"].tolist() == [2001]
    assert info.stn_list["use_eyear"].tolist() == [2001]


def test_station_matching_jobs_default_is_conservative(monkeypatch):
    from openbench.data.station_matcher import _station_matching_jobs

    monkeypatch.delenv("OPENBENCH_STATION_MATCHER_JOBS", raising=False)
    monkeypatch.setattr("openbench.data.station_matcher.os.cpu_count", lambda: 64)

    assert _station_matching_jobs(100) == 4
    assert _station_matching_jobs(2) == 2
    assert _station_matching_jobs(100, requested=8) == 8
