"""Regression tests for station matching output lifecycle."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
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
        sim_grid_res=0.25,
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

    run_station_matching(info, str(dataset_path), method="direct")

    paths = list(info.stn_list["ref_dir"])
    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert all("__idx" in path for path in paths)
    for path in paths:
        with xr.open_dataset(path) as station_ds:
            assert "discharge" in station_ds


def test_station_matching_uses_source_qualified_ids_for_consolidated_sources(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array([0, 0])),
            "data_source_name": ("station", np.array(["GRDC", "CAMELS_BR"], dtype=object)),
            "lon": ("station", np.array([10.0, 11.0])),
            "lat": ("station", np.array([20.0, 21.0])),
            "discharge": (("station", "time"), np.array([[1.0, 2.0], [3.0, 4.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_grid_res=0.25,
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

    run_station_matching(info, str(dataset_path), method="direct")

    assert info.stn_list["ID"].tolist() == ["GRDC_0", "CAMELS_BR_0"]
    assert info.stn_list["ID"].is_unique


def test_unique_station_ids_only_qualifies_duplicates_with_filename_safe_ids():
    from openbench.data.station_matcher import _unique_station_ids

    ids = _unique_station_ids(
        np.array(["7", "0", "0", "GRDC_0"]),
        np.array(["X", "GRDC", "a/b", "Y"], dtype=object),
    )

    assert ids[0] == "7"
    assert ids[2] == "a%2Fb_0"
    # "GRDC_0" collides with the qualified duplicate, so both fall back to row indices.
    assert ids[1] == "GRDC_0_idx1"
    assert ids[3] == "GRDC_0_idx3"
    assert len(set(ids)) == len(ids)
    assert not any(char in station_id for station_id in ids for char in '<>:"/\\|?*')
    assert _unique_station_ids(np.array(["1", "1"])) == ["1_idx0", "1_idx1"]


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
        sim_grid_res=0.25,
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
        run_station_matching(info, str(dataset_path), method="direct")
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
        sim_grid_res=0.25,
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

    run_station_matching(info, str(dataset_path), method="direct")

    assert info.stn_list["ID"].tolist() == ["AR_0000001"]
    assert Path(info.stn_list["ref_dir"].iloc[0]).exists()


def test_station_matching_counts_single_year_and_wraps_longitude(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array(["A"], dtype=object)),
            "lon": ("station", np.array([190.0])),
            "lat": ("station", np.array([20.0])),
            "discharge": (("station", "time"), np.array([[1.0, 2.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_grid_res=0.25,
        sim_source="SimA",
        sim_syear=2000,
        sim_eyear=2000,
        syear=2000,
        eyear=2000,
        min_year=1,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    run_station_matching(info, str(dataset_path), method="direct")

    assert info.stn_list["ID"].tolist() == ["A"]
    assert info.stn_list["use_syear"].tolist() == [2000]
    assert info.stn_list["use_eyear"].tolist() == [2000]
    assert info.stn_list["ref_lon"].tolist() == [-170.0]


def test_station_matching_reads_only_candidate_stations_and_target_years(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("1999-01-01", periods=5, freq="YS")
    xr.Dataset(
        {
            "station": ("station", np.array(["A", "B"], dtype=object)),
            "lon": ("station", np.array([-60.0, 10.0])),
            "lat": ("station", np.array([0.0, 20.0])),
            "area": ("station", np.array([10_000.0, 10_000.0])),
            "discharge": (
                ("station", "time"),
                np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]),
            ),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_grid_res=0.25,
        sim_source="SimA",
        sim_syear=2001,
        sim_eyear=2002,
        syear=2001,
        eyear=2002,
        min_year=1,
        min_lon=-80,
        max_lon=-35,
        min_lat=-22,
        max_lat=12,
    )

    run_station_matching(info, str(dataset_path), method="direct", n_jobs=1)

    assert info.stn_list["ID"].tolist() == ["A"]
    with xr.open_dataset(info.stn_list["ref_dir"].iloc[0]) as station_ds:
        assert station_ds.sizes["time"] == 2
        np.testing.assert_allclose(station_ds["discharge"].values, [3.0, 4.0])


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
        run_station_matching(info, str(dataset_path), method="cama_allocation")
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
        sim_grid_res=0.25,
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

    run_station_matching(info, str(dataset_path), method="direct", station_dim="station")

    with xr.open_dataset(info.stn_list.loc[info.stn_list["ID"] == "B", "ref_dir"].iloc[0]) as station_ds:
        np.testing.assert_allclose(station_ds["discharge"].values, [10.0, 20.0, 30.0])


def test_station_extraction_manual_fallback_uses_cyclic_longitude_distance():
    from openbench.data._processing_station_extract import StationExtractionMixin

    class Processor(StationExtractionMixin):
        compare_grid_res = 1.0

    dataset = xr.Dataset(
        {"value": (("time", "lat", "lon"), np.array([[[42.0]]]))},
        coords={"time": pd.date_range("2000-01-01", periods=1), "lat": [0.0], "lon": [-170.0]},
    )
    station = pd.Series({"ID": "A", "ref_lat": 0.0, "ref_lon": 190.0})

    extracted = Processor().extract_single_station_data(dataset, station, "sim")

    assert float(extracted["lon"].values[0]) == -170.0
    assert float(extracted["value"].values[0, 0, 0]) == pytest.approx(42.0)


def test_station_extraction_falls_back_to_same_source_coordinates_when_peer_is_missing():
    from openbench.data._processing_station_extract import StationExtractionMixin

    class Processor(StationExtractionMixin):
        compare_grid_res = 1.0

    dataset = xr.Dataset(
        {"value": (("time", "lat", "lon"), np.array([[[42.0]]]))},
        coords={"time": pd.date_range("2000-01-01", periods=1), "lat": [1.0], "lon": [2.0]},
    )
    station = pd.Series({"ID": "A", "sim_lat": 1.0, "sim_lon": 2.0})

    extracted = Processor().extract_single_station_data(dataset, station, "sim")

    assert float(extracted["value"].values[0, 0, 0]) == pytest.approx(42.0)


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

    run_station_matching(info, str(dataset_path), method="cama_allocation", n_jobs=1)

    assert info.stn_list["use_syear"].tolist() == [2001]
    assert info.stn_list["use_eyear"].tolist() == [2001]
    with xr.open_dataset(info.stn_list["ref_dir"].iloc[0]) as station_ds:
        np.testing.assert_allclose(station_ds["discharge"].values, [np.nan, 2.0, np.nan], equal_nan=True)


def test_direct_station_matching_writes_missing_sentinels_as_nan(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = tmp_path / "stations.nc"
    times = pd.date_range("2000-01-01", periods=3, freq="D")
    xr.Dataset(
        {
            "station": ("station", np.array(["A"], dtype=object)),
            "lon": ("station", np.array([10.0])),
            "lat": ("station", np.array([20.0])),
            "discharge": (("station", "time"), np.array([[10.0, -999.0, 10.0]])),
        },
        coords={"time": times},
    ).to_netcdf(dataset_path)

    info = SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_grid_res=0.25,
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

    run_station_matching(info, str(dataset_path), method="direct")

    with xr.open_dataset(info.stn_list["ref_dir"].iloc[0]) as station_ds:
        np.testing.assert_allclose(station_ds["discharge"].values, [10.0, np.nan, 10.0], equal_nan=True)


def test_station_matching_jobs_default_is_conservative(monkeypatch):
    from openbench.data.station_matcher import _station_matching_jobs

    monkeypatch.delenv("OPENBENCH_STATION_MATCHER_JOBS", raising=False)
    monkeypatch.setattr("openbench.data.station_matcher.os.cpu_count", lambda: 64)

    assert _station_matching_jobs(100) == 4
    assert _station_matching_jobs(2) == 2
    assert _station_matching_jobs(100, requested=8) == 8


def test_cama_resolution_rejects_unknown_resolution():
    from openbench.data.station_matcher import get_resolution_suffix

    with pytest.raises(ValueError, match="Unsupported CaMA"):
        get_resolution_suffix(0.07)


@pytest.mark.parametrize(
    ("sim_grid_res", "expected"),
    [(0.25, 3000.0), (0.1, 500.0), (0.0833, 350.0), (0.05, 150.0), (0.0167, 100.0)],
)
def test_resolution_min_uparea_is_fixed_per_cama_resolution(sim_grid_res, expected):
    from openbench.data.station_matcher import resolution_min_uparea

    assert resolution_min_uparea(sim_grid_res) == expected


@pytest.mark.parametrize("sim_grid_res", [0.5, None, ""])
def test_resolution_min_uparea_rejects_unsupported_resolution(sim_grid_res):
    from openbench.data.station_matcher import resolution_min_uparea

    with pytest.raises(ValueError, match="no minimum upstream area"):
        resolution_min_uparea(sim_grid_res)


def _uparea_stations(tmp_path, areas, alloc_errs=None):
    dataset_path = tmp_path / "stations.nc"
    n = len(areas)
    if alloc_errs is None:
        alloc_errs = np.zeros(n)
    xr.Dataset(
        {
            "station": ("station", np.arange(n)),
            "lon": ("station", np.full(n, 10.0)),
            "lat": ("station", np.full(n, 20.0)),
            "area": ("station", np.asarray(areas, dtype=float)),
            "cama_lon_03min": ("station", np.full(n, 10.0)),
            "cama_lat_03min": ("station", np.full(n, 20.0)),
            "cama_alloc_err_03min": ("station", np.asarray(alloc_errs)),
            "discharge": (("station", "time"), np.ones((n, 2))),
        },
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="D")},
    ).to_netcdf(dataset_path)
    return dataset_path


def _uparea_info(tmp_path, sim_grid_res):
    return SimpleNamespace(
        casedir=str(tmp_path / "case"),
        sim_grid_res=sim_grid_res,
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


@pytest.mark.parametrize(
    ("sim_grid_res", "expected_ids"),
    [(0.25, ["1", "2"]), (0.1, ["0", "1", "2"])],
)
def test_direct_station_matching_enforces_resolution_min_uparea(tmp_path, sim_grid_res, expected_ids):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = _uparea_stations(tmp_path, [2000.0, 5000.0, np.nan])
    info = _uparea_info(tmp_path, sim_grid_res)

    run_station_matching(info, str(dataset_path), method="direct", n_jobs=1)

    assert info.stn_list["ID"].tolist() == expected_ids
    assert info.min_uparea == {0.25: 3000.0, 0.1: 500.0}[sim_grid_res]


def test_cama_station_matching_enforces_resolution_min_uparea(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = _uparea_stations(tmp_path, [100.0, 200.0])
    info = _uparea_info(tmp_path, 0.05)

    run_station_matching(info, str(dataset_path), method="cama_allocation", n_jobs=1)

    assert info.stn_list["ID"].tolist() == ["1"]


def test_direct_station_matching_rejects_unsupported_resolution(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    dataset_path = _uparea_stations(tmp_path, [5000.0])
    info = _uparea_info(tmp_path, 0.5)

    with pytest.raises(ValueError, match="no minimum upstream area"):
        run_station_matching(info, str(dataset_path), method="direct", n_jobs=1)
    assert not hasattr(info, "stn_list")


def test_cama_station_matching_drops_missing_or_large_alloc_err(tmp_path):
    from openbench.data.station_matcher import MAX_CAMA_ALLOC_ERR, run_station_matching

    alloc_errs = [np.nan, 0.0, MAX_CAMA_ALLOC_ERR, 0.25, -0.1, -0.3]
    dataset_path = _uparea_stations(tmp_path, [1000.0] * len(alloc_errs), alloc_errs)
    info = _uparea_info(tmp_path, 0.05)

    run_station_matching(info, str(dataset_path), method="cama_allocation", n_jobs=1)

    assert info.stn_list["ID"].tolist() == ["1", "2", "4"]


def test_cama_station_matching_compares_float32_alloc_err_at_stored_precision(tmp_path):
    from openbench.data.station_matcher import MAX_CAMA_ALLOC_ERR, run_station_matching

    limit = np.float32(MAX_CAMA_ALLOC_ERR)
    alloc_errs = np.array([limit, -limit, np.nextafter(limit, np.float32(1)), np.nan], dtype=np.float32)
    dataset_path = _uparea_stations(tmp_path, [1000.0] * len(alloc_errs), alloc_errs)
    with xr.open_dataset(dataset_path) as ds:
        assert ds["cama_alloc_err_03min"].dtype == np.float32
    info = _uparea_info(tmp_path, 0.05)

    run_station_matching(info, str(dataset_path), method="cama_allocation", n_jobs=1)

    assert info.stn_list["ID"].tolist() == ["0", "1"]


def test_resolve_station_dataset_prefers_full_then_dist(tmp_path):
    from openbench.data.station_matcher import resolve_station_dataset

    full = tmp_path / "Flow_Daily_full.nc"
    dist = tmp_path / "Flow_Daily_dist.nc"
    assert resolve_station_dataset(tmp_path, full.name) is None
    dist.touch()
    assert resolve_station_dataset(tmp_path, full.name) == dist
    full.touch()
    assert resolve_station_dataset(tmp_path, full.name) == full


def test_resolve_station_dataset_only_falls_back_for_full_files(tmp_path):
    from openbench.data.station_matcher import resolve_station_dataset

    (tmp_path / "GRDC_daily_dist.nc").touch()

    assert resolve_station_dataset(tmp_path, "GRDC_daily.nc") is None


def test_station_matching_falls_back_to_other_upstream_area_name(tmp_path, caplog):
    import logging

    from openbench.data.station_matcher import run_station_matching

    dataset_path = _uparea_stations(tmp_path, [100.0, 200.0])  # stored as "area"
    info = _uparea_info(tmp_path, 0.05)

    with caplog.at_level(logging.WARNING):
        run_station_matching(info, str(dataset_path), method="cama_allocation", area_var="upstream_area", n_jobs=1)

    assert info.stn_list["ID"].tolist() == ["1"]
    assert "using 'area' as upstream area" in caplog.text


def test_station_matching_warns_when_dataset_has_no_upstream_area(tmp_path, caplog):
    import logging

    from openbench.data.station_matcher import run_station_matching

    dataset_path = _uparea_stations(tmp_path, [100.0])
    with xr.open_dataset(dataset_path) as ds:
        no_area = ds.drop_vars("area").load()
    no_area.to_netcdf(tmp_path / "no_area.nc")
    info = _uparea_info(tmp_path, 0.05)

    with caplog.at_level(logging.WARNING):
        run_station_matching(
            info, str(tmp_path / "no_area.nc"), method="cama_allocation", area_var="upstream_area", n_jobs=1
        )

    assert info.stn_list["ID"].tolist() == ["0"]
    assert "minimum upstream area of 150 km2 is not applied" in caplog.text


def _sediment_stations(tmp_path):
    dataset_path = tmp_path / "sediment.nc"
    xr.Dataset(
        {
            "station": ("station", np.array(["S1"], dtype=object)),
            "lon": ("station", [10.0]),
            "lat": ("station", [20.0]),
            "upstream_area": ("station", [5000.0]),
            "cama_lon_15min": ("station", [10.0]),
            "cama_lat_15min": ("station", [20.0]),
            "cama_alloc_err_15min": ("station", [0.0]),
            "discharge": (("station", "time"), [[1.0, 2.0]]),
            "ssc": (("station", "time"), [[30.0, 40.0]]),
        },
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="D")},
    ).to_netcdf(dataset_path)
    return dataset_path


def test_station_matching_reads_the_item_variable_and_names_station_files_after_it(tmp_path):
    from openbench.data.station_matcher import run_station_matching

    info = _uparea_info(tmp_path, 0.25)
    run_station_matching(info, str(_sediment_stations(tmp_path)), area_var="upstream_area", varname="ssc", n_jobs=1)

    with xr.open_dataset(info.stn_list["ref_dir"].iloc[0]) as station_ds:
        assert list(station_ds.data_vars) == ["ssc"]
        np.testing.assert_allclose(station_ds["ssc"].values, [30.0, 40.0])


def test_station_matching_falls_back_to_discharge_var_and_keeps_the_item_name(tmp_path):
    """An older catalog names the item variable "discharge" while the file says "Disch"."""
    from openbench.data.station_matcher import run_station_matching

    path = _sediment_stations(tmp_path)
    with xr.open_dataset(path) as ds:
        renamed = ds.rename({"discharge": "Disch"}).load()
    renamed.to_netcdf(tmp_path / "grdc_like.nc")
    info = _uparea_info(tmp_path, 0.25)

    run_station_matching(
        info,
        str(tmp_path / "grdc_like.nc"),
        area_var="upstream_area",
        discharge_var="Disch",
        varname="discharge",
        varname_falls_back=True,
        n_jobs=1,
    )

    with xr.open_dataset(info.stn_list["ref_dir"].iloc[0]) as station_ds:
        assert list(station_ds.data_vars) == ["discharge"]
        np.testing.assert_allclose(station_ds["discharge"].values, [1.0, 2.0])


@pytest.mark.parametrize("varname", ["ssc", "ssl"])
def test_station_matching_never_reads_discharge_for_a_missing_sediment_variable(tmp_path, varname):
    from openbench.data.station_matcher import run_station_matching
    from openbench.util.exceptions import DataProcessingError

    path = _sediment_stations(tmp_path)
    with xr.open_dataset(path) as ds:
        discharge_only = ds.drop_vars("ssc").load()
    discharge_only.to_netcdf(tmp_path / "discharge_only.nc")
    info = _uparea_info(tmp_path, 0.25)

    with pytest.raises(DataProcessingError, match=f"'{varname}'"):
        run_station_matching(
            info, str(tmp_path / "discharge_only.nc"), area_var="upstream_area", varname=varname, n_jobs=1
        )
    assert not hasattr(info, "stn_list")
