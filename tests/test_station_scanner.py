"""Tests for station simulation directory scanning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def test_scan_station_sim_dir_keeps_flat_station_files_without_coordinates(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    path = tmp_path / "sim_AT-Neu_2004-2005.nc"
    ds = xr.Dataset(
        {"f_fevpa": (["time"], np.zeros(731, dtype=np.float32))},
        coords={"time": pd.date_range("2004-01-01", periods=731, freq="D")},
    )
    ds.to_netcdf(path)

    df = scan_station_sim_dir(str(tmp_path))

    assert list(df["ID"]) == ["AT-Neu"]
    assert df.loc[0, "use_syear"] == 2004
    assert df.loc[0, "use_eyear"] == 2005
    assert df.loc[0, "sim_dir"] == str(path)
    assert pd.isna(df.loc[0, "sim_lon"])
    assert pd.isna(df.loc[0, "sim_lat"])


def test_scan_station_sim_dir_rejects_unitless_numeric_time_as_year_range(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    path = tmp_path / "US-AAA.nc"
    ds = xr.Dataset(
        {"QFLX_EVAP_TOT": (["time"], np.zeros(12, dtype=np.float32))},
        coords={"time": np.arange(12), "lat": 40.0, "lon": -100.0},
    )
    ds.to_netcdf(path)

    with pytest.raises(FileNotFoundError, match="No valid station NC files"):
        scan_station_sim_dir(str(tmp_path))


def test_scan_station_sim_dir_skips_duplicate_child_symlink(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    case_dir = tmp_path / "US-AAA"
    path = case_dir / "US-AAA.nc"
    ds = xr.Dataset(
        {"QFLX_EVAP_TOT": (["time"], np.zeros(12, dtype=np.float32))},
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS"), "lat": 40.0, "lon": -100.0},
    )
    path.parent.mkdir()
    ds.to_netcdf(path)
    try:
        (tmp_path / "US-AAA-alias").symlink_to(case_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not permitted on this platform")

    df = scan_station_sim_dir(str(tmp_path))

    assert list(df["ID"]) == ["US-AAA"]


def test_scan_station_sim_dir_uses_parent_station_case_metadata(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    root = tmp_path / "Initial_test"
    stn_dir = root / "stn"
    stn_dir.mkdir(parents=True)
    path = stn_dir / "sim_US-ARM_2004-2005.nc"
    ds = xr.Dataset(
        {"f_lfevpa": (["time", "patch"], np.zeros((731, 1), dtype=np.float32))},
        coords={"time": pd.date_range("2004-01-01", periods=731, freq="D")},
    )
    ds.to_netcdf(path)
    (root / "station_case.csv").write_text(
        "scenario,ID,SYEAR,EYEAR,LON,LAT,DIR\ncase,US-ARM,2004,2005,-97.4888,36.6058,stn/sim_US-ARM_2004-2005.nc\n"
    )

    df = scan_station_sim_dir(str(stn_dir))

    assert list(df["ID"]) == ["US-ARM"]
    assert df.loc[0, "sim_lon"] == -97.4888
    assert df.loc[0, "sim_lat"] == 36.6058
    assert df.loc[0, "use_syear"] == 2004
    assert df.loc[0, "use_eyear"] == 2005


def test_scan_station_sim_dir_ignores_static_auxiliary_child_dirs(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    history = tmp_path / "history"
    history.mkdir()
    for month in range(1, 3):
        ds = xr.Dataset(
            {"f_qintr": (["time"], np.full(1, month, dtype=np.float32))},
            coords={
                "time": pd.date_range(f"2004-{month:02d}-01", periods=1, freq="MS"),
                "lon": 10.0,
                "lat": 40.0,
            },
        )
        ds.to_netcdf(history / f"case_hist_2004-{month:02d}.nc")

    landdata = tmp_path / "landdata"
    landdata.mkdir()
    xr.Dataset(
        {"landmask": (["patch"], np.ones(1, dtype=np.float32))},
        coords={"patch": [0]},
    ).to_netcdf(landdata / "case_landdata.nc")

    df, dropped = scan_station_sim_dir(str(tmp_path), output_dir=str(tmp_path / "merged"), return_dropped=True)

    assert dropped == []
    assert list(df["ID"]) == ["history"]
    assert df.loc[0, "use_syear"] == 2004
    assert df.loc[0, "use_eyear"] == 2004


def test_extract_station_data_honors_configured_num_cores(tmp_path, monkeypatch):
    import openbench.data.processing as processing
    from openbench.data.processing import GridDatasetProcessing

    output = tmp_path / "flat.nc"
    xr.Dataset(
        {"value": (["time", "lat", "lon"], np.zeros((1, 1, 1), dtype=np.float32))},
        coords={"time": [0], "lat": [1.0], "lon": [2.0]},
    ).to_netcdf(output)

    calls = []

    class FakeParallel:
        def __init__(self, n_jobs):
            calls.append(n_jobs)

        def __call__(self, tasks):
            list(tasks)
            return []

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return (func, args, kwargs)

        return wrapper

    proc = GridDatasetProcessing.__new__(GridDatasetProcessing)
    proc.num_cores = 1
    proc.station_list = pd.DataFrame({"ID": ["US-ARM"]})
    proc.get_output_filename = lambda params: str(output)
    proc._extract_stn_parallel = lambda *args, **kwargs: None

    monkeypatch.setattr(processing, "Parallel", FakeParallel)
    monkeypatch.setattr(processing, "delayed", fake_delayed)

    proc.extract_station_data({"datasource": "ref"})

    assert calls == [1]


def test_extract_station_data_loads_flat_dataset_before_parallel(tmp_path, monkeypatch):
    """Station extraction should detach from the flat NC before passing data to workers."""
    from openbench.data.processing import GridDatasetProcessing

    output = tmp_path / "flat.nc"
    xr.Dataset(
        {"value": (("time", "lat", "lon"), np.ones((1, 1, 1), dtype=np.float32))},
        coords={"time": pd.date_range("2000-01-01", periods=1), "lat": [1.0], "lon": [2.0]},
    ).to_netcdf(output)

    proc = GridDatasetProcessing.__new__(GridDatasetProcessing)
    proc.num_cores = 1
    proc.station_list = pd.DataFrame({"ID": ["US-ARM"]})
    proc.get_output_filename = lambda params: str(output)
    proc._extract_stn_parallel = lambda *args, **kwargs: None

    original_load = xr.Dataset.load
    calls = {"load": 0}

    def spy_load(self, **kwargs):
        calls["load"] += 1
        return original_load(self, **kwargs)

    monkeypatch.setattr(xr.Dataset, "load", spy_load)

    proc.extract_station_data({"datasource": "ref"})

    assert calls["load"] >= 1
    assert not output.exists()


def test_process_station_data_honors_configured_num_cores(monkeypatch):
    import openbench.data.processing as processing
    from openbench.data.processing import StationDatasetProcessing

    calls = []

    class FakeParallel:
        def __init__(self, n_jobs):
            calls.append(n_jobs)

        def __call__(self, tasks):
            list(tasks)
            return []

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return (func, args, kwargs)

        return wrapper

    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.num_cores = 1
    proc.station_list = pd.DataFrame(
        {"ID": ["US-ARM"], "use_syear": [2004], "use_eyear": [2005], "sim_dir": ["unused.nc"]}
    )
    proc._make_stn_parallel = lambda *args, **kwargs: None

    monkeypatch.setattr(processing, "Parallel", FakeParallel)
    monkeypatch.setattr(processing, "delayed", fake_delayed)

    proc.process_station_data({"datasource": "sim"})

    assert calls == [1]


def test_setup_output_directories_writes_canonical_station_list_from_sim_fulllist(tmp_path: Path):
    from openbench.data.processing import BaseDatasetProcessing

    fulllist = tmp_path / "sim_stations.csv"
    pd.DataFrame(
        [
            {
                "ID": "US-ARM",
                "sim_lon": -97.4888,
                "sim_lat": 36.6058,
                "use_syear": 2004,
                "use_eyear": 2005,
                "sim_dir": "sim_US-ARM_2004-2005.nc",
            }
        ]
    ).to_csv(fulllist, index=False)

    proc = BaseDatasetProcessing.__new__(BaseDatasetProcessing)
    proc.ref_data_type = "grid"
    proc.sim_data_type = "stn"
    proc.ref_source = "FLUXCOM_LowRes"
    proc.sim_source = "stn"
    proc.ref_fulllist = ""
    proc.sim_fulllist = str(fulllist)
    proc.sim_dir = str(tmp_path)
    proc.casedir = str(tmp_path / "case")

    proc.setup_output_directories()

    canonical = tmp_path / "case" / "stn_FLUXCOM_LowRes_stn_list.txt"
    assert canonical.exists()
    saved = pd.read_csv(canonical)
    assert list(saved["ID"]) == ["US-ARM"]
    assert proc.ref_fulllist == str(canonical)


def test_station_evaluation_runs_sequential_when_num_cores_is_one(tmp_path, monkeypatch):
    import openbench.core.evaluation as evaluation
    from openbench.core.evaluation import Evaluation_stn

    stnlist = tmp_path / "stn_FLUXCOM_LowRes_stn_list.txt"
    pd.DataFrame(
        [
            {
                "ID": "US-ARM",
                "sim_lon": -97.4888,
                "sim_lat": 36.6058,
                "use_syear": 2004,
                "use_eyear": 2005,
            }
        ]
    ).to_csv(stnlist, index=False)

    ev = Evaluation_stn.__new__(Evaluation_stn)
    ev.casedir = str(tmp_path)
    ev.item = "Latent_Heat"
    ev.ref_source = "FLUXCOM_LowRes"
    ev.sim_source = "stn"
    ev.ref_fulllist = str(stnlist)
    ev.num_cores = 1
    ev.output_manager = None
    ev.make_evaluation_parallel = lambda station_list, i: {
        "KGESS": 1.0,
        "RMSE": 0.0,
        "correlation": 1.0,
        "bias": 0.0,
        "Overall_Score": 1.0,
    }

    def fail_parallel(*args, **kwargs):
        raise PermissionError("parallel path should not be used for one core")

    monkeypatch.setattr(evaluation, "parallel_map", fail_parallel)
    monkeypatch.setattr(evaluation, "make_plot_index_stn", lambda self: None)

    ev.make_evaluation_P()

    metrics = tmp_path / "metrics" / "Latent_Heat_stn_FLUXCOM_LowRes_stn_evaluations.csv"
    assert metrics.exists()
    saved = pd.read_csv(metrics)
    assert list(saved["ID"]) == ["US-ARM"]


def test_station_taylor_comparison_runs_sequential_when_num_cores_is_one(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison
    from openbench.core.comparison import ComparisonProcessing

    casedir = tmp_path
    metrics_dir = casedir / "metrics"
    data_dir = casedir / "data" / "stn_FLUXCOM_LowRes_stn"
    metrics_dir.mkdir()
    data_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "ID": "US-ARM",
                "sim_lon": -97.4888,
                "sim_lat": 36.6058,
                "use_syear": 2004,
                "use_eyear": 2005,
                "bias": 0.0,
                "RMSE": 0.0,
                "correlation": 1.0,
                "Overall_Score": 1.0,
            }
        ]
    ).to_csv(metrics_dir / "Latent_Heat_stn_FLUXCOM_LowRes_stn_evaluations.csv", index=False)

    time = pd.date_range("2004-01-01", periods=2, freq="D")
    xr.Dataset({"f_lfevpa": ("time", np.array([1.0, 2.0], dtype=np.float32))}, coords={"time": time}).to_netcdf(
        data_dir / "Latent_Heat_sim_US-ARM_2004_2005.nc"
    )
    xr.Dataset({"le": ("time", np.array([1.0, 2.0], dtype=np.float32))}, coords={"time": time}).to_netcdf(
        data_dir / "Latent_Heat_ref_US-ARM_2004_2005.nc"
    )

    cp = ComparisonProcessing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Day",
                "num_cores": 1,
                "weight": "none",
            }
        },
        scores=[],
        metrics=[],
    )

    def fail_parallel(*args, **kwargs):
        raise PermissionError("parallel path should not be used for one core")

    monkeypatch.setattr(comparison, "Parallel", fail_parallel)
    monkeypatch.setattr(comparison, "make_scenarios_comparison_Taylor_Diagram", lambda *args, **kwargs: None)

    cp.scenarios_Taylor_Diagram_comparison(
        str(casedir),
        {
            "general": {"Latent_Heat_sim_source": ["stn"]},
            "Latent_Heat": {"stn_data_type": "stn", "stn_varname": "f_lfevpa"},
        },
        {
            "general": {"Latent_Heat_ref_source": "FLUXCOM_LowRes"},
            "Latent_Heat": {"FLUXCOM_LowRes_data_type": "grid", "FLUXCOM_LowRes_varname": "le"},
        },
        ["Latent_Heat"],
        [],
        [],
        {},
    )

    assert (casedir / "comparisons" / "Taylor_Diagram" / "taylor_diagram__Latent_Heat__FLUXCOM_LowRes.csv").exists()


def test_merge_site_uses_atomic_netcdf_write():
    source = Path("src/openbench/data/station_scanner.py").read_text()

    assert ".to_netcdf(" not in source


def test_scan_station_sim_dir_parses_compact_year_range(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    path = tmp_path / "DE-Tha_20002010.nc"
    ds = xr.Dataset(
        {"Q": (["time"], np.zeros(2, dtype=np.float32))},
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="YS"), "lat": 1.0, "lon": 2.0},
    )
    ds.to_netcdf(path)

    df = scan_station_sim_dir(str(tmp_path))

    assert df.loc[0, "ID"] == "DE-Tha"
    assert df.loc[0, "use_syear"] == 2000
    assert df.loc[0, "use_eyear"] == 2010


def test_scan_station_sim_dir_loads_common_station_sidecar_names(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    path = tmp_path / "US-ABC.nc"
    ds = xr.Dataset(
        {"Q": (["time"], np.zeros(2, dtype=np.float32))},
        coords={"time": pd.date_range("2001-01-01", periods=2, freq="YS")},
    )
    ds.to_netcdf(path)
    (tmp_path / "stations.csv").write_text("ID,LON,LAT,SYEAR,EYEAR\nUS-ABC,10,20,1999,2005\n", encoding="utf-8")

    df = scan_station_sim_dir(str(tmp_path))

    assert df.loc[0, "sim_lon"] == 10
    assert df.loc[0, "sim_lat"] == 20
    assert df.loc[0, "use_syear"] == 1999
    assert df.loc[0, "use_eyear"] == 2005


def test_nested_multi_station_rescan_rebuilds_stale_merged_year_range(tmp_path: Path):
    from openbench.data.station_scanner import scan_station_sim_dir

    root = tmp_path / "raw"
    site = root / "US-AAA"
    merge_dir = tmp_path / "merged"
    site.mkdir(parents=True)
    merge_dir.mkdir()

    def write_part(name: str, start: str):
        path = site / name
        ds = xr.Dataset(
            {"QFLX_EVAP_TOT": ("time", np.array([1.0], dtype=np.float32))},
            coords={"time": pd.to_datetime([start]), "lat": 40.0, "lon": -100.0},
        )
        ds.to_netcdf(path)
        return path

    write_part("part_2000_a.nc", "2000-01-01")
    write_part("part_2000_b.nc", "2000-02-01")
    first = scan_station_sim_dir(str(root), output_dir=str(merge_dir))
    assert first.loc[0, "use_eyear"] == 2000
    assert Path(first.loc[0, "sim_dir"]).name == "sim_US-AAA_2000_2000.nc"

    write_part("part_2001.nc", "2001-01-01")
    second = scan_station_sim_dir(str(root), output_dir=str(merge_dir))

    assert second.loc[0, "use_syear"] == 2000
    assert second.loc[0, "use_eyear"] == 2001
    assert Path(second.loc[0, "sim_dir"]).name == "sim_US-AAA_2000_2001.nc"
