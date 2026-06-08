"""Tests for simulation output discovery and metadata inference."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _write_grid_nc(
    path: Path,
    var_name: str = "QFLX_EVAP_TOT",
    periods: int = 12,
    start: str = "2000-01-01",
    freq: str = "MS",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, periods=periods, freq=freq)
    ds = xr.Dataset(
        {var_name: (["time", "lat", "lon"], np.zeros((periods, 3, 4)))},
        coords={"time": times, "lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds[var_name].attrs["units"] = "mm day-1"
    ds.to_netcdf(path)


def _write_static_grid_nc(path: Path, var_name: str = "QFLX_EVAP_TOT") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {var_name: (["lat", "lon"], np.zeros((3, 4)))},
        coords={"lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds[var_name].attrs["units"] = "mm day-1"
    ds.to_netcdf(path)


def _write_station_nc(
    path: Path,
    *,
    site_id: str = "US-AAA",
    lon: float = -100.0,
    lat: float = 40.0,
    start: str = "2000-01-01",
    periods: int = 2,
    freq: str = "MS",
    var_name: str = "QFLX_EVAP_TOT",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, periods=periods, freq=freq)
    ds = xr.Dataset(
        {var_name: (["time"], np.zeros(periods))},
        coords={"time": times, "lon": lon, "lat": lat},
        attrs={"station_id": site_id},
    )
    ds[var_name].attrs["units"] = "mm day-1"
    ds.to_netcdf(path)


def test_scan_simulation_roots_discovers_cases_at_default_depth_five(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "project" / "CoLM2024" / "group" / "CaseA" / "history"
    _write_grid_nc(history / "QFLX_EVAP_TOT_2000.nc")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    case = result.cases[0]
    assert case.root_dir == history
    assert case.model == "CoLM2024"
    assert case.depth == 5


def test_scan_simulation_roots_skips_generated_derived_output_dirs(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    derived = root / "CaseA_derived"
    _write_grid_nc(history / "QFLX_EVAP_TOT_2000.nc")
    _write_grid_nc(derived / "canopy_CaseA_2000.nc", var_name="Canopy_Evaporation")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]


def test_scan_simulation_roots_skips_symlink_cycles(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_grid_nc(history / "QFLX_EVAP_TOT_2000.nc")
    try:
        (root / "loop").symlink_to(root, target_is_directory=True)
    except OSError:
        import pytest

        pytest.skip("symlink creation is not permitted on this platform")

    result = scan_simulation_roots([root], model_name="CoLM2024", case_depth=8)

    assert [case.label for case in result.cases] == ["CaseA"]


def test_scan_simulation_roots_uses_case_label_when_root_is_case_dir(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    case_root = root / "CaseA"
    history = case_root / "history"
    _write_grid_nc(history / "QFLX_EVAP_TOT_2000.nc")

    result = scan_simulation_roots([case_root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    assert result.cases[0].root_dir == history


def test_scan_simulation_roots_uses_case_label_when_root_is_data_dir(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_grid_nc(history / "QFLX_EVAP_TOT_2000.nc")

    result = scan_simulation_roots([history], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    assert result.cases[0].root_dir == history


def test_scan_simulation_roots_discovers_nested_station_collection_as_one_case(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    case_root = root / "CaseA"
    _write_station_nc(case_root / "US-AAA" / "history" / "US-AAA_2000.nc", site_id="US-AAA")
    _write_station_nc(
        case_root / "US-AAA" / "history" / "US-AAA_2001.nc",
        site_id="US-AAA",
        start="2001-01-01",
    )
    _write_station_nc(case_root / "US-BBB" / "history" / "US-BBB_2000.nc", site_id="US-BBB")
    _write_station_nc(
        case_root / "US-BBB" / "history" / "US-BBB_2001.nc",
        site_id="US-BBB",
        start="2001-01-01",
    )

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    case = result.cases[0]
    assert case.root_dir == case_root
    assert case.data_type == "stn"
    assert case.data_groupby == "Single"
    assert case.station_layout == "nested_multi"
    assert case.variables == ["QFLX_EVAP_TOT"]


def test_scan_simulation_roots_ignores_station_aux_dirs_and_counts_all_history_files(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    case_root = root / "CaseA"
    for month in range(1, 13):
        _write_station_nc(
            case_root / "history" / f"CaseA_hist_2004-{month:02d}.nc",
            site_id="CaseA",
            start=f"2004-{month:02d}-01",
            periods=1,
        )
    _write_static_grid_nc(case_root / "landdata" / "CaseA_landdata.nc")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    case = result.cases[0]
    assert case.station_layout == "nested_multi"
    assert case.time_count == 12
    assert case.time_start == "2004-01-01T00:00:00"
    assert case.time_end == "2004-12-01T00:00:00"
    assert case.years == [2004, 2004]


def test_scan_simulation_roots_keeps_sibling_flat_station_cases_separate(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    _write_station_nc(root / "CaseA" / "US-AAA_2000.nc", site_id="US-AAA")
    _write_station_nc(root / "CaseB" / "US-BBB_2000.nc", site_id="US-BBB")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA", "CaseB"]
    assert [case.root_dir for case in result.cases] == [root / "CaseA", root / "CaseB"]
    assert [case.station_layout for case in result.cases] == ["flat", "flat"]


def test_materialize_station_cases_keeps_unicode_case_labels_distinct(tmp_path: Path, monkeypatch):
    import openbench.data.station_scanner as station_scanner
    from openbench.data.sim_scanner import (
        SimulationCase,
        SimulationScanResult,
        materialize_station_cases,
    )

    case_a = SimulationCase(
        label="测试",
        root_dir=tmp_path / "simulations" / "a",
        model="CoLM",
        depth=1,
        data_type="stn",
        station_layout="flat",
    )
    case_b = SimulationCase(
        label="案例",
        root_dir=tmp_path / "simulations" / "b",
        model="CoLM",
        depth=1,
        data_type="stn",
        station_layout="flat",
    )
    result = SimulationScanResult(roots=[tmp_path / "simulations"], cases=[case_a, case_b])

    def fake_scan_station_sim_dir(root_dir, *, output_dir, num_workers=4, return_dropped=False):
        station_id = Path(root_dir).name
        return (
            pd.DataFrame(
                {
                    "ID": [station_id],
                    "sim_dir": [str(Path(root_dir) / f"{station_id}.nc")],
                }
            ),
            [],
        )

    monkeypatch.setattr(station_scanner, "scan_station_sim_dir", fake_scan_station_sim_dir)

    materialize_station_cases(result, tmp_path / "station_lists")

    assert case_a.fulllist.parent.name == "测试"
    assert case_b.fulllist.parent.name == "案例"
    assert case_a.fulllist != case_b.fulllist
    assert case_a.fulllist.exists()
    assert case_b.fulllist.exists()


def test_scan_simulation_roots_infers_flat_station_tim_res_per_station_not_across_sites(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    for index in range(10):
        site_id = f"US-A{index:02d}"
        _write_station_nc(
            root / "CaseA" / f"sim_{site_id}_2004-2005.nc",
            site_id=site_id,
            lon=-100.0 - index,
            lat=40.0 + index,
            start="2004-01-01",
            periods=731,
            freq="D",
        )

    result = scan_simulation_roots([root], model_name="CoLM")

    case = result.cases[0]
    assert case.data_type == "stn"
    assert case.tim_res == "Day"
    assert case.years == [2004, 2005]
    assert case.time_count == 731
    assert case.time_start == "2004-01-01T00:00:00"
    assert case.time_end == "2005-12-31T00:00:00"


def test_scan_simulation_roots_groups_named_station_history_dirs(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    case_root = root / "CaseA"
    _write_station_nc(case_root / "Loobos" / "history" / "Loobos_2000.nc", site_id="Loobos")
    _write_station_nc(case_root / "Hyytiala" / "history" / "Hyytiala_2000.nc", site_id="Hyytiala")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    case = result.cases[0]
    assert case.root_dir == case_root
    assert case.station_layout == "nested_single"


def test_scan_simulation_roots_groups_named_station_direct_dirs(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    case_root = root / "CaseA"
    _write_station_nc(case_root / "Loobos" / "Loobos_2000.nc", site_id="Loobos")
    _write_station_nc(case_root / "Hyytiala" / "Hyytiala_2000.nc", site_id="Hyytiala")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    assert [case.label for case in result.cases] == ["CaseA"]
    case = result.cases[0]
    assert case.root_dir == case_root
    assert case.station_layout == "nested_single"


def test_scan_simulation_roots_deduplicates_duplicate_case_labels(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    _write_grid_nc(root_a / "CaseA" / "history" / "QFLX_EVAP_TOT_2000.nc")
    _write_grid_nc(root_b / "CaseA" / "history" / "QFLX_EVAP_TOT_2000.nc")

    result = scan_simulation_roots([root_a, root_b], model_name="CoLM2024")

    assert len(result.cases) == 2
    labels = sorted(case.label for case in result.cases)
    assert labels == sorted({"CaseA", "CaseA__root_b"})
    assert len({case.label for case in result.cases}) == 2
    duplicate_case = next(case for case in result.cases if case.label != "CaseA")
    assert duplicate_case.provenance["label"] == "deduplicated_by_root"
    assert duplicate_case.provenance["original_label"] == "CaseA"


def test_scan_simulation_roots_infers_grid_metadata_from_nc_and_filenames(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_grid_nc(history / "caseA_QFLX_EVAP_TOT_2000_hist.nc")
    _write_grid_nc(history / "caseA_QFLX_EVAP_TOT_2001_hist.nc", start="2001-01-01")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.data_type == "grid"
    assert case.tim_res == "Month"
    assert case.grid_res == 0.5
    assert case.data_groupby == "Year"
    assert case.years == [2000, 2001]
    assert case.prefix == "caseA_QFLX_EVAP_TOT_"
    assert case.suffix == "_hist"
    assert case.variables == ["QFLX_EVAP_TOT"]


def test_scan_simulation_roots_treats_single_year_token_file_as_year_groupby(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_grid_nc(history / "caseA_QFLX_EVAP_TOT_2000_hist.nc")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.data_groupby == "Year"
    assert case.prefix == "caseA_QFLX_EVAP_TOT_"
    assert case.suffix == "_hist"


def test_scan_simulation_roots_prefers_output_month_token_over_experiment_year_range(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CLM5"
    path = history / "IHistClm50BgcCropGSWP_CWD-1901-2014.clm2.h0.2002-02.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"QFLX_EVAP_TOT": (["time", "lat", "lon"], np.zeros((1, 2, 2), dtype=np.float32))},
        coords={"time": np.array([36950]), "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds["time"].attrs["units"] = "days since 1901-01-01 00:00:00"
    ds["time"].attrs["calendar"] = "noleap"
    ds["QFLX_EVAP_TOT"].attrs["units"] = "mm day-1"
    ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="CLM5")

    case = result.cases[0]
    assert case.data_groupby == "Month"
    assert case.tim_res == "Month"
    assert case.years == [2002, 2002]
    assert case.prefix == "IHistClm50BgcCropGSWP_CWD-1901-2014.clm2.h0."
    assert case.suffix == ""


def test_scan_simulation_roots_uses_month_file_shape_when_coverage_has_gaps(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "GLDAS"
    stamps = [f"{year}{month:02d}" for year in (2000, 2001) for month in (1, 3, 5, 7, 9, 11)]
    for stamp in stamps:
        year = int(stamp[:4])
        month = int(stamp[4:])
        ds = xr.Dataset(
            {"Evap_tavg": (["time", "lat", "lon"], np.zeros((1, 2, 2), dtype=np.float32))},
            coords={
                "time": np.array([f"{year}-{month:02d}-01"], dtype="datetime64[ns]"),
                "lat": [0.0, 0.25],
                "lon": [100.0, 100.25],
            },
        )
        ds["Evap_tavg"].attrs["units"] = "kg m-2 s-1"
        path = history / f"GLDAS_NOAH025_M.A{stamp}.021.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="GLDAS2")

    case = result.cases[0]
    assert case.data_groupby == "Month"
    assert case.tim_res == "Month"
    assert case.provenance["tim_res"] == "time"


def test_time_info_from_file_disables_timedelta_decode_warning(tmp_path: Path):
    from openbench.data.sim_scanner import _time_info_from_file

    path = tmp_path / "GLDAS_NOAH025_M.A200001.021.nc"
    ds = xr.Dataset(
        {
            "SNOW_PERSISTENCE": (
                ["time"],
                np.array([1.0], dtype=np.float32),
                {"units": "days"},
            )
        },
        coords={"time": pd.date_range("2000-01-01", periods=1, freq="MS")},
    )
    ds.to_netcdf(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = _time_info_from_file(path)

    assert info["has_time"] is True
    assert not any("SNOW_PERSISTENCE" in str(item.message) and "timedelta64" in str(item.message) for item in caught)


def test_infer_time_coverage_samples_large_single_step_monthly_files(
    tmp_path: Path,
    monkeypatch,
):
    import openbench.data.sim_scanner as sim_scanner

    for year in range(2000, 2010):
        for month in range(1, 13):
            (tmp_path / f"Case_hist_{year}-{month:02d}.nc").write_text("")

    calls = []

    def fake_time_info(file_path: Path) -> dict:
        calls.append(file_path)
        return {
            "readable": True,
            "has_time": True,
            "values": [],
            "time_size": 1,
            "time_units": "days since 2000-01-01 00:00:00",
        }

    monkeypatch.setattr(sim_scanner, "_time_info_from_file", fake_time_info)

    coverage = sim_scanner._infer_time_coverage(tmp_path)

    assert len(calls) <= 12
    assert coverage["time_start"] == "2000-01-01T00:00:00"
    assert coverage["time_end"] == "2009-12-01T00:00:00"
    assert coverage["time_count"] == 120
    assert coverage["years"] == [2000, 2009]


def test_scan_simulation_roots_infers_daily_tim_res_from_monthly_files(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    path = history / "caseA_QFLX_EVAP_TOT_2000-01.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"QFLX_EVAP_TOT": (["time", "lat", "lon"], np.zeros((31, 3, 4)))},
        coords={"time": np.arange(31), "lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds["time"].attrs["units"] = "calendar days since 2000-01-01 00:00:00 ; "
    ds["QFLX_EVAP_TOT"].attrs["units"] = "mm day-1"
    ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.data_groupby == "Month"
    assert case.tim_res == "Day"
    assert case.provenance["tim_res"] in {"nc", "time"}


def test_scan_simulation_roots_infers_monthly_tim_res_from_calendar_month_axis(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    path = history / "YEE2_JRA-55_outflw_M2000_GLB025.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"outflw": (["time", "lat", "lon"], np.zeros((12, 3, 4)))},
        coords={"time": np.arange(12), "lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds["time"].attrs["units"] = "calendar months since 2000-01-01 00:00:00 ; "
    ds["outflw"].attrs["units"] = "m3/s"
    ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="TE")

    case = result.cases[0]
    assert case.data_groupby == "Year"
    assert case.tim_res == "Month"
    assert case.provenance["tim_res"] == "time"


def test_scan_simulation_roots_infers_multi_month_tim_res_from_calendar_month_axis(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    path = history / "YEE2_JRA-55_CDSTM_M2000_GLB050.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"CDSTM": (["time", "lat", "lon"], np.zeros((12, 3, 4)))},
        coords={"time": np.arange(0, 36, 3), "lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds["time"].attrs["units"] = "calendar months since 2000-01-01 00:00:00 ; "
    ds["CDSTM"].attrs["units"] = "m3"
    ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="TE")

    case = result.cases[0]
    assert case.data_groupby == "Year"
    assert case.tim_res == "Month"
    assert case.time_start == "2000-01-01T00:00:00"
    assert case.time_end == "2000-12-01T00:00:00"
    assert case.provenance["tim_res"] == "time"


def test_scan_simulation_roots_counts_time_for_selected_filename_stream(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CLM5"
    for stream in ("h0", "h1"):
        for month in (1, 2):
            path = history / f"IHistClm50BgcCropGSWP_CWD-1901-2014.clm2.{stream}.2000-{month:02d}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            ds = xr.Dataset(
                {"QFLX_EVAP_TOT": (["time", "lat", "lon"], np.zeros((1, 2, 2), dtype=np.float32))},
                coords={
                    "time": np.array([f"2000-{month:02d}-01"], dtype="datetime64[ns]"),
                    "lat": [0.0, 1.0],
                    "lon": [0.0, 1.0],
                },
            )
            ds["QFLX_EVAP_TOT"].attrs["units"] = "mm day-1"
            ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="CLM5")

    case = result.cases[0]
    assert case.prefix == "IHistClm50BgcCropGSWP_CWD-1901-2014.clm2.h0."
    assert case.time_count == 2
    assert case.time_start == "2000-01-01T00:00:00"
    assert case.time_end == "2000-02-01T00:00:00"


def test_scan_simulation_roots_infers_variable_file_overrides_from_model_profile(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "TE"
    files = [
        ("LTNT", "GLB050", 0.5),
        ("SENS", "GLB050", 0.5),
        ("outflw", "GLB025", 0.25),
        ("alb", "GLB050", 0.5),
    ]
    for var_name, suffix, step in files:
        path = history / f"YEE2_JRA-55_{var_name}_M2000_{suffix}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {var_name: (["time", "lat", "lon"], np.zeros((12, 3, 4), dtype=np.float32))},
            coords={
                "time": np.arange(12),
                "lat": [10.0, 10.0 + step, 10.0 + 2 * step],
                "lon": [100.0, 100.0 + step, 100.0 + 2 * step, 100.0 + 3 * step],
            },
        )
        ds["time"].attrs["units"] = "calendar months since 2000-01-01 00:00:00 ; "
        ds[var_name].attrs["units"] = "1"
        ds.to_netcdf(path)

    result = scan_simulation_roots([root], model_name="TE")

    case = result.cases[0]
    assert case.variable_overrides["Latent_Heat"] == {
        "prefix": "YEE2_JRA-55_LTNT_M",
        "suffix": "_GLB050",
    }
    assert case.variable_overrides["Sensible_Heat"] == {
        "prefix": "YEE2_JRA-55_SENS_M",
        "suffix": "_GLB050",
    }
    assert case.variable_overrides["Streamflow"] == {
        "prefix": "YEE2_JRA-55_outflw_M",
        "suffix": "_GLB025",
        "grid_res": 0.25,
    }
    assert case.variable_overrides["Albedo"] == {
        "varname": "alb",
        "prefix": "YEE2_JRA-55_alb_M",
        "suffix": "_GLB050",
    }


def test_scan_simulation_roots_reports_time_coverage_from_nc_time(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_grid_nc(history / "merged_without_year.nc", periods=24)

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.data_groupby == "Single"
    assert case.years == [2000, 2001]
    assert case.time_start == "2000-01-01T00:00:00"
    assert case.time_end == "2001-12-01T00:00:00"
    assert case.time_count == 24
    assert case.time_span_days == 700
    assert case.provenance["years"] == "nc"
    assert case.provenance["time_coverage"] == "nc"


def test_scan_simulation_roots_uses_all_no_date_files_for_time_coverage(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    for index, timestamp in enumerate(pd.date_range("2000-01-01", periods=24, freq="MS")):
        _write_grid_nc(history / f"chunk_{index:02d}.nc", periods=1, start=str(timestamp.date()))

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.time_count == 24
    assert case.years == [2000, 2001]
    assert case.time_start == "2000-01-01T00:00:00"
    assert case.time_end == "2001-12-01T00:00:00"
    assert case.tim_res == "Month"
    # Multi-undated chunks now keep the longest common stem prefix so downstream
    # lookups have something to glob, and the case is flagged unresolved.
    assert case.prefix == "chunk_"
    assert case.suffix == ""
    assert "multi_undated_files" in case.unresolved
    assert case.temporal_kind != "climatology-year"


def test_scan_simulation_roots_does_not_treat_unitless_numeric_time_as_1970(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    history.mkdir(parents=True)
    ds = xr.Dataset(
        {"QFLX_EVAP_TOT": (["time", "lat", "lon"], np.zeros((12, 2, 2)))},
        coords={"time": np.arange(12), "lat": [0.0, 0.5], "lon": [10.0, 10.5]},
    )
    ds.to_netcdf(history / "merged_without_dates.nc")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.time_count == 12
    assert case.years is None
    assert case.time_start is None
    assert case.time_end is None
    assert case.tim_res is None


def test_scan_simulation_roots_marks_monthly_climatology_candidate_from_hint_and_12_months(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_grid_nc(history / "monthly_climatology.nc", periods=12)

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.temporal_kind is None
    assert case.temporal_kind_candidate == "climatology-month"
    assert case.tim_res == "Month"
    assert case.data_groupby == "Single"
    assert case.provenance["temporal_kind"] == "candidate"


def test_scan_simulation_roots_detects_annual_climatology_without_time_dimension(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CaseA" / "history"
    _write_static_grid_nc(history / "annual_climatology.nc")

    result = scan_simulation_roots([root], model_name="CoLM2024")

    case = result.cases[0]
    assert case.temporal_kind == "climatology-year"
    assert case.tim_res == "climatology-year"
    assert case.time_count == 0
    assert case.provenance["temporal_kind"] == "auto"


def test_scan_simulation_roots_auto_model_matches_existing_profile_name_in_path(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "CoLM2024_experiment" / "history"
    _write_grid_nc(history / "QFLX_EVAP_TOT_2000.nc")

    result = scan_simulation_roots([root], model_name="auto")

    case = result.cases[0]
    assert case.model == "CoLM2024"
    assert case.provenance["model"] == "path"
    assert result.unresolved == []


def test_scan_simulation_roots_auto_model_matches_variable_case_insensitively(tmp_path: Path):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "experiment" / "history"
    _write_grid_nc(history / "CANOPY_EVAPORATION_2000.nc", var_name="CANOPY_EVAPORATION")

    result = scan_simulation_roots([root], model_name="auto")

    case = result.cases[0]
    assert case.model == "CoLM2024"
    assert case.provenance["model"] == "variables"
    assert result.unresolved == []


def test_scan_simulation_roots_auto_model_prefers_strong_variable_match_over_generic_path(
    tmp_path: Path,
):
    from openbench.data.sim_scanner import scan_simulation_roots

    root = tmp_path / "simulations"
    history = root / "colm-routing"
    history.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    data = np.zeros((2, 3, 4))
    ds = xr.Dataset(
        {
            "f_discharge": (["time", "lat", "lon"], data),
            "f_wdpth_ucat": (["time", "lat", "lon"], data),
            "volresv": (["time", "lat", "lon"], data),
            "qresv_in": (["time", "lat", "lon"], data),
            "qresv_out": (["time", "lat", "lon"], data),
        },
        coords={"time": times, "lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds.to_netcdf(history / "Global_Grid_50km_IGBP_VG_hist_unitcat_setgrid_2000-01.nc")

    result = scan_simulation_roots([root], model_name="auto")

    case = result.cases[0]
    assert case.model == "CoLM2024"
    assert case.provenance["model"] == "variables"
