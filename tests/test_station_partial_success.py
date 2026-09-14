"""Only known station data gaps may be downgraded to partial success."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core.evaluation import Evaluation_stn
from openbench.data.processing import StationDatasetProcessing


def _processor(tmp_path):
    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.casedir, proc.item, proc.ref_source, proc.sim_source = str(tmp_path), "Runoff", "Ref", "Sim"
    proc.ref_varname = proc.sim_varname = ["flow"]
    proc.ref_varunit = proc.sim_varunit = "m3 s-1"
    proc.compare_tim_res = "D"
    proc.num_cores = 1
    proc._try_compute_from_profile = lambda *args: None
    proc._is_climatology_mode = lambda: False
    proc._resample_to_compare_resolution = lambda ds, *args: ds
    proc.check_coordinate = lambda ds: ds
    proc.check_dataset_time_integrity = lambda ds, *args: ds
    proc.process_units = lambda ds, unit: (ds, unit)
    proc.select_timerange = lambda ds, *args: ds
    return proc


@pytest.mark.parametrize("gap", ["variable", "nan", "empty_time"])
def test_preprocessing_known_data_gap_removes_stale_output(tmp_path, gap):
    proc = _processor(tmp_path)
    raw = tmp_path / "raw.nc"
    xr.Dataset(
        {"other" if gap == "variable" else "flow": ("time", [np.nan if gap == "nan" else 1.0] * 2)},
        coords={"time": pd.date_range("1990-01-01" if gap == "empty_time" else "2000-01-01", periods=2)},
    ).to_netcdf(raw)
    stations = pd.DataFrame([{"ID": "A", "use_syear": 2000, "use_eyear": 2000, "ref_dir": str(raw)}])
    output = tmp_path / "data/stn_Ref_Sim/Runoff_ref_A_2000_2000.nc"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"stale")
    result = proc._make_stn_parallel(stations, "ref", 0)
    assert result["ok"] is False and result["station"] == "A" and result["error"]
    assert not output.exists()
    assert output.with_suffix(".skip.txt").read_text() == result["error"]


@pytest.mark.parametrize("stage", ["compute", "filter", "filter_import", "units", "write"])
def test_preprocessing_program_errors_propagate(tmp_path, monkeypatch, stage):
    import openbench.data.custom as custom

    proc = _processor(tmp_path)
    raw = tmp_path / "raw.nc"
    xr.Dataset(
        {"other" if stage in {"filter", "filter_import", "compute"} else "flow": ("time", [1.0, 2.0])},
        coords={"time": pd.date_range("2000-01-01", periods=2)},
    ).to_netcdf(raw)
    proc.station_list = pd.DataFrame([{"ID": "A", "use_syear": 2000, "use_eyear": 2000, "ref_dir": str(raw)}])

    def broken(*args, **kwargs):
        raise AttributeError("implementation bug")

    if stage == "filter":
        monkeypatch.setattr(custom, "load_filter", lambda name: SimpleNamespace(filter_Ref=broken))
    elif stage == "filter_import":
        monkeypatch.setenv("OPENBENCH_CUSTOM_DIR", str(tmp_path))
        (tmp_path / "Ref_filter.py").write_text("raise AttributeError('implementation bug')\n")
    else:
        method = {"compute": "_try_compute_from_profile", "units": "process_units", "write": "save_station_data"}[stage]
        setattr(proc, method, broken)
    with pytest.raises(AttributeError, match="implementation bug"):
        proc.process_station_data({"datasource": "ref"})


def _evaluator(tmp_path, monkeypatch):
    ev = Evaluation_stn.__new__(Evaluation_stn)
    ev.casedir, ev.item, ev.ref_source, ev.sim_source = str(tmp_path), "Runoff", "Ref", "Sim"
    ev.ref_varname = ev.sim_varname = ["flow"]
    ev.compare_tim_res, ev.num_cores, ev.output_manager = "Day", 1, None
    ev.metrics, ev.scores = ["bias"], []
    monkeypatch.setattr("openbench.core.evaluation.plot_stn", lambda *args: None)
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda *args: None)
    stations = pd.DataFrame(
        [
            {"ID": name, "sim_lon": i, "sim_lat": i, "use_syear": 2000, "use_eyear": 2000}
            for i, name in enumerate(["A", "B", "C"])
        ]
    )
    stations.to_csv(tmp_path / "stn_Ref_Sim_list.txt", index=False)
    directory = tmp_path / "data/stn_Ref_Sim"
    directory.mkdir(parents=True)
    for name in stations.ID:
        for source in ["sim", "ref"]:
            xr.Dataset(
                {"flow": ("time", [1.0, 2.0, 3.0])},
                coords={"time": pd.date_range("2000-01-01", periods=3)},
            ).to_netcdf(directory / f"Runoff_{source}_{name}_2000_2000.nc")
    return ev, directory


@pytest.mark.parametrize("all_failed", [False, True])
@pytest.mark.parametrize("samples", [1, 3])
def test_real_nan_stations_are_skipped_not_good_rows(tmp_path, monkeypatch, caplog, all_failed, samples):
    ev, directory = _evaluator(tmp_path, monkeypatch)
    for name in ["A", "B", "C"] if all_failed else ["A", "C"]:
        xr.Dataset(
            {"flow": ("time", [np.nan] * samples)},
            coords={"time": pd.date_range("2000-01-01", periods=samples)},
        ).to_netcdf(directory / f"Runoff_ref_{name}_2000_2000.nc")
    if all_failed:
        with pytest.raises(RuntimeError, match="no valid station"):
            ev.make_evaluation_P()
    else:
        ev.make_evaluation_P()
        frame = pd.read_csv(tmp_path / "metrics/Runoff_stn_Ref_Sim_evaluations.csv")
        status = pd.read_csv(directory / "Runoff_evaluation_status.csv").fillna({"reason": ""})
        assert status.ID.tolist() == ["A", "B", "C"]
        assert status.status.tolist() == ["unavailable", "ok", "unavailable"]
        assert status.reason.iloc[0] == status.reason.iloc[2] == "no shared finite sim/ref pairs"
        assert frame.ID.tolist() == ["B"]
        assert frame.sim_lon.tolist() == [1]
        assert frame.bias.tolist() == [0.0]
        assert ev.station_summary["succeeded"] == 1
        assert [s["station"] for s in ev.station_summary["skipped"]] == ["A", "C"]
        assert "partial success" in caplog.text


@pytest.mark.parametrize("stage", ["alignment", "metric", "config", "worker_result", "missing_output"])
def test_evaluation_program_errors_are_not_station_skips(tmp_path, monkeypatch, stage):
    ev, directory = _evaluator(tmp_path, monkeypatch)

    def broken(*args, **kwargs):
        raise AttributeError("implementation bug")

    if stage == "alignment":
        ev._align_station_times = broken
    elif stage == "metric":
        ev.KGESS = broken
    elif stage == "config":
        ev.metrics = ["unknown_metric"]
    elif stage == "missing_output":
        (directory / "Runoff_ref_A_2000_2000.nc").unlink()
    else:
        ev.make_evaluation_parallel = lambda *args: None
    error = {
        "alignment": AttributeError,
        "metric": AttributeError,
        "config": ValueError,
        "worker_result": RuntimeError,
        "missing_output": FileNotFoundError,
    }
    with pytest.raises(error[stage]):
        ev.make_evaluation_P()


@pytest.mark.parametrize("gap", ["missing_variable", "no_overlap"])
def test_evaluation_reports_data_gap_reason(tmp_path, monkeypatch, gap):
    ev, directory = _evaluator(tmp_path, monkeypatch)
    path = directory / "Runoff_ref_A_2000_2000.nc"
    if gap == "missing_variable":
        proc = _processor(tmp_path)
        raw = tmp_path / "raw_missing.nc"
        xr.Dataset({"other": ("time", [1.0])}, coords={"time": [np.datetime64("2000-01-01")]}).to_netcdf(raw)
        stations = pd.DataFrame([{"ID": "A", "use_syear": 2000, "use_eyear": 2000, "ref_dir": str(raw)}])
        assert proc._make_stn_parallel(stations, "ref", 0)["ok"] is False
    else:
        xr.Dataset(
            {"flow": ("time", [1.0, 2.0, 3.0])},
            coords={"time": pd.date_range("2001-01-01", periods=3)},
        ).to_netcdf(path)
    ev.make_evaluation_P()
    skipped = ev.station_summary["skipped"]
    assert len(skipped) == 1 and skipped[0]["station"] == "A"
    assert ("Variable 'flow' not found" if gap == "missing_variable" else "no overlapping") in skipped[0]["reason"]


def test_grid_station_empty_time_skip_cannot_reuse_stale_data(tmp_path):
    proc = _processor(tmp_path)
    stations = pd.DataFrame([{"ID": "A", "use_syear": 2000, "use_eyear": 2000}])
    output = tmp_path / "data/stn_Ref_Sim/Runoff_ref_A_2000_2000.nc"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"stale")
    proc.extract_single_station_data = lambda ds, *args: ds
    proc.process_extracted_data = lambda ds, *args: None
    proc._extract_stn_parallel("ref", xr.Dataset(), stations, 0)
    assert not output.exists()
    assert "No data in time range" in output.with_suffix(".skip.txt").read_text()
    data = xr.DataArray([1.0, 2.0], dims="time", name="flow")
    proc.save_extracted_data(data, stations.iloc[0], "ref")
    assert output.exists() and not output.with_suffix(".skip.txt").exists()


@pytest.mark.parametrize("broken", [False, True])
def test_versioned_station_filter_keeps_fallback_but_not_import_errors(tmp_path, monkeypatch, broken):
    from openbench.data.custom import load_filter

    monkeypatch.setenv("OPENBENCH_CUSTOM_DIR", str(tmp_path))
    (tmp_path / "Ref_filter.py").write_text(
        "import missing_filter_dependency\n" if broken else "def filter_Ref(*args): return args\n"
    )
    if broken:
        with pytest.raises(ModuleNotFoundError, match="missing_filter_dependency"):
            load_filter("Ref2.0")
    else:
        assert callable(load_filter("Ref2.0").filter_Ref)


@pytest.mark.parametrize("partial", [False, True])
def test_runner_propagates_station_summary_and_only_caches_complete_data(tmp_path, monkeypatch, partial):
    from openbench.runner.cache import EvaluationCache
    from openbench.runner.task_execution import evaluate_single

    summary = {
        "total": 2,
        "succeeded": 1 if partial else 2,
        "skipped": [{"station": "A", "reason": "no shared finite sim/ref pairs"}] if partial else [],
    }
    evaluator = SimpleNamespace(station_summary=summary, make_evaluation_P=lambda: None)
    monkeypatch.setattr("openbench.core.evaluation.Evaluation_stn", lambda *args: evaluator)
    cache = EvaluationCache(tmp_path)
    cache.mark_done("key", "hash")
    bindings = SimpleNamespace(build_evaluation_fig_nml=lambda: SimpleNamespace(to_fig_nml=lambda: {}))
    result = evaluate_single(
        {
            "var_name": "Runoff",
            "sim_source": "Sim",
            "ref_source": "Ref",
            "cache_key": "key",
            "config_hash": "hash",
            "use_cache": False,
            "update_cache": True,
            "cache_dir": tmp_path,
            "bindings": bindings,
            "ref_preprocessed": True,
        },
        build_bridge_runtime_info_fn=lambda task: {"ref_data_type": "stn", "casedir": str(tmp_path)},
        bindings_only_drawing_fn=lambda bindings: False,
        task_output_requirement_fn=lambda *args: [],
        missing_expected_outputs_fn=lambda *args: [],
    )
    assert result["status"] == "success" and result["station_summary"] == summary
    assert EvaluationCache(tmp_path).is_cached("key", "hash") is not partial
