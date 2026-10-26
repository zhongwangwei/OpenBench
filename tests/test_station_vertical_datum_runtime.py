"""One WSE correction path for metrics, station figures and only-drawing refreshes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core.evaluation import Evaluation_stn
from openbench.data.processing import StationDatasetProcessing
from openbench.data.vertical_datum import VerticalDatumError


def _pgm(path, model, value):
    path.write_bytes(
        f"P5\n# Description {model}\n# Offset 0\n# Scale 1\n4 3\n65535\n".encode()
        + np.full((3, 4), value, dtype=">u2").tobytes()
    )
    return str(path)


def _evaluation(tmp_path, monkeypatch):
    ev = Evaluation_stn.__new__(Evaluation_stn)
    ev.casedir, ev.item, ev.ref_source, ev.sim_source = str(tmp_path), "Water_Surface_Elevation", "Ref", "Sim"
    ev.ref_varname = ev.sim_varname = ["height"]
    ev.metrics, ev.scores, ev.compare_tim_res, ev.num_cores = ["bias"], [], "Day", 1
    ev.output_manager, ev.fig_nml = None, {}
    stations = pd.DataFrame([dict(ID="0001", use_syear=2000, use_eyear=2000, ref_lat=0.0, ref_lon=0.0)])
    directory = tmp_path / "data/stn_Ref_Sim"
    directory.mkdir(parents=True)
    for source in ("sim", "ref"):
        xr.DataArray(
            [1.0, 2.0, 3.0], dims="time", coords={"time": pd.date_range("2000-01-01", periods=3)}, attrs={"units": "m"}
        ).to_dataset(name="height").to_netcdf(directory / f"{ev.item}_{source}_0001_2000_2000.nc")
    ev.ref_fulllist = str(tmp_path / "stations.csv")
    stations.to_csv(ev.ref_fulllist, index=False)
    monkeypatch.setattr("openbench.core.evaluation.plot_stn", lambda *args: None)
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda *args: None)
    return ev, stations, directory


def _enable(ev, tmp_path):
    ev.ref_vertical_datum = {
        "source_geoid": _pgm(tmp_path / "08.pgm", "EGM2008", 10),
        "target_geoid": _pgm(tmp_path / "96.pgm", "EGM96", 6),
    }


@pytest.mark.parametrize("as_dict", [False, True])
def test_correction_reaches_metrics_and_plot_once_without_writes(tmp_path, monkeypatch, as_dict):
    ev, stations, directory = _evaluation(tmp_path, monkeypatch)
    if as_dict:
        stations = stations.to_dict(orient="list")
    before = {p: p.read_bytes() for p in directory.glob("*.nc")}
    assert ev.make_evaluation_parallel(stations, 0)["bias"] == pytest.approx(0.0)
    _enable(ev, tmp_path)
    captured = []
    monkeypatch.setattr("openbench.core.evaluation.plot_stn", lambda *args: captured.append(args))
    for _ in range(2):
        assert ev.make_evaluation_parallel(stations, 0)["bias"] == pytest.approx(-4.0)
        np.testing.assert_allclose(captured[-1][2], [5.0, 6.0, 7.0])
    assert all(p.read_bytes() == content for p, content in before.items())
    ev.ref_vertical_datum["source_geoid"] = str(tmp_path / "missing.pgm")
    with pytest.raises(VerticalDatumError, match="Cannot read geoid"):
        ev.make_evaluation_parallel(stations, 0)


def test_only_drawing_refreshes_stale_metrics_using_same_correction(tmp_path, monkeypatch):
    from openbench.visualization.only_drawing import Evaluation_stn_only_drawing

    ev, stations, directory = _evaluation(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in directory.glob("*.nc")}
    ev.make_evaluation_P()
    path = tmp_path / "metrics" / f"{ev.item}_stn_Ref_Sim_evaluations.csv"
    assert pd.read_csv(path).bias.iloc[0] == 0.0
    _enable(ev, tmp_path)
    captured = []
    monkeypatch.setattr("openbench.core.evaluation.plot_stn", lambda *args: captured.append(args))
    monkeypatch.setattr("openbench.core.evaluation.create_output_manager", lambda *args: None)
    draw = Evaluation_stn_only_drawing(ev.__dict__, {})
    for _ in range(2):
        draw.make_evaluation_P()
        assert pd.read_csv(path).bias.iloc[0] == pytest.approx(-4.0)
        np.testing.assert_allclose(captured[-1][2], [5.0, 6.0, 7.0])
    assert all(p.read_bytes() == content for p, content in before.items())
    assert draw.station_summary["succeeded"] == 1


def test_default_only_drawing_does_not_evaluate(monkeypatch):
    from openbench.visualization.only_drawing import Evaluation_stn_only_drawing

    draw = Evaluation_stn_only_drawing({"sim_varname": "s", "ref_varname": "o"}, {})
    plotted = []
    monkeypatch.setattr("openbench.visualization.only_drawing.make_plot_index_stn", lambda obj: plotted.append(obj))
    monkeypatch.setattr(
        "openbench.core.evaluation.Evaluation_stn", lambda *args: pytest.fail("legacy drawing must remain read-only")
    )
    draw.make_evaluation_P()
    assert plotted == [draw]


def _processor():
    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.item, proc.ref_source, proc.sim_source = "Water_Surface_Elevation", "Ref", "Model"
    proc.ref_varname = proc.sim_varname = ["height"]
    proc.ref_varunit = proc.sim_varunit = "m"
    proc._try_compute_from_profile = lambda *args, **kwargs: None
    proc._is_climatology_mode = lambda: False
    proc._resample_to_compare_resolution = lambda ds, *args: ds
    proc.check_coordinate = lambda ds: ds
    proc.check_dataset_time_integrity = lambda ds, *args: ds
    proc.select_timerange = lambda ds, *args: ds
    proc.compare_tim_res = "D"
    proc.ref_vertical_datum = {"source_geoid": "08.pgm", "target_geoid": "96.pgm"}
    return proc


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"units": "cm"},
        {"units": "mm"},
        {"units": "m2"},
        {"units": "m", "vertical_datum": "EGM96"},
        {"units": "m", "openbench_vertical_datum_offset_m": 4.0},
    ],
)
def test_reference_preprocessing_rejects_wrong_units_or_already_corrected(attrs):
    raw = xr.Dataset({"height": ("time", [1.0, 2.0], attrs)}, coords={"time": pd.date_range("2000-01-01", periods=2)})
    with pytest.raises(VerticalDatumError):
        _processor().process_single_station_data(raw, 2000, 2000, "ref")


def test_model_requires_no_quantity_datum_or_native_provenance():
    proc = _processor()
    raw = xr.Dataset(
        {"height": ("time", [1.0, 2.0], {"units": "m"})}, coords={"time": pd.date_range("2000-01-01", periods=2)}
    )
    actual = proc.process_single_station_data(raw, 2000, 2000, "sim")
    np.testing.assert_allclose(actual, [1.0, 2.0])


@pytest.mark.parametrize(
    "patch,ok",
    [
        ({}, True),
        ({"item": "River_Water_Level"}, False),
        ({"item": "Dam_Elevation"}, False),
        ({"ref_data_type": "grid"}, False),
        ({"ref_varunit": "mm"}, False),
        ({"sim_vertical_datum": {"source": "EGM96"}}, False),
    ],
)
def test_runner_rejects_wrong_scope_before_preprocessing(monkeypatch, patch, ok):
    from openbench.runner.task_execution import evaluate_single

    info = dict(
        item="Water_Surface_Elevation",
        ref_data_type="stn",
        sim_data_type="grid",
        ref_varunit="m",
        sim_varunit="m",
        ref_vertical_datum={"source_geoid": "08.pgm", "target_geoid": "96.pgm"},
        **{},
    )
    info.update(patch)
    prepared = []
    monkeypatch.setattr(
        "openbench.data.processing.DatasetProcessing",
        lambda info: SimpleNamespace(prepare_source=lambda side: prepared.append(side)),
    )
    monkeypatch.setattr(
        "openbench.core.evaluation.Evaluation_stn", lambda *args: SimpleNamespace(make_evaluation_P=lambda: None)
    )
    bindings = SimpleNamespace(build_evaluation_fig_nml=lambda: SimpleNamespace(to_fig_nml=lambda: {}))
    task = dict(
        var_name=info["item"],
        sim_source="Model",
        ref_source="Ref",
        cache_key="x",
        config_hash="x",
        use_cache=False,
        bindings=bindings,
    )
    actual = evaluate_single(
        task,
        build_bridge_runtime_info_fn=lambda task: info,
        bindings_only_drawing_fn=lambda _: False,
        task_output_requirement_fn=lambda *args: [],
        missing_expected_outputs_fn=lambda *args: [],
    )
    assert actual["status"] == ("success" if ok else "error")
    assert prepared == (["ref", "sim"] if ok else [])


def test_runner_only_drawing_uses_cached_station_series_without_preprocessing(tmp_path, monkeypatch):
    from openbench.runner.task_execution import evaluate_single

    ev, _, directory = _evaluation(tmp_path, monkeypatch)
    _enable(ev, tmp_path)
    info = {**ev.__dict__, "ref_data_type": "stn", "sim_data_type": "grid", "ref_varunit": "m", "sim_varunit": "m"}
    before = {p: p.read_bytes() for p in directory.glob("*.nc")}
    monkeypatch.setattr(
        "openbench.data.processing.DatasetProcessing",
        lambda *args: pytest.fail("only_drawing must not preprocess inputs"),
    )
    monkeypatch.setattr("openbench.core.evaluation.create_output_manager", lambda *args: None)
    bindings = SimpleNamespace(build_evaluation_fig_nml=lambda: SimpleNamespace(to_fig_nml=lambda: {}))
    task = dict(
        var_name=ev.item,
        sim_source="Sim",
        ref_source="Ref",
        cache_key="x",
        config_hash="x",
        use_cache=False,
        bindings=bindings,
    )
    result = evaluate_single(
        task,
        build_bridge_runtime_info_fn=lambda task: info,
        bindings_only_drawing_fn=lambda _: True,
        task_output_requirement_fn=lambda *args: [],
        missing_expected_outputs_fn=lambda *args: [],
    )
    assert result["status"] == "success", result
    assert pd.read_csv(tmp_path / "metrics" / f"{ev.item}_stn_Ref_Sim_evaluations.csv").bias.iloc[0] == pytest.approx(
        -4.0
    )
    assert all(p.read_bytes() == content for p, content in before.items())
