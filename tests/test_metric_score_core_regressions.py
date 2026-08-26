from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.config.adapter import build_runner_config, to_legacy_config
from openbench.config.schema import EvaluationConfig, OpenBenchConfig, ProjectConfig, ReferenceConfig, SimulationEntry
from openbench.core.evaluation import (
    Evaluation_grid,
    Evaluation_stn,
    _apply_pairwise_valid_mask,
    _has_any_valid_pair,
    _mfm_shared_metric_names,
)
from openbench.core.metrics import metrics
from openbench.core.registry import IMPLEMENTED_METRICS, IMPLEMENTED_SCORES
from openbench.core.scores import scores
from openbench.data.climatology import process_climatology_evaluation
from openbench.gui.config_manager import ConfigManager
from openbench.runner.preflight import missing_expected_outputs, output_file_is_readable


def _cfg(metrics_value=None, scores_value=None):
    kwargs = {}
    if metrics_value is not None:
        kwargs["metrics"] = metrics_value
    if scores_value is not None:
        kwargs["scores"] = scores_value
    return OpenBenchConfig(
        project=ProjectConfig(name="case", output_dir="./out", years=[2001, 2002]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "Ref"}),
        simulation={"Sim": SimulationEntry(model="Sim", root_dir="/sim")},
        **kwargs,
    )


def test_config_preserves_explicit_empty_metrics_and_scores(tmp_path):
    cfg = _cfg(metrics_value=[], scores_value=[])

    assert build_runner_config(cfg).metrics == []
    assert build_runner_config(cfg).scores == []
    legacy = to_legacy_config(cfg)
    assert legacy["metrics"] == {}
    assert legacy["scores"] == {}

    gui_cfg = {
        "general": {"basename": "case", "basedir": str(tmp_path), "syear": 2001, "eyear": 2002},
        "evaluation_items": {"Runoff": True},
        "ref_data": {"general": {"Runoff_ref_source": "Ref"}},
        "sim_data": {
            "general": {"Runoff_sim_source": ["Sim"]},
            "source_configs": {"Sim": {"general": {"root_dir": "/sim"}}},
        },
        "metrics": {},
        "scores": {},
    }
    exported = __import__("yaml").safe_load(ConfigManager().generate_config_yaml(gui_cfg))
    assert exported["metrics"] == []
    assert exported["scores"] == []


def test_config_omitted_metrics_and_scores_keep_defaults():
    cfg = _cfg()

    assert build_runner_config(cfg).metrics == ["bias", "RMSE", "correlation"]
    assert build_runner_config(cfg).scores == ["Overall_Score"]


def test_pairwise_valid_mask_rejects_inf_pairs():
    sim = xr.DataArray([1.0, np.inf], dims=["time"])
    obs = xr.DataArray([1.0, 2.0], dims=["time"])

    masked_sim, masked_obs = _apply_pairwise_valid_mask(sim, obs)

    assert bool(_has_any_valid_pair(sim, obs)) is True
    assert np.isnan(masked_sim.values[1])
    assert np.isnan(masked_obs.values[1])
    assert not _has_any_valid_pair(xr.DataArray([np.inf], dims=["time"]), xr.DataArray([1.0], dims=["time"]))


def test_nspatial_score_preserves_singleton_lat_lon():
    arr = xr.DataArray(
        np.arange(2.0).reshape(2, 1, 1),
        coords={"time": [1, 2], "lat": [30.0], "lon": [120.0]},
        dims=["time", "lat", "lon"],
    )

    out = scores().nSpatialScore(arr, arr)

    assert out.dims == ("lat", "lon")
    assert out.sizes["lat"] == 1
    assert out.sizes["lon"] == 1


def test_process_score_preserves_singleton_lat_lon_and_reorders_dims(tmp_path, monkeypatch):
    monkeypatch.setattr("openbench.core.evaluation.gc.collect", lambda: None)
    ev = Evaluation_grid.__new__(Evaluation_grid)
    ev.casedir = str(tmp_path)
    ev.item = "Runoff"
    ev.ref_source = "Ref"
    ev.sim_source = "Sim"
    ev.output_manager = None
    ev.CustomScore = lambda _s, _o: xr.DataArray(
        [[0.7]],
        coords={"lon": [120.0], "lat": [30.0]},
        dims=["lon", "lat"],
    )
    template = xr.DataArray(
        np.ones((2, 1, 1)),
        coords={"time": [1, 2], "lat": [30.0], "lon": [120.0]},
        dims=["time", "lat", "lon"],
    )

    ev.process_score("CustomScore", template, template)

    with xr.open_dataset(tmp_path / "scores" / "Runoff_ref_Ref_sim_Sim_CustomScore.nc") as ds:
        assert ds["CustomScore"].dims == ("lat", "lon")
        assert ds.sizes["lat"] == 1
        assert ds.sizes["lon"] == 1
        assert float(ds["CustomScore"].isel(lat=0, lon=0)) == pytest.approx(0.7)


def test_parallel_metric_uses_threading_backend_and_propagates_errors(monkeypatch):
    calls = []

    def fake_parallel_map(func, items, **kwargs):
        calls.append(kwargs)
        return [func(item) for item in items]

    monkeypatch.setattr("openbench.core.evaluation._HAS_PARALLEL_ENGINE", True)
    monkeypatch.setattr("openbench.core.evaluation.parallel_map", fake_parallel_map)
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_grid", lambda _self: None)
    monkeypatch.setattr(
        "openbench.core.evaluation.open_dataset_chunked",
        lambda path: xr.Dataset(
            {"v": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
            coords={"time": [1, 2], "lat": [0.0], "lon": [0.0]},
        ),
    )
    monkeypatch.setattr("openbench.util.names.select_data_array", lambda ds, *_args: ds["v"])
    monkeypatch.setattr(
        Evaluation_grid,
        "bad",
        lambda self, *_args: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )

    ev = Evaluation_grid.__new__(Evaluation_grid)
    ev.casedir = "/tmp"
    ev.item = "Runoff"
    ev.ref_source = "Ref"
    ev.sim_source = "Sim"
    ev.ref_varname = "v"
    ev.sim_varname = "v"
    ev.metrics = ["bad", "bias"]
    ev.scores = []
    ev.num_cores = 2
    ev.output_manager = None
    ev.compare_tim_res = "Month"

    with pytest.raises(RuntimeError, match="boom"):
        ev.make_Evaluation()
    assert calls and calls[0]["backend"] == "threading"


@pytest.mark.parametrize(
    "name",
    [
        "nBiasScore",
        "nRMSEScore",
        "nSeasonalityScore",
        "Overall_Score",
        "L",
        "ubRMSE",
        "ubNSE",
        "pc_ampli",
        "dr",
        "MFM_omega",
        "MFM_varphi",
        "MFM_eta",
        "MFM",
        "index_agreement",
    ],
)
def test_annual_climatology_rejects_single_time_unsafe_outputs(name):
    assert name in IMPLEMENTED_METRICS or name in IMPLEMENTED_SCORES
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=3), "lat": [0.0], "lon": [0.0]},
    )

    _ref, _sim, supported = process_climatology_evaluation(
        ds, ds, ["bias", name], compare_tim_res="climatology-year", syear=2001
    )

    assert supported == ["bias"]


def test_annual_climatology_keeps_single_time_relative_extrema():
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=3), "lat": [0.0], "lon": [0.0]},
    )

    _ref, _sim, supported = process_climatology_evaluation(
        ds, ds, ["pc_max", "pc_min"], compare_tim_res="climatology-year", syear=2001
    )

    assert supported == ["pc_max", "pc_min"]


def test_annual_climatology_rejects_overall_score():
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=3), "lat": [0.0], "lon": [0.0]},
    )

    ref, sim, supported = process_climatology_evaluation(
        ds, ds, ["bias", "Overall_Score"], compare_tim_res="climatology-year", syear=2001
    )

    assert ref is not None and sim is not None
    assert supported == ["bias"]


def test_r2_metrics_are_clipped(monkeypatch):
    monkeypatch.setattr("xarray.corr", lambda *_args, **_kwargs: xr.DataArray([1.0000001, -1.0000001], dims=["x"]))
    m = metrics()
    arr = xr.DataArray(np.ones((2, 2)), dims=["time", "x"])

    out = m.correlation_R2(arr, arr)

    assert np.all((out.values >= 0) & (out.values <= 1))


def test_station_requested_columns_each_need_finite_values(tmp_path, monkeypatch):
    stnlist = tmp_path / "stn_Ref_Sim_list.txt"
    pd.DataFrame({"ID": ["a", "b"]}).to_csv(stnlist, index=False)
    ev = Evaluation_stn.__new__(Evaluation_stn)
    ev.casedir = str(tmp_path)
    ev.item = "Runoff"
    ev.ref_source = "Ref"
    ev.sim_source = "Sim"
    ev.metrics = ["bias", "RMSE"]
    ev.scores = []
    ev.output_manager = None
    ev.num_cores = 1
    ev.make_evaluation_parallel = lambda _list, i: {"bias": 1.0, "RMSE": np.nan}
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda _self: None)

    with pytest.raises(RuntimeError, match="RMSE"):
        ev.make_evaluation_P()


def test_preflight_netcdf_requires_requested_variable_and_finite_data(tmp_path):
    good = tmp_path / "good.nc"
    wrong = tmp_path / "wrong.nc"
    legacy_value = tmp_path / "legacy_value.nc"
    bad = tmp_path / "bad.nc"
    all_bad = tmp_path / "all_bad.nc"
    xr.Dataset({"bias": (("lat", "lon"), [[1.0]])}).to_netcdf(good)
    xr.Dataset({"RMSE": (("lat", "lon"), [[1.0]])}).to_netcdf(wrong)
    xr.Dataset({"value": (("lat", "lon"), [[1.0]])}).to_netcdf(legacy_value)
    xr.Dataset({"bias": (("lat", "lon"), [[np.inf]])}).to_netcdf(bad)
    xr.Dataset({"bias": ("x", [np.inf]), "RMSE": ("x", [np.nan])}).to_netcdf(all_bad)

    assert output_file_is_readable(good, "bias")
    assert not output_file_is_readable(wrong, "bias")
    assert not output_file_is_readable(legacy_value, "bias")
    assert not output_file_is_readable(bad, "bias")
    assert not output_file_is_readable(all_bad)


def test_preflight_station_columns_require_finite_requested_values(tmp_path):
    out = tmp_path / "out"
    path = out / "metrics" / "Runoff_stn_Ref_Sim_evaluations.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"bias": [1.0], "RMSE": [np.inf]}).to_csv(path, index=False)
    task = {
        "var_name": "Runoff",
        "ref_source": "Ref",
        "sim_source": "Sim",
        "output_requirements": {"metrics": ["bias", "RMSE"], "scores": [], "ref_data_type": "stn"},
    }

    missing = missing_expected_outputs(out, task)

    assert path in missing


def test_mfm_shared_detection_uses_unique_metric_names():
    assert _mfm_shared_metric_names(["MFM", "MFM"]) == set()
    assert _mfm_shared_metric_names(["MFM", "MFM_eta", "MFM_eta"]) == {"MFM", "MFM_eta"}


def test_grid_evaluation_reuses_shared_mfm_components(monkeypatch):
    ev = Evaluation_grid.__new__(Evaluation_grid)
    ev.metrics = ["MFM", "MFM_eta", "MFM_omega"]
    ev.item = "Runoff"
    ev.ref_source = "Ref"
    ev.sim_source = "Sim"
    ev.output_manager = None
    counts = {"omega": 0, "varphi": 0, "eta": 0, "mfm": 0}
    saved = []
    arr = xr.DataArray([[1.0]], coords={"lat": [0.0], "lon": [0.0]}, dims=("lat", "lon"))

    def component(name, value):
        def _inner(_s, _o):
            counts[name] += 1
            return arr * value

        return _inner

    ev.MFM_omega = component("omega", 0.8)
    ev.MFM_varphi = component("varphi", 0.7)
    ev.MFM_eta = component("eta", 0.6)
    ev.MFM = component("mfm", 0.1)

    def save_metric(name, da, vkey=""):
        saved.append((name, float(da.values.squeeze())))

    monkeypatch.setattr(ev, "_save_metric_array", save_metric)
    template = xr.DataArray(
        np.ones((4, 1, 1)),
        coords={"time": pd.date_range("2001-01-01", periods=4), "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )

    ev._process_metrics_in_order(template, template)

    assert counts == {"omega": 1, "varphi": 1, "eta": 1, "mfm": 0}
    assert [name for name, _ in saved] == ev.metrics
    expected_mfm = 1 - np.sqrt(((1 - 0.8) ** 2 + (1 - 0.7) ** 2 + (1 - 0.6) ** 2) / 3)
    assert saved[0][1] == pytest.approx(expected_mfm)


def test_station_evaluation_reuses_shared_mfm_components(tmp_path, monkeypatch):
    monkeypatch.setattr("openbench.core.evaluation.plot_stn", lambda *args, **kwargs: None)
    ev = Evaluation_stn.__new__(Evaluation_stn)
    ev.casedir = str(tmp_path)
    ev.item = "Runoff"
    ev.ref_source = "Ref"
    ev.sim_source = "Sim"
    ev.ref_varname = ["v"]
    ev.sim_varname = ["v"]
    ev.metrics = ["MFM", "MFM_eta"]
    ev.scores = []
    ev.compare_tim_res = ""
    counts = {"omega": 0, "varphi": 0, "eta": 0, "mfm": 0}

    def component(name, value):
        def _inner(_s, _o):
            counts[name] += 1
            return xr.DataArray(value)

        return _inner

    ev.MFM_omega = component("omega", 0.8)
    ev.MFM_varphi = component("varphi", 0.7)
    ev.MFM_eta = component("eta", 0.6)
    ev.MFM = component("mfm", 0.1)
    data_dir = tmp_path / "data" / "stn_Ref_Sim"
    data_dir.mkdir(parents=True)
    time = pd.date_range("2001-01-01", periods=4)
    ds = xr.Dataset({"v": ("time", np.arange(4.0) + 1)}, coords={"time": time})
    ds.to_netcdf(data_dir / "Runoff_sim_S1_2001_2001.nc")
    ds.to_netcdf(data_dir / "Runoff_ref_S1_2001_2001.nc")
    station_list = {"ID": ["S1"], "use_syear": [2001], "use_eyear": [2001], "ref_lat": [0.0], "ref_lon": [0.0]}

    row = ev.make_evaluation_parallel(station_list, 0)

    assert counts == {"omega": 1, "varphi": 1, "eta": 1, "mfm": 0}
    assert float(row["MFM_eta"]) == pytest.approx(0.6)
    expected_mfm = 1 - np.sqrt(((1 - 0.8) ** 2 + (1 - 0.7) ** 2 + (1 - 0.6) ** 2) / 3)
    assert float(row["MFM"]) == pytest.approx(expected_mfm)


def test_parallel_metric_keeps_parallel_map_with_shared_mfm(monkeypatch):
    calls = []
    counts = {"omega": 0, "varphi": 0, "eta": 0, "mfm": 0, "other": 0}

    def fake_parallel_map(func, items, **kwargs):
        calls.append((list(items), kwargs))
        return [func(item) for item in items]

    monkeypatch.setattr("openbench.core.evaluation._HAS_PARALLEL_ENGINE", True)
    monkeypatch.setattr("openbench.core.evaluation.parallel_map", fake_parallel_map)
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_grid", lambda _self: None)
    monkeypatch.setattr(
        "openbench.core.evaluation.open_dataset_chunked",
        lambda path: xr.Dataset(
            {"v": (("time", "lat", "lon"), np.ones((4, 1, 1)))},
            coords={"time": pd.date_range("2001-01-01", periods=4), "lat": [0.0], "lon": [0.0]},
        ),
    )
    monkeypatch.setattr("openbench.util.names.select_data_array", lambda ds, *_args: ds["v"])

    def component(name, value):
        def _inner(self, _s, _o):
            counts[name] += 1
            return xr.DataArray([[value]], coords={"lat": [0.0], "lon": [0.0]}, dims=("lat", "lon"))

        return _inner

    monkeypatch.setattr(Evaluation_grid, "MFM_omega", component("omega", 0.8))
    monkeypatch.setattr(Evaluation_grid, "MFM_varphi", component("varphi", 0.7))
    monkeypatch.setattr(Evaluation_grid, "MFM_eta", component("eta", 0.6))
    monkeypatch.setattr(Evaluation_grid, "MFM", component("mfm", 0.1))
    monkeypatch.setattr(Evaluation_grid, "other_metric", component("other", 0.2), raising=False)
    monkeypatch.setattr(Evaluation_grid, "_save_metric_array", lambda *args, **kwargs: None)

    ev = Evaluation_grid.__new__(Evaluation_grid)
    ev.casedir = "/tmp"
    ev.item = "Runoff"
    ev.ref_source = "Ref"
    ev.sim_source = "Sim"
    ev.ref_varname = "v"
    ev.sim_varname = "v"
    ev.metrics = ["MFM", "MFM_eta", "other_metric"]
    ev.scores = []
    ev.num_cores = 2
    ev.output_manager = None
    ev.compare_tim_res = "Month"

    ev.make_Evaluation()

    assert calls and calls[0][0] == ev.metrics
    assert calls[0][1]["backend"] == "threading"
    assert counts == {"omega": 1, "varphi": 1, "eta": 1, "mfm": 0, "other": 1}
