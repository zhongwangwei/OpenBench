import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.config.schema import (
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
    UncertaintyConfig,
)
from openbench.runner.uncertainty import (
    _cached_uncertainty_outputs_complete,
    _load_station_pairs,
    _write_grid_spread,
    _write_station_spread,
    run_uncertainty,
)


def _cfg(tmp_path: Path, simulations: list[str], references: list[str]) -> OpenBenchConfig:
    return OpenBenchConfig(
        project=ProjectConfig(name="case", output_dir=str(tmp_path), years=[2000, 2001], generate_report=False),
        evaluation=EvaluationConfig(variables=["Flow"]),
        reference=ReferenceConfig(sources={"Flow": references}),
        simulation={name: SimulationEntry(model=name, root_dir=".") for name in simulations},
        metrics=["RMSE"],
        scores=[],
        uncertainty=UncertaintyConfig(
            enabled=True,
            metrics=["RMSE"],
            n_resamples=20,
            confidence_level=0.9,
            block_length=2,
            seed=4,
        ),
    )


def _bindings(simulations: list[str], references: list[str], data_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        runner_cfg=SimpleNamespace(general={"compare_tim_res": "Month"}),
        namelists=SimpleNamespace(
            reference={
                "Flow": {
                    **{f"{name}_varname": "flow" for name in references},
                    **{f"{name}_data_type": data_type for name in references},
                }
            },
            simulation={
                "Flow": {
                    **{f"{name}_varname": "flow" for name in simulations},
                    **{f"{name}_data_type": data_type for name in simulations},
                }
            },
        ),
    )


def _tasks(bindings, simulations: list[str], references: list[str]):
    return [
        {"var_name": "Flow", "sim_source": sim, "ref_source": ref, "bindings": bindings}
        for ref in references
        for sim in simulations
    ]


def test_grid_uncertainty_outputs_keep_model_and_reference_axes_separate(tmp_path):
    output = tmp_path / "case"
    for folder in ("data", "metrics"):
        (output / folder).mkdir(parents=True)
    simulations = ["A", "B"]
    references = ["R1", "R2"]
    bindings = _bindings(simulations, references, "grid")
    time = np.concatenate([np.arange(10), np.arange(20, 30)])
    coords = {"time": time, "lat": [0.0, 10.0], "lon": [100.0, 110.0]}
    shape = (20, 2, 2)
    for name, value in {"A": 0.1, "B": 2.0}.items():
        xr.DataArray(np.full(shape, value), coords=coords, dims=("time", "lat", "lon"), name="flow").to_netcdf(
            output / "data" / f"Flow_sim_{name}_flow.nc"
        )
    for name, value in {"R1": 0.0, "R2": 3.0}.items():
        xr.DataArray(np.full(shape, value), coords=coords, dims=("time", "lat", "lon"), name="flow").to_netcdf(
            output / "data" / f"Flow_ref_{name}_flow.nc"
        )
    for ref, sim, value in [
        ("R1", "A", 0.1),
        ("R1", "B", 2.0),
        ("R2", "A", 2.9),
        ("R2", "B", 1.0),
    ]:
        xr.DataArray(np.full((2, 2), value), dims=("lat", "lon"), name="RMSE").to_netcdf(
            output / "metrics" / f"Flow_ref_{ref}_sim_{sim}_RMSE.nc"
        )

    errors = run_uncertainty(
        _cfg(tmp_path, simulations, references),
        _tasks(bindings, simulations, references),
        output,
        ["RMSE"],
        make_phase_error_fn=lambda phase, message, **details: {"phase": phase, "message": message, **details},
    )

    assert errors == []
    summary = json.loads((output / "uncertainty" / "summary.json").read_text())
    assert len(summary["bootstrap"]) == 4
    assert all(row["segment_count"] == 2 for row in summary["bootstrap"])
    estimates = {(row["reference"], row["simulation"]): row["estimate"] for row in summary["bootstrap"]}
    assert estimates == pytest.approx({("R1", "A"): 0.1, ("R1", "B"): 2.0, ("R2", "A"): 2.9, ("R2", "B"): 1.0})
    assert summary["verdicts"][0]["status"] == "reference_sensitive"
    assert len(summary["products"]["model_spread"]) == 2
    assert len(summary["products"]["reference_sensitivity"]) == 2
    with xr.open_dataset(output / summary["products"]["model_spread"][0]) as product:
        assert {"ensemble_mean", "model_spread", "member_count", "coefficient_of_variation"} <= set(product.data_vars)


def test_station_uncertainty_writes_network_bootstrap_and_csv_products(tmp_path):
    output = tmp_path / "case"
    (output / "metrics").mkdir(parents=True)
    simulations = ["A", "B"]
    bindings = _bindings(simulations, ["R"], "stn")
    sim_time = pd.date_range("2000-01-31", periods=20, freq="ME")
    ref_time = pd.date_range("2000-01-01", periods=20, freq="MS") + pd.Timedelta(days=14)
    for simulation, offsets in {"A": [0.5, 1.0], "B": [1.5, 2.0]}.items():
        folder = output / "data" / f"stn_R_{simulation}"
        folder.mkdir(parents=True)
        for station, offset in zip(["one", "two"], offsets):
            xr.DataArray(
                np.arange(20) + offset,
                coords={"time": sim_time},
                dims="time",
                name="flow",
            ).to_netcdf(folder / f"Flow_sim_{station}_2000_2001.nc")
            xr.DataArray(np.arange(20), coords={"time": ref_time}, dims="time", name="flow").to_netcdf(
                folder / f"Flow_ref_{station}_2000_2001.nc"
            )
        pd.DataFrame({"ID": ["one", "two"], "RMSE": offsets}).to_csv(
            output / "metrics" / f"Flow_stn_R_{simulation}_evaluations.csv",
            index=False,
        )

    errors = run_uncertainty(
        _cfg(tmp_path, simulations, ["R"]),
        _tasks(bindings, simulations, ["R"]),
        output,
        ["RMSE"],
        make_phase_error_fn=lambda phase, message, **details: {"phase": phase, "message": message, **details},
    )

    assert errors == []
    summary = json.loads((output / "uncertainty" / "summary.json").read_text())
    assert summary["bootstrap"][0]["scope"] == "station_network"
    assert summary["bootstrap"][0]["station_count"] == 2
    product = pd.read_csv(output / summary["products"]["model_spread"][0])
    assert {"ID", "ensemble_mean", "model_spread", "member_count", "coefficient_of_variation"} <= set(product)


def test_grid_verdict_uses_only_common_spatial_support(tmp_path):
    output = tmp_path / "case"
    for folder in ("data", "metrics"):
        (output / folder).mkdir(parents=True)
    simulations = ["A", "B"]
    references = ["R"]
    bindings = _bindings(simulations, references, "grid")
    time = np.arange(20)
    coords = {"time": time, "lat": [0.0], "lon": [0.0, 1.0, 2.0]}
    ref = np.broadcast_to(np.array([0.0, 50.0, 100.0]), (20, 1, 3)).copy()
    sim_a = ref.copy()
    sim_a[:, :, 2] = np.nan
    sim_b = ref.copy()
    sim_b[:, :, 0] = np.nan
    for name, values in {"A": sim_a, "B": sim_b}.items():
        xr.DataArray(values, coords=coords, dims=("time", "lat", "lon"), name="flow").to_netcdf(
            output / "data" / f"Flow_sim_{name}_flow.nc"
        )
        xr.DataArray(np.zeros((1, 3)), dims=("lat", "lon"), name="RMSE").to_netcdf(
            output / "metrics" / f"Flow_ref_R_sim_{name}_RMSE.nc"
        )
    xr.DataArray(ref, coords=coords, dims=("time", "lat", "lon"), name="flow").to_netcdf(
        output / "data" / "Flow_ref_R_flow.nc"
    )

    errors = run_uncertainty(
        _cfg(tmp_path, simulations, references),
        _tasks(bindings, simulations, references),
        output,
        ["RMSE"],
        make_phase_error_fn=lambda phase, message, **details: {"phase": phase, "message": message, **details},
    )

    assert errors == []
    summary = json.loads((output / "uncertainty" / "summary.json").read_text())
    assert summary["verdicts"][0]["status"] == "indistinguishable"
    assert summary["verdicts"][0]["references"]["R"]["estimate"] == 0.0


def test_station_loader_uses_current_evaluation_rows_and_best_time_alignment(tmp_path):
    output = tmp_path / "case"
    folder = output / "data" / "stn_R_A"
    folder.mkdir(parents=True)
    (output / "metrics").mkdir()
    sim_time = np.array(["2000-01-31", "2000-02-28", "2000-03-31", "2000-04-30"], dtype="datetime64[D]")
    ref_time = np.array(["2000-01-30", "2000-02-28", "2000-03-30", "2000-04-30"], dtype="datetime64[D]")
    for station in ("current", "stale"):
        xr.DataArray(np.arange(4.0), coords={"time": sim_time}, dims="time", name="flow").to_netcdf(
            folder / f"Flow_sim_{station}_2000_2000.nc"
        )
        xr.DataArray(np.arange(4.0), coords={"time": ref_time}, dims="time", name="flow").to_netcdf(
            folder / f"Flow_ref_{station}_2000_2000.nc"
        )
    pd.DataFrame({"ID": ["current"], "use_syear": [2000], "use_eyear": [2000], "RMSE": [0.0]}).to_csv(
        output / "metrics" / "Flow_stn_R_A_evaluations.csv", index=False
    )
    task = _tasks(_bindings(["A"], ["R"], "stn"), ["A"], ["R"])[0]

    pairs = _load_station_pairs(task, output)

    assert list(pairs) == ["current"]
    assert pairs["current"][0].sizes["time"] == 4


def test_station_reference_spread_matches_locations_not_source_ids(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "spread.csv"
    pd.DataFrame({"ID": ["A"], "ref_lat": [10.0], "ref_lon": [20.0], "RMSE": [1.0]}).to_csv(first, index=False)
    pd.DataFrame({"ID": ["B"], "ref_lat": [10.0], "ref_lon": [20.0], "RMSE": [3.0]}).to_csv(second, index=False)

    assert _write_station_spread(
        [("R1", first), ("R2", second)],
        output,
        axis="reference",
        metric="RMSE",
    )
    result = pd.read_csv(output)

    assert len(result) == 1
    assert result.loc[0, "member_count"] == 2
    assert result.loc[0, "reference_sensitivity"] == 1.0


def test_grid_spread_does_not_report_zero_with_one_member(tmp_path):
    output = tmp_path / "spread.nc"
    first = xr.DataArray([1.0, np.nan], dims="x")
    second = xr.DataArray([np.nan, 2.0], dims="x")

    _write_grid_spread([("A", first), ("B", second)], output, axis="model", metric="RMSE")

    with xr.open_dataset(output) as result:
        assert result["member_count"].values.tolist() == [1, 1]
        assert np.isnan(result["model_spread"]).all()


def test_uncertainty_cache_requires_all_declared_outputs(tmp_path):
    uncertainty = tmp_path / "uncertainty"
    uncertainty.mkdir()
    summary = {
        "schema_version": 1,
        "config": {
            "enabled": True,
            "metrics": ["RMSE"],
            "n_resamples": 20,
            "confidence_level": 0.9,
            "block_length": 2,
            "seed": 4,
        },
        "bootstrap": [],
        "products": {"model_spread": [], "reference_sensitivity": [], "skipped": []},
        "verdicts": [],
    }
    (uncertainty / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (uncertainty / "verdicts.json").write_text(
        json.dumps({"schema_version": 1, "verdicts": []}),
        encoding="utf-8",
    )

    cfg = _cfg(tmp_path, ["A"], ["R"])
    assert not _cached_uncertainty_outputs_complete(cfg, tmp_path)
    (uncertainty / "bootstrap_summary.csv").write_text("metric\\nRMSE\\n", encoding="utf-8")
    assert _cached_uncertainty_outputs_complete(cfg, tmp_path)


def test_local_runner_invokes_enabled_uncertainty_phase(tmp_path, monkeypatch):
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    import openbench.runner.orchestration as orchestration

    cfg = _cfg(tmp_path, ["A"], ["R"])
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Flow": True},
        metrics=["RMSE"],
        scores=[],
        comparisons=[],
        statistics=[],
        general={"only_drawing": False, "unified_mask": False},
    )
    bindings = SimpleNamespace(
        runner_cfg=runner_cfg,
        namelists=adapter.LegacyNamelists(main={}, reference={}, simulation={}),
        figures={},
        has_grid_evaluation=lambda variables: adapter.GridEvaluationEvidence(True),
    )
    task = {"var_name": "Flow", "sim_source": "A", "ref_source": "R", "bindings": bindings}
    calls = []

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda config: bindings)
    monkeypatch.setattr(local_runner, "_build_evaluation_tasks", lambda **kwargs: [task])
    monkeypatch.setattr(local_runner, "_preprocess_variable_tasks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        local_runner,
        "_evaluate_ready_tasks",
        lambda *args, **kwargs: [{"variable": "Flow", "sim": "A", "ref": "R", "status": "success", "skipped": False}],
    )
    monkeypatch.setattr(
        local_runner,
        "_run_uncertainty",
        lambda config, tasks, output_dir, metrics: calls.append((config, tasks, output_dir, metrics)) or [],
    )
    monkeypatch.setattr(orchestration, "_write_run_manifest", lambda *args: None)

    result = local_runner.run_evaluation(cfg, force=True)

    assert result["status"] == "success"
    assert len(calls) == 1
    assert calls[0][3] == ["RMSE"]


def test_local_runner_removes_stale_uncertainty_summary_when_disabled(tmp_path, monkeypatch):
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    import openbench.runner.orchestration as orchestration

    cfg = _cfg(tmp_path, ["A"], ["R"])
    cfg.uncertainty.enabled = False
    output = tmp_path / "case" / "uncertainty"
    output.mkdir(parents=True)
    summary = output / "summary.json"
    summary.write_text('{"config":{"enabled":true}}', encoding="utf-8")
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Flow": True},
        metrics=["RMSE"],
        scores=[],
        comparisons=[],
        statistics=[],
        general={"only_drawing": False, "unified_mask": False},
    )
    bindings = SimpleNamespace(
        runner_cfg=runner_cfg,
        namelists=adapter.LegacyNamelists(main={}, reference={}, simulation={}),
        figures={},
        has_grid_evaluation=lambda variables: adapter.GridEvaluationEvidence(True),
    )
    task = {"var_name": "Flow", "sim_source": "A", "ref_source": "R", "bindings": bindings}

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda config: bindings)
    monkeypatch.setattr(local_runner, "_build_evaluation_tasks", lambda **kwargs: [task])
    monkeypatch.setattr(local_runner, "_preprocess_variable_tasks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        local_runner,
        "_evaluate_ready_tasks",
        lambda *args, **kwargs: [{"variable": "Flow", "sim": "A", "ref": "R", "status": "success", "skipped": False}],
    )
    monkeypatch.setattr(orchestration, "_write_run_manifest", lambda *args: None)

    result = local_runner.run_evaluation(cfg, force=True)

    assert result["status"] == "success"
    assert not summary.exists()
