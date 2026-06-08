"""Tests for runner.local orchestration and status reporting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openbench.config.schema import (
    ComparisonConfig,
    DaskConfig,
    EvaluationConfig,
    IOConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
    StatisticsConfig,
)
from openbench.runner.local import run_evaluation


def _make_cfg(
    tmp_path,
    *,
    comparison_enabled=True,
    statistics_enabled=False,
    generate_report=False,
) -> OpenBenchConfig:
    return OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=generate_report,
        ),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "TestRef"}),
        simulation={"SimA": SimulationEntry(model="ModelA", root_dir=str(tmp_path))},
        comparison=ComparisonConfig(enabled=comparison_enabled, items=["HeatMap"]),
        statistics=StatisticsConfig(enabled=statistics_enabled, items=["Mean"]),
    )


def _legacy_payload(tmp_path):
    return {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
        },
        "evaluation_items": {"Runoff": True},
        "metrics": {"bias": True},
        "scores": {"Overall_Score": True},
        "comparisons": {"HeatMap": True},
        "statistics": {"Mean": True},
    }


def _namelists(tmp_path):
    main_nl = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
        },
        "evaluation_items": {"Runoff": True},
        "metrics": {"bias": True},
        "scores": {"Overall_Score": True},
        "comparisons": {"HeatMap": True},
        "statistics": {"Mean": True},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "TestRef"},
        "Runoff": {"TestRef_data_type": "grid"},
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {"SimA_data_type": "grid"},
    }
    return main_nl, ref_nml, sim_nml


def _write_fake_grid_outputs(
    casedir,
    var_name,
    ref_source,
    sim_source,
    *,
    metrics=("bias", "RMSE", "correlation"),
    scores=("Overall_Score",),
):
    """Write placeholder grid outputs for runner orchestration tests.

    These tests stub the evaluator, so a "successful" fake evaluator must
    still satisfy the runner's requested-output contract.
    """
    from pathlib import Path

    case = Path(casedir)
    stem = f"{var_name}_ref_{ref_source}_sim_{sim_source}"
    if metrics:
        (case / "metrics").mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            _write_fake_netcdf(case / "metrics" / f"{stem}_{metric}.nc")
    if scores:
        (case / "scores").mkdir(parents=True, exist_ok=True)
        for score in scores:
            _write_fake_netcdf(case / "scores" / f"{stem}_{score}.nc")


def _write_fake_station_outputs(casedir, var_name, ref_source, sim_source, *, metrics=True, scores=True):
    """Write placeholder station CSV outputs for runner orchestration tests."""
    from pathlib import Path

    case = Path(casedir)
    filename = f"{var_name}_stn_{ref_source}_{sim_source}_evaluations.csv"
    metric_columns = ["bias", "RMSE", "correlation"] if metrics is True else list(metrics or [])
    score_columns = ["Overall_Score"] if scores is True else list(scores or [])

    def _csv_text(columns):
        return ",".join(["ID", *columns]) + "\n" + ",".join(["S1", *(["1.0"] * len(columns))]) + "\n"

    if metrics:
        (case / "metrics").mkdir(parents=True, exist_ok=True)
        (case / "metrics" / filename).write_text(_csv_text(metric_columns))
    if scores:
        (case / "scores").mkdir(parents=True, exist_ok=True)
        (case / "scores" / filename).write_text(_csv_text(score_columns))


def _write_fake_netcdf(path):
    """Write a tiny valid NetCDF file for output-readability checks."""
    import numpy as np
    import xarray as xr

    xr.Dataset({"value": ("sample", np.array([1.0]))}).to_netcdf(path)


class _ExplodingNamelist:
    """Sentinel object for asserting runner stops reading namelists directly."""

    def __getitem__(self, key):
        raise AssertionError("runner should use bindings methods instead of direct namelist reads")

    def get(self, *args, **kwargs):
        raise AssertionError("runner should use bindings methods instead of direct namelist reads")


def test_runner_declares_bridge_runtime_field_contract():
    """The runner should publish the exact bridge fields it consumes."""
    import openbench.runner.local as local_runner

    assert local_runner.BRIDGE_RUNTIME_FIELDS == {
        "casedir",
        "ref_varname",
        "sim_varname",
        "ref_data_type",
        "sim_data_type",
        "compare_tim_res",
        "compare_grid_res",
        "compare_tzone",
        "regrid_backend",
        "unified_mask",
    }


def test_evaluation_task_worker_count_caps_to_cpu_count(monkeypatch):
    import openbench.runner.local as local_runner

    monkeypatch.setattr(local_runner.os, "cpu_count", lambda: 2)

    assert local_runner._evaluation_task_worker_count(64, 10) == 2
    assert local_runner._evaluation_task_worker_count(64, 1) == 1
    assert local_runner._evaluation_task_worker_count(0, 10) == 1


def test_task_level_parallel_safe_rejects_station_tasks(tmp_path):
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    bindings = adapter.RunnerBindings(
        runner_cfg=adapter.RunnerConfig(
            basename="case",
            basedir=str(tmp_path),
            evaluation_items={"Runoff": True},
            metrics=["bias"],
            scores=[],
            comparisons=[],
            statistics=[],
            general={"compare_tim_res": "Month", "compare_tzone": 0.0, "compare_grid_res": 1.0},
        ),
        namelists=adapter.LegacyNamelists(main={}, reference={}, simulation={}),
        figures=adapter.LegacyFigureConfig(raw={}),
    )

    assert not local_runner._task_level_parallel_safe(
        [{"bindings": bindings, "ref_data_type": "grid", "sim_data_type": "stn"}]
    )
    assert not local_runner._task_level_parallel_safe(
        [{"bindings": bindings, "ref_data_type": "stn", "sim_data_type": "grid"}]
    )


def test_ready_tasks_parallelize_only_when_unified_mask_disabled(monkeypatch):
    """Task-level eval parallelism is opt-in via unified_mask=False + num_cores>1."""
    import openbench.runner.local as local_runner

    tasks = [{"var_name": "Runoff", "sim_source": f"Sim{i}", "ref_source": "Ref"} for i in range(3)]
    evaluated = []

    def fake_evaluate(task):
        evaluated.append(task["sim_source"])
        return {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
        }

    class FakeExecutor:
        calls = []

        def __init__(self, max_workers):
            self.max_workers = max_workers
            FakeExecutor.calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, items):
            return [func(item) for item in items]

    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(local_runner, "_task_level_parallel_safe", lambda tasks: True)
    monkeypatch.setattr(local_runner, "ProcessPoolExecutor", FakeExecutor)

    results = local_runner._evaluate_ready_tasks(
        tasks,
        num_cores=2,
        unified_mask=False,
        only_drawing=False,
    )

    assert [result["sim"] for result in results] == ["Sim0", "Sim1", "Sim2"]
    assert evaluated == ["Sim0", "Sim1", "Sim2"]
    assert FakeExecutor.calls == [2]


def test_ready_tasks_stay_serial_for_unified_mask_or_only_drawing(monkeypatch):
    """Shared-mask and renderer paths must not use task-level parallel workers."""
    import openbench.runner.local as local_runner

    tasks = [{"var_name": "Runoff", "sim_source": "SimA", "ref_source": "Ref"}]
    evaluated = []

    def fake_evaluate(task):
        evaluated.append(task["sim_source"])
        return {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
        }

    class ExplodingExecutor:
        def __init__(self, *args, **kwargs):  # pragma: no cover - should never be called
            raise AssertionError("serial paths must not create a ProcessPoolExecutor")

    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(local_runner, "_task_level_parallel_safe", lambda tasks: True)
    monkeypatch.setattr(local_runner, "ProcessPoolExecutor", ExplodingExecutor)

    local_runner._evaluate_ready_tasks(tasks * 2, num_cores=4, unified_mask=True, only_drawing=False)
    local_runner._evaluate_ready_tasks(tasks * 2, num_cores=4, unified_mask=False, only_drawing=True)
    local_runner._evaluate_ready_tasks(
        tasks * 2,
        num_cores=4,
        unified_mask=False,
        only_drawing=False,
        dask_distributed=True,
    )

    assert evaluated == ["SimA", "SimA", "SimA", "SimA", "SimA", "SimA"]


def test_ready_tasks_parallelize_unified_mask_across_ref_groups(monkeypatch):
    """Unified-mask tasks can parallelize across refs after preprocessing is complete."""
    import openbench.runner.local as local_runner

    tasks = [
        {"var_name": "Runoff", "sim_source": "SimA", "ref_source": "Ref1", "ref_preprocessed": True},
        {"var_name": "Runoff", "sim_source": "SimB", "ref_source": "Ref1", "ref_preprocessed": True},
        {"var_name": "Runoff", "sim_source": "SimA", "ref_source": "Ref2", "ref_preprocessed": True},
    ]
    evaluated = []

    def fake_evaluate(task):
        evaluated.append((task["ref_source"], task["sim_source"]))
        return {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
        }

    class FakeExecutor:
        calls = []

        def __init__(self, max_workers):
            FakeExecutor.calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, groups):
            groups = list(groups)
            assert [[task["ref_source"] for task in group] for group in groups] == [["Ref1", "Ref1"], ["Ref2"]]
            return [func(group) for group in groups]

    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(local_runner, "_task_level_parallel_safe", lambda tasks: True)
    monkeypatch.setattr(local_runner, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(local_runner.os, "cpu_count", lambda: 8)

    results = local_runner._evaluate_ready_tasks(
        tasks,
        num_cores=4,
        unified_mask=True,
        only_drawing=False,
    )

    assert [result["ref"] for result in results] == ["Ref1", "Ref1", "Ref2"]
    assert evaluated == [("Ref1", "SimA"), ("Ref1", "SimB"), ("Ref2", "SimA")]
    assert FakeExecutor.calls == [2]


def test_unified_mask_parallel_requires_preprocessed_refs(monkeypatch):
    import openbench.runner.local as local_runner

    tasks = [
        {"var_name": "Runoff", "sim_source": "SimA", "ref_source": "Ref1"},
        {"var_name": "Runoff", "sim_source": "SimA", "ref_source": "Ref2"},
    ]
    evaluated = []

    def fake_evaluate(task):
        evaluated.append(task["ref_source"])
        return {"variable": task["var_name"], "sim": task["sim_source"], "ref": task["ref_source"], "status": "success"}

    class ExplodingExecutor:
        def __init__(self, *args, **kwargs):  # pragma: no cover - should never be called
            raise AssertionError("unpreprocessed unified-mask tasks must stay serial")

    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(local_runner, "_task_level_parallel_safe", lambda tasks: True)
    monkeypatch.setattr(local_runner, "ProcessPoolExecutor", ExplodingExecutor)

    local_runner._evaluate_ready_tasks(tasks, num_cores=4, unified_mask=True, only_drawing=False)

    assert evaluated == ["Ref1", "Ref2"]


def test_bridge_access_helper_is_shared_by_preprocess_and_evaluate(tmp_path, monkeypatch):
    """Runner bridge reads should flow through one shared helper."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    helper_calls = []

    def fake_build_bridge_runtime_info(task):
        helper_calls.append((task["var_name"], task["sim_source"], task["ref_source"]))
        return {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": task["ref_source"],
            "sim_source": task["sim_source"],
        }

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            return None

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info
            self.fig_nml = fig_nml

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "Runoff",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fake_build_bridge_runtime_info)
    monkeypatch.setattr(local_runner, "_apply_unified_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert helper_calls == [
        ("Runoff", "SimA", "TestRef"),
        ("Runoff", "SimA", "TestRef"),
    ]


def test_runner_builds_runtime_context_without_mutating_reader_state(tmp_path, monkeypatch):
    """Runner-owned fields should not be written back onto the reader dict."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.config.legacy_processors as legacy_processors
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    reader_state = {
        "casedir": str(tmp_path / "case"),
        "ref_varname": "runoff_ref",
        "sim_varname": "runoff_sim",
        "ref_data_type": "grid",
        "sim_data_type": "grid",
    }
    processor_infos = []
    evaluation_infos = []

    class FakeInfoReader:
        def __init__(self, *args, **kwargs):
            for key, value in reader_state.items():
                setattr(self, key, value)

    class FakeProcessor:
        def __init__(self, info):
            processor_infos.append(dict(info))
            self.info = info

        def prepare_source(self, datasource):
            return None

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            evaluation_infos.append(dict(info))
            self.info = info
            self.fig_nml = fig_nml

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "Runoff",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(legacy_processors, "GeneralInfoReader", FakeInfoReader)
    monkeypatch.setattr(local_runner, "_apply_unified_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert "ref_source" not in reader_state
    assert "sim_source" not in reader_state
    assert processor_infos == [
        {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": "TestRef",
            "sim_source": "SimA",
        }
    ]
    assert evaluation_infos == [
        {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": "TestRef",
            "sim_source": "SimA",
        }
    ]


def test_runtime_context_uses_bindings_bridge_info_builder(monkeypatch):
    """Runner should delegate bridge-info extraction through the task bindings object."""
    import openbench.config.adapter as adapter_module
    import openbench.runner.local as local_runner

    build_calls = []
    task = {
        "bindings": type(
            "Bindings",
            (),
            {
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: (
                    build_calls.append((var_name, sim_source, ref_source))
                    or adapter_module.BridgeRuntimeInfo(
                        payload={
                            "casedir": "/tmp/case",
                            "ref_varname": "runoff_ref",
                            "sim_varname": "runoff_sim",
                            "ref_data_type": "grid",
                            "sim_data_type": "grid",
                        }
                    )
                )
            },
        )(),
        "main_nl": {},
        "sim_nml": {},
        "ref_nml": {},
        "metric_vars": [],
        "score_vars": [],
        "comparison_vars": [],
        "statistic_vars": [],
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "RefA",
    }

    runtime = local_runner._build_runtime_context(task)

    assert isinstance(runtime.bridge_info, adapter_module.BridgeRuntimeInfo)
    assert runtime.bridge_info.to_info() == {
        "casedir": "/tmp/case",
        "ref_varname": "runoff_ref",
        "sim_varname": "runoff_sim",
        "ref_data_type": "grid",
        "sim_data_type": "grid",
    }
    assert build_calls == [("Runoff", "SimA", "RefA")]
    assert runtime.ref_source == "RefA"
    assert runtime.sim_source == "SimA"


def test_runtime_context_prefers_bindings_runtime_info_builder(monkeypatch):
    """Runner should read runtime info from task bindings after removing the free adapter helper."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    task = {
        "bindings": type(
            "Bindings",
            (),
            {
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: adapter.BridgeRuntimeInfo(
                    payload={
                        "casedir": "/tmp/case",
                        "ref_varname": "runoff_ref",
                        "sim_varname": "runoff_sim",
                        "ref_data_type": "grid",
                        "sim_data_type": "grid",
                    }
                )
            },
        )(),
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "RefA",
    }

    runtime = local_runner._build_runtime_context(task)

    assert not hasattr(adapter, "build_runtime_info")
    assert runtime.bridge_info.to_info()["casedir"] == "/tmp/case"
    assert runtime.ref_source == "RefA"
    assert runtime.sim_source == "SimA"


def test_runner_uses_explicit_processing_api(tmp_path, monkeypatch):
    """Runner preprocessing should call prepare_source() instead of string-mode process()."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    prepare_calls = []

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            prepare_calls.append(datasource)

        def process(self, data=None, **kwargs):
            raise AssertionError("runner should use prepare_source for datasource preprocessing")

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info
            self.fig_nml = fig_nml

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "Runoff",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(
        local_runner,
        "_build_bridge_runtime_info",
        lambda task: {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": task["ref_source"],
            "sim_source": task["sim_source"],
        },
    )
    monkeypatch.setattr(local_runner, "_apply_unified_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert prepare_calls == ["ref", "sim"]


def test_runner_uses_runner_config_without_direct_legacy_dict_dependency(tmp_path, monkeypatch):
    """Runner should not need adapter.to_legacy_config() once runner-facing inputs exist."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
        },
    )

    monkeypatch.setattr(adapter, "build_runner_config", lambda cfg: runner_cfg)
    monkeypatch.setattr(
        adapter,
        "to_legacy_config",
        lambda cfg: (_ for _ in ()).throw(AssertionError("runner should not call to_legacy_config")),
    )
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(
        local_runner,
        "_build_bridge_runtime_info",
        lambda task: {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": task["ref_source"],
            "sim_source": task["sim_source"],
        },
    )
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        },
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert result["variables"] == ["Runoff"]
    assert result["metrics"] == ["bias"]


def test_runner_uses_single_adapter_bindings_entrypoint(tmp_path, monkeypatch):
    """Runner should consume one adapter bindings object instead of separate legacy builders."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
        },
    )

    monkeypatch.setattr(
        adapter,
        "build_runner_bindings",
        lambda cfg: type(
            "Bindings",
            (),
            {
                "runner_cfg": runner_cfg,
                "main_nl": main_nl,
                "ref_nml": ref_nml,
                "sim_nml": sim_nml,
                "fig_nml": {},
                "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
                "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(True),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(tmp_path / "case"),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
            },
        )(),
        raising=False,
    )
    monkeypatch.setattr(
        adapter,
        "build_runner_config",
        lambda cfg: (_ for _ in ()).throw(AssertionError("runner should use build_runner_bindings")),
    )
    monkeypatch.setattr(
        adapter,
        "build_legacy_namelists",
        lambda cfg: (_ for _ in ()).throw(AssertionError("runner should use build_runner_bindings")),
    )
    monkeypatch.setattr(
        adapter,
        "build_fig_nml",
        lambda: (_ for _ in ()).throw(AssertionError("runner should use build_runner_bindings")),
    )
    monkeypatch.setattr(
        local_runner,
        "_build_bridge_runtime_info",
        lambda task: {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": task["ref_source"],
            "sim_source": task["sim_source"],
        },
    )
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        },
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert result["variables"] == ["Runoff"]
    assert result["metrics"] == ["bias"]


def test_runner_tasks_no_longer_expose_loose_binding_sections(tmp_path, monkeypatch):
    """Runner should build task inputs through bindings instead of reading ref/sim namelists directly."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "main_nl": main_nl,
            "ref_nml": _ExplodingNamelist(),
            "sim_nml": _ExplodingNamelist(),
            "fig_nml": {},
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
        },
    )()

    captured_tasks = []

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: (
            captured_tasks.append(task)
            or {
                "variable": task["var_name"],
                "sim": task["sim_source"],
                "ref": task["ref_source"],
                "status": "success",
                "skipped": False,
                "cache_key": task["cache_key"],
                "config_hash": task["config_hash"],
            }
        ),
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert len(captured_tasks) == 1
    task = captured_tasks[0]
    assert task["bindings"] is bindings
    assert "main_nl" not in task
    assert "ref_nml" not in task
    assert "sim_nml" not in task
    assert "fig_nml" not in task
    assert "metric_vars" not in task
    assert "score_vars" not in task
    assert "comparison_vars" not in task
    assert "statistic_vars" not in task


def test_runner_post_phase_helpers_consume_bindings_only(tmp_path, monkeypatch):
    """Post-phase helpers should receive bindings instead of loose namelist sections."""
    cfg = _make_cfg(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)
    _write_fake_grid_outputs(
        tmp_path / "case",
        "Runoff",
        "TestRef",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "main_nl": main_nl,
            "ref_nml": ref_nml,
            "sim_nml": sim_nml,
            "fig_nml": {},
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {},
        },
    )()

    comparison_calls = []

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(
        local_runner,
        "_run_comparison",
        lambda *args: comparison_calls.append(args) or [],
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg, comparison_only=True)

    assert result["status"] == "success"
    assert len(comparison_calls) == 1
    args = comparison_calls[0]
    assert len(args) == 3
    assert args[0] is bindings
    assert args[1] == ["HeatMap"]
    assert str(args[2]).endswith("/case")


def test_runner_statistics_helper_consumes_bindings_only(tmp_path, monkeypatch):
    """Statistics gating and invocation should stay behind the bindings boundary."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False, statistics_enabled=True)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "main_nl": main_nl,
            "ref_nml": _ExplodingNamelist(),
            "sim_nml": _ExplodingNamelist(),
            "fig_nml": {},
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(True),
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
        },
    )()

    statistics_calls = []

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        },
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])
    monkeypatch.setattr(local_runner, "_run_statistics", lambda *args: statistics_calls.append(args) or [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert len(statistics_calls) == 1
    args = statistics_calls[0]
    assert len(args) == 3
    assert args[0] is bindings
    assert args[1] == ["Mean"]
    assert str(args[2]).endswith("/case")


def test_runner_report_helper_consumes_bindings_report_config(tmp_path, monkeypatch):
    """Report helper should build report config through bindings instead of raw ref/sim namelists."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False, generate_report=True)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner
    import openbench.util.report as report_module

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": True,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    report_configs = []
    generated_outputs = {"html": str(tmp_path / "case" / "reports" / "index.html")}
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "main_nl": main_nl,
            "ref_nml": _ExplodingNamelist(),
            "sim_nml": _ExplodingNamelist(),
            "fig_nml": {},
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
            "build_report_config": lambda self: adapter.ReportContext(
                runner_cfg=runner_cfg,
                namelists=adapter.LegacyNamelists(
                    main={"general": runner_cfg.general},
                    reference={"general": {"Runoff_ref_source": "TestRef"}},
                    simulation={"general": {"Runoff_sim_source": ["SimA"]}},
                ),
            ),
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
        },
    )()

    class FakeReportGenerator:
        def __init__(self, report_config, output_dir):
            report_configs.append((report_config, output_dir))

        def generate_report(self):
            return generated_outputs

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        },
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])
    monkeypatch.setattr(report_module, "ReportGenerator", FakeReportGenerator)

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert report_configs == [
        (
            bindings.build_report_config().to_report_config(),
            str(tmp_path / "case"),
        )
    ]


def test_statistics_helper_uses_bindings_statistics_context(tmp_path, monkeypatch):
    """Statistics helper should get main_nl-derived payload from bindings instead of reading raw sections."""
    import openbench.config.adapter as adapter_module
    import openbench.core.statistics.Mod_Statistics as stats_module
    import openbench.runner.local as local_runner

    stats_calls = []
    namelists = adapter_module.LegacyNamelists(
        main={
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "num_cores": 1,
                "compare_tim_res": "Month",
                "compare_grid_res": 0.5,
                "syear": 2000,
                "eyear": 2001,
            }
        },
        reference={},
        simulation={},
    )
    context = adapter_module.StatisticsContext(
        namelists=namelists,
        stats_dir=str(tmp_path / "case" / "statistics"),
        stats_nml={
            "general": {"Mean_data_source": "Mean_Data"},
            "Mean": {"Mean_Data_dir": str(tmp_path / "case" / "data")},
        },
        num_cores=1,
        statistic_fig={"Mean": {"title": "Mean"}},
    )
    bindings = type(
        "Bindings",
        (),
        {
            "main_nl": _ExplodingNamelist(),
            "fig_nml": _ExplodingNamelist(),
            "build_statistics_context": lambda self, statistic_vars: context,
        },
    )()

    class FakeStatisticsProcessing:
        def __init__(self, main_nl, stats_nml, stats_dir, num_cores):
            stats_calls.append(("init", main_nl, stats_nml, stats_dir, num_cores))

        def scenarios_Basic_analysis(self, statistic, stat_cfg, stat_fig):
            stats_calls.append(("run", statistic, stat_cfg, stat_fig))

    monkeypatch.setattr(stats_module, "StatisticsProcessing", FakeStatisticsProcessing)

    errors = local_runner._run_statistics(bindings, ["Mean"])

    assert errors == []
    assert stats_calls == [
        ("init", context.namelists.main, context.stats_nml, context.stats_dir, 1),
        ("run", "Mean", context.stats_nml["Mean"], context.statistic_fig["Mean"]),
    ]


def test_comparison_helper_uses_bindings_comparison_context(tmp_path, monkeypatch):
    """Comparison helper should get its legacy-shaped payload from bindings, not raw sections."""
    import openbench.config.adapter as adapter_module
    import openbench.core.comparison as comparison_module
    import openbench.runner.local as local_runner

    comparison_calls = []
    namelists = adapter_module.LegacyNamelists(
        main={"general": {"basename": "case"}},
        reference={"general": {"Runoff_ref_source": "TestRef"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}},
    )
    context = adapter_module.ComparisonContext(
        namelists=namelists,
        evaluation_items=["Runoff"],
        score_vars=["Overall_Score"],
        metric_vars=["bias"],
        comparison_fig={"HeatMap": {"title": "HeatMap"}},
    )
    bindings = type(
        "Bindings",
        (),
        {
            "main_nl": _ExplodingNamelist(),
            "ref_nml": _ExplodingNamelist(),
            "sim_nml": _ExplodingNamelist(),
            "fig_nml": _ExplodingNamelist(),
            "runner_cfg": type("RunnerCfg", (), {})(),
            "build_comparison_context": lambda self: context,
        },
    )()

    class FakeComparisonProcessing:
        def __init__(self, main_nl, score_vars, metric_vars):
            comparison_calls.append(("init", main_nl, score_vars, metric_vars))

        def scenarios_HeatMap_comparison(
            self,
            basedir,
            sim_nml,
            ref_nml,
            evaluation_items,
            score_vars,
            metric_vars,
            fig_opts,
        ):
            comparison_calls.append(
                ("run", basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, fig_opts)
            )

    monkeypatch.setattr(comparison_module, "ComparisonProcessing", FakeComparisonProcessing)

    # _run_comparison filters items to those with score/metric output files on
    # disk; create a properly-named output for "Runoff" so the filter keeps it.
    # The filter requires <item>_(ref|stn)_* per real evaluation output naming.
    case_dir = tmp_path / "case"
    _write_fake_grid_outputs(case_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=("Overall_Score",))

    errors = local_runner._run_comparison(bindings, ["HeatMap"], case_dir)

    assert errors == []
    assert comparison_calls == [
        ("init", context.namelists.main, context.score_vars, context.metric_vars),
        (
            "run",
            str(tmp_path / "case"),
            context.namelists.simulation,
            context.namelists.reference,
            context.evaluation_items,
            context.score_vars,
            context.metric_vars,
            context.comparison_fig["HeatMap"],
        ),
    ]


def test_groupby_helper_uses_bindings_groupby_context(tmp_path, monkeypatch):
    """Groupby helper should get its payload from bindings instead of raw legacy sections."""
    import openbench.config.adapter as adapter_module
    import openbench.core.climatezone_groupby as climatezone_module
    import openbench.core.landcover_groupby as landcover_module
    import openbench.runner.local as local_runner

    cfg = type(
        "Cfg",
        (),
        {
            "project": type(
                "Project",
                (),
                {
                    "IGBP_groupby": True,
                    "PFT_groupby": True,
                    "climate_zone_groupby": True,
                },
            )(),
        },
    )()
    groupby_calls = []
    namelists = adapter_module.LegacyNamelists(
        main={"general": {"basename": "case"}},
        reference={"general": {"Runoff_ref_source": "TestRef"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}},
    )
    context = adapter_module.GroupbyContext(
        namelists=namelists,
        evaluation_items=["Runoff"],
        score_vars=["Overall_Score"],
        metric_vars=["bias"],
        validation_fig={"title": "validation"},
        climate_zone_fig={"title": "cz"},
    )
    bindings = type(
        "Bindings",
        (),
        {
            "main_nl": _ExplodingNamelist(),
            "ref_nml": _ExplodingNamelist(),
            "sim_nml": _ExplodingNamelist(),
            "fig_nml": _ExplodingNamelist(),
            "runner_cfg": type("RunnerCfg", (), {})(),
            "build_groupby_context": lambda self: context,
        },
    )()

    class FakeLCGroupby:
        def __init__(self, main_nl, score_vars, metric_vars):
            groupby_calls.append(("lc-init", main_nl, score_vars, metric_vars))

        def scenarios_IGBP_groupby_comparison(
            self, basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, validation_fig
        ):
            groupby_calls.append(
                ("igbp", basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, validation_fig)
            )

        def scenarios_PFT_groupby_comparison(
            self, basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, validation_fig
        ):
            groupby_calls.append(
                ("pft", basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, validation_fig)
            )

    class FakeCZGroupby:
        def __init__(self, main_nl, score_vars, metric_vars):
            groupby_calls.append(("cz-init", main_nl, score_vars, metric_vars))

        def scenarios_CZ_groupby_comparison(
            self, basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, cz_fig
        ):
            groupby_calls.append(("cz", basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, cz_fig))

    monkeypatch.setattr(landcover_module, "LC_groupby", FakeLCGroupby)
    monkeypatch.setattr(climatezone_module, "CZ_groupby", FakeCZGroupby)

    case_dir = tmp_path / "case"
    _write_fake_grid_outputs(case_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=("Overall_Score",))

    errors = local_runner._run_groupby(cfg, bindings, case_dir)

    assert errors == []
    assert groupby_calls == [
        ("lc-init", context.namelists.main, context.score_vars, context.metric_vars),
        (
            "igbp",
            str(tmp_path / "case"),
            context.namelists.simulation,
            context.namelists.reference,
            context.evaluation_items,
            context.score_vars,
            context.metric_vars,
            context.validation_fig,
        ),
        ("lc-init", context.namelists.main, context.score_vars, context.metric_vars),
        (
            "pft",
            str(tmp_path / "case"),
            context.namelists.simulation,
            context.namelists.reference,
            context.evaluation_items,
            context.score_vars,
            context.metric_vars,
            context.validation_fig,
        ),
        ("cz-init", context.namelists.main, context.score_vars, context.metric_vars),
        (
            "cz",
            str(tmp_path / "case"),
            context.namelists.simulation,
            context.namelists.reference,
            context.evaluation_items,
            context.score_vars,
            context.metric_vars,
            context.climate_zone_fig,
        ),
    ]


def test_evaluate_single_uses_bindings_evaluation_fig_nml(tmp_path, monkeypatch):
    """Single-task evaluation should get figure config from bindings, not read fig_nml directly."""
    import openbench.core.evaluation as evaluation_module
    import openbench.runner.local as local_runner

    captured = []
    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "bindings": type(
            "Bindings",
            (),
            {
                "fig_nml": _ExplodingNamelist(),
                "build_evaluation_fig_nml": lambda self: type(
                    "EvaluationFigureContext",
                    (),
                    {"to_fig_nml": lambda self: {"Validation": {"title": "ok"}}},
                )(),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(tmp_path / "case"),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
            },
        )(),
        "cache_key": "Runoff__SimA__TestRef",
        "config_hash": "deadbeef",
        "use_cache": False,
        "cache_dir": str(tmp_path / "case"),
        "ref_preprocessed": True,
    }

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            captured.append((info, fig_nml))

        def make_Evaluation(self):
            return None

    monkeypatch.setattr(evaluation_module, "Evaluation_grid", FakeGridEvaluation)

    result = local_runner._evaluate_single(task)

    assert result["status"] == "success"
    assert captured == [
        (
            {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
                "ref_source": "TestRef",
                "sim_source": "SimA",
            },
            {"Validation": {"title": "ok"}},
        )
    ]


def test_evaluate_single_keyerror_is_reported_as_error_not_cached_success(tmp_path, monkeypatch):
    """A KeyError from evaluation can mean missing data, not just a plot issue."""
    import openbench.core.evaluation as evaluation_module
    import openbench.runner.local as local_runner

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            pass

        def make_Evaluation(self):
            raise KeyError("missing_var")

    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "bindings": type(
            "Bindings",
            (),
            {
                "build_evaluation_fig_nml": lambda self: type(
                    "Fig",
                    (),
                    {"to_fig_nml": lambda self: {}},
                )(),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(tmp_path / "case"),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
            },
        )(),
        "cache_key": "Runoff__SimA__TestRef",
        "config_hash": "deadbeef",
        "use_cache": False,
        "cache_dir": str(tmp_path / "case"),
        "ref_preprocessed": True,
    }

    monkeypatch.setattr(evaluation_module, "Evaluation_grid", FakeGridEvaluation)

    result = local_runner._evaluate_single(task)

    assert result["status"] == "error"
    assert "missing_var" in result["error"]


def test_evaluate_single_does_not_cache_when_requested_outputs_are_missing(tmp_path, monkeypatch):
    """Evaluator success without requested files is a failed task, not cacheable."""
    import json

    import openbench.core.evaluation as evaluation_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    output_dir.mkdir()

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            pass

        def make_Evaluation(self):
            return None

    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "bindings": type(
            "Bindings",
            (),
            {
                "build_evaluation_fig_nml": lambda self: type(
                    "Fig",
                    (),
                    {"to_fig_nml": lambda self: {}},
                )(),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(output_dir),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
            },
        )(),
        "cache_key": "Runoff__SimA__TestRef",
        "config_hash": "deadbeef",
        "use_cache": True,
        "cache_dir": str(output_dir),
        "ref_preprocessed": True,
        "output_requirements": {"metrics": ["bias"], "scores": []},
    }

    monkeypatch.setattr(evaluation_module, "Evaluation_grid", FakeGridEvaluation)

    result = local_runner._evaluate_single(task)

    assert result["status"] == "error"
    assert "requested outputs are missing" in result["error"]
    cache_file = output_dir / ".openbench_cache.json"
    if cache_file.exists():
        assert json.loads(cache_file.read_text()) == {}


def test_evaluate_single_force_mode_still_fails_when_requested_outputs_are_missing(tmp_path, monkeypatch):
    """Bypassing cache must not bypass output completeness validation."""
    import openbench.core.evaluation as evaluation_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    output_dir.mkdir()

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            pass

        def make_Evaluation(self):
            return None

    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "bindings": type(
            "Bindings",
            (),
            {
                "build_evaluation_fig_nml": lambda self: type(
                    "Fig",
                    (),
                    {"to_fig_nml": lambda self: {}},
                )(),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(output_dir),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
            },
        )(),
        "cache_key": "Runoff__SimA__TestRef",
        "config_hash": "deadbeef",
        "use_cache": False,
        "cache_dir": str(output_dir),
        "ref_preprocessed": True,
        "output_requirements": {"metrics": ["bias"], "scores": []},
    }

    monkeypatch.setattr(evaluation_module, "Evaluation_grid", FakeGridEvaluation)

    result = local_runner._evaluate_single(task)

    assert result["status"] == "error"
    assert "requested outputs are missing" in result["error"]


def test_evaluate_single_does_not_cache_when_requested_netcdf_is_unreadable(tmp_path, monkeypatch):
    """Evaluator success with a corrupt requested NetCDF is a failed task."""
    import json

    import openbench.core.evaluation as evaluation_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    output_dir.mkdir()

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir()
            (metrics_dir / "Runoff_ref_TestRef_sim_SimA_bias.nc").write_text("not a netcdf")

    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "bindings": type(
            "Bindings",
            (),
            {
                "build_evaluation_fig_nml": lambda self: type(
                    "Fig",
                    (),
                    {"to_fig_nml": lambda self: {}},
                )(),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(output_dir),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
            },
        )(),
        "cache_key": "Runoff__SimA__TestRef",
        "config_hash": "deadbeef",
        "use_cache": True,
        "cache_dir": str(output_dir),
        "ref_preprocessed": True,
        "output_requirements": {"metrics": ["bias"], "scores": []},
    }

    monkeypatch.setattr(evaluation_module, "Evaluation_grid", FakeGridEvaluation)

    result = local_runner._evaluate_single(task)

    assert result["status"] == "error"
    assert "requested outputs are missing or unreadable" in result["error"]
    cache_file = output_dir / ".openbench_cache.json"
    if cache_file.exists():
        assert json.loads(cache_file.read_text()) == {}


def test_comparison_only_requires_existing_outputs(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    comparison_called = []

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(local_runner, "_run_comparison", lambda *args, **kwargs: comparison_called.append(True))
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: None)

    result = run_evaluation(cfg, comparison_only=True)

    assert result["status"] == "error"
    assert comparison_called == []
    assert any("missing prerequisite outputs" in err["message"] for err in result["errors"])


def test_comparison_only_aborts_when_any_requested_task_output_is_missing(tmp_path, monkeypatch):
    """comparison-only must not generate figures from a partial task set."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    cfg = OpenBenchConfig(
        project=ProjectConfig(name="case", output_dir=str(tmp_path), years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "TestRef"}),
        simulation={
            "SimA": SimulationEntry(model="ModelA", root_dir=str(tmp_path)),
            "SimB": SimulationEntry(model="ModelB", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=True, items=["HeatMap"]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )
    _write_fake_grid_outputs(
        tmp_path / "case",
        "Runoff",
        "TestRef",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "iter_task_sources": lambda self, variables: [
                adapter.EvaluationSource("Runoff", "SimA", "TestRef"),
                adapter.EvaluationSource("Runoff", "SimB", "TestRef"),
            ],
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(True),
        },
    )()

    comparison_called = []
    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(local_runner, "_run_comparison", lambda *args, **kwargs: comparison_called.append(True) or [])

    result = run_evaluation(cfg, comparison_only=True)

    assert result["status"] == "error"
    assert comparison_called == []
    assert result["evaluated"] == []
    assert any("missing prerequisite outputs" in err["message"] for err in result["errors"])


def test_post_phase_errors_are_reported_in_results(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    cfg.metrics = ["bias"]
    cfg.scores = ["Overall_Score"]
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)
    _write_fake_grid_outputs(
        tmp_path / "case",
        "Runoff",
        "TestRef",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(
        local_runner,
        "_run_comparison",
        lambda *args, **kwargs: [{"phase": "comparison", "message": "comparison failed"}],
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg, comparison_only=True)

    assert result["status"] == "partial"
    assert any(err["phase"] == "comparison" for err in result["errors"])


def test_preprocessing_errors_are_reported_and_skip_evaluation(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)

    import openbench.config.adapter as adapter
    import openbench.config.legacy_processors as legacy_processors
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    class FakeInfoReader:
        def __init__(self, *args, **kwargs):
            self.casedir = str(tmp_path / "case")
            self.ref_varname = "runoff_ref"
            self.ref_data_type = "grid"
            self.sim_data_type = "grid"

    class FailingProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            if datasource == "sim":
                raise RuntimeError("sim preprocessing exploded")

    evaluated = []

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(legacy_processors, "GeneralInfoReader", FakeInfoReader)
    monkeypatch.setattr(processing, "DatasetProcessing", FailingProcessor)
    monkeypatch.setattr(local_runner, "_evaluate_single", lambda task: evaluated.append(task) or {})
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "error"
    assert evaluated == []
    assert any(err["phase"] == "preprocess" for err in result["errors"])
    assert any("sim preprocessing exploded" in err["message"] for err in result["errors"])


def test_normal_run_skips_post_phases_after_partial_evaluation_failure(tmp_path, monkeypatch):
    """A normal run must not build comparison/statistics/report outputs from a partial task set."""
    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=True,
        ),
        evaluation=EvaluationConfig(variables=["Runoff", "ET"]),
        reference=ReferenceConfig(sources={"Runoff": "RefA", "ET": "RefA"}),
        simulation={"SimA": SimulationEntry(model="ModelA", root_dir=str(tmp_path))},
        comparison=ComparisonConfig(enabled=True, items=["HeatMap"]),
        statistics=StatisticsConfig(enabled=True, items=["Mean"]),
    )
    cfg.project.IGBP_groupby = True
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True, "ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": False,
            "generate_report": True,
            "only_drawing": False,
        },
    )

    class Bindings:
        def __init__(self):
            self.runner_cfg = runner_cfg
            self.namelists = adapter.LegacyNamelists(main={"general": runner_cfg.general}, reference={}, simulation={})
            self.figures = adapter.LegacyFigureConfig(raw={})

        def iter_task_sources(self, variables):
            return [
                adapter.EvaluationSource("Runoff", "SimA", "RefA"),
                adapter.EvaluationSource("ET", "SimA", "RefA"),
            ]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "ref",
                "sim_varname": "sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            }

        def has_grid_evaluation(self, variables):
            return adapter.GridEvaluationEvidence(True)

    def fake_evaluate(task):
        if task["var_name"] == "ET":
            return {
                "variable": "ET",
                "sim": "SimA",
                "ref": "RefA",
                "status": "error",
                "error": "ET failed",
                "cache_key": task["cache_key"],
                "config_hash": task["config_hash"],
                "skipped": False,
            }
        return {
            "variable": "Runoff",
            "sim": "SimA",
            "ref": "RefA",
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        }

    post_calls = []
    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(local_runner, "_run_comparison", lambda *args, **kwargs: post_calls.append("comparison") or [])
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: post_calls.append("groupby") or [])
    monkeypatch.setattr(local_runner, "_run_statistics", lambda *args, **kwargs: post_calls.append("statistics") or [])
    monkeypatch.setattr(local_runner, "_run_report", lambda *args, **kwargs: post_calls.append("report") or [])

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "partial"
    assert [item["variable"] for item in result["evaluated"]] == ["Runoff"]
    assert any(err["phase"] == "evaluation" and "ET failed" in err["message"] for err in result["errors"])
    assert post_calls == []


def test_partial_run_cache_marks_only_successful_tasks(tmp_path, monkeypatch):
    """A partial multi-variable run must not cache the failed task's config hash."""
    import json

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner
    from openbench.runner.cache import make_cache_key

    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=False,
        ),
        evaluation=EvaluationConfig(variables=["Runoff", "ET"]),
        reference=ReferenceConfig(sources={"Runoff": "RefA", "ET": "RefA"}),
        simulation={"SimA": SimulationEntry(model="ModelA", root_dir=str(tmp_path))},
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )
    output_dir = tmp_path / "case"
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True, "ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": False,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": False,
        },
    )

    class Bindings:
        def iter_task_sources(self, variables):
            return [
                adapter.EvaluationSource("Runoff", "SimA", "RefA"),
                adapter.EvaluationSource("ET", "SimA", "RefA"),
            ]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {
                "casedir": str(output_dir),
                "ref_varname": "ref",
                "sim_varname": "sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            }

        def build_evaluation_fig_nml(self):
            return type("Fig", (), {"to_fig_nml": lambda self: {}})()

        def has_grid_evaluation(self, variables):
            return adapter.GridEvaluationEvidence(False)

    Bindings.runner_cfg = runner_cfg
    Bindings.namelists = adapter.LegacyNamelists(main={"general": runner_cfg.general}, reference={}, simulation={})
    Bindings.figures = adapter.LegacyFigureConfig(raw={})

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            return None

    calls = []

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            var_name = ["Runoff", "ET"][len(calls)]
            calls.append(var_name)
            if var_name == "ET":
                raise RuntimeError("ET failed after Runoff cached")
            _write_fake_grid_outputs(
                self.info["casedir"],
                var_name,
                self.info["ref_source"],
                self.info["sim_source"],
                metrics=("bias",),
                scores=("Overall_Score",),
            )

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "partial"
    assert [entry["variable"] for entry in result["evaluated"]] == ["Runoff"]
    assert any(err["phase"] == "evaluation" and "ET failed" in err["message"] for err in result["errors"])

    cache = json.loads((output_dir / ".openbench_cache.json").read_text())
    assert make_cache_key("Runoff", "SimA", "RefA") in cache
    assert make_cache_key("ET", "SimA", "RefA") not in cache


def test_runner_returns_error_when_no_tasks_are_queued(tmp_path, monkeypatch):
    """An empty task plan should not be reported as a successful evaluation."""
    cfg = _make_cfg(tmp_path, comparison_enabled=False, generate_report=True)

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": True,
            "generate_report": True,
            "only_drawing": False,
        },
    )

    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(main={}, reference={}, simulation={}),
            "figures": adapter.LegacyFigureConfig(raw={}),
            "iter_task_sources": lambda self, variables: [],
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
        },
    )()

    report_calls = []
    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(local_runner, "_run_report", lambda *args: report_calls.append(args) or [])

    result = run_evaluation(cfg)

    assert result["status"] == "error"
    assert result["evaluated"] == []
    assert any(err["phase"] == "preflight" for err in result["errors"])
    assert report_calls == []


def test_task_config_hash_changes_when_runtime_inputs_change(tmp_path, monkeypatch):
    """Cache hashes must include years and source paths, not just var/sim/ref."""
    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    captured: list[str] = []

    def make_bindings(cfg):
        sim_root = cfg.simulation["SimA"].root_dir
        runner_cfg = adapter.RunnerConfig(
            basename="case",
            basedir=str(tmp_path),
            evaluation_items={"Runoff": True},
            metrics=["bias"],
            scores=["Overall_Score"],
            comparisons=[],
            statistics=[],
            general={
                "basename": "case",
                "basedir": str(tmp_path),
                "syear": cfg.project.years[0],
                "eyear": cfg.project.years[1],
                "min_lat": -90,
                "max_lat": 90,
                "min_lon": -180,
                "max_lon": 180,
                "compare_tim_res": "Month",
                "compare_grid_res": 0.5,
                "compare_tzone": 0,
                "num_cores": 1,
                "unified_mask": True,
                "time_alignment": "intersection",
                "generate_report": False,
                "only_drawing": False,
                "weight": "area",
            },
        )
        namelists = adapter.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={
                "general": {"Runoff_ref_source": "TestRef"},
                "Runoff": {
                    "TestRef_dir": str(tmp_path / "ref"),
                    "TestRef_varname": "runoff_ref",
                    "TestRef_varunit": "mm day-1",
                    "TestRef_data_groupby": "Year",
                    "TestRef_tim_res": "Month",
                    "TestRef_grid_res": 0.5,
                },
            },
            simulation={
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {
                    "SimA_dir": sim_root,
                    "SimA_varname": "runoff_sim",
                    "SimA_varunit": "mm day-1",
                    "SimA_data_groupby": "Year",
                    "SimA_tim_res": "Month",
                    "SimA_grid_res": 0.5,
                },
            },
        )

        return type(
            "Bindings",
            (),
            {
                "runner_cfg": runner_cfg,
                "namelists": namelists,
                "figures": adapter.LegacyFigureConfig(raw={}),
                "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(tmp_path / "case"),
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                },
                "build_evaluation_fig_nml": lambda self: type("Fig", (), {"to_fig_nml": lambda self: {}})(),
                "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
            },
        )()

    monkeypatch.setattr(adapter, "build_runner_bindings", make_bindings)
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: (
            captured.append(task["config_hash"])
            or {
                "variable": task["var_name"],
                "sim": task["sim_source"],
                "ref": task["ref_source"],
                "status": "success",
                "skipped": False,
                "cache_key": task["cache_key"],
                "config_hash": task["config_hash"],
            }
        ),
    )

    cfg_a = _make_cfg(tmp_path, comparison_enabled=False)
    cfg_a.simulation["SimA"].root_dir = str(tmp_path / "sim-a")
    cfg_b = _make_cfg(tmp_path, comparison_enabled=False)
    cfg_b.project.years = [2002, 2003]
    cfg_b.simulation["SimA"].root_dir = str(tmp_path / "sim-b")

    run_evaluation(cfg_a, force=True)
    run_evaluation(cfg_b, force=True)

    assert len(captured) == 2
    assert captured[0] != captured[1]


def test_task_config_hash_changes_when_shared_unified_mask_peer_sims_change(tmp_path):
    """intersection/strict unified_mask hashes must include peer sims sharing the ref mask."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": True,
            "time_alignment": "intersection",
            "regrid_backend": "cdo_remapcon",
            "only_drawing": False,
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"Runoff": {"TestRef_varname": "runoff_ref", "TestRef_data_type": "grid"}},
        simulation={
            "Runoff": {
                "SimA_varname": "runoff_sim",
                "SimA_data_type": "grid",
                "SimB_varname": "runoff_sim",
                "SimB_data_type": "grid",
            }
        },
    )
    bindings = type("Bindings", (), {"runner_cfg": runner_cfg, "namelists": namelists})()

    cfg_one = _make_cfg(tmp_path, comparison_enabled=False)
    cfg_one.project.unified_mask = True
    cfg_one.project.time_alignment = "intersection"
    cfg_two = _make_cfg(tmp_path, comparison_enabled=False)
    cfg_two.project.unified_mask = True
    cfg_two.project.time_alignment = "intersection"
    cfg_two.simulation["SimB"] = SimulationEntry(model="ModelB", root_dir=str(tmp_path / "sim-b"))

    def hash_for(cfg):
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    assert hash_for(cfg_one) != hash_for(cfg_two)


def test_task_config_hash_payload_serializes_simulation_entry_as_data(tmp_path):
    """Cache hash payload should not depend on dataclass repr formatting."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": True,
            "time_alignment": "intersection",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={"Runoff": {"SimA_dir": str(tmp_path), "SimA_varname": "runoff_sim"}},
            ),
        },
    )()

    payload = local_runner._task_hash_payload(
        cfg=cfg,
        bindings=bindings,
        var_name="Runoff",
        sim_source="SimA",
        ref_source="TestRef",
        metric_vars=["bias"],
        score_vars=["Overall_Score"],
        comparison_vars=[],
        statistic_vars=[],
    )

    assert payload["simulation"]["config"] == {
        "model": "ModelA",
        "root_dir": str(tmp_path),
        "data_type": None,
        "grid_res": None,
        "tim_res": None,
        "data_groupby": None,
        "prefix": None,
        "suffix": None,
        "fulllist": None,
        "variables": None,
    }


def test_task_config_hash_changes_when_input_file_mtime_changes(tmp_path):
    """Evaluation cache hashes must include source input file metadata."""
    import os

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    ref_dir = tmp_path / "ref"
    sim_dir = tmp_path / "sim"
    ref_dir.mkdir()
    sim_dir.mkdir()
    ref_file = ref_dir / "ref_runoff.nc"
    sim_file = sim_dir / "sim_runoff.nc"
    ref_file.write_text("ref", encoding="utf-8")
    sim_file.write_text("sim", encoding="utf-8")
    os.utime(ref_file, ns=(1_000_000_000, 1_000_000_000))
    os.utime(sim_file, ns=(1_000_000_000, 1_000_000_000))

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={
                    "Runoff": {
                        "TestRef_dir": str(ref_dir),
                        "TestRef_prefix": "ref_",
                        "TestRef_varname": "runoff",
                        "TestRef_suffix": "",
                    }
                },
                simulation={
                    "Runoff": {
                        "SimA_dir": str(sim_dir),
                        "SimA_prefix": "sim_",
                        "SimA_varname": "runoff",
                        "SimA_suffix": "",
                    }
                },
            ),
        },
    )()

    def cache_hash():
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    first = cache_hash()
    os.utime(sim_file, ns=(2_000_000_000, 2_000_000_000))

    assert cache_hash() != first


def test_task_config_hash_changes_when_input_file_content_changes_without_mtime(tmp_path):
    """Evaluation cache hashes should not stale-hit after same-size content edits."""
    import os

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    sim_file = sim_dir / "sim_runoff.nc"
    sim_file.write_bytes(b"old-data")
    fixed_time = (1_000_000_000, 1_000_000_000)
    os.utime(sim_file, ns=fixed_time)

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={
                    "Runoff": {
                        "SimA_dir": str(sim_dir),
                        "SimA_prefix": "sim_",
                        "SimA_varname": "runoff",
                        "SimA_suffix": "",
                    }
                },
            ),
        },
    )()

    def cache_hash():
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    first = cache_hash()
    sim_file.write_bytes(b"new-data")
    os.utime(sim_file, ns=fixed_time)

    assert cache_hash() != first


def test_task_config_hash_changes_when_middle_of_large_input_changes_without_mtime(tmp_path):
    """Sampled input digests must catch NetCDF-like data-section rewrites."""
    import os

    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    sim_file = sim_dir / "sim_runoff.nc"
    sim_file.write_bytes(b"H" * 8192 + b"A" * 8192 + b"T" * 8192 + b"Z" * 8192)
    fixed_time = (1_000_000_000, 1_000_000_000)
    os.utime(sim_file, ns=fixed_time)

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={
                    "Runoff": {
                        "SimA_dir": str(sim_dir),
                        "SimA_prefix": "sim_",
                        "SimA_varname": "runoff",
                        "SimA_suffix": "",
                    }
                },
            ),
        },
    )()

    def cache_hash():
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    first = cache_hash()
    sim_file.write_bytes(b"H" * 8192 + b"B" * 8192 + b"T" * 8192 + b"Z" * 8192)
    os.utime(sim_file, ns=fixed_time)

    assert cache_hash() != first


def test_task_config_hash_includes_regrid_backend_signature(tmp_path, monkeypatch):
    """Restart cache hashes must change when effective regrid environment changes."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "regrid_backend": "cdo_remapcon",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={"Runoff": {"SimA_varname": "runoff_sim"}},
            ),
        },
    )()

    def cache_hash():
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    monkeypatch.setattr(local_runner, "_regrid_backend_signature", lambda: {"cdo": {"available": False}})
    without_cdo = cache_hash()
    monkeypatch.setattr(local_runner, "_regrid_backend_signature", lambda: {"cdo": {"available": True}})

    assert cache_hash() != without_cdo


def test_cached_task_requires_all_requested_metric_and_score_outputs(tmp_path):
    """A cache hit with only one requested output file is stale, not reusable."""
    import json

    import openbench.runner.local as local_runner
    from openbench.runner.cache import make_cache_key

    output_dir = tmp_path / "case"
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "scores").mkdir()
    key = make_cache_key("Runoff", "SimA", "TestRef")
    config_hash = "deadbeef"
    (output_dir / ".openbench_cache.json").write_text(json.dumps({key: config_hash}))
    _write_fake_grid_outputs(output_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=())

    result = local_runner._cached_task_result(
        {
            "var_name": "Runoff",
            "sim_source": "SimA",
            "ref_source": "TestRef",
            "cache_key": key,
            "config_hash": config_hash,
            "use_cache": True,
            "cache_dir": str(output_dir),
            "metric_vars": ["bias", "RMSE"],
            "score_vars": ["Overall_Score"],
        }
    )

    assert result is None


def test_cached_task_rejects_unreadable_requested_netcdf_output(tmp_path):
    """A cache hit with corrupt requested NetCDF output is stale, not reusable."""
    import json

    import openbench.runner.local as local_runner
    from openbench.runner.cache import make_cache_key

    output_dir = tmp_path / "case"
    (output_dir / "metrics").mkdir(parents=True)
    key = make_cache_key("Runoff", "SimA", "TestRef")
    config_hash = "deadbeef"
    (output_dir / ".openbench_cache.json").write_text(json.dumps({key: config_hash}))
    (output_dir / "metrics" / "Runoff_ref_TestRef_sim_SimA_bias.nc").write_text("not a netcdf")

    result = local_runner._cached_task_result(
        {
            "var_name": "Runoff",
            "sim_source": "SimA",
            "ref_source": "TestRef",
            "cache_key": key,
            "config_hash": config_hash,
            "use_cache": True,
            "cache_dir": str(output_dir),
            "metric_vars": ["bias"],
            "score_vars": [],
        }
    )

    assert result is None


def test_comparison_only_requires_all_requested_metric_and_score_outputs(tmp_path):
    """Comparison-only mode must reject partial evaluation output sets."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "scores").mkdir()
    _write_fake_grid_outputs(output_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=())

    errors = local_runner._validate_comparison_only_inputs(
        output_dir,
        [
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "TestRef",
                "metric_vars": ["bias", "RMSE"],
                "score_vars": ["Overall_Score"],
            }
        ],
    )

    assert errors
    assert errors[0]["phase"] == "preflight"
    assert "missing prerequisite outputs" in errors[0]["message"]


def test_comparison_only_rejects_unreadable_requested_netcdf_output(tmp_path):
    """Comparison-only mode must reject corrupt pre-existing evaluation outputs."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "metrics" / "Runoff_ref_TestRef_sim_SimA_bias.nc").write_text("not a netcdf")

    errors = local_runner._validate_comparison_only_inputs(
        output_dir,
        [
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "TestRef",
                "metric_vars": ["bias"],
                "score_vars": [],
            }
        ],
    )

    assert errors
    assert errors[0]["phase"] == "preflight"
    assert "missing prerequisite outputs" in errors[0]["message"]
    assert str(output_dir / "metrics" / "Runoff_ref_TestRef_sim_SimA_bias.nc") in errors[0]["missing_outputs"]


def test_comparison_only_rejects_missing_second_reference_outputs(tmp_path):
    """A complete RefA output set must not satisfy a missing RefB task."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    _write_fake_grid_outputs(
        output_dir,
        "Runoff",
        "RefA",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    errors = local_runner._validate_comparison_only_inputs(
        output_dir,
        [
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "RefA",
                "metric_vars": ["bias"],
                "score_vars": ["Overall_Score"],
            },
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "RefB",
                "metric_vars": ["bias"],
                "score_vars": ["Overall_Score"],
            },
        ],
    )

    assert len(errors) == 1
    assert errors[0]["ref"] == "RefB"
    assert all("RefB" in path for path in errors[0]["missing_outputs"])


def test_comparison_only_station_task_requires_station_csv_not_grid_nc(tmp_path):
    """Station comparison-only tasks must not be satisfied by grid-shaped NC outputs."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    _write_fake_grid_outputs(
        output_dir,
        "Runoff",
        "RefA",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    errors = local_runner._validate_comparison_only_inputs(
        output_dir,
        [
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "RefA",
                "metric_vars": ["bias"],
                "score_vars": ["Overall_Score"],
                "ref_data_type": "grid",
                "sim_data_type": "stn",
            }
        ],
    )

    assert len(errors) == 1
    assert all(path.endswith("_evaluations.csv") for path in errors[0]["missing_outputs"])


def test_comparison_only_station_task_accepts_complete_station_csv_outputs(tmp_path):
    """Station comparison-only tasks should pass when metrics and scores CSVs are present."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    _write_fake_station_outputs(
        output_dir,
        "Runoff",
        "RefA",
        "SimA",
        metrics=True,
        scores=True,
    )

    errors = local_runner._validate_comparison_only_inputs(
        output_dir,
        [
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "RefA",
                "metric_vars": ["bias"],
                "score_vars": ["Overall_Score"],
                "ref_data_type": "grid",
                "sim_data_type": "stn",
            }
        ],
    )

    assert errors == []


def test_comparison_only_exact_outputs_do_not_match_prefix_overlap_sources(tmp_path):
    """RefA/SimA tasks must not reuse complete outputs for RefAB/SimAB."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    _write_fake_grid_outputs(
        output_dir,
        "Runoff",
        "RefAB",
        "SimAB",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    errors = local_runner._validate_comparison_only_inputs(
        output_dir,
        [
            {
                "var_name": "Runoff",
                "sim_source": "SimA",
                "ref_source": "RefA",
                "metric_vars": ["bias"],
                "score_vars": ["Overall_Score"],
            }
        ],
    )

    assert len(errors) == 1
    assert all("RefA_sim_SimA" in path for path in errors[0]["missing_outputs"])


def test_comparison_only_preflight_requires_complete_requested_outputs(tmp_path, monkeypatch):
    """Dry-run comparison-only preflight should use the same full-output check."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    cfg.metrics = ["bias", "RMSE"]
    cfg.scores = ["Overall_Score"]
    output_dir = tmp_path / "case"
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "scores").mkdir()
    _write_fake_grid_outputs(output_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=())

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias", "RMSE"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
        },
    )()

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)

    errors = local_runner.comparison_only_preflight_errors(cfg)

    assert errors
    assert errors[0]["phase"] == "preflight"
    assert "missing prerequisite outputs" in errors[0]["message"]


def test_fully_cached_variable_skips_preprocessing_before_cache_hit(tmp_path, monkeypatch):
    """A complete variable cache hit should not touch source data before skipping."""
    import json

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache, make_cache_key

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    output_dir = tmp_path / "case"
    _write_fake_grid_outputs(output_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=("Overall_Score",))

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "unified_mask": True,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": False,
            "weight": "area",
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_varname": "runoff_ref"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_varname": "runoff_sim"}},
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": namelists,
            "figures": adapter.LegacyFigureConfig(raw={}),
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(output_dir),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
        },
    )()
    cache_hash = EvaluationCache.hash_config(
        local_runner._task_hash_payload(
            cfg=cfg,
            bindings=bindings,
            var_name="Runoff",
            sim_source="SimA",
            ref_source="TestRef",
            metric_vars=["bias"],
            score_vars=["Overall_Score"],
            comparison_vars=[],
            statistic_vars=[],
        )
    )
    (output_dir / ".openbench_cache.json").write_text(
        json.dumps({make_cache_key("Runoff", "SimA", "TestRef"): cache_hash})
    )

    class ExplodingProcessor:
        def __init__(self, info):
            raise AssertionError("cache hit should skip preprocessing")

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(processing, "DatasetProcessing", ExplodingProcessor)
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: (_ for _ in ()).throw(AssertionError("cache hit should skip evaluation")),
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert result["evaluated"] == [
        {"variable": "Runoff", "sim": "SimA", "ref": "TestRef", "status": "success", "skipped": True}
    ]


def test_force_reruns_and_refreshes_cache_entry(tmp_path, monkeypatch):
    """force=True should bypass cache reads but still refresh cache after successful recompute."""
    import json

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache, make_cache_key

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    output_dir = tmp_path / "case"
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "unified_mask": False,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": False,
            "weight": "area",
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_varname": "runoff_ref"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_varname": "runoff_sim"}},
    )

    class Bindings:
        def __init__(self):
            self.runner_cfg = runner_cfg
            self.namelists = namelists
            self.figures = adapter.LegacyFigureConfig(raw={})

        def iter_task_sources(self, variables):
            return [adapter.EvaluationSource("Runoff", "SimA", "TestRef")]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {
                "casedir": str(output_dir),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            }

        def build_evaluation_fig_nml(self):
            return type("Fig", (), {"to_fig_nml": lambda self: {}})()

        def has_grid_evaluation(self, variables):
            return adapter.GridEvaluationEvidence(False)

    bindings = Bindings()
    expected_hash = EvaluationCache.hash_config(
        local_runner._task_hash_payload(
            cfg=cfg,
            bindings=bindings,
            var_name="Runoff",
            sim_source="SimA",
            ref_source="TestRef",
            metric_vars=["bias"],
            score_vars=["Overall_Score"],
            comparison_vars=[],
            statistic_vars=[],
        )
    )
    key = make_cache_key("Runoff", "SimA", "TestRef")
    output_dir.mkdir(parents=True)
    (output_dir / ".openbench_cache.json").write_text(json.dumps({key: "stalehash"}))
    calls = []

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            calls.append(("preprocess", datasource))

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            calls.append(("evaluate", self.info["sim_source"]))
            _write_fake_grid_outputs(
                self.info["casedir"],
                "Runoff",
                self.info["ref_source"],
                self.info["sim_source"],
                metrics=("bias",),
                scores=("Overall_Score",),
            )

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "success", result.get("errors")
    assert ("preprocess", "ref") in calls
    assert ("preprocess", "sim") in calls
    assert ("evaluate", "SimA") in calls
    assert json.loads((output_dir / ".openbench_cache.json").read_text())[key] == expected_hash


def test_only_drawing_uses_only_drawing_evaluators_and_skips_preprocess(tmp_path, monkeypatch):
    """project.only_drawing should re-render from existing outputs, not preprocess/recompute."""
    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation_module
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    cfg.project.only_drawing = True
    _write_fake_grid_outputs(
        tmp_path / "case",
        "Runoff",
        "TestRef",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": True,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": True,
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
    )

    Bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": namelists,
            "figures": adapter.LegacyFigureConfig(raw={}),
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
            "build_evaluation_fig_nml": lambda self: type("Fig", (), {"to_fig_nml": lambda self: {}})(),
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
        },
    )

    calls = []

    class FakeOnlyDrawingGrid:
        def __init__(self, info, fig_nml):
            calls.append(("only", info["ref_source"], info["sim_source"]))

        def make_Evaluation(self):
            calls.append(("draw",))

    class ExplodingProcessor:
        def __init__(self, info):
            raise AssertionError("only_drawing must not preprocess data")

    class ExplodingGrid:
        def __init__(self, info, fig_nml):
            raise AssertionError("only_drawing must not use compute evaluator")

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(processing, "DatasetProcessing", ExplodingProcessor)
    monkeypatch.setattr(evaluation_module, "Evaluation_grid", ExplodingGrid)
    monkeypatch.setattr(only_drawing_module, "Evaluation_grid_only_drawing", FakeOnlyDrawingGrid)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "success", result.get("errors")
    assert calls == [("only", "TestRef", "SimA"), ("draw",)]


def test_only_drawing_requires_existing_outputs_before_rendering(tmp_path, monkeypatch):
    """only_drawing must not report success or call visualizers when outputs are missing."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    cfg.project.only_drawing = True

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": True,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": True,
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
    )

    class Bindings:
        def __init__(self):
            self.runner_cfg = runner_cfg
            self.namelists = namelists
            self.figures = adapter.LegacyFigureConfig(raw={})

        def iter_task_sources(self, variables):
            return [adapter.EvaluationSource("Runoff", "SimA", "TestRef")]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            }

        def build_evaluation_fig_nml(self):
            return type("Fig", (), {"to_fig_nml": lambda self: {}})()

        def has_grid_evaluation(self, variables):
            return adapter.GridEvaluationEvidence(False)

    class ExplodingOnlyDrawingGrid:
        def __init__(self, info, fig_nml):
            raise AssertionError("only_drawing must preflight outputs before rendering")

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(only_drawing_module, "Evaluation_grid_only_drawing", ExplodingOnlyDrawingGrid)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "error"
    assert result["evaluated"] == []
    assert any("missing prerequisite outputs" in err["message"] for err in result["errors"])


def test_only_drawing_skips_statistics_recomputation(tmp_path, monkeypatch):
    """only_drawing should not run core statistics, which recomputes and rewrites stats outputs."""
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    cfg = _make_cfg(tmp_path, comparison_enabled=False, statistics_enabled=True)
    cfg.project.only_drawing = True
    _write_fake_grid_outputs(
        tmp_path / "case",
        "Runoff",
        "TestRef",
        "SimA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": True,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": True,
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
    )

    class Bindings:
        def __init__(self):
            self.runner_cfg = runner_cfg
            self.namelists = namelists
            self.figures = adapter.LegacyFigureConfig(raw={})

        def iter_task_sources(self, variables):
            return [adapter.EvaluationSource("Runoff", "SimA", "TestRef")]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            }

        def build_evaluation_fig_nml(self):
            return type("Fig", (), {"to_fig_nml": lambda self: {}})()

        def has_grid_evaluation(self, variables):
            return adapter.GridEvaluationEvidence(True)

    class FakeOnlyDrawingGrid:
        def __init__(self, info, fig_nml):
            pass

        def make_Evaluation(self):
            pass

    statistics_calls = []

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(only_drawing_module, "Evaluation_grid_only_drawing", FakeOnlyDrawingGrid)
    monkeypatch.setattr(local_runner, "_run_statistics", lambda *args, **kwargs: statistics_calls.append(args) or [])
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "success", result.get("errors")
    assert statistics_calls == []


def test_cz_groupby_only_drawing_reuses_existing_csv_without_recomputing(tmp_path, monkeypatch):
    """CZ only_drawing should render from existing groupby CSVs, not remap/recompute NetCDF outputs."""
    from pathlib import Path

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    group_dir = case_dir / "comparisons" / "CZ_groupby" / "SimA___RefA"
    group_dir.mkdir(parents=True)
    metrics_csv = group_dir / "Runoff_SimA___RefA_metrics.csv"
    scores_csv = group_dir / "Runoff_SimA___RefA_scores.csv"
    metrics_csv.write_text("ID\t1\tAll\nFullName\tAf\tOverall\nbias\t1.0\t1.0\n")
    scores_csv.write_text("ID\t1\tAll\nFullName\tAf\tOverall\nOverall_Score\t0.8\t0.8\n")

    calls = []

    def fake_heatmap(path, selected, data_type, option):
        calls.append((Path(path).name, tuple(selected), data_type, option.get("groupby")))

    def exploding_open_dataset(*args, **kwargs):
        raise AssertionError("CZ only_drawing must not open/remap Climate_zone or metric NetCDF inputs")

    monkeypatch.setattr(only_drawing_module, "make_CZ_based_heat_map", fake_heatmap)
    monkeypatch.setattr(only_drawing_module.xr, "open_dataset", exploding_open_dataset)

    renderer = only_drawing_module.CZ_groupby_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=["bias"],
    )
    renderer.scenarios_CZ_groupby_comparison(
        str(case_dir),
        {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        {
            "general": {"Runoff_ref_source": "RefA"},
            "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
        },
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        {},
    )

    assert calls == [
        ("Runoff_SimA___RefA_metrics.csv", ("bias",), "metric", "CZ_groupby"),
        ("Runoff_SimA___RefA_scores.csv", ("Overall_Score",), "score", "CZ_groupby"),
    ]
    assert not (case_dir / "comparisons" / "CZ_groupby" / "CZ_remap.nc").exists()


def test_cz_groupby_only_drawing_prefers_safe_groupby_names(tmp_path, monkeypatch):
    """CZ only_drawing should resolve safe groupby names when source names contain path separators."""
    from pathlib import Path

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module
    from openbench.util.filenames import groupby_pair_dirname, groupby_table_filename

    case_dir = tmp_path / "case"
    evaluation_item = "Run/off"
    sim_source = "Sim/A___B"
    ref_source = "Ref:C*"
    group_dir = case_dir / "comparisons" / "CZ_groupby" / groupby_pair_dirname(sim_source, ref_source)
    group_dir.mkdir(parents=True)
    scores_csv = group_dir / groupby_table_filename(evaluation_item, sim_source, ref_source, "scores")
    scores_csv.write_text("ID\t1\tAll\nFullName\tAf\tOverall\nOverall_Score\t0.8\t0.8\n")

    calls = []

    def fake_heatmap(path, selected, data_type, option):
        calls.append((Path(path), tuple(selected), data_type, option.get("path"), option.get("item")))

    monkeypatch.setattr(only_drawing_module, "make_CZ_based_heat_map", fake_heatmap)

    renderer = only_drawing_module.CZ_groupby_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )
    renderer.scenarios_CZ_groupby_comparison(
        str(case_dir),
        {
            "general": {f"{evaluation_item}_sim_source": [sim_source]},
            evaluation_item: {f"{sim_source}_data_type": "grid"},
        },
        {
            "general": {f"{evaluation_item}_ref_source": ref_source},
            evaluation_item: {f"{ref_source}_data_type": "grid", f"{ref_source}_varname": "runoff_ref"},
        },
        [evaluation_item],
        ["Overall_Score"],
        [],
        {},
    )

    assert calls == [
        (
            scores_csv,
            ("Overall_Score",),
            "score",
            str(group_dir),
            [evaluation_item, sim_source, ref_source],
        )
    ]


def test_cz_groupby_only_drawing_missing_requested_table_fails_fast(tmp_path, monkeypatch):
    """Requested only_drawing groupby outputs should fail instead of warning and silently succeeding."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    monkeypatch.setattr(
        only_drawing_module,
        "make_CZ_based_heat_map",
        lambda *args, **kwargs: pytest.fail("missing producer table should fail before rendering"),
    )

    renderer = only_drawing_module.CZ_groupby_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(FileNotFoundError, match="only_drawing missing required file"):
        renderer.scenarios_CZ_groupby_comparison(
            str(tmp_path / "case"),
            {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_cz_groupby_score_only_sets_item_and_does_not_log_metric_error(tmp_path, monkeypatch, caplog):
    """CZ score-only full runs should not depend on the metrics branch to set renderer context."""
    import logging

    import numpy as np
    import xarray as xr

    import openbench.core.climatezone_groupby as cz_module

    case_dir = tmp_path / "case"
    static_dir = tmp_path / "static"
    (case_dir / "scores").mkdir(parents=True)
    static_dir.mkdir()

    lat = np.array([0.25, 0.75])
    lon = np.array([0.25, 0.75])
    xr.Dataset(
        {"climate_zone": (("lat", "lon"), np.ones((2, 2), dtype=np.int32))},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(static_dir / "Climate_zone.nc")
    xr.Dataset(
        {"Overall_Score": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(case_dir / "scores" / "Runoff_ref_RefA_sim_SimA_Overall_Score.nc")

    calls = []
    monkeypatch.setenv("OPENBENCH_DATASET_DIR", str(static_dir))
    monkeypatch.setattr(
        cz_module,
        "make_CZ_based_heat_map",
        lambda path, selected, data_type, option: calls.append((path, tuple(selected), data_type, dict(option))),
    )

    main_nml = {
        "general": {
            "basedir": str(tmp_path),
            "basename": "case",
            "compare_grid_res": 0.5,
            "compare_tim_res": "Month",
            "min_lat": 0,
            "max_lat": 1,
            "min_lon": 0,
            "max_lon": 1,
        }
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "RefA"},
        "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
    }

    caplog.set_level(logging.ERROR)
    cz_module.CZ_groupby(main_nml, scores=["Overall_Score"], metrics=[]).scenarios_CZ_groupby_comparison(
        str(case_dir),
        sim_nml,
        ref_nml,
        ["Runoff"],
        ["Overall_Score"],
        [],
        {},
    )

    assert calls
    assert calls[0][2] == "score"
    assert calls[0][3]["item"] == ["Runoff", "SimA", "RefA"]
    assert "No scores for climate zone class comparison" not in caplog.text


@pytest.mark.parametrize(
    ("groupby_name", "static_filename", "static_var", "method_name", "active_classes"),
    [
        ("CZ_groupby", "Climate_zone.nc", "climate_zone", "scenarios_CZ_groupby_comparison", (1, 2)),
        ("IGBP_groupby", "IGBP.nc", "IGBP", "scenarios_IGBP_groupby_comparison", (1, 2)),
        ("PFT_groupby", "PFT.nc", "PFT", "scenarios_PFT_groupby_comparison", (0, 1)),
    ],
)
@pytest.mark.parametrize("weight", ["area", "mass"])
def test_lc_cz_groupby_score_weighting_and_empty_classes_are_consistent(
    tmp_path,
    monkeypatch,
    groupby_name,
    static_filename,
    static_var,
    method_name,
    active_classes,
    weight,
):
    """CZ/IGBP/PFT score tables should use the same masked weighted-mean semantics."""
    import numpy as np
    import xarray as xr

    from openbench.core.climatezone_groupby import CZ_groupby
    from openbench.core.landcover_groupby import LC_groupby
    from openbench.util.filenames import groupby_class_netcdf_stem, groupby_pair_dirname, groupby_table_filename

    case_dir = tmp_path / "case"
    static_dir = tmp_path / "static"
    (case_dir / "scores").mkdir(parents=True)
    (case_dir / "metrics").mkdir()
    (case_dir / "data").mkdir()
    static_dir.mkdir()

    lat = np.array([0.25, 0.75])
    lon = np.array([0.25, 0.75])
    class_values = np.array([[active_classes[0], active_classes[1]], [active_classes[0], active_classes[1]]])
    score_values = np.array([[1.0, 10.0], [3.0, np.nan]])
    constant_values = np.array([[7.0, np.nan], [7.0, np.nan]])
    all_nan_values = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    mass_ref_values = np.array([[[1.0, 9.0], [2.0, 4.0]]])

    xr.Dataset({static_var: (("lat", "lon"), class_values)}, coords={"lat": lat, "lon": lon}).to_netcdf(
        static_dir / static_filename
    )
    for score_name, values in {
        "Overall_Score": score_values,
        "Constant_Score": constant_values,
        "All_NaN_Score": all_nan_values,
    }.items():
        xr.Dataset({score_name: (("lat", "lon"), values)}, coords={"lat": lat, "lon": lon}).to_netcdf(
            case_dir / "scores" / f"Runoff_ref_RefA_sim_SimA_{score_name}.nc"
        )
    xr.Dataset({"bias": (("lat", "lon"), score_values)}, coords={"lat": lat, "lon": lon}).to_netcdf(
        case_dir / "metrics" / "Runoff_ref_RefA_sim_SimA_bias.nc"
    )
    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), mass_ref_values)},
        coords={"time": [0], "lat": lat, "lon": lon},
    ).to_netcdf(case_dir / "data" / "Runoff_ref_RefA_runoff_ref.nc")

    monkeypatch.setenv("OPENBENCH_DATASET_DIR", str(static_dir))
    monkeypatch.setattr("openbench.core.climatezone_groupby.make_CZ_based_heat_map", lambda *args, **kwargs: None)
    monkeypatch.setattr("openbench.core.landcover_groupby.make_LC_based_heat_map", lambda *args, **kwargs: None)

    main_nml = {
        "general": {
            "basedir": str(tmp_path),
            "basename": "case",
            "compare_grid_res": 0.5,
            "compare_tim_res": "Month",
            "min_lat": 0,
            "max_lat": 1,
            "min_lon": 0,
            "max_lon": 1,
            "weight": weight,
        }
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "RefA"},
        "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
    }

    score_names = ["Overall_Score", "Constant_Score", "All_NaN_Score"]
    metric_names = ["bias"]
    groupby = CZ_groupby(main_nml, scores=score_names, metrics=metric_names)
    if groupby_name != "CZ_groupby":
        groupby = LC_groupby(main_nml, scores=score_names, metrics=metric_names)
    getattr(groupby, method_name)(str(case_dir), sim_nml, ref_nml, ["Runoff"], score_names, metric_names, {})

    table_path = (
        case_dir
        / "comparisons"
        / groupby_name
        / groupby_pair_dirname("SimA", "RefA")
        / groupby_table_filename("Runoff", "SimA", "RefA", "scores")
    )
    assert table_path.exists()
    assert not (case_dir / "comparisons" / groupby_name / "SimA___RefA" / "Runoff_SimA___RefA_scores.csv").exists()
    class_prefix = {"CZ_groupby": "CZ", "IGBP_groupby": "IGBP", "PFT_groupby": "PFT"}[groupby_name]
    class_count = {"CZ": 30, "IGBP": 17, "PFT": 16}[class_prefix]
    score_stem = groupby_class_netcdf_stem("Runoff", "RefA", "SimA", "Overall_Score", class_prefix)
    bundle_path = table_path.parent / f"{score_stem}__classes.nc"
    assert bundle_path.exists()
    assert len(list(table_path.parent.glob(f"{score_stem}__*.nc"))) == 1
    with xr.open_dataset(bundle_path) as bundle:
        assert "class" in bundle.dims
        assert bundle.sizes["class"] == class_count
    metric_stem = groupby_class_netcdf_stem("Runoff", "RefA", "SimA", "bias", class_prefix)
    metric_bundle_path = table_path.parent / f"{metric_stem}__classes.nc"
    assert metric_bundle_path.exists()
    assert len(list(table_path.parent.glob(f"{metric_stem}__*.nc"))) == 1
    with xr.open_dataset(metric_bundle_path) as bundle:
        assert "class" in bundle.dims
        assert bundle.sizes["class"] == class_count

    rows = [line.split("\t") for line in table_path.read_text().splitlines()[1:]]
    weighted_row, constant_row, all_nan_row = rows
    area_weights = np.cos(np.deg2rad(lat))[:, None]
    if weight == "area":
        weights = np.broadcast_to(area_weights, score_values.shape)
    else:
        weights = area_weights * mass_ref_values[0]

    finite = np.isfinite(score_values)
    expected_overall = float((score_values[finite] * weights[finite]).sum() / weights[finite].sum())
    first_mask = class_values == active_classes[0]
    first_valid = first_mask & finite
    expected_first = float((score_values[first_valid] * weights[first_valid]).sum() / weights[first_valid].sum())
    second_mask = class_values == active_classes[1]
    second_valid = second_mask & finite
    expected_second = float((score_values[second_valid] * weights[second_valid]).sum() / weights[second_valid].sum())

    assert weighted_row[0] == "Overall_Score"
    assert weighted_row[1] == f"{expected_first:.3f}"
    assert weighted_row[2] == f"{expected_second:.3f}"
    assert weighted_row[3] == "N/A"
    assert weighted_row[-1] == f"{expected_overall:.3f}"

    assert constant_row[0] == "Constant_Score"
    assert constant_row[1] == "7.000"
    assert constant_row[2] == "N/A"
    assert constant_row[3] == "N/A"
    assert constant_row[-1] == "7.000"

    assert all_nan_row[0] == "All_NaN_Score"
    assert all_nan_row[1] == "N/A"
    assert all_nan_row[2] == "N/A"
    assert all_nan_row[3] == "N/A"
    assert all_nan_row[-1] == "N/A"


def test_igbp_groupby_only_drawing_falls_back_to_legacy_root_scores_csv(tmp_path, monkeypatch):
    """IGBP only_drawing should read scores from the old root location when pre-fix runs wrote them there."""
    from pathlib import Path

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    legacy_dir = case_dir / "comparisons" / "IGBP_groupby"
    legacy_dir.mkdir(parents=True)
    legacy_scores = legacy_dir / "Runoff_SimA___RefA_scores.csv"
    legacy_scores.write_text("ID\t1\tAll\nFullName\tevergreen_needleleaf_forest\tOverall\nOverall_Score\t0.8\t0.8\n")

    calls = []

    def fake_heatmap(path, selected, data_type, option):
        calls.append((Path(path), tuple(selected), data_type, option.get("path"), option.get("item")))

    monkeypatch.setattr(only_drawing_module, "make_LC_based_heat_map", fake_heatmap)

    renderer = only_drawing_module.LC_groupby_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )
    renderer.scenarios_IGBP_groupby_comparison(
        str(case_dir),
        {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        {
            "general": {"Runoff_ref_source": "RefA"},
            "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
        },
        ["Runoff"],
        ["Overall_Score"],
        [],
        {},
    )

    assert calls == [
        (
            legacy_scores,
            ("Overall_Score",),
            "score",
            f"{case_dir}/comparisons/IGBP_groupby/SimA__RefA/",
            ["Runoff", "SimA", "RefA"],
        )
    ]


def test_comparison_only_drawing_igbp_delegates_to_groupby_fallback(tmp_path, monkeypatch):
    """ComparisonProcessing only_drawing should not keep a divergent IGBP/PFT groupby implementation."""
    from pathlib import Path

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    legacy_dir = case_dir / "comparisons" / "IGBP_groupby"
    legacy_dir.mkdir(parents=True)
    legacy_scores = legacy_dir / "Runoff_SimA___RefA_scores.csv"
    legacy_scores.write_text("ID\t1\tAll\nFullName\tevergreen_needleleaf_forest\tOverall\nOverall_Score\t0.8\t0.8\n")

    calls = []

    def fake_heatmap(path, selected, data_type, option):
        calls.append((Path(path), tuple(selected), data_type, option.get("item")))

    monkeypatch.setattr(only_drawing_module, "make_LC_based_heat_map", fake_heatmap)

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )
    renderer.scenarios_IGBP_groupby_comparison(
        str(case_dir),
        {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        {
            "general": {"Runoff_ref_source": "RefA"},
            "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
        },
        ["Runoff"],
        ["Overall_Score"],
        [],
        {},
    )

    assert calls == [(legacy_scores, ("Overall_Score",), "score", ["Runoff", "SimA", "RefA"])]


def test_core_comparison_groupby_delegates_to_landcover_groupby(tmp_path, monkeypatch):
    """Core ComparisonProcessing should not keep a divergent IGBP/PFT groupby producer implementation."""
    import openbench.core.landcover_groupby as landcover_module
    from openbench.core.comparison import ComparisonProcessing

    calls = []

    class FakeLCGroupby:
        def __init__(self, main_nml, scores, metrics):
            calls.append(("init", main_nml, tuple(scores), tuple(metrics)))

        def scenarios_IGBP_groupby_comparison(
            self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
        ):
            calls.append(("igbp", casedir, evaluation_items, tuple(scores), tuple(metrics), option))

        def scenarios_PFT_groupby_comparison(
            self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
        ):
            calls.append(("pft", casedir, evaluation_items, tuple(scores), tuple(metrics), option))

    monkeypatch.setattr(landcover_module, "LC_groupby", FakeLCGroupby)

    main_nml = {
        "general": {
            "basedir": str(tmp_path),
            "basename": "case",
            "compare_grid_res": 0.5,
            "compare_tim_res": "Month",
        }
    }
    processor = ComparisonProcessing(main_nml, ["Overall_Score"], ["bias"])

    processor.scenarios_IGBP_groupby_comparison(
        str(tmp_path / "case"),
        {"general": {"Runoff_sim_source": ["SimA"]}},
        {"general": {"Runoff_ref_source": "RefA"}},
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        {"title": "igbp"},
    )
    processor.scenarios_PFT_groupby_comparison(
        str(tmp_path / "case"),
        {"general": {"Runoff_sim_source": ["SimA"]}},
        {"general": {"Runoff_ref_source": "RefA"}},
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        {"title": "pft"},
    )

    assert calls == [
        ("init", main_nml, ("Overall_Score",), ("bias",)),
        ("igbp", str(tmp_path / "case"), ["Runoff"], ("Overall_Score",), ("bias",), {"title": "igbp"}),
        ("init", main_nml, ("Overall_Score",), ("bias",)),
        ("pft", str(tmp_path / "case"), ["Runoff"], ("Overall_Score",), ("bias",), {"title": "pft"}),
    ]


def test_parallel_coordinates_only_drawing_does_not_write_remove_nan_csv(tmp_path, monkeypatch):
    """Parallel Coordinates only_drawing should not create sanitized intermediate CSVs."""
    from pathlib import Path

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    comparison_dir = case_dir / "comparisons" / "Parallel_Coordinates"
    comparison_dir.mkdir(parents=True)
    source_csv = comparison_dir / "Parallel_Coordinates_evaluations.csv"
    source_csv.write_text(
        "Item\tReference\tSimulation\tOverall_Score\tbias\nRunoff\tRefA\tSimA\t0.8\t1.0\nRunoff\tRefA\tSimB\t0.7\t2.0\n"
    )
    remove_nan_csv = comparison_dir / "Parallel_Coordinates_evaluations_remove_nan.csv"

    calls = []

    def fake_parallel_coordinates(path, basedir, evaluation_items, scores, metrics, option):
        calls.append((Path(path).name, tuple(scores), tuple(metrics)))

    monkeypatch.setattr(
        only_drawing_module,
        "make_scenarios_comparison_parallel_coordinates",
        fake_parallel_coordinates,
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=["bias"],
    )
    renderer.scenarios_Parallel_Coordinates_comparison(
        str(case_dir),
        {
            "general": {"Runoff_sim_source": ["SimA", "SimB"]},
            "Runoff": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
        },
        {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        {},
    )

    assert calls == [("Parallel_Coordinates_evaluations.csv", ("Overall_Score",), ("bias",))]
    assert not remove_nan_csv.exists()


def test_parallel_coordinates_only_drawing_ignores_stale_remove_nan_csv(tmp_path, monkeypatch):
    """only_drawing must render from canonical CSV, not a stale sanitized artifact from an older run."""
    from pathlib import Path

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    comparison_dir = case_dir / "comparisons" / "Parallel_Coordinates"
    comparison_dir.mkdir(parents=True)
    (comparison_dir / "Parallel_Coordinates_evaluations.csv").write_text(
        "Item\tReference\tSimulation\tOverall_Score\tbias\nRunoff\tRefA\tSimA\t0.8\t1.0\n"
    )
    (comparison_dir / "Parallel_Coordinates_evaluations_remove_nan.csv").write_text(
        "Item\tReference\tSimulation\tOldScore\nRunoff\tRefA\tSimA\t0.1\n"
    )

    calls = []

    def fake_parallel_coordinates(path, basedir, evaluation_items, scores, metrics, option):
        calls.append((Path(path).name, tuple(scores), tuple(metrics)))

    monkeypatch.setattr(
        only_drawing_module,
        "make_scenarios_comparison_parallel_coordinates",
        fake_parallel_coordinates,
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=["bias"],
    )
    renderer.scenarios_Parallel_Coordinates_comparison(
        str(case_dir),
        {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        {},
    )

    assert calls == [("Parallel_Coordinates_evaluations.csv", ("Overall_Score",), ("bias",))]


def test_statistical_comparison_only_drawing_reports_missing_precomputed_output(tmp_path):
    """Statistical comparison only_drawing should error when its full-run NC output is absent."""
    import openbench.config.adapter as adapter_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {"only_drawing": True}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={
                        "general": {
                            "basename": "case",
                            "basedir": str(tmp_path.parent),
                            "compare_grid_res": 0.5,
                            "compare_tim_res": "Month",
                        }
                    },
                    reference={
                        "general": {"Runoff_ref_source": "RefA"},
                        "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
                    },
                    simulation={
                        "general": {"Runoff_sim_source": ["SimA"]},
                        "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
                    },
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={"Standard_Deviation": {}},
            )

        def iter_task_sources(self, variables):
            return [adapter_module.EvaluationSource("Runoff", "SimA", "RefA")]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {"ref_data_type": "grid", "sim_data_type": "grid"}

    errors = local_runner._run_comparison(Bindings(), ["Standard_Deviation"], output_dir)

    assert errors
    assert errors[0]["phase"] == "comparison"
    assert "Standard_Deviation comparison failed" in errors[0]["message"]


@pytest.mark.parametrize("comparison_item", ["HeatMap", "RadarMap", "Taylor_Diagram", "Target_Diagram"])
def test_comparison_only_drawing_reports_missing_intermediate_csv(tmp_path, comparison_item):
    """Comparison only_drawing should error when the full-run comparison CSV is absent."""
    import openbench.config.adapter as adapter_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {"only_drawing": True}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={
                        "general": {
                            "basename": "case",
                            "basedir": str(tmp_path.parent),
                            "compare_grid_res": 0.5,
                            "compare_tim_res": "Month",
                        }
                    },
                    reference={
                        "general": {"Runoff_ref_source": "RefA"},
                        "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
                    },
                    simulation={
                        "general": {"Runoff_sim_source": ["SimA"]},
                        "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
                    },
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={comparison_item: {}},
            )

        def iter_task_sources(self, variables):
            return [adapter_module.EvaluationSource("Runoff", "SimA", "RefA")]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {"ref_data_type": "grid", "sim_data_type": "grid"}

    errors = local_runner._run_comparison(Bindings(), [comparison_item], output_dir)

    assert errors
    assert errors[0]["phase"] == "comparison"
    assert f"{comparison_item} comparison failed" in errors[0]["message"]


def test_only_drawing_missing_file_errors_include_actionable_context(tmp_path):
    """only_drawing missing intermediates should name the mode, path, and full-run remedy."""
    import openbench.config.adapter as adapter_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {"only_drawing": True}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={
                        "general": {
                            "basename": "case",
                            "basedir": str(tmp_path.parent),
                            "compare_grid_res": 0.5,
                            "compare_tim_res": "Month",
                        }
                    },
                    reference={
                        "general": {"Runoff_ref_source": "RefA"},
                        "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
                    },
                    simulation={
                        "general": {"Runoff_sim_source": ["SimA"]},
                        "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
                    },
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={"Taylor_Diagram": {}},
            )

        def iter_task_sources(self, variables):
            return [adapter_module.EvaluationSource("Runoff", "SimA", "RefA")]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {"ref_data_type": "grid", "sim_data_type": "grid"}

    errors = local_runner._run_comparison(Bindings(), ["Taylor_Diagram"], output_dir)

    assert errors
    message = errors[0]["message"]
    assert "only_drawing" in message
    assert "missing required file" in message
    assert "taylor_diagram__Runoff__RefA.csv" in message
    assert "only_drawing=False" in message


@pytest.mark.parametrize(
    ("comparison_method", "plot_func"),
    [
        ("scenarios_Whisker_Plot_comparison", "make_scenarios_comparison_Whisker_Plot"),
        ("scenarios_Ridgeline_Plot_comparison", "make_scenarios_comparison_Ridgeline_Plot"),
    ],
)
def test_distribution_only_drawing_reports_missing_grid_evaluation_input(
    tmp_path,
    monkeypatch,
    comparison_method,
    plot_func,
):
    """Distribution only_drawing should not silently plot a subset when one sim output is missing."""
    import xarray as xr

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    scores_dir = case_dir / "scores"
    scores_dir.mkdir(parents=True)
    xr.Dataset({"Overall_Score": ("sample", [0.8])}).to_netcdf(scores_dir / "Runoff_ref_RefA_sim_SimA_Overall_Score.nc")
    plot_calls = []
    monkeypatch.setattr(
        only_drawing_module,
        plot_func,
        lambda *args, **kwargs: plot_calls.append(args),
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(FileNotFoundError):
        getattr(renderer, comparison_method)(
            str(case_dir),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
            },
            {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


def test_kernel_density_only_drawing_reports_missing_station_evaluation_input(tmp_path, monkeypatch):
    """KDE only_drawing should not silently plot a subset when one station sim CSV is missing."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    scores_dir = case_dir / "scores"
    scores_dir.mkdir(parents=True)
    (scores_dir / "Runoff_stn_RefA_SimA_evaluations.csv").write_text("station,Overall_Score\nS1,0.8\n")
    plot_calls = []
    monkeypatch.setattr(
        only_drawing_module,
        "make_scenarios_comparison_Kernel_Density_Estimate",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(FileNotFoundError):
        renderer.scenarios_Kernel_Density_Estimate_comparison(
            str(case_dir),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {"SimA_data_type": "stn", "SimB_data_type": "stn", "SimA_varname": "", "SimB_varname": ""},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "stn", "RefA_varname": ""},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


@pytest.mark.parametrize(
    ("comparison_method", "plot_func"),
    [
        ("scenarios_Kernel_Density_Estimate_comparison", "make_scenarios_comparison_Kernel_Density_Estimate"),
        ("scenarios_Whisker_Plot_comparison", "make_scenarios_comparison_Whisker_Plot"),
        ("scenarios_Ridgeline_Plot_comparison", "make_scenarios_comparison_Ridgeline_Plot"),
    ],
)
def test_only_drawing_distribution_plot_failures_propagate(tmp_path, monkeypatch, comparison_method, plot_func):
    """only_drawing should fail the run when a distribution figure renderer fails."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    scores_dir = case_dir / "scores"
    scores_dir.mkdir(parents=True)
    for sim_source in ("SimA", "SimB"):
        (scores_dir / f"Runoff_stn_RefA_{sim_source}_evaluations.csv").write_text("station,Overall_Score\nS1,0.8\n")

    def fail_plot(*args, **kwargs):
        raise RuntimeError("plot boom")

    monkeypatch.setattr(only_drawing_module, plot_func, fail_plot)
    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(RuntimeError, match="plot boom"):
        getattr(renderer, comparison_method)(
            str(case_dir),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {"SimA_data_type": "stn", "SimB_data_type": "stn", "SimA_varname": "", "SimB_varname": ""},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "stn", "RefA_varname": ""},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


@pytest.mark.parametrize("data_type", ["grid", "stn"])
def test_relative_score_only_drawing_propagates_missing_files(tmp_path, monkeypatch, data_type):
    """Relative Score only_drawing should report missing precomputed relative-score files."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {"SimA_data_type": data_type},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "RefA"},
        "Runoff": {"RefA_data_type": data_type},
    }

    with pytest.raises(FileNotFoundError):
        renderer.scenarios_Relative_Score_comparison(
            str(tmp_path / "case"),
            sim_nml,
            ref_nml,
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_relative_score_only_drawing_plot_failures_propagate(tmp_path, monkeypatch):
    """Relative Score only_drawing should not hide renderer failures after inputs exist."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    relative_dir = case_dir / "comparisons" / "Relative_Score"
    relative_dir.mkdir(parents=True)
    (relative_dir / "Runoff_stn_RefA_SimA_relative_scores.csv").write_text(
        "ID,relative_Overall_Score_SimA,ref_lon,ref_lat\nS1,0.8,100.0,30.0\n"
    )

    def fail_plot(*args, **kwargs):
        raise RuntimeError("relative plot boom")

    monkeypatch.setattr(only_drawing_module, "make_scenarios_comparison_Relative_Score", fail_plot)
    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(RuntimeError, match="relative plot boom"):
        renderer.scenarios_Relative_Score_comparison(
            str(case_dir),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "stn"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "stn"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_relative_score_only_drawing_rejects_nonfinite_grid_input(tmp_path, monkeypatch):
    """Relative Score only_drawing should reject stale all-NaN precomputed grid relative scores."""
    import numpy as np
    import xarray as xr

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    relative_dir = case_dir / "comparisons" / "Relative_Score"
    relative_dir.mkdir(parents=True)
    xr.Dataset(
        {
            "relative_Overall_Score": (
                ("lat", "lon"),
                np.array([[np.nan]]),
            )
        },
        coords={"lat": [30.0], "lon": [100.0]},
    ).to_netcdf(relative_dir / "Runoff_ref_RefA_sim_SimA_RelativeOverall_Score.nc")
    plot_calls = []
    monkeypatch.setattr(
        only_drawing_module,
        "make_scenarios_comparison_Relative_Score",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(ValueError, match="no finite data"):
        renderer.scenarios_Relative_Score_comparison(
            str(case_dir),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


def test_portrait_only_drawing_rejects_nonfinite_summary_csv(tmp_path, monkeypatch):
    """Portrait only_drawing should not render from a stale all-NaN summary CSV."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    portrait_dir = case_dir / "comparisons" / "Portrait_Plot_seasonal"
    portrait_dir.mkdir(parents=True)
    (portrait_dir / "Portrait_Plot_seasonal.csv").write_text(
        "Item\tReference\tSimulation\tbias_DJF\tbias_MAM\tbias_JJA\tbias_SON\nRunoff\tRefA\tSimA\tnan\tnan\tnan\tnan\n"
    )
    plot_calls = []
    monkeypatch.setattr(
        only_drawing_module,
        "make_scenarios_comparison_Portrait_Plot_seasonal",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=[],
        metrics=["bias"],
    )

    with pytest.raises(ValueError, match="no finite data"):
        renderer.scenarios_Portrait_Plot_seasonal_comparison(
            str(case_dir),
            {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
            {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
            ["Runoff"],
            [],
            ["bias"],
            {},
        )

    assert plot_calls == []


def test_diff_plot_only_drawing_reports_missing_precomputed_input(tmp_path, monkeypatch):
    """Diff Plot only_drawing should preflight expected anomaly/diff files before renderer execution."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    (case_dir / "comparisons" / "Diff_Plot").mkdir(parents=True)
    plot_calls = []
    monkeypatch.setattr(
        only_drawing_module,
        "make_scenarios_comparison_Diff_Plot",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(FileNotFoundError, match="Overall_Score_anomaly"):
        renderer.scenarios_Diff_Plot_comparison(
            str(case_dir),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {
                    "SimA_data_type": "grid",
                    "SimA_varname": "runoff_sim",
                    "SimA_varunit": "mm",
                    "SimB_data_type": "grid",
                    "SimB_varname": "runoff_sim",
                    "SimB_varunit": "mm",
                },
            },
            {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


def test_basic_only_drawing_reports_missing_precomputed_grid_input(tmp_path, monkeypatch):
    """Basic only_drawing should fail fast when the requested precomputed grid map is absent."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    case_dir = tmp_path / "case"
    (case_dir / "comparisons" / "Mean").mkdir(parents=True)
    plot_calls = []
    monkeypatch.setattr(only_drawing_module, "make_geo_plot_index", lambda *args, **kwargs: plot_calls.append(args))

    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=[],
        metrics=["Mean"],
    )

    with pytest.raises(FileNotFoundError, match="Runoff_ref_RefA_runoff_ref_Mean.nc"):
        renderer.scenarios_Basic_comparison(
            str(case_dir),
            {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            [],
            ["Mean"],
            {"key": "Mean"},
        )

    assert plot_calls == []


def test_correlation_only_drawing_fails_fast_for_station_sources(tmp_path, monkeypatch):
    """Requested Correlation only_drawing should not warn/continue when inputs are unsupported."""
    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    plot_calls = []
    monkeypatch.setattr(only_drawing_module, "make_Correlation", lambda *args, **kwargs: plot_calls.append(args))
    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=[],
        metrics=["Correlation"],
    )

    with pytest.raises(ValueError, match="Correlation only_drawing cannot render requested figure"):
        renderer.scenarios_Correlation_comparison(
            str(tmp_path / "case"),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {"SimA_data_type": "stn", "SimB_data_type": "grid"},
            },
            {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
            ["Runoff"],
            [],
            ["Correlation"],
            {},
        )

    assert plot_calls == []


def test_only_drawing_statistical_plot_failures_propagate(tmp_path, monkeypatch):
    """Statistical only_drawing should not report success when a figure renderer fails."""
    import numpy as np
    import xarray as xr

    import openbench.visualization.Mod_Only_Drawing as only_drawing_module

    output_file = (
        tmp_path / "case" / "comparisons" / "Standard_Deviation" / "Standard_Deviation_Runoff_sim_SimA_runoff_sim.nc"
    )
    output_file.parent.mkdir(parents=True)
    xr.Dataset({"Standard_Deviation": ("sample", np.array([1.0]))}).to_netcdf(output_file)

    def fail_plot(*args, **kwargs):
        raise RuntimeError("standard deviation plot boom")

    monkeypatch.setattr(only_drawing_module, "make_Standard_Deviation", fail_plot)
    renderer = only_drawing_module.ComparisonProcessing_only_drawing(
        {
            "general": {
                "basedir": str(tmp_path),
                "basename": "case",
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
            }
        },
        scores=["Overall_Score"],
        metrics=[],
    )

    with pytest.raises(RuntimeError, match="standard deviation plot boom"):
        renderer.scenarios_Standard_Deviation_comparison(
            str(tmp_path / "case"),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_run_evaluation_closes_optional_dask_on_success(tmp_path, monkeypatch):
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    handle = (object(), object())
    calls = []

    def start(num_cores, *, only_drawing, comparison_only, local_directory, dask_config, station_heavy):
        calls.append(("start", num_cores, only_drawing, comparison_only, local_directory, dask_config, station_heavy))
        return handle

    monkeypatch.setattr(local_runner, "_start_optional_dask_client", start)
    monkeypatch.setattr(local_runner, "_close_optional_dask_client", lambda item: calls.append(("close", item)))
    monkeypatch.setattr(local_runner, "_run_evaluation_impl", lambda *args, **kwargs: {"status": "success"})

    assert local_runner.run_evaluation(cfg, force=True) == {"status": "success"}
    assert calls == [
        (
            "start",
            max(1, __import__("os").cpu_count() or 1),
            False,
            False,
            str(tmp_path / "case" / "scratch" / "dask"),
            cfg.project.dask,
            False,
        ),
        ("close", handle),
    ]


def test_run_evaluation_passes_station_heavy_to_dask_guard(tmp_path, monkeypatch):
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    cfg.simulation["SimA"].data_type = "stn"
    calls = []

    def start(num_cores, *, only_drawing, comparison_only, local_directory, dask_config, station_heavy):
        calls.append(station_heavy)
        return None

    monkeypatch.setattr(local_runner, "_start_optional_dask_client", start)
    monkeypatch.setattr(local_runner, "_close_optional_dask_client", lambda item: None)
    monkeypatch.setattr(local_runner, "_run_evaluation_impl", lambda *args, **kwargs: {"status": "success"})

    assert local_runner.run_evaluation(cfg, force=True) == {"status": "success"}
    assert calls == [True]


def test_run_evaluation_closes_optional_dask_on_error(tmp_path, monkeypatch):
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    handle = (object(), object())
    calls = []

    monkeypatch.setattr(local_runner, "_start_optional_dask_client", lambda *a, **k: handle)
    monkeypatch.setattr(local_runner, "_close_optional_dask_client", lambda item: calls.append(item))

    def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(local_runner, "_run_evaluation_impl", fail)

    with pytest.raises(RuntimeError, match="boom"):
        local_runner.run_evaluation(cfg, force=True)
    assert calls == [handle]


def test_run_evaluation_applies_project_io_env_defaults_for_run(tmp_path, monkeypatch):
    import os

    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    cfg.project.io = IOConfig(
        netcdf_compression=True,
        netcdf_compression_level=4,
        mfdataset_batch_size=25,
        mfdataset_auto_batch_min_files=50,
        mfdataset_auto_batch_memory_fraction=0.5,
    )
    captured = {}

    def fake_impl(*_args, **_kwargs):
        for key in (
            "OPENBENCH_NETCDF_COMPRESSION",
            "OPENBENCH_NETCDF_COMP_LEVEL",
            "OPENBENCH_MFDATASET_BATCH_SIZE",
            "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES",
            "OPENBENCH_MFDATASET_AUTO_BATCH_MEMORY_FRACTION",
        ):
            captured[key] = os.environ.get(key)
        return {"status": "success"}

    monkeypatch.delenv("OPENBENCH_NETCDF_COMPRESSION", raising=False)
    monkeypatch.delenv("OPENBENCH_NETCDF_COMP_LEVEL", raising=False)
    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.delenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", raising=False)
    monkeypatch.delenv("OPENBENCH_MFDATASET_AUTO_BATCH_MEMORY_FRACTION", raising=False)
    monkeypatch.setattr(local_runner, "_start_optional_dask_client", lambda *a, **k: None)
    monkeypatch.setattr(local_runner, "_run_evaluation_impl", fake_impl)

    assert local_runner.run_evaluation(cfg, force=True) == {"status": "success"}

    assert captured == {
        "OPENBENCH_NETCDF_COMPRESSION": "1",
        "OPENBENCH_NETCDF_COMP_LEVEL": "4",
        "OPENBENCH_MFDATASET_BATCH_SIZE": "25",
        "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES": "50",
        "OPENBENCH_MFDATASET_AUTO_BATCH_MEMORY_FRACTION": "0.5",
    }
    assert os.environ.get("OPENBENCH_NETCDF_COMPRESSION") is None
    assert os.environ.get("OPENBENCH_MFDATASET_BATCH_SIZE") is None


def test_run_evaluation_project_io_does_not_override_existing_env(tmp_path, monkeypatch):
    import os

    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    cfg.project.io = IOConfig(netcdf_compression=True, netcdf_compression_level=4, mfdataset_batch_size=25)
    captured = {}

    def fake_impl(*_args, **_kwargs):
        captured["compression"] = os.environ.get("OPENBENCH_NETCDF_COMPRESSION")
        captured["batch"] = os.environ.get("OPENBENCH_MFDATASET_BATCH_SIZE")
        return {"status": "success"}

    monkeypatch.setenv("OPENBENCH_NETCDF_COMPRESSION", "0")
    monkeypatch.setenv("OPENBENCH_MFDATASET_BATCH_SIZE", "7")
    monkeypatch.setattr(local_runner, "_start_optional_dask_client", lambda *a, **k: None)
    monkeypatch.setattr(local_runner, "_run_evaluation_impl", fake_impl)

    assert local_runner.run_evaluation(cfg, force=True) == {"status": "success"}
    assert captured == {"compression": "0", "batch": "7"}


def test_start_optional_dask_client_disabled_by_default(monkeypatch):
    import openbench.runner.local as local_runner

    monkeypatch.delenv("OPENBENCH_DASK", raising=False)
    monkeypatch.delenv("OPENBENCH_DASK_DISTRIBUTED", raising=False)

    assert local_runner._start_optional_dask_client(4) is None


def test_start_optional_dask_client_uses_env_options(monkeypatch):
    import sys
    import types

    import openbench.runner.local as local_runner

    calls = {}

    class FakeCluster:
        def __init__(self, **kwargs):
            calls["cluster_kwargs"] = kwargs
            self.closed = False

        def close(self):
            self.closed = True
            calls["cluster_closed"] = True

    class FakeClient:
        dashboard_link = "http://dashboard"

        def __init__(self, cluster, **kwargs):
            calls["client_cluster"] = cluster
            calls["client_kwargs"] = kwargs

        def close(self):
            calls["client_closed"] = True

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.Client = FakeClient
    fake_distributed.LocalCluster = FakeCluster
    monkeypatch.setitem(sys.modules, "distributed", fake_distributed)
    monkeypatch.setenv("OPENBENCH_DASK_DISTRIBUTED", "1")
    monkeypatch.setenv("OPENBENCH_DASK_WORKERS", "8")
    monkeypatch.setenv("OPENBENCH_DASK_THREADS_PER_WORKER", "2")
    monkeypatch.setenv("OPENBENCH_DASK_PROCESSES", "off")
    monkeypatch.setenv("OPENBENCH_DASK_DASHBOARD_ADDRESS", ":8787")

    handle = local_runner._start_optional_dask_client(4)
    assert handle is not None
    local_runner._close_optional_dask_client(handle)

    assert calls["cluster_kwargs"] == {
        "n_workers": 4,
        "threads_per_worker": 2,
        "processes": False,
        "memory_limit": "auto",
        "dashboard_address": ":8787",
        "scheduler_port": 0,
        "host": "127.0.0.1",
        "protocol": "tcp",
        "local_directory": None,
        "silence_logs": __import__("logging").WARNING,
    }
    assert calls["client_cluster"] is handle[1]
    assert calls["client_kwargs"] == {"set_as_default": True}
    assert calls["client_closed"] is True
    assert calls["cluster_closed"] is True


def test_start_optional_dask_client_skips_station_tasks_by_default(monkeypatch):
    import sys
    import types

    import openbench.runner.local as local_runner

    class ExplodingClient:
        def __init__(self, *_args, **_kwargs):  # pragma: no cover - guard should return before import use
            raise AssertionError("station-heavy guard should skip dask startup")

    class ExplodingCluster:
        def __init__(self, **_kwargs):  # pragma: no cover - guard should return before import use
            raise AssertionError("station-heavy guard should skip dask startup")

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.Client = ExplodingClient
    fake_distributed.LocalCluster = ExplodingCluster
    monkeypatch.setitem(sys.modules, "distributed", fake_distributed)
    monkeypatch.setenv("OPENBENCH_DASK", "1")
    monkeypatch.delenv("OPENBENCH_DASK_ALLOW_STATION", raising=False)

    assert (
        local_runner._start_optional_dask_client(
            4,
            tasks=[{"ref_data_type": "stn", "sim_data_type": "grid"}],
        )
        is None
    )


def test_start_optional_dask_client_can_force_station_tasks(monkeypatch):
    import sys
    import types

    import openbench.runner.local as local_runner

    calls = {}

    class FakeCluster:
        def __init__(self, **kwargs):
            calls["cluster_kwargs"] = kwargs

        def close(self):
            calls["cluster_closed"] = True

    class FakeClient:
        dashboard_link = None

        def __init__(self, cluster, **kwargs):
            calls["client_cluster"] = cluster
            calls["client_kwargs"] = kwargs

        def close(self):
            calls["client_closed"] = True

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.Client = FakeClient
    fake_distributed.LocalCluster = FakeCluster
    monkeypatch.setitem(sys.modules, "distributed", fake_distributed)
    monkeypatch.setenv("OPENBENCH_DASK", "1")
    monkeypatch.setenv("OPENBENCH_DASK_ALLOW_STATION", "1")

    handle = local_runner._start_optional_dask_client(
        4,
        tasks=[{"ref_data_type": "stn", "sim_data_type": "grid"}],
    )
    assert handle is not None
    local_runner._close_optional_dask_client(handle)

    assert calls["cluster_kwargs"]["n_workers"] == 4
    assert calls["client_kwargs"] == {"set_as_default": True}


def test_start_optional_dask_client_connects_external_scheduler(monkeypatch):
    import sys
    import types

    import openbench.runner.local as local_runner

    calls = {}

    class FakeClient:
        def __init__(self, address, **kwargs):
            calls["address"] = address
            calls["kwargs"] = kwargs

        def close(self):
            calls["closed"] = True

    class ExplodingCluster:
        def __init__(self, **_kwargs):  # pragma: no cover - should never be called
            raise AssertionError("external scheduler must not create LocalCluster")

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.Client = FakeClient
    fake_distributed.LocalCluster = ExplodingCluster
    monkeypatch.setitem(sys.modules, "distributed", fake_distributed)
    monkeypatch.setenv("OPENBENCH_DASK", "1")
    monkeypatch.setenv("OPENBENCH_DASK_DISTRIBUTED", "0")
    monkeypatch.setenv("OPENBENCH_DASK_SCHEDULER", "tcp://scheduler:8786")

    handle = local_runner._start_optional_dask_client(4)
    assert handle is not None
    assert handle[1] is None
    local_runner._close_optional_dask_client(handle)

    assert calls == {
        "address": "tcp://scheduler:8786",
        "kwargs": {"set_as_default": True},
        "closed": True,
    }


def test_unhandled_post_phase_error_still_cleans_pair_ref_override(tmp_path, monkeypatch):
    """Temporary per-pair ref files must be removed even if a post phase fails."""
    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path, comparison_enabled=True)
    cfg.project.IGBP_groupby = True
    pair_ref = tmp_path / "case" / "data" / "Runoff_ref_TestRef_SimA_runoff_ref.nc"

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": True,
            "time_alignment": "per_pair",
            "generate_report": False,
            "only_drawing": False,
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
        simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
    )

    Bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": namelists,
            "figures": adapter.LegacyFigureConfig(raw={}),
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
        },
    )

    def fake_evaluate(task):
        pair_ref.parent.mkdir(parents=True, exist_ok=True)
        pair_ref.write_text("temporary")
        task["ref_file_override"] = str(pair_ref)
        return {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        }

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(
        local_runner,
        "_run_groupby",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("groupby exploded")),
    )

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "partial"
    assert any(err["phase"] == "groupby" and "groupby exploded" in err["message"] for err in result["errors"])
    assert not pair_ref.exists()


def test_groupby_phase_obeys_comparison_enabled_gate(tmp_path, monkeypatch):
    """Groupby toggles should not run when the comparison phase is disabled."""
    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    cfg.project.IGBP_groupby = True

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "num_cores": 1,
            "unified_mask": True,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
                simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
            ),
            "figures": adapter.LegacyFigureConfig(raw={}),
            "iter_task_sources": lambda self, variables: [adapter.EvaluationSource("Runoff", "SimA", "TestRef")],
            "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                "casedir": str(tmp_path / "case"),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            },
            "has_grid_evaluation": lambda self, variables: adapter.GridEvaluationEvidence(False),
        },
    )()

    groupby_calls = []
    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(
        processing,
        "DatasetProcessing",
        type("FakeProcessor", (), {"__init__": lambda self, info: None, "prepare_source": lambda self, ds: None}),
    )
    monkeypatch.setattr(
        local_runner,
        "_evaluate_single",
        lambda task: {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "skipped": False,
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
        },
    )
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: groupby_calls.append(args) or [])

    result = run_evaluation(cfg, force=True)

    assert result["status"] == "success"
    assert groupby_calls == []


def test_evaluate_single_strict_rejects_equal_length_mismatched_time(tmp_path, monkeypatch):
    """strict alignment should fail on timestamp mismatch instead of positional pairing."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.core.evaluation as evaluation_module
    import openbench.runner.local as local_runner

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    lat = [0.0]
    lon = [10.0]
    ref_time = pd.date_range("2001-01-01", periods=2, freq="D")
    sim_time = pd.date_range("2001-01-02", periods=2, freq="D")
    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": ref_time, "lat": lat, "lon": lon},
    ).to_netcdf(data_dir / "Runoff_ref_TestRef_runoff_ref.nc")
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": sim_time, "lat": lat, "lon": lon},
    ).to_netcdf(data_dir / "Runoff_sim_SimA_runoff_sim.nc")

    monkeypatch.setattr(evaluation_module, "make_plot_index_grid", lambda *args, **kwargs: None)

    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "bindings": type(
            "Bindings",
            (),
            {
                "build_evaluation_fig_nml": lambda self: type(
                    "Fig",
                    (),
                    {"to_fig_nml": lambda self: {}},
                )(),
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: {
                    "casedir": str(case_dir),
                    "item": "Runoff",
                    "ref_varname": "runoff_ref",
                    "sim_varname": "runoff_sim",
                    "ref_data_type": "grid",
                    "sim_data_type": "grid",
                    "metrics": [],
                    "scores": [],
                    "time_alignment": "strict",
                },
            },
        )(),
        "cache_key": "Runoff__SimA__TestRef",
        "config_hash": "deadbeef",
        "use_cache": False,
        "cache_dir": str(case_dir),
        "ref_preprocessed": True,
    }

    result = local_runner._evaluate_single(task)

    assert result["status"] == "error"
    assert "time" in result["error"].lower()
    assert "mismatch" in result["error"].lower()


def test_unified_mask_strict_rejects_equal_length_mismatched_time(tmp_path):
    """strict unified_mask should fail fast on timestamp mismatch during preprocessing."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.runner.local as local_runner

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    lat = [0.0]
    lon = [10.0]
    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=2, freq="D"), "lat": lat, "lon": lon},
    ).to_netcdf(data_dir / "Runoff_ref_TestRef_runoff_ref.nc")
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": pd.date_range("2001-01-02", periods=2, freq="D"), "lat": lat, "lon": lon},
    ).to_netcdf(data_dir / "Runoff_sim_SimA_runoff_sim.nc")

    with pytest.raises(ValueError, match="time values mismatch"):
        local_runner._apply_unified_mask(
            {
                "casedir": str(case_dir),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "time_alignment": "strict",
            },
            "Runoff",
            "TestRef",
            "SimA",
        )


def test_unified_mask_write_failure_preserves_existing_ref_file(tmp_path, monkeypatch):
    """Unified mask should not corrupt the ref NC if rewriting the mask fails."""
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.runner.local as local_runner

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    ref_path = data_dir / "Runoff_ref_TestRef_runoff_ref.nc"
    sim_path = data_dir / "Runoff_sim_SimA_runoff_sim.nc"
    time = pd.date_range("2001-01-01", periods=2, freq="D")
    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": time, "lat": [0.0], "lon": [10.0]},
    ).to_netcdf(ref_path)
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": time, "lat": [0.0], "lon": [10.0]},
    ).to_netcdf(sim_path)

    def failing_to_netcdf(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial invalid netcdf")
        raise OSError("simulated write failure")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", failing_to_netcdf)

    with pytest.raises(OSError, match="simulated write failure"):
        local_runner._apply_unified_mask(
            {
                "casedir": str(case_dir),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "time_alignment": "intersection",
            },
            "Runoff",
            "TestRef",
            "SimA",
        )

    with xr.open_dataset(ref_path) as ds:
        np.testing.assert_allclose(ds["runoff_ref"].values, np.ones((2, 1, 1)))


def test_basic_comparison_per_pair_reads_pair_ref_file(tmp_path, monkeypatch):
    """Basic grid comparison should use per-pair masked refs when alignment is per_pair."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.core.comparison as comparison_module

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    time = pd.date_range("2001-01-01", periods=2, freq="D")
    lat = [0.0]
    lon = [10.0]
    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": time, "lat": lat, "lon": lon},
    ).to_netcdf(data_dir / "Runoff_ref_TestRef_SimA_runoff_ref.nc")

    main_nml = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "compare_grid_res": 0.5,
            "compare_tim_res": "Day",
            "time_alignment": "per_pair",
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "num_cores": 1,
            "weight": "area",
        }
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "TestRef"},
        "Runoff": {
            "TestRef_data_type": "grid",
            "TestRef_varname": "runoff_ref",
        },
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {
            "SimA_data_type": "grid",
            "SimA_varname": "runoff_sim",
        },
    }
    monkeypatch.setattr(comparison_module, "make_geo_plot_index", lambda *args, **kwargs: None)

    comparison = comparison_module.ComparisonProcessing(main_nml, ["Overall_Score"], ["bias"])
    comparison.scenarios_Basic_comparison(
        str(case_dir),
        sim_nml,
        ref_nml,
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        {"key": "Mean"},
    )

    assert (case_dir / "comparisons" / "Mean" / "Runoff_ref_TestRef_sim_SimA_runoff_ref_Mean.nc").exists()


# --- Multi-reference preprocessing (regression for c9fcbdc HIGH bug) ---


def test_preprocess_runs_for_each_ref_source(tmp_path, monkeypatch):
    """Each (variable, ref_source) pair must trigger ref preprocessing.

    Earlier code used `ref_done: bool` per variable, so configs with
    `reference: {Var: [RefA, RefB]}` only preprocessed RefA — RefB was
    silently skipped, producing tasks that referenced un-preprocessed
    NC files. Fix: track refs_done as set[str] keyed by ref_source.

    This test feeds 2 refs × 2 sims = 4 tasks and asserts both refs
    receive a prepare_source("ref") call.
    """
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=False,
            unified_mask=False,
        ),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        # Multi-ref: list of two source names
        reference=ReferenceConfig(sources={"Runoff": ["RefA", "RefB"]}),
        simulation={
            "SimA": SimulationEntry(model="MA", root_dir=str(tmp_path)),
            "SimB": SimulationEntry(model="MB", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )

    legacy = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": False,
            "generate_report": False,
        },
        "evaluation_items": {"Runoff": True},
        "metrics": {"bias": True},
        "scores": {"Overall_Score": True},
        "comparisons": {},
        "statistics": {},
    }
    main_nl = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
            "time_alignment": "intersection",
        },
    }
    ref_nml = {
        "general": {"Runoff_ref_source": ["RefA", "RefB"]},
        "Runoff": {"RefA_data_type": "grid", "RefB_data_type": "grid"},
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA", "SimB"]},
        "Runoff": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
    }

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    # Track prepare_source calls so we can verify each ref preprocesses
    prepare_calls: list[tuple[str, str, str]] = []  # (datasource, ref_source, sim_source)

    def fake_build_bridge_runtime_info(task):
        return {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "rv",
            "sim_varname": "sv",
            "ref_data_type": "grid",
            "sim_data_type": "grid",
            "ref_source": task["ref_source"],
            "sim_source": task["sim_source"],
        }

    class FakeProcessor:
        def __init__(self, info):
            self._ref_source = info["ref_source"]
            self._sim_source = info["sim_source"]

        def prepare_source(self, datasource):
            prepare_calls.append((datasource, self._ref_source, self._sim_source))

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "Runoff",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fake_build_bridge_runtime_info)
    monkeypatch.setattr(local_runner, "_apply_unified_mask", lambda *a, **k: None)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *a, **k: [])

    result = run_evaluation(cfg)
    assert result["status"] == "success"

    # Critical assertion: prepare_source("ref") MUST be called for BOTH refs.
    # Pre-fix this only fired for RefA (single ref_done=True flag).
    ref_calls = {(ds, ref) for ds, ref, _sim in prepare_calls if ds == "ref"}
    assert ("ref", "RefA") in ref_calls, "RefA was not preprocessed"
    assert ("ref", "RefB") in ref_calls, f"RefB was not preprocessed — multi-ref regression: got {sorted(ref_calls)}"
    # And exactly one ref preprocess per ref_source (not duplicated)
    assert len([c for c in prepare_calls if c[0] == "ref" and c[1] == "RefA"]) == 1
    assert len([c for c in prepare_calls if c[0] == "ref" and c[1] == "RefB"]) == 1

    # Sim should be preprocessed for every (sim, ref) task: 4 calls
    sim_calls = [c for c in prepare_calls if c[0] == "sim"]
    assert len(sim_calls) == 4, f"expected 4 sim preprocess calls (2 sim × 2 ref), got {len(sim_calls)}"


def test_preprocess_mixed_grid_and_stn_sims_with_same_grid_ref(tmp_path, monkeypatch):
    """Same grid ref reused across [SimGrid, SimStn] sims must trigger THREE ref preps.

    Edge case: with simulation: {SimGrid: grid, SimStn: stn} and a single
    grid reference, processing.extract_station_data_if_needed runs for the
    stn-involved task and DELETES the flat ref NC. A naive ref-once-per-
    ref_source dedupe would skip the third (RefA, SimGrid) task, leaving
    its evaluation pointed at the deleted flat NC.

    Fix: dedupe key is (ref_source, "_grid") for grid×grid AND
    (ref_source, sim_source) when any side is stn. This test pins that
    behavior — RefA gets at LEAST one grid-side prep AND one stn-side prep.
    """
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=False,
            unified_mask=False,
        ),
        evaluation=EvaluationConfig(variables=["LH"]),
        # Single grid ref, but reused across grid AND stn sim
        reference=ReferenceConfig(sources={"LH": "RefGrid"}),
        simulation={
            "SimGrid": SimulationEntry(model="MG", root_dir=str(tmp_path)),
            "SimStn": SimulationEntry(model="MS", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )

    legacy = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": False,
            "generate_report": False,
        },
        "evaluation_items": {"LH": True},
        "metrics": {"bias": True},
        "scores": {"Overall_Score": True},
        "comparisons": {},
        "statistics": {},
    }
    main_nl = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
            "time_alignment": "intersection",
        },
    }
    ref_nml = {
        "general": {"LH_ref_source": "RefGrid"},
        "LH": {"RefGrid_data_type": "grid"},
    }
    sim_nml = {
        "general": {"LH_sim_source": ["SimGrid", "SimStn"]},
        # Mixed: SimGrid is grid, SimStn is stn
        "LH": {"SimGrid_data_type": "grid", "SimStn_data_type": "stn"},
    }

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    prepare_calls: list[tuple[str, str, str, str]] = []  # (ds, ref, sim, sim_dtype)

    def fake_build_bridge_runtime_info(task):
        sim = task["sim_source"]
        sim_dtype = "grid" if sim == "SimGrid" else "stn"
        return {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "rv",
            "sim_varname": "sv",
            "ref_data_type": "grid",  # same ref for all tasks
            "sim_data_type": sim_dtype,
            "ref_source": task["ref_source"],
            "sim_source": sim,
        }

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            prepare_calls.append(
                (
                    datasource,
                    self.info["ref_source"],
                    self.info["sim_source"],
                    self.info["sim_data_type"],
                )
            )

    class FakeStnEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_evaluation_P(self):
            _write_fake_station_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fake_build_bridge_runtime_info)
    monkeypatch.setattr(local_runner, "_apply_unified_mask", lambda *a, **k: None)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGridEvaluation)
    monkeypatch.setattr(evaluation, "Evaluation_stn", FakeStnEvaluation)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *a, **k: [])

    result = run_evaluation(cfg)
    assert result["status"] == "success", result.get("errors")

    # The ref preprocessing must run THREE times for RefGrid:
    # - once for the grid×grid path (SimGrid) — flat NC created
    # - once for the stn-involved path (SimStn) — stn extraction deletes flat
    # - once at end of loop to RESTORE flat (because grid×grid evaluation
    #   downstream reads data/<var>_ref_<ref>_<varname>.nc)
    ref_calls_for_refgrid = [c for c in prepare_calls if c[0] == "ref" and c[1] == "RefGrid"]
    assert len(ref_calls_for_refgrid) == 3, (
        "Same grid ref reused across grid and stn sims should trigger 3 ref preps "
        "(grid + stn-pair + restoration); "
        f"got {ref_calls_for_refgrid}"
    )
    sim_dtypes_seen = {c[3] for c in ref_calls_for_refgrid}
    assert sim_dtypes_seen == {"grid", "stn"}, f"ref preps should cover both sim types, got {sim_dtypes_seen}"


def test_cache_validation_pattern_includes_ref_source(tmp_path):
    """Cache hit must verify outputs contain THIS task's ref_source.

    Regression: cache validation in _evaluate_single used the glob
    pattern f"{var_name}*{sim_source}*" which omitted ref_source. With
    multi-ref configs, an earlier (Var, SimA, RefA) task's outputs
    (Var_ref_RefA_sim_SimA_*.nc) would let a later (Var, SimA, RefB)
    cache check falsely pass — RefB had never run but _evaluate_single
    returned skipped=True. Fix reuses _find_existing_outputs which
    correctly includes ref_source in the glob.
    """
    from openbench.runner.cache import EvaluationCache, make_cache_key

    cache_dir = tmp_path
    (cache_dir / "scores").mkdir()
    (cache_dir / "metrics").mkdir()

    # Simulate prior run of (Runoff, SimA, RefA) leaving files behind
    (cache_dir / "scores" / "Runoff_ref_RefA_sim_SimA_Overall_Score.nc").touch()
    (cache_dir / "metrics" / "Runoff_ref_RefA_sim_SimA_bias.nc").touch()

    # Now check (Runoff, SimA, RefB) — DIFFERENT ref, same sim
    cache = EvaluationCache(cache_dir)
    refb_key = make_cache_key("Runoff", "SimA", "RefB")
    refb_hash = "fakehash_for_RefB"
    # Mark RefB as cached so cache.is_cached() returns True
    cache.mark_done(refb_key, refb_hash)

    # Pre-fix: this would falsely return skipped=True because the loose
    # pattern matched RefA's files. Post-fix: pattern includes ref_source,
    # no RefB files exist → cache treated as stale → re-evaluation triggered.
    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "RefB",  # ← different ref
        "cache_key": refb_key,
        "config_hash": refb_hash,
        "use_cache": True,
        "cache_dir": str(cache_dir),
        "bindings": None,
    }

    # We don't run the full _evaluate_single (it requires real bindings/info),
    # but we can directly assert the underlying pattern: _find_existing_outputs
    # for RefB must NOT return RefA's files.
    from openbench.runner.local import _find_existing_outputs

    matches = _find_existing_outputs(cache_dir, task)
    assert matches == [], (
        f"_find_existing_outputs for (Runoff, SimA, RefB) should return [] "
        f"but found {[m.name for m in matches]} — multi-ref cache validation regression"
    )

    # Sanity: RefA's lookup DOES find them
    task_refa = dict(task, ref_source="RefA")
    matches_a = _find_existing_outputs(cache_dir, task_refa)
    assert len(matches_a) == 2, f"RefA should find its 2 files, got {matches_a}"


def test_find_existing_outputs_does_not_match_prefix_overlap(tmp_path):
    """Cache/output lookup must not confuse RefA with RefAB or SimA with SimAB."""
    from openbench.runner.local import _find_existing_outputs

    (tmp_path / "scores").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "scores" / "Runoff_ref_RefAB_sim_SimAB_Overall_Score.nc").touch()
    (tmp_path / "metrics" / "Runoff_stn_RefAB_SimAB_evaluations.csv").touch()

    task = {"var_name": "Runoff", "ref_source": "RefA", "sim_source": "SimA"}
    matches = _find_existing_outputs(tmp_path, task)
    assert matches == [], (
        f"Lookup for (Runoff, RefA, SimA) must not match files for (Runoff, RefAB, SimAB): {[m.name for m in matches]}"
    )

    exact_task = {"var_name": "Runoff", "ref_source": "RefAB", "sim_source": "SimAB"}
    exact_matches = {m.name for m in _find_existing_outputs(tmp_path, exact_task)}
    assert exact_matches == {
        "Runoff_ref_RefAB_sim_SimAB_Overall_Score.nc",
        "Runoff_stn_RefAB_SimAB_evaluations.csv",
    }


def test_unified_mask_persists_across_stn_prep_deletion_intersection(tmp_path, monkeypatch):
    """Restoring a stn-deleted flat ref must preserve accumulated unified_mask."""

    import numpy as np
    import xarray as xr

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=False,
            unified_mask=True,
            time_alignment="intersection",
        ),
        evaluation=EvaluationConfig(variables=["LH"]),
        reference=ReferenceConfig(sources={"LH": "RefGrid"}),
        simulation={
            "SimGrid1": SimulationEntry(model="MG1", root_dir=str(tmp_path)),
            "SimStn": SimulationEntry(model="MS", root_dir=str(tmp_path)),
            "SimGrid2": SimulationEntry(model="MG2", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )

    casedir = tmp_path / "case"
    flat_nc = casedir / "data" / "LH_ref_RefGrid_rv.nc"
    times = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]")
    coords = {"time": times, "lat": [10.0, 20.0], "lon": [100.0, 110.0]}

    def write_ref():
        flat_nc.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"rv": (("time", "lat", "lon"), np.ones((2, 2, 2), dtype=float))},
            coords=coords,
        ).to_netcdf(flat_nc)

    def write_sim(sim_source):
        values = np.ones((2, 2, 2), dtype=float)
        if sim_source == "SimGrid1":
            values[0, 0, 0] = np.nan
        elif sim_source == "SimGrid2":
            values[1, 1, 1] = np.nan
        xr.Dataset(
            {"sv": (("time", "lat", "lon"), values)},
            coords=coords,
        ).to_netcdf(casedir / "data" / f"LH_sim_{sim_source}_sv.nc")

    class Fig:
        def to_fig_nml(self):
            return {}

    class GridEvidence:
        has_grid = True

    class Bindings:
        def __init__(self):
            self.runner_cfg = SimpleNamespace(
                basedir=str(tmp_path),
                basename="case",
                general={
                    "basename": "case",
                    "basedir": str(tmp_path),
                    "num_cores": 1,
                    "unified_mask": True,
                    "generate_report": False,
                },
                evaluation_items={"LH": True},
                metrics={"bias"},
                scores={"Overall_Score"},
                comparisons=set(),
                statistics=set(),
            )

        def iter_task_sources(self, variables):
            for sim in ("SimGrid1", "SimStn", "SimGrid2"):
                yield SimpleNamespace(var_name="LH", sim_source=sim, ref_source="RefGrid")

        def build_evaluation_fig_nml(self):
            return Fig()

        def has_grid_evaluation(self, variables):
            return GridEvidence()

    def fake_info(task):
        sim = task["sim_source"]
        info = {
            "casedir": str(casedir),
            "ref_varname": "rv",
            "sim_varname": "sv",
            "ref_data_type": "grid",
            "sim_data_type": "stn" if sim == "SimStn" else "grid",
            "ref_source": task["ref_source"],
            "sim_source": sim,
        }
        if task.get("ref_file_override"):
            info["ref_file_override"] = task["ref_file_override"]
        return info

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            if datasource == "ref":
                write_ref()
                if self.info["sim_data_type"] == "stn":
                    flat_nc.unlink()
            elif datasource == "sim" and self.info["sim_data_type"] == "grid":
                write_sim(self.info["sim_source"])

    class FakeGrid:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    class FakeStn:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_evaluation_P(self):
            _write_fake_station_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fake_info)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGrid)
    monkeypatch.setattr(evaluation, "Evaluation_stn", FakeStn)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *a, **k: [])

    result = run_evaluation(cfg, force=True)
    assert result["status"] == "success", result.get("errors")

    with xr.open_dataset(flat_nc) as ds:
        values = ds["rv"].values
        assert np.isnan(values[0, 0, 0]), "Mask from SimGrid1 was lost"
        assert np.isnan(values[1, 1, 1]), "Mask from SimGrid2 was lost"
        assert values[0, 1, 1] == 1.0, "Valid cells should remain valid"


def test_unified_mask_per_pair_handles_deleted_flat(tmp_path, monkeypatch):
    """per_pair mode must restore the flat ref before copying later pair refs."""

    import numpy as np
    import xarray as xr

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=False,
            unified_mask=True,
            time_alignment="per_pair",
        ),
        evaluation=EvaluationConfig(variables=["LH"]),
        reference=ReferenceConfig(sources={"LH": "RefGrid"}),
        simulation={
            "SimGrid1": SimulationEntry(model="MG1", root_dir=str(tmp_path)),
            "SimStn": SimulationEntry(model="MS", root_dir=str(tmp_path)),
            "SimGrid2": SimulationEntry(model="MG2", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )

    casedir = tmp_path / "case"
    flat_nc = casedir / "data" / "LH_ref_RefGrid_rv.nc"
    times = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]")
    coords = {"time": times, "lat": [10.0, 20.0], "lon": [100.0, 110.0]}
    seen_pair_refs: dict[str, np.ndarray] = {}

    def write_ref():
        flat_nc.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"rv": (("time", "lat", "lon"), np.ones((2, 2, 2), dtype=float))},
            coords=coords,
        ).to_netcdf(flat_nc)

    def write_sim(sim_source):
        values = np.ones((2, 2, 2), dtype=float)
        if sim_source == "SimGrid1":
            values[0, 0, 0] = np.nan
        elif sim_source == "SimGrid2":
            values[1, 1, 1] = np.nan
        xr.Dataset(
            {"sv": (("time", "lat", "lon"), values)},
            coords=coords,
        ).to_netcdf(casedir / "data" / f"LH_sim_{sim_source}_sv.nc")

    class Fig:
        def to_fig_nml(self):
            return {}

    class GridEvidence:
        has_grid = True

    class Bindings:
        def __init__(self):
            self.runner_cfg = SimpleNamespace(
                basedir=str(tmp_path),
                basename="case",
                general={
                    "basename": "case",
                    "basedir": str(tmp_path),
                    "num_cores": 1,
                    "unified_mask": True,
                    "generate_report": False,
                },
                evaluation_items={"LH": True},
                metrics={"bias"},
                scores={"Overall_Score"},
                comparisons=set(),
                statistics=set(),
            )

        def iter_task_sources(self, variables):
            for sim in ("SimGrid1", "SimStn", "SimGrid2"):
                yield SimpleNamespace(var_name="LH", sim_source=sim, ref_source="RefGrid")

        def build_evaluation_fig_nml(self):
            return Fig()

        def has_grid_evaluation(self, variables):
            return GridEvidence()

    def fake_info(task):
        sim = task["sim_source"]
        info = {
            "casedir": str(casedir),
            "ref_varname": "rv",
            "sim_varname": "sv",
            "ref_data_type": "grid",
            "sim_data_type": "stn" if sim == "SimStn" else "grid",
            "ref_source": task["ref_source"],
            "sim_source": sim,
        }
        if task.get("ref_file_override"):
            info["ref_file_override"] = task["ref_file_override"]
        return info

    class FakeProcessor:
        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            if datasource == "ref":
                write_ref()
                if self.info["sim_data_type"] == "stn":
                    flat_nc.unlink()
            elif datasource == "sim" and self.info["sim_data_type"] == "grid":
                write_sim(self.info["sim_source"])

    class FakeGrid:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            ref_override = self.info.get("ref_file_override")
            assert ref_override, "per_pair grid evaluation must use ref_file_override"
            with xr.open_dataset(ref_override) as ds:
                seen_pair_refs[self.info["sim_source"]] = ds["rv"].values.copy()
            _write_fake_grid_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    class FakeStn:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_evaluation_P(self):
            _write_fake_station_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: Bindings())
    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fake_info)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGrid)
    monkeypatch.setattr(evaluation, "Evaluation_stn", FakeStn)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *a, **k: [])

    result = run_evaluation(cfg, force=True)
    assert result["status"] == "success", result.get("errors")
    assert set(seen_pair_refs) == {"SimGrid1", "SimGrid2"}
    assert np.isnan(seen_pair_refs["SimGrid1"][0, 0, 0])
    assert not np.isnan(seen_pair_refs["SimGrid1"][1, 1, 1])
    assert np.isnan(seen_pair_refs["SimGrid2"][1, 1, 1])
    assert not np.isnan(seen_pair_refs["SimGrid2"][0, 0, 0])


def test_preprocess_grid_stn_grid_sequence_restores_deleted_flat(tmp_path, monkeypatch):
    """Grid → stn → grid sequence with same ref must end with flat NC present.

    Regression: bf561bc dedupes grid×grid prep by (ref_source, "_grid").
    Sequence [SimGrid1, SimStn, SimGrid2] for a single grid ref:
      - Task 1 (grid): prep runs, flat NC at data/<var>_ref_<ref>_<v>.nc created
      - Task 2 (stn): prep runs, extract_station_data DELETES flat NC
      - Task 3 (grid): dedupe key (RefA, "_grid") already done → SKIPPED.
        Flat NC is gone; downstream Evaluation_grid reads the missing path
        and crashes with FileNotFoundError.

    Fix: end-of-loop restoration. After all per-task prep, if any ref had
    BOTH stn-involved prep AND grid×grid tasks, re-run grid prep once to
    restore the flat NC.

    This test simulates the deletion in the FakeProcessor, then asserts
    the flat NC exists on disk after _preprocess_variable returns.
    """
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            generate_report=False,
            unified_mask=False,
        ),
        evaluation=EvaluationConfig(variables=["LH"]),
        reference=ReferenceConfig(sources={"LH": "RefGrid"}),
        # Three sims: grid, stn, grid — the sequence the reviewer probed
        simulation={
            "SimGrid1": SimulationEntry(model="MG1", root_dir=str(tmp_path)),
            "SimStn": SimulationEntry(model="MS", root_dir=str(tmp_path)),
            "SimGrid2": SimulationEntry(model="MG2", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )

    legacy = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": False,
            "generate_report": False,
        },
        "evaluation_items": {"LH": True},
        "metrics": {"bias": True},
        "scores": {"Overall_Score": True},
        "comparisons": {},
        "statistics": {},
    }
    main_nl = {
        "general": {
            "basename": "case",
            "basedir": str(tmp_path),
            "num_cores": 1,
            "unified_mask": False,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "syear": 2000,
            "eyear": 2001,
            "time_alignment": "intersection",
        },
    }
    ref_nml = {
        "general": {"LH_ref_source": "RefGrid"},
        "LH": {"RefGrid_data_type": "grid"},
    }
    sim_nml = {
        "general": {"LH_sim_source": ["SimGrid1", "SimStn", "SimGrid2"]},
        "LH": {
            "SimGrid1_data_type": "grid",
            "SimStn_data_type": "stn",
            "SimGrid2_data_type": "grid",
        },
    }

    import openbench.config.adapter as adapter
    import openbench.core.evaluation as evaluation
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner

    case_root = tmp_path / "case" / "case"
    data_dir = case_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    flat_nc = data_dir / "LH_ref_RefGrid_rv.nc"

    prepare_calls: list[tuple[str, str, str, str]] = []  # (ds, ref, sim, sim_dtype)

    def fake_build_bridge_runtime_info(task):
        sim = task["sim_source"]
        sim_dtype = "stn" if sim == "SimStn" else "grid"
        return {
            "casedir": str(tmp_path / "case"),
            "ref_varname": "rv",
            "sim_varname": "sv",
            "ref_data_type": "grid",
            "sim_data_type": sim_dtype,
            "ref_source": task["ref_source"],
            "sim_source": sim,
        }

    class FakeProcessorWithStnDeletion:
        """Simulates real DatasetProcessing: stn-involved ref prep deletes flat."""

        def __init__(self, info):
            self.info = info

        def prepare_source(self, datasource):
            prepare_calls.append(
                (
                    datasource,
                    self.info["ref_source"],
                    self.info["sim_source"],
                    self.info["sim_data_type"],
                )
            )
            if datasource == "ref":
                # Always create the flat NC (mirroring real grid prep step)
                flat_nc.write_text("")
                # If stn-involved, simulate extract_station_data deleting flat
                if self.info["ref_data_type"] == "stn" or self.info["sim_data_type"] == "stn":
                    flat_nc.unlink()

    class FakeStn:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_evaluation_P(self):
            _write_fake_station_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    class FakeGrid:
        def __init__(self, info, fig_nml):
            self.info = info

        def make_Evaluation(self):
            _write_fake_grid_outputs(
                self.info["casedir"],
                "LH",
                self.info["ref_source"],
                self.info["sim_source"],
            )

    monkeypatch.setattr(adapter, "to_legacy_config", lambda cfg: legacy)
    monkeypatch.setattr(adapter, "build_legacy_namelists", lambda cfg: (main_nl, ref_nml, sim_nml))
    monkeypatch.setattr(adapter, "build_fig_nml", lambda: {})
    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fake_build_bridge_runtime_info)
    monkeypatch.setattr(local_runner, "_apply_unified_mask", lambda *a, **k: None)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessorWithStnDeletion)
    monkeypatch.setattr(evaluation, "Evaluation_grid", FakeGrid)
    monkeypatch.setattr(evaluation, "Evaluation_stn", FakeStn)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *a, **k: [])

    result = run_evaluation(cfg)
    assert result["status"] == "success", result.get("errors")

    # Critical assertion: after _preprocess_variable completes, the flat NC
    # MUST exist on disk for downstream grid evaluation to succeed.
    assert flat_nc.exists(), (
        f"Flat ref NC missing after preprocessing. The grid→stn→grid sequence "
        f"left the flat NC deleted. prepare_source calls: {prepare_calls}"
    )

    # Sequence verification: grid prep ran for SimGrid1 (initial), stn prep
    # ran for SimStn (deleting flat), then grid prep ran again for restoration.
    ref_call_seq = [c[3] for c in prepare_calls if c[0] == "ref"]
    # Expect at least one "grid" call after the "stn" call in the sequence
    last_stn_idx = max(
        (i for i, t in enumerate(ref_call_seq) if t == "stn"),
        default=-1,
    )
    assert last_stn_idx >= 0, "Expected at least one stn-path ref prep"
    grid_after_stn = [t for t in ref_call_seq[last_stn_idx + 1 :] if t == "grid"]
    assert len(grid_after_stn) >= 1, (
        f"Expected at least one grid ref prep AFTER the stn-path prep (restoration). Sequence: {ref_call_seq}"
    )


def test_run_comparison_filter_does_not_match_prefix_overlap_items(tmp_path):
    """When the user has variables ['Runoff', 'Runoff_2'] and only Runoff_2
    has output files, the comparison item filter for 'Runoff' must NOT pass.

    Previously used startswith(f"{item}_") which substring-matched: 'Runoff_'
    is a prefix of 'Runoff_2_*.nc', so Runoff would falsely be flagged as
    having outputs and ComparisonProcessing would be invoked with an item
    that has no actual data — same shape of bug as the scanner cache pattern
    fixed in 3b80d20.
    """
    from unittest.mock import MagicMock

    from openbench.runner.local import _run_comparison

    output_dir = tmp_path
    (output_dir / "scores").mkdir()
    (output_dir / "metrics").mkdir()

    # Only Runoff_2 has real outputs (overlapping prefix scenario)
    _write_fake_grid_outputs(output_dir, "Runoff_2", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))
    # Runoff has NO outputs

    captured_items = {"items": None}

    class FakeComparisonProcessing:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_HeatMap_comparison(self, basedir, sim_nml, ref_nml, items, *args, **kwargs):
            captured_items["items"] = list(items)

    import openbench.core.comparison as comp_mod

    orig_attr = getattr(comp_mod, "ComparisonProcessing", None)
    comp_mod.ComparisonProcessing = FakeComparisonProcessing
    try:
        bindings = MagicMock()
        bindings.build_comparison_context.return_value = SimpleNamespace(
            namelists=SimpleNamespace(main={}, simulation={}, reference={}),
            score_vars=["Overall_Score"],
            metric_vars=["bias"],
            evaluation_items=["Runoff", "Runoff_2"],
            comparison_fig={"HeatMap": {}},
        )
        errors = _run_comparison(bindings, ["HeatMap"], output_dir)
    finally:
        if orig_attr is not None:
            comp_mod.ComparisonProcessing = orig_attr

    assert errors == [], f"Comparison should not error: {errors}"
    # Filter must reject Runoff (no actual outputs), pass Runoff_2 only
    assert captured_items["items"] == ["Runoff_2"], (
        f"Expected only Runoff_2 (has outputs), got: {captured_items['items']}. "
        "Runoff_ as substring would have falsely included Runoff."
    )


def test_run_comparison_reports_incomplete_task_outputs_instead_of_partial_items(tmp_path, monkeypatch):
    """Comparison helper should not run on a variable with only one of two sim outputs."""
    import openbench.config.adapter as adapter_module
    import openbench.core.comparison as comparison_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))

    called = []

    class FakeComparisonProcessing:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_HeatMap_comparison(self, *args, **kwargs):
            called.append(args)

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={"general": {}},
                    simulation={"general": {}},
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={"HeatMap": {}},
            )

        def iter_task_sources(self, variables):
            return [
                adapter_module.EvaluationSource("Runoff", "SimA", "RefA"),
                adapter_module.EvaluationSource("Runoff", "SimB", "RefA"),
            ]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {"ref_data_type": "grid", "sim_data_type": "grid"}

    monkeypatch.setattr(comparison_module, "ComparisonProcessing", FakeComparisonProcessing)

    errors = local_runner._run_comparison(Bindings(), ["HeatMap"], output_dir)

    assert called == []
    assert errors
    assert errors[0]["phase"] == "preflight"
    assert errors[0]["sim"] == "SimB"


def test_run_comparison_uses_namelist_fallback_for_station_column_preflight(tmp_path, monkeypatch):
    """Post-phase preflight must still catch incomplete station CSVs without iter_task_sources."""
    import openbench.config.adapter as adapter_module
    import openbench.core.comparison as comparison_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_station_outputs(
        output_dir,
        "Runoff",
        "RefA",
        "SimA",
        metrics=("RMSE",),
        scores=("Overall_Score",),
    )
    called = []

    class FakeComparisonProcessing:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_HeatMap_comparison(self, *args, **kwargs):
            called.append(args)

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={
                        "general": {"Runoff_ref_source": "RefA"},
                        "Runoff": {"RefA_data_type": "grid"},
                    },
                    simulation={
                        "general": {"Runoff_sim_source": ["SimA"]},
                        "Runoff": {"SimA_data_type": "stn"},
                    },
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={"HeatMap": {}},
            )

    monkeypatch.setattr(comparison_module, "ComparisonProcessing", FakeComparisonProcessing)

    errors = local_runner._run_comparison(Bindings(), ["HeatMap"], output_dir)

    assert called == []
    assert errors
    assert errors[0]["phase"] == "preflight"
    assert str(errors[0]["missing_outputs"][0]).endswith("Runoff_stn_RefA_SimA_evaluations.csv")


def test_run_comparison_constructs_processor_once(tmp_path, monkeypatch):
    """Comparison helper should not instantiate the processor twice for one phase."""
    import openbench.config.adapter as adapter_module
    import openbench.core.comparison as comparison_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))
    calls = {"init": 0, "method": 0}

    class FakeComparisonProcessing:
        def __init__(self, *args, **kwargs):
            calls["init"] += 1

        def scenarios_HeatMap_comparison(self, *args, **kwargs):
            calls["method"] += 1

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={"general": {}},
                    simulation={"general": {}},
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={"HeatMap": {}},
            )

    monkeypatch.setattr(comparison_module, "ComparisonProcessing", FakeComparisonProcessing)

    errors = local_runner._run_comparison(Bindings(), ["HeatMap"], output_dir)

    assert errors == []
    assert calls == {"init": 1, "method": 1}


def test_run_comparison_errors_include_item_source_and_original_exception(tmp_path, monkeypatch):
    import openbench.config.adapter as adapter_module
    import openbench.core.comparison as comparison_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))

    class FakeComparisonProcessing:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_HeatMap_comparison(self, *args, **kwargs):
            raise ValueError("first comparison failure")

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {}})()

        def build_comparison_context(self):
            return adapter_module.ComparisonContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={"general": {}},
                    simulation={"general": {}},
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                comparison_fig={"HeatMap": {}},
            )

    monkeypatch.setattr(comparison_module, "ComparisonProcessing", FakeComparisonProcessing)

    errors = local_runner._run_comparison(Bindings(), ["HeatMap"], output_dir)

    assert len(errors) == 1
    assert errors[0]["phase"] == "comparison"
    assert errors[0]["item"] == "HeatMap"
    assert errors[0]["source"] == "scenarios_HeatMap_comparison"
    assert "first comparison failure" in errors[0]["message"]


def test_run_groupby_filter_does_not_include_items_without_outputs(tmp_path, monkeypatch):
    """Groupby should not receive variables that produced no metrics/scores."""
    import openbench.config.adapter as adapter_module
    import openbench.core.landcover_groupby as landcover_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    (output_dir / "scores").mkdir()
    (output_dir / "metrics").mkdir()
    _write_fake_grid_outputs(output_dir, "Runoff_2", "RefA", "SimA", metrics=(), scores=("Overall_Score",))

    captured_items = {"items": None}

    cfg = type(
        "Cfg",
        (),
        {
            "project": type(
                "Project",
                (),
                {"IGBP_groupby": True, "PFT_groupby": False, "climate_zone_groupby": False},
            )(),
        },
    )()

    class FakeLCGroupby:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_IGBP_groupby_comparison(
            self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, validation_fig
        ):
            captured_items["items"] = list(evaluation_items)

    monkeypatch.setattr(landcover_module, "LC_groupby", FakeLCGroupby)

    bindings = type(
        "Bindings",
        (),
        {
            "build_groupby_context": lambda self: adapter_module.GroupbyContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={"general": {}},
                    simulation={"general": {}},
                ),
                evaluation_items=["Runoff", "Runoff_2"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                validation_fig={},
                climate_zone_fig={},
            ),
            "runner_cfg": type("RunnerCfg", (), {"general": {"only_drawing": False}})(),
        },
    )()

    errors = local_runner._run_groupby(cfg, bindings, output_dir)

    assert errors == []
    assert captured_items["items"] == ["Runoff_2"]


def test_run_groupby_uses_namelist_fallback_for_station_column_preflight(tmp_path, monkeypatch):
    """Groupby should not proceed on station CSVs that are readable but lack requested columns."""
    import openbench.config.adapter as adapter_module
    import openbench.core.landcover_groupby as landcover_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_station_outputs(
        output_dir,
        "Runoff",
        "RefA",
        "SimA",
        metrics=("RMSE",),
        scores=("Overall_Score",),
    )
    called = []
    cfg = type(
        "Cfg",
        (),
        {
            "project": type(
                "Project",
                (),
                {"IGBP_groupby": True, "PFT_groupby": False, "climate_zone_groupby": False},
            )(),
        },
    )()

    class FakeLCGroupby:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_IGBP_groupby_comparison(self, *args, **kwargs):
            called.append(args)

    class Bindings:
        runner_cfg = type("RunnerCfg", (), {"general": {"only_drawing": False}})()

        def build_groupby_context(self):
            return adapter_module.GroupbyContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={
                        "general": {"Runoff_ref_source": "RefA"},
                        "Runoff": {"RefA_data_type": "grid"},
                    },
                    simulation={
                        "general": {"Runoff_sim_source": ["SimA"]},
                        "Runoff": {"SimA_data_type": "stn"},
                    },
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                validation_fig={},
                climate_zone_fig={},
            )

    monkeypatch.setattr(landcover_module, "LC_groupby", FakeLCGroupby)

    errors = local_runner._run_groupby(cfg, Bindings(), output_dir)

    assert called == []
    assert errors
    assert errors[0]["phase"] == "preflight"
    assert str(errors[0]["missing_outputs"][0]).endswith("Runoff_stn_RefA_SimA_evaluations.csv")


def test_run_groupby_errors_include_item_source_and_original_exception(tmp_path, monkeypatch):
    import openbench.config.adapter as adapter_module
    import openbench.core.landcover_groupby as landcover_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=("Overall_Score",))
    cfg = type(
        "Cfg",
        (),
        {
            "project": type(
                "Project",
                (),
                {"IGBP_groupby": True, "PFT_groupby": False, "climate_zone_groupby": False},
            )(),
        },
    )()

    class FakeLCGroupby:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_IGBP_groupby_comparison(self, *args, **kwargs):
            raise RuntimeError("first groupby failure")

    bindings = type(
        "Bindings",
        (),
        {
            "build_groupby_context": lambda self: adapter_module.GroupbyContext(
                namelists=adapter_module.LegacyNamelists(
                    main={"general": {"basename": "case"}},
                    reference={"general": {}},
                    simulation={"general": {}},
                ),
                evaluation_items=["Runoff"],
                score_vars=["Overall_Score"],
                metric_vars=["bias"],
                validation_fig={},
                climate_zone_fig={},
            ),
            "runner_cfg": type("RunnerCfg", (), {"general": {"only_drawing": False}})(),
        },
    )()

    monkeypatch.setattr(landcover_module, "LC_groupby", FakeLCGroupby)

    errors = local_runner._run_groupby(cfg, bindings, output_dir)

    assert len(errors) == 1
    assert errors[0]["phase"] == "groupby"
    assert errors[0]["item"] == "IGBP_groupby"
    assert errors[0]["source"] == "LC_groupby"
    assert "first groupby failure" in errors[0]["message"]


def test_run_statistics_filter_does_not_include_items_without_outputs(tmp_path, monkeypatch):
    """Statistics context should be built only for variables with actual outputs."""
    import openbench.config.adapter as adapter_module
    import openbench.core.statistics.Mod_Statistics as stats_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    (output_dir / "scores").mkdir()
    (output_dir / "metrics").mkdir()
    _write_fake_grid_outputs(output_dir, "Runoff_2", "RefA", "SimA", metrics=("bias",), scores=())

    captured_items = {"items": None}
    namelists = adapter_module.LegacyNamelists(
        main={"general": {"basename": "case"}},
        reference={"general": {}},
        simulation={"general": {}},
    )
    context = adapter_module.StatisticsContext(
        namelists=namelists,
        stats_dir=str(output_dir / "statistics"),
        stats_nml={"general": {"Mean_data_source": "Runoff_2"}, "Mean": {}},
        num_cores=1,
        statistic_fig={"Mean": {}},
    )

    class FakeStatisticsProcessing:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_Basic_analysis(self, statistic, stat_cfg, stat_fig):
            return None

    def build_statistics_context(self, statistic_vars, evaluation_items=None):
        captured_items["items"] = list(evaluation_items)
        return context

    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": type(
                "RunnerCfg",
                (),
                {"evaluation_items": {"Runoff": True, "Runoff_2": True}},
            )(),
            "build_statistics_context": build_statistics_context,
        },
    )()

    monkeypatch.setattr(stats_module, "StatisticsProcessing", FakeStatisticsProcessing)

    errors = local_runner._run_statistics(bindings, ["Mean"], output_dir)

    assert errors == []
    assert captured_items["items"] == ["Runoff_2"]


def test_run_statistics_uses_namelist_fallback_for_station_column_preflight(tmp_path):
    """Statistics preflight should reject incomplete configured station outputs before building context."""
    import openbench.config.adapter as adapter_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_station_outputs(
        output_dir,
        "Runoff",
        "RefA",
        "SimA",
        metrics=("RMSE",),
        scores=("Overall_Score",),
    )
    context_built = []

    class Bindings:
        runner_cfg = type(
            "RunnerCfg",
            (),
            {
                "evaluation_items": {"Runoff": True},
                "metrics": ["bias"],
                "scores": ["Overall_Score"],
            },
        )()
        namelists = adapter_module.LegacyNamelists(
            main={"general": {"basename": "case"}},
            reference={
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid"},
            },
            simulation={
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "stn"},
            },
        )

        def build_statistics_context(self, *args, **kwargs):
            context_built.append(True)
            raise AssertionError("statistics context should not be built after failed preflight")

    errors = local_runner._run_statistics(Bindings(), ["Mean"], output_dir)

    assert context_built == []
    assert errors
    assert errors[0]["phase"] == "preflight"
    assert str(errors[0]["missing_outputs"][0]).endswith("Runoff_stn_RefA_SimA_evaluations.csv")


def test_run_statistics_errors_include_item_source_and_original_exception(tmp_path, monkeypatch):
    import openbench.config.adapter as adapter_module
    import openbench.core.statistics.Mod_Statistics as stats_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    _write_fake_grid_outputs(output_dir, "Runoff", "RefA", "SimA", metrics=("bias",), scores=())
    context = adapter_module.StatisticsContext(
        namelists=adapter_module.LegacyNamelists(
            main={"general": {"basename": "case"}},
            reference={"general": {}},
            simulation={"general": {}},
        ),
        stats_dir=str(output_dir / "statistics"),
        stats_nml={"general": {"Mean_data_source": "Runoff"}, "Mean": {}},
        num_cores=1,
        statistic_fig={"Mean": {}},
    )

    class FakeStatisticsProcessing:
        def __init__(self, *args, **kwargs):
            pass

        def scenarios_Basic_analysis(self, *args, **kwargs):
            raise RuntimeError("first statistics failure")

    class Bindings:
        runner_cfg = type(
            "RunnerCfg",
            (),
            {"evaluation_items": {"Runoff": True}, "metrics": ["bias"], "scores": []},
        )()

        def build_statistics_context(self, *args, **kwargs):
            return context

    monkeypatch.setattr(stats_module, "StatisticsProcessing", FakeStatisticsProcessing)

    errors = local_runner._run_statistics(Bindings(), ["Mean"], output_dir)

    assert len(errors) == 1
    assert errors[0]["phase"] == "statistics"
    assert errors[0]["item"] == "Mean"
    assert errors[0]["source"] == "scenarios_Basic_analysis"
    assert "first statistics failure" in errors[0]["message"]


@pytest.mark.parametrize(
    ("method_name", "plot_func"),
    [
        ("scenarios_HeatMap_comparison", "make_scenarios_scores_comparison_heat_map"),
        ("scenarios_RadarMap_comparison", "make_scenarios_comparison_radar_map"),
    ],
)
def test_core_comparison_station_missing_score_column_raises(tmp_path, monkeypatch, method_name, plot_func):
    """Direct core comparison calls must not silently convert missing station score columns to N/A."""
    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "Runoff_stn_RefA_SimA_evaluations.csv").write_text("ID,OtherScore\nS1,1.0\n")

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )
    monkeypatch.setattr(comparison_module, plot_func, lambda *args, **kwargs: None)

    with pytest.raises(KeyError, match="Overall_Score"):
        getattr(processor, method_name)(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "stn", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


@pytest.mark.parametrize(
    ("method_name", "plot_func"),
    [
        ("scenarios_HeatMap_comparison", "make_scenarios_scores_comparison_heat_map"),
        ("scenarios_RadarMap_comparison", "make_scenarios_comparison_radar_map"),
    ],
)
@pytest.mark.parametrize("case", ["missing_file", "missing_variable"])
def test_core_comparison_grid_missing_score_output_raises(tmp_path, monkeypatch, method_name, plot_func, case):
    """Direct core comparison calls must not silently turn missing grid score outputs into N/A."""
    import numpy as np
    import xarray as xr

    import openbench.core.comparison as comparison_module

    if case == "missing_variable":
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        xr.Dataset({"OtherScore": ("sample", np.array([1.0]))}).to_netcdf(
            scores_dir / "Runoff_ref_RefA_sim_SimA_Overall_Score.nc"
        )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )
    monkeypatch.setattr(comparison_module, plot_func, lambda *args, **kwargs: None)

    expected = FileNotFoundError if case == "missing_file" else KeyError
    with pytest.raises(expected, match="Overall_Score"):
        getattr(processor, method_name)(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


@pytest.mark.parametrize(
    ("method_name", "plot_func"),
    [
        ("scenarios_Taylor_Diagram_comparison", "make_scenarios_comparison_Taylor_Diagram"),
        ("scenarios_Target_Diagram_comparison", "make_scenarios_comparison_Target_Diagram"),
    ],
)
def test_core_diagram_station_all_sites_skipped_raises(tmp_path, monkeypatch, method_name, plot_func):
    """Taylor/Target station diagrams should fail when every listed site lacks task input files."""
    import pandas as pd

    import openbench.core.comparison as comparison_module

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ID": "S1",
                "use_syear": 2001,
                "use_eyear": 2002,
                "bias": 0.0,
                "RMSE": 0.0,
                "correlation": 1.0,
                "Overall_Score": 1.0,
            }
        ]
    ).to_csv(metrics_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False)

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Day",
                "num_cores": 1,
                "weight": "none",
            }
        },
        [],
        [],
    )
    monkeypatch.setattr(comparison_module, plot_func, lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="no usable station data"):
        getattr(processor, method_name)(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "stn", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            [],
            [],
            {},
        )


@pytest.mark.parametrize(
    ("method_name", "plot_func"),
    [
        ("scenarios_Kernel_Density_Estimate_comparison", "make_scenarios_comparison_Kernel_Density_Estimate"),
        ("scenarios_Whisker_Plot_comparison", "make_scenarios_comparison_Whisker_Plot"),
        ("scenarios_Ridgeline_Plot_comparison", "make_scenarios_comparison_Ridgeline_Plot"),
    ],
)
def test_core_distribution_empty_filtered_series_raises(tmp_path, monkeypatch, method_name, plot_func):
    """Distribution comparisons should fail before plotting when a requested series has no finite values."""
    import numpy as np
    import xarray as xr

    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    xr.Dataset({"Overall_Score": ("sample", np.array([np.nan, np.nan]))}).to_netcdf(
        scores_dir / "Runoff_ref_RefA_sim_SimA_Overall_Score.nc"
    )
    plot_calls = []
    monkeypatch.setattr(comparison_module, plot_func, lambda *args, **kwargs: plot_calls.append(args))

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(ValueError, match="no finite data"):
        getattr(processor, method_name)(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


@pytest.mark.parametrize(
    ("method_name", "plot_func"),
    [
        ("scenarios_Kernel_Density_Estimate_comparison", "make_scenarios_comparison_Kernel_Density_Estimate"),
        ("scenarios_Whisker_Plot_comparison", "make_scenarios_comparison_Whisker_Plot"),
        ("scenarios_Ridgeline_Plot_comparison", "make_scenarios_comparison_Ridgeline_Plot"),
    ],
)
def test_core_distribution_plot_failures_propagate(tmp_path, monkeypatch, method_name, plot_func):
    """Core distribution comparisons should fail the phase when the renderer fails."""
    import numpy as np
    import xarray as xr

    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    xr.Dataset({"Overall_Score": ("sample", np.array([0.8, 0.9]))}).to_netcdf(
        scores_dir / "Runoff_ref_RefA_sim_SimA_Overall_Score.nc"
    )

    def fail_plot(*args, **kwargs):
        raise RuntimeError("plot boom")

    monkeypatch.setattr(comparison_module, plot_func, fail_plot)

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(RuntimeError, match="plot boom"):
        getattr(processor, method_name)(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_core_parallel_coordinates_empty_station_score_raises(tmp_path, monkeypatch):
    """Parallel Coordinates should not drop an all-empty requested score and still render a partial plot."""
    import pandas as pd

    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    pd.DataFrame({"ID": ["S1", "S2"], "Overall_Score": [float("nan"), float("nan")]}).to_csv(
        scores_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False
    )
    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_parallel_coordinates",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(ValueError, match="no finite data"):
        processor.scenarios_Parallel_Coordinates_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "stn", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


def test_core_parallel_coordinates_preserves_existing_csv_when_rebuild_fails(tmp_path, monkeypatch):
    """A failed rebuild should not replace a previous complete Parallel Coordinates CSV with a partial file."""
    import openbench.core.comparison as comparison_module

    comparison_dir = tmp_path / "comparisons" / "Parallel_Coordinates"
    comparison_dir.mkdir(parents=True)
    output_csv = comparison_dir / "Parallel_Coordinates_evaluations.csv"
    old_content = "Item\tReference\tSimulation\tOverall_Score\nRunoff\tRefA\tOldSim\t0.9\n"
    output_csv.write_text(old_content)
    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_parallel_coordinates",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(FileNotFoundError, match="Runoff_stn_RefA_SimA"):
        processor.scenarios_Parallel_Coordinates_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "stn", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert output_csv.read_text() == old_content
    assert plot_calls == []


def test_core_relative_score_plot_failures_propagate(tmp_path, monkeypatch):
    """Core Relative Score should fail the comparison phase when the renderer fails."""
    import pandas as pd

    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    for sim_source, value in {"SimA": 0.8, "SimB": 0.6}.items():
        pd.DataFrame(
            {
                "ID": ["S1"],
                "Overall_Score": [value],
                "ref_lon": [100.0],
                "ref_lat": [30.0],
            }
        ).to_csv(scores_dir / f"Runoff_stn_RefA_{sim_source}_evaluations.csv", index=False)

    def fail_plot(*args, **kwargs):
        raise RuntimeError("relative plot boom")

    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Relative_Score", fail_plot)

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(RuntimeError, match="relative plot boom"):
        processor.scenarios_Relative_Score_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {
                    "SimA_data_type": "stn",
                    "SimA_varname": "runoff_sim",
                    "SimB_data_type": "stn",
                    "SimB_varname": "runoff_sim",
                },
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "stn", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_core_relative_score_nonfinite_station_result_raises(tmp_path, monkeypatch):
    """Relative Score should not emit all-NaN/inf station relative scores when model spread is zero."""
    import pandas as pd

    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    for sim_source in ["SimA", "SimB"]:
        pd.DataFrame(
            {
                "ID": ["S1"],
                "Overall_Score": [0.8],
                "ref_lon": [100.0],
                "ref_lat": [30.0],
            }
        ).to_csv(scores_dir / f"Runoff_stn_RefA_{sim_source}_evaluations.csv", index=False)
    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_Relative_Score",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(ValueError, match="no finite data"):
        processor.scenarios_Relative_Score_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {
                    "SimA_data_type": "stn",
                    "SimA_varname": "runoff_sim",
                    "SimB_data_type": "stn",
                    "SimB_varname": "runoff_sim",
                },
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "stn", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


def test_core_diff_plot_missing_station_score_input_raises(tmp_path, monkeypatch):
    """Diff Plot should not log-and-render when a requested station score input is absent."""
    import pandas as pd

    import openbench.core.comparison as comparison_module

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    pd.DataFrame(
        {
            "ID": ["S1"],
            "Overall_Score": [0.8],
            "ref_lon": [100.0],
            "ref_lat": [30.0],
        }
    ).to_csv(scores_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False)
    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_Diff_Plot",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        ["Overall_Score"],
        [],
    )

    with pytest.raises(FileNotFoundError, match="SimB"):
        processor.scenarios_Diff_Plot_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {
                    "SimA_data_type": "stn",
                    "SimA_varname": "runoff_sim",
                    "SimB_data_type": "stn",
                    "SimB_varname": "runoff_sim",
                },
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "stn", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )

    assert plot_calls == []


def test_core_diff_plot_station_pairwise_difference_aligns_by_id(tmp_path, monkeypatch):
    """Station Diff Plot pairwise outputs must subtract matching station IDs, not CSV row numbers."""
    import pandas as pd

    import openbench.core.comparison as comparison_module
    from openbench.util.filenames import diff_station_anomaly_filename, diff_station_difference_filename

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        {
            "ID": ["S1", "S2"],
            "bias": [10.0, 20.0],
            "ref_lon": [100.0, 101.0],
            "ref_lat": [30.0, 31.0],
        }
    ).to_csv(metrics_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["S2", "S1"],
            "bias": [1.0, 2.0],
            "ref_lon": [101.0, 100.0],
            "ref_lat": [31.0, 30.0],
        }
    ).to_csv(metrics_dir / "Runoff_stn_RefA_SimB_evaluations.csv", index=False)

    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_Diff_Plot",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        [],
        ["bias"],
    )

    processor.scenarios_Diff_Plot_comparison(
        str(tmp_path),
        {
            "general": {"Runoff_sim_source": ["SimA", "SimB"]},
            "Runoff": {
                "SimA_data_type": "stn",
                "SimA_varname": "runoff_sim_a",
                "SimB_data_type": "stn",
                "SimB_varname": "runoff_sim_b",
            },
        },
        {
            "general": {"Runoff_ref_source": "RefA"},
            "Runoff": {"RefA_data_type": "stn", "RefA_varname": "runoff_ref"},
        },
        ["Runoff"],
        [],
        ["bias"],
        {},
    )

    output = (
        tmp_path
        / "comparisons"
        / "Diff_Plot"
        / diff_station_difference_filename(
            "Runoff",
            "RefA",
            "SimA",
            "runoff_sim_a",
            "SimB",
            "runoff_sim_b",
            "bias",
        )
    )
    result = pd.read_csv(output)

    assert result["ID"].tolist() == ["S1", "S2"]
    assert result["bias_diff"].tolist() == [8.0, 19.0]
    ensemble = pd.read_csv(tmp_path / "comparisons" / "Diff_Plot" / "Runoff_stn_RefA_ensemble_mean_bias.csv")
    anomaly = pd.read_csv(
        tmp_path / "comparisons" / "Diff_Plot" / diff_station_anomaly_filename("Runoff", "RefA", "SimA", "bias")
    )

    assert ensemble["ID"].tolist() == ["S1", "S2"]
    assert ensemble["bias_ensemble_mean"].tolist() == [6.0, 10.5]
    assert anomaly["ID"].tolist() == ["S1", "S2"]
    assert anomaly["bias_anomaly"].tolist() == [4.0, 9.5]
    assert len(plot_calls) == 1


def test_core_portrait_grid_all_nan_metric_raises_before_render(tmp_path, monkeypatch):
    """Portrait seasonal should fail when a requested seasonal metric has no finite values."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.core.comparison as comparison_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    times = pd.date_range("2001-01-01", periods=12, freq="MS")
    field = xr.DataArray(
        np.ones((12, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [30.0], "lon": [100.0]},
        name="runoff",
    )
    field.to_dataset(name="runoff_ref").to_netcdf(data_dir / "Runoff_ref_RefA_runoff_ref.nc")
    field.to_dataset(name="runoff_sim").to_netcdf(data_dir / "Runoff_sim_SimA_runoff_sim.nc")
    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_Portrait_Plot_seasonal",
        lambda *args, **kwargs: plot_calls.append(args),
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        [],
        ["all_nan_metric"],
    )
    processor.all_nan_metric = lambda s, o: xr.full_like(s.isel(time=0, drop=True), np.nan)

    with pytest.raises(ValueError, match="no finite data"):
        processor.scenarios_Portrait_Plot_seasonal_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            [],
            ["all_nan_metric"],
            {},
        )

    assert plot_calls == []


def test_core_portrait_unknown_metric_raises_value_error_not_system_exit(tmp_path, monkeypatch):
    """Portrait seasonal should report unknown metrics as exceptions, not terminate the process."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.core.comparison as comparison_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    times = pd.date_range("2001-01-01", periods=4, freq="QS-DEC")
    coords = {"time": times, "lat": [0.0], "lon": [0.0]}
    xr.Dataset({"runoff_ref": (("time", "lat", "lon"), np.ones((4, 1, 1)))}, coords=coords).to_netcdf(
        data_dir / "Runoff_ref_RefA_runoff_ref.nc"
    )
    xr.Dataset({"runoff_sim": (("time", "lat", "lon"), np.ones((4, 1, 1)))}, coords=coords).to_netcdf(
        data_dir / "Runoff_sim_SimA_runoff_sim.nc"
    )
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_Portrait_Plot_seasonal",
        lambda *args, **kwargs: None,
    )

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
                "num_cores": 1,
            }
        },
        [],
        ["NotAMetric"],
    )

    with pytest.raises(ValueError, match="No such metric: NotAMetric"):
        processor.scenarios_Portrait_Plot_seasonal_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            [],
            ["NotAMetric"],
            {},
        )


def test_core_standard_deviation_missing_grid_input_raises(tmp_path, monkeypatch):
    """Standard Deviation comparison should not silently skip a missing requested grid input."""
    import openbench.core.comparison as comparison_module

    plot_calls = []
    monkeypatch.setattr(comparison_module, "make_Standard_Deviation", lambda *args, **kwargs: plot_calls.append(args))

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        [],
        [],
    )

    with pytest.raises(FileNotFoundError, match="required simulation input is missing"):
        processor.scenarios_Standard_Deviation_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
            },
            {
                "general": {"Runoff_ref_source": "RefA"},
                "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
            },
            ["Runoff"],
            [],
            [],
            {},
        )

    assert plot_calls == []


def test_core_correlation_plot_failures_propagate(tmp_path, monkeypatch):
    """Correlation comparison should fail the phase when its renderer fails."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.core.comparison as comparison_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    times = pd.date_range("2001-01-01", periods=3, freq="MS")
    coords = {"time": times, "lat": [30.0], "lon": [100.0]}
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.array([[[1.0]], [[2.0]], [[3.0]]]))},
        coords=coords,
    ).to_netcdf(data_dir / "Runoff_sim_SimA_runoff_sim.nc")
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.array([[[1.5]], [[2.5]], [[3.5]]]))},
        coords=coords,
    ).to_netcdf(data_dir / "Runoff_sim_SimB_runoff_sim.nc")

    def fail_plot(*args, **kwargs):
        raise RuntimeError("correlation plot boom")

    monkeypatch.setattr(comparison_module, "make_Correlation", fail_plot)

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
            }
        },
        [],
        [],
    )

    with pytest.raises(RuntimeError, match="correlation plot boom"):
        processor.scenarios_Correlation_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA", "SimB"]},
                "Runoff": {
                    "SimA_data_type": "grid",
                    "SimA_varname": "runoff_sim",
                    "SimB_data_type": "grid",
                    "SimB_varname": "runoff_sim",
                },
            },
            {"general": {}},
            ["Runoff"],
            [],
            [],
            {},
        )


def test_post_phase_output_filter_ignores_unreadable_outputs(tmp_path):
    """Post-phase filters should not pass variables whose only outputs are corrupt."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path
    (output_dir / "metrics").mkdir()
    (output_dir / "metrics" / "Runoff_ref_RefA_sim_SimA_bias.nc").write_text("not a netcdf")
    _write_fake_grid_outputs(output_dir, "ET", "RefA", "SimA", metrics=("bias",), scores=())

    filtered = local_runner._filter_evaluation_items_with_outputs(output_dir, ["Runoff", "ET"])

    assert filtered == ["ET"]


def test_partially_cached_variable_with_unified_mask_off_skips_cached_task_preprocessing(tmp_path, monkeypatch):
    """With unified_mask disabled, a cached task should not touch source data even if sibling tasks rerun."""
    import json

    import openbench.config.adapter as adapter
    import openbench.data.processing as processing
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache, make_cache_key

    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            unified_mask=False,
            generate_report=False,
        ),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "TestRef"}),
        simulation={
            "SimA": SimulationEntry(model="ModelA", root_dir=str(tmp_path)),
            "SimB": SimulationEntry(model="ModelB", root_dir=str(tmp_path)),
        },
        comparison=ComparisonConfig(enabled=False, items=[]),
        statistics=StatisticsConfig(enabled=False, items=[]),
    )
    output_dir = tmp_path / "case"
    _write_fake_grid_outputs(output_dir, "Runoff", "TestRef", "SimA", metrics=("bias",), scores=("Overall_Score",))

    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "unified_mask": False,
            "time_alignment": "intersection",
            "generate_report": False,
            "only_drawing": False,
            "weight": "area",
        },
    )
    namelists = adapter.LegacyNamelists(
        main={"general": runner_cfg.general},
        reference={
            "general": {"Runoff_ref_source": "TestRef"},
            "Runoff": {"TestRef_varname": "runoff_ref", "TestRef_data_type": "grid"},
        },
        simulation={
            "general": {"Runoff_sim_source": ["SimA", "SimB"]},
            "Runoff": {
                "SimA_varname": "runoff_sim",
                "SimA_data_type": "grid",
                "SimB_varname": "runoff_sim",
                "SimB_data_type": "grid",
            },
        },
    )

    class Bindings:
        def __init__(self):
            self.runner_cfg = runner_cfg
            self.namelists = namelists
            self.figures = adapter.LegacyFigureConfig(raw={})

        def iter_task_sources(self, variables):
            return [
                adapter.EvaluationSource("Runoff", "SimA", "TestRef"),
                adapter.EvaluationSource("Runoff", "SimB", "TestRef"),
            ]

        def build_runtime_info_for(self, var_name, sim_source, ref_source):
            return {
                "casedir": str(output_dir),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "ref_data_type": "grid",
                "sim_data_type": "grid",
            }

        def has_grid_evaluation(self, variables):
            return adapter.GridEvaluationEvidence(False)

    bindings = Bindings()
    sim_a_hash = EvaluationCache.hash_config(
        local_runner._task_hash_payload(
            cfg=cfg,
            bindings=bindings,
            var_name="Runoff",
            sim_source="SimA",
            ref_source="TestRef",
            metric_vars=["bias"],
            score_vars=["Overall_Score"],
            comparison_vars=[],
            statistic_vars=[],
        )
    )
    (output_dir / ".openbench_cache.json").write_text(
        json.dumps({make_cache_key("Runoff", "SimA", "TestRef"): sim_a_hash})
    )

    preprocessed_sims = []

    class FakeProcessor:
        def __init__(self, info):
            self.info = info
            if info["sim_source"] == "SimA":
                raise AssertionError("cached SimA should not be preprocessed")

        def prepare_source(self, datasource):
            preprocessed_sims.append((self.info["sim_source"], datasource))

    evaluated_sims = []

    def fake_evaluate(task):
        evaluated_sims.append(task["sim_source"])
        return {
            "variable": task["var_name"],
            "sim": task["sim_source"],
            "ref": task["ref_source"],
            "status": "success",
            "cache_key": task["cache_key"],
            "config_hash": task["config_hash"],
            "skipped": False,
        }

    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)
    monkeypatch.setattr(processing, "DatasetProcessing", FakeProcessor)
    monkeypatch.setattr(local_runner, "_evaluate_single", fake_evaluate)
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])

    result = run_evaluation(cfg)

    assert result["status"] == "success", result.get("errors")
    assert {item["sim"]: item["skipped"] for item in result["evaluated"]} == {"SimA": True, "SimB": False}
    assert evaluated_sims == ["SimB"]
    assert all(sim == "SimB" for sim, _ in preprocessed_sims)


def test_station_output_preflight_requires_requested_columns(tmp_path):
    """Station CSV outputs are shared files, so preflight must verify requested columns, not only file existence."""
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    metrics_dir = output_dir / "metrics"
    scores_dir = output_dir / "scores"
    metrics_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)
    filename = "Runoff_stn_TestRef_SimA_evaluations.csv"
    (metrics_dir / filename).write_text("ID,RMSE\nS1,1.0\n")
    (scores_dir / filename).write_text("ID,Overall_Score\nS1,0.8\n")

    task = {
        "var_name": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "output_requirements": {
            "metrics": ["bias"],
            "scores": ["Overall_Score"],
            "ref_data_type": "stn",
            "sim_data_type": "grid",
        },
    }

    missing = local_runner._missing_expected_outputs(output_dir, task)

    assert metrics_dir / filename in missing
    assert scores_dir / filename not in missing


def test_output_type_detection_uses_namelists_without_runtime_info_side_effects(tmp_path, monkeypatch):
    """Output preflight should derive grid/station naming from namelists, not station-file readers."""
    import openbench.config.adapter as adapter_module
    import openbench.runner.local as local_runner

    output_dir = tmp_path / "case"
    _write_fake_station_outputs(
        output_dir,
        "Runoff",
        "DemoRef",
        "CaseA",
        metrics=("bias",),
        scores=("Overall_Score",),
    )
    bindings = type(
        "Bindings",
        (),
        {
            "namelists": adapter_module.LegacyNamelists(
                main={"general": {}},
                reference={
                    "general": {"Runoff_ref_source": "DemoRef"},
                    "Runoff": {"DemoRef_data_type": "grid"},
                },
                simulation={
                    "general": {"Runoff_sim_source": ["CaseA"]},
                    "Runoff": {"CaseA_data_type": "stn"},
                },
            )
        },
    )()

    def fail_runtime_info(task):
        raise AssertionError("output type detection should not build runtime info")

    monkeypatch.setattr(local_runner, "_build_bridge_runtime_info", fail_runtime_info)

    missing = local_runner._missing_expected_outputs(
        output_dir,
        {
            "var_name": "Runoff",
            "sim_source": "CaseA",
            "ref_source": "DemoRef",
            "bindings": bindings,
            "output_requirements": {"metrics": ["bias"], "scores": ["Overall_Score"]},
        },
    )

    assert missing == []


def test_start_optional_dask_client_uses_project_dask_config(monkeypatch):
    import sys
    import types

    import openbench.runner.local as local_runner

    calls = {}

    class FakeCluster:
        def __init__(self, **kwargs):
            calls["cluster_kwargs"] = kwargs

        def close(self):
            calls["cluster_closed"] = True

    class FakeClient:
        dashboard_link = None

        def __init__(self, cluster, **kwargs):
            calls["client_cluster"] = cluster
            calls["client_kwargs"] = kwargs

        def close(self):
            calls["client_closed"] = True

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.Client = FakeClient
    fake_distributed.LocalCluster = FakeCluster
    monkeypatch.setitem(sys.modules, "distributed", fake_distributed)
    for name in (
        "OPENBENCH_DASK",
        "OPENBENCH_DASK_DISTRIBUTED",
        "OPENBENCH_DASK_N_WORKERS",
        "OPENBENCH_DASK_THREADS_PER_WORKER",
        "OPENBENCH_DASK_PROCESSES",
        "OPENBENCH_DASK_MEMORY_LIMIT",
        "OPENBENCH_DASK_DASHBOARD_ADDRESS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = DaskConfig(
        enabled=True,
        n_workers=3,
        threads_per_worker=2,
        processes=False,
        memory_limit="2GB",
        dashboard_address=":0",
    )

    handle = local_runner._start_optional_dask_client(8, local_directory="/tmp/dask", dask_config=config)
    assert handle is not None
    local_runner._close_optional_dask_client(handle)

    assert calls["cluster_kwargs"]["n_workers"] == 3
    assert calls["cluster_kwargs"]["threads_per_worker"] == 2
    assert calls["cluster_kwargs"]["processes"] is False
    assert calls["cluster_kwargs"]["memory_limit"] == "2GB"
    assert calls["cluster_kwargs"]["dashboard_address"] == ":0"
    assert calls["cluster_kwargs"]["local_directory"] == "/tmp/dask"


def test_dask_env_overrides_project_dask_disabled(monkeypatch):
    import openbench.runner.local as local_runner

    monkeypatch.setenv("OPENBENCH_DASK", "1")
    assert local_runner._dask_distributed_requested(DaskConfig(enabled=False)) is True

    monkeypatch.setenv("OPENBENCH_DASK", "0")
    assert local_runner._dask_distributed_requested(DaskConfig(enabled=True)) is False


def test_project_dask_local_directory_prefers_yaml(tmp_path, monkeypatch):
    import openbench.runner.local as local_runner

    cfg = _make_cfg(tmp_path)
    cfg.project.dask.local_directory = "~/openbench-dask"
    monkeypatch.delenv("OPENBENCH_DASK_LOCAL_DIRECTORY", raising=False)

    assert local_runner._project_dask_local_directory(cfg).endswith("openbench-dask")


def test_task_config_hash_includes_openbench_algorithm_version(tmp_path, monkeypatch):
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "regrid_backend": "openbench_conservative",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={"Runoff": {"SimA_varname": "runoff_sim"}},
            ),
        },
    )()

    def cache_hash():
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    monkeypatch.setattr(local_runner, "_openbench_version", lambda: "3.0.0")
    first = cache_hash()
    monkeypatch.setattr(local_runner, "_openbench_version", lambda: "3.0.1")

    assert cache_hash() != first


def test_task_config_hash_includes_selected_regrid_backend(tmp_path):
    import openbench.config.adapter as adapter
    import openbench.runner.local as local_runner
    from openbench.runner.cache import EvaluationCache

    cfg = _make_cfg(tmp_path, comparison_enabled=False)
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "regrid_backend": "openbench_conservative",
            "only_drawing": False,
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={"Runoff": {"SimA_varname": "runoff_sim"}},
            ),
        },
    )()

    def cache_hash():
        return EvaluationCache.hash_config(
            local_runner._task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
            )
        )

    first = cache_hash()
    runner_cfg.general["regrid_backend"] = "xesmf_conservative"

    assert cache_hash() != first
