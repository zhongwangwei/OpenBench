"""Tests for runner.local orchestration and status reporting."""

from __future__ import annotations

from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
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
            name="case", output_dir=str(tmp_path), years=[2000, 2001],
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
        "unified_mask",
    }


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
            return None

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
            return None

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
                "build_runtime_info_for": lambda self, var_name, sim_source, ref_source: build_calls.append(
                    (var_name, sim_source, ref_source)
                ) or adapter_module.BridgeRuntimeInfo(
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
            return None

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
    monkeypatch.setattr(local_runner, "_evaluate_single", lambda task: {
        "variable": task["var_name"],
        "sim": task["sim_source"],
        "ref": task["ref_source"],
        "status": "success",
        "skipped": False,
        "cache_key": task["cache_key"],
        "config_hash": task["config_hash"],
    })
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
                "iter_task_sources": lambda self, variables: [
                    adapter.EvaluationSource("Runoff", "SimA", "TestRef")
                ],
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
    monkeypatch.setattr(local_runner, "_evaluate_single", lambda task: {
        "variable": task["var_name"],
        "sim": task["sim_source"],
        "ref": task["ref_source"],
        "status": "success",
        "skipped": False,
        "cache_key": task["cache_key"],
        "config_hash": task["config_hash"],
    })
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
            "iter_task_sources": lambda self, variables: [
                adapter.EvaluationSource("Runoff", "SimA", "TestRef")
            ],
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
    monkeypatch.setattr(local_runner, "_evaluate_single", lambda task: captured_tasks.append(task) or {
        "variable": task["var_name"],
        "sim": task["sim_source"],
        "ref": task["ref_source"],
        "status": "success",
        "skipped": False,
        "cache_key": task["cache_key"],
        "config_hash": task["config_hash"],
    })
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
    metrics_dir = tmp_path / "case" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "Runoff_ref_TestRef_sim_SimA_bias.nc").write_text("placeholder")

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
            "iter_task_sources": lambda self, variables: [
                adapter.EvaluationSource("Runoff", "SimA", "TestRef")
            ],
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
            "iter_task_sources": lambda self, variables: [
                adapter.EvaluationSource("Runoff", "SimA", "TestRef")
            ],
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
    monkeypatch.setattr(local_runner, "_evaluate_single", lambda task: {
        "variable": task["var_name"],
        "sim": task["sim_source"],
        "ref": task["ref_source"],
        "status": "success",
        "skipped": False,
        "cache_key": task["cache_key"],
        "config_hash": task["config_hash"],
    })
    monkeypatch.setattr(local_runner, "_run_groupby", lambda *args, **kwargs: [])
    monkeypatch.setattr(local_runner, "_run_statistics", lambda *args: statistics_calls.append(args) or [])

    result = run_evaluation(cfg)

    assert result["status"] == "success"
    assert len(statistics_calls) == 1
    args = statistics_calls[0]
    assert len(args) == 2
    assert args[0] is bindings
    assert args[1] == ["Mean"]


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
            "iter_task_sources": lambda self, variables: [
                adapter.EvaluationSource("Runoff", "SimA", "TestRef")
            ],
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
    monkeypatch.setattr(local_runner, "_evaluate_single", lambda task: {
        "variable": task["var_name"],
        "sim": task["sim_source"],
        "ref": task["ref_source"],
        "status": "success",
        "skipped": False,
        "cache_key": task["cache_key"],
        "config_hash": task["config_hash"],
    })
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
    # disk; create a dummy output for "Runoff" so the filter keeps it.
    case_dir = tmp_path / "case"
    scores_dir = case_dir / "scores"
    scores_dir.mkdir(parents=True)
    (scores_dir / "Runoff_dummy_score.nc").write_text("placeholder")

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
            groupby_calls.append(
                ("cz", basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, cz_fig)
            )

    monkeypatch.setattr(landcover_module, "LC_groupby", FakeLCGroupby)
    monkeypatch.setattr(climatezone_module, "CZ_groupby", FakeCZGroupby)

    errors = local_runner._run_groupby(cfg, bindings, tmp_path / "case")

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


def test_post_phase_errors_are_reported_in_results(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    legacy = _legacy_payload(tmp_path)
    main_nl, ref_nml, sim_nml = _namelists(tmp_path)
    metrics_dir = tmp_path / "case" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "Runoff_ref_TestRef_sim_SimA_bias.nc").write_text("placeholder")

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
            name="case", output_dir=str(tmp_path), years=[2000, 2001],
            generate_report=False, unified_mask=False,
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
            "basename": "case", "basedir": str(tmp_path),
            "num_cores": 1, "unified_mask": False,
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
            "basename": "case", "basedir": str(tmp_path),
            "num_cores": 1, "unified_mask": False,
            "compare_tim_res": "Month", "compare_grid_res": 0.5,
            "compare_tzone": 0, "syear": 2000, "eyear": 2001,
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
            "ref_varname": "rv", "sim_varname": "sv",
            "ref_data_type": "grid", "sim_data_type": "grid",
            "ref_source": task["ref_source"], "sim_source": task["sim_source"],
        }

    class FakeProcessor:
        def __init__(self, info):
            self._ref_source = info["ref_source"]
            self._sim_source = info["sim_source"]

        def prepare_source(self, datasource):
            prepare_calls.append((datasource, self._ref_source, self._sim_source))

    class FakeGridEvaluation:
        def __init__(self, info, fig_nml): pass
        def make_Evaluation(self): return None

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
    assert ("ref", "RefB") in ref_calls, (
        "RefB was not preprocessed — multi-ref regression: "
        f"got {sorted(ref_calls)}"
    )
    # And exactly one ref preprocess per ref_source (not duplicated)
    assert len([c for c in prepare_calls if c[0] == "ref" and c[1] == "RefA"]) == 1
    assert len([c for c in prepare_calls if c[0] == "ref" and c[1] == "RefB"]) == 1

    # Sim should be preprocessed for every (sim, ref) task: 4 calls
    sim_calls = [c for c in prepare_calls if c[0] == "sim"]
    assert len(sim_calls) == 4, f"expected 4 sim preprocess calls (2 sim × 2 ref), got {len(sim_calls)}"
