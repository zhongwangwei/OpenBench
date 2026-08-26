"""Tests for config adapter — new format to legacy format bridge."""

from types import SimpleNamespace

import pytest

import openbench.config.adapter as adapter_module
from openbench.config.adapter import to_legacy_config
from openbench.config.loader import ConfigError
from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
    StatisticsConfig,
)


def test_minimal_config_adapter():
    """Convert minimal new config to legacy format."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="test", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a"}),
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data")},
    )
    legacy = to_legacy_config(cfg)

    assert legacy["general"]["basename"] == "test"
    assert legacy["general"]["basedir"] == "./output"
    assert legacy["general"]["syear"] == 2004
    assert legacy["general"]["eyear"] == 2010
    assert "Evapotranspiration" in legacy["evaluation_items"]
    assert legacy["evaluation_items"]["Evapotranspiration"] is True


def test_build_runner_config_returns_typed_inputs():
    """Adapter should expose runner-facing typed inputs alongside the legacy wrapper."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="runner", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a"}),
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data", tim_res="Day", grid_res=0.25)},
    )

    runner_cfg = adapter_module.build_runner_config(cfg)

    assert runner_cfg.basename == "runner"
    assert runner_cfg.basedir == "./output"
    assert runner_cfg.compare_tim_res == "Day"
    assert runner_cfg.compare_grid_res == 0.25
    assert runner_cfg.compare_tzone == 0
    assert runner_cfg.evaluation_items == {"Evapotranspiration": True}
    assert runner_cfg.metrics == ["bias", "RMSE", "correlation"]
    assert runner_cfg.scores == ["Overall_Score"]
    assert runner_cfg.comparisons == ["Taylor_Diagram", "HeatMap"]
    assert runner_cfg.statistics == []


def test_build_runner_config_preserves_explicit_empty_scores():
    """Omitted scores keep the default, but ``scores: []`` disables scores."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="runner", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a"}),
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data")},
        scores=[],
    )

    runner_cfg = adapter_module.build_runner_config(cfg)
    legacy = to_legacy_config(cfg)

    assert runner_cfg.scores == []
    assert legacy["scores"] == {}


def test_build_runner_config_uses_model_profile_resolution_when_entry_missing(monkeypatch):
    """Runner target resolution should match model-profile fallback used by sim bindings."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="runner", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a"}),
        simulation={"CaseA": SimulationEntry(model="ProfileModel", root_dir="/data")},
    )
    registry = SimpleNamespace(get_model=lambda name: SimpleNamespace(tim_res="Day", grid_res=0.25))
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)

    runner_cfg = adapter_module.build_runner_config(cfg)

    assert runner_cfg.compare_tim_res == "Day"
    assert runner_cfg.compare_grid_res == 0.25


def test_adapter_preserves_inline_sim_variable_convert(monkeypatch):
    """Inline variable-level convert expressions must reach legacy sim_nml."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="convert", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["GPP"]),
        reference=ReferenceConfig(sources={"GPP": "FLUXCOM"}),
        simulation={
            "CaseA": SimulationEntry(
                model="InlineModel",
                root_dir="/data",
                variables={"GPP": {"varname": "gpp_raw", "varunit": "mol m-2 s-1", "convert": "value * 12.011"}},
            )
        },
    )
    registry = SimpleNamespace(
        get_model=lambda name: None,
        get_resolution_variants=lambda name: {},
        get_reference=lambda name, **kwargs: None,
    )
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    _main, _ref, sim = adapter_module.build_legacy_namelists(cfg)

    assert sim["GPP"]["CaseA_convert"] == "value * 12.011"


def test_build_runner_config_rejects_model_profile_resolution_conflict(monkeypatch):
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="runner", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "Monthly": SimulationEntry(model="MonthlyModel", root_dir="/monthly"),
            "Daily": SimulationEntry(model="DailyModel", root_dir="/daily"),
        },
    )
    models = {
        "monthlymodel": SimpleNamespace(tim_res="Month", grid_res=0.5),
        "dailymodel": SimpleNamespace(tim_res="Day", grid_res=0.25),
    }
    registry = SimpleNamespace(get_model=lambda name: models.get(name.lower()))
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)

    with pytest.raises(ConfigError, match="ambiguous across simulations"):
        adapter_module.build_runner_config(cfg)


def test_build_runner_config_rejects_project_name_path_with_config_error():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="../../escape", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a"}),
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data")},
    )

    with pytest.raises(ConfigError, match="project.name must be a simple directory name"):
        adapter_module.build_runner_config(cfg)


def test_adapter_declares_legacy_general_key_contract():
    """The adapter should publish the exact legacy general keys it owns."""
    assert adapter_module.LEGACY_GENERAL_KEYS == {
        "basename",
        "basedir",
        "syear",
        "eyear",
        "min_year",
        "min_lat",
        "max_lat",
        "min_lon",
        "max_lon",
        "num_cores",
        "evaluation",
        "comparison",
        "statistics",
        "debug_mode",
        "only_drawing",
        "IGBP_groupby",
        "PFT_groupby",
        "Climate_zone_groupby",
        "unified_mask",
        "time_alignment",
        "generate_report",
        "weight",
        "compare_tim_res",
        "compare_tzone",
        "compare_grid_res",
        "regrid_backend",
    }


def test_full_config_adapter(monkeypatch):
    """Convert full config with all options."""
    monkeypatch.setattr(adapter_module, "effective_cpu_count", lambda count: count)
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="full",
            output_dir="/data/out",
            years=[2000, 2020],
            min_year_threshold=5,
            lat_range=[-60, 90],
            lon_range=[-180, 180],
            tim_res="Month",
            grid_res=0.5,
            num_cores=8,
            time_alignment="per_pair",
        ),
        evaluation=EvaluationConfig(variables=["GPP", "Latent_Heat"]),
        reference=ReferenceConfig(sources={"GPP": "FLUXCOM", "Latent_Heat": "FLUXCOM"}),
        simulation={
            "CoLM": SimulationEntry(model="CoLM2024", root_dir="/d1"),
            "CLM": SimulationEntry(model="CLM5", root_dir="/d2"),
        },
        metrics=["bias", "RMSE"],
        scores=["nBiasScore"],
        comparison=ComparisonConfig(enabled=True, items=["Taylor_Diagram"]),
        statistics=StatisticsConfig(enabled=True, items=["ANOVA"]),
    )
    legacy = to_legacy_config(cfg)
    bindings = adapter_module.build_runner_bindings(cfg)
    stats_ctx = bindings.build_statistics_context(["ANOVA"], ["GPP"])

    assert legacy["general"]["num_cores"] == 8
    assert legacy["general"]["comparison"] is True
    assert legacy["general"]["statistics"] is True
    assert legacy["general"]["min_year"] == 5
    assert legacy["metrics"] == {"bias": True, "RMSE": True}
    assert legacy["scores"] == {"nBiasScore": True}
    assert legacy["comparisons"] == {"Taylor_Diagram": True}
    assert legacy["statistics"] == {"ANOVA": True}
    assert stats_ctx.stats_nml["ANOVA"]["analysis_type"] == "oneway"


def test_defaults_applied():
    """Verify sensible defaults when options not specified."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="t", output_dir=".", years=[2000, 2010]),
        evaluation=EvaluationConfig(variables=["GPP"]),
        reference=ReferenceConfig(sources={"GPP": "FLUXCOM"}),
        simulation={"M": SimulationEntry(model="M", root_dir="/d")},
    )
    legacy = to_legacy_config(cfg)

    assert legacy["general"]["num_cores"] >= 1
    assert legacy["general"]["unified_mask"] is True
    assert legacy["general"]["generate_report"] is True
    assert legacy["general"]["comparison"] is False
    assert "bias" in legacy["metrics"]  # Default metrics


def test_effective_target_resolution_follows_shared_simulation_context():
    """Legacy compare_* settings should inherit the effective target resolution."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="shared-res", output_dir=".", years=[2000, 2010]),
        evaluation=EvaluationConfig(variables=["GPP"]),
        reference=ReferenceConfig(sources={"GPP": "FLUXCOM"}),
        simulation={
            "SimA": SimulationEntry(model="M", root_dir="/d1", tim_res="Day", grid_res=0.25),
            "SimB": SimulationEntry(model="M", root_dir="/d2", tim_res="Day", grid_res=0.25),
        },
    )

    legacy = to_legacy_config(cfg)

    assert legacy["general"]["compare_tim_res"] == "Day"
    assert legacy["general"]["compare_grid_res"] == 0.25


def test_explicit_comparison_resolution_overrides_simulation_context():
    """Explicit project-level resolution should remain the source of truth for legacy compare_* fields."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="explicit-res",
            output_dir=".",
            years=[2000, 2010],
            tim_res="Month",
            grid_res=0.5,
        ),
        evaluation=EvaluationConfig(variables=["GPP"]),
        reference=ReferenceConfig(sources={"GPP": "FLUXCOM"}),
        simulation={
            "SimA": SimulationEntry(model="M", root_dir="/d1", tim_res="Day", grid_res=0.25),
            "SimB": SimulationEntry(model="M", root_dir="/d2", tim_res="Day", grid_res=0.25),
        },
    )

    legacy = to_legacy_config(cfg)

    assert legacy["general"]["compare_tim_res"] == "Month"
    assert legacy["general"]["compare_grid_res"] == 0.5


def test_conflicting_simulation_resolution_requires_explicit_comparison_context():
    """Adapter should reject ambiguous target resolution the same way resolver does."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="ambiguous-res", output_dir=".", years=[2000, 2010]),
        evaluation=EvaluationConfig(variables=["GPP"]),
        reference=ReferenceConfig(sources={"GPP": "FLUXCOM"}),
        simulation={
            "SimA": SimulationEntry(model="M", root_dir="/d1", tim_res="Day", grid_res=0.25),
            "SimB": SimulationEntry(model="M", root_dir="/d2", tim_res="Month", grid_res=0.5),
        },
    )

    with pytest.raises(ConfigError, match="ambiguous across simulations"):
        to_legacy_config(cfg)


def test_runner_bindings_build_runtime_info_is_self_contained(monkeypatch):
    """RunnerBindings should not depend on the free build_runtime_info helper anymore."""
    import openbench.config.runtime_info as runtime_info

    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": {}},
            reference={"general": {}},
            simulation={"general": {}},
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    class FakeInfoReader:
        def __init__(self, **kwargs):
            self.item = kwargs["item"]
            self.sim_source = kwargs["sim_source"]
            self.ref_source = kwargs["ref_source"]
            self.metric_vars = kwargs["metric_vars"]
            self.score_vars = kwargs["score_vars"]
            self.comparison_vars = kwargs["comparison_vars"]
            self.statistic_vars = kwargs["statistic_vars"]

    monkeypatch.setattr(runtime_info, "GeneralInfoReader", FakeInfoReader)

    info = bindings.build_runtime_info_for("Runoff", "SimA", "TestRef")

    assert not hasattr(adapter_module, "build_runtime_info")
    assert isinstance(info, adapter_module.BridgeRuntimeInfo)
    assert info.to_info() == {
        "item": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "metric_vars": ["bias"],
        "score_vars": ["Overall_Score"],
        "comparison_vars": ["HeatMap"],
        "statistic_vars": ["Mean"],
    }


def test_runner_bindings_build_runtime_info_without_reader_to_dict(monkeypatch):
    """Adapter should snapshot public reader attrs without depending on GeneralInfoReader.to_dict()."""
    import openbench.config.runtime_info as runtime_info

    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": {}},
            reference={"general": {}},
            simulation={"general": {}},
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    class FakeInfoReader:
        def __init__(self, **kwargs):
            self.item = kwargs["item"]
            self.sim_source = kwargs["sim_source"]
            self.ref_source = kwargs["ref_source"]
            self.metric_vars = kwargs["metric_vars"]
            self.score_vars = kwargs["score_vars"]
            self.comparison_vars = kwargs["comparison_vars"]
            self.statistic_vars = kwargs["statistic_vars"]
            self._private = "ignore-me"

    monkeypatch.setattr(runtime_info, "GeneralInfoReader", FakeInfoReader)

    info = bindings.build_runtime_info_for("Runoff", "SimA", "TestRef")

    assert info.to_info() == {
        "item": "Runoff",
        "sim_source": "SimA",
        "ref_source": "TestRef",
        "metric_vars": ["bias"],
        "score_vars": ["Overall_Score"],
        "comparison_vars": ["HeatMap"],
        "statistic_vars": ["Mean"],
    }


def test_bridge_runtime_info_snapshots_input_payload():
    """Typed runtime payload should snapshot mutable reader state at the adapter boundary."""
    payload = {
        "casedir": "/tmp/case",
        "ref_varname": "runoff_ref",
    }

    info = adapter_module.BridgeRuntimeInfo(payload=payload)
    payload["ref_varname"] = "mutated"

    assert info.to_info() == {
        "casedir": "/tmp/case",
        "ref_varname": "runoff_ref",
    }


def test_runner_bindings_build_typed_phase_contexts():
    """Phase helpers should receive typed context objects instead of raw dict payloads."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
            simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        ),
        figures=adapter_module.LegacyFigureConfig(
            raw={
                "Comparison": {"HeatMap": {"title": "heat"}},
                "Validation": {"title": "validation"},
                "Climate_zone_groupby": {"title": "cz"},
                "Statistic": {"Mean": {"title": "mean"}},
            }
        ),
    )

    comparison = bindings.build_comparison_context()
    groupby = bindings.build_groupby_context()
    statistics = bindings.build_statistics_context(["Mean"])
    report = bindings.build_report_config()
    evaluation_fig = bindings.build_evaluation_fig_nml()

    assert isinstance(bindings.namelists, adapter_module.LegacyNamelists)
    assert bindings.namelists.main == {"general": runner_cfg.general}
    assert bindings.namelists.reference == {
        "general": {"Runoff_ref_source": "TestRef"},
        "Runoff": {"TestRef_data_type": "grid"},
    }
    assert bindings.namelists.simulation == {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {"SimA_data_type": "grid"},
    }
    assert isinstance(bindings.figures, adapter_module.LegacyFigureConfig)
    assert bindings.figures.raw["Comparison"] == {"HeatMap": {"title": "heat"}}
    assert bindings.figures.validation == {"title": "validation"}
    assert bindings.figures.statistics == {"Mean": {"title": "mean"}}
    assert bindings.figures.climate_zone_groupby == {"title": "cz"}
    assert not hasattr(bindings, "main_nl")
    assert not hasattr(bindings, "ref_nml")
    assert not hasattr(bindings, "sim_nml")
    assert not hasattr(bindings, "fig_nml")

    assert isinstance(comparison, adapter_module.ComparisonContext)
    assert isinstance(comparison.namelists, adapter_module.LegacyNamelists)
    assert comparison.namelists is bindings.namelists
    assert comparison.evaluation_items == ["Runoff"]
    assert comparison.comparison_fig == {"HeatMap": {"title": "heat"}}

    assert isinstance(groupby, adapter_module.GroupbyContext)
    assert isinstance(groupby.namelists, adapter_module.LegacyNamelists)
    assert groupby.namelists is bindings.namelists
    assert groupby.validation_fig == {"title": "validation"}
    assert groupby.climate_zone_fig == {"title": "cz"}

    assert isinstance(statistics, adapter_module.StatisticsContext)
    assert isinstance(statistics.namelists, adapter_module.LegacyNamelists)
    assert statistics.namelists is bindings.namelists
    assert statistics.num_cores == 1
    assert statistics.statistic_fig == {"Mean": {"title": "mean"}}

    assert isinstance(report, adapter_module.ReportContext)
    assert report.evaluation_items == ["Runoff"]
    assert report.runner_cfg is runner_cfg
    assert report.namelists is bindings.namelists
    assert report.to_report_config()["ref_nml"] == {
        "general": {"Runoff_ref_source": "TestRef"},
        "Runoff": {"TestRef_data_type": "grid"},
    }

    assert isinstance(evaluation_fig, adapter_module.EvaluationFigureContext)
    assert evaluation_fig.figures is bindings.figures
    assert evaluation_fig.to_fig_nml()["Comparison"] == {"HeatMap": {"title": "heat"}}


def test_runner_bindings_iter_task_sources_returns_typed_entries():
    """Task-source iteration should use typed entries instead of positional tuples."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
            simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    sources = bindings.iter_task_sources(["Runoff"])

    assert sources == [
        adapter_module.EvaluationSource(
            var_name="Runoff",
            sim_source="SimA",
            ref_source="TestRef",
        )
    ]


def test_runner_bindings_has_grid_evaluation_returns_typed_evidence():
    """Grid applicability should return typed evidence instead of a bare bool."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=["HeatMap"],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"Runoff_ref_source": "TestRef"}, "Runoff": {"TestRef_data_type": "grid"}},
            simulation={"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "grid"}},
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    evidence = bindings.has_grid_evaluation(["Runoff"])

    assert isinstance(evidence, adapter_module.GridEvaluationEvidence)
    assert evidence.has_grid is True


def test_iter_task_sources_multi_ref_cartesian_product():
    """When ref_source is a list, every (ref, sim) pair becomes a task.

    Regression: v3.0a1 lost v2.x's multi-reference capability when the
    schema was tightened to dict[str, str]. The fix restores str|list[str]
    on the schema, loader, resolver, and adapter; this test pins the
    Cartesian product behavior at the adapter level.
    """
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={
                "general": {"ET_ref_source": ["GLEAM", "FLUXCOM"]},
                "ET": {"GLEAM_data_type": "grid", "FLUXCOM_data_type": "grid"},
            },
            simulation={
                "general": {"ET_sim_source": ["SimA", "SimB"]},
                "ET": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
            },
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    sources = bindings.iter_task_sources(["ET"])

    # 2 sims × 2 refs = 4 tasks
    assert len(sources) == 4
    pairs = {(s.sim_source, s.ref_source) for s in sources}
    assert pairs == {
        ("SimA", "GLEAM"),
        ("SimB", "GLEAM"),
        ("SimA", "FLUXCOM"),
        ("SimB", "FLUXCOM"),
    }


def test_iter_task_sources_single_ref_str_works_unchanged():
    """Regression guard: single-string ref_source must still produce 1 task per sim."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"ET_ref_source": "GLEAM"}, "ET": {"GLEAM_data_type": "grid"}},
            simulation={
                "general": {"ET_sim_source": ["SimA", "SimB"]},
                "ET": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
            },
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    sources = bindings.iter_task_sources(["ET"])
    assert {(s.sim_source, s.ref_source) for s in sources} == {
        ("SimA", "GLEAM"),
        ("SimB", "GLEAM"),
    }


def test_has_grid_evaluation_full_cartesian_with_mixed_sim_types():
    """A later grid-grid pair must make statistics applicable."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={
                "general": {"ET_ref_source": "RefGrid"},
                "ET": {"RefGrid_data_type": "grid"},
            },
            simulation={
                # Mixed sim types: stn first, grid second (the regression scenario)
                "general": {"ET_sim_source": ["SimStn", "SimGrid"]},
                "ET": {"SimStn_data_type": "stn", "SimGrid_data_type": "grid"},
            },
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    evidence = bindings.has_grid_evaluation(["ET"])
    assert evidence.has_grid is True


def test_has_grid_evaluation_rejects_station_grid_pairs():
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={"basename": "case", "basedir": ".", "syear": 2000, "eyear": 2001},
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"ET_ref_source": "RefStn"}, "ET": {"RefStn_data_type": "stn"}},
            simulation={"general": {"ET_sim_source": ["SimGrid"]}, "ET": {"SimGrid_data_type": "grid"}},
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    assert bindings.has_grid_evaluation(["ET"]).has_grid is False


def test_has_grid_evaluation_without_simulation_source_is_false():
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=[],
        scores=[],
        comparisons=[],
        statistics=["Mean"],
        general={"basename": "case", "basedir": "."},
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"ET_ref_source": "RefGrid"}, "ET": {"RefGrid_data_type": "grid"}},
            simulation={"general": {}, "ET": {}},
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    assert bindings.has_grid_evaluation(["ET"]).has_grid is False


def test_statistics_context_skips_station_pairs_within_mixed_sources(tmp_path):
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Mean"],
        general={
            "basename": "case",
            "basedir": str(tmp_path),
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={
                "general": {"ET_ref_source": "RefGrid"},
                "ET": {"RefGrid_data_type": "grid", "RefGrid_varname": "E"},
            },
            simulation={
                "general": {"ET_sim_source": ["SimStn", "SimGrid"]},
                "ET": {
                    "SimStn_data_type": "stn",
                    "SimStn_varname": "station_E",
                    "SimGrid_data_type": "grid",
                    "SimGrid_varname": "grid_E",
                },
            },
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    context = bindings.build_statistics_context(["Mean"])

    assert context.stats_nml["general"]["Mean_data_source"] == "ET_SimGrid"
    assert "ET_SimStn_dir" not in context.stats_nml["Mean"]


def test_has_grid_evaluation_pure_stn_x_stn_returns_false():
    """Pure stn × stn (ref=stn, all sims=stn) should still return has_grid=False."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={"general": {"ET_ref_source": "RefStn"}, "ET": {"RefStn_data_type": "stn"}},
            simulation={
                "general": {"ET_sim_source": ["SimStn1", "SimStn2"]},
                "ET": {"SimStn1_data_type": "stn", "SimStn2_data_type": "stn"},
            },
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    evidence = bindings.has_grid_evaluation(["ET"])
    assert evidence.has_grid is False


def test_statistics_context_per_pair_uses_pair_specific_ref_prefix():
    """per_pair statistics must read the same masked ref copies as evaluation."""
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"ET": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=["Hellinger_Distance"],
        general={
            "basename": "case",
            "basedir": ".",
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "num_cores": 1,
            "syear": 2000,
            "eyear": 2001,
            "time_alignment": "per_pair",
        },
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": runner_cfg.general},
            reference={
                "general": {"ET_ref_source": "RefA"},
                "ET": {
                    "RefA_data_type": "grid",
                    "RefA_varname": "et_ref",
                    "RefA_varunit": "mm day-1",
                },
            },
            simulation={
                "general": {"ET_sim_source": ["SimA", "SimB"]},
                "ET": {
                    "SimA_data_type": "grid",
                    "SimA_varname": "et_sim",
                    "SimA_varunit": "mm day-1",
                    "SimB_data_type": "grid",
                    "SimB_varname": "et_sim",
                    "SimB_varunit": "mm day-1",
                },
            },
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    context = bindings.build_statistics_context(["Hellinger_Distance"])

    assert context.stats_nml["Hellinger_Distance"]["ET_SimA2_prefix"] == "ET_ref_RefA_SimA_et_ref"
    assert context.stats_nml["Hellinger_Distance"]["ET_SimB2_prefix"] == "ET_ref_RefA_SimB_et_ref"


def test_adapter_uses_reference_override_and_fallbacks(monkeypatch, tmp_path):
    from openbench.data.registry.schema import FallbackVar, ReferenceDataset, VariableMapping

    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "alt.nc4").write_bytes(b"x")
    ref = ReferenceDataset(
        name="RefA",
        description="",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Single",
        timezone=0,
        years=[2000, 2001],
        root_dir="/wrong",
        variables={
            "Runoff": VariableMapping(
                varname="runoff",
                varunit="mm day-1",
                prefix="",
                suffix="",
                fallbacks=[FallbackVar(varname="q", varunit="kg m-2 s-1", convert="value * 86400")],
                prefix_fallback=["alt_"],
            )
        },
        grid_res=0.5,
    )
    registry = SimpleNamespace(
        get_model=lambda name: SimpleNamespace(tim_res="Month", grid_res=0.5, data_type="grid", variables={}),
        get_resolution_variants=lambda name: {},
        get_reference=lambda name, **kwargs: ref,
    )
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    monkeypatch.setenv("OPENBENCH_TEST_REF_ROOT", str(ref_root))
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="override", output_dir="./out", years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(
            data_root="/common",
            sources={"Runoff": "RefA"},
            overrides={"RefA": {"root_dir": "$OPENBENCH_TEST_REF_ROOT", "variables": {"Runoff": {"varname": "q2"}}}},
        ),
        simulation={"SimA": SimulationEntry(model="M", root_dir="/sim", tim_res="Month", grid_res=0.5)},
    )

    _main, ref_nml, _sim = adapter_module.build_legacy_namelists(cfg)

    section = ref_nml["Runoff"]
    assert section["RefA_dir"] == str(ref_root)
    assert section["RefA_varname"] == "q2"
    assert section["RefA_prefix_fallback"] == ["alt_"]
    assert section["RefA_fallbacks"][0]["varname"] == "q"
    assert section["RefA_fallbacks"][0]["convert"] == "value * 86400"


def test_find_nc_dir_does_not_substitute_lowres(tmp_path):
    mid = tmp_path / "Grid" / "MidRes" / "Water" / "Runoff" / "Demo"
    low = tmp_path / "Grid" / "LowRes" / "Water" / "Runoff" / "Demo"
    mid.mkdir(parents=True)
    low.mkdir(parents=True)
    (low / "demo.nc").write_bytes(b"x")

    assert adapter_module._find_nc_dir(str(mid), str(tmp_path / "Grid" / "MidRes"), "Water/Runoff/Demo") == str(mid)


def test_adapter_reference_data_root_still_overrides_catalog_root(monkeypatch):
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    ref = ReferenceDataset(
        name="RefA",
        description="",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Single",
        timezone=0,
        years=[2000, 2001],
        root_dir="/catalog/root",
        variables={"Runoff": VariableMapping(varname="runoff", varunit="mm day-1")},
        grid_res=0.5,
    )
    registry = SimpleNamespace(
        get_model=lambda name: SimpleNamespace(tim_res="Month", grid_res=0.5, data_type="grid", variables={}),
        get_resolution_variants=lambda name: {},
        get_reference=lambda name, **kwargs: ref,
    )
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="root", output_dir="./out", years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(data_root="/config/root", sources={"Runoff": "RefA"}),
        simulation={"SimA": SimulationEntry(model="M", root_dir="/sim", tim_res="Month", grid_res=0.5)},
    )

    _main, ref_nml, _sim = adapter_module.build_legacy_namelists(cfg)

    assert ref_nml["Runoff"]["RefA_dir"] == "/config/root"


def test_adapter_rejects_programmatic_unsafe_labels():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="unsafe", output_dir="./out", years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "CON"}),
        simulation={"SimA ": SimulationEntry(model="M", root_dir="/sim", tim_res="Month", grid_res=0.5)},
    )

    with pytest.raises(ConfigError, match="path-safe"):
        adapter_module.build_runner_config(cfg)


def test_build_namelists_preserves_inline_sim_variable_runtime_fields():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="case", output_dir="./out", years=[2000, 2000]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "GRDC"}),
        simulation={
            "CaseA": SimulationEntry(
                model="UnknownModel",
                root_dir="/sim",
                variables={
                    "Runoff": {
                        "varname": "q",
                        "sub_dir": "hydro",
                        "compute": "ds['rain'] + ds['snow']",
                        "fallbacks": [{"varname": "q_alt", "varunit": "mm"}],
                        "prefix_fallback": ["_hydro_"],
                    }
                },
            )
        },
    )

    _, _, sim_nml = adapter_module.build_legacy_namelists(cfg)
    section = sim_nml["Runoff"]

    assert section["CaseA_dir"].replace("\\", "/") == "/sim/hydro"
    assert section["CaseA_compute"] == "ds['rain'] + ds['snow']"
    assert section["CaseA_fallbacks"] == [{"varname": "q_alt", "varunit": "mm"}]
    assert section["CaseA_prefix_fallback"] == ["_hydro_"]
