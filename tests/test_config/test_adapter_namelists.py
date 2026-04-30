"""Test legacy namelist building from new config."""

import openbench.config.adapter as adapter_module
from openbench.config.adapter import build_legacy_namelists
from openbench.config.schema import (
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
)


def test_adapter_declares_legacy_root_section_contract():
    """The adapter should publish the legacy top-level sections it still emits."""
    assert adapter_module.LEGACY_ROOT_SECTION_KEYS == {
        "general",
        "evaluation_items",
        "metrics",
        "scores",
        "comparisons",
        "statistics",
    }


def test_build_legacy_namelists():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="test", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data/CoLM2024")},
    )

    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    # Check main_nl
    assert main_nl["general"]["basename"] == "test"
    assert "Evapotranspiration" in main_nl["evaluation_items"]

    # Check ref_nml
    assert ref_nml["general"]["Evapotranspiration_ref_source"] == "GLEAM_v4.2a_LowRes"
    assert "Evapotranspiration" in ref_nml
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_varname"] == "E"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_data_type"] == "grid"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_varunit"] == "mm day-1"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_prefix"] == "E_"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_suffix"] == "_GLEAM_v4.2a"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_tim_res"] == "Month"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_grid_res"] in (0.25, 0.5)
    # years come from registry (may be auto-detected or config fallback)
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_syear"] >= 1980
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_eyear"] >= 2005
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_timezone"] == 0

    # Check sim_nml
    assert "Evapotranspiration_sim_source" in sim_nml["general"]
    assert "CoLM2024" in sim_nml["general"]["Evapotranspiration_sim_source"]
    assert "Evapotranspiration" in sim_nml
    assert sim_nml["Evapotranspiration"]["CoLM2024_varname"] == "f_fevpa"
    assert sim_nml["Evapotranspiration"]["CoLM2024_varunit"] in ("mm day-1", "mm s-1")
    assert sim_nml["Evapotranspiration"]["CoLM2024_data_type"] == "grid"
    assert sim_nml["Evapotranspiration"]["CoLM2024_grid_res"] == 0.5
    assert sim_nml["Evapotranspiration"]["CoLM2024_dir"] == "/data/CoLM2024"


def test_build_namelists_multiple_simulations():
    """Multiple sim entries produce a list of sim sources per variable."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="multi", output_dir="/out", years=[2000, 2020]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/d1"),
            "CLM5": SimulationEntry(model="CLM5", root_dir="/d2"),
        },
    )

    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    sources = sim_nml["general"]["Evapotranspiration_sim_source"]
    assert "CoLM2024" in sources
    assert "CLM5" in sources
    assert len(sources) == 2


def test_build_namelists_inline_override():
    """Inline variable overrides take precedence over model profile."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="override", output_dir="/out", years=[2005, 2015]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "MyModel": SimulationEntry(
                model="CoLM2024",
                root_dir="/custom",
                variables={
                    "Evapotranspiration": {
                        "varname": "custom_et",
                        "varunit": "kg m-2 s-1",
                    }
                },
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    assert sim_nml["Evapotranspiration"]["MyModel_varname"] == "custom_et"
    assert sim_nml["Evapotranspiration"]["MyModel_varunit"] == "kg m-2 s-1"
    # dir still comes from root_dir
    assert sim_nml["Evapotranspiration"]["MyModel_dir"] == "/custom"


def test_build_namelists_unknown_model_with_inline():
    """An unknown model name still works if inline variables are provided."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="unknown", output_dir="/out", years=[2000, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "Custom": SimulationEntry(
                model="UnknownModel",
                root_dir="/x",
                variables={
                    "Evapotranspiration": {"varname": "et", "varunit": "mm/d"},
                },
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    assert sim_nml["Evapotranspiration"]["Custom_varname"] == "et"
    assert sim_nml["Evapotranspiration"]["Custom_varunit"] == "mm/d"


def test_reference_fulllist_relative_to_root_dir_is_resolved(monkeypatch, tmp_path):
    """Registry fulllist values may be root_dir-relative; legacy processing needs an existing path."""
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping
    import openbench.data.registry.manager as manager_module

    ref = ReferenceDataset(
        name="DemoStn",
        description="Demo station",
        category="Water",
        data_type="stn",
        tim_res="Day",
        data_groupby="single",
        timezone=0,
        years=[2000, 2001],
        root_dir=str(tmp_path),
        variables={
            "Runoff": VariableMapping(varname="Q", varunit="m3 s-1"),
        },
        fulllist="list/subset.csv",
    )

    class FakeRegistry:
        last_resolve_reason = ""

        def get_reference(self, name, sim_tim_res=None, sim_grid_res=None):
            return ref if name == "DemoStn" else None

        def get_resolution_variants(self, name):
            return {}

        def get_model(self, name):
            return None

    monkeypatch.setattr(manager_module, "get_registry", lambda: FakeRegistry())

    cfg = OpenBenchConfig(
        project=ProjectConfig(name="fulllist", output_dir="/out", years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "DemoStn"}),
        simulation={
            "Custom": SimulationEntry(
                model="UnknownModel",
                root_dir="/sim",
                variables={
                    "Runoff": {"varname": "runoff", "varunit": "m3 s-1"},
                },
            ),
        },
    )

    _, ref_nml, _ = build_legacy_namelists(cfg)

    assert ref_nml["Runoff"]["DemoStn_fulllist"] == str(tmp_path / "list" / "subset.csv")
