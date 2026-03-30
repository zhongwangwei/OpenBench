"""Tests for config schema dataclasses."""

from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
    SimulationEntry,
    StatisticsConfig,
)


def test_project_config_defaults():
    p = ProjectConfig(name="test", output_dir="./output", years=[2004, 2010])
    assert p.name == "test"
    assert p.output_dir == "./output"
    assert p.years == [2004, 2010]
    assert p.min_year_threshold == 3
    assert p.lat_range == [-90.0, 90.0]
    assert p.lon_range == [-180.0, 180.0]


def test_project_config_custom():
    p = ProjectConfig(
        name="custom",
        output_dir="/data/out",
        years=[2000, 2020],
        min_year_threshold=5,
        lat_range=[-60, 90],
        lon_range=[-180, 180],
    )
    assert p.min_year_threshold == 5
    assert p.lat_range == [-60, 90]


def test_evaluation_config():
    e = EvaluationConfig(variables=["Evapotranspiration", "GPP"])
    assert len(e.variables) == 2


def test_simulation_entry_minimal():
    s = SimulationEntry(model="CoLM2024", root_dir="/data/CoLM2024")
    assert s.model == "CoLM2024"
    assert s.root_dir == "/data/CoLM2024"
    assert s.data_type is None
    assert s.variables is None


def test_simulation_entry_with_overrides():
    s = SimulationEntry(
        model="CLM5",
        root_dir="/data/CLM5",
        data_type="grid",
        grid_res=1.0,
        tim_res="Month",
        variables={"GPP": {"varname": "FPSN"}},
    )
    assert s.data_type == "grid"
    assert s.grid_res == 1.0
    assert s.variables["GPP"]["varname"] == "FPSN"


def test_options_config_defaults():
    o = OptionsConfig()
    assert o.num_cores is None
    assert o.time_alignment == "intersection"
    assert o.unified_mask is True
    assert o.generate_report is True
    assert o.IGBP_groupby is False
    assert o.PFT_groupby is False
    assert o.climate_zone_groupby is False
    assert o.debug_mode is False
    assert o.only_drawing is False


def test_comparison_config_defaults():
    c = ComparisonConfig()
    assert c.enabled is False


def test_statistics_config_defaults():
    s = StatisticsConfig()
    assert s.enabled is False


def test_openbench_config_minimal():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="test", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference={"Evapotranspiration": "GLEAM_v4.2a"},
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data")},
    )
    assert cfg.project.name == "test"
    assert cfg.metrics is None
    assert cfg.scores is None
    assert cfg.comparison.enabled is False
    assert cfg.statistics.enabled is False
    assert cfg.options.time_alignment == "intersection"


def test_openbench_config_full():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="full", output_dir="./out", years=[2000, 2020]),
        evaluation=EvaluationConfig(variables=["GPP", "Latent_Heat"]),
        reference={"GPP": "FLUXCOM", "Latent_Heat": "FLUXCOM"},
        simulation={
            "CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data/colm"),
            "CLM5": SimulationEntry(model="CLM5", root_dir="/data/clm5"),
        },
        metrics=["bias", "RMSE", "correlation"],
        scores=["nBiasScore", "nRMSEScore"],
        comparison=ComparisonConfig(enabled=True),
        statistics=StatisticsConfig(enabled=True, items=["Z_Score", "ANOVA"]),
        options=OptionsConfig(num_cores=16, time_alignment="per_pair"),
    )
    assert len(cfg.simulation) == 2
    assert cfg.metrics == ["bias", "RMSE", "correlation"]
    assert cfg.comparison.enabled is True
    assert cfg.options.num_cores == 16
    assert cfg.options.time_alignment == "per_pair"
