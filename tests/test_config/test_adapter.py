"""Tests for config adapter — new format to legacy format bridge."""

from openbench.config.adapter import to_legacy_config
from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
    SimulationEntry,
    StatisticsConfig,
)


def test_minimal_config_adapter():
    """Convert minimal new config to legacy format."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="test", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference={"Evapotranspiration": "GLEAM_v4.2a"},
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data")},
    )
    legacy = to_legacy_config(cfg)

    assert legacy["general"]["basename"] == "test"
    assert legacy["general"]["basedir"] == "./output"
    assert legacy["general"]["syear"] == 2004
    assert legacy["general"]["eyear"] == 2010
    assert "Evapotranspiration" in legacy["evaluation_items"]
    assert legacy["evaluation_items"]["Evapotranspiration"] is True


def test_full_config_adapter():
    """Convert full config with all options."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="full",
            output_dir="/data/out",
            years=[2000, 2020],
            min_year_threshold=5,
            lat_range=[-60, 90],
            lon_range=[-180, 180],
        ),
        evaluation=EvaluationConfig(variables=["GPP", "Latent_Heat"]),
        reference={"GPP": "FLUXCOM", "Latent_Heat": "FLUXCOM"},
        simulation={
            "CoLM": SimulationEntry(model="CoLM2024", root_dir="/d1"),
            "CLM": SimulationEntry(model="CLM5", root_dir="/d2"),
        },
        metrics=["bias", "RMSE"],
        scores=["nBiasScore"],
        comparison=ComparisonConfig(enabled=True, items=["Taylor_Diagram"]),
        statistics=StatisticsConfig(enabled=True, items=["ANOVA"]),
        options=OptionsConfig(num_cores=8, time_alignment="per_pair"),
    )
    legacy = to_legacy_config(cfg)

    assert legacy["general"]["num_cores"] == 8
    assert legacy["general"]["comparison"] is True
    assert legacy["general"]["statistics"] is True
    assert legacy["general"]["min_year"] == 5
    assert legacy["metrics"] == {"bias": True, "RMSE": True}
    assert legacy["scores"] == {"nBiasScore": True}
    assert legacy["comparisons"] == {"Taylor_Diagram": True}
    assert legacy["statistics"] == {"ANOVA": True}


def test_defaults_applied():
    """Verify sensible defaults when options not specified."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="t", output_dir=".", years=[2000, 2010]),
        evaluation=EvaluationConfig(variables=["GPP"]),
        reference={"GPP": "FLUXCOM"},
        simulation={"M": SimulationEntry(model="M", root_dir="/d")},
    )
    legacy = to_legacy_config(cfg)

    assert legacy["general"]["num_cores"] >= 1
    assert legacy["general"]["unified_mask"] is True
    assert legacy["general"]["generate_report"] is True
    assert legacy["general"]["comparison"] is False
    assert "bias" in legacy["metrics"]  # Default metrics
