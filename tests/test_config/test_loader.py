"""Tests for config loader."""

from pathlib import Path

import pytest

from openbench.config.loader import ConfigError, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_minimal():
    cfg = load_config(FIXTURES / "minimal.yaml")
    assert cfg.project.name == "test-minimal"
    assert cfg.project.years == [2004, 2010]
    assert cfg.evaluation.variables == ["Evapotranspiration"]
    assert cfg.reference["Evapotranspiration"] == "GLEAM_v4.2a"
    assert "CoLM2024" in cfg.simulation
    assert cfg.simulation["CoLM2024"].model == "CoLM2024"
    # Defaults applied
    assert cfg.options.time_alignment == "intersection"
    assert cfg.comparison.enabled is False


def test_load_full():
    cfg = load_config(FIXTURES / "full.yaml")
    assert cfg.project.name == "test-full"
    assert cfg.project.min_year_threshold == 5
    assert cfg.project.lat_range == [-60, 90]
    assert len(cfg.evaluation.variables) == 3
    assert len(cfg.simulation) == 2
    assert cfg.metrics == ["bias", "RMSE", "correlation"]
    assert cfg.scores == ["nBiasScore", "nRMSEScore"]
    assert cfg.comparison.enabled is True
    assert cfg.statistics.enabled is True
    assert cfg.statistics.items == ["Z_Score", "ANOVA"]
    assert cfg.options.num_cores == 16
    assert cfg.options.time_alignment == "per_pair"


def test_load_with_defaults_merge():
    cfg = load_config(FIXTURES / "with_defaults.yaml")
    # _defaults should be merged into each entry
    assert "_defaults" not in cfg.simulation
    assert cfg.simulation["CoLM2014"].data_type == "grid"
    assert cfg.simulation["CoLM2014"].grid_res == 0.5
    assert cfg.simulation["CoLM2014"].tim_res == "Month"
    assert cfg.simulation["CoLM2024"].data_type == "grid"
    # CLM5 has variable override but inherits data_type from _defaults
    assert cfg.simulation["CLM5"].data_type == "grid"
    assert cfg.simulation["CLM5"].variables is not None
    assert cfg.simulation["CLM5"].variables["Evapotranspiration"]["varname"] == "QFLX_EVAP_TOT"


def test_invalid_years_raises():
    with pytest.raises(ConfigError, match="start year.*must be.*end year"):
        load_config(FIXTURES / "invalid_years.yaml")


def test_missing_project_raises():
    with pytest.raises(ConfigError, match="project"):
        load_config(FIXTURES / "missing_project.yaml")


def test_invalid_time_alignment():
    """Construct a config dict with bad time_alignment."""
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
        "options": {"time_alignment": "invalid_value"},
    }
    with pytest.raises(ConfigError, match="time_alignment"):
        _build_config(raw)


def test_unreferenced_variable_warning():
    """Variable in evaluation but not in reference should error."""
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP", "Latent_Heat"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }
    with pytest.raises(ConfigError, match="Latent_Heat"):
        _build_config(raw)


def test_load_nonexistent_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("/nonexistent/file.yaml"))


def test_load_non_yaml():
    """Reject non-YAML files."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b'{"key": "value"}')
        f.flush()
        with pytest.raises(ConfigError, match="YAML"):
            load_config(Path(f.name))
