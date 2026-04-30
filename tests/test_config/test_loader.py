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
    assert cfg.reference.sources["Evapotranspiration"] == "GLEAM_v4.2a_LowRes"
    assert "CoLM2024" in cfg.simulation
    assert cfg.simulation["CoLM2024"].model == "CoLM2024"
    # Defaults applied
    assert cfg.project.time_alignment == "intersection"
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
    assert cfg.project.num_cores == 16
    assert cfg.project.time_alignment == "per_pair"


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


def test_load_with_defaults_deep_merges_variable_overrides():
    """Variable overrides should preserve unspecified fields from _defaults."""
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["Evapotranspiration", "Runoff"]},
        "reference": {
            "Evapotranspiration": "GLEAM_v4.2a_LowRes",
            "Runoff": "GRDC_Monthly",
        },
        "simulation": {
            "_defaults": {
                "model": "BaseModel",
                "root_dir": "/defaults",
                "variables": {
                    "Evapotranspiration": {
                        "varname": "QVEGE",
                        "varunit": "mm day-1",
                        "prefix": "hist_",
                        "fallbacks": [
                            {
                                "varname": "QSOIL",
                                "varunit": "kg m-2 s-1",
                                "convert": "value * 86400",
                            }
                        ],
                    },
                    "Runoff": {
                        "varname": "QRUNOFF",
                        "varunit": "mm day-1",
                    },
                },
            },
            "CaseA": {
                "root_dir": "/case-a",
                "variables": {
                    "Evapotranspiration": {
                        "varname": "QFLX_EVAP_TOT",
                    }
                },
            },
        },
    }

    cfg = _build_config(raw)

    case_vars = cfg.simulation["CaseA"].variables
    assert case_vars is not None
    assert case_vars["Evapotranspiration"]["varname"] == "QFLX_EVAP_TOT"
    assert case_vars["Evapotranspiration"]["varunit"] == "mm day-1"
    assert case_vars["Evapotranspiration"]["prefix"] == "hist_"
    assert case_vars["Evapotranspiration"]["fallbacks"] == [
        {
            "varname": "QSOIL",
            "varunit": "kg m-2 s-1",
            "convert": "value * 86400",
        }
    ]
    assert case_vars["Runoff"] == {
        "varname": "QRUNOFF",
        "varunit": "mm day-1",
    }


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
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010], "time_alignment": "invalid_value"},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
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


# --- Multi-reference per variable (v2.x compatibility) ---

def _minimal_with_reference(ref_block):
    return {
        "project": {"name": "test", "output_dir": "/tmp/x", "years": [2010, 2014]},
        "evaluation": {"variables": ["Evapotranspiration"]},
        "reference": ref_block,
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }


def test_reference_accepts_list_value():
    """v3 schema must accept list[str] form for ref_source (v2.x compat)."""
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({
        "Evapotranspiration": ["GLEAM_v4.2a", "FLUXCOM-X-BASE"],
    })
    cfg = _build_config(raw)
    assert cfg.reference.sources["Evapotranspiration"] == [
        "GLEAM_v4.2a",
        "FLUXCOM-X-BASE",
    ]


def test_reference_single_item_list_collapses_to_string():
    """A list with one element is stored as plain string for downstream simplicity."""
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({
        "Evapotranspiration": ["GLEAM_v4.2a"],
    })
    cfg = _build_config(raw)
    assert cfg.reference.sources["Evapotranspiration"] == "GLEAM_v4.2a"


def test_reference_comma_separated_string_splits():
    """Comma-separated source names (legacy NML form) auto-split into list."""
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({
        "Evapotranspiration": "GLEAM_v4.2a,FLUXCOM",
    })
    cfg = _build_config(raw)
    assert cfg.reference.sources["Evapotranspiration"] == ["GLEAM_v4.2a", "FLUXCOM"]


def test_reference_rejects_empty_list():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({"Evapotranspiration": []})
    with pytest.raises(ConfigError, match="empty list"):
        _build_config(raw)


def test_reference_rejects_non_string_in_list():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({"Evapotranspiration": ["GLEAM", 42]})
    with pytest.raises(ConfigError, match="must be a string"):
        _build_config(raw)
