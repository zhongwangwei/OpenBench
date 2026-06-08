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


def test_null_reference_section_raises_config_error():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": None,
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }

    with pytest.raises(ConfigError, match="'reference' must be a mapping"):
        _build_config(raw)


@pytest.mark.parametrize("section", ["project", "evaluation", "simulation"])
def test_non_mapping_required_section_raises_config_error(section):
    """Malformed required sections must raise ConfigError, not a bare traceback (M3)."""
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "RefA"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }
    raw[section] = "not-a-mapping"

    with pytest.raises(ConfigError, match=f"'{section}' must be a mapping"):
        _build_config(raw)


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


def test_project_tim_res_is_validated_and_normalized():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010], "tim_res": "month"},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }

    cfg = _build_config(raw)

    assert cfg.project.tim_res == "Month"

    raw["project"]["tim_res"] = "Weekly"
    with pytest.raises(ConfigError, match="project.tim_res"):
        _build_config(raw)


def test_comparison_section_must_be_mapping():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
        "comparison": "enabled",
    }

    with pytest.raises(ConfigError, match="comparison"):
        _build_config(raw)


def test_project_weight_is_normalized_for_known_values_and_rejects_invalid_values():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010], "weight": "None"},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }

    cfg = _build_config(raw)

    assert cfg.project.weight == "none"

    raw["project"]["weight"] = "inverse-distance"
    with pytest.raises(ConfigError, match="project.weight must be one of"):
        _build_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lat_range", [10], "project.lat_range"),
        ("lon_range", ["west", 180], "project.lon_range"),
        ("lat_range", [-100, 20], "project.lat_range"),
        ("lon_range", [10, -10], "project.lon_range"),
    ],
)
def test_invalid_project_bounds_raise_config_error(field, value, message):
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010], field: value},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }

    with pytest.raises(ConfigError, match=message):
        _build_config(raw)


def test_simulation_variables_must_be_mapping():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {
            "M": {
                "model": "M",
                "root_dir": "/d",
                "variables": "not-a-mapping",
            }
        },
    }

    with pytest.raises(ConfigError, match="simulation.M.variables"):
        _build_config(raw)


def test_simulation_variable_overrides_must_be_mapping():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {
            "M": {
                "model": "M",
                "root_dir": "/d",
                "variables": {"GPP": "not-a-mapping"},
            }
        },
    }

    with pytest.raises(ConfigError, match="simulation.M.variables.GPP"):
        _build_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", 123, "simulation.M.model"),
        ("root_dir", 123, "simulation.M.root_dir"),
        ("data_type", "table", "simulation.M.data_type"),
        ("tim_res", "Weekly", "simulation.M.tim_res"),
        ("grid_res", 0, "simulation.M.grid_res"),
        ("prefix", 123, "simulation.M.prefix"),
        ("suffix", 123, "simulation.M.suffix"),
        ("fulllist", 123, "simulation.M.fulllist"),
    ],
)
def test_simulation_scalar_fields_are_validated(field, value, message):
    from openbench.config.loader import _build_config

    entry = {"model": "M", "root_dir": "/d", field: value}
    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": entry},
    }

    with pytest.raises(ConfigError, match=message):
        _build_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("grid_res", 0, "project.grid_res"),
        ("grid_res", "0.5", "project.grid_res"),
        ("timezone", "UTC", "project.timezone"),
    ],
)
def test_project_numeric_fields_are_validated(field, value, message):
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010], field: value},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }

    with pytest.raises(ConfigError, match=message):
        _build_config(raw)


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        ("comparison", {"enabled": "yes"}, "comparison.enabled"),
        ("comparison", {"items": "HeatMap"}, "comparison.items"),
        ("comparison", {"items": ["HeatMap", 7]}, r"comparison.items\[1\]"),
        ("statistics", {"enabled": "yes"}, "statistics.enabled"),
        ("statistics", {"items": "Mean"}, "statistics.items"),
        ("statistics", {"items": ["Mean", 7]}, r"statistics.items\[1\]"),
    ],
)
def test_optional_workflow_sections_are_validated(section, payload, message):
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
        section: payload,
    }

    with pytest.raises(ConfigError, match=message):
        _build_config(raw)


def test_simulation_tim_res_accepts_extended_values():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d", "tim_res": "3month"}},
    }

    cfg = _build_config(raw)

    assert cfg.simulation["M"].tim_res == "3month"


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


def test_include_allows_parent_reference_inside_project_root(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    shared_dir = project / "shared"
    config_dir.mkdir(parents=True)
    shared_dir.mkdir()
    (shared_dir / "reference.yaml").write_text("GPP: FLUXCOM\n")
    config = config_dir / "openbench.yaml"
    config.write_text(
        """
project:
  name: include-parent
  output_dir: ./out
  years: [2000, 2001]
evaluation:
  variables: [GPP]
reference: !include ../shared/reference.yaml
simulation:
  M:
    model: M
    root_dir: /data
""".lstrip()
    )

    monkeypatch.chdir(project)

    cfg = load_config(config)

    assert cfg.reference.sources["GPP"] == "FLUXCOM"


def test_include_rejects_parent_escape_outside_allowed_project_root(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    (tmp_path / "outside.yaml").write_text("GPP: SECRET\n")
    config = config_dir / "openbench.yaml"
    config.write_text(
        """
project:
  name: include-escape
  output_dir: ./out
  years: [2000, 2001]
evaluation:
  variables: [GPP]
reference: !include ../../outside.yaml
simulation:
  M:
    model: M
    root_dir: /data
""".lstrip()
    )

    monkeypatch.chdir(project)

    with pytest.raises(ConfigError, match="outside allowed roots"):
        load_config(config)


def test_include_can_read_external_file_when_root_is_explicitly_allowed(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    external = tmp_path / "shared-outside"
    config_dir.mkdir(parents=True)
    external.mkdir()
    (external / "reference.yaml").write_text("GPP: FLUXCOM\n")
    config = config_dir / "openbench.yaml"
    config.write_text(
        f"""
project:
  name: include-explicit-root
  output_dir: ./out
  years: [2000, 2001]
evaluation:
  variables: [GPP]
reference: !include {external / "reference.yaml"}
simulation:
  M:
    model: M
    root_dir: /data
""".lstrip()
    )

    monkeypatch.chdir(project)
    monkeypatch.setenv("OPENBENCH_INCLUDE_ROOTS", str(external))

    cfg = load_config(config)

    assert cfg.reference.sources["GPP"] == "FLUXCOM"


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

    raw = _minimal_with_reference(
        {
            "Evapotranspiration": ["GLEAM_v4.2a", "FLUXCOM-X-BASE"],
        }
    )
    cfg = _build_config(raw)
    assert cfg.reference.sources["Evapotranspiration"] == [
        "GLEAM_v4.2a",
        "FLUXCOM-X-BASE",
    ]


def test_reference_single_item_list_collapses_to_string():
    """A list with one element is stored as plain string for downstream simplicity."""
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference(
        {
            "Evapotranspiration": ["GLEAM_v4.2a"],
        }
    )
    cfg = _build_config(raw)
    assert cfg.reference.sources["Evapotranspiration"] == "GLEAM_v4.2a"


def test_reference_comma_separated_string_splits():
    """Comma-separated source names (legacy NML form) auto-split into list."""
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference(
        {
            "Evapotranspiration": "GLEAM_v4.2a,FLUXCOM",
        }
    )
    cfg = _build_config(raw)
    assert cfg.reference.sources["Evapotranspiration"] == ["GLEAM_v4.2a", "FLUXCOM"]


def test_reference_data_root_must_be_string():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference(
        {
            "data_root": 42,
            "Evapotranspiration": "GLEAM_v4.2a",
        }
    )
    with pytest.raises(ConfigError, match="reference.data_root must be a string"):
        _build_config(raw)


def test_reference_data_root_must_not_be_empty():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference(
        {
            "data_root": "  ",
            "Evapotranspiration": "GLEAM_v4.2a",
        }
    )
    with pytest.raises(ConfigError, match="reference.data_root must not be empty"):
        _build_config(raw)


def test_reference_rejects_empty_comma_separated_source():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({"Evapotranspiration": " , , "})
    with pytest.raises(
        ConfigError,
        match="reference.Evapotranspiration must include at least one source name",
    ):
        _build_config(raw)


def test_reference_rejects_empty_source_string():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({"Evapotranspiration": "  "})
    with pytest.raises(ConfigError, match="reference.Evapotranspiration must not be empty"):
        _build_config(raw)


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


def test_reference_rejects_empty_source_in_list():
    from openbench.config.loader import _build_config

    raw = _minimal_with_reference({"Evapotranspiration": ["GLEAM", " "]})
    with pytest.raises(ConfigError, match=r"reference\.Evapotranspiration\[1\] must not be empty"):
        _build_config(raw)


def test_loader_normalizes_uppercase_data_type_and_known_weight_values():
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2001], "weight": "Area"},
        "evaluation": {"variables": ["Runoff"]},
        "reference": {"Runoff": "GRDC_Monthly"},
        "simulation": {"CaseA": {"model": "M", "root_dir": "/tmp", "data_type": "GRID"}},
    }

    cfg = _build_config(raw)
    assert cfg.project.weight == "area"
    assert cfg.simulation["CaseA"].data_type == "grid"


def _minimal_raw_config(**project_overrides):
    project = {"name": "tier-b2", "output_dir": ".", "years": [2000, 2001]}
    project.update(project_overrides)
    return {
        "project": project,
        "evaluation": {"variables": ["Runoff"]},
        "reference": {"Runoff": "GRDC_Monthly"},
        "simulation": {"CaseA": {"model": "CoLM2024", "root_dir": "/tmp"}},
    }


def test_dask_threads_per_worker_zero_is_preserved():
    from openbench.config.loader import _build_config

    cfg = _build_config(_minimal_raw_config(dask={"enabled": True, "threads_per_worker": 0}))

    assert cfg.project.dask.threads_per_worker == 0


def test_invalid_project_weight_raises_config_error():
    from openbench.config.loader import _build_config

    with pytest.raises(ConfigError, match="project.weight must be one of"):
        _build_config(_minimal_raw_config(weight="bogus"))


def test_simulation_defaults_must_be_mapping():
    from openbench.config.loader import _build_config

    raw = _minimal_raw_config()
    raw["simulation"] = {"_defaults": "not-a-mapping", "CaseA": {"model": "CoLM2024", "root_dir": "/tmp"}}

    with pytest.raises(ConfigError, match="simulation._defaults must be a mapping"):
        _build_config(raw)
