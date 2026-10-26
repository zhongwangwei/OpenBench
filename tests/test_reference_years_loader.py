"""Reference coverage overrides must survive GUI export and reject malformed windows."""

from pathlib import Path

import pytest
import yaml

from openbench.config.loader import ConfigError, load_config


@pytest.fixture
def raw_config():
    fixture = Path(__file__).parent / "test_config" / "fixtures" / "minimal.yaml"
    return yaml.safe_load(fixture.read_text())


def _load(tmp_path, raw):
    path = tmp_path / "openbench.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.mark.parametrize("scope", ["dataset", "variable"])
@pytest.mark.parametrize(
    "years",
    ["2001", 2001, [], [2001], [2001, 2002, 2003], [True, 2002], [2001.0, 2002], ["2001", "2002"], [2002, 2001]],
)
def test_reject_invalid_reference_years(tmp_path, raw_config, scope, years):
    override = {"years": years}
    if scope == "variable":
        override = {"variables": {"Evapotranspiration": override}}
    raw_config["reference"]["overrides"] = {"GLEAM_v4.3a_LowRes": override}
    with pytest.raises(ConfigError, match=r"reference\.overrides\..*\.years"):
        _load(tmp_path, raw_config)


@pytest.mark.parametrize("years", [[2001, 2001], [2001, 2005], None])
def test_variable_years_roundtrip_and_clear(tmp_path, raw_config, years):
    raw_config["reference"]["overrides"] = {
        "GLEAM_v4.3a_LowRes": {"years": [2000, 2010], "variables": {"Evapotranspiration": {"years": years}}}
    }
    cfg = _load(tmp_path, raw_config)
    override = cfg.reference.overrides["GLEAM_v4.3a_LowRes"]
    assert override["years"] == [2000, 2010]
    assert override["variables"]["Evapotranspiration"]["years"] == years


def test_simulation_years_remain_unsupported(tmp_path, raw_config):
    raw_config["simulation"]["CoLM2024"]["variables"] = {"Evapotranspiration": {"years": [2000, 2010]}}
    with pytest.raises(ConfigError, match="years"):
        _load(tmp_path, raw_config)


def test_gui_export_loads_variable_years(tmp_path):
    pytest.importorskip("PySide6")
    from openbench.gui.config_manager import ConfigManager

    raw = {
        "general": {"basename": "coverage", "basedir": "/out", "syear": 2000, "eyear": 2010},
        "evaluation_items": {"Evapotranspiration": True},
        "metrics": {"RMSE": True},
        "ref_data": {
            "general": {"Evapotranspiration_ref_source": "GLEAM_v4.3a_LowRes"},
            "source_configs": {
                "Evapotranspiration::GLEAM_v4.3a_LowRes": {
                    "_explicit_override": True,
                    "_years_scope": "variable",
                    "years": [2001, 2005],
                    "general": {"syear": 2001, "eyear": 2005},
                    "varname": "E",
                    "varunit": "mm day-1",
                }
            },
        },
        "sim_data": {
            "general": {"Evapotranspiration_sim_source": "CoLM2024"},
            "source_configs": {"CoLM2024": {"general": {"model": "CoLM2024", "root_dir": "/sim"}}},
        },
    }
    path = tmp_path / "export.yaml"
    path.write_text(ConfigManager().generate_config_yaml(raw))
    cfg = load_config(path)
    assert cfg.reference.overrides["GLEAM_v4.3a_LowRes"]["variables"]["Evapotranspiration"]["years"] == [2001, 2005]


@pytest.mark.parametrize("variable_years", ["absent", None, [2002, 2004]])
def test_dataset_null_years_is_noop_without_clearing_binding(tmp_path, raw_config, variable_years):
    from openbench.config.adapter import _apply_reference_override
    from openbench.data.registry.manager import _build_reference
    from openbench.data.registry.schema import reference_years

    override = {"years": None}
    if variable_years != "absent":
        override["variables"] = {"Evapotranspiration": {"years": variable_years}}
    raw_config["reference"]["overrides"] = {"GLEAM_v4.3a_LowRes": override}
    cfg = _load(tmp_path, raw_config)
    normalized = cfg.reference.overrides["GLEAM_v4.3a_LowRes"]
    assert "years" not in normalized
    assert override["years"] is None
    ref = _build_reference(
        {
            "name": "GLEAM_v4.3a_LowRes",
            "data_type": "grid",
            "tim_res": "Month",
            "years": [1980, 2020],
            "variables": {"Evapotranspiration": {"varname": "E", "varunit": "mm day-1", "years": [2001, 2010]}},
        }
    )
    effective, mapping = _apply_reference_override(
        ref, ref.variables["Evapotranspiration"], "Evapotranspiration", normalized
    )
    expected = [2001, 2010] if variable_years == "absent" else variable_years or [1980, 2020]
    assert reference_years(effective, mapping) == expected
    assert ref.variables["Evapotranspiration"].years == [2001, 2010]
