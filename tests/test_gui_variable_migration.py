import copy

import pytest
import yaml

pytest.importorskip("PySide6")

from openbench.gui.config_manager import ConfigManager, migrate_gui_variable_config
from openbench.gui.controller import WizardController


def _legacy_config():
    return {
        "general": {"basename": "demo", "basedir": "/tmp/out", "comparison": False, "statistics": False},
        "evaluation_items": {
            "Water_Evaporation": True,
            "Open_Water_Evaporation": False,
            "Bare_Soil_Evaporation": True,
        },
        "ref_data": {
            "general": {
                "Water_Evaporation_ref_source": "GLEAM",
                "Open_Water_Evaporation_ref_source": ["ALT"],
                "Bare_Soil_Evaporation_ref_source": "GLEAM",
            },
            "source_configs": {
                "Water_Evaporation::GLEAM": {"general": {"root_dir": "/ref"}, "varname": "Ew"},
                "Bare_Soil_Evaporation::GLEAM": {"general": {"root_dir": "/ref"}, "varname": "Eb"},
                "LegacyRef": {"general": {"root_dir": "/legacy-ref"}, "Bare_Soil_Evaporation": {"varname": "Eb"}},
            },
            "Bare_Soil_Evaporation": {"legacy": "ref-section"},
        },
        "sim_data": {
            "general": {
                "Water_Evaporation_sim_source": ["CaseA"],
                "Bare_Soil_Evaporation_sim_source": ["CaseA"],
            },
            "source_configs": {
                "CaseA": {
                    "general": {"model": "GLDAS", "root_dir": "/sim"},
                    "variables": {
                        "Water_Evaporation": {"varname": "Ew"},
                        "Bare_Soil_Evaporation": {"varname": "Eb"},
                    },
                }
            },
            "_scanned_cases": [
                {
                    "variables": ["Water_Evaporation", "Bare_Soil_Evaporation", "Runoff"],
                    "variable_overrides": {"Bare_Soil_Evaporation": {"prefix": "b_"}},
                    "metadata": {
                        "variables": ["Bare_Soil_Evaporation"],
                        "variable_overrides": {"Water_Evaporation": {"prefix": "w_"}},
                    },
                }
            ],
            "Water_Evaporation": {"legacy": "sim-section"},
        },
        "metrics": {"bias": True},
        "scores": {},
        "comparisons": {},
        "statistics": {},
    }


def test_migrate_gui_variable_config_updates_all_logical_variable_slots_without_mutating_input():
    original = _legacy_config()
    before = copy.deepcopy(original)

    migrated = migrate_gui_variable_config(original)

    assert original == before
    assert migrated["evaluation_items"] == {"Open_Water_Evaporation": True, "Soil_Evaporation": True}
    assert migrated["ref_data"]["general"] == {
        "Open_Water_Evaporation_ref_source": ["GLEAM", "ALT"],
        "Soil_Evaporation_ref_source": "GLEAM",
    }
    assert set(migrated["ref_data"]["source_configs"]) == {
        "Open_Water_Evaporation::GLEAM",
        "Soil_Evaporation::GLEAM",
        "LegacyRef",
    }
    assert migrated["ref_data"]["source_configs"]["LegacyRef"]["Soil_Evaporation"] == {"varname": "Eb"}
    assert migrated["ref_data"]["Soil_Evaporation"] == {"legacy": "ref-section"}
    case_cfg = migrated["sim_data"]["source_configs"]["CaseA"]
    assert set(case_cfg["variables"]) == {"Open_Water_Evaporation", "Soil_Evaporation"}
    scanned = migrated["sim_data"]["_scanned_cases"][0]
    assert scanned["variables"] == ["Open_Water_Evaporation", "Soil_Evaporation", "Runoff"]
    assert set(scanned["variable_overrides"]) == {"Soil_Evaporation"}
    assert scanned["metadata"]["variables"] == ["Soil_Evaporation"]
    assert set(scanned["metadata"]["variable_overrides"]) == {"Open_Water_Evaporation"}
    assert migrate_gui_variable_config(migrated) == migrated


def test_migrate_gui_variable_config_merges_distinct_scalar_sources_but_keeps_equal_scalar_shape():
    config = {
        "evaluation_items": {"Water_Evaporation": True},
        "ref_data": {
            "general": {
                "Water_Evaporation_ref_source": "GLEAM",
                "Open_Water_Evaporation_ref_source": "ALT",
                "Bare_Soil_Evaporation_ref_source": "Same",
                "Soil_Evaporation_ref_source": "Same",
            }
        },
    }

    migrated = migrate_gui_variable_config(config)

    assert migrated["ref_data"]["general"]["Open_Water_Evaporation_ref_source"] == ["GLEAM", "ALT"]
    assert migrated["ref_data"]["general"]["Soil_Evaporation_ref_source"] == "Same"


def test_migrate_gui_variable_config_normalizes_var_name_when_alias_and_canonical_collide():
    config = {
        "evaluation_items": {"Bare_Soil_Evaporation": True, "Soil_Evaporation": False},
        "ref_data": {
            "general": {},
            "source_configs": {
                "Bare_Soil_Evaporation::GLEAM": {"_var_name": "Bare_Soil_Evaporation", "varname": "Eb"},
                "Soil_Evaporation::GLEAM": {"_var_name": "Bare_Soil_Evaporation", "varname": "Eb"},
            },
        },
    }

    migrated = migrate_gui_variable_config(config)

    assert migrated["evaluation_items"] == {"Soil_Evaporation": True}
    assert migrated["ref_data"]["source_configs"] == {
        "Soil_Evaporation::GLEAM": {"_var_name": "Soil_Evaporation", "varname": "Eb"}
    }


def test_controller_setter_migrates_loaded_legacy_config_at_boundary(qapp):
    controller = WizardController()

    controller.config = _legacy_config()

    assert "Water_Evaporation" not in controller.config["evaluation_items"]
    assert controller.config["evaluation_items"]["Open_Water_Evaporation"] is True
    assert controller.config["evaluation_items"]["Soil_Evaporation"] is True
    assert "Bare_Soil_Evaporation_ref_source" not in controller.config["ref_data"]["general"]


def test_generate_config_yaml_exports_legacy_loaded_variable_names_as_canonical():
    data = yaml.safe_load(ConfigManager().generate_config_yaml(_legacy_config()))

    assert data["evaluation"]["variables"] == ["Open_Water_Evaporation", "Soil_Evaporation"]
    assert data["reference"]["Open_Water_Evaporation"] == ["GLEAM", "ALT"]
    assert data["reference"]["Soil_Evaporation"] == "GLEAM"
    assert data["simulation"]["CaseA"]["variables"] == {
        "Open_Water_Evaporation": {"varname": "Ew"},
        "Soil_Evaporation": {"varname": "Eb"},
    }


def test_migrate_gui_variable_config_rejects_conflicting_explicit_variable_configs():
    config = _legacy_config()
    config["sim_data"]["source_configs"]["CaseA"]["variables"]["Soil_Evaporation"] = {"varname": "different"}
    before = copy.deepcopy(config)

    with pytest.raises(ValueError, match="Conflicting values"):
        migrate_gui_variable_config(config)

    assert config == before


def test_migrated_config_stays_idempotent_after_yaml_save_load(tmp_path):
    manager = ConfigManager()
    migrated = migrate_gui_variable_config(_legacy_config())
    path = tmp_path / "config.yaml"

    manager.save_to_yaml(migrated, str(path))
    loaded = manager.load_from_yaml(str(path))

    assert migrate_gui_variable_config(loaded) == migrated
