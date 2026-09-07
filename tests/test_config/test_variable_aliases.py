"""Logical aliases must not rename native fields or lose variable overrides."""

from copy import deepcopy

import pytest

from openbench.config.loader import ConfigError, _build_config


@pytest.mark.parametrize(
    ("old", "canonical"),
    [("Bare_Soil_Evaporation", "Soil_Evaporation"), ("Water_Evaporation", "Open_Water_Evaporation")],
)
def test_loader_migrates_aliases_and_preserves_native_fields(old, canonical):
    raw = {
        "project": {"name": "aliases", "output_dir": ".", "years": [2000, 2001]},
        "evaluation": {"variables": [old, canonical]},
        "reference": {
            old: "Ref",
            "overrides": {"Ref": {"variables": {old: {"varname": old, "sub_dir": old}}}},
        },
        "simulation": {
            "_defaults": {"variables": {old: {"varunit": "mm day-1", "prefix": old}}},
            "Case": {"model": old, "root_dir": old, "variables": {canonical: {"varname": old}}},
        },
    }
    original = deepcopy(raw)

    cfg = _build_config(raw)

    assert cfg.evaluation.variables == [canonical]
    assert cfg.reference.sources == {canonical: "Ref"}
    assert cfg.reference.overrides["Ref"]["variables"] == {canonical: {"varname": old, "sub_dir": old}}
    assert cfg.simulation["Case"].variables == {canonical: {"varunit": "mm day-1", "prefix": old, "varname": old}}
    assert cfg.simulation["Case"].model == old
    assert cfg.simulation["Case"].root_dir == old
    assert raw == original


def test_loader_rejects_conflicting_alias_overrides():
    raw = {
        "project": {"name": "aliases", "output_dir": ".", "years": [2000, 2001]},
        "evaluation": {"variables": ["Soil_Evaporation"]},
        "reference": {"Soil_Evaporation": "Ref"},
        "simulation": {
            "Case": {
                "model": "CoLM2024",
                "root_dir": "/sim",
                "variables": {
                    "Bare_Soil_Evaporation": {"varname": "old"},
                    "Soil_Evaporation": {"varname": "new"},
                },
            }
        },
    }
    with pytest.raises(ConfigError, match="duplicate variable"):
        _build_config(raw)


@pytest.mark.parametrize("variable", ["Soil_Evaporation", "Bare_Soil_Evaporation"])
def test_alias_config_binds_real_reference_and_model(variable, monkeypatch, tmp_path):
    from openbench.config.adapter import build_legacy_namelists
    from openbench.data.registry.manager import RegistryManager
    from openbench.gui.config_manager import model_definition_from_registry

    registry = RegistryManager(user_dir=tmp_path)
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    cfg = _build_config(
        {
            "project": {"name": "aliases", "output_dir": ".", "years": [2000, 2001]},
            "evaluation": {"variables": [variable]},
            "reference": {variable: "GLEAM_v4.2a_LowRes"},
            "simulation": {"Case": {"model": "CoLM2024", "root_dir": "/sim"}},
        }
    )
    main, ref, sim = build_legacy_namelists(cfg)
    assert main["evaluation_items"] == {"Soil_Evaporation": True}
    assert ref["Soil_Evaporation"]["GLEAM_v4.2a_LowRes_varname"] == "Eb"
    assert sim["Soil_Evaporation"]["Case_varname"] == "f_fevpg"
    model_def = model_definition_from_registry("CoLM2024", ["Soil_Evaporation"], registry=registry)
    assert model_def["Soil_Evaporation"]["varname"] == "f_fevpg"


def test_gui_migration_round_trip_preserves_explicit_reference_fields():
    import yaml

    from openbench.gui.config_manager import ConfigManager

    old = "Bare_Soil_Evaporation"
    manager = ConfigManager()
    gui_config = {
        "general": {"basename": "aliases", "basedir": "/out"},
        "evaluation_items": {old: True},
        "ref_data": {
            "general": {f"{old}_ref_source": "GLEAM"},
            "source_configs": {
                f"{old}::GLEAM": {
                    "_explicit_override": True,
                    "general": {"root_dir": f"/ref/{old}"},
                    "varname": old,
                    "prefix": old,
                    "sub_dir": old,
                }
            },
        },
    }

    exported = yaml.safe_load(manager.generate_config_yaml(gui_config))
    override = exported["reference"]["overrides"]["GLEAM"]
    assert override == {
        "root_dir": f"/ref/{old}",
        "variables": {"Soil_Evaporation": {"varname": old, "prefix": old, "sub_dir": old}},
    }
    restored = manager.unified_to_gui_config(exported)
    reexported = yaml.safe_load(manager.generate_config_yaml(restored))
    assert reexported["reference"] == exported["reference"]
