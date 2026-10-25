"""Logical aliases must not rename native fields or lose variable overrides."""

from copy import deepcopy

import pytest

from openbench.config.loader import ConfigError, _build_config


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Water_Evaporation", "Open_Water_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
        ("transpiration", "Canopy_Transpiration"),
        ("RUNOFF", "Total_Runoff"),
        ("CANOPY_INTERCEPTION", "Canopy_Evaporation"),
        ("canopy_interception", "Canopy_Evaporation"),
    ],
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


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
    ],
)
def test_loader_rejects_conflicting_alias_overrides(old, canonical):
    raw = {
        "project": {"name": "aliases", "output_dir": ".", "years": [2000, 2001]},
        "evaluation": {"variables": [canonical]},
        "reference": {canonical: "Ref"},
        "simulation": {
            "Case": {
                "model": "CoLM2024",
                "root_dir": "/sim",
                "variables": {
                    old: {"varname": "old"},
                    canonical: {"varname": "new"},
                },
            }
        },
    }
    with pytest.raises(ConfigError, match="duplicate variable"):
        _build_config(raw)


@pytest.mark.parametrize(
    ("variable", "canonical", "reference", "ref_var", "sim_var"),
    [
        ("Soil_Evaporation", "Soil_Evaporation", "GLEAM_v4.2a_LowRes", "Eb", "f_fevpg"),
        ("Bare_Soil_Evaporation", "Soil_Evaporation", "GLEAM_v4.2a_LowRes", "Eb", "f_fevpg"),
        ("Transpiration", "Canopy_Transpiration", "GLEAM_v4.3a_LowRes", "Et", "f_etr"),
        ("Canopy_Transpiration", "Canopy_Transpiration", "GLEAM_v4.3a_LowRes", "Et", "f_etr"),
        ("Runoff", "Total_Runoff", "GRFR_LowRes", "ro", "f_rnof"),
        ("Total_Runoff", "Total_Runoff", "GRFR_LowRes", "ro", "f_rnof"),
    ],
)
def test_alias_config_binds_real_reference_and_model(
    variable, canonical, reference, ref_var, sim_var, monkeypatch, tmp_path
):
    from openbench.config.adapter import build_legacy_namelists
    from openbench.data.registry.manager import RegistryManager
    from openbench.gui.config_manager import model_definition_from_registry

    registry = RegistryManager(user_dir=tmp_path)
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    cfg = _build_config(
        {
            "project": {"name": "aliases", "output_dir": ".", "years": [2000, 2001]},
            "evaluation": {"variables": [variable]},
            "reference": {variable: reference},
            "simulation": {"Case": {"model": "CoLM2024", "root_dir": "/sim"}},
        }
    )
    main, ref, sim = build_legacy_namelists(cfg)
    assert main["evaluation_items"] == {canonical: True}
    assert ref[canonical][f"{reference}_varname"] == ref_var
    assert sim[canonical]["Case_varname"] == sim_var
    model_def = model_definition_from_registry("CoLM2024", [canonical], registry=registry)
    assert model_def[canonical]["varname"] == sim_var


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
    ],
)
def test_gui_migration_round_trip_preserves_explicit_reference_fields(old, canonical):
    import yaml

    from openbench.gui.config_manager import ConfigManager

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
        "variables": {canonical: {"varname": old, "prefix": old, "sub_dir": old}},
    }
    restored = manager.unified_to_gui_config(exported)
    reexported = yaml.safe_load(manager.generate_config_yaml(restored))
    assert reexported["reference"] == exported["reference"]
