"""Old scan catalogs must not make every registry consumer unavailable."""

from copy import deepcopy

import pytest
import yaml

from openbench.data.registry import scanner
from openbench.data.registry.manager import RegistryManager
from openbench.data.registry.scanner import ScannedDataset


@pytest.mark.parametrize("kind", ["references", "models"])
@pytest.mark.parametrize("individual", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_legacy_alias_collision_keeps_profile_and_original_file(tmp_path, caplog, kind, individual, reverse):
    old = {"varname": "old_native", "varunit": "mm day-1", "sub_dir": "Water/Transpiration/Demo"}
    new = {"varname": "new_native", "varunit": "mm hr-1", "sub_dir": "Water/Canopy_Transpiration/Demo"}
    pairs = [("Transpiration", old), ("Canopy_Transpiration", new)]
    if reverse:
        pairs.reverse()
    entry = {
        "name": "Demo",
        "data_type": "grid",
        "tim_res": "Month",
        "description": "user edit",
        "variables": dict(pairs + [("Custom", {"varname": "custom", "varunit": "1"})]),
    }
    directory = tmp_path / kind
    directory.mkdir()
    path = directory / (
        "demo.yaml" if individual else ("reference_catalog.yaml" if kind == "references" else "model_catalog.yaml")
    )
    content = yaml.safe_dump({"Demo": entry}, sort_keys=False)
    path.write_text(content)
    registry = RegistryManager(user_dir=tmp_path)
    result = registry.get_reference("Demo") if kind == "references" else registry.get_model("Demo")
    assert result is not None
    assert result.description == "user edit"
    assert result.variables["Canopy_Transpiration"].varname == "new_native"
    assert result.variables["Canopy_Transpiration"].varunit == "mm hr-1"
    assert result.variables["Custom"].varname == "custom"
    assert "Transpiration" not in result.variables
    assert path.read_text() == content
    assert str(path) in caplog.text
    assert "Canopy_Transpiration" in caplog.text and "Transpiration" in caplog.text
    assert "unchanged" in caplog.text


def test_legacy_alias_migration_applies_to_existing_bundled_profile(tmp_path, caplog):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "CoLM2024": {
                    "description": "custom model",
                    "variables": {
                        "Transpiration": {"varname": "old"},
                        "Canopy_Transpiration": {"varname": "custom_new"},
                    },
                }
            }
        )
    )
    model = RegistryManager(user_dir=tmp_path).get_model("CoLM2024")
    assert model.description == "custom model"
    assert model.variables["Canopy_Transpiration"].varname == "custom_new"
    assert "Surface_Runoff" in model.variables


@pytest.mark.parametrize("reverse", [False, True])
def test_scan_old_and_new_directories_uses_canonical_without_mutating_input(tmp_path, monkeypatch, caplog, reverse):
    monkeypatch.setattr(scanner, "get_reference_profile", lambda name: None)
    old, canonical = "Transpiration", "Canopy_Transpiration"
    pairs = [(old, f"Water/{old}/Demo"), (canonical, f"Water/{canonical}/Demo")]
    if reverse:
        pairs.reverse()
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables=dict(pairs),
    )
    original = deepcopy(scanned.variables)
    result = scanner._build_variables(scanned, {}, None, None)
    assert set(result) == {canonical}
    assert result[canonical]["sub_dir"] == f"Water/{canonical}/Demo"
    assert scanned.variables == original
    assert "Transpiration" in caplog.text


@pytest.mark.parametrize("placeholder", [{}, {"varname": "", "varunit": ""}, None])
def test_alias_placeholder_and_explicit_deletion_remain_distinct(tmp_path, placeholder):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "CoLM2024": {
                    "variables": {"Canopy_Transpiration": placeholder, "Transpiration": {"varname": "legacy_custom"}}
                }
            }
        )
    )
    variables = RegistryManager(user_dir=tmp_path).get_model("CoLM2024").variables
    if placeholder is None:
        assert "Canopy_Transpiration" not in variables
    else:
        assert variables["Canopy_Transpiration"].varname == "legacy_custom"


def test_batch_rescan_migrates_old_aliases_with_backup_and_keeps_other_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "get_reference_profile", lambda name: None)
    path = tmp_path / "reference_catalog.yaml"
    existing = {
        "Demo_LowRes": {
            "name": "Demo_LowRes",
            "data_type": "grid",
            "tim_res": "Month",
            "description": "custom note",
            "variables": {
                "Transpiration": {"varname": "legacy", "varunit": "mm day-1"},
                "Canopy_Transpiration": {"varname": "preferred", "varunit": "mm hr-1"},
            },
        }
    }
    content = yaml.safe_dump(existing)
    path.write_text(content)
    datasets = [
        ScannedDataset(
            name=name,
            resolution="LowRes",
            category="Water",
            data_type="grid",
            root_dir=str(tmp_path),
            variables={"Transpiration": "old", "Canopy_Transpiration": "new"},
        )
        for name in ("Demo", "Other")
    ]
    scanner.register_scanned_datasets_batch(datasets, catalog_path=path)
    updated = yaml.safe_load(path.read_text())
    assert set(updated) == {"Demo_LowRes", "Other_LowRes"}
    assert updated["Demo_LowRes"]["description"] == "custom note"
    assert updated["Demo_LowRes"]["variables"]["Canopy_Transpiration"]["varname"] == "preferred"
    assert set(updated["Other_LowRes"]["variables"]) == {"Canopy_Transpiration"}
    assert path.with_suffix(".yaml.bak").read_text() == content
    assert any(p.read_text() == content for p in tmp_path.glob("*.yaml.*.bak"))


def test_canonical_directory_keeps_its_remote_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "get_reference_profile", lambda name: None)
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Transpiration": "old", "Canopy_Transpiration": "new"},
    )
    scanned.nc_inspections = {
        "Transpiration": {"varname": "old_native", "varunit": "mm day-1"},
        "Canopy_Transpiration": {"varname": "new_native", "varunit": "mm hr-1"},
    }
    result = scanner._build_variables(scanned, {}, None, None)
    assert result["Canopy_Transpiration"]["varname"] == "new_native"
    assert result["Canopy_Transpiration"]["varunit"] == "mm hr-1"


@pytest.mark.parametrize("kind", ["references", "models"])
def test_explicit_save_of_migrated_entry_preserves_original_backup(tmp_path, monkeypatch, kind):
    from openbench.data.registry import manager

    monkeypatch.setattr(manager, "get_user_config_dir", lambda *args: tmp_path)
    directory = tmp_path / kind
    directory.mkdir()
    filename = "reference_catalog.yaml" if kind == "references" else "model_catalog.yaml"
    path = directory / filename
    entry = {
        "name": "Demo",
        "data_type": "grid",
        "tim_res": "Month",
        "variables": {"Transpiration": {"varname": "old"}, "Canopy_Transpiration": {"varname": "canonical"}},
    }
    content = yaml.safe_dump({"Demo": entry})
    path.write_text(content)
    registry = RegistryManager(user_dir=tmp_path)
    if kind == "references":
        registry.save_reference("Demo", registry.get_reference("Demo"))
    else:
        registry.save_model("Demo", registry.get_model("Demo"))
    assert set(yaml.safe_load(path.read_text())["Demo"]["variables"]) == {"Canopy_Transpiration"}
    assert path.with_suffix(".yaml.bak").read_text() == content


@pytest.mark.parametrize("kind", ["references", "models"])
@pytest.mark.parametrize("bundled", [False, True])
def test_independent_alias_file_save_matches_registry_replay(tmp_path, monkeypatch, kind, bundled):
    from dataclasses import replace

    from openbench.data.registry import manager, overlay_audit

    monkeypatch.setattr(manager, "get_user_config_dir", lambda *args: tmp_path)
    name = ("GLEAM_v4.3a_LowRes" if kind == "references" else "CoLM2024") if bundled else "Demo"
    directory = tmp_path / kind
    directory.mkdir()
    path = directory / ("reference_catalog.yaml" if kind == "references" else "model_catalog.yaml")
    raw = {
        "name": name,
        "data_type": "grid",
        "tim_res": "Month",
        "variables": {
            "Transpiration": {"varname": "old", "varunit": "mm day-1"},
            "Canopy_Transpiration": {"varname": "canonical", "varunit": "mm hr-1"},
        },
    }
    # Exercise both catalog replay and the independent-file replay that wins afterwards.
    path.write_text(yaml.safe_dump({name: raw}))
    separate = directory / "legacy.yaml"
    content = yaml.safe_dump({name: raw})
    separate.write_text(content)
    registry = RegistryManager(user_dir=tmp_path)
    get = registry.get_reference if kind == "references" else registry.get_model
    save = registry.save_reference if kind == "references" else registry.save_model
    loaded = get(name)
    overlay_audit.check_saved_entry_against_separate_files(kind, path, name, loaded)
    save(name, loaded)
    save(name, replace(loaded, description="updated description"))
    fresh = RegistryManager(user_dir=tmp_path)
    restored = fresh.get_reference(name) if kind == "references" else fresh.get_model(name)
    assert restored.description == "updated description"
    assert restored.variables["Canopy_Transpiration"].varname == "canonical"
    assert separate.read_text() == content
    changed = replace(restored.variables["Canopy_Transpiration"], varname="would_be_undone")
    with pytest.raises(overlay_audit.SeparateFileOverride, match="variables.Canopy_Transpiration"):
        save(name, replace(restored, variables={**restored.variables, "Canopy_Transpiration": changed}))
