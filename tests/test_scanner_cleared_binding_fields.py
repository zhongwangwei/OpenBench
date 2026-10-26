"""A rescan must preserve explicitly cleared reference-binding metadata."""

import copy

import pytest
import yaml

from openbench.data.registry import manager, overlay_audit, scanner
from openbench.data.registry.schema import reference_years


@pytest.mark.parametrize("field", ["years", "vertical_datum"])
def test_rescan_preserves_binding_clear_through_descriptor_delta_and_save(tmp_path, monkeypatch, field):
    raw = yaml.safe_load((manager.REGISTRY_DIR / "reference_catalog.yaml").read_text())
    name = "ERA5LAND_LowRes"
    base = copy.deepcopy(raw[name])
    variable = "Latent_Heat"
    if field == "vertical_datum":
        # A synthetic bundled declaration exercises the same nullable field contract.
        base["variables"][variable][field] = {"source_geoid": "08.pgm", "target_geoid": "96.pgm"}
    assert base["variables"][variable][field] is not None
    clear = {"variables": {variable: {field: None}}}
    existing = scanner._merge_descriptor_overlay(base, clear)
    scanned = scanner.ScannedDataset(
        name="ERA5LAND",
        resolution="LowRes",
        category="Heat",
        data_type="grid",
        root_dir=str(tmp_path / "data"),
        variables={variable: "Heat/Latent_Heat/ERA5LAND"},
    )
    catalog = {}
    scanner._register_to_dict(scanned, catalog, existing_descriptor=existing)
    descriptor = catalog[name]
    assert field in descriptor["variables"][variable]
    assert descriptor["variables"][variable][field] is None
    delta = scanner._descriptor_overlay_diff(base, descriptor)
    assert delta["variables"][variable][field] is None
    base_obj = manager._build_reference({**base, "name": name})
    effective = manager._deep_merge_reference(base_obj, delta)
    assert getattr(effective.variables[variable], field) is None
    if field == "years":
        assert reference_years(effective, effective.variables[variable]) == [1950, 2022]

    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    (bundled_dir / "reference_catalog.yaml").write_text(yaml.safe_dump({name: base}))
    (bundled_dir / "model_catalog.yaml").write_text("{}")
    monkeypatch.setattr(manager, "REGISTRY_DIR", bundled_dir)
    monkeypatch.setattr(overlay_audit, "_bundled_catalog", lambda kind: {name: base} if kind == "references" else {})
    user_dir = tmp_path / "user"
    path = user_dir / "references" / "reference_catalog.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({name: clear}))
    monkeypatch.setattr(manager, "get_writable_reference_catalog_path", lambda: path)
    monkeypatch.setattr(scanner, "_load_builtin_catalog_for_user_overlay", lambda _: {name: base})
    scanner.register_scanned_dataset(scanned, catalog_path=path)
    saved = yaml.safe_load(path.read_text())[name]
    assert saved["variables"][variable][field] is None
    # Registry reload applies the saved sparse overlay to its bundled descriptor.
    reloaded = manager.RegistryManager(user_dir=user_dir).get_reference(name)
    assert getattr(reloaded.variables[variable], field) is None
    if field == "years":
        assert reference_years(reloaded, reloaded.variables[variable]) == [1950, 2022]


def test_rescan_keeps_other_null_and_empty_fields_omitted(tmp_path):
    scanned = scanner.ScannedDataset(
        name="NoProfile",
        resolution="LowRes",
        category="Heat",
        data_type="grid",
        root_dir=str(tmp_path / "data"),
        variables={"Latent_Heat": "Heat/Latent_Heat/NoProfile"},
    )
    existing = {
        "variables": {
            "Latent_Heat": {
                "varname": "le",
                "varunit": "W m-2",
                "prefix": None,
                "suffix": None,
                "compute": None,
                "years": "",
                "vertical_datum": "",
            }
        }
    }
    variables = scanner._build_variables(scanned, {}, existing, None)
    for field in ("prefix", "suffix", "compute", "years", "vertical_datum"):
        assert field not in variables["Latent_Heat"]


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("explicit_clear", [False, True])
def test_profile_defaults_do_not_revive_explicit_binding_clear(tmp_path, monkeypatch, composite, explicit_clear):
    variable = "Water_Surface_Elevation"
    defaults = {
        "varname": "height",
        "varunit": "m",
        "prefix": "profile_",
        "years": [2001, 2010],
        "vertical_datum": {"source_geoid": "08.pgm", "target_geoid": "96.pgm"},
    }
    monkeypatch.setattr(scanner, "get_reference_profile", lambda _: {"variables": {variable: defaults}})
    existing_var = {"varname": "height", "varunit": "m", "prefix": None}
    if explicit_clear:
        existing_var.update(years=None, vertical_datum=None)
    existing = {"variables": {variable: existing_var}}
    scanned = scanner.ScannedDataset(
        name="ProfileClear",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path / "data"),
        variables={"Composite" if composite else variable: "leaf"},
    )
    catalog = {}
    scanner._register_to_dict(scanned, catalog, existing_descriptor=existing)
    result = catalog[scanned.registry_name]["variables"][variable]
    for field in ("years", "vertical_datum"):
        assert result[field] == (None if explicit_clear else defaults[field])
    assert result["prefix"] == "profile_"  # Other nulls keep the prior fill-default behavior.


def test_profile_explicit_overwrite_still_replaces_nullable_binding_fields():
    target = {"years": None, "vertical_datum": None}
    defaults = {"years": [2001, 2010], "vertical_datum": {"source_geoid": "08.pgm", "target_geoid": "96.pgm"}}
    scanner._merge_profile_var_fields(target, defaults, "height", overwrite_existing=True)
    assert all(target[field] == value for field, value in defaults.items())
