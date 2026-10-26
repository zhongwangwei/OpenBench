"""Datum declarations survive configuration without changing legacy defaults."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from openbench.config import adapter
from openbench.data.registry import manager
from openbench.data.registry.schema import VariableMapping

SPEC = dict(
    quantity="orthometric_height", source="EGM2008", target="EGM96", source_geoid="08.pgm", target_geoid="96.pgm"
)


def reference():
    return manager._build_reference(
        dict(
            name="DatumRef",
            data_type="stn",
            tim_res="Day",
            root_dir="/ref",
            years=[2000, 2001],
            variables={"Water_Surface_Elevation": dict(varname="height", varunit="m", vertical_datum=SPEC)},
        )
    )


def test_reference_mapping_roundtrip_and_clear():
    assert "vertical_datum" not in VariableMapping("v", "m").to_dict()
    ref = reference()
    rebuilt = manager._build_reference(ref.to_dict())
    assert rebuilt.variables["Water_Surface_Elevation"].vertical_datum == SPEC
    cleared = manager._deep_merge_reference(
        rebuilt, {"variables": {"Water_Surface_Elevation": {"vertical_datum": None}}}
    )
    assert cleared.variables["Water_Surface_Elevation"].vertical_datum is None


@pytest.mark.parametrize("value", ["EGM2008", {}, {**SPEC, "oops": 1}, {**SPEC, "source": "EGM84"}])
def test_validation(value):
    from openbench.config.loader import ConfigError, _validated_variables_mapping

    with pytest.raises((ConfigError, ValueError)):
        _validated_variables_mapping({"Water_Surface_Elevation": {"vertical_datum": value}}, "simulation.M.variables")


def test_adapter_runtime_json_paths(monkeypatch, tmp_path):
    from openbench.config.runtime_info import GeneralInfoReader
    from openbench.config.schema import (
        EvaluationConfig,
        OpenBenchConfig,
        ProjectConfig,
        ReferenceConfig,
        SimulationEntry,
    )

    ref = replace(reference(), root_dir=str(tmp_path / "reference"))
    model = manager._build_model(
        dict(
            name="M",
            variables={"Water_Surface_Elevation": dict(varname="height", varunit="m")},
        )
    )
    registry = SimpleNamespace(
        get_model=lambda name: model, get_resolution_variants=lambda name: {}, get_reference=lambda name, **kw: ref
    )
    monkeypatch.setattr(manager, "get_registry", lambda: registry)
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="datum", output_dir=str(tmp_path), years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Water_Surface_Elevation"]),
        reference=ReferenceConfig(sources={"Water_Surface_Elevation": ref.name}),
        simulation={"Sim": SimulationEntry(model="M", root_dir="/sim", tim_res="Day", grid_res=0.5)},
    )
    _, ref_nml, sim_nml = adapter.build_legacy_namelists(cfg)
    assert "Sim_vertical_datum" not in ref_nml["Water_Surface_Elevation"]
    # Exercise the actual Fortran namelist serialization boundary.
    import f90nml

    path = tmp_path / "ref.nml"
    f90nml.write(ref_nml, str(path))
    restored = f90nml.read(str(path))
    spec = json.loads(restored["water_surface_elevation"]["datumref_vertical_datum"])
    assert spec["source_geoid"] == str(tmp_path / "reference" / "08.pgm")
    reader = GeneralInfoReader.__new__(GeneralInfoReader)
    reader.item = "Water_Surface_Elevation"
    restored["Water_Surface_Elevation"]["DatumRef_fulllist"] = "/ref/stations.csv"
    reader._set_source_attributes(sim_nml, restored, "Water_Surface_Elevation", "Sim", "DatumRef")
    assert reader.ref_vertical_datum == spec
    assert reader.sim_vertical_datum is None
    monkeypatch.setenv("DATUM_ROOT", str(tmp_path))
    assert adapter._resolved_vertical_datum({**SPEC, "source_geoid": "$DATUM_ROOT/08.pgm"}, "/ref")[
        "source_geoid"
    ] == str(tmp_path / "08.pgm")


def test_geoid_signature_detects_same_path_change(tmp_path):
    from openbench.runner.hashing import COMMON_ALGORITHM_SOURCE_MODULES, input_file_signature

    for name in ("08.pgm", "96.pgm"):
        (tmp_path / name).write_bytes(b"first")
    section = {"S_dir": str(tmp_path), "S_vertical_datum": json.dumps(SPEC)}
    before = input_file_signature(section, "S")
    (tmp_path / "08.pgm").write_bytes(b"other")
    after = input_file_signature(section, "S")
    assert len(before["files"]) == 2
    assert before != after
    assert "openbench.data.vertical_datum" in COMMON_ALGORITHM_SOURCE_MODULES


def test_gui_hidden_and_path_roundtrip():
    from openbench.gui.config_manager import _transform_vertical_datum

    transformed = _transform_vertical_datum(SPEC, lambda p: "/remote" + p, "/stations")
    assert transformed["source_geoid"] == "/remote/stations/08.pgm"
    assert transformed["target_geoid"] == "/remote/stations/96.pgm"
    assert SPEC["source_geoid"] == "08.pgm"
    pytest.importorskip("PySide6")
    from openbench.gui.pages.page_registry import _merge_reference_editor_dataset

    ref = reference()
    from openbench.gui.pages.page_ref_data import _registry_source_data

    source = _registry_source_data("Water_Surface_Elevation", ref.name, SimpleNamespace(get_reference=lambda name: ref))
    assert source["vertical_datum"] == SPEC
    edited = replace(ref, variables={"Water_Surface_Elevation": VariableMapping("h", "m")})
    assert _merge_reference_editor_dataset(ref, edited).variables["Water_Surface_Elevation"].vertical_datum == SPEC


def test_overlay_behavior_field():
    from openbench.data.registry.overlay_audit import _MERGED_VARIABLE_FIELDS

    assert all("vertical_datum" in _MERGED_VARIABLE_FIELDS[k] for k in ("references", "models"))


def test_saved_reference_clearing_bundled_datum(monkeypatch, tmp_path):
    import yaml

    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "reference_catalog.yaml").write_text(yaml.safe_dump({"DatumRef": reference().to_dict()}))
    (bundled / "model_catalog.yaml").write_text("{}")
    user = tmp_path / "user"
    catalog = user / "references" / "reference_catalog.yaml"
    monkeypatch.setattr(manager, "REGISTRY_DIR", bundled)
    monkeypatch.setattr(manager, "get_writable_reference_catalog_path", lambda: catalog)
    registry = manager.RegistryManager(user_dir=user)
    ref = registry.get_reference("DatumRef")
    mapping = replace(ref.variables["Water_Surface_Elevation"], vertical_datum=None)
    registry.save_reference("DatumRef", replace(ref, variables={"Water_Surface_Elevation": mapping}))
    raw = yaml.safe_load(catalog.read_text())
    assert raw["DatumRef"]["variables"]["Water_Surface_Elevation"]["vertical_datum"] is None
    assert (
        manager.RegistryManager(user_dir=user)
        .get_reference("DatumRef")
        .variables["Water_Surface_Elevation"]
        .vertical_datum
        is None
    )


@pytest.mark.parametrize("spec", [SPEC, None])
def test_gui_unified_roundtrip_remote_paths(spec):
    import yaml

    from openbench.gui.config_manager import ConfigManager

    config = dict(
        project=dict(name="datum", output_dir="/output", years=[2000, 2001]),
        evaluation=dict(variables=["Water_Surface_Elevation"]),
        reference={
            "Water_Surface_Elevation": "DatumRef",
            "overrides": {"DatumRef": {"variables": {"Water_Surface_Elevation": {"vertical_datum": spec}}}},
        },
        simulation={
            "Sim": {
                "model": "CoLM2024",
                "root_dir": "/sim",
            }
        },
        metrics=["RMSE"],
    )
    gui = ConfigManager()
    state = gui.unified_to_gui_config(config)
    output = yaml.safe_load(gui.generate_config_yaml(state, path_transform=lambda p: "/remote/" + p.lstrip("/")))
    ref = output["reference"]["overrides"]["DatumRef"]["variables"]["Water_Surface_Elevation"]["vertical_datum"]
    if spec is None:
        assert ref is None
    else:
        assert ref["source_geoid"] == "08.pgm"  # Catalog root remains runtime-owned.
        assert ref["target_geoid"] == "96.pgm"


def test_relative_root_resolution_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    spec = adapter._resolved_vertical_datum(SPEC, "data")
    assert spec["source_geoid"] == str(tmp_path / "data/08.pgm")
    assert adapter._resolved_vertical_datum(spec, "data") == spec


@pytest.mark.parametrize("route", ["legacy", "organized", "copy"])
def test_gui_source_file_exports_resolve_geoid_paths(tmp_path, route):
    import yaml

    from openbench.gui.config_manager import ConfigManager

    manager = ConfigManager()
    item = "Water_Surface_Elevation"
    content = {"general": {}, item: {"vertical_datum": SPEC}}
    dest = tmp_path / "export.yaml"
    if route == "legacy":
        manager._write_source_config(content, str(dest), [item], str(tmp_path))
    elif route == "organized":
        manager._write_source_config_organized(content, str(dest), [item], str(tmp_path), "ref")
    else:
        source = tmp_path / "source.yaml"
        source.write_text(yaml.safe_dump(content))
        manager._copy_namelist_filtered(str(source), str(dest), [item], str(tmp_path))
    result = yaml.safe_load(dest.read_text())[item]["vertical_datum"]
    assert result["source_geoid"] == "08.pgm"
    assert result["target_geoid"] == "96.pgm"


@pytest.mark.parametrize("route", ["legacy", "organized", "copy"])
def test_gui_datum_base_is_source_not_repository(tmp_path, route):
    import yaml

    from openbench.gui.config_manager import ConfigManager

    gui = ConfigManager()
    item = "Water_Surface_Elevation"
    general = {"root_dir": str(tmp_path / "data" / "stations")}
    content = {"general": general, item: {"sub_dir": "heights", "vertical_datum": SPEC}}
    dest = tmp_path / "out.yaml"
    if route == "legacy":
        gui._write_source_config(content, str(dest), [item], str(tmp_path))
    elif route == "organized":
        gui._write_source_config_organized(
            {"_general": general, item: content[item]}, str(dest), [item], str(tmp_path), "ref"
        )
    else:
        source = tmp_path / "input.yaml"
        source.write_text(yaml.safe_dump(content))
        gui._copy_namelist_filtered(str(source), str(dest), [item], str(tmp_path))
    result = yaml.safe_load(dest.read_text())[item]["vertical_datum"]
    assert result["source_geoid"] == str(tmp_path / "data" / "stations" / "08.pgm")
    assert result["target_geoid"] == str(tmp_path / "data" / "stations" / "96.pgm")


def test_gui_remote_resolves_datum_before_mapping():
    import yaml

    from openbench.gui.config_manager import ConfigManager

    item = "Water_Surface_Elevation"
    config = dict(
        project=dict(name="datum", output_dir="/out", years=[2000, 2001]),
        evaluation={"variables": [item]},
        reference={
            item: "DatumRef",
            "data_root": "/fallback",
            "overrides": {
                "DatumRef": {
                    "root_dir": "/data/stations",
                    "variables": {item: {"vertical_datum": SPEC, "sub_dir": "height"}},
                }
            },
        },
        simulation={"Sim": {"model": "CoLM2024", "root_dir": "/data/model"}},
        metrics=["RMSE"],
    )
    gui = ConfigManager()
    result = yaml.safe_load(
        gui.generate_config_yaml(gui.unified_to_gui_config(config), path_transform=lambda p: "/remote" + p)
    )
    ref = result["reference"]["overrides"]["DatumRef"]["variables"][item]["vertical_datum"]
    assert ref["source_geoid"] == "/remote/data/stations/08.pgm"
    state = gui.unified_to_gui_config(result)
    again = yaml.safe_load(gui.generate_config_yaml(state))
    assert again["reference"]["overrides"]["DatumRef"]["variables"][item]["vertical_datum"] == ref


@pytest.mark.parametrize("root", ["/data/stations", r"C:\data\stations", r"\\server\stations"])
def test_gui_datum_paths_on_windows_client(monkeypatch, root):
    import ntpath

    from openbench.gui import config_manager

    # Simulate the client's path operations without changing the host filesystem.
    monkeypatch.setattr(config_manager, "os", SimpleNamespace(path=ntpath))
    seen = []

    def transform(path):
        seen.append(path)
        return "/remote/" + path.replace("\\", "/").lstrip("/")

    result = config_manager._transform_vertical_datum(SPEC, transform, root)
    expected = root + "/08.pgm" if root.startswith("/") else ntpath.join(root, "08.pgm")
    assert seen[0] == expected
    assert "\\" not in result["source_geoid"]
    assert config_manager._transform_vertical_datum(result, lambda path: path, root) == result
