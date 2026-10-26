"""Water-surface elevation is selectable, with correction configured on observations only."""

import ast
from pathlib import Path

import pytest
import yaml

from openbench.config.loader import ConfigError, load_config

SPEC = {"source_geoid": "egm2008-2_5.pgm", "target_geoid": "egm96-5.pgm"}


def _config():
    return yaml.safe_load((Path(__file__).parent / "test_config/fixtures/minimal.yaml").read_text())


def _load(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


def test_gui_standard_water_surface_elevation(tmp_path):
    from openbench.data.registry import manager

    tree = ast.parse((Path(manager.__file__).parents[2] / "gui/pages/page_evaluation.py").read_text())
    groups = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EVALUATION_ITEMS" for target in node.targets)
    )
    assert "Water_Surface_Elevation" in groups["Water Bodies"]
    assert sum(items.count("Water_Surface_Elevation") for items in groups.values()) == 1
    model = manager.RegistryManager(user_dir=tmp_path).get_model("TE")
    assert model.variables["Water_Surface_Elevation"].varname == "sfcelv"
    assert model.variables["Water_Surface_Elevation"].varunit == "m"


@pytest.mark.parametrize("scope", ["CoLM2024", "_defaults"])
@pytest.mark.parametrize("value", [SPEC, None])
def test_loader_rejects_simulation_vertical_datum(tmp_path, scope, value):
    config = _config()
    config["simulation"].setdefault(scope, {})["variables"] = {"Water_Surface_Elevation": {"vertical_datum": value}}
    with pytest.raises(ConfigError, match=rf"simulation\.{scope}.*vertical_datum.*reference-only"):
        _load(tmp_path, config)


@pytest.mark.parametrize("value", [SPEC, None])
def test_reference_vertical_datum_still_loads(tmp_path, value):
    config = _config()
    config["evaluation"]["variables"] = ["Water_Surface_Elevation"]
    config["reference"] = {
        "Water_Surface_Elevation": "MyStations",
        "overrides": {"MyStations": {"variables": {"Water_Surface_Elevation": {"vertical_datum": value}}}},
    }
    result = _load(tmp_path, config)
    assert result.reference.overrides["MyStations"]["variables"]["Water_Surface_Elevation"]["vertical_datum"] == value
