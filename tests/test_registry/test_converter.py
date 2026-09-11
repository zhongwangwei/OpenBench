"""Tests for reference definition converter."""

from pathlib import Path

import yaml

from openbench.data.registry.converter import convert_old_reference


def test_convert_gleam(tmp_path):
    """Convert the bundled legacy GLEAM definition without a private checkout."""
    old_path = Path(__file__).resolve().parents[1] / "test_config/fixtures/old_json/ref_def/GLEAM.json"
    out_path = tmp_path / "converted" / "GLEAM_v4.2a.yaml"
    convert_old_reference(old_path, out_path, name="GLEAM_v4.2a", category="Water")

    assert out_path.exists()
    data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert data["name"] == "GLEAM_v4.2a"
    assert data["category"] == "Water"
    assert data["data_type"] == "grid"
    assert data["grid_res"] == 0.25
    assert data["years"] == [1980, 2023]
    assert data["tim_res"] == "Month"
    assert data["data_groupby"] == "Year"
    assert data["variables"]["Evapotranspiration"] == {
        "varname": "E",
        "varunit": "mm day-1",
        "prefix": "E_",
        "suffix": "_GLEAM",
        "sub_dir": "Water/Evapotranspiration/GLEAM",
    }
