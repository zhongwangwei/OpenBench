"""Tests for reference definition converter."""

import tempfile
from pathlib import Path

from openbench.data.registry.converter import convert_old_reference

OLD_REF_DIR = Path("OpenBench-wei/nml/nml-yaml/Ref_variables_definition_LowRes")


def test_convert_gleam():
    """Convert old GLEAM definition to new format."""
    old_path = OLD_REF_DIR / "GLEAM_v4.2a.yaml"
    if not old_path.exists():
        return  # Skip if old code not present

    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "GLEAM_v4.2a.yaml"
        convert_old_reference(old_path, out_path, name="GLEAM_v4.2a", category="Water")

        assert out_path.exists()

        with open(out_path) as f:
            data = yaml.safe_load(f)

        assert data["name"] == "GLEAM_v4.2a"
        assert data["data_type"] == "grid"
        assert "variables" in data
        assert "Evapotranspiration" in data["variables"]
        assert data["variables"]["Evapotranspiration"]["varname"] == "E"
