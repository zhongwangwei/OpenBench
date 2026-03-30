"""Tests for config migration tool."""

import tempfile
from pathlib import Path

import pytest
import yaml

from openbench.config.migration import migrate_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_migrate_json_config():
    """Migrate old JSON config set to new YAML format."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        output_path = Path(f.name)

    result = migrate_config(FIXTURES / "old_json" / "main.json", output_path)

    assert output_path.exists()
    with open(output_path) as f:
        data = yaml.safe_load(f)

    # Check project section
    assert data["project"]["name"] == "test-migrate"
    assert data["project"]["output_dir"] == "./output"
    assert data["project"]["years"] == [2004, 2010]

    # Check evaluation - only true items
    assert "Evapotranspiration" in data["evaluation"]["variables"]
    assert "Latent_Heat" in data["evaluation"]["variables"]
    assert "GPP" not in data["evaluation"]["variables"]

    # Check reference
    assert "Evapotranspiration" in data["reference"]

    # Check simulation
    assert "CoLM2024" in data["simulation"]
    assert data["simulation"]["CoLM2024"]["root_dir"] == "/data/CoLM2024/output"

    # Check metrics - only true ones
    assert "bias" in data["metrics"]
    assert "RMSE" in data["metrics"]
    assert "percent_bias" not in data["metrics"]

    # Check scores - only true ones
    assert "nBiasScore" in data["scores"]
    assert "Overall_Score" not in data["scores"]

    # Check comparison
    assert data["comparison"]["enabled"] is True

    # Check options
    assert data["options"]["num_cores"] == 8

    # Validate the output can be loaded by the new loader
    from openbench.config.loader import load_config

    cfg = load_config(output_path)
    assert cfg.project.name == "test-migrate"

    output_path.unlink()


def test_migrate_returns_summary():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        output_path = Path(f.name)

    result = migrate_config(FIXTURES / "old_json" / "main.json", output_path)

    assert "files_read" in result
    assert result["files_read"] >= 1
    assert "variables" in result
    assert "simulations" in result

    output_path.unlink()


def test_migrate_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        migrate_config(Path("/no/such/file.json"), Path("/tmp/out.yaml"))
