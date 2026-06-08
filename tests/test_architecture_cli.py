"""Architecture cleanup regression checks."""

from __future__ import annotations

import pathlib

from click.testing import CliRunner

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cli_uses_ref_not_legacy_data_command():
    from openbench.cli.main import cli

    runner = CliRunner()
    # `ref` is the single documented name for reference-dataset management.
    # The legacy `data` namespace was renamed and must not be re-registered.
    assert runner.invoke(cli, ["ref", "--help"]).exit_code == 0
    assert runner.invoke(cli, ["data", "--help"]).exit_code != 0


def test_cli_profile_rescue_logic_lives_outside_data_god_module():
    data_source = (ROOT / "src/openbench/cli/data.py").read_text(encoding="utf-8")
    rescue_source = (ROOT / "src/openbench/cli/_profile_rescue.py").read_text(encoding="utf-8")

    assert "_profile_rescue" in data_source
    assert "def _prompt_grid_composite_profile(" in rescue_source
    assert "def _write_reference_profiles(" in rescue_source
    assert "_prompt_grid_composite_profile = _profile_rescue._prompt_grid_composite_profile" in data_source
    assert "Create reference profile for {skip_path}" not in data_source


def test_cli_ref_scan_logic_lives_outside_data_god_module():
    data_source = (ROOT / "src/openbench/cli/data.py").read_text(encoding="utf-8")
    scan_source = (ROOT / "src/openbench/cli/_scan.py").read_text(encoding="utf-8")

    assert "_scan" in data_source
    assert "def run_scan(" in scan_source
    assert "register_scanned_datasets_batch(" in scan_source
    assert "register_scanned_datasets_batch(" not in data_source
    assert "def scan(" in data_source


def test_cli_ref_register_logic_lives_outside_data_god_module():
    data_source = (ROOT / "src/openbench/cli/data.py").read_text(encoding="utf-8")
    register_source = (ROOT / "src/openbench/cli/_register.py").read_text(encoding="utf-8")

    assert "_register" in data_source
    assert "def register_reference(" in register_source
    assert "def register_reference_profile(" in register_source
    assert "_catalog_write_lock" in register_source
    assert "parse_fallbacks(" in register_source
    assert "parse_fallbacks(" not in data_source
    assert "def register(" in data_source


def test_cli_ref_simple_commands_live_outside_data_god_module():
    data_source = (ROOT / "src/openbench/cli/data.py").read_text(encoding="utf-8")
    ref_source = (ROOT / "src/openbench/cli/_ref_commands.py").read_text(encoding="utf-8")
    scan_support_source = (ROOT / "src/openbench/cli/_scan_support.py").read_text(encoding="utf-8")

    assert "_ref_commands" in data_source
    assert "_scan_support" in data_source
    for name in ("list_datasets", "delete_reference", "generate_station_list"):
        assert f"def {name}(" in ref_source
    assert "def print_scan_skip_report(" in scan_support_source
    assert "def _print_scan_skip_report(" not in data_source
    assert "def generate_station_list(" in data_source  # Click wrapper remains
    assert "from openbench.data.registry.scanner import generate_station_list" not in data_source


def test_cli_ref_display_and_optimize_logic_lives_outside_data_god_module():
    data_source = (ROOT / "src/openbench/cli/data.py").read_text(encoding="utf-8")
    display_source = (ROOT / "src/openbench/cli/_display.py").read_text(encoding="utf-8")
    optimize_source = (ROOT / "src/openbench/cli/_optimize.py").read_text(encoding="utf-8")

    assert "_display" in data_source
    assert "_optimize" in data_source
    assert "def show_reference(" in display_source
    assert "def optimize_reference(" in optimize_source
    assert "Base-name references are resolved" in display_source
    assert "write_mfdataset_zarr" in optimize_source
    assert "write_mfdataset_zarr" not in data_source
