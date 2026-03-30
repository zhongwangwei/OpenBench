"""CLI integration tests — verify commands work end-to-end."""

from pathlib import Path

from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()
FIXTURES = Path(__file__).parent / "test_config" / "fixtures"


def test_check_valid_config():
    result = runner.invoke(cli, ["check", str(FIXTURES / "minimal.yaml")])
    assert result.exit_code == 0
    assert "Config valid" in result.output


def test_check_invalid_config():
    result = runner.invoke(cli, ["check", str(FIXTURES / "invalid_years.yaml")])
    assert result.exit_code == 1


def test_run_dry_run():
    result = runner.invoke(cli, ["run", str(FIXTURES / "full.yaml"), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "test-full" in result.output


def test_run_actual():
    result = runner.invoke(cli, ["run", str(FIXTURES / "minimal.yaml")])
    assert result.exit_code == 0
    assert "Evaluation complete" in result.output


def test_data_list():
    result = runner.invoke(cli, ["data", "list"])
    assert result.exit_code == 0
    assert "GLEAM" in result.output


def test_data_list_filter():
    result = runner.invoke(cli, ["data", "list", "--variable", "Evapotranspiration"])
    assert result.exit_code == 0
    assert "GLEAM" in result.output


def test_model_list():
    result = runner.invoke(cli, ["model", "list"])
    assert result.exit_code == 0
    assert "CoLM2024" in result.output


def test_model_show():
    result = runner.invoke(cli, ["model", "show", "CoLM2024"])
    assert result.exit_code == 0
    assert "f_fevpa" in result.output


def test_model_show_not_found():
    result = runner.invoke(cli, ["model", "show", "NonExistent"])
    assert result.exit_code == 1


def test_migrate():
    old_config = FIXTURES / "old_json" / "main.json"
    if not old_config.exists():
        return

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        out = f.name

    result = runner.invoke(cli, ["migrate", str(old_config), "-o", out])
    assert result.exit_code == 0
    assert "Written to" in result.output

    Path(out).unlink()


def test_version():
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "3.0.0a1" in result.output
