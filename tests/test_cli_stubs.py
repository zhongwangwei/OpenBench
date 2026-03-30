"""Test that all CLI commands are registered and show help."""

import click
from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()


def test_run_help():
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run evaluation" in result.output


def test_check_help():
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0
    assert "Validate" in result.output


def test_data_list_help():
    result = runner.invoke(cli, ["data", "list", "--help"])
    assert result.exit_code == 0


def test_data_download_help():
    result = runner.invoke(cli, ["data", "download", "--help"])
    assert result.exit_code == 0


def test_data_status_help():
    result = runner.invoke(cli, ["data", "status", "--help"])
    assert result.exit_code == 0


def test_data_path_help():
    result = runner.invoke(cli, ["data", "path", "--help"])
    assert result.exit_code == 0


def test_data_optimize_help():
    result = runner.invoke(cli, ["data", "optimize", "--help"])
    assert result.exit_code == 0


def test_model_list_help():
    result = runner.invoke(cli, ["model", "list", "--help"])
    assert result.exit_code == 0


def test_model_show_help():
    result = runner.invoke(cli, ["model", "show", "--help"])
    assert result.exit_code == 0


def test_model_register_help():
    result = runner.invoke(cli, ["model", "register", "--help"])
    assert result.exit_code == 0


def test_model_remove_var_help():
    result = runner.invoke(cli, ["model", "remove-var", "--help"])
    assert result.exit_code == 0


def test_migrate_help():
    result = runner.invoke(cli, ["migrate", "--help"])
    assert result.exit_code == 0


def test_init_help():
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0


def test_gui_help():
    result = runner.invoke(cli, ["gui", "--help"])
    assert result.exit_code == 0


def test_all_commands_registered():
    """Verify all expected commands are accessible via the CLI group."""
    # Use list_commands() for LazyGroup compatibility
    ctx = click.Context(cli)
    command_names = set(cli.list_commands(ctx))
    expected = {"run", "check", "data", "model", "migrate", "init", "gui", "version"}
    assert expected == command_names, f"Missing: {expected - command_names}, Extra: {command_names - expected}"
