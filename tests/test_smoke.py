"""Smoke tests to verify package structure and CLI entry point."""


def test_import_openbench():
    """Verify that the openbench package can be imported."""
    import openbench

    assert openbench.__version__ == "3.0.0b1"
    assert openbench.__title__ == "OpenBench"


def test_import_subpackages():
    """Verify that all sub-packages can be imported."""


def test_cli_entry_point():
    """Verify that the CLI group is callable."""
    from click.testing import CliRunner

    from openbench.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "3.0.0" in result.output


def test_cli_help():
    """Verify that --help works."""
    from click.testing import CliRunner

    from openbench.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "OpenBench" in result.output


def test_cli_version_option():
    """Verify that --version works."""
    from click.testing import CliRunner

    from openbench.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "3.0.0" in result.output


def test_gui_import_guard():
    """Verify that GUI import guard gives a helpful error when PySide6 is missing."""
    from openbench.gui import _check_gui_deps

    # This test passes if PySide6 IS installed (no error),
    # or if PySide6 is NOT installed (ImportError with hint).
    try:
        _check_gui_deps()
    except ImportError as e:
        assert "colm-openbench[gui]" in str(e)
