"""Smoke tests to verify package structure and CLI entry point."""

from pathlib import Path


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


def test_smoke_test_refuses_non_empty_work_dir(tmp_path):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    work_dir = tmp_path / "smoke-work"
    work_dir.mkdir()
    sentinel = work_dir / "keep.txt"
    sentinel.write_text("do not delete")

    result = CliRunner().invoke(cli, ["smoke-test", "--work-dir", str(work_dir)])

    assert result.exit_code != 0
    assert "already exists and is not empty" in result.output
    assert sentinel.read_text() == "do not delete"


def test_smoke_test_allows_empty_existing_work_dir(monkeypatch, tmp_path):
    import openbench.cli.smoke as smoke_module

    def fake_prepare(work_dir: Path):
        sample_root = work_dir / "Initial_test"
        sample_root.mkdir(parents=True)
        home = work_dir / "home"
        home.mkdir()
        config = work_dir / "smoke.yaml"
        config.write_text("project: {}\n")
        return sample_root, home, config

    monkeypatch.setattr(smoke_module, "_prepare_work_dir", fake_prepare)
    monkeypatch.setattr(smoke_module, "_run_openbench_subcommand", lambda *args, **kwargs: 0)

    from click.testing import CliRunner

    from openbench.cli.main import cli

    work_dir = tmp_path / "empty-work"
    work_dir.mkdir()

    result = CliRunner().invoke(cli, ["smoke-test", "--work-dir", str(work_dir)])

    assert result.exit_code == 0
    assert (work_dir / "Initial_test").is_dir()
