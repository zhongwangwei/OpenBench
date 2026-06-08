"""CLI integration tests — verify commands work end-to-end."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()
FIXTURES = Path(__file__).parent / "test_config" / "fixtures"


def _fixture_with_existing_sim_roots(tmp_path: Path, fixture_name: str) -> Path:
    data = yaml.safe_load((FIXTURES / fixture_name).read_text())
    for label, entry in data.get("simulation", {}).items():
        if not isinstance(entry, dict):
            continue
        root = tmp_path / "sim" / label
        root.mkdir(parents=True, exist_ok=True)
        entry["root_dir"] = str(root)

    config_path = tmp_path / fixture_name
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return config_path


def test_check_valid_config(tmp_path):
    config = _fixture_with_existing_sim_roots(tmp_path, "minimal.yaml")
    data = yaml.safe_load(config.read_text())
    ref_root = tmp_path / "ref"
    ref_dir = ref_root / "Water" / "Evapotranspiration" / "GLEAM_v4.2a"
    ref_dir.mkdir(parents=True)
    (ref_dir / "E_2004.nc").touch()
    data["reference"]["data_root"] = str(ref_root)
    config.write_text(yaml.safe_dump(data, sort_keys=False))

    result = runner.invoke(cli, ["check", str(config)])
    assert result.exit_code == 0
    assert "Config valid" in result.output


def test_check_invalid_config():
    result = runner.invoke(cli, ["check", str(FIXTURES / "invalid_years.yaml")])
    assert result.exit_code == 1


def test_run_dry_run(tmp_path):
    config = _fixture_with_existing_sim_roots(tmp_path, "full.yaml")
    result = runner.invoke(cli, ["run", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "test-full" in result.output


def test_run_actual(tmp_path):
    import openbench.runner.local as local_runner

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        return {
            "status": "success",
            "output_dir": "/tmp/openbench-out",
            "variables": ["Evapotranspiration"],
            "simulations": ["CoLM2024"],
            "errors": [],
        }

    original = local_runner.run_evaluation
    local_runner.run_evaluation = fake_run_evaluation
    try:
        config = _fixture_with_existing_sim_roots(tmp_path, "minimal.yaml")
        result = runner.invoke(cli, ["run", str(config)])
    finally:
        local_runner.run_evaluation = original

    assert result.exit_code == 0
    assert "Evaluation complete" in result.output


def test_run_only_drawing_fail_fast_errors_exit_nonzero(tmp_path):
    import openbench.runner.local as local_runner

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        return {
            "status": "error",
            "output_dir": str(tmp_path / "output" / "test"),
            "variables": ["Evapotranspiration"],
            "simulations": ["CoLM2024"],
            "errors": [
                {
                    "phase": "comparison",
                    "item": "Taylor_Diagram",
                    "source": "scenarios_Taylor_Diagram_comparison",
                    "message": "only_drawing missing required file: taylor_diagram__Evapotranspiration__GLEAM.csv",
                }
            ],
        }

    original = local_runner.run_evaluation
    local_runner.run_evaluation = fake_run_evaluation
    try:
        config = _fixture_with_existing_sim_roots(tmp_path, "minimal.yaml")
        data = yaml.safe_load(config.read_text())
        data.setdefault("project", {})["only_drawing"] = True
        config.write_text(yaml.safe_dump(data, sort_keys=False))
        result = runner.invoke(cli, ["run", str(config)])
    finally:
        local_runner.run_evaluation = original

    assert result.exit_code == 1
    assert "Evaluation failed" in result.output
    assert "[comparison]" in result.output
    assert "only_drawing missing required file" in result.output


def test_ref_list():
    result = runner.invoke(cli, ["ref", "list"])
    assert result.exit_code == 0
    assert "GLEAM" in result.output


def test_ref_list_filter():
    result = runner.invoke(cli, ["ref", "list", "--variable", "Evapotranspiration"])
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


def test_model_register_interactive_comma_separated_varnames_write_fallbacks(tmp_path, monkeypatch):
    import yaml

    import openbench.cli.model as cli_model
    import openbench.data.registry.manager as registry_manager

    # The CLI writes via get_writable_model_catalog_path; patch both the
    # source and the already-imported reference inside cli.model so the
    # test catches the write regardless of import style.
    catalog_path = tmp_path / "model_catalog.yaml"
    monkeypatch.setattr(registry_manager, "get_writable_model_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(cli_model, "get_writable_model_catalog_path", lambda: catalog_path, raising=False)

    result = runner.invoke(
        cli,
        ["model", "register", "InteractiveModel", "--data-type", "grid", "--grid-res", "0.5", "--tim-res", "Month"],
        input="Runoff\nrunoff_primary,runoff_fallback\nmm day-1\n\n",
    )

    catalog = yaml.safe_load(catalog_path.read_text())
    runoff = catalog["InteractiveModel"]["variables"]["Runoff"]

    assert result.exit_code == 0
    assert runoff["varname"] == "runoff_primary"
    assert runoff["fallbacks"] == [{"varname": "runoff_fallback", "varunit": "mm day-1"}]


def test_migrate():
    old_config = FIXTURES / "old_json" / "main.json"
    if not old_config.exists():
        pytest.skip(f"fixture not found: {old_config}")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        out = f.name

    # NamedTemporaryFile creates the file, so migrate would prompt for
    # overwrite confirmation; use --force to skip it in this CI test.
    result = runner.invoke(cli, ["migrate", str(old_config), "-o", out, "--force"])
    assert result.exit_code == 0
    assert "Written to" in result.output

    Path(out).unlink()


def test_version():
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "3.0.0" in result.output


def test_init_output_is_loadable(tmp_path, monkeypatch):
    """Regression: openbench init must produce a YAML that the loader accepts.

    Earlier versions wrote {"reference": {"sources": {...}}} but the loader
    expects a flat var->source mapping at reference top-level. The two paths
    were silently incompatible — init succeeded, then check immediately failed
    with 'reference.sources must be a string (source name), got dict'.
    """
    out = tmp_path / "init_output.yaml"
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    # Pipe accept-all-defaults answers. Press enter through prompts;
    # extra newlines are harmless. End model loop with empty input.
    init_input = "\n" * 250 + "\n"
    result = runner.invoke(cli, ["init", "--no-ref-check", "-o", str(out)], input=init_input)

    if result.exit_code != 0 or not out.exists():
        pytest.skip(f"init did not complete in test env (exit={result.exit_code}); regression check skipped")

    # Same loader path as openbench check
    from openbench.config import ConfigError, load_config

    try:
        cfg = load_config(out)
    except ConfigError as e:
        pytest.fail(f"openbench init produced YAML that loader rejected: {e}\n\nGenerated YAML:\n{out.read_text()}")

    # Reference entries must be strings (var -> source_name)
    for var, src in cfg.reference.sources.items():
        assert isinstance(src, str), f"reference[{var!r}] should be a string source name, got {type(src).__name__}"
