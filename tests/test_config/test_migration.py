"""Tests for config migration tool."""

import tempfile
from pathlib import Path

import pytest
import tomllib
import yaml
from click.testing import CliRunner

from openbench.cli.main import cli
from openbench.config.migration import migrate_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_migrate_json_config():
    """Migrate old JSON config set to new YAML format."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        output_path = Path(f.name)

    migrate_config(FIXTURES / "old_json" / "main.json", output_path)

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

    # Runtime options are flattened into project in migrated configs.
    assert data["project"]["num_cores"] == 8

    # Validate the output can be loaded by the new loader
    from openbench.config.loader import load_config

    cfg = load_config(output_path)
    assert cfg.project.name == "test-migrate"

    output_path.unlink()


def test_migrated_yaml_has_no_deprecated_sections():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        output_path = Path(f.name)

    migrate_config(FIXTURES / "old_json" / "main.json", output_path)

    with open(output_path) as f:
        data = yaml.safe_load(f)

    assert "options" not in data
    assert data["project"]["num_cores"] == 8

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


def test_migration_preserves_zero_and_lowercase_nml_options(tmp_path: Path):
    main = tmp_path / "main.yaml"
    output = tmp_path / "openbench.yaml"
    main.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "basename": "migrated",
                    "basedir": str(tmp_path / "out"),
                    "syear": 2000,
                    "eyear": 2001,
                    "num_cores": 0,
                    "min_year": 0,
                    "igbp_groupby": True,
                    "pft_groupby": True,
                    "climate_zone_groupby": True,
                    "compare_tim_res": "8day",
                },
                "evaluation_items": {"GPP": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    migrate_config(main, output)

    migrated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert migrated["project"]["num_cores"] == 0
    assert migrated["project"]["min_year_threshold"] == 0
    assert migrated["project"]["IGBP_groupby"] is True
    assert migrated["project"]["PFT_groupby"] is True
    assert migrated["project"]["climate_zone_groupby"] is True
    assert migrated["project"]["tim_res"] == "8Day"


def test_repeated_migrate_on_modern_yaml_is_idempotent(tmp_path: Path):
    first = tmp_path / "openbench.yaml"
    second = tmp_path / "openbench-again.yaml"
    data = {
        "project": {"name": "modern", "output_dir": str(tmp_path / "out"), "years": [2000, 2001]},
        "evaluation": {"variables": ["Runoff"]},
        "reference": {"Runoff": "DemoRef"},
        "simulation": {
            "SimA": {
                "model": "InlineModel",
                "root_dir": str(tmp_path / "sim"),
                "variables": {"Runoff": {"varname": "runoff"}},
            }
        },
        "metrics": ["bias"],
    }
    first.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = migrate_config(first, second)

    assert result["already_modern"] is True
    assert yaml.safe_load(second.read_text(encoding="utf-8")) == data


def test_f90nml_is_declared_for_namelist_migration_extra():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert any(requirement.startswith("f90nml") for requirement in optional["migration"])
    assert "migration" in optional["all"][0]


def test_migrated_config_validation_and_run_dry_run_agree(tmp_path: Path, monkeypatch):
    ref_root = tmp_path / "ref"
    sim_root = tmp_path / "sim"
    ref_root.mkdir()
    sim_root.mkdir()
    import numpy as np
    import xarray as xr

    xr.Dataset({"runoff": ("time", np.array([1.0]))}).to_netcdf(ref_root / "runoff_2000.nc")

    old = tmp_path / "old"
    old.mkdir()
    ref_def = old / "ref_def.yaml"
    ref_def.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "root_dir": str(ref_root),
                    "data_type": "grid",
                    "tim_res": "Month",
                    "grid_res": 0.5,
                    "data_groupby": "Year",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ref_nml = old / "ref.yaml"
    ref_nml.write_text(
        yaml.safe_dump(
            {
                "general": {"Runoff_ref_source": "DemoRef"},
                "def_nml": {"DemoRef": str(ref_def)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sim_def = old / "sim_def.yaml"
    sim_def.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "root_dir": str(sim_root),
                    "model_namelist": "InlineModel.nml",
                    "data_type": "grid",
                    "tim_res": "Month",
                    "grid_res": 0.5,
                    "data_groupby": "Year",
                },
                "Runoff": {"varname": "runoff", "varunit": "mm day-1"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sim_nml = old / "sim.yaml"
    sim_nml.write_text(
        yaml.safe_dump(
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "def_nml": {"SimA": str(sim_def)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main = old / "main.yaml"
    main.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "basename": "migrated",
                    "basedir": str(tmp_path / "out"),
                    "syear": 2000,
                    "eyear": 2001,
                    "reference_nml": str(ref_nml),
                    "simulation_nml": str(sim_nml),
                    "comparison": False,
                    "statistics": False,
                },
                "evaluation_items": {"Runoff": True},
                "metrics": {"bias": True},
                "scores": {"Overall_Score": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    migrated = tmp_path / "migrated.yaml"
    migrate_config(main, migrated)

    from types import SimpleNamespace

    from openbench.data.registry import manager as mgr_mod

    class Registry:
        def get_resolution_variants(self, name):
            return {}

        def get_reference(self, name, **kwargs):
            return SimpleNamespace(
                name=name,
                data_type="grid",
                tim_res="Month",
                grid_res=0.5,
                data_groupby="Year",
                timezone=0,
                years=[2000, 2001],
                root_dir=str(ref_root),
                variables={
                    "Runoff": SimpleNamespace(
                        varname="runoff",
                        varunit="mm day-1",
                        sub_dir="",
                        prefix="",
                        suffix="",
                        fulllist=None,
                    )
                },
                _provenance={},
            )

        def get_model(self, name):
            return None

        def list_models(self):
            return []

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: Registry())
    runner = CliRunner()

    check_result = runner.invoke(cli, ["check", str(migrated)])
    dry_run_result = runner.invoke(cli, ["run", "--dry-run", str(migrated)])

    assert check_result.exit_code == 0, check_result.output
    assert dry_run_result.exit_code == 0, dry_run_result.output


def test_migrate_treats_null_enable_sections_as_empty(tmp_path):
    main = tmp_path / "main.yaml"
    out = tmp_path / "out.yaml"
    main.write_text(
        "general:\n"
        "  basename: null-sections\n"
        "  basedir: ./out\n"
        "  syear: 2000\n"
        "  eyear: 2001\n"
        "evaluation_items: null\n"
        "metrics: null\n"
        "scores: null\n"
        "comparisons: null\n",
        encoding="utf-8",
    )

    migrate_config(main, out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["evaluation"]["variables"] == []
    assert "metrics" not in data
    assert "scores" not in data
    assert "comparison" not in data


def test_migrate_preserves_variable_overrides_for_known_model(tmp_path, monkeypatch):
    """Known model profiles must not swallow legacy per-variable overrides."""
    from types import SimpleNamespace

    from openbench.data.registry import manager as mgr_mod

    old = tmp_path / "old"
    old.mkdir()
    sim_def = old / "sim_def.yaml"
    sim_def.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "model_namelist": "CoLM2024.nml",
                    "root_dir": "/sim/root",
                    "data_type": "grid",
                    "tim_res": "Month",
                },
                "Runoff": {
                    "varname": "custom_runoff",
                    "varunit": "mm day-1",
                    "prefix": "custom_",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sim_nml = old / "sim.yaml"
    sim_nml.write_text(
        yaml.safe_dump(
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "def_nml": {"SimA": str(sim_def)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main = old / "main.yaml"
    main.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "basename": "migrated",
                    "basedir": str(tmp_path / "out"),
                    "syear": 2000,
                    "eyear": 2001,
                    "simulation_nml": str(sim_nml),
                },
                "evaluation_items": {"Runoff": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class Registry:
        def get_model(self, name):
            return SimpleNamespace(name=name) if name == "CoLM2024" else None

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: Registry())
    out = tmp_path / "migrated.yaml"

    migrate_config(main, out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["simulation"]["SimA"]["variables"]["Runoff"] == {
        "varname": "custom_runoff",
        "varunit": "mm day-1",
        "prefix": "custom_",
    }


def test_migrate_preserves_falsey_and_path_simulation_fields(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    sim_def = old / "sim_def.yaml"
    sim_def.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "model_namelist": "UnknownModel.nml",
                    "root_dir": "/sim/root",
                    "grid_res": 0,
                    "prefix": "",
                    "suffix": "_daily.nc",
                    "fulllist": "stations.csv",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sim_nml = old / "sim.yaml"
    sim_nml.write_text(
        yaml.safe_dump(
            {
                "general": {"Runoff_sim_source": ["SimA"]},
                "def_nml": {"SimA": str(sim_def)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main = old / "main.yaml"
    main.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "basename": "migrated",
                    "basedir": str(tmp_path / "out"),
                    "syear": 2000,
                    "eyear": 2001,
                    "simulation_nml": str(sim_nml),
                },
                "evaluation_items": {"Runoff": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = tmp_path / "migrated.yaml"

    migrate_config(main, out)
    entry = yaml.safe_load(out.read_text(encoding="utf-8"))["simulation"]["SimA"]

    assert entry["grid_res"] == 0
    assert entry["prefix"] == ""
    assert entry["suffix"] == "_daily.nc"
    assert entry["fulllist"] == "stations.csv"


def test_migrate_normalizes_simulation_level_tim_res(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    sim_def = old / "sim_def.yaml"
    sim_def.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "model_namelist": "UnknownModel.nml",
                    "root_dir": "/sim/root",
                    "tim_res": "monthly",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sim_nml = old / "sim.yaml"
    sim_nml.write_text(
        yaml.safe_dump(
            {"general": {"Runoff_sim_source": ["SimA"]}, "def_nml": {"SimA": str(sim_def)}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main = old / "main.yaml"
    main.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "basename": "migrated",
                    "basedir": str(tmp_path / "out"),
                    "syear": 2000,
                    "eyear": 2001,
                    "simulation_nml": str(sim_nml),
                },
                "evaluation_items": {"Runoff": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = tmp_path / "migrated.yaml"
    migrate_config(main, out)

    assert yaml.safe_load(out.read_text(encoding="utf-8"))["simulation"]["SimA"]["tim_res"] == "Month"
