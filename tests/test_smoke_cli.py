"""Tests for the bundled OpenBench smoke-test command."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from importlib.resources import files
from pathlib import Path

import pandas as pd
import yaml
from click.testing import CliRunner

from openbench.cli.main import cli


def test_smoke_test_help_is_registered():
    runner = CliRunner()

    result = runner.invoke(cli, ["smoke-test", "--help"])

    assert result.exit_code == 0
    assert "Run the bundled Initial_test smoke fixture" in result.output
    assert "--run" in result.output
    assert "--keep" in result.output


def test_smoke_fixture_resources_are_packaged():
    smoke_dir = files("openbench") / "dataset" / "smoke"

    assert (smoke_dir / "initial_test.tar.gz").is_file()
    assert (smoke_dir / "openbench-smoke.yaml").is_file()
    with tarfile.open(smoke_dir / "initial_test.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
    assert "Initial_test/Reference/Initial_test/GLEAM4.2a_monthly/E_2004_GLEAM_v4.2a_MO.nc" in names
    assert "Initial_test/Reference/Initial_test/GLEAM_hybrid_PLUMBER2/AU-How_2004-2005.nc" in names
    assert "Initial_test/Reference/Initial_test/PLUMBER2/AU-How_2004-2005_OzFlux_Flux.nc" in names
    assert "Initial_test/Simulation/Initial_test/grid/grid_case_hist_2004-01.nc" in names
    assert "Initial_test/Simulation/Initial_test/stn/sim_AU-How_2004-2005.nc" in names
    assert "Initial_test/nml/nml-json/main-Initial_test.json" in names


def test_smoke_test_extracts_fixture_and_invokes_check(tmp_path, monkeypatch):
    from openbench.cli import smoke as smoke_module

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(smoke_module.subprocess, "run", fake_run)

    work_dir = tmp_path / "work"
    runner = CliRunner()
    result = runner.invoke(cli, ["smoke-test", "--work-dir", str(work_dir), "--keep"])

    assert result.exit_code == 0, result.output
    assert calls

    cmd, kwargs = calls[0]
    assert cmd[:3] == [sys.executable, "-m", "openbench"]
    assert cmd[3] == "check"
    assert Path(cmd[4]).is_file()
    assert kwargs["check"] is False
    assert kwargs["env"]["HOME"] == str(work_dir / "home")

    sample_root = work_dir / "Initial_test"
    assert (sample_root / "Reference" / "Initial_test" / "GLEAM4.2a_monthly").is_dir()
    assert (sample_root / "Simulation" / "Initial_test" / "grid").is_dir()
    assert (sample_root / "Simulation" / "Initial_test" / "stn").is_dir()
    assert (sample_root / "nml" / "nml-json" / "main-Initial_test.json").is_file()

    config = yaml.safe_load(Path(cmd[4]).read_text())
    assert config["evaluation"]["variables"] == [
        "Evapotranspiration",
        "Latent_Heat",
        "Sensible_Heat",
    ]
    assert config["reference"] == {
        "data_root": str(sample_root / "Reference" / "Initial_test"),
        "Evapotranspiration": ["OpenBench_Smoke_GLEAM4_2a", "OpenBench_Smoke_GLEAM_hybrid_PLUMBER2"],
        "Latent_Heat": ["OpenBench_Smoke_ILAMB_monthly", "OpenBench_Smoke_PLUMBER2"],
        "Sensible_Heat": ["OpenBench_Smoke_ILAMB_monthly", "OpenBench_Smoke_PLUMBER2"],
    }
    assert set(config["simulation"]) == {"grid_case", "station_case"}
    assert config["simulation"]["station_case"]["fulllist"] == str(work_dir / "lists" / "station_case.csv")

    catalog = yaml.safe_load((work_dir / "home" / ".openbench" / "references" / "reference_catalog.yaml").read_text())
    assert catalog["OpenBench_Smoke_GLEAM4_2a"]["root_dir"] == str(sample_root / "Reference" / "Initial_test")
    assert set(catalog) == {
        "OpenBench_Smoke_GLEAM4_2a",
        "OpenBench_Smoke_GLEAM_hybrid_PLUMBER2",
        "OpenBench_Smoke_ILAMB_monthly",
        "OpenBench_Smoke_PLUMBER2",
    }
    assert catalog["OpenBench_Smoke_GLEAM_hybrid_PLUMBER2"]["fulllist"] == str(
        work_dir / "lists" / "GLEAM_hybrid_PLUMBER2.csv"
    )
    assert catalog["OpenBench_Smoke_PLUMBER2"]["fulllist"] == str(work_dir / "lists" / "PLUMBER2.csv")

    station_list = pd.read_csv(work_dir / "lists" / "station_case.csv")
    ref_list = pd.read_csv(work_dir / "lists" / "PLUMBER2.csv")
    assert station_list["DIR"].map(lambda value: Path(value).is_absolute()).all()
    assert ref_list["DIR"].map(lambda value: Path(value).is_absolute()).all()


def test_smoke_config_enables_total_score_heatmap(tmp_path, monkeypatch):
    from openbench.cli import smoke as smoke_module

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(smoke_module.subprocess, "run", fake_run)

    work_dir = tmp_path / "work"
    result = CliRunner().invoke(cli, ["smoke-test", "--work-dir", str(work_dir), "--keep"])

    assert result.exit_code == 0, result.output
    config = yaml.safe_load((work_dir / "openbench-smoke.yaml").read_text())
    assert config["comparison"] == {"enabled": True, "items": ["HeatMap"]}


def test_smoke_run_reports_required_total_score_artifacts(tmp_path, monkeypatch):
    from openbench.cli import smoke as smoke_module

    def fake_run(cmd, **kwargs):
        config_path = Path(cmd[4])
        output_dir = Path(yaml.safe_load(config_path.read_text())["project"]["output_dir"])
        case_dir = output_dir / "openbench_initial_test_smoke"
        (case_dir / "comparisons" / "HeatMap").mkdir(parents=True)
        (case_dir / "scores").mkdir()
        (case_dir / "comparisons" / "HeatMap" / "scenarios_Overall_Score_comparison_heatmap.jpg").write_bytes(b"jpg")
        (case_dir / "comparisons" / "HeatMap" / "scenarios_Overall_Score_comparison.csv").write_text("x\n")
        (case_dir / "scores" / "Evapotranspiration__ref__Ref__sim__Sim__Overall_Score.jpg").write_bytes(b"jpg")
        (case_dir / "scores" / "Evapotranspiration__stn__Ref__Sim__Overall_Score.jpg").write_bytes(b"jpg")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(smoke_module.subprocess, "run", fake_run)

    work_dir = tmp_path / "work"
    result = CliRunner().invoke(cli, ["smoke-test", "--work-dir", str(work_dir), "--run", "--keep"])

    assert result.exit_code == 0, result.output
    assert "Smoke result artifacts:" in result.output
    assert "Total score heatmap:" in result.output
    assert "scenarios_Overall_Score_comparison_heatmap.jpg" in result.output
    assert "Overall score spatial maps:" in result.output
    assert "Evapotranspiration__ref__Ref__sim__Sim__Overall_Score.jpg" in result.output
    assert "Evapotranspiration__stn__Ref__Sim__Overall_Score.jpg" in result.output
