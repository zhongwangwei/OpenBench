"""CLI tests for openbench sim scan."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()


def _write_case_nc(
    root: Path,
    label: str = "CaseA",
    var_name: str = "QFLX_EVAP_TOT",
    filename: str | None = None,
) -> Path:
    history = root / label / "history"
    history.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    ds = xr.Dataset(
        {var_name: (["time", "lat", "lon"], np.zeros((12, 3, 4)))},
        coords={"time": times, "lat": [10.0, 10.5, 11.0], "lon": [100.0, 100.5, 101.0, 101.5]},
    )
    ds[var_name].attrs["units"] = "mm day-1"
    ds.to_netcdf(history / (filename or f"{var_name}_2000.nc"))
    return history


def _write_station_case_nc(
    root: Path,
    *,
    label: str = "CaseA",
    site_id: str = "US-AAA",
    lon: float = -100.0,
    lat: float = 40.0,
    year: int = 2000,
    var_name: str = "QFLX_EVAP_TOT",
) -> Path:
    history = root / label / site_id / "history"
    history.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(f"{year}-01-01", periods=2, freq="MS")
    ds = xr.Dataset(
        {var_name: (["time"], np.zeros(2))},
        coords={"time": times, "lon": lon, "lat": lat},
        attrs={"station_id": site_id},
    )
    ds[var_name].attrs["units"] = "mm day-1"
    ds.to_netcdf(history / f"{site_id}_{year}.nc")
    return root / label


def test_sim_scan_auto_writes_timestamped_simulation_yaml_and_report(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    history = _write_case_nc(root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-143012", raising=False)

    result = runner.invoke(cli, ["sim", "scan", str(root), "--model", "CoLM2024", "--auto"])

    assert result.exit_code == 0
    sim_path = tmp_path / "openbench_sim_scan_20260501-143012.yaml"
    report_path = tmp_path / "openbench_sim_scan_report_20260501-143012.yaml"
    assert sim_path.exists()
    assert report_path.exists()

    sim_data = yaml.safe_load(sim_path.read_text())
    assert sim_data["simulation"]["CaseA"]["model"] == "CoLM2024"
    assert sim_data["simulation"]["CaseA"]["root_dir"] == str(history)
    assert sim_data["simulation"]["CaseA"]["tim_res"] == "Month"
    assert sim_data["simulation"]["CaseA"]["grid_res"] == 0.5
    assert sim_data["simulation"]["CaseA"]["data_groupby"] == "Year"
    assert sim_data["simulation"]["CaseA"]["prefix"] == "QFLX_EVAP_TOT_"

    report = yaml.safe_load(report_path.read_text())
    assert report["summary"]["cases"] == 1
    assert report["cases"][0]["label"] == "CaseA"
    assert report["cases"][0]["time_start"] == "2000-01-01T00:00:00"
    assert report["cases"][0]["time_end"] == "2000-12-01T00:00:00"
    assert report["cases"][0]["time_count"] == 12
    assert report["cases"][0]["time_span_days"] == 335


def test_sim_scan_station_collection_writes_fulllist_and_merged_station_files(
    tmp_path,
    monkeypatch,
):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    case_root = _write_station_case_nc(root, site_id="US-AAA", year=2000)
    _write_station_case_nc(root, site_id="US-AAA", year=2001)
    _write_station_case_nc(root, site_id="US-BBB", lon=-101.0, lat=41.0, year=2000)
    _write_station_case_nc(root, site_id="US-BBB", lon=-101.0, lat=41.0, year=2001)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-160500", raising=False)

    result = runner.invoke(
        cli,
        ["sim", "scan", str(root), "--model", "CoLM2024", "--station-workers", "1", "--auto"],
    )

    assert result.exit_code == 0, result.output
    sim_path = tmp_path / "openbench_sim_scan_20260501-160500.yaml"
    sim_data = yaml.safe_load(sim_path.read_text())
    entry = sim_data["simulation"]["CaseA"]
    assert entry["root_dir"] == str(case_root)
    assert entry["data_type"] == "stn"
    assert entry["data_groupby"] == "Single"
    assert "prefix" not in entry
    assert "suffix" not in entry

    fulllist = Path(entry["fulllist"])
    assert fulllist.exists()
    rows = pd.read_csv(fulllist)
    assert sorted(rows["ID"]) == ["US-AAA", "US-BBB"]
    assert set(rows.columns) >= {"ID", "sim_lon", "sim_lat", "use_syear", "use_eyear", "sim_dir"}
    assert all(Path(path).exists() for path in rows["sim_dir"])

    report = yaml.safe_load((tmp_path / "openbench_sim_scan_report_20260501-160500.yaml").read_text())
    assert report["cases"][0]["station_layout"] == "nested_multi"
    assert report["cases"][0]["station_count"] == 2
    assert Path(report["cases"][0]["merged_dir"]).exists()


def test_sim_scan_default_station_output_follows_explicit_output_parent(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "simulations"
    _write_station_case_nc(root, site_id="US-AAA", year=2000)
    output_dir = tmp_path / "configs"
    output_dir.mkdir()
    output_path = output_dir / "foo.yaml"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "CoLM2024",
            "--output",
            str(output_path),
            "--station-workers",
            "1",
            "--auto",
        ],
    )

    assert result.exit_code == 0, result.output
    sim_data = yaml.safe_load(output_path.read_text())
    raw_fulllist = sim_data["simulation"]["CaseA"]["fulllist"]
    fulllist = Path(raw_fulllist)
    if not fulllist.is_absolute():
        fulllist = (output_path.parent / fulllist).resolve()
    assert fulllist.parent.parent == output_dir / "foo_sim_station_lists"
    assert not list(cwd.glob("openbench_sim_scan_station_lists_*"))


def test_sim_scan_station_collection_merges_when_any_site_has_multiple_files(
    tmp_path,
    monkeypatch,
):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    _write_station_case_nc(root, site_id="US-AAA", year=2000)
    _write_station_case_nc(root, site_id="US-AAA", year=2001)
    _write_station_case_nc(root, site_id="US-BBB", lon=-101.0, lat=41.0, year=2000)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-161500", raising=False)

    result = runner.invoke(
        cli,
        ["sim", "scan", str(root), "--model", "CoLM2024", "--station-workers", "1", "--auto"],
    )

    assert result.exit_code == 0, result.output
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-161500.yaml").read_text())
    fulllist = Path(sim_data["simulation"]["CaseA"]["fulllist"])
    rows = pd.read_csv(fulllist).set_index("ID")

    assert rows.loc["US-AAA", "use_syear"] == 2000
    assert rows.loc["US-AAA", "use_eyear"] == 2001
    assert Path(rows.loc["US-AAA", "sim_dir"]).parent.name == "merged"
    assert Path(rows.loc["US-BBB", "sim_dir"]).parent.name == "merged"


def test_sim_scan_writes_inferred_prefix_suffix_for_runtime_lookup(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    _write_case_nc(root, filename="caseA_QFLX_EVAP_TOT_2000_hist.nc")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-162500", raising=False)

    result = runner.invoke(cli, ["sim", "scan", str(root), "--model", "CoLM2024", "--auto"])

    assert result.exit_code == 0, result.output
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-162500.yaml").read_text())
    entry = sim_data["simulation"]["CaseA"]
    assert entry["data_groupby"] == "Year"
    assert entry["prefix"] == "caseA_QFLX_EVAP_TOT_"
    assert entry["suffix"] == "_hist"


def test_sim_scan_writes_variable_file_overrides(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    history = root / "TE"
    for var_name, suffix, step in (("LTNT", "GLB050", 0.5), ("outflw", "GLB025", 0.25)):
        path = history / f"YEE2_JRA-55_{var_name}_M2000_{suffix}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {var_name: (["time", "lat", "lon"], np.zeros((12, 3, 4), dtype=np.float32))},
            coords={
                "time": np.arange(12),
                "lat": [10.0, 10.0 + step, 10.0 + 2 * step],
                "lon": [100.0, 100.0 + step, 100.0 + 2 * step, 100.0 + 3 * step],
            },
        )
        ds["time"].attrs["units"] = "calendar months since 2000-01-01 00:00:00 ; "
        ds[var_name].attrs["units"] = "1"
        ds.to_netcdf(path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-172500", raising=False)

    result = runner.invoke(cli, ["sim", "scan", str(root), "--model", "TE", "--auto"])

    assert result.exit_code == 0, result.output
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-172500.yaml").read_text())
    entry = sim_data["simulation"]["TE"]
    assert entry["variables"]["Latent_Heat"] == {
        "prefix": "YEE2_JRA-55_LTNT_M",
        "suffix": "_GLB050",
    }
    assert entry["variables"]["Streamflow"] == {
        "prefix": "YEE2_JRA-55_outflw_M",
        "suffix": "_GLB025",
        "grid_res": 0.25,
    }


def test_sim_scan_duplicate_labels_do_not_overwrite_yaml_entries(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    hist_a = _write_case_nc(root_a, label="CaseA")
    hist_b = _write_case_nc(root_b, label="CaseA")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-163500", raising=False)

    result = runner.invoke(
        cli,
        ["sim", "scan", str(root_a), str(root_b), "--model", "CoLM2024", "--auto"],
    )

    assert result.exit_code == 0, result.output
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-163500.yaml").read_text())
    simulation = sim_data["simulation"]
    assert "CaseA" in simulation
    duplicate_label = "CaseA__root_b"
    assert duplicate_label in simulation, sorted(simulation)
    assert simulation["CaseA"]["root_dir"] == str(hist_a)
    assert simulation[duplicate_label]["root_dir"] == str(hist_b)


def test_sim_scan_dry_run_does_not_write_outputs(tmp_path, monkeypatch):
    root = tmp_path / "simulations"
    _write_case_nc(root)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["sim", "scan", str(root), "--model", "CoLM2024", "--dry-run"])

    assert result.exit_code == 0
    assert "[DRY RUN]" in result.output
    assert list(tmp_path.glob("openbench_sim_scan_*.yaml")) == []
    assert list(tmp_path.glob("openbench_sim_scan_report_*.yaml")) == []


def test_sim_scan_rejects_file_root(tmp_path):
    nc_path = tmp_path / "single_case.nc"
    ds = xr.Dataset(
        {"QFLX_EVAP_TOT": (["time", "lat", "lon"], np.zeros((1, 2, 2)))},
        coords={"time": pd.date_range("2000-01-01", periods=1), "lat": [0.0, 0.5], "lon": [10.0, 10.5]},
    )
    ds.to_netcdf(nc_path)

    result = runner.invoke(cli, ["sim", "scan", str(nc_path), "--model", "CoLM2024", "--dry-run"])

    assert result.exit_code != 0
    assert "directory" in result.output.lower()


def test_sim_scan_expands_root_environment_variable(tmp_path, monkeypatch):
    root = tmp_path / "simulations"
    _write_case_nc(root)
    monkeypatch.setenv("OPENBENCH_TMP_SIM_ROOT", str(root))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["sim", "scan", "$OPENBENCH_TMP_SIM_ROOT", "--model", "CoLM2024", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "[DRY RUN]" in result.output


def test_sim_scan_expands_output_paths(tmp_path, monkeypatch):
    root = tmp_path / "simulations"
    _write_case_nc(root)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setenv("OPENBENCH_TMP_CONFIG_DIR", str(config_dir))

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "CoLM2024",
            "--output",
            "$OPENBENCH_TMP_CONFIG_DIR/out.yaml",
            "--report",
            "$OPENBENCH_TMP_CONFIG_DIR/report.yaml",
            "--station-output",
            "$OPENBENCH_TMP_CONFIG_DIR/stations",
            "--auto",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (config_dir / "out.yaml").exists()
    assert (config_dir / "report.yaml").exists()


def test_sim_scan_multiple_roots_uses_simulation_defaults(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    hist_a = _write_case_nc(root_a, label="CaseA")
    hist_b = _write_case_nc(root_b, label="CaseB")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-143500", raising=False)

    result = runner.invoke(
        cli,
        ["sim", "scan", str(root_a), str(root_b), "--model", "CoLM2024", "--auto"],
    )

    assert result.exit_code == 0
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-143500.yaml").read_text())
    simulation = sim_data["simulation"]
    assert simulation["_defaults"] == {
        "model": "CoLM2024",
        "data_type": "grid",
        "tim_res": "Month",
        "grid_res": 0.5,
        "data_groupby": "Year",
    }
    assert simulation["CaseA"] == {"root_dir": str(hist_a), "prefix": "QFLX_EVAP_TOT_"}
    assert simulation["CaseB"] == {"root_dir": str(hist_b), "prefix": "QFLX_EVAP_TOT_"}


def test_sim_scan_climatology_option_forces_monthly_climatology(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    _write_case_nc(root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-150500", raising=False)

    result = runner.invoke(
        cli,
        ["sim", "scan", str(root), "--model", "CoLM2024", "--climatology", "month", "--auto"],
    )

    assert result.exit_code == 0
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-150500.yaml").read_text())
    assert sim_data["simulation"]["CaseA"]["tim_res"] == "climatology-month"

    report = yaml.safe_load((tmp_path / "openbench_sim_scan_report_20260501-150500.yaml").read_text())
    assert report["cases"][0]["temporal_kind"] == "climatology-month"
    assert report["cases"][0]["provenance"]["temporal_kind"] == "user"


def test_sim_scan_auto_requires_explicit_climatology_for_monthly_candidate(tmp_path, monkeypatch):
    root = tmp_path / "simulations"
    _write_case_nc(root, filename="monthly_climatology.nc")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["sim", "scan", str(root), "--model", "CoLM2024", "--auto"])

    assert result.exit_code != 0
    assert "climatology-month" in result.output
    assert "--climatology month" in result.output
    assert "--climatology off" in result.output
    assert list(tmp_path.glob("openbench_sim_scan_*.yaml")) == []
    assert list(tmp_path.glob("openbench_sim_scan_report_*.yaml")) == []


def test_sim_scan_interactive_confirmation_applies_monthly_climatology_candidate(
    tmp_path,
    monkeypatch,
):
    import openbench.cli.sim as sim_module

    root = tmp_path / "simulations"
    _write_case_nc(root, filename="monthly_climatology.nc")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260501-151500", raising=False)

    result = runner.invoke(
        cli,
        ["sim", "scan", str(root), "--model", "CoLM2024"],
        input="y\ny\n",
    )

    assert result.exit_code == 0
    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260501-151500.yaml").read_text())
    assert sim_data["simulation"]["CaseA"]["tim_res"] == "climatology-month"

    report = yaml.safe_load((tmp_path / "openbench_sim_scan_report_20260501-151500.yaml").read_text())
    assert report["cases"][0]["temporal_kind"] == "climatology-month"
    assert report["cases"][0]["temporal_kind_candidate"] is None
    assert report["cases"][0]["provenance"]["temporal_kind"] == "user-confirmed"


def test_sim_scan_auto_model_requires_resolved_model_before_writing(tmp_path, monkeypatch):
    root = tmp_path / "simulations"
    _write_case_nc(root, label="MysteryCase", var_name="unknown_flux")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["sim", "scan", str(root), "--model", "auto", "--auto"])

    assert result.exit_code != 0
    assert "unresolved" in result.output.lower()
    assert "--model NewModel --register-model" in result.output
    assert "openbench model register" in result.output
    assert list(tmp_path.glob("openbench_sim_scan_*.yaml")) == []


def test_sim_scan_register_model_creates_draft_profile_for_unknown_model(tmp_path, monkeypatch):
    import openbench.cli.sim as sim_module

    home = tmp_path / "home"
    root = tmp_path / "simulations"
    _write_case_nc(root, label="MysteryCase", var_name="unknown_flux")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_module, "_timestamp", lambda: "20260502-101500", raising=False)

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "NewModel",
            "--register-model",
            "--auto",
        ],
    )

    assert result.exit_code == 0, result.output

    catalog_path = home / ".openbench" / "models" / "model_catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    profile = catalog["NewModel"]
    assert profile["name"] == "NewModel"
    assert profile["data_type"] == "grid"
    assert profile["tim_res"] == "Month"
    assert profile["grid_res"] == 0.5
    assert profile["variables"]["unknown_flux"] == {
        "varname": "unknown_flux",
        "varunit": "mm day-1",
    }

    sim_data = yaml.safe_load((tmp_path / "openbench_sim_scan_20260502-101500.yaml").read_text())
    assert sim_data["simulation"]["MysteryCase"]["model"] == "NewModel"


def test_sim_scan_dry_run_register_model_previews_even_when_profile_exists(tmp_path, monkeypatch):
    home = tmp_path / "home"
    catalog_path = home / ".openbench" / "models" / "model_catalog.yaml"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "ExistingModel": {
                    "name": "ExistingModel",
                    "description": "existing",
                    "data_type": "grid",
                    "grid_res": 0.5,
                    "tim_res": "Month",
                    "variables": {"unknown_flux": {"varname": "unknown_flux", "varunit": "mm day-1"}},
                }
            }
        )
    )
    root = tmp_path / "simulations"
    _write_case_nc(root, label="MysteryCase", var_name="unknown_flux")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "ExistingModel",
            "--register-model",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Would register draft model profile 'ExistingModel'" in result.output
    assert "[DRY RUN] No files written." in result.output


def test_sim_scan_register_model_rejects_mixed_case_data_types(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = tmp_path / "simulations"
    _write_case_nc(root, label="GridCase", var_name="unknown_flux")
    _write_station_case_nc(root, label="StationCase", var_name="station_flux")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "MixedModel",
            "--register-model",
            "--auto",
        ],
    )

    assert result.exit_code != 0
    assert "mixed data_type" in result.output
    assert not (home / ".openbench" / "models" / "model_catalog.yaml").exists()


def test_sim_scan_register_model_rolls_back_profile_when_later_write_fails(
    tmp_path,
    monkeypatch,
):
    import openbench.data.sim_scanner as sim_scanner

    home = tmp_path / "home"
    root = tmp_path / "simulations"
    _write_case_nc(root, label="MysteryCase", var_name="unknown_flux")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)

    def fail_materialize(*args, **kwargs):
        raise RuntimeError("station materialization failed")

    monkeypatch.setattr(sim_scanner, "materialize_station_cases", fail_materialize)

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "RollbackModel",
            "--register-model",
            "--auto",
        ],
    )

    assert result.exit_code != 0
    catalog_path = home / ".openbench" / "models" / "model_catalog.yaml"
    if catalog_path.exists():
        assert "RollbackModel" not in (yaml.safe_load(catalog_path.read_text()) or {})


def test_sim_scan_writes_report_before_rejecting_partial_station_materialization(
    tmp_path,
    monkeypatch,
):
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    root = tmp_path / "simulations"
    root.mkdir()
    output = tmp_path / "sim.yaml"
    report_path = tmp_path / "report.yaml"
    case = SimulationCase(
        label="CaseA",
        root_dir=root / "CaseA",
        source_root=root,
        model="CoLM2024",
        depth=1,
        data_type="stn",
        tim_res="Day",
        data_groupby="Single",
        station_layout="flat",
        station_count=2,
    )

    def fake_scan_simulation_roots(*args, **kwargs):
        return SimulationScanResult(roots=[root], cases=[case])

    def fake_materialize_station_cases(result, *args, **kwargs):
        result.cases[0].station_dropped_sites = ["US-BAD"]

    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", fake_materialize_station_cases)

    result = runner.invoke(
        cli,
        [
            "sim",
            "scan",
            str(root),
            "--model",
            "CoLM2024",
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--auto",
        ],
    )

    assert result.exit_code == 1
    assert "Station materialization dropped sites for: CaseA" in result.output
    assert str(report_path) in result.output
    assert report_path.exists()
    assert not output.exists()
    report = yaml.safe_load(report_path.read_text())
    assert report["cases"][0]["station_dropped_sites"] == ["US-BAD"]


def test_sim_scan_register_model_uses_shared_model_profile_writer(tmp_path, monkeypatch):
    import openbench.cli.model as model_module
    import openbench.cli.sim as sim_module

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    calls = []

    def fake_writer(name, profile, *, overwrite=False, exists_message=None):
        calls.append(
            {
                "name": name,
                "profile": profile,
                "overwrite": overwrite,
                "exists_message": exists_message,
            }
        )
        return tmp_path / "model_catalog.yaml"

    monkeypatch.setattr(model_module, "_write_model_profile_descriptor", fake_writer, raising=False)

    result = sim_module._register_model_profile_from_scan(
        "SharedWriterModel",
        {"name": "SharedWriterModel", "variables": {"Runoff": {"varname": "ro", "varunit": "mm"}}},
        overwrite=True,
    )

    assert result == tmp_path / "model_catalog.yaml"
    assert calls == [
        {
            "name": "SharedWriterModel",
            "profile": {
                "name": "SharedWriterModel",
                "variables": {"Runoff": {"varname": "ro", "varunit": "mm"}},
            },
            "overwrite": True,
            "exists_message": (
                "Use `openbench model register` to update variable mappings, "
                "pass --overwrite-model to merge the scanned draft, or omit --register-model."
            ),
        }
    ]


def test_infer_tim_res_from_time_coverage_preserves_subdaily_labels():
    from openbench.data.sim_scanner import _infer_tim_res_from_time_coverage

    assert _infer_tim_res_from_time_coverage({"time_count": 3, "time_span_seconds": 6 * 3600}) == "3Hour"
    assert _infer_tim_res_from_time_coverage({"time_count": 3, "time_span_seconds": 12 * 3600}) == "6Hour"


def test_rebase_station_artifacts_updates_fulllist_and_case_paths(tmp_path):
    from types import SimpleNamespace

    from openbench.cli.sim import _rebase_station_artifacts

    old_root = tmp_path / "station.tmp"
    new_root = tmp_path / "station"
    old_case = old_root / "CaseA"
    old_merged = old_case / "merged"
    old_merged.mkdir(parents=True)
    fulllist = old_case / "CaseA_stations.csv"
    fulllist.write_text("ID,sim_dir\nS1,%s\n" % (old_merged / "S1.nc"))
    new_root.parent.mkdir(parents=True, exist_ok=True)
    old_root.replace(new_root)

    case = SimpleNamespace(
        fulllist=new_root / "CaseA" / "CaseA_stations.csv",
        merged_dir=old_merged,
    )
    result = SimpleNamespace(cases=[case])

    _rebase_station_artifacts(result, old_root, new_root)

    assert case.merged_dir == new_root / "CaseA" / "merged"
    assert str(new_root / "CaseA" / "merged" / "S1.nc") in case.fulllist.read_text()
