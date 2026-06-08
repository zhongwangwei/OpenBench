"""Test that all CLI commands are registered and show help."""

from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import yaml
from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()


def _write_fake_cli_netcdf(path: Path):
    import numpy as np
    import xarray as xr

    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"value": ("sample", np.array([1.0]))}).to_netcdf(path)


def _write_fake_cli_grid_outputs(output_dir: Path, var_name: str, ref_source: str, sim_source: str):
    stem = f"{var_name}_ref_{ref_source}_sim_{sim_source}"
    _write_fake_cli_netcdf(output_dir / "metrics" / f"{stem}_bias.nc")
    _write_fake_cli_netcdf(output_dir / "scores" / f"{stem}_Overall_Score.nc")


class _InitFakeReference:
    def __init__(
        self,
        name="FLUXCOM_LowRes",
        variable="Latent_Heat",
        category="Energy",
        data_type="grid",
        tim_res="Month",
        grid_res=0.5,
        years=None,
        root_dir=None,
    ):
        self.name = name
        self.category = category
        self.data_type = data_type
        self.tim_res = tim_res
        self.grid_res = grid_res
        self.years = years
        self.root_dir = root_dir
        self.variables = {variable: {}}


def _install_single_reference_registry(monkeypatch, refs=None, models=None):
    import openbench.data.registry as registry_package

    refs = refs if refs is not None else [_InitFakeReference()]
    models = models if models is not None else []

    class FakeRegistryManager:
        def list_references(self):
            return refs

        def references_for_variable(self, variable):
            return [ref for ref in refs if variable in ref.variables]

        def list_models(self):
            return models

    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    return refs


def test_run_help():
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run evaluation" in result.output


def test_check_help():
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0
    assert "Validate" in result.output


def test_ref_list_help():
    result = runner.invoke(cli, ["ref", "list", "--help"])
    assert result.exit_code == 0


def test_ref_help():
    result = runner.invoke(cli, ["ref", "--help"])
    assert result.exit_code == 0
    assert "Manage reference datasets" in result.output


def test_ref_reuses_reference_subcommands():
    result = runner.invoke(cli, ["ref", "scan", "--help"])
    assert result.exit_code == 0
    assert "Scan a directory for reference datasets" in result.output


def test_ref_scan_writes_user_reference_catalog(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="TestRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    captured = []

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: [SimpleNamespace(variants={"default": variant})],
    )

    def fake_register_scanned_datasets_batch(datasets, catalog_path=None, **kwargs):
        captured.append((datasets, catalog_path))
        return catalog_path

    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        fake_register_scanned_datasets_batch,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--auto"])

    assert result.exit_code == 0, result.output
    assert captured == [([variant], home / ".openbench" / "references" / "reference_catalog.yaml")]


def test_ref_scan_reports_actual_catalog_path_returned_by_batch(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    written_path = tmp_path / "actual_reference_catalog.yaml"
    variant = SimpleNamespace(
        registry_name="TestRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: [SimpleNamespace(variants={"default": variant})],
    )
    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        lambda *args, **kwargs: written_path,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--auto"])

    assert result.exit_code == 0, result.output
    assert str(written_path) in result.output


def test_ref_scan_auto_fails_on_ambiguous_nc_variables(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="AmbiguousRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: [SimpleNamespace(variants={"default": variant})],
    )

    def fake_register_scanned_datasets_batch(datasets, catalog_path=None, on_multi_var=None, **kwargs):
        on_multi_var(
            "Runoff",
            "Water/Runoff/AmbiguousRef",
            [
                {"name": "primary", "unit": "", "dims": []},
                {"name": "alternate", "unit": "", "dims": []},
            ],
        )

    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        fake_register_scanned_datasets_batch,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--auto"])

    assert result.exit_code != 0
    assert "--auto cannot pick a variable" in result.output
    assert "primary, alternate" in result.output


def test_ref_scan_auto_pick_first_keeps_legacy_ambiguous_selection(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="AmbiguousRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    selected = []

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: [SimpleNamespace(variants={"default": variant})],
    )

    def fake_register_scanned_datasets_batch(datasets, catalog_path=None, on_multi_var=None, **kwargs):
        selected.append(
            on_multi_var(
                "Runoff",
                "Water/Runoff/AmbiguousRef",
                [
                    {"name": "primary", "unit": "", "dims": []},
                    {"name": "alternate", "unit": "", "dims": []},
                ],
            )
        )
        return catalog_path

    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        fake_register_scanned_datasets_batch,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--auto", "--pick-first"])

    assert result.exit_code == 0, result.output
    assert selected == ["primary"]


def test_ref_scan_interactive_ambiguous_variable_choice_reprompts(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="AmbiguousRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    selected = []

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: [SimpleNamespace(variants={"default": variant})],
    )

    def fake_register_scanned_datasets_batch(datasets, catalog_path=None, on_multi_var=None, **kwargs):
        selected.append(
            on_multi_var(
                "Runoff",
                "Water/Runoff/AmbiguousRef",
                [
                    {"name": "primary", "unit": "", "dims": []},
                    {"name": "alternate", "unit": "", "dims": []},
                ],
            )
        )
        return catalog_path

    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        fake_register_scanned_datasets_batch,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root)], input="y\n9\n2\n")

    assert result.exit_code == 0, result.output
    assert "Variable choice out of range" in result.output
    assert selected == ["alternate"]


def test_ref_scan_ambiguous_variable_empty_candidates_fails_cleanly(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="EmptyCandidatesRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: [SimpleNamespace(variants={"default": variant})],
    )

    def fake_register_scanned_datasets_batch(datasets, catalog_path=None, on_multi_var=None, **kwargs):
        on_multi_var("Runoff", "Water/Runoff/EmptyCandidatesRef", [])
        return catalog_path

    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        fake_register_scanned_datasets_batch,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root)], input="y\n")

    assert result.exit_code != 0
    assert "No NetCDF variables found" in result.output
    assert "Water/Runoff/EmptyCandidatesRef" in result.output


def test_ref_scan_pick_first_requires_auto(tmp_path):
    ref_root = tmp_path / "Reference"
    ref_root.mkdir()

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--pick-first", "--dry-run"])

    assert result.exit_code != 0
    assert "--pick-first requires --auto" in result.output


def test_ref_scan_auto_fails_when_unsupported_folders_would_be_skipped(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="ValidRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    registered = []

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/Bad",
                reason="ambiguous_nc_subdirectories",
                hint="Register manually.",
            )
        )
        return [SimpleNamespace(variants={"default": variant})]

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        lambda datasets, **kwargs: registered.extend(datasets),
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--auto"])

    assert result.exit_code != 0
    assert "unsupported folder" in result.output.lower()
    assert "Grid/LowRes/Composite/Bad" in result.output
    assert "--allow-skip" in result.output
    assert registered == []


def test_ref_scan_auto_allow_skip_continues_with_supported_datasets(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="ValidRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    registered = []

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/Bad",
                reason="ambiguous_nc_subdirectories",
                hint="Register manually.",
            )
        )
        return [SimpleNamespace(variants={"default": variant})]

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        lambda datasets, catalog_path=None, **kwargs: registered.extend(datasets) or catalog_path,
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--auto", "--allow-skip"])

    assert result.exit_code == 0, result.output
    assert "Continuing because --allow-skip" in result.output
    assert registered == [variant]


def test_ref_scan_interactive_requires_confirmation_for_unsupported_skips(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="ValidRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    registered = []

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Station/Water/Streamflow/Bad",
                reason="nc_files_too_deep",
                hint="Move files up.",
            )
        )
        return [SimpleNamespace(variants={"default": variant})]

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(
        scanner_module,
        "register_scanned_datasets_batch",
        lambda datasets, **kwargs: registered.extend(datasets),
    )

    result = runner.invoke(cli, ["ref", "scan", str(ref_root)], input="n\n")

    assert result.exit_code != 0
    assert "Skip 1 unsupported folder" in result.output
    assert "Scan cancelled" in result.output
    assert registered == []


def test_ref_scan_dry_run_reports_unsupported_skips_without_failing(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/Bad",
                reason="ambiguous_nc_subdirectories",
            )
        )
        return []

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Unsupported folder(s) skipped by scanner: 1" in result.output
    assert "No new datasets found" in result.output


def test_ref_scan_dry_run_warns_how_to_commit_when_skips_are_present(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    variant = SimpleNamespace(
        registry_name="ValidRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/Bad",
                reason="ambiguous_nc_subdirectories",
            )
        )
        return [SimpleNamespace(variants={"default": variant})]

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "[DRY RUN] Would register 1 dataset" in result.output
    assert "1 unsupported folder(s) would still need --allow-skip or interactive profile handling" in result.output


def test_ref_scan_profile_rescue_does_not_partially_write_if_prompt_fails(tmp_path, monkeypatch):
    import openbench.cli.data as cli_data
    import openbench.data.registry.scanner as scanner_module

    skipped = [
        scanner_module.ScanSkip(
            path="Grid/LowRes/Composite/First",
            reason="ambiguous_nc_subdirectories",
        ),
        scanner_module.ScanSkip(
            path="Grid/LowRes/Composite/Second",
            reason="ambiguous_nc_subdirectories",
        ),
    ]
    writes = []

    def fake_prompt(item, ref_root):
        if item is skipped[1]:
            raise click.ClickException("cannot rescue")
        return (
            "FirstProfile",
            {
                "scan": {
                    "layout": "grid_composite_files",
                    "root_sub_dir": "Grid/LowRes/Composite/First",
                },
                "variables": {},
            },
        )

    monkeypatch.setattr(cli_data, "_prompt_reference_profile_for_scan_skip", fake_prompt)
    monkeypatch.setattr(cli_data, "_write_reference_profile", lambda name, profile: writes.append(name))

    with pytest.raises(click.ClickException, match="cannot rescue"):
        cli_data._create_profiles_for_scan_skips(skipped, tmp_path)

    assert writes == []


def test_write_reference_profile_confirms_before_overwriting_scan_config(tmp_path, monkeypatch):
    import openbench.cli.data as cli_data

    home = tmp_path / "home"
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump(
            {
                "ExistingProfile": {
                    "scan": {
                        "layout": "grid_composite_files",
                        "root_sub_dir": "Grid/LowRes/Composite/Original",
                    },
                    "variables": {
                        "Latent_Heat": {
                            "root_sub_dir": "Grid/LowRes/Composite/Original/land",
                            "varname": "LE",
                            "varunit": "W m-2",
                        },
                    },
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: False)

    with pytest.raises(click.ClickException, match="Profile scan overwrite cancelled"):
        cli_data._write_reference_profile(
            "ExistingProfile",
            {
                "scan": {
                    "layout": "grid_composite_files",
                    "root_sub_dir": "Grid/MidRes/Composite/New",
                },
                "variables": {
                    "Streamflow": {
                        "root_sub_dir": "Grid/MidRes/Composite/New/cama",
                        "varname": "discharge",
                        "varunit": "m3 s-1",
                    },
                },
            },
        )

    stored = yaml.safe_load(profile_path.read_text())["ExistingProfile"]
    assert stored["scan"]["root_sub_dir"] == "Grid/LowRes/Composite/Original"
    assert "Streamflow" not in stored["variables"]


def test_ref_scan_profile_rescue_stops_when_same_skips_remain(tmp_path, monkeypatch):
    import openbench.cli.data as cli_data
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    calls = []

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/StillBad",
                reason="ambiguous_nc_subdirectories",
            )
        )
        return []

    def fake_create_profiles(skipped, root):
        calls.append([item.path for item in skipped])
        return 1

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(cli_data, "_create_profiles_for_scan_skips", fake_create_profiles)

    result = runner.invoke(cli, ["ref", "scan", str(ref_root)], input="p\n")

    assert result.exit_code != 0
    assert "Profile creation did not resolve unsupported folder(s)" in result.output
    assert calls == [["Grid/LowRes/Composite/StillBad"]]


def test_ref_scan_profile_rescue_does_not_stall_on_same_path_new_reason(tmp_path, monkeypatch):
    import openbench.cli.data as cli_data
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    calls = []

    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        reason = "ambiguous_nc_subdirectories" if len(calls) == 0 else "mixed_grid_resolutions_in_profile"
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/StillBad",
                reason=reason,
            )
        )
        return []

    def fake_create_profiles(skipped, root):
        calls.append([item.reason for item in skipped])
        return 1

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(cli_data, "_create_profiles_for_scan_skips", fake_create_profiles)

    result = runner.invoke(cli, ["ref", "scan", str(ref_root)], input="p\ns\n")

    assert result.exit_code == 0, result.output
    assert "mixed_grid_resolutions_in_profile" in result.output
    assert "Profile creation did not resolve unsupported folder(s)" not in result.output
    assert calls == [["ambiguous_nc_subdirectories"]]


def test_ref_scan_interactive_can_add_profile_and_rescan(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    for child, nc_var, unit in (
        ("land", "LE", "W m-2"),
        ("cama", "discharge", "m3 s-1"),
    ):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "BadComposite" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {nc_var: (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": unit})},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{child}_2010.nc")

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root)],
        input="\n".join(
            [
                "p",  # create/update profile and rescan
                "",  # profile name default: BadComposite_LowRes
                "",  # file glob for cama
                "Streamflow",  # standard name for cama
                "",  # NetCDF variable default: discharge
                "",  # unit default: m3 s-1
                "",  # file glob for land
                "Latent_Heat",  # standard name for land
                "",  # NetCDF variable default: LE
                "",  # unit default: W m-2
                "y",  # register after rescan
            ]
        )
        + "\n",
    )

    assert result.exit_code == 0, result.output
    assert "[p] create/update reference profile and rescan" in result.output
    assert "[r]" not in result.output
    assert "Updated reference profile: BadComposite_LowRes" in result.output
    assert "Registered 1 dataset" in result.output

    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text())["BadComposite_LowRes"]
    assert profile["scan"] == {
        "layout": "grid_composite_files",
        "root_sub_dir": "Grid/LowRes/Composite/BadComposite",
    }
    assert profile["variables"]["Latent_Heat"] == {
        "root_sub_dir": "Grid/LowRes/Composite/BadComposite/land",
        "varname": "LE",
        "varunit": "W m-2",
    }
    assert profile["variables"]["Streamflow"] == {
        "root_sub_dir": "Grid/LowRes/Composite/BadComposite/cama",
        "varname": "discharge",
        "varunit": "m3 s-1",
    }

    catalog_path = home / ".openbench" / "references" / "reference_catalog.yaml"
    entry = yaml.safe_load(catalog_path.read_text())["BadComposite_LowRes"]
    assert entry["variables"]["Latent_Heat"]["sub_dir"] == "Composite/BadComposite/land"
    assert entry["variables"]["Streamflow"]["sub_dir"] == "Composite/BadComposite/cama"


def test_ref_scan_profile_rescue_accepts_multiple_vars_from_one_nc(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    cama_dir = ref_root / "Grid" / "LowRes" / "Composite" / "BadComposite" / "cama"
    cama_dir.mkdir(parents=True)
    xr.Dataset(
        {"discharge": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": "m3 s-1"})},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    ).to_netcdf(cama_dir / "cama_2010.nc")

    land_dir = ref_root / "Grid" / "LowRes" / "Composite" / "BadComposite" / "land"
    land_dir.mkdir(parents=True)
    xr.Dataset(
        {
            "LE": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": "W m-2"}),
            "H": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": "W m-2"}),
            "Rn": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": "W m-2"}),
        },
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    ).to_netcdf(land_dir / "land_2010.nc")

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root)],
        input="\n".join(
            [
                "p",
                "",
                "",
                "Streamflow",
                "",
                "",
                "",
                "Latent_Heat",
                "bad",
                "le",
                "9",
                "1",
                "",
                "Sensible_Heat",
                "2",
                "",
                "Net_Radiation",
                "Rn",
                "",
                "",
                "y",
            ]
        )
        + "\n",
    )

    assert result.exit_code == 0, result.output
    assert "Multiple NetCDF variables detected" in result.output
    assert "Invalid NetCDF variable 'bad'. Choose one of: LE, H, Rn." in result.output
    assert "Invalid NetCDF variable 'le'. Did you mean 'LE'?" in result.output
    assert "Variable choice out of range: 9 (expected 1-3)." in result.output
    assert "Select variable number" not in result.output

    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    variables = yaml.safe_load(profile_path.read_text())["BadComposite_LowRes"]["variables"]
    assert variables["Latent_Heat"]["varname"] == "LE"
    assert variables["Sensible_Heat"]["varname"] == "H"
    assert variables["Net_Radiation"]["varname"] == "Rn"
    assert variables["Streamflow"]["varname"] == "discharge"

    catalog_path = home / ".openbench" / "references" / "reference_catalog.yaml"
    entry_vars = yaml.safe_load(catalog_path.read_text())["BadComposite_LowRes"]["variables"]
    assert entry_vars["Latent_Heat"]["varname"] == "LE"
    assert entry_vars["Sensible_Heat"]["varname"] == "H"
    assert entry_vars["Net_Radiation"]["varname"] == "Rn"
    assert entry_vars["Latent_Heat"]["sub_dir"] == "Composite/BadComposite/land"
    assert entry_vars["Sensible_Heat"]["sub_dir"] == "Composite/BadComposite/land"
    assert entry_vars["Net_Radiation"]["sub_dir"] == "Composite/BadComposite/land"


def test_ref_scan_profile_rescue_writes_file_glob_and_allows_child_skip(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    for child, nc_var in (("land", "LE"), ("mask", "mask")):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "MixedSource" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {nc_var: (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": "1"})},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{child}_2010.nc")

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root)],
        input="\n".join(
            [
                "p",
                "",
                "*land*.nc",
                "Latent_Heat",
                "",
                "W m-2",
                "",
                "",
                "y",
            ]
        )
        + "\n",
    )

    assert result.exit_code == 0, result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    variables = yaml.safe_load(profile_path.read_text())["MixedSource_LowRes"]["variables"]
    assert variables == {
        "Latent_Heat": {
            "root_sub_dir": "Grid/LowRes/Composite/MixedSource/land",
            "file_glob": "*land*.nc",
            "varname": "LE",
            "varunit": "W m-2",
        }
    }
    catalog_path = home / ".openbench" / "references" / "reference_catalog.yaml"
    entry_vars = yaml.safe_load(catalog_path.read_text())["MixedSource_LowRes"]["variables"]
    assert set(entry_vars) == {"Latent_Heat"}


def test_ref_scan_profile_rescue_creates_station_profile(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    base = ref_root / "Station" / "Carbon" / "CH4_Flux" / "MyStn"
    for year in ("2010", "2011"):
        nc_dir = base / year
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {
                "CH4_flux": (
                    ["time"],
                    np.array([1.0], dtype=np.float32),
                    {"units": "g m-2 day-1"},
                )
            },
            coords={"time": np.array([f"{year}-01-01"], dtype="datetime64[ns]")},
        )
        ds.to_netcdf(nc_dir / f"MyStn_{year}.nc")

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root)],
        input="\n".join(
            [
                "p",
                "",
                "",
                "CH4_Flux",
                "",
                "",
                "y",
            ]
        )
        + "\n",
    )

    assert result.exit_code == 0, result.output
    assert "Updated reference profile: MyStn_Station" in result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text())["MyStn_Station"]
    assert profile["scan"] == {
        "layout": "station_direct",
        "root_sub_dir": "Station/Carbon/CH4_Flux/MyStn",
        "file_glob": "**/*.nc",
    }
    assert profile["variables"] == {"CH4_Flux": {"varname": "CH4_flux", "varunit": "g m-2 day-1"}}
    catalog_path = home / ".openbench" / "references" / "reference_catalog.yaml"
    entry = yaml.safe_load(catalog_path.read_text())["MyStn_Station"]
    assert entry["data_type"] == "stn"
    assert entry["variables"]["CH4_Flux"]["varname"] == "CH4_flux"


def test_ref_scan_profile_rescue_creates_grid_nested_profile_for_deep_nc(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    nc_dir = ref_root / "Grid" / "LowRes" / "Water" / "Runoff" / "DeepDS" / "a" / "b" / "c"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {"ro": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32), {"units": "mm day-1"})},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "deep_2010.nc")

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root)],
        input="\n".join(
            [
                "p",
                "",
                "",
                "",
                "",
                "y",
            ]
        )
        + "\n",
    )

    assert result.exit_code == 0, result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text())["DeepDS_LowRes"]
    assert profile["scan"] == {
        "layout": "grid_nested_root",
        "root_sub_dir": "Grid/LowRes/Water/Runoff/DeepDS",
    }
    assert profile["variables"] == {
        "Runoff": {
            "sub_dir": "Water/Runoff/DeepDS/a/b/c",
            "varname": "ro",
            "varunit": "mm day-1",
        }
    }


def test_ref_scan_ignore_action_writes_ignore_profile_and_rescans(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    for child in ("raw_a", "raw_b"):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "RawComposite" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {"x": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{child}_2010.nc")

    result = runner.invoke(cli, ["ref", "scan", str(ref_root)], input="i\n")

    assert result.exit_code == 0, result.output
    assert "[i] add ignore profile and rescan" in result.output
    assert "Updated 1 ignore profile" in result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profiles = yaml.safe_load(profile_path.read_text())
    assert any(
        profile.get("scan")
        == {
            "layout": "ignore",
            "root_sub_dir": "Grid/LowRes/Composite/RawComposite",
        }
        for profile in profiles.values()
    )


def test_ref_scan_does_not_report_skips_for_already_registered_dataset(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    catalog_path = home / ".openbench" / "references" / "reference_catalog.yaml"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "BadComposite_LowRes": {
                    "name": "BadComposite_LowRes",
                    "category": "Other",
                    "data_type": "grid",
                    "tim_res": "Month",
                    "data_groupby": "Single",
                    "root_dir": str(ref_root / "Grid" / "LowRes"),
                    "variables": {
                        "Latent_Heat": {
                            "varname": "LE",
                            "varunit": "W m-2",
                            "sub_dir": "Composite/BadComposite/land",
                        }
                    },
                }
            }
        )
    )

    for child, nc_var in (("land", "LE"), ("cama", "discharge")):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "BadComposite" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {nc_var: (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{child}_2010.nc")

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Unsupported folder(s) skipped" not in result.output
    assert "No new datasets found" in result.output


def test_ref_scan_expands_root_environment_variable(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    seen = []

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OPENBENCH_TMP_SCAN_ROOT", str(ref_root))
    monkeypatch.setattr(
        scanner_module,
        "find_new_datasets",
        lambda root, on_progress=None, on_skip=None: seen.append(root) or [],
    )

    result = runner.invoke(cli, ["ref", "scan", "$OPENBENCH_TMP_SCAN_ROOT", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert seen == [str(ref_root)]


def test_ref_scan_only_filters_rescan_results(tmp_path, monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    keep = DatasetGroup(base_name="KeepMe")
    keep.variants["LowRes"] = ScannedDataset(
        name="KeepMe",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
    )
    drop = DatasetGroup(base_name="DropMe")
    drop.variants["LowRes"] = ScannedDataset(
        name="DropMe",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
    )

    import openbench.data.registry.scanner as scanner_module

    monkeypatch.setattr(
        scanner_module,
        "scan_reference_directory",
        lambda root, on_progress=None, on_skip=None: [keep, drop],
    )

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root), "--rescan", "--only", "Keep*", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "KeepMe_LowRes" in result.output
    assert "DropMe_LowRes" not in result.output


def test_ref_scan_only_ignores_unmatched_skips_when_committing(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset, ScanSkip

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    keep = DatasetGroup(base_name="KeepMe")
    keep_variant = ScannedDataset(
        name="KeepMe",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
    )
    keep.variants["LowRes"] = keep_variant
    captured = []

    def fake_scan(root, on_progress=None, on_skip=None):
        on_skip(
            ScanSkip(
                path="Grid/LowRes/Water/Runoff/DropMe",
                reason="ambiguous_nc_subdirectories",
                hint="unrelated to --only",
            )
        )
        return [keep]

    def fake_register(datasets, catalog_path=None, **kwargs):
        captured.extend(datasets)
        return catalog_path

    monkeypatch.setattr(scanner_module, "scan_reference_directory", fake_scan)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register)

    result = runner.invoke(
        cli,
        ["ref", "scan", str(ref_root), "--rescan", "--only", "Keep*", "--auto"],
    )

    assert result.exit_code == 0, result.output
    assert captured == [keep_variant]
    assert "DropMe" not in result.output


def test_ref_scan_dry_run_previews_profile_rescue(tmp_path, monkeypatch):
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    def fake_find_new_datasets(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Water/Runoff/DailyMonthly",
                reason="ambiguous_nc_subdirectories",
                hint="choose a child",
            )
        )
        return []

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)

    result = runner.invoke(cli, ["ref", "scan", str(ref_root), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Profile rescue preview" in result.output
    assert "DailyMonthly_LowRes" in result.output
    assert "grid_dataset_choice" in result.output
    assert not (home / ".openbench" / "references" / "reference_profiles.yaml").exists()


def test_ref_scan_rejects_file_before_scanning(tmp_path):
    ref_file = tmp_path / "reference.yaml"
    ref_file.write_text("{}\n")

    result = runner.invoke(cli, ["ref", "scan", str(ref_file), "--dry-run"])

    assert result.exit_code != 0
    assert "must be a directory" in result.output
    assert "Scanning" not in result.output


def test_ref_register_writes_user_reference_catalog(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    user_catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    assert "ManualDS" in yaml.safe_load(user_catalog.read_text())


def test_ref_register_data_type_autodetect_handles_iterdir_errors(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    original_iterdir = Path.iterdir

    def fail_data_root_iterdir(self):
        if self == data_root:
            raise PermissionError("blocked")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fail_data_root_iterdir)

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(data_root),
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Could not inspect subdirectories" in result.output
    user_catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    assert yaml.safe_load(user_catalog.read_text())["ManualDS"]["data_type"] == "grid"


def test_ref_register_reports_corrupt_catalog_without_traceback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    data_root.mkdir()
    catalog.parent.mkdir(parents=True)
    catalog.write_text("bad: [\n")

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 1
    assert "Failed to load existing catalog" in result.output
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, RuntimeError)


def test_ref_register_rejects_missing_root_dir_without_traceback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    missing = tmp_path / "missing"

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(missing),
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_ref_register_rejects_invalid_resolution_and_years(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))

    bad_grid = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "BadGridRef",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "--grid-res",
            "-1",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )
    bad_years = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "BadYearsRef",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "--years",
            "2025",
            "2020",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert bad_grid.exit_code == 1
    assert "--grid-res must be a positive value" in bad_grid.output
    assert bad_years.exit_code == 1
    assert "--years start year must be <= end year" in bad_years.output
    catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    assert not catalog.exists() or not (yaml.safe_load(catalog.read_text()) or {})


def test_ref_register_rejects_station_grid_res(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualStation"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "StationRef",
            "--root-dir",
            str(data_root),
            "--data-type",
            "stn",
            "--grid-res",
            "0.5",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code != 0
    assert "--grid-res is not valid for station reference datasets" in result.output


def test_ref_register_variable_option_accepts_prefix_suffix(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "-v",
            "Runoff:ro:mm day-1:runoff_:_monthly",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())["ManualDS"]
    assert descriptor["variables"]["Runoff"] == {
        "varname": "ro",
        "varunit": "mm day-1",
        "prefix": "runoff_",
        "suffix": "_monthly",
    }


def test_ref_register_accepts_climatology_tim_res(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "--tim-res",
            "climatology-month",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())["ManualDS"]
    assert descriptor["tim_res"] == "climatology-month"


def test_ref_register_expands_root_dir_environment_variable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OPENBENCH_TMP_REF_ROOT", str(data_root))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            "$OPENBENCH_TMP_REF_ROOT",
            "--data-type",
            "grid",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())["ManualDS"]
    assert descriptor["root_dir"] == str(data_root)


def test_ref_register_expands_fulllist_environment_variable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualStn"
    data_root.mkdir()
    station_list = tmp_path / "stations.csv"
    station_list.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\n")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OPENBENCH_TMP_FULLLIST", str(station_list))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualStn",
            "--root-dir",
            str(data_root),
            "--data-type",
            "stn",
            "--fulllist",
            "$OPENBENCH_TMP_FULLLIST",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())[
        "ManualStn"
    ]
    assert descriptor["fulllist"] == str(station_list)


def test_ref_register_resolves_relative_fulllist_against_root_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualStn"
    list_dir = data_root / "list"
    list_dir.mkdir(parents=True)
    (list_dir / "stations.csv").write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\n")

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualStn",
            "--root-dir",
            str(data_root),
            "--data-type",
            "stn",
            "--fulllist",
            "list/stations.csv",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())[
        "ManualStn"
    ]
    assert descriptor["fulllist"] == str(data_root / "list" / "stations.csv")


def test_ref_register_accepts_scanner_tim_res_values(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_root = tmp_path / "ManualDS"
    data_root.mkdir()

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "ManualDS",
            "--root-dir",
            str(data_root),
            "--data-type",
            "grid",
            "--tim-res",
            "8Day",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())["ManualDS"]
    assert descriptor["tim_res"] == "8Day"


def test_parse_fallbacks_preserves_conversion_with_colons():
    from openbench.cli._parsing import parse_fallbacks, parse_variables

    variables = parse_variables(("Runoff:ro:mm day-1",))

    parse_fallbacks(
        ("Runoff:q:mm day-1:mapping['a:b'] if value > 0 else defaults['c:d']",),
        variables,
    )

    fallback = variables["Runoff"]["fallbacks"][0]
    assert fallback["convert"] == "mapping['a:b'] if value > 0 else defaults['c:d']"


def test_ref_register_updates_existing_reference_via_user_overlay(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    home = tmp_path / "home"
    existing = ReferenceDataset(
        name="BuiltinDS",
        description="Built-in descriptor",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2001, 2002],
        root_dir="/builtin/root",
        grid_res=0.5,
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_reference(self, name):
            return existing if name == "BuiltinDS" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(
        cli,
        ["ref", "register", "BuiltinDS", "-v", "Evapotranspiration:ET:mm day-1"],
    )

    assert result.exit_code == 0, result.output
    user_catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    descriptor = yaml.safe_load(user_catalog.read_text())["BuiltinDS"]
    assert descriptor["root_dir"] == "/builtin/root"
    assert descriptor["variables"]["Runoff"]["varname"] == "ro"
    assert descriptor["variables"]["Evapotranspiration"]["varname"] == "ET"


def test_ref_register_updates_existing_reference_metadata_options(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    home = tmp_path / "home"
    existing = ReferenceDataset(
        name="BuiltinDS",
        description="Built-in descriptor",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2001, 2002],
        root_dir="/builtin/root",
        grid_res=0.5,
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_reference(self, name):
            return existing if name == "BuiltinDS" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "BuiltinDS",
            "--data-type",
            "grid",
            "--tim-res",
            "Day",
            "--grid-res",
            "0.25",
            "--category",
            "Hydrology",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())[
        "BuiltinDS"
    ]
    assert descriptor["data_type"] == "grid"
    assert descriptor["tim_res"] == "Day"
    assert descriptor["grid_res"] == 0.25
    assert descriptor["category"] == "Hydrology"


def test_ref_register_existing_grid_to_station_drops_grid_res(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    home = tmp_path / "home"
    existing = ReferenceDataset(
        name="BuiltinDS",
        description="Built-in descriptor",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2001, 2002],
        root_dir="/builtin/root",
        grid_res=0.5,
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_reference(self, name):
            return existing if name == "BuiltinDS" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(
        cli,
        [
            "ref",
            "register",
            "BuiltinDS",
            "--data-type",
            "stn",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())[
        "BuiltinDS"
    ]
    assert descriptor["data_type"] == "stn"
    assert "grid_res" not in descriptor


def test_ref_register_does_not_default_existing_station_reference_to_grid(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    home = tmp_path / "home"
    existing = ReferenceDataset(
        name="MyStn",
        description="Station reference",
        category="Energy",
        data_type="stn",
        tim_res="Hour",
        data_groupby="Single",
        timezone=0,
        years=[2001, 2002],
        root_dir="/builtin/station",
        variables={"Latent_Heat": VariableMapping(varname="Qle", varunit="W m-2")},
    )

    class FakeRegistryManager:
        def get_reference(self, name):
            return existing if name == "MyStn" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["ref", "register", "MyStn", "-v", "Sensible_Heat:Qh:W m-2"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())["MyStn"]
    assert descriptor["data_type"] == "stn"
    assert descriptor["variables"]["Sensible_Heat"]["varname"] == "Qh"


def test_ref_register_reports_metadata_only_update(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    home = tmp_path / "home"
    existing = ReferenceDataset(
        name="BuiltinDS",
        description="Built-in descriptor",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2001, 2002],
        root_dir="/builtin/root",
        grid_res=0.5,
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_reference(self, name):
            return existing if name == "BuiltinDS" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["ref", "register", "BuiltinDS", "--category", "Hydrology"])

    assert result.exit_code == 0, result.output
    assert "metadata updated" in result.output


def test_ref_register_handles_null_variables_in_catalog(tmp_path, monkeypatch):
    home = tmp_path / "home"
    catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "NullRef:\n"
        "  name: NullRef\n"
        "  description: demo\n"
        "  category: Water\n"
        "  data_type: grid\n"
        "  tim_res: Month\n"
        "  data_groupby: Year\n"
        "  timezone: 0\n"
        "  years: [2000, 2001]\n"
        "  root_dir: /tmp\n"
        "  variables: null\n"
    )

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(cli, ["ref", "register", "NullRef", "-v", "Runoff:ro:mm day-1"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(catalog.read_text())["NullRef"]
    assert descriptor["variables"]["Runoff"]["varname"] == "ro"


def test_ref_register_merges_against_latest_catalog_under_write_lock(tmp_path, monkeypatch):
    import copy
    from contextlib import nullcontext

    import openbench.data.registry.manager as manager_module
    import openbench.data.registry.scanner as scanner_module

    catalog_path = tmp_path / "reference_catalog.yaml"
    stale_catalog = {
        "RaceRef": {
            "name": "RaceRef",
            "description": "demo",
            "category": "Water",
            "data_type": "grid",
            "tim_res": "Month",
            "data_groupby": "Year",
            "timezone": 0,
            "years": [2000, 2001],
            "root_dir": "/tmp",
            "variables": {"V1": {"varname": "v1", "varunit": ""}},
        }
    }
    latest_catalog = copy.deepcopy(stale_catalog)
    latest_catalog["RaceRef"]["variables"]["V4"] = {"varname": "v4", "varunit": ""}
    loads = [stale_catalog, latest_catalog]
    written = {}

    monkeypatch.setattr(manager_module, "get_writable_reference_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(scanner_module, "_safe_load_catalog", lambda path: copy.deepcopy(loads.pop(0)))
    monkeypatch.setattr(scanner_module, "_catalog_write_lock", lambda path: nullcontext())
    monkeypatch.setattr(
        scanner_module,
        "_backup_then_write",
        lambda path, catalog: written.update({"catalog": copy.deepcopy(catalog)}),
    )
    monkeypatch.setattr(scanner_module, "_invalidate_registry_caches", lambda: None)

    result = runner.invoke(cli, ["ref", "register", "RaceRef", "-v", "V3:v3:mm day-1"])

    assert result.exit_code == 0, result.output
    variables = written["catalog"]["RaceRef"]["variables"]
    assert set(variables) == {"V1", "V3", "V4"}


def test_ref_register_rechecks_new_dataset_root_dir_under_write_lock(tmp_path, monkeypatch):
    import copy
    from contextlib import nullcontext

    import openbench.data.registry.manager as manager_module
    import openbench.data.registry.scanner as scanner_module

    catalog_path = tmp_path / "reference_catalog.yaml"
    stale_catalog = {
        "RaceRef": {
            "name": "RaceRef",
            "description": "demo",
            "category": "Water",
            "data_type": "grid",
            "tim_res": "Month",
            "data_groupby": "Year",
            "timezone": 0,
            "years": [2000, 2001],
            "root_dir": "/tmp",
            "variables": {"V1": {"varname": "v1", "varunit": ""}},
        }
    }
    loads = [stale_catalog, {}]
    written = {}

    monkeypatch.setattr(manager_module, "get_writable_reference_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(scanner_module, "_safe_load_catalog", lambda path: copy.deepcopy(loads.pop(0)))
    monkeypatch.setattr(scanner_module, "_catalog_write_lock", lambda path: nullcontext())
    monkeypatch.setattr(
        scanner_module,
        "_backup_then_write",
        lambda path, catalog: written.update({"catalog": copy.deepcopy(catalog)}),
    )
    monkeypatch.setattr(scanner_module, "_invalidate_registry_caches", lambda: None)

    result = runner.invoke(cli, ["ref", "register", "RaceRef", "-v", "V3:v3:mm day-1"])

    assert result.exit_code != 0
    assert "--root-dir is required for new datasets" in result.output
    assert written == {}


def test_ref_register_profile_reports_updated_for_existing_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    first = runner.invoke(
        cli,
        ["ref", "register-profile", "ManualProfile", "-v", "Runoff:ro:mm day-1"],
    )
    assert first.exit_code == 0, first.output
    assert "Created profile 'ManualProfile'" in first.output

    second = runner.invoke(
        cli,
        ["ref", "register-profile", "ManualProfile", "--description", "Updated profile"],
    )

    assert second.exit_code == 0, second.output
    assert "Updated profile 'ManualProfile'" in second.output
    assert "Created profile 'ManualProfile'" not in second.output


def test_ref_register_profile_handles_null_variables_in_catalog(tmp_path, monkeypatch):
    home = tmp_path / "home"
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("NullProfile:\n  description: demo\n  variables: null\n")

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        ["ref", "register-profile", "NullProfile", "-v", "Runoff:ro:mm day-1"],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(profile_path.read_text())["NullProfile"]
    assert descriptor["variables"]["Runoff"]["varname"] == "ro"


def test_ref_register_profile_accepts_prefix_suffix_and_fallback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register-profile",
            "ManualProfile",
            "-v",
            "Runoff:ro:mm day-1:runoff_:_monthly",
            "-f",
            "Runoff:runoff_alt:mm day-1:value['a:b']",
        ],
    )

    assert result.exit_code == 0, result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    runoff = yaml.safe_load(profile_path.read_text())["ManualProfile"]["variables"]["Runoff"]
    assert runoff["prefix"] == "runoff_"
    assert runoff["suffix"] == "_monthly"
    assert runoff["fallbacks"] == [{"varname": "runoff_alt", "varunit": "mm day-1", "convert": "value['a:b']"}]


def test_ref_register_profile_accepts_category_and_validates_data_groupby(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    bad = runner.invoke(
        cli,
        [
            "ref",
            "register-profile",
            "ManualProfile",
            "-v",
            "Runoff:ro:mm day-1",
            "--data-groupby",
            "weekly",
        ],
    )
    assert bad.exit_code != 0
    assert "Invalid value for '--data-groupby'" in bad.output

    result = runner.invoke(
        cli,
        [
            "ref",
            "register-profile",
            "ManualProfile",
            "-v",
            "Runoff:ro:mm day-1",
            "--category",
            "Water",
            "--data-groupby",
            "Day",
        ],
    )
    assert result.exit_code == 0, result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text())["ManualProfile"]
    assert profile["category"] == "Water"
    assert profile["data_groupby"] == "Day"


def test_ref_register_profile_rejects_climatology_data_groupby(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "ref",
            "register-profile",
            "ManualProfile",
            "-v",
            "Runoff:ro:mm day-1",
            "--data-groupby",
            "climatology-month",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--data-groupby'" in result.output
    profile_path = home / ".openbench" / "references" / "reference_profiles.yaml"
    assert not profile_path.exists()


def test_ref_register_profile_merges_against_latest_catalog_under_write_lock(tmp_path, monkeypatch):
    import copy
    from contextlib import nullcontext

    import openbench.data.registry.manager as manager_module
    import openbench.data.registry.scanner as scanner_module

    profile_path = tmp_path / "reference_profiles.yaml"
    stale_profiles = {
        "RaceProfile": {
            "description": "demo",
            "variables": {"V1": {"varname": "v1", "varunit": ""}},
        }
    }
    latest_profiles = copy.deepcopy(stale_profiles)
    latest_profiles["RaceProfile"]["variables"]["V4"] = {"varname": "v4", "varunit": ""}
    loads = [stale_profiles, latest_profiles]
    written = {}

    monkeypatch.setattr(manager_module, "get_writable_reference_profiles_path", lambda: profile_path)
    monkeypatch.setattr(scanner_module, "_safe_load_catalog", lambda path: copy.deepcopy(loads.pop(0)))
    monkeypatch.setattr(scanner_module, "_catalog_write_lock", lambda path: nullcontext())
    monkeypatch.setattr(
        scanner_module,
        "_backup_then_write",
        lambda path, catalog: written.update({"catalog": copy.deepcopy(catalog)}),
    )
    monkeypatch.setattr(scanner_module, "_invalidate_registry_caches", lambda: None)

    result = runner.invoke(cli, ["ref", "register-profile", "RaceProfile", "-v", "V3:v3:mm day-1"])

    assert result.exit_code == 0, result.output
    variables = written["catalog"]["RaceProfile"]["variables"]
    assert set(variables) == {"V1", "V3", "V4"}


def test_ref_download_help():
    result = runner.invoke(cli, ["ref", "download", "--help"])
    assert result.exit_code == 0


def test_ref_download_returns_nonzero_until_implemented():
    result = runner.invoke(cli, ["ref", "download", "GLEAM"])
    assert result.exit_code != 0
    assert "not yet implemented" in result.output


def test_ref_status_help():
    result = runner.invoke(cli, ["ref", "status", "--help"])
    assert result.exit_code == 0


def test_ref_path_help():
    result = runner.invoke(cli, ["ref", "path", "--help"])
    assert result.exit_code == 0


def test_ref_optimize_help():
    result = runner.invoke(cli, ["ref", "optimize", "--help"])
    assert result.exit_code == 0


def test_ref_optimize_rejects_station_datasets(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package

    root = tmp_path / "station"
    root.mkdir()

    class FakeRegistryManager:
        def get_reference(self, name):
            return SimpleNamespace(name=name, root_dir=str(root), data_type="stn")

    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["ref", "optimize", "StationRef"])

    assert result.exit_code != 0
    assert "station datasets" in result.output


def test_ref_optimize_preserves_existing_zarr_when_conversion_fails(tmp_path, monkeypatch):
    import openbench.data.coordinates as coordinates
    import openbench.data.registry as registry_package
    import openbench.util.dataset_loader as dataset_loader

    root = tmp_path / "grid"
    root.mkdir()
    nc_file = root / "sample.nc"
    nc_file.write_text("placeholder")
    zarr_dir = tmp_path / "grid.zarr"
    zarr_dir.mkdir()
    sentinel = zarr_dir / "sentinel"
    sentinel.write_text("old-ok")

    class FakeRegistryManager:
        def get_reference(self, name):
            return SimpleNamespace(name=name, root_dir=str(root), data_type="grid")

    def failing_write_mfdataset_zarr(_nc_files, zarr_path, **_kwargs):
        Path(zarr_path).mkdir(parents=True)
        (Path(zarr_path) / "partial").write_text("bad")
        raise RuntimeError("boom")

    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(coordinates, "glob_nc", lambda root, recursive=True: [nc_file])
    monkeypatch.setattr(dataset_loader, "write_mfdataset_zarr", failing_write_mfdataset_zarr)

    result = runner.invoke(cli, ["ref", "optimize", "GridRef"], input="y\n")

    assert result.exit_code != 0
    assert "Conversion failed" in result.output
    assert sentinel.read_text() == "old-ok"
    assert not list(tmp_path.glob(".grid.zarr.tmp-*"))


def test_generate_station_list_rejects_file_input_before_scanning(tmp_path):
    dataset_file = tmp_path / "station.nc"
    dataset_file.write_text("not a netcdf")

    result = runner.invoke(cli, ["ref", "generate-station-list", str(dataset_file)])

    assert result.exit_code != 0
    assert "must be a directory" in result.output
    assert "No NC files found" not in result.output


def test_sim_help():
    result = runner.invoke(cli, ["sim", "--help"])
    assert result.exit_code == 0
    assert "Manage simulation outputs" in result.output


def test_sim_scan_help():
    result = runner.invoke(cli, ["sim", "scan", "--help"])
    assert result.exit_code == 0
    assert "Scan directories for simulation cases" in result.output
    assert "--case-depth" in result.output


def test_data_namespace_is_not_registered():
    result = runner.invoke(cli, ["data", "--help"])
    assert result.exit_code != 0
    assert "No such command 'data'" in result.output


def test_model_list_help():
    result = runner.invoke(cli, ["model", "list", "--help"])
    assert result.exit_code == 0


def test_model_show_help():
    result = runner.invoke(cli, ["model", "show", "--help"])
    assert result.exit_code == 0


def test_model_register_help():
    result = runner.invoke(cli, ["model", "register", "--help"])
    assert result.exit_code == 0


def test_register_help_examples_are_not_wrapped_into_one_line():
    for command in (
        ["ref", "register"],
        ["ref", "register-profile"],
        ["ref", "scan"],
        ["model", "register"],
    ):
        result = runner.invoke(cli, [*command, "--help"], terminal_width=88)
        assert result.exit_code == 0
        assert "Examples:\n  openbench" in result.output
        assert "Examples:     openbench" not in result.output
        assert "\\   -" not in result.output

    scan_result = runner.invoke(cli, ["ref", "scan", "--help"], terminal_width=88)
    assert "Examples:\n  openbench ref scan /Volumes/work/Reference" in scan_result.output
    assert "\n  openbench ref scan /Volumes/work/Reference --dry-run" in scan_result.output
    assert "\n  openbench ref scan /Volumes/work/Reference --rescan --auto" in scan_result.output


def test_model_register_updates_existing_builtin_model_via_user_overlay(tmp_path, monkeypatch):
    import openbench.data.registry.manager as manager_module
    from openbench.data.registry.schema import ModelProfile, VariableMapping

    home = tmp_path / "home"
    existing = ModelProfile(
        name="BuiltinModel",
        description="Built-in model",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_model(self, name):
            return existing if name == "BuiltinModel" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(manager_module, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(
        cli,
        ["model", "register", "BuiltinModel", "-v", "Evapotranspiration:ET:mm day-1"],
    )

    assert result.exit_code == 0, result.output
    user_catalog = home / ".openbench" / "models" / "model_catalog.yaml"
    descriptor = yaml.safe_load(user_catalog.read_text())["BuiltinModel"]
    assert descriptor["variables"]["Runoff"]["varname"] == "ro"
    assert descriptor["variables"]["Evapotranspiration"]["varname"] == "ET"


def test_model_register_reports_metadata_only_update(tmp_path, monkeypatch):
    import openbench.data.registry.manager as manager_module
    from openbench.data.registry.schema import ModelProfile, VariableMapping

    home = tmp_path / "home"
    existing = ModelProfile(
        name="BuiltinModel",
        description="Built-in model",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_model(self, name):
            return existing if name == "BuiltinModel" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(manager_module, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["model", "register", "BuiltinModel", "--tim-res", "Day"])

    assert result.exit_code == 0, result.output
    assert "metadata updated" in result.output


def test_model_register_accepts_climatology_tim_res(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ManualModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "climatology-year",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "models" / "model_catalog.yaml").read_text())["ManualModel"]
    assert descriptor["tim_res"] == "climatology-year"


def test_model_register_accepts_scanner_tim_res_values(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ManualModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "3month",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "models" / "model_catalog.yaml").read_text())["ManualModel"]
    assert descriptor["tim_res"] == "3month"


def test_model_register_preserves_canonical_profile_name_when_input_case_differs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(cli, ["model", "register", "colm2024", "-v", "Snow_Depth:sd:m"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "models" / "model_catalog.yaml").read_text())["CoLM2024"]
    assert descriptor["name"] == "CoLM2024"
    assert descriptor["variables"]["Snow_Depth"]["varname"] == "sd"


def test_model_register_handles_null_variables_in_catalog(tmp_path, monkeypatch):
    home = tmp_path / "home"
    catalog = home / ".openbench" / "models" / "model_catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "NullVars:\n  name: NullVars\n  description: demo\n  data_type: grid\n  tim_res: Month\n  variables: null\n"
    )

    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(cli, ["model", "register", "NullVars", "-v", "Runoff:ro:mm day-1"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(catalog.read_text())["NullVars"]
    assert descriptor["variables"]["Runoff"]["varname"] == "ro"


def test_model_register_cancels_new_profile_with_no_variables(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "EmptyModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
        ],
        input="\n\n\n",
    )

    assert result.exit_code == 0, result.output
    assert "No variables defined" in result.output
    catalog = home / ".openbench" / "models" / "model_catalog.yaml"
    assert not catalog.exists() or "EmptyModel" not in (yaml.safe_load(catalog.read_text()) or {})


def test_model_register_merges_against_latest_catalog_under_write_lock(tmp_path, monkeypatch):
    import copy
    from contextlib import nullcontext

    import openbench.data.registry.manager as manager_module
    import openbench.data.registry.scanner as scanner_module

    catalog_path = tmp_path / "model_catalog.yaml"
    stale_catalog = {
        "RaceModel": {
            "name": "RaceModel",
            "description": "demo",
            "data_type": "grid",
            "tim_res": "Month",
            "variables": {"V1": {"varname": "v1", "varunit": ""}},
        }
    }
    latest_catalog = copy.deepcopy(stale_catalog)
    latest_catalog["RaceModel"]["variables"]["V4"] = {"varname": "v4", "varunit": ""}
    loads = [stale_catalog, latest_catalog]
    written = {}

    monkeypatch.setattr(manager_module, "get_writable_model_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(scanner_module, "_safe_load_catalog", lambda path: copy.deepcopy(loads.pop(0)))
    monkeypatch.setattr(scanner_module, "_catalog_write_lock", lambda path: nullcontext())
    monkeypatch.setattr(
        scanner_module,
        "_backup_then_write",
        lambda path, catalog: written.update({"catalog": copy.deepcopy(catalog)}),
    )
    monkeypatch.setattr(scanner_module, "_invalidate_registry_caches", lambda: None)

    result = runner.invoke(cli, ["model", "register", "RaceModel", "-v", "V3:v3:mm day-1"])

    assert result.exit_code == 0, result.output
    variables = written["catalog"]["RaceModel"]["variables"]
    assert set(variables) == {"V1", "V3", "V4"}


def test_model_register_uses_canonical_name_in_default_description(tmp_path, monkeypatch):
    import openbench.data.registry.manager as manager_module
    from openbench.data.registry.schema import ModelProfile

    home = tmp_path / "home"
    existing = ModelProfile(
        name="CanonicalModel",
        description="",
        data_type="grid",
        tim_res="Month",
        variables={},
    )
    existing_dict = existing.to_dict()
    existing_dict.pop("description")

    class FakeRegistryManager:
        def get_model(self, name):
            if name == "alias":
                return SimpleNamespace(name="CanonicalModel", to_dict=lambda: existing_dict)
            return None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(manager_module, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["model", "register", "alias", "-v", "Runoff:ro:mm day-1"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "models" / "model_catalog.yaml").read_text())["CanonicalModel"]
    assert "description" not in descriptor


def test_ref_register_preserves_canonical_reference_name_when_input_case_differs(tmp_path, monkeypatch):
    import openbench.data.registry as registry_package
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    home = tmp_path / "home"
    existing = ReferenceDataset(
        name="CanonicalRef",
        description="Built-in descriptor",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2001, 2002],
        root_dir="/builtin/root",
        grid_res=0.5,
        variables={"Runoff": VariableMapping(varname="ro", varunit="mm day-1")},
    )

    class FakeRegistryManager:
        def get_reference(self, name):
            return existing if name.lower() == "canonicalref" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["ref", "register", "canonicalref", "-v", "Snow_Depth:sd:m"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "references" / "reference_catalog.yaml").read_text())[
        "CanonicalRef"
    ]
    assert descriptor["name"] == "CanonicalRef"
    assert descriptor["variables"]["Snow_Depth"]["varname"] == "sd"


def test_model_remove_var_help():
    result = runner.invoke(cli, ["model", "remove-var", "--help"])
    assert result.exit_code == 0


def test_model_remove_var_can_create_overlay_from_builtin_profile(tmp_path, monkeypatch):
    import openbench.data.registry.manager as manager_module
    from openbench.data.registry.schema import ModelProfile, VariableMapping

    home = tmp_path / "home"
    existing = ModelProfile(
        name="BuiltinModel",
        description="Built-in model",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        variables={
            "Runoff": VariableMapping(varname="ro", varunit="mm day-1"),
            "Evapotranspiration": VariableMapping(varname="et", varunit="mm day-1"),
        },
    )

    class FakeRegistryManager:
        def get_model(self, name):
            return existing if name == "BuiltinModel" else None

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(manager_module, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(cli, ["model", "remove-var", "BuiltinModel", "Runoff"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load((home / ".openbench" / "models" / "model_catalog.yaml").read_text())["BuiltinModel"]
    assert "Runoff" not in descriptor["variables"]
    assert "Evapotranspiration" in descriptor["variables"]


def test_migrate_help():
    result = runner.invoke(cli, ["migrate", "--help"])
    assert result.exit_code == 0


def test_migrate_rejects_directory_input_before_migration(tmp_path):
    config_dir = tmp_path / "old_config"
    config_dir.mkdir()

    result = runner.invoke(cli, ["migrate", str(config_dir), "-o", str(tmp_path / "out.yaml")])

    assert result.exit_code != 0
    assert "file" in result.output.lower()


def test_init_help():
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "--no-interactive" not in result.output


def test_init_no_interactive_is_explicitly_rejected():
    result = runner.invoke(cli, ["init", "--no-interactive"])

    assert result.exit_code != 0
    assert "--no-interactive" in result.output
    assert "not implemented" in result.output.lower()


def test_init_registry_overlays_creates_empty_catalog_overlays(tmp_path):
    import openbench.data.custom as custom_package
    from openbench.cli.init_cmd import ensure_user_registry_overlays

    ensure_user_registry_overlays(tmp_path)

    expected_empty_overlays = [
        tmp_path / "references" / "reference_catalog.yaml",
        tmp_path / "references" / "reference_profiles.yaml",
        tmp_path / "models" / "model_catalog.yaml",
    ]
    for target in expected_empty_overlays:
        assert target.exists()
        assert yaml.safe_load(target.read_text()) == {}

    custom_sources = sorted(Path(custom_package.__file__).parent.glob("*_filter.py"))
    assert custom_sources
    for source in custom_sources:
        target = tmp_path / "custom" / source.name
        assert target.exists()
        assert target.read_text() == source.read_text()

    assert (tmp_path / "custom").is_dir()


def test_copy_default_file_if_missing_preserves_seeded_target_when_source_is_removed(tmp_path):
    import openbench.cli.init_cmd as init_module

    base_dir = tmp_path / "user"
    target = base_dir / "custom" / "Old_filter.py"
    source = tmp_path / "package" / "Old_filter.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")
    manifest = {
        "custom/Old_filter.py": {
            "sha256": init_module._file_sha256(target),
        }
    }

    init_module._copy_default_file_if_missing(source, target, base_dir, manifest)

    assert target.read_text() == "VALUE = 1\n"
    assert "custom/Old_filter.py" not in manifest


def test_init_registry_overlays_do_not_shadow_builtin_catalog_updates(tmp_path, monkeypatch):
    import openbench.data.registry.manager as manager_module
    from openbench.cli.init_cmd import ensure_user_registry_overlays

    def descriptor(tim_res):
        return {
            "Foo": {
                "name": "Foo",
                "description": "Foo reference",
                "category": "Water",
                "data_type": "grid",
                "tim_res": tim_res,
                "data_groupby": "Year",
                "timezone": 0,
                "years": [2000, 2001],
                "grid_res": 0.5,
                "variables": {"Runoff": {"varname": "ro", "varunit": "mm day-1"}},
            }
        }

    package_registry = tmp_path / "package-registry"
    package_registry.mkdir()
    (package_registry / "reference_catalog.yaml").write_text(yaml.safe_dump(descriptor("Month")))
    (package_registry / "reference_profiles.yaml").write_text("{}\n")
    (package_registry / "model_catalog.yaml").write_text("{}\n")
    monkeypatch.setattr(manager_module, "REGISTRY_DIR", package_registry)

    user_dir = tmp_path / "user"
    ensure_user_registry_overlays(user_dir)
    user_catalog = user_dir / "references" / "reference_catalog.yaml"
    user_data = yaml.safe_load(user_catalog.read_text()) or {}
    user_data["UserOnly"] = {
        **descriptor("Day")["Foo"],
        "name": "UserOnly",
        "description": "User-only reference",
    }
    user_catalog.write_text(yaml.safe_dump(user_data))

    (package_registry / "reference_catalog.yaml").write_text(yaml.safe_dump(descriptor("Day")))
    ensure_user_registry_overlays(user_dir)

    assert "Foo" not in (yaml.safe_load(user_catalog.read_text()) or {})
    mgr = manager_module.RegistryManager(user_dir=user_dir)
    assert mgr.get_reference("Foo").tim_res == "Day"
    assert mgr.get_reference("UserOnly").tim_res == "Day"


def test_init_registry_overlays_preserves_existing_user_files(tmp_path):
    from openbench.cli.init_cmd import ensure_user_registry_overlays

    reference_catalog = tmp_path / "references" / "reference_catalog.yaml"
    reference_profiles = tmp_path / "references" / "reference_profiles.yaml"
    model_catalog = tmp_path / "models" / "model_catalog.yaml"
    reference_catalog.parent.mkdir(parents=True)
    model_catalog.parent.mkdir(parents=True)
    reference_catalog.write_text("UserReference: {}\n")
    reference_profiles.write_text("UserProfile: {}\n")
    model_catalog.write_text("UserModel: {}\n")

    ensure_user_registry_overlays(tmp_path)

    assert reference_catalog.read_text() == "UserReference: {}\n"
    assert reference_profiles.read_text() == "UserProfile: {}\n"
    assert model_catalog.read_text() == "UserModel: {}\n"
    assert (tmp_path / "custom").is_dir()


def test_init_registry_overlays_converts_untouched_seeded_catalogs_to_empty_overlays(tmp_path):
    from openbench.cli.init_cmd import _file_sha256, ensure_user_registry_overlays

    user_dir = tmp_path / "user"
    reference_catalog = user_dir / "references" / "reference_catalog.yaml"
    reference_profiles = user_dir / "references" / "reference_profiles.yaml"
    model_catalog = user_dir / "models" / "model_catalog.yaml"
    reference_catalog.parent.mkdir(parents=True)
    model_catalog.parent.mkdir(parents=True)

    reference_catalog.write_text("BuiltInReferenceV1: {}\n")
    reference_profiles.write_text("BuiltInProfileV1: {}\n")
    model_catalog.write_text("BuiltInModelV1: {}\n")
    manifest = {
        "references/reference_catalog.yaml": {"sha256": _file_sha256(reference_catalog)},
        "references/reference_profiles.yaml": {"sha256": _file_sha256(reference_profiles)},
        "models/model_catalog.yaml": {"sha256": _file_sha256(model_catalog)},
    }
    (user_dir / ".seeded_defaults.yaml").write_text(yaml.safe_dump(manifest))

    ensure_user_registry_overlays(user_dir)

    assert yaml.safe_load(reference_catalog.read_text()) == {}
    assert yaml.safe_load(reference_profiles.read_text()) == {}
    assert yaml.safe_load(model_catalog.read_text()) == {}


def test_init_registry_overlays_preserves_modified_seeded_files(tmp_path, monkeypatch):
    import openbench.data.registry.manager as manager_module
    from openbench.cli.init_cmd import ensure_user_registry_overlays

    package_registry = tmp_path / "package-registry"
    package_registry.mkdir()
    (package_registry / "reference_catalog.yaml").write_text("BuiltInReferenceV1: {}\n")
    (package_registry / "reference_profiles.yaml").write_text("BuiltInProfileV1: {}\n")
    (package_registry / "model_catalog.yaml").write_text("BuiltInModelV1: {}\n")
    monkeypatch.setattr(manager_module, "REGISTRY_DIR", package_registry)

    user_dir = tmp_path / "user"
    ensure_user_registry_overlays(user_dir)
    user_reference = user_dir / "references" / "reference_catalog.yaml"
    user_reference.write_text("UserReferenceEdit: {}\n")
    (package_registry / "reference_catalog.yaml").write_text("BuiltInReferenceV2: {}\n")

    ensure_user_registry_overlays(user_dir)

    assert user_reference.read_text() == "UserReferenceEdit: {}\n"


def test_init_command_initializes_user_registry_overlays(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    called = []

    def fake_ensure_user_registry_overlays():
        called.append(True)

    monkeypatch.setattr(
        init_module,
        "ensure_user_registry_overlays",
        fake_ensure_user_registry_overlays,
    )
    _install_single_reference_registry(monkeypatch)

    output = tmp_path / "openbench.yaml"
    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n\n\n\n\n\n\n\n",
    )

    assert result.exit_code == 0
    assert called == [True]
    assert output.exists()


def test_init_uses_timestamped_default_output(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(
        init_module,
        "_default_init_output_path",
        lambda: Path("openbench_init_20260501-140305.yaml"),
        raising=False,
    )
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(cli, ["init", "--no-ref-check"], input="\n\n\n\n\n\n\n\n\n")

    assert result.exit_code == 0
    assert (tmp_path / "openbench_init_20260501-140305.yaml").exists()
    assert not (tmp_path / "openbench.yaml").exists()


def test_init_requires_reference_root_on_fresh_empty_overlay(tmp_path, monkeypatch):
    output = tmp_path / "openbench.yaml"
    home = tmp_path / "home"

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    result = runner.invoke(cli, ["init", "-o", str(output)], input="\n" * 300)

    assert result.exit_code != 0
    assert "Reference catalog is missing or empty" in result.output
    assert "Reference data root" in result.output
    assert not output.exists()


def test_init_scans_simulation_roots_into_generated_config(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        name = "GLEAM"
        category = "Water"
        data_type = "grid"
        tim_res = "Month"
        variables = {"Evapotranspiration": {}}

    class FakeRegistryManager:
        def list_references(self):
            return [FakeReference()]

        def references_for_variable(self, variable):
            return [FakeReference()] if variable == "Evapotranspiration" else []

        def list_models(self):
            return []

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "CoLM2024"
    case_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"
    calls = []

    def fake_scan_simulation_roots(roots, **kwargs):
        calls.append((roots, kwargs))
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="CoLM2024",
                    root_dir=case_root,
                    model="CoLM",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=1.875,
                    data_groupby="Month",
                    prefix="Global_",
                    suffix="_monthly",
                    variable_overrides={
                        "Evapotranspiration": {
                            "prefix": "Global_QFLX_EVAP_TOT_",
                            "suffix": "_monthly",
                            "varname": "QFLX_EVAP_TOT",
                        },
                    },
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n1996\n1996\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            [str(sim_root)],
            {
                "model_name": "auto",
                "case_depth": 5,
                "case_pattern": None,
                "exclude": (),
                "climatology": "auto",
            },
        )
    ]
    config = yaml.safe_load(output.read_text())
    entry = config["simulation"]["CoLM2024"]
    assert entry["model"] == "CoLM"
    assert entry["root_dir"] == str(case_root)
    assert entry["tim_res"] == "Month"
    assert entry["grid_res"] == 1.875
    assert entry["data_groupby"] == "Month"
    # Case-level prefix is suppressed when variable overrides expose a different
    # stream pattern, so unmapped variables don't inherit the wrong prefix.
    assert "prefix" not in entry
    assert "suffix" not in entry
    assert entry["variables"]["Evapotranspiration"] == {
        "prefix": "Global_QFLX_EVAP_TOT_",
        "suffix": "_monthly",
        "varname": "QFLX_EVAP_TOT",
    }


def test_init_passes_explicit_sim_model_to_scan(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        name = "FLUXCOM_LowRes"
        category = "Energy"
        data_type = "grid"
        tim_res = "Month"
        grid_res = 0.5
        variables = {"Latent_Heat": {}}

    class FakeRegistryManager:
        def list_references(self):
            return [FakeReference()]

        def references_for_variable(self, variable):
            return [FakeReference()] if variable == "Latent_Heat" else []

        def list_models(self):
            return []

    sim_root = tmp_path / "Initial_test"
    grid_root = sim_root / "grid"
    grid_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"
    calls = []

    def fake_scan_simulation_roots(roots, **kwargs):
        calls.append((roots, kwargs))
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="grid",
                    root_dir=grid_root,
                    model="CoLM",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=2.0,
                    data_groupby="Month",
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        [
            "init",
            "--no-ref-check",
            "--sim-root",
            str(sim_root),
            "--sim-model",
            "CoLM",
            "-o",
            str(output),
        ],
        input="\n\n2004\n2005\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert calls[0][1]["model_name"] == "CoLM"
    config = yaml.safe_load(output.read_text())
    assert config["simulation"]["grid"]["model"] == "CoLM"


def test_init_caps_min_year_threshold_to_project_span(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        name = "FLUXCOM_LowRes"
        category = "Energy"
        data_type = "grid"
        tim_res = "Month"
        grid_res = 0.5
        variables = {"Latent_Heat": {}}

    class FakeRegistryManager:
        def list_references(self):
            return [FakeReference()]

        def references_for_variable(self, variable):
            return [FakeReference()] if variable == "Latent_Heat" else []

        def list_models(self):
            return []

    sim_root = tmp_path / "Initial_test"
    stn_root = sim_root / "stn"
    stn_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="stn",
                    root_dir=stn_root,
                    model="CoLM",
                    depth=1,
                    data_type="stn",
                    tim_res="Day",
                    data_groupby="Single",
                    fulllist=tmp_path / "stations.csv",
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        [
            "init",
            "--no-ref-check",
            "--sim-root",
            str(sim_root),
            "--sim-model",
            "CoLM",
            "-o",
            str(output),
        ],
        input="\n\n2004\n2005\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    project = yaml.safe_load(output.read_text())["project"]
    assert project["min_year_threshold"] == 2


def test_init_sets_project_resolution_from_selected_reference_when_sim_scan_is_mixed(
    tmp_path,
    monkeypatch,
):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        name = "GLEAM_v4.2a_MidRes"
        category = "Water"
        data_type = "grid"
        tim_res = "Month"
        grid_res = 0.25
        variables = {"Evapotranspiration": {}}

    class FakeRegistryManager:
        def list_references(self):
            return [FakeReference()]

        def references_for_variable(self, variable):
            return [FakeReference()] if variable == "Evapotranspiration" else []

        def list_models(self):
            return []

    sim_root = tmp_path / "Simulation"
    sim_root.mkdir()
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="Coarse",
                    root_dir=sim_root / "Coarse",
                    model="M1",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.5,
                    data_groupby="Month",
                ),
                SimulationCase(
                    label="Fine",
                    root_dir=sim_root / "Fine",
                    model="M2",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.25,
                    data_groupby="Month",
                ),
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n1996\n1996\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    project = yaml.safe_load(output.read_text())["project"]
    assert project["tim_res"] == "Month"
    assert project["grid_res"] == 0.25


def test_init_writes_loadable_yaml_with_commented_template_options(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.config import load_config
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        def __init__(self, name, variable, category="Energy", tim_res="Month", grid_res=0.5):
            self.name = name
            self.category = category
            self.data_type = "grid"
            self.tim_res = tim_res
            self.grid_res = grid_res
            self.variables = {variable: {}}

    refs = [
        FakeReference("FLUXCOM_LowRes", "Latent_Heat"),
        FakeReference("ERA5LAND_LowRes", "Latent_Heat"),
        FakeReference("GLEAM_v4.2a_MidRes", "Evapotranspiration", category="Water", grid_res=0.25),
    ]

    class FakeRegistryManager:
        def list_references(self):
            return refs

        def references_for_variable(self, variable):
            return [ref for ref in refs if variable in ref.variables]

        def list_models(self):
            return []

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="Case01",
                    root_dir=case_root,
                    model="CoLM",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.5,
                    data_groupby="Month",
                    prefix="Case01_hist_",
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n2004\n2005\n1\n1\n\nn\n",
    )

    assert result.exit_code == 0, result.output
    text = output.read_text()
    cfg = load_config(output)

    assert cfg.evaluation.variables == ["Latent_Heat"]
    assert cfg.reference.sources["Latent_Heat"] == "FLUXCOM_LowRes"
    assert cfg.metrics == ["bias", "RMSE", "correlation"]
    assert cfg.scores == ["Overall_Score"]
    assert cfg.comparison.items == ["Taylor_Diagram", "HeatMap"]

    assert "# lat_range: [-90.0, 90.0]" in text
    assert "# - Evapotranspiration" in text
    assert "# Latent_Heat: ERA5LAND_LowRes" in text
    assert "# - KGE" in text
    assert "# - nBiasScore" in text
    assert "# - Target_Diagram" in text
    assert "# items:" in text
    assert "# suffix:" in text
    assert "# fulllist: /path/to/station_list.csv" in text


def test_init_deduplicates_variables_that_appear_in_multiple_categories(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        def __init__(self, name, variable, category):
            self.name = name
            self.category = category
            self.data_type = "grid"
            self.tim_res = "Month"
            self.grid_res = 0.5
            self.variables = {variable: {}}

    refs = [
        FakeReference("EnergyLE", "Latent_Heat", "Energy"),
        FakeReference("HeatLE", "Latent_Heat", "Heat"),
        FakeReference("RunoffRef", "Runoff", "Water"),
    ]

    class FakeRegistryManager:
        def list_references(self):
            return refs

        def references_for_variable(self, variable):
            if variable == "Latent_Heat":
                return [refs[0]]
            if variable == "Runoff":
                return [refs[2]]
            return []

        def list_models(self):
            return []

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="Case01",
                    root_dir=case_root,
                    model="CoLM",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.5,
                    data_groupby="Month",
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n2004\n2005\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text())
    assert config["evaluation"]["variables"] == ["Latent_Heat", "Runoff"]
    assert result.output.count("Latent_Heat") == 2  # variable list + reference line


def test_init_rejects_out_of_range_reference_choice(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package

    refs = [
        _InitFakeReference(name="FirstRef", variable="Latent_Heat"),
        _InitFakeReference(name="SecondRef", variable="Latent_Heat"),
    ]

    class FakeRegistryManager:
        def list_references(self):
            return refs

        def references_for_variable(self, variable):
            return refs if variable == "Latent_Heat" else []

        def list_models(self):
            return []

    output = tmp_path / "openbench.yaml"
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n2004\n2004\n1\n99\n",
    )

    assert result.exit_code != 0
    assert "out of range" in result.output.lower()
    assert not output.exists()


def test_init_accepts_variable_and_reference_names(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    refs = [
        _InitFakeReference(
            name="FLUXCOM_LowRes",
            variable="Latent_Heat",
            grid_res=0.5,
            years=[2001, 2020],
            root_dir="/refs/fluxcom",
        ),
        _InitFakeReference(
            name="ERA5LAND_LowRes",
            variable="Latent_Heat",
            grid_res=0.25,
            years=[1981, 2022],
            root_dir="/refs/era5land",
        ),
    ]
    output = tmp_path / "openbench.yaml"

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch, refs=refs)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n2004\n2004\nlatent_heat\nera5land_lowres\n\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text())
    assert config["evaluation"]["variables"] == ["Latent_Heat"]
    assert config["reference"]["Latent_Heat"] == "ERA5LAND_LowRes"
    assert "/refs/era5land" in result.output
    assert "1981-2022" in result.output


def test_init_reference_choice_zero_skips_variable(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    refs = [
        _InitFakeReference(name="EnergyRef", variable="Latent_Heat"),
        _InitFakeReference(name="RunoffRef", variable="Runoff", category="Water"),
        _InitFakeReference(name="AltEnergyRef", variable="Latent_Heat"),
    ]
    output = tmp_path / "openbench.yaml"

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch, refs=refs)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n2004\n2004\nall\n0\n\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text())
    assert config["evaluation"]["variables"] == ["Runoff"]
    assert config["reference"] == {"Runoff": "RunoffRef"}


def test_init_aborts_when_all_selected_variables_have_no_reference(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package

    refs = [_InitFakeReference(name="BrokenIndexRef", variable="Latent_Heat")]
    output = tmp_path / "openbench.yaml"

    class FakeRegistryManager:
        def list_references(self):
            return refs

        def references_for_variable(self, variable):
            return []

        def list_models(self):
            return []

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n2004\n2004\nall\n",
    )

    assert result.exit_code != 0
    assert "No reference data selected" in result.output
    assert not output.exists()


def test_init_reloads_reference_status_after_overlay_creation(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    home = tmp_path / "home"
    catalog = home / ".openbench" / "references" / "reference_catalog.yaml"
    captured = []

    def fake_ensure_user_registry_overlays():
        catalog.parent.mkdir(parents=True)
        catalog.write_text("SeededReference: {}\n")
        return home / ".openbench"

    def fake_preflight(status, **kwargs):
        captured.append(status)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", fake_ensure_user_registry_overlays)
    monkeypatch.setattr(init_module, "_init_reference_registry_preflight", fake_preflight)
    _install_single_reference_registry(monkeypatch)

    output = tmp_path / "openbench.yaml"
    result = runner.invoke(cli, ["init", "-o", str(output)], input="\n\n2004\n2004\n\n\n\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert captured
    assert captured[0].exists is True
    assert captured[0].empty is False
    assert captured[0].catalog_path == catalog


def test_init_writes_default_statistics_items_when_enabled(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="Case01",
                    root_dir=case_root,
                    model="CoLM",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.5,
                    data_groupby="Month",
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n2004\n2004\n\nn\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(output.read_text())["statistics"] == {
        "enabled": True,
        "items": ["Mean", "Median", "Min", "Max", "Sum"],
    }


def test_init_warns_when_reference_or_simulation_years_do_not_overlap_project_years(
    tmp_path,
    monkeypatch,
):
    import openbench.cli.init_cmd as init_module
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"
    refs = [
        _InitFakeReference(
            name="OldRef",
            variable="Latent_Heat",
            years=[1980, 1985],
        )
    ]

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="Case01",
                    root_dir=case_root,
                    model="CoLM",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.5,
                    data_groupby="Month",
                    years=[1990, 1991],
                )
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch, refs=refs)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n2004\n2004\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "OldRef" in result.output
    assert "Case01" in result.output
    assert "outside project years 2004-2004" in result.output
    assert output.exists()


def test_init_rejects_end_year_before_start_year(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(tmp_path / "openbench.yaml")],
        input="\n\n2010\n2004\n",
    )

    assert result.exit_code != 0
    assert "Start year must be <= end year" in result.output


def test_init_rejects_project_name_path_before_writing(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    output = tmp_path / "openbench.yaml"
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="../bad\n",
    )

    assert result.exit_code != 0
    assert "project.name must be a simple directory name" in result.output
    assert not output.exists()


def test_init_no_ref_check_allows_template_when_reference_registry_has_no_variables(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch, refs=[])
    output = tmp_path / "openbench.yaml"

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n2004\n2004\nLatent_Heat, Runoff\n\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text())
    assert config["evaluation"]["variables"] == ["Latent_Heat", "Runoff"]
    assert config["reference"] == {}
    assert "# Latent_Heat: <reference_dataset>" in output.read_text()
    assert "Fill in reference placeholders" in result.output
    assert "Next: openbench check" not in result.output


def test_init_rejects_unresolved_environment_variable_in_project_output_dir(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBENCH_MISSING_PROJECT_OUT", raising=False)
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)
    output = tmp_path / "openbench.yaml"

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n$OPENBENCH_MISSING_PROJECT_OUT/output\n",
    )

    assert result.exit_code != 0
    assert "project.output_dir contains unresolved environment variable" in result.output
    assert not output.exists()


def test_init_prompts_for_missing_grid_resolution_when_simulation_scan_is_mixed(
    tmp_path,
    monkeypatch,
):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry as registry_package
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    class FakeReference:
        name = "StationRef"
        category = "Water"
        data_type = "stn"
        tim_res = "Month"
        grid_res = None
        variables = {"Runoff": {}}

    class FakeRegistryManager:
        def list_references(self):
            return [FakeReference()]

        def references_for_variable(self, variable):
            return [FakeReference()] if variable == "Runoff" else []

        def list_models(self):
            return []

    sim_root = tmp_path / "Simulation"
    sim_root.mkdir()
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[
                SimulationCase(
                    label="Coarse",
                    root_dir=sim_root / "Coarse",
                    model="M1",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.5,
                    data_groupby="Month",
                ),
                SimulationCase(
                    label="Fine",
                    root_dir=sim_root / "Fine",
                    model="M2",
                    depth=1,
                    data_type="grid",
                    tim_res="Month",
                    grid_res=0.25,
                    data_groupby="Month",
                ),
            ],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr(registry_package, "RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n2004\n2004\n\nn\nn\n0.5\n",
    )

    assert result.exit_code == 0, result.output
    project = yaml.safe_load(output.read_text())["project"]
    assert project["tim_res"] == "Month"
    assert project["grid_res"] == 0.5
    assert "Target grid_res" in result.output


def test_init_metric_and_score_options_are_implemented():
    import openbench.cli.init_cmd as init_module
    from openbench.core.metrics import metrics
    from openbench.core.scores import scores

    missing_metrics = [name for name in init_module.METRIC_OPTIONS if not hasattr(metrics, name)]
    missing_scores = [name for name in init_module.SCORE_OPTIONS if not hasattr(scores, name)]
    disabled_metrics = {"rSD", "PBIAS_HF", "PBIAS_LF"} & set(init_module.METRIC_OPTIONS)

    assert missing_metrics == []
    assert missing_scores == []
    assert disabled_metrics == set()


def test_run_basic_comparison_uses_basic_figure_options(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module
    import openbench.runner.local as local_runner

    (tmp_path / "metrics").mkdir()
    _write_fake_cli_netcdf(tmp_path / "metrics" / "Runoff_ref_TestRef_sim_TestSim_bias.nc")
    seen_options = []

    class FakeComparisonProcessing:
        def __init__(self, main, scores, metrics):
            pass

        def scenarios_Basic_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
            seen_options.append(dict(option))

    class FakeBindings:
        def build_comparison_context(self):
            return SimpleNamespace(
                namelists=SimpleNamespace(main={}, simulation={}, reference={}),
                evaluation_items=["Runoff"],
                score_vars=[],
                metric_vars=["bias"],
                comparison_fig={"Basic": {"cmap": "viridis"}},
            )

    monkeypatch.setattr(comparison_module, "ComparisonProcessing", FakeComparisonProcessing)

    errors = local_runner._run_comparison(FakeBindings(), ["Mean"], tmp_path)

    assert errors == []
    assert seen_options == [{"cmap": "viridis", "key": "Mean"}]


def test_run_basic_statistics_uses_basic_figure_options(monkeypatch):
    import openbench.core.statistics.Mod_Statistics as statistics_module
    import openbench.runner.local as local_runner

    seen_options = []

    class FakeStatisticsProcessing:
        def __init__(self, main_nl, stats_nml, stats_dir, num_cores=1):
            pass

        def scenarios_Basic_analysis(self, statistic, statistic_nml, option):
            seen_options.append((statistic, dict(option)))

    class FakeBindings:
        def build_statistics_context(self, statistic_vars):
            return SimpleNamespace(
                namelists=SimpleNamespace(main={"general": {}}, simulation={}, reference={}),
                stats_dir="/tmp/openbench-stat-test",
                stats_nml={"general": {}, "Mean": {}},
                num_cores=1,
                statistic_fig={"Basic": {"cmap": "viridis"}},
            )

    monkeypatch.setattr(statistics_module, "StatisticsProcessing", FakeStatisticsProcessing)

    errors = local_runner._run_statistics(FakeBindings(), ["Mean"])

    assert errors == []
    assert seen_options == [("Mean", {"cmap": "viridis"})]


def test_init_keeps_all_variables_in_template_when_some_are_skipped(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    refs = [
        _InitFakeReference(name="GoodRef", variable="Latent_Heat"),
        _InitFakeReference(name="BrokenIndexRef", variable="Runoff", category="Water"),
    ]

    class FakeRegistryManager:
        def list_references(self):
            return refs

        def references_for_variable(self, variable):
            return [refs[0]] if variable == "Latent_Heat" else []

        def list_models(self):
            return []

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    output = tmp_path / "openbench.yaml"

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(
            roots=[sim_root],
            cases=[SimulationCase(label="Case01", root_dir=case_root, model="CoLM", depth=1)],
        )

    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    monkeypatch.setattr("openbench.data.registry.RegistryManager", FakeRegistryManager)
    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "--sim-root", str(sim_root), "-o", str(output)],
        input="\n\n2004\n2004\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(output.read_text())["evaluation"]["variables"] == ["Latent_Heat"]
    assert "# - Runoff" in output.read_text()


def test_init_reference_preflight_missing_catalog_runs_scan_then_registers(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    catalog_path = tmp_path / "user" / "references" / "reference_catalog.yaml"
    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    scanned = [object()]
    calls = []
    home = tmp_path / "home"

    status = init_module.ReferenceCatalogStatus(
        catalog_path=catalog_path,
        exists=False,
        empty=True,
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)
    monkeypatch.setattr(init_module, "_resolve_reference_root", lambda value: ref_root)
    monkeypatch.setattr(
        init_module,
        "_scan_reference_variants",
        lambda root: calls.append(("scan", root)) or scanned,
    )
    monkeypatch.setattr(
        init_module,
        "_register_reference_variants",
        lambda variants, path, **kwargs: calls.append(("register", variants, path, kwargs.get("ref_root"))),
    )
    monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: True)

    init_module._init_reference_registry_preflight(status)

    assert calls == [
        ("scan", ref_root),
        ("register", scanned, catalog_path, ref_root),
    ]
    settings = yaml.safe_load((home / ".openbench" / "settings.yaml").read_text())
    assert settings["reference_root"] == str(ref_root.resolve())


def test_init_reference_preflight_existing_catalog_can_skip_update(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    catalog_path = tmp_path / "user" / "references" / "reference_catalog.yaml"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text("Dataset: {}\n")
    status = init_module.ReferenceCatalogStatus(
        catalog_path=catalog_path,
        exists=True,
        empty=False,
    )

    monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        init_module,
        "_scan_reference_variants",
        lambda root: (_ for _ in ()).throw(AssertionError("scan should be skipped")),
        raising=False,
    )

    init_module._init_reference_registry_preflight(status)


def test_init_refresh_ref_registers_without_second_confirmation(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    catalog_path = tmp_path / "user" / "references" / "reference_catalog.yaml"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text("Dataset: {}\n")
    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    scanned = [object()]
    calls = []
    home = tmp_path / "home"
    status = init_module.ReferenceCatalogStatus(
        catalog_path=catalog_path,
        exists=True,
        empty=False,
    )

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(init_module, "_resolve_reference_root", lambda value: ref_root)
    monkeypatch.setattr(
        init_module,
        "_scan_reference_variants",
        lambda root: calls.append(("scan", root)) or scanned,
    )
    monkeypatch.setattr(
        init_module,
        "_register_reference_variants",
        lambda variants, path, **kwargs: calls.append(("register", variants, path, kwargs.get("ref_root"))) or path,
    )
    monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no confirm")))

    init_module._init_reference_registry_preflight(status, refresh_ref=True)

    assert calls == [
        ("scan", ref_root),
        ("register", scanned, catalog_path, ref_root),
    ]


def test_register_reference_variants_restores_ref_env_after_registration(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    catalog = tmp_path / "reference_catalog.yaml"
    captured = []

    def fake_register_scanned_datasets_batch(variants, catalog_path=None, **kwargs):
        captured.append(__import__("os").environ.get("OPENBENCH_REF_ROOT"))
        return catalog_path

    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/old/reference")
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register_scanned_datasets_batch)
    monkeypatch.setattr("openbench.data.registry.manager.clear_registry_cache", lambda: None)

    init_module._register_reference_variants([object()], catalog, ref_root=ref_root)

    assert captured == [str(ref_root.resolve())]
    assert __import__("os").environ["OPENBENCH_REF_ROOT"] == "/old/reference"


def test_scan_reference_variants_reports_progress_and_restores_ref_env(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    variant = SimpleNamespace(registry_name="TestRef", data_type="grid", category="Water", variables={}, file_count=1)
    captured = []

    def fake_scan_reference_directory(root, on_progress=None, on_skip=None):
        captured.append((root, on_progress, on_skip))
        if on_progress:
            on_progress("scanning child")
        return [SimpleNamespace(variants={"default": variant})]

    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/old/reference")
    monkeypatch.setattr(scanner_module, "scan_reference_directory", fake_scan_reference_directory)

    variants = init_module._scan_reference_variants(ref_root)

    assert variants == [variant]
    assert captured[0][0] == ref_root
    assert callable(captured[0][1])
    assert callable(captured[0][2])
    assert __import__("os").environ["OPENBENCH_REF_ROOT"] == "/old/reference"


def test_scan_reference_variants_reports_unsupported_folders(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()

    def fake_scan_reference_directory(root, on_progress=None, on_skip=None):
        on_skip(
            scanner_module.ScanSkip(
                path="Grid/LowRes/Composite/Bad",
                reason="ambiguous_nc_subdirectories",
                hint="Register manually.",
            )
        )
        return []

    monkeypatch.setattr(scanner_module, "scan_reference_directory", fake_scan_reference_directory)

    result = runner.invoke(
        click.Command("scan", callback=lambda: init_module._scan_reference_variants(ref_root)),
        input="\n",
    )

    assert result.exit_code == 0, result.output
    assert "Unsupported folder" in result.output
    assert "Grid/LowRes/Composite/Bad" in result.output


def test_scan_reference_variants_can_create_profile_for_unsupported_folders(tmp_path, monkeypatch):
    import openbench.cli.data as cli_data
    import openbench.cli.init_cmd as init_module
    import openbench.data.registry.scanner as scanner_module

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    skip = scanner_module.ScanSkip(
        path="Grid/LowRes/Composite/Bad",
        reason="ambiguous_nc_subdirectories",
        hint="Register manually.",
    )
    variant = SimpleNamespace(
        registry_name="RecoveredRef",
        data_type="grid",
        category="Water",
        variables={"Runoff": "Runoff"},
        file_count=1,
    )
    calls = []
    profile_calls = []

    def fake_scan_reference_directory(root, on_progress=None, on_skip=None):
        calls.append(root)
        if len(calls) == 1:
            on_skip(skip)
            return []
        return [SimpleNamespace(variants={"default": variant})]

    monkeypatch.setattr(scanner_module, "scan_reference_directory", fake_scan_reference_directory)
    monkeypatch.setattr(cli_data, "_prompt_scan_skip_action", lambda *args, **kwargs: "p")
    monkeypatch.setattr(cli_data, "_profile_rescue_supported", lambda item: True)
    monkeypatch.setattr(
        cli_data,
        "_create_profiles_for_scan_skips",
        lambda skipped, root: profile_calls.append((list(skipped), root)) or 1,
    )

    variants = init_module._scan_reference_variants(ref_root)

    assert variants == [variant]
    assert calls == [ref_root, ref_root]
    assert profile_calls == [([skip], ref_root)]


def test_reference_catalog_status_marks_corrupt_catalog(tmp_path):
    import openbench.cli.init_cmd as init_module

    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    catalog.write_text("bad: [\n")

    status = init_module._reference_catalog_status(user_dir=tmp_path)

    assert status.exists is True
    assert status.empty is False
    assert status.corrupted is True
    with pytest.raises(click.ClickException, match="Reference catalog is corrupt"):
        init_module._init_reference_registry_preflight(status)


def test_resolve_reference_root_expands_prompted_user_path(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    home = tmp_path / "home"
    ref_root = home / "Reference"
    ref_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    @click.command()
    def command():
        click.echo(init_module._resolve_reference_root())

    result = runner.invoke(command, input="~/Reference\n")

    assert result.exit_code == 0, result.output
    assert str(ref_root) in result.output


def test_resolve_reference_root_warns_when_saved_root_is_missing(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    from openbench.config.user_settings import remember_reference_root

    home = tmp_path / "home"
    valid_root = tmp_path / "Reference"
    missing_root = tmp_path / "MissingReference"
    valid_root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)
    remember_reference_root(missing_root)

    @click.command()
    def command():
        click.echo(init_module._resolve_reference_root())

    result = runner.invoke(command, input=f"{valid_root}\n")

    assert result.exit_code == 0, result.output
    assert "is not available" in result.output
    assert str(valid_root) in result.output


def test_parse_variable_selection_accepts_names_case_insensitively():
    import openbench.cli.init_cmd as init_module

    selected = init_module._parse_variable_selection(
        "latent_heat, 1",
        ["Runoff", "Latent_Heat"],
    )

    assert selected == ["Latent_Heat", "Runoff"]


def test_prompt_manual_simulations_accepts_numeric_model_name_without_known_models(monkeypatch):
    import openbench.cli.init_cmd as init_module

    class FakeRegistryManager:
        def list_models(self):
            return []

    answers = iter(["2024", "/data/2024", "Run2024", ""])
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: next(answers))

    simulation = init_module._prompt_manual_simulations(FakeRegistryManager())

    assert simulation == {"Run2024": {"model": "2024", "root_dir": "/data/2024"}}


def test_prompt_manual_simulations_rejects_duplicate_labels(monkeypatch):
    import openbench.cli.init_cmd as init_module

    class FakeRegistryManager:
        def list_models(self):
            return []

    answers = iter(
        [
            "ModelA",
            "/data/a",
            "Case",
            "ModelB",
            "/data/b",
            "Case",
            "ModelB",
            "/data/b",
            "CaseB",
            "",
        ]
    )
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: next(answers))

    simulation = init_module._prompt_manual_simulations(FakeRegistryManager())

    assert simulation == {
        "Case": {"model": "ModelA", "root_dir": "/data/a"},
        "CaseB": {"model": "ModelB", "root_dir": "/data/b"},
    }


def test_init_creates_output_parent_directory(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    output = tmp_path / "missing" / "nested" / "openbench.yaml"
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="\n\n2004\n2004\n\n\n\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_init_confirms_before_overwriting_output(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    output = tmp_path / "openbench.yaml"
    output.write_text("original\n")
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(cli, ["init", "--no-ref-check", "-o", str(output)], input="n\n")

    assert result.exit_code != 0
    assert "Output file already exists" in result.output
    assert output.read_text() == "original\n"


def test_init_rejects_directory_output_without_traceback(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    output = tmp_path / "openbench-dir"
    output.mkdir()
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", str(output)],
        input="y\n\n\n2004\n2004\n\n\n\nn\nn\n",
    )

    assert result.exit_code != 0
    assert "Output path must be a file" in result.output
    assert "Traceback" not in result.output


def test_init_rejects_unresolved_environment_variable_in_output(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBENCH_MISSING_OUT", raising=False)
    monkeypatch.setattr(init_module, "ensure_user_registry_overlays", lambda: tmp_path / "user")
    _install_single_reference_registry(monkeypatch)

    result = runner.invoke(
        cli,
        ["init", "--no-ref-check", "-o", "$OPENBENCH_MISSING_OUT/openbench.yaml"],
        input="\n\n2004\n2004\n\n\n\nn\nn\n",
    )

    assert result.exit_code != 0
    assert "unresolved environment variable" in result.output
    assert not (tmp_path / "$OPENBENCH_MISSING_OUT").exists()


def test_scan_simulation_config_prompts_for_model_when_auto_inference_is_unresolved(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    calls = []

    def fake_scan_simulation_roots(roots, **kwargs):
        calls.append(kwargs["model_name"])
        unresolved = SimulationCase(
            label="Case01",
            root_dir=case_root,
            model="UNRESOLVED",
            depth=1,
            unresolved=["model"],
        )
        resolved = SimulationCase(
            label="Case01",
            root_dir=case_root,
            model="CoLM",
            depth=1,
            data_type="grid",
            tim_res="Month",
            grid_res=0.5,
            data_groupby="Month",
        )
        if len(calls) == 1:
            return SimulationScanResult(roots=[sim_root], cases=[unresolved], unresolved=[unresolved])
        return SimulationScanResult(roots=[sim_root], cases=[resolved])

    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", lambda *args, **kwargs: None)
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: "CoLM")

    simulation = init_module._scan_simulation_config(
        [str(sim_root)],
        model_name="auto",
        output_path=tmp_path / "openbench.yaml",
        case_depth=5,
        case_pattern=None,
        exclude=(),
        climatology="auto",
    )

    assert calls == ["auto", "CoLM"]
    assert simulation["Case01"]["model"] == "CoLM"


def test_scan_simulation_config_writes_portable_station_fulllist(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    output = tmp_path / "configs" / "openbench.yaml"
    case = SimulationCase(
        label="Case01",
        root_dir=case_root,
        model="CoLM",
        depth=1,
        data_type="stn",
        tim_res="Day",
        data_groupby="Single",
        station_layout="flat",
    )

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(roots=[sim_root], cases=[case])

    materialize_calls = []

    def fake_materialize_station_cases(result, output_dir, **kwargs):
        materialize_calls.append((Path(output_dir), kwargs))
        result.cases[0].fulllist = Path(output_dir) / "Case01" / "Case01_stations.csv"

    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", fake_materialize_station_cases)

    simulation = init_module._scan_simulation_config(
        [str(sim_root)],
        model_name="CoLM",
        output_path=output,
        case_depth=5,
        case_pattern=None,
        exclude=(),
        climatology="auto",
    )

    assert materialize_calls == [(tmp_path / "configs" / "openbench_sim_station_lists", {"allow_partial": False})]
    assert simulation["Case01"]["fulllist"] == "openbench_sim_station_lists/Case01/Case01_stations.csv"


def test_scan_simulation_config_aborts_on_partial_station_materialization(tmp_path, monkeypatch):
    import openbench.cli.init_cmd as init_module
    import openbench.data.sim_scanner as sim_scanner
    from openbench.data.sim_scanner import SimulationCase, SimulationScanResult

    sim_root = tmp_path / "Simulation"
    case_root = sim_root / "Case01"
    case_root.mkdir(parents=True)
    case = SimulationCase(
        label="Case01",
        root_dir=case_root,
        model="CoLM",
        depth=1,
        data_type="stn",
        tim_res="Day",
        data_groupby="Single",
        station_layout="flat",
    )

    def fake_scan_simulation_roots(roots, **kwargs):
        return SimulationScanResult(roots=[sim_root], cases=[case])

    def fake_materialize_station_cases(result, output_dir, **kwargs):
        result.cases[0].station_dropped_sites = ["US-ABC"]

    monkeypatch.setattr(sim_scanner, "scan_simulation_roots", fake_scan_simulation_roots)
    monkeypatch.setattr(sim_scanner, "materialize_station_cases", fake_materialize_station_cases)

    with pytest.raises(click.ClickException, match="Station materialization dropped sites"):
        init_module._scan_simulation_config(
            [str(sim_root)],
            model_name="CoLM",
            output_path=tmp_path / "openbench.yaml",
            case_depth=5,
            case_pattern=None,
            exclude=(),
            climatology="auto",
        )


def test_gui_help():
    result = runner.invoke(cli, ["gui", "--help"])
    assert result.exit_code == 0


def test_gui_remote_option_accepts_profile_name_and_is_explicitly_rejected(monkeypatch):
    import openbench.gui as gui_package
    import openbench.gui.app as gui_app

    called = []
    monkeypatch.setattr(gui_package, "_check_gui_deps", lambda: None)
    monkeypatch.setattr(gui_app, "launch", lambda **kwargs: called.append(kwargs))

    result = runner.invoke(cli, ["gui", "--remote", "cluster"])
    assert result.exit_code != 0
    assert "not implemented" in result.output.lower()
    assert called == []


def test_gui_rejects_directory_config_path(tmp_path, monkeypatch):
    import openbench.gui as gui_package
    import openbench.gui.app as gui_app

    config_dir = tmp_path / "config.yaml"
    config_dir.mkdir()
    called = []
    monkeypatch.setattr(gui_package, "_check_gui_deps", lambda: None)
    monkeypatch.setattr(gui_app, "launch", lambda **kwargs: called.append(kwargs))

    result = runner.invoke(cli, ["gui", str(config_dir)])

    assert result.exit_code != 0
    assert "file" in result.output.lower()
    assert called == []


def test_simulation_root_validation_flags_windows_style_unresolved_env():
    from openbench.cli._simulation_validation import simulation_root_errors

    cfg = SimpleNamespace(
        simulation={
            "CaseA": SimpleNamespace(root_dir="%OPENBENCH_SIM_ROOT%/case"),
        }
    )

    errors = simulation_root_errors(cfg)

    assert errors == [
        (
            "CaseA",
            "Simulation root contains unresolved environment variable: %OPENBENCH_SIM_ROOT%/case",
        )
    ]


def test_check_rejects_directory_config_path(tmp_path):
    config_dir = tmp_path / "config.yaml"
    config_dir.mkdir()

    result = runner.invoke(cli, ["check", str(config_dir)])

    assert result.exit_code != 0
    assert "file" in result.output.lower()


def test_run_rejects_directory_config_path(tmp_path):
    config_dir = tmp_path / "config.yaml"
    config_dir.mkdir()

    result = runner.invoke(cli, ["run", str(config_dir), "--dry-run"])

    assert result.exit_code != 0
    assert "file" in result.output.lower()


def test_run_dry_run_strict_catches_bad_reference(tmp_path, monkeypatch):
    """run --dry-run with strict_reference=true must fail on unresolved references."""
    import yaml

    config_path = tmp_path / "strict_test.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "strict_dry",
                    "output_dir": str(tmp_path),
                    "years": [2004, 2005],
                    "strict_reference": True,
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "NONEXISTENT_DATASET"},
                "simulation": {
                    "Case01": {
                        "model": "TestModel",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                    }
                },
            }
        )
    )

    result = runner.invoke(cli, ["run", "--dry-run", str(config_path)])
    assert result.exit_code != 0
    assert "NONEXISTENT_DATASET" in result.output or "resolution" in result.output.lower()


def test_run_strict_reference_fails_before_runner_like_dry_run(tmp_path, monkeypatch):
    """run should resolve references before invoking the runner, just like dry-run."""
    import yaml

    config_path = tmp_path / "strict_run.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "strict_run",
                    "output_dir": str(tmp_path),
                    "years": [2004, 2005],
                    "strict_reference": True,
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "NONEXISTENT_DATASET"},
                "simulation": {
                    "Case01": {
                        "model": "TestModel",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )
    called = []
    monkeypatch.setattr("openbench.runner.local.run_evaluation", lambda *args, **kwargs: called.append(True))

    result = runner.invoke(cli, ["run", str(config_path)])

    assert result.exit_code != 0
    assert "NONEXISTENT_DATASET" in result.output or "resolution" in result.output.lower()
    assert called == []


def test_run_lenient_reference_status_fails_before_runner_like_dry_run(tmp_path, monkeypatch):
    """run should not enter the runner with unresolved references in non-strict mode."""
    config_path = tmp_path / "lenient_run.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "lenient_run", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "NONEXISTENT_DATASET"},
                "simulation": {
                    "Case01": {
                        "model": "TestModel",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )
    called = []
    monkeypatch.setattr("openbench.runner.local.run_evaluation", lambda *args, **kwargs: called.append(True))

    result = runner.invoke(cli, ["run", str(config_path)])

    assert result.exit_code != 0
    assert "NONEXISTENT_DATASET" in result.output or "Reference resolution errors" in result.output
    assert called == []


def test_run_rejects_duplicate_variable_overrides(tmp_path):
    config_path = tmp_path / "duplicate_vars.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "duplicate_vars", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "Case01": {
                        "model": "TestModel",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )

    result = runner.invoke(
        cli,
        [
            "run",
            str(config_path),
            "--dry-run",
            "--variable",
            "Latent_Heat",
            "--variable",
            "Latent_Heat",
        ],
    )

    assert result.exit_code != 0
    assert "--variable values must be unique: Latent_Heat" in result.output


def test_check_fails_on_missing_simulation_root(tmp_path):
    config_path = tmp_path / "missing_sim_root.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "missing_sim", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": "/path/to/data",
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
                "comparison": {"enabled": True, "items": ["HeatMap"]},
            }
        )
    )

    result = runner.invoke(cli, ["check", str(config_path)])

    assert result.exit_code == 1
    assert "Simulation root does not exist" in result.output
    assert "Ready to run" not in result.output


def test_check_fails_when_resolved_reference_root_is_missing(tmp_path, monkeypatch):
    from openbench.data.registry import manager as mgr_mod

    config_path = tmp_path / "missing_ref_root.yaml"
    missing_ref = tmp_path / "missing-reference"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "missing_ref", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestRef"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                    }
                },
            }
        )
    )

    class MockRegistry:
        def get_reference(self, name, **kwargs):
            return SimpleNamespace(
                name=name,
                data_type="grid",
                tim_res="Month",
                grid_res=0.5,
                years=[2000, 2010],
                root_dir=str(missing_ref),
                variables={
                    "Runoff": SimpleNamespace(
                        varname="ro",
                        varunit="mm day-1",
                        sub_dir="",
                        prefix="",
                        suffix="",
                        fulllist=None,
                    )
                },
                _provenance={},
            )

        def get_resolution_variants(self, name):
            return {}

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: MockRegistry())

    result = runner.invoke(cli, ["check", str(config_path)])

    assert result.exit_code == 1
    assert "Reference root does not exist" in result.output
    assert str(missing_ref) in result.output
    assert "Ready to run" not in result.output


def test_check_warns_when_resolved_reference_root_has_no_netcdf_files(tmp_path, monkeypatch):
    from openbench.data.registry import manager as mgr_mod

    ref_root = tmp_path / "empty-reference"
    ref_root.mkdir()
    config_path = tmp_path / "empty_ref_root.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "empty_ref", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestRef"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                    }
                },
            }
        )
    )

    class MockRegistry:
        def get_reference(self, name, **kwargs):
            return SimpleNamespace(
                name=name,
                data_type="grid",
                tim_res="Month",
                grid_res=0.5,
                years=[2000, 2010],
                root_dir=str(ref_root),
                variables={
                    "Runoff": SimpleNamespace(
                        varname="ro",
                        varunit="mm day-1",
                        sub_dir="",
                        prefix="",
                        suffix="",
                        fulllist=None,
                    )
                },
                _provenance={},
            )

        def get_resolution_variants(self, name):
            return {}

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: MockRegistry())

    result = runner.invoke(cli, ["check", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "no NetCDF files found" in result.output
    assert "Ready to run" in result.output


def test_run_dry_run_fails_on_missing_simulation_root(tmp_path):
    config_path = tmp_path / "missing_sim_root_dry.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "missing_sim", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": "/path/to/data",
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
                "comparison": {"enabled": True, "items": ["HeatMap"]},
            }
        )
    )

    result = runner.invoke(cli, ["run", "--dry-run", str(config_path)])

    assert result.exit_code == 1
    assert "Simulation root does not exist" in result.output


def test_run_comparison_only_skips_missing_simulation_root_preflight(tmp_path, monkeypatch):
    import yaml

    config_path = tmp_path / "comparison_only.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "compare_only", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": "/path/to/data",
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
                "comparison": {"enabled": True, "items": ["HeatMap"]},
            }
        )
    )

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        return {
            "status": "success",
            "output_dir": str(tmp_path),
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", "--comparison-only", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Simulation root does not exist" not in result.output


def test_run_remote_fails_before_local_preflight_or_dump_config(tmp_path):
    config_path = tmp_path / "remote.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "remote_case", "output_dir": str(output_dir), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {"MyModel": {"model": "MyModel", "root_dir": str(tmp_path / "missing")}},
            }
        )
    )

    result = runner.invoke(cli, ["run", str(config_path), "--remote", "cluster", "--dump-config"])

    assert result.exit_code != 0
    assert "Remote execution not yet implemented" in result.output
    assert "Simulation root does not exist" not in result.output
    assert not (output_dir / "remote_case" / "debug").exists()


def test_run_variable_alias_matches_legacy_variables_option(tmp_path, monkeypatch):
    config_path = tmp_path / "variables.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "vars", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["A", "B", "C"]},
                "reference": {"A": "RefA", "B": "RefB", "C": "RefC"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "variables": {
                            "A": {"varname": "A"},
                            "B": {"varname": "B"},
                            "C": {"varname": "C"},
                        },
                    }
                },
            }
        )
    )

    seen = []

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        seen.append(list(cfg.evaluation.variables))
        return {
            "status": "success",
            "output_dir": str(tmp_path),
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.cli.run._resolve_references_for_run", lambda cfg: [])
    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    new_result = runner.invoke(cli, ["run", str(config_path), "--variable", "A", "--variable", "B"])
    old_result = runner.invoke(cli, ["run", str(config_path), "--variables", "A", "--variables", "B"])

    assert new_result.exit_code == 0, new_result.output
    assert old_result.exit_code == 0, old_result.output
    assert seen == [["A", "B"], ["A", "B"]]


def test_run_force_option_bypasses_cache_without_editing_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "force.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "force_case", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )

    seen_force = []

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        seen_force.append(force)
        return {
            "status": "success",
            "output_dir": str(tmp_path),
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path), "--force"])

    assert result.exit_code == 0, result.output
    assert seen_force == [True]


def test_run_rejects_unknown_variable_override_before_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "bad_var.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "bad_var", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {"MyModel": {"model": "MyModel", "root_dir": str(tmp_path)}},
            }
        )
    )

    called = []
    monkeypatch.setattr("openbench.runner.local.run_evaluation", lambda *args, **kwargs: called.append(True))

    result = runner.invoke(cli, ["run", str(config_path), "--variable", "Typo"])

    assert result.exit_code == 1
    assert "not in evaluation.variables" in result.output
    assert called == []


def test_run_dry_run_reuses_check_preflight_for_unknown_metric(tmp_path):
    config_path = tmp_path / "bad_metric.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "bad_metric", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {"MyModel": {"model": "MyModel", "root_dir": str(tmp_path)}},
                "metrics": ["Definitely_Not_A_Metric"],
            }
        )
    )

    result = runner.invoke(cli, ["run", "--dry-run", str(config_path)])

    assert result.exit_code == 1
    assert "Unknown metric" in result.output


def test_run_comparison_only_requires_comparison_enabled_before_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "comparison_disabled.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "comparison_disabled", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {"MyModel": {"model": "MyModel", "root_dir": "/missing/root"}},
                "comparison": {"enabled": False},
            }
        )
    )

    called = []
    monkeypatch.setattr("openbench.runner.local.run_evaluation", lambda *args, **kwargs: called.append(True))

    result = runner.invoke(cli, ["run", "--comparison-only", str(config_path)])

    assert result.exit_code == 1
    assert "comparison.enabled" in result.output
    assert called == []


def test_run_dry_run_comparison_only_checks_existing_outputs(tmp_path):
    config_path = tmp_path / "comparison_missing_outputs.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "comparison_missing_outputs", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": "/missing/root",
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
                "comparison": {"enabled": True, "items": ["HeatMap"]},
            }
        )
    )

    result = runner.invoke(cli, ["run", "--dry-run", "--comparison-only", str(config_path)])

    assert result.exit_code == 1
    assert "missing prerequisite outputs" in result.output


def test_run_dry_run_only_drawing_checks_existing_outputs(tmp_path):
    config_path = tmp_path / "only_drawing_missing_outputs.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "only_drawing_missing_outputs",
                    "output_dir": str(tmp_path),
                    "years": [2004, 2005],
                    "only_drawing": True,
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
                "metrics": ["bias"],
                "scores": ["Overall_Score"],
                "comparison": {"enabled": False},
            }
        )
    )

    result = runner.invoke(cli, ["run", "--dry-run", str(config_path)])

    assert result.exit_code == 1
    assert "missing prerequisite outputs" in result.output


def test_run_only_drawing_skips_missing_simulation_root_preflight(tmp_path, monkeypatch):
    config_path = tmp_path / "only_drawing.yaml"
    case_dir = tmp_path / "only_drawing"
    _write_fake_cli_grid_outputs(case_dir, "Latent_Heat", "FLUXCOM_LowRes", "MyModel")
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "only_drawing",
                    "output_dir": str(tmp_path),
                    "years": [2004, 2005],
                    "only_drawing": True,
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": "/missing/raw/sim/root",
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
                "metrics": ["bias"],
                "scores": ["Overall_Score"],
                "comparison": {"enabled": False},
            }
        )
    )

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        return {
            "status": "success",
            "output_dir": str(case_dir),
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Simulation root does not exist" not in result.output


def test_run_expands_env_paths_before_runner(tmp_path, monkeypatch):
    env_root = tmp_path / "envroot"
    sim_root = env_root / "sim"
    out_root = env_root / "out"
    sim_root.mkdir(parents=True)
    monkeypatch.setenv("OPENBENCH_RUN_ENVROOT", str(env_root))

    config_path = tmp_path / "env_paths.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "env_paths",
                    "output_dir": "$OPENBENCH_RUN_ENVROOT/out",
                    "years": [2004, 2005],
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": "$OPENBENCH_RUN_ENVROOT/sim",
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )

    seen = []

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        seen.append((cfg.project.output_dir, cfg.simulation["MyModel"].root_dir))
        return {
            "status": "success",
            "output_dir": cfg.project.output_dir,
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path)])

    assert result.exit_code == 0, result.output
    assert seen == [(str(out_root), str(sim_root))]


def test_run_output_dir_option_overrides_yaml_before_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "output_override.yaml"
    yaml_out = tmp_path / "yaml_out"
    cli_out = tmp_path / "cli_out"
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "output_override",
                    "output_dir": str(yaml_out),
                    "years": [2004, 2005],
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )

    seen = []

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        seen.append(cfg.project.output_dir)
        return {
            "status": "success",
            "output_dir": cfg.project.output_dir,
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path), "--output-dir", str(cli_out)])

    assert result.exit_code == 0, result.output
    assert seen == [str(cli_out)]


def test_run_preflights_station_simulation_fulllist_before_runner(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    config_path = tmp_path / "missing_fulllist.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "missing_fulllist", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(sim_root),
                        "data_type": "stn",
                        "fulllist": "missing_sites.csv",
                    }
                },
            }
        )
    )

    called = []
    monkeypatch.setattr("openbench.runner.local.run_evaluation", lambda *args, **kwargs: called.append(True))

    result = runner.invoke(cli, ["run", str(config_path)])

    assert result.exit_code == 1
    assert "Station fulllist does not exist" in result.output
    assert called == []


def test_run_writes_per_run_log_with_debug_records(tmp_path, monkeypatch):
    import logging

    config_path = tmp_path / "run_log.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "run_log_case", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {
                    "MyModel": {
                        "model": "MyModel",
                        "root_dir": str(tmp_path),
                        "variables": {"Latent_Heat": {"varname": "Latent_Heat"}},
                    }
                },
            }
        )
    )

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        logger = logging.getLogger("openbench.tests.run_log")
        logger.debug("debug marker for run.log")
        logger.info("info marker for run.log")
        return {
            "status": "success",
            "output_dir": str(Path(cfg.project.output_dir) / cfg.project.name),
            "variables": cfg.evaluation.variables,
            "simulations": list(cfg.simulation),
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path)])

    log_path = tmp_path / "run_log_case" / "run.log"
    assert result.exit_code == 0, result.output
    assert log_path.exists()
    text = log_path.read_text()
    assert "debug marker for run.log" in text
    assert "info marker for run.log" in text


def test_run_rejects_project_name_path_before_evaluation(tmp_path, monkeypatch):
    config_path = tmp_path / "unsafe_project_name.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "../../etc/passwd", "output_dir": str(tmp_path), "years": [2004, 2005]},
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"Latent_Heat": "FLUXCOM_LowRes"},
                "simulation": {"MyModel": {"model": "MyModel", "root_dir": str(tmp_path)}},
            }
        )
    )
    called = []

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        called.append(cfg.project.name)
        return {
            "status": "success",
            "output_dir": str(tmp_path),
            "variables": ["Latent_Heat"],
            "simulations": ["MyModel"],
        }

    monkeypatch.setattr("openbench.runner.local.run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path)])

    assert result.exit_code != 0
    assert "project.name must be a simple directory name" in result.output
    assert called == []


def test_check_resolution_ambiguity_shows_comparison_guidance(tmp_path, monkeypatch):
    """check should tell users how to fix cross-simulation resolution ambiguity."""
    import yaml

    config_path = tmp_path / "ambiguous_check.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "ambiguous", "output_dir": str(tmp_path), "years": [2000, 2001]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {
                    "S1": {"model": "M1", "root_dir": str(tmp_path), "tim_res": "Month", "grid_res": 0.5},
                    "S2": {"model": "M2", "root_dir": str(tmp_path), "tim_res": "Day", "grid_res": 0.25},
                },
            }
        )
    )

    from openbench.data.registry import manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: object())

    result = runner.invoke(cli, ["check", str(config_path)])
    assert result.exit_code == 1
    assert "ambiguous across simulations" in result.output
    assert "project.tim_res" in result.output
    assert "project.grid_res" in result.output
    assert "project:" in result.output
    assert "tim_res: Month" in result.output
    assert "grid_res: 0.25" in result.output


def test_run_dry_run_resolution_ambiguity_shows_comparison_guidance(tmp_path, monkeypatch):
    """run --dry-run should show a direct fix for cross-simulation resolution ambiguity."""
    import yaml

    config_path = tmp_path / "ambiguous_dry_run.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "ambiguous", "output_dir": str(tmp_path), "years": [2000, 2001]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {
                    "S1": {"model": "M1", "root_dir": str(tmp_path), "tim_res": "Month", "grid_res": 0.5},
                    "S2": {"model": "M2", "root_dir": str(tmp_path), "tim_res": "Day", "grid_res": 0.25},
                },
            }
        )
    )

    from openbench.data.registry import manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: object())

    result = runner.invoke(cli, ["run", "--dry-run", str(config_path)])
    assert result.exit_code == 1
    assert "Reference resolution failed" in result.output
    assert "ambiguous across simulations" in result.output
    assert "project.tim_res" in result.output
    assert "project.grid_res" in result.output
    assert "project:" in result.output
    assert "tim_res: Month" in result.output
    assert "grid_res: 0.25" in result.output


def test_all_commands_registered():
    """Verify all expected commands are accessible via the CLI group."""
    # Use list_commands() for LazyGroup compatibility
    ctx = click.Context(cli)
    command_names = set(cli.list_commands(ctx))
    expected = {"run", "check", "ref", "sim", "model", "migrate", "init", "cache", "gui", "version"}
    assert expected == command_names, f"Missing: {expected - command_names}, Extra: {command_names - expected}"


def test_lazy_group_list_commands_includes_decorated_commands_dynamically():
    @click.command("diagnostic")
    def diagnostic():
        click.echo("ok")

    cli.add_command(diagnostic)
    try:
        command_names = set(cli.list_commands(click.Context(cli)))
    finally:
        cli.commands.pop("diagnostic", None)

    assert "diagnostic" in command_names
    assert "version" in command_names


def test_check_warns_on_default_provenance_tim_res(tmp_path, monkeypatch):
    """check should warn when tim_res provenance is 'default'."""
    from types import SimpleNamespace

    import yaml

    config_path = tmp_path / "prov_test.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "prov", "output_dir": str(tmp_path), "years": [2000, 2001]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {"S1": {"model": "M", "root_dir": str(tmp_path), "tim_res": "Month", "grid_res": 0.5}},
            }
        )
    )

    # Mock registry to return a ref with default provenance
    class MockRef:
        name = "TestDS"
        data_type = "grid"
        tim_res = "Month"
        grid_res = 0.5
        _provenance = {"tim_res": "default", "grid_res": "nc"}
        variables = {"Runoff": SimpleNamespace(varname="ro", varunit="mm")}

    class MockRegistry:
        def get_reference(self, name, **kw):
            return MockRef()

        def get_resolution_variants(self, name):
            return {}

    from openbench.data.registry import manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: MockRegistry())

    result = runner.invoke(cli, ["check", str(config_path)])
    assert "not confirmed" in result.output.lower() or "unconfirmed" in result.output.lower()


def test_run_dry_run_strict_fails_on_default_provenance(tmp_path, monkeypatch):
    """run --dry-run should fail in strict mode on low-confidence provenance."""
    from types import SimpleNamespace

    import yaml

    config_path = tmp_path / "prov_strict.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {
                    "name": "prov_strict",
                    "output_dir": str(tmp_path),
                    "years": [2000, 2001],
                    "strict_reference": True,
                },
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {"S1": {"model": "M", "root_dir": str(tmp_path), "tim_res": "Month", "grid_res": 0.5}},
            }
        )
    )

    class MockRef:
        name = "TestDS"
        data_type = "grid"
        tim_res = "Month"
        grid_res = 0.5
        _provenance = {"tim_res": "default", "grid_res": "nc"}
        variables = {"Runoff": SimpleNamespace(varname="ro", varunit="mm")}

    class MockRegistry:
        def get_reference(self, name, **kw):
            return MockRef()

        def get_resolution_variants(self, name):
            return {}

    from openbench.data.registry import manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: MockRegistry())

    result = runner.invoke(cli, ["run", "--dry-run", str(config_path)])
    assert result.exit_code != 0
    assert "strict_reference" in result.output or "unconfirmed default" in result.output.lower()


def test_run_exits_nonzero_when_runner_reports_errors(tmp_path, monkeypatch):
    """run command should surface runner failures as a non-zero exit."""
    import yaml

    config_path = tmp_path / "partial.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "partial", "output_dir": str(tmp_path), "years": [2000, 2001]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {
                    "S1": {"model": "M", "root_dir": str(tmp_path), "variables": {"Runoff": {"varname": "Runoff"}}}
                },
            }
        )
    )

    def fake_run_evaluation(cfg, force=False, comparison_only=False):
        return {
            "status": "error",
            "output_dir": str(tmp_path),
            "variables": ["Runoff"],
            "simulations": ["S1"],
            "errors": [{"phase": "comparison", "message": "missing prerequisite outputs"}],
        }

    import openbench.runner.local as local_runner

    monkeypatch.setattr("openbench.cli.run._resolve_references_for_run", lambda cfg: [])
    monkeypatch.setattr(local_runner, "run_evaluation", fake_run_evaluation)

    result = runner.invoke(cli, ["run", str(config_path)])
    assert result.exit_code == 1
    assert "missing prerequisite outputs" in result.output.lower()


def test_run_dry_run_dump_config_is_read_only(tmp_path, monkeypatch):
    """run --dry-run --dump-config should not write debug artifacts."""
    from dataclasses import dataclass

    import yaml

    config_path = tmp_path / "dump.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "dump_case", "output_dir": str(tmp_path), "years": [2000, 2001]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {
                    "S1": {
                        "model": "M",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                        "variables": {"Runoff": {"varname": "Runoff"}},
                    }
                },
            }
        )
    )

    import openbench.config.adapter as adapter_mod

    @dataclass
    class FakeRunnerConfig:
        basedir: str
        basename: str

    class ExplodingCompatBindings:
        def __init__(self):
            self.runner_cfg = FakeRunnerConfig(basedir=str(tmp_path), basename="dump_case")
            self.namelists = adapter_mod.LegacyNamelists(
                main={"general": {}},
                reference={"general": {}},
                simulation={"general": {}},
            )
            self.figures = adapter_mod.LegacyFigureConfig(raw={})

        @property
        def main_nl(self):
            raise AssertionError("cli should use bindings.namelists, not compatibility properties")

        @property
        def ref_nml(self):
            raise AssertionError("cli should use bindings.namelists, not compatibility properties")

        @property
        def sim_nml(self):
            raise AssertionError("cli should use bindings.namelists, not compatibility properties")

        @property
        def fig_nml(self):
            raise AssertionError("cli should use bindings.figures, not compatibility properties")

    monkeypatch.setattr(adapter_mod, "build_runner_bindings", lambda cfg: ExplodingCompatBindings())

    from openbench.data.registry import manager as mgr_mod

    class MockRegistry:
        def get_resolution_variants(self, name):
            return {}

        def get_reference(self, name, **kwargs):
            return SimpleNamespace(
                name=name,
                data_type="grid",
                tim_res="Month",
                grid_res=0.5,
                years=[2000, 2001],
                root_dir=str(tmp_path),
                variables={"Runoff": SimpleNamespace(varname="Runoff", varunit="", sub_dir="", prefix="", suffix="")},
                _provenance={},
            )

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: MockRegistry())

    result = runner.invoke(cli, ["run", "--dump-config", "--dry-run", str(config_path)])

    debug_dir = tmp_path / "dump_case" / "debug"
    assert result.exit_code == 0
    assert "Debug config dump skipped in dry-run mode" in result.output
    assert not debug_dir.exists()


def test_run_dump_config_writes_runner_facing_debug_artifacts_for_real_run(tmp_path, monkeypatch):
    """run --dump-config should write runner-focused debug files instead of legacy_config.yaml."""
    from dataclasses import dataclass

    import yaml

    config_path = tmp_path / "dump.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project": {"name": "dump_case", "output_dir": str(tmp_path), "years": [2000, 2001]},
                "evaluation": {"variables": ["Runoff"]},
                "reference": {"Runoff": "TestDS"},
                "simulation": {
                    "S1": {
                        "model": "M",
                        "root_dir": str(tmp_path),
                        "tim_res": "Month",
                        "grid_res": 0.5,
                        "variables": {"Runoff": {"varname": "Runoff"}},
                    }
                },
            }
        )
    )

    import openbench.config.adapter as adapter_mod
    import openbench.runner.local as local_runner

    @dataclass
    class FakeRunnerConfig:
        basedir: str
        basename: str

    class ExplodingCompatBindings:
        def __init__(self):
            self.runner_cfg = FakeRunnerConfig(basedir=str(tmp_path), basename="dump_case")
            self.namelists = adapter_mod.LegacyNamelists(
                main={"general": {}},
                reference={"general": {}},
                simulation={"general": {}},
            )
            self.figures = adapter_mod.LegacyFigureConfig(raw={})

    monkeypatch.setattr(adapter_mod, "build_runner_bindings", lambda cfg: ExplodingCompatBindings())
    monkeypatch.setattr(
        local_runner,
        "run_evaluation",
        lambda cfg, force=False, comparison_only=False: {
            "status": "success",
            "output_dir": str(tmp_path / "dump_case"),
            "variables": ["Runoff"],
            "simulations": ["S1"],
        },
    )

    from openbench.data.registry import manager as mgr_mod

    class MockRegistry:
        def get_resolution_variants(self, name):
            return {}

        def get_reference(self, name, **kwargs):
            return SimpleNamespace(
                name=name,
                data_type="grid",
                tim_res="Month",
                grid_res=0.5,
                years=[2000, 2001],
                root_dir=str(tmp_path),
                variables={"Runoff": SimpleNamespace(varname="Runoff", varunit="", sub_dir="", prefix="", suffix="")},
                _provenance={},
            )

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: MockRegistry())

    result = runner.invoke(cli, ["run", "--dump-config", str(config_path)])

    debug_dir = tmp_path / "dump_case" / "debug"
    assert result.exit_code == 0, result.output
    assert (debug_dir / "runner_config.yaml").exists()
    assert not (debug_dir / "legacy_config.yaml").exists()


def test_ref_show_resolution_guidance_matches_target_resolution_context(monkeypatch):
    """ref show should explain variant selection via target resolution, not highest frequency."""
    from types import SimpleNamespace

    class MockRegistry:
        def get_resolution_variants(self, name):
            return {
                "LowRes": SimpleNamespace(
                    name="CARE_LowRes",
                    data_type="grid",
                    grid_res=0.5,
                    tim_res="Day",
                    years=[2000, 2010],
                    variables={"Runoff": object()},
                    root_dir="/data/low",
                ),
                "MidRes": SimpleNamespace(
                    name="CARE_MidRes",
                    data_type="grid",
                    grid_res=0.25,
                    tim_res="Month",
                    years=[2000, 2010],
                    variables={"Runoff": object()},
                    root_dir="/data/mid",
                ),
            }

        def get_reference(self, name):
            return None

    import openbench.data.registry as registry_pkg

    monkeypatch.setattr(registry_pkg, "RegistryManager", lambda: MockRegistry())

    result = runner.invoke(cli, ["ref", "show", "CARE"])
    assert result.exit_code == 0
    assert "project.tim_res" in result.output
    assert "project.grid_res" in result.output
    assert "shared simulation resolution" in result.output.lower()
    assert "highest frequency" not in result.output.lower()


def test_ref_show_base_name_with_single_variant_shows_dataset(monkeypatch):
    """ref show should resolve a base name when only one resolution variant exists."""
    from types import SimpleNamespace

    mapping = SimpleNamespace(varname="ro", varunit="mm day-1", fallbacks=[])
    ref = SimpleNamespace(
        name="AH4GUC_LowRes",
        description="AH4GUC test",
        category="Water",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        years=[2000, 2010],
        variables={"Runoff": mapping},
        root_dir="/data/ah4guc",
    )

    class MockRegistry:
        def get_resolution_variants(self, name):
            return {"LowRes": ref} if name == "AH4GUC" else {}

        def get_reference(self, name):
            return None

    import openbench.data.registry as registry_pkg

    monkeypatch.setattr(registry_pkg, "RegistryManager", lambda: MockRegistry())

    result = runner.invoke(cli, ["ref", "show", "AH4GUC"])

    assert result.exit_code == 0, result.output
    assert "AH4GUC_LowRes" in result.output
    assert "Runoff" in result.output


def test_ref_list_and_show_support_json_format(monkeypatch):
    """ref list/show should support machine-readable JSON like model list/show."""
    import json
    from types import SimpleNamespace

    mapping = SimpleNamespace(varname="ro", varunit="mm day-1", fallbacks=[])
    ref = SimpleNamespace(
        name="JSONRef",
        description="JSON test",
        category="Water",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        data_groupby="Year",
        timezone=0,
        years=[2000, 2010],
        variables={"Runoff": mapping},
        root_dir="/data/jsonref",
    )

    class MockRegistry:
        def list_references(self):
            return [ref]

        def references_for_variable(self, variable):
            return [ref] if variable == "Runoff" else []

        def get_resolution_variants(self, name):
            return {}

        def get_reference(self, name):
            return ref if name == "JSONRef" else None

    import openbench.data.registry as registry_pkg

    monkeypatch.setattr(registry_pkg, "RegistryManager", lambda: MockRegistry())

    list_result = runner.invoke(cli, ["ref", "list", "--format", "json"])
    show_result = runner.invoke(cli, ["ref", "show", "JSONRef", "--format", "json"])

    assert list_result.exit_code == 0, list_result.output
    assert show_result.exit_code == 0, show_result.output
    assert json.loads(list_result.output)[0]["name"] == "JSONRef"
    assert json.loads(show_result.output)["variables"]["Runoff"]["varname"] == "ro"


def test_parse_variables_reports_unclosed_quotes_as_click_exception():
    from openbench.cli._parsing import parse_variables

    result = None
    try:
        parse_variables(('Runoff name="unterminated unit=mm',))
    except Exception as exc:
        result = exc

    assert result is not None
    assert result.__class__.__name__ == "ClickException"
    assert "No closing quotation" in str(result)


def test_cache_clear_regrid_uses_explicit_directory(tmp_path):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    cache_dir = tmp_path / "regrid-cache"
    cache_dir.mkdir()
    (cache_dir / "weights-demo.npz").write_bytes(b"cache")
    (cache_dir / "notes.txt").write_text("keep", encoding="utf-8")

    result = CliRunner().invoke(cli, ["cache", "clear", "--regrid", "--dir", str(cache_dir), "--yes"])

    assert result.exit_code == 0, result.output
    assert not (cache_dir / "weights-demo.npz").exists()
    assert (cache_dir / "notes.txt").exists()


def test_cache_status_regrid_json_prunes_and_reports(tmp_path):
    import json

    from click.testing import CliRunner

    from openbench.cli.main import cli

    cache_dir = tmp_path / "regrid-cache"
    cache_dir.mkdir()
    (cache_dir / "weights-demo.npz").write_bytes(b"cache")

    result = CliRunner().invoke(cli, ["cache", "status", "--regrid", "--dir", str(cache_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["regrid"]["files"] == 1
    assert payload["regrid"]["bytes"] == 5


def test_cache_clear_regrid_prompt_shows_resolved_env_directory(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    cache_dir = tmp_path / "env-regrid-cache"
    cache_dir.mkdir()
    (cache_dir / "weights-demo.npz").write_bytes(b"cache")
    monkeypatch.setenv("OPENBENCH_REGRID_WEIGHT_CACHE_DIR", str(cache_dir))

    result = CliRunner().invoke(cli, ["cache", "clear", "--regrid"], input="n\n")

    assert result.exit_code != 0
    assert f"Delete regrid weight cache files in {cache_dir}?" in result.output
    assert (cache_dir / "weights-demo.npz").exists()


def test_ref_convert_old_command_uses_registry_converter(tmp_path):
    old_path = tmp_path / "old.yaml"
    out_path = tmp_path / "new.yaml"
    old_path.write_text(
        "general:\n"
        "  data_type: grid\n"
        "  tim_res: Month\n"
        "  syear: 2000\n"
        "  eyear: 2001\n"
        "Runoff:\n"
        "  varname: ro\n"
        "  varunit: mm day-1\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli,
        ["ref", "convert-old", str(old_path), str(out_path), "--name", "OldRunoff", "--category", "Water"],
    )

    assert result.exit_code == 0, result.output
    converted = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert converted["name"] == "OldRunoff"
    assert converted["category"] == "Water"
    assert converted["variables"]["Runoff"]["varname"] == "ro"
