"""Tests for registration of scanned datasets."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import openbench.cli.data as cli_data
import openbench.data.registry.manager as registry_manager_module
import openbench.data.registry.scanner as scanner_module
from openbench.data.registry.scanner import ScannedDataset, register_scanned_dataset


def test_register_scanned_dataset_does_not_persist_unverified_default_years(tmp_path: Path):
    catalog = tmp_path / "reference_catalog.yaml"
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )

    register_scanned_dataset(scanned, catalog_path=catalog)

    text = catalog.read_text(encoding="utf-8")
    assert "1980" not in text
    assert "2023" not in text
    assert "years:" not in text


def test_bundled_reference_catalog_has_no_scanner_temporary_variable_fields():
    catalog_path = Path(scanner_module.__file__).with_name("reference_catalog.yaml")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    leaks = []
    for dataset_name, descriptor in catalog.items():
        for variable_name, variable in (descriptor.get("variables") or {}).items():
            leaks.extend(f"{dataset_name}.{variable_name}.{field}" for field in variable if field.startswith("_"))

    assert leaks == []


def test_inspect_nc_file_detects_tim_res_from_time_dimension(tmp_path: Path):
    """_inspect_nc_file should detect tim_res from time[1] - time[0]."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    # Monthly data: ~30 day intervals
    times = pd.date_range("2000-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)
    ds = xr.Dataset(
        {"temp": (["time", "lat", "lon"], np.zeros((12, 5, 10)))},
        coords={"time": times, "lat": np.arange(5), "lon": np.arange(10)},
    )
    ds.to_netcdf(tmp_path / "temp_2000_test.nc")

    result = _inspect_nc_file(tmp_path)
    assert result.get("detected_tim_res") == "Month"

    # Daily data: 1 day intervals
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    times_d = pd.date_range("2000-01-01", periods=365, freq="D")
    ds_d = xr.Dataset(
        {"precip": (["time", "lat", "lon"], np.zeros((365, 5, 10)))},
        coords={"time": times_d, "lat": np.arange(5), "lon": np.arange(10)},
    )
    ds_d.to_netcdf(daily_dir / "precip_2000_daily.nc")

    result_d = _inspect_nc_file(daily_dir)
    assert result_d.get("detected_tim_res") == "Day"


def test_detect_tim_res_recognizes_30min_dataset_names(tmp_path: Path):
    """Half-hourly station folders should not fall back to default monthly metadata."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _detect_tim_res, _tim_res_rank

    half_hour_dir = tmp_path / "30min"
    half_hour_dir.mkdir()
    times = pd.date_range("2000-01-01", periods=4, freq="30min")
    ds = xr.Dataset(
        {"LE": (["station", "time"], np.zeros((1, len(times))))},
        coords={"station": [0], "time": times},
    )
    ds.to_netcdf(half_hour_dir / "OpenBench_FLUX_30min_full.nc")

    assert _detect_tim_res(half_hour_dir) == "30min"
    assert _tim_res_rank("30min") > _tim_res_rank("Hour")


def test_inspect_nc_file_marks_monthly_climatology_candidate_without_overriding_tim_res(
    tmp_path: Path,
):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    ds = xr.Dataset(
        {"temp": (["time", "lat", "lon"], np.zeros((12, 5, 10)))},
        coords={"time": times, "lat": np.arange(5), "lon": np.arange(10)},
    )
    ds.to_netcdf(tmp_path / "monthly_climatology.nc")

    result = _inspect_nc_file(tmp_path)

    assert result.get("detected_tim_res") == "Month"
    assert result.get("climatology_candidate") == "climatology-month"


def test_register_scanned_dataset_merges_existing_variable_descriptor_by_scanned_variable_key(
    tmp_path: Path,
):
    catalog = tmp_path / "reference_catalog.yaml"
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )
    existing_descriptor = {
        "name": "Demo_LowRes",
        "category": "Water",
        "data_type": "grid",
        "tim_res": "Month",
        "data_groupby": "Year",
        "timezone": 0,
        "years": [1990, 1991],
        "root_dir": "/legacy/root",
        "grid_res": 1.0,
        "variables": {
            "Evapotranspiration": {
                "varname": "ET",
                "varunit": "mm",
                "prefix": "pre_",
                "suffix": "_suf",
            },
            "NotUsed": {
                "varname": "SHOULD_NOT_APPLY",
                "varunit": "ignored",
            },
        },
    }

    register_scanned_dataset(
        scanned,
        catalog_path=catalog,
        existing_descriptor=existing_descriptor,
    )

    data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    descriptor = data["Demo_LowRes"]
    variable = descriptor["variables"]["Evapotranspiration"]

    assert variable["varname"] == "ET"
    assert variable["varunit"] == "mm"
    assert variable["prefix"] == "pre_"
    assert variable["suffix"] == "_suf"
    assert "NotUsed" not in descriptor["variables"]
    assert descriptor["root_dir"] == str(tmp_path)
    assert descriptor["grid_res"] == 0.5


def test_register_scanned_dataset_does_not_match_existing_variable_descriptors_by_varname(
    tmp_path: Path,
):
    catalog = tmp_path / "reference_catalog.yaml"
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )
    existing_descriptor = {
        "name": "Demo_LowRes",
        "variables": {
            "ET": {
                "varname": "ET",
                "varunit": "mm",
                "prefix": "pre_",
                "suffix": "_suf",
            }
        },
    }

    register_scanned_dataset(
        scanned,
        catalog_path=catalog,
        existing_descriptor=existing_descriptor,
    )

    data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    variable = data["Demo_LowRes"]["variables"]["Evapotranspiration"]

    assert variable["varname"] == "Evapotranspiration"
    assert variable["varunit"] == ""
    assert "prefix" not in variable
    assert "suffix" not in variable


def test_cli_scan_prefers_base_name_existing_descriptor_before_registry_name(
    monkeypatch,
    tmp_path: Path,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    variant = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )
    group = SimpleNamespace(base_name="Demo", variants={"LowRes": variant})

    calls = []
    captured = {}

    class FakeVar:
        def __init__(self, varname: str, varunit: str):
            self.varname = varname
            self.varunit = varunit
            self.prefix = ""
            self.suffix = ""

    class FakeRef:
        def __init__(self, label: str):
            self.variables = {label: FakeVar(label, f"{label}-unit")}

    class FakeRegistryManager:
        def __init__(self):
            pass

        def get_reference(self, name: str):
            calls.append(name)
            if name == variant.name:
                return FakeRef("base")
            if name == variant.registry_name:
                return FakeRef("variant")
            return None

    def fake_find_new_datasets(ref_root, on_progress=None, on_skip=None):
        return [group]

    def fake_register_batch(datasets, on_multi_var=None, on_progress=None, catalog_path=None):
        for ds in datasets:
            captured["scanned"] = ds.registry_name
        return tmp_path / "reference_catalog.yaml"

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register_batch)
    monkeypatch.setattr(cli_data.click, "echo", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_data.click, "secho", lambda *args, **kwargs: None)

    cli_data.scan.callback(str(tmp_path), auto=True, dry_run=False)

    assert captured["scanned"] == "Demo_LowRes"
    settings = yaml.safe_load((home / ".openbench" / "settings.yaml").read_text(encoding="utf-8"))
    assert settings["reference_root"] == str(tmp_path.resolve())


def test_cli_scan_rescan_registers_existing_variants(monkeypatch, tmp_path: Path):
    """--rescan should update scanned variants even when they already exist."""
    from openbench.data.registry.scanner import DatasetGroup

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    variant = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "."},
    )
    group = DatasetGroup(base_name="Demo", variants={"LowRes": variant})
    captured = {}

    def fake_scan_reference_directory(ref_root, on_progress=None, on_skip=None):
        captured["scanned_root"] = ref_root
        return [group]

    def fake_find_new_datasets(*args, **kwargs):
        raise AssertionError("--rescan must use scan_reference_directory, not find_new_datasets")

    def fake_register_batch(datasets, on_multi_var=None, on_progress=None, catalog_path=None):
        captured["registered"] = [ds.registry_name for ds in datasets]
        return tmp_path / "reference_catalog.yaml"

    monkeypatch.setattr(scanner_module, "scan_reference_directory", fake_scan_reference_directory)
    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register_batch)
    monkeypatch.setattr(cli_data.click, "echo", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_data.click, "secho", lambda *args, **kwargs: None)

    cli_data.scan.callback(str(tmp_path), auto=True, dry_run=False, rescan=True)

    assert captured["scanned_root"] == str(tmp_path)
    assert captured["registered"] == ["Demo_LowRes"]
    assert (home / ".openbench" / "settings.yaml").exists()


def test_cli_scan_rejects_file_ref_root(tmp_path: Path):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    ref_root = tmp_path / "not_a_directory"
    ref_root.write_text("not a reference tree")

    result = CliRunner().invoke(cli, ["ref", "scan", str(ref_root), "--dry-run"])

    assert result.exit_code != 0
    assert "directory" in result.output.lower()


def test_cli_scan_restores_ref_root_env_after_dry_run(monkeypatch, tmp_path: Path):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    result = CliRunner().invoke(cli, ["ref", "scan", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert "OPENBENCH_REF_ROOT" not in __import__("os").environ


def test_cli_scan_restores_previous_ref_root_env(monkeypatch, tmp_path: Path):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/old/reference/root")

    result = CliRunner().invoke(cli, ["ref", "scan", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert __import__("os").environ["OPENBENCH_REF_ROOT"] == "/old/reference/root"


def test_cli_scan_rejects_out_of_range_multi_variable_choice(monkeypatch, tmp_path: Path):
    from click.testing import CliRunner

    from openbench.cli.main import cli
    from openbench.data.registry.scanner import DatasetGroup

    home = tmp_path / "home"
    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    variant = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(ref_root),
        variables={"Runoff": "."},
        file_count=1,
    )
    group = DatasetGroup(base_name="Demo", variants={"LowRes": variant})
    selected = []

    def fake_find_new_datasets(ref_root, on_progress=None, on_skip=None):
        return [group]

    def fake_register_batch(datasets, on_multi_var=None, on_progress=None, catalog_path=None):
        selected.append(
            on_multi_var(
                "Runoff",
                ".",
                [
                    {"name": "primary", "unit": "", "dims": []},
                    {"name": "alternate", "unit": "", "dims": []},
                ],
            )
        )

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register_batch)

    result = CliRunner().invoke(cli, ["ref", "scan", str(ref_root)], input="y\n99\n")

    assert result.exit_code != 0
    assert "out of range" in result.output.lower()
    assert selected == []


def test_register_scanned_dataset_writes_station_list_next_to_catalog(tmp_path: Path):
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import register_scanned_dataset

    nc_root = tmp_path / "station_nc"
    nc_root.mkdir()
    ds = xr.Dataset(
        {
            "discharge": (["time"], np.array([1.0])),
            "lat": ([], 10.0),
            "lon": ([], 20.0),
        },
        coords={"time": np.array(["2001-01-01"], dtype="datetime64[ns]")},
    )
    ds.to_netcdf(nc_root / "station_2001.nc")

    catalog_path = tmp_path / "references" / "reference_catalog.yaml"
    scanned = ScannedDataset(
        name="DemoStation",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(nc_root),
        variables={"Streamflow": ""},
        file_count=1,
    )

    register_scanned_dataset(scanned, catalog_path=catalog_path)

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    expected = catalog_path.parent / "station_lists" / "DemoStation.csv"
    assert catalog["DemoStation"]["fulllist"] == expected.as_posix()
    assert expected.exists()


def test_register_scanned_builtin_identical_descriptor_does_not_shadow_user_overlay(
    tmp_path: Path,
    monkeypatch,
):
    from openbench.data.registry.scanner import register_scanned_datasets_batch

    package_registry = tmp_path / "package_registry"
    package_registry.mkdir()
    base_descriptor = {
        "name": "Demo_LowRes",
        "description": "Demo reference dataset (LowRes)",
        "category": "Water",
        "data_type": "grid",
        "tim_res": "Month",
        "data_groupby": "Year",
        "timezone": 0,
        "root_dir": str(tmp_path),
        "grid_res": 0.5,
        "variables": {
            "Runoff": {
                "varname": "Runoff",
                "varunit": "",
                "sub_dir": ".",
            }
        },
        "_provenance": {
            "data_groupby": "scan",
            "tim_res": "default",
            "grid_res": "default",
        },
    }
    (package_registry / "reference_catalog.yaml").write_text(yaml.safe_dump({"Demo_LowRes": base_descriptor}))
    (package_registry / "reference_profiles.yaml").write_text("{}\n")
    (package_registry / "model_catalog.yaml").write_text("{}\n")

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(registry_manager_module, "REGISTRY_DIR", package_registry)

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "."},
        file_count=1,
    )

    catalog_path = register_scanned_datasets_batch([scanned])

    assert yaml.safe_load(catalog_path.read_text(encoding="utf-8")) == {}


def test_batch_register_preserves_existing_descriptor_fields(tmp_path: Path):
    """Re-scanning should preserve hand-edited varname/varunit from existing catalog."""
    from openbench.data.registry.scanner import register_scanned_datasets_batch

    catalog_path = tmp_path / "reference_catalog.yaml"

    # Pre-populate catalog with hand-edited entry
    existing = {
        "Demo_LowRes": {
            "name": "Demo_LowRes",
            "data_type": "grid",
            "tim_res": "Month",
            "data_groupby": "Year",
            "root_dir": str(tmp_path),
            "variables": {
                "Evapotranspiration": {
                    "varname": "my_custom_et",
                    "varunit": "mm/day",
                    "prefix": "my_prefix_",
                    "suffix": "_my_suffix",
                    "sub_dir": "Water/Evapotranspiration/Demo",
                },
            },
        }
    }
    catalog_path.write_text(yaml.dump(existing))

    # Create a dataset dir with a dummy NC so scanner can inspect
    nc_dir = tmp_path / "Water" / "Evapotranspiration" / "Demo"
    nc_dir.mkdir(parents=True)
    import numpy as np
    import xarray as xr

    ds = xr.Dataset({"ET": (["time", "lat", "lon"], np.zeros((12, 2, 2)))})
    ds.to_netcdf(nc_dir / "ET_2000_demo.nc")

    variant = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )

    register_scanned_datasets_batch([variant], catalog_path=catalog_path)

    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    var = data["Demo_LowRes"]["variables"]["Evapotranspiration"]

    # Hand-edited fields should be preserved (not overwritten by scanner)
    assert var["varname"] == "my_custom_et"
    assert var["varunit"] == "mm/day"
    assert var["prefix"] == "my_prefix_"
    assert var["suffix"] == "_my_suffix"


def test_single_register_preserves_existing_descriptor_fields(tmp_path: Path):
    """Single-dataset registration should preserve existing edits like batch registration."""
    catalog_path = tmp_path / "reference_catalog.yaml"

    existing = {
        "Demo_LowRes": {
            "name": "Demo_LowRes",
            "description": "Hand edited description",
            "category": "Custom",
            "data_type": "grid",
            "tim_res": "Day",
            "data_groupby": "Year",
            "timezone": 8,
            "root_dir": str(tmp_path / "old"),
            "years": [1999, 2000],
            "variables": {
                "Runoff": {
                    "varname": "Q",
                    "varunit": "m3 s-1",
                    "prefix": "q_",
                    "suffix": "_daily",
                }
            },
        }
    }
    catalog_path.write_text(yaml.safe_dump(existing))

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "missing-dir"},
        tim_res="Month",
    )

    register_scanned_dataset(scanned, catalog_path=catalog_path)

    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["Demo_LowRes"]
    assert entry["description"] == "Hand edited description"
    assert entry["category"] == "Custom"
    assert entry["timezone"] == 8
    assert entry["years"] == [1999, 2000]
    assert entry["variables"]["Runoff"]["varname"] == "Q"
    assert entry["variables"]["Runoff"]["varunit"] == "m3 s-1"
    assert entry["variables"]["Runoff"]["prefix"] == "q_"
    assert entry["variables"]["Runoff"]["suffix"] == "_daily"


def test_register_profile_uses_writable_references_dir_and_clears_caches(
    monkeypatch,
    tmp_path: Path,
):
    writable_profile = tmp_path / "user-registry" / "references" / "reference_profiles.yaml"
    fake_package_cli = tmp_path / "pkg" / "cli" / "data.py"
    fake_package_cli.parent.mkdir(parents=True)
    fake_package_registry = fake_package_cli.parent.parent / "data" / "registry"
    fake_package_registry.mkdir(parents=True)

    registry_cleared = []
    profile_cleared = []

    monkeypatch.setattr(cli_data, "__file__", str(fake_package_cli))
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_reference_profiles_path",
        lambda: writable_profile,
    )
    monkeypatch.setattr(
        registry_manager_module,
        "clear_registry_cache",
        lambda: registry_cleared.append(True),
    )
    monkeypatch.setattr(
        scanner_module,
        "clear_reference_profile_cache",
        lambda: profile_cleared.append(True),
        raising=False,
    )

    cli_data.register_profile.callback(
        "DemoProfile",
        variable=("Runoff:ro:mm day-1",),
        tim_res="Month",
        data_groupby="Year",
        fulllist=None,
        description=None,
    )

    assert writable_profile.exists()
    assert not (fake_package_registry / "reference_profiles.yaml").exists()
    saved = yaml.safe_load(writable_profile.read_text(encoding="utf-8"))
    assert saved["DemoProfile"]["variables"]["Runoff"]["varname"] == "ro"
    assert registry_cleared == [True]
    assert profile_cleared == [True]


def test_reference_profiles_load_from_writable_references_dir(monkeypatch, tmp_path: Path):
    writable_profile = tmp_path / "user-registry" / "references" / "reference_profiles.yaml"
    writable_profile.parent.mkdir(parents=True)
    fake_scanner_file = tmp_path / "pkg" / "registry" / "scanner.py"
    fake_scanner_file.parent.mkdir(parents=True)

    writable_profile.write_text(
        yaml.dump(
            {
                "DemoProfile": {
                    "variables": {
                        "Runoff": {"varname": "ro", "varunit": "mm day-1"},
                    }
                }
            }
        )
    )

    # The package profile lookup now goes through importlib.resources
    # via `_package_reference_profiles_path`, which doesn't honour
    # `__file__` overrides. Patch the helper directly. This test wants
    # the package profile to be absent so only the writable file is
    # consulted — point at a non-existent path under the fake scanner
    # directory.
    monkeypatch.setattr(scanner_module, "__file__", str(fake_scanner_file))
    monkeypatch.setattr(
        scanner_module,
        "_package_reference_profiles_path",
        lambda: fake_scanner_file.parent / "reference_profiles.yaml",
    )
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_reference_profiles_path",
        lambda: writable_profile,
    )
    scanner_module._REFERENCE_PROFILES = None

    profile = scanner_module.get_reference_profile("DemoProfile_LowRes")

    assert profile is not None
    assert profile["variables"]["Runoff"]["varname"] == "ro"


def test_reference_profiles_user_overlay_deep_merges_package_profile(monkeypatch, tmp_path: Path):
    """User reference_profiles.yaml should update package profiles, not replace them."""
    package_registry = tmp_path / "pkg" / "registry"
    package_registry.mkdir(parents=True)
    fake_scanner_file = package_registry / "scanner.py"
    fake_scanner_file.write_text("")
    package_profile = package_registry / "reference_profiles.yaml"
    package_profile.write_text(
        yaml.dump(
            {
                "DemoProfile": {
                    "description": "package profile",
                    "tim_res": "Day",
                    "variables": {
                        "Runoff": {"varname": "ro", "varunit": "mm day-1"},
                        "Latent_Heat": {"varname": "LE", "varunit": "W m-2"},
                    },
                }
            }
        )
    )

    writable_profile = tmp_path / "user-registry" / "references" / "reference_profiles.yaml"
    writable_profile.parent.mkdir(parents=True)
    writable_profile.write_text(
        yaml.dump(
            {
                "DemoProfile": {
                    "variables": {
                        "Runoff": {"varname": "user_ro"},
                    },
                }
            }
        )
    )

    # The package profile lookup now goes through importlib.resources
    # via `_package_reference_profiles_path`, which doesn't honour
    # `__file__` overrides. Patch the helper directly.
    monkeypatch.setattr(scanner_module, "__file__", str(fake_scanner_file))
    monkeypatch.setattr(
        scanner_module,
        "_package_reference_profiles_path",
        lambda: package_registry / "reference_profiles.yaml",
    )
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_reference_profiles_path",
        lambda: writable_profile,
    )
    scanner_module._REFERENCE_PROFILES = None

    profile = scanner_module.get_reference_profile("DemoProfile")

    assert profile["description"] == "package profile"
    assert profile["tim_res"] == "Day"
    assert profile["variables"]["Runoff"] == {
        "varname": "user_ro",
        "varunit": "mm day-1",
    }
    assert profile["variables"]["Latent_Heat"]["varname"] == "LE"


def test_reference_profiles_invalid_yaml_logs_warning(monkeypatch, tmp_path: Path, caplog):
    """Malformed user/profile YAML should be visible instead of silently ignored."""
    import logging

    package_registry = tmp_path / "pkg" / "registry"
    package_registry.mkdir(parents=True)
    fake_scanner_file = package_registry / "scanner.py"
    fake_scanner_file.write_text("")
    (package_registry / "reference_profiles.yaml").write_text("bad: [unclosed\n")

    writable_profile = tmp_path / "user-registry" / "references" / "reference_profiles.yaml"
    # The package profile lookup now goes through importlib.resources
    # via `_package_reference_profiles_path`, which doesn't honour
    # `__file__` overrides. Patch the helper directly.
    monkeypatch.setattr(scanner_module, "__file__", str(fake_scanner_file))
    monkeypatch.setattr(
        scanner_module,
        "_package_reference_profiles_path",
        lambda: package_registry / "reference_profiles.yaml",
    )
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_reference_profiles_path",
        lambda: writable_profile,
    )
    scanner_module._REFERENCE_PROFILES = None

    with caplog.at_level(logging.WARNING):
        profile = scanner_module.get_reference_profile("DemoProfile")

    assert profile is None
    assert "Failed to load reference profiles" in caplog.text
    assert "reference_profiles.yaml" in caplog.text


def test_reference_profiles_load_legacy_user_root_path(monkeypatch, tmp_path: Path):
    """Existing ~/.openbench/reference_profiles.yaml files remain readable."""
    writable_profile = tmp_path / "user-registry" / "references" / "reference_profiles.yaml"
    legacy_profile = tmp_path / "user-registry" / "reference_profiles.yaml"
    legacy_profile.parent.mkdir(parents=True)
    fake_scanner_file = tmp_path / "pkg" / "registry" / "scanner.py"
    fake_scanner_file.parent.mkdir(parents=True)

    legacy_profile.write_text(
        yaml.dump(
            {
                "LegacyProfile": {
                    "variables": {
                        "Runoff": {"varname": "legacy_ro", "varunit": "mm day-1"},
                    }
                }
            }
        )
    )

    # The package profile lookup now goes through importlib.resources
    # via `_package_reference_profiles_path`, which doesn't honour
    # `__file__` overrides. Patch the helper directly. This test wants
    # the package profile to NOT exist so the legacy/writable path
    # fallback runs — point at a non-existent file.
    monkeypatch.setattr(scanner_module, "__file__", str(fake_scanner_file))
    monkeypatch.setattr(
        scanner_module,
        "_package_reference_profiles_path",
        lambda: fake_scanner_file.parent / "reference_profiles.yaml",
    )
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_reference_profiles_path",
        lambda: writable_profile,
    )
    monkeypatch.setattr(
        registry_manager_module,
        "get_legacy_reference_profiles_path",
        lambda: legacy_profile,
        raising=False,
    )
    scanner_module._REFERENCE_PROFILES = None

    profile = scanner_module.get_reference_profile("LegacyProfile")

    assert profile is not None
    assert profile["variables"]["Runoff"]["varname"] == "legacy_ro"


def test_provenance_reaches_reference_dataset_from_catalog(tmp_path: Path):
    """_provenance written by scanner must be loadable via RegistryManager."""
    from openbench.data.registry.scanner import register_scanned_datasets_batch

    # RegistryManager looks for user references at user_dir/references/reference_catalog.yaml
    references_dir = tmp_path / "references"
    references_dir.mkdir()
    catalog_path = references_dir / "reference_catalog.yaml"

    variant = ScannedDataset(
        name="ProvTest",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Water/Runoff/ProvTest"},
    )

    register_scanned_datasets_batch([variant], catalog_path=catalog_path)

    # Verify provenance is written by the scanner.
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    prov = data["ProvTest_LowRes"].get("_provenance")
    assert prov is not None
    assert "tim_res" in prov

    # Verify provenance survives the load chain (catalog load + dir-glob
    # second-pass merge through _deep_merge_reference).
    from openbench.data.registry.manager import RegistryManager

    mgr = RegistryManager(user_dir=tmp_path)
    ref = mgr.get_reference("ProvTest_LowRes")
    assert ref is not None
    assert ref._provenance is not None
    assert ref._provenance["tim_res"] == "default"
    assert ref._provenance["grid_res"] == "default"


def test_data_type_correction_grid_to_stn_generates_fulllist(tmp_path: Path):
    """When NC content says stn but directory says grid, descriptor should
    have no grid_res and should attempt fulllist generation."""
    from openbench.data.registry.scanner import _register_to_dict

    # Create a station-like NC file (lat=1, lon=1) in a "grid" scanned dataset
    nc_dir = tmp_path / "Water" / "Evapotranspiration" / "FakeGrid"
    nc_dir.mkdir(parents=True)
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {
            "ET": (["time", "y", "x"], np.zeros((12, 1, 1))),
            "lat": ([], 47.0),
            "lon": ([], 11.0),
        }
    )
    ds.to_netcdf(nc_dir / "FakeGrid_2004_2005.nc")

    variant = ScannedDataset(
        name="FakeGrid",
        resolution="LowRes",
        category="Water",
        data_type="grid",  # directory says grid
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/FakeGrid"},
    )

    catalog = {}
    _register_to_dict(variant, catalog)
    entry = catalog["FakeGrid_LowRes"]

    # NC detection should have corrected to stn
    assert entry["data_type"] == "stn"
    # grid_res should be removed
    assert "grid_res" not in entry
    # provenance should reflect NC correction
    assert entry["_provenance"].get("data_type") == "nc"


def test_data_type_correction_stn_to_grid_adds_grid_res(tmp_path: Path):
    """When NC content says grid but directory says stn, descriptor should
    get grid_res and no fulllist."""
    from openbench.data.registry.scanner import _register_to_dict

    # Create a grid-like NC file (lat=180, lon=360) in a "stn" scanned dataset
    nc_dir = tmp_path / "Heat" / "Sensible_Heat" / "FakeStn"
    nc_dir.mkdir(parents=True)
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {
            "H": (["time", "lat", "lon"], np.zeros((12, 180, 360))),
        },
        coords={
            "lat": np.linspace(-89.5, 89.5, 180),
            "lon": np.linspace(-179.5, 179.5, 360),
        },
    )
    ds.to_netcdf(nc_dir / "H_2004_fakestn.nc")

    variant = ScannedDataset(
        name="FakeStn",
        resolution="Station",
        category="Heat",
        data_type="stn",  # directory says stn
        root_dir=str(tmp_path),
        variables={"Sensible_Heat": "Heat/Sensible_Heat/FakeStn"},
    )

    catalog = {}
    _register_to_dict(variant, catalog)
    entry = catalog["FakeStn"]

    # NC detection should have corrected to grid
    assert entry["data_type"] == "grid"
    # grid_res should be present
    assert "grid_res" in entry
    # No fulllist for grid data
    assert "fulllist" not in entry
    # provenance should reflect NC correction
    assert entry["_provenance"].get("data_type") == "nc"


def test_scanner_discovers_nc4_files(tmp_path: Path):
    """Scanner should count and inspect .nc4 files the same as .nc."""
    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Runoff" / "TestNC4"
    nc_dir.mkdir(parents=True)
    import numpy as np
    import xarray as xr

    for year in (2000, 2001):
        ds = xr.Dataset(
            {"ro": (["time", "lat", "lon"], np.zeros((12, 10, 20)))},
            coords={"lat": np.arange(10), "lon": np.arange(20)},
        )
        ds.to_netcdf(nc_dir / f"ro_{year}_test.nc4")

    variant = ScannedDataset(
        name="TestNC4",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Water/Runoff/TestNC4"},
    )

    catalog = {}
    _register_to_dict(variant, catalog)
    entry = catalog["TestNC4_LowRes"]

    # Should detect as grid (not Single, since 2 files)
    assert entry["data_type"] == "grid"
    assert entry["data_groupby"] != "Single"
    # Should have inspected the .nc4 file and found varname
    assert entry["variables"]["Runoff"]["varname"] == "ro"


def test_find_data_files_finds_nc4(tmp_path: Path):
    """_find_data_files and _find_single_file should find .nc4 files."""
    import numpy as np
    import xarray as xr

    from openbench.data.processing import BaseDatasetProcessing

    # Create a .nc4 file (not .nc)
    ds = xr.Dataset({"ET": (["time", "lat", "lon"], np.zeros((12, 5, 5)))})
    ds.to_netcdf(tmp_path / "prefix_2000_suffix.nc4")

    # Minimal processor setup
    class FakeProcessor(BaseDatasetProcessing):
        pass

    proc = FakeProcessor.__new__(FakeProcessor)
    proc.sim_source = "sim"

    # _find_data_files should find the .nc4
    found = proc._find_data_files(str(tmp_path), "prefix_", 2000, "_suffix", "sim")
    assert len(found) > 0, "Should find .nc4 file"

    # _find_single_file for a single file
    single = tmp_path / "single_data.nc4"
    ds.to_netcdf(single)
    path = proc._find_single_file(str(tmp_path), "single_data", "", "sim")
    assert path.endswith(".nc4")


def test_inspect_nc_file_detects_grid_res_from_lat_dimension(tmp_path: Path):
    """_inspect_nc_file should detect grid_res from lat[1] - lat[0]."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    # 0.25 degree grid
    ds = xr.Dataset(
        {"var": (["time", "lat", "lon"], np.zeros((12, 720, 1440)))},
        coords={
            "lat": np.linspace(-89.875, 89.875, 720),
            "lon": np.linspace(-179.875, 179.875, 1440),
        },
    )
    ds.to_netcdf(tmp_path / "var_2000_test.nc")

    result = _inspect_nc_file(tmp_path)
    assert result.get("detected_grid_res") is not None
    assert abs(result["detected_grid_res"] - 0.25) < 0.01


def test_register_to_dict_nc_tim_res_overrides_profile(tmp_path: Path):
    """NC-detected tim_res should win over profile tim_res."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    # Create a daily NC file in a dataset that has a profile saying tim_res=Day
    nc_dir = tmp_path / "Water" / "StreamFlow" / "Daily"
    nc_dir.mkdir(parents=True)
    times = pd.date_range("2000-01-01", periods=365, freq="D")
    ds = xr.Dataset(
        {"discharge": (["time", "lat", "lon"], np.zeros((365, 5, 10)))},
        coords={"time": times, "lat": np.arange(5), "lon": np.arange(10)},
    )
    ds.to_netcdf(nc_dir / "discharge_2000_daily.nc")

    variant = ScannedDataset(
        name="Daily",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"StreamFlow": "Water/StreamFlow/Daily"},
    )

    catalog = {}
    _register_to_dict(variant, catalog)
    entry = catalog["Daily"]

    # NC detection (1-day interval) should take priority
    assert entry["tim_res"] == "Day"
    assert entry["_provenance"]["tim_res"] == "nc"


def test_register_to_dict_warns_when_nc_tim_res_conflicts_with_filename(
    tmp_path: Path,
    caplog,
):
    """NC tim_res stays authoritative, but filename/NC disagreement should be visible."""
    import logging

    import netCDF4

    from openbench.data.registry.scanner import _detect_tim_res, _register_to_dict

    nc_dir = tmp_path / "Water" / "AirTemp" / "Demo"
    nc_dir.mkdir(parents=True)
    with netCDF4.Dataset(nc_dir / "Tair_hourly_2020.nc", "w") as nc:
        nc.createDimension("time", 2)
        nc.createDimension("lat", 2)
        nc.createDimension("lon", 2)
        time = nc.createVariable("time", "f8", ("time",))
        time.units = "days since 2020-01-01"
        time[:] = [0, 31]
        lat = nc.createVariable("lat", "f4", ("lat",))
        lat[:] = [0.0, 1.0]
        lon = nc.createVariable("lon", "f4", ("lon",))
        lon[:] = [0.0, 1.0]
        tair = nc.createVariable("Tair", "f4", ("time", "lat", "lon"))
        tair.units = "K"

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Air_Temperature": "Water/AirTemp/Demo"},
        tim_res=_detect_tim_res(nc_dir),
    )

    catalog: dict = {}
    with caplog.at_level(logging.WARNING, logger="openbench.data.registry.scanner"):
        _register_to_dict(scanned, catalog)

    entry = catalog["Demo_LowRes"]
    assert scanned.tim_res == "Hour"
    assert entry["tim_res"] == "Month"
    assert entry["_provenance"]["tim_res"] == "nc"
    assert "filename/path suggests tim_res='Hour'" in caplog.text
    assert "NC time coordinate suggests 'Month'" in caplog.text


def test_register_to_dict_warns_and_does_not_persist_monthly_climatology_candidate(
    tmp_path: Path,
    caplog,
):
    import logging

    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _detect_tim_res, _register_to_dict

    nc_dir = tmp_path / "Water" / "AirTemp" / "MonthlyClimatology"
    nc_dir.mkdir(parents=True)
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    ds = xr.Dataset(
        {"Tair": (["time", "lat", "lon"], np.zeros((12, 2, 2)))},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds.to_netcdf(nc_dir / "Tair_monthly_climatology.nc")

    scanned = ScannedDataset(
        name="MonthlyClimatology",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Air_Temperature": "Water/AirTemp/MonthlyClimatology"},
        tim_res=_detect_tim_res(nc_dir),
    )

    catalog: dict = {}
    with caplog.at_level(logging.WARNING, logger="openbench.data.registry.scanner"):
        _register_to_dict(scanned, catalog)

    entry = catalog["MonthlyClimatology_LowRes"]
    assert entry["tim_res"] == "Month"
    assert "_climatology_candidate" not in entry["variables"]["Air_Temperature"]
    assert "climatology-month" in caplog.text
    assert "explicit confirmation" in caplog.text


def test_register_to_dict_native_nc_grid_res_wins_over_resolution_bucket(tmp_path: Path):
    """Catalog grid_res should describe native NC spacing, not the directory bucket."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Runoff" / "TestRes"
    nc_dir.mkdir(parents=True)
    # 0.25 degree grid, but in a "LowRes" dataset (RESOLUTION_MAP says 0.5)
    ds = xr.Dataset(
        {"ro": (["time", "lat", "lon"], np.zeros((12, 720, 1440)))},
        coords={
            "lat": np.linspace(-89.875, 89.875, 720),
            "lon": np.linspace(-179.875, 179.875, 1440),
        },
    )
    ds.to_netcdf(nc_dir / "ro_2000_test.nc")

    variant = ScannedDataset(
        name="TestRes",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Water/Runoff/TestRes"},
    )

    catalog = {}
    _register_to_dict(variant, catalog)
    entry = catalog["TestRes_LowRes"]

    # NC says 0.25, RESOLUTION_MAP["LowRes"] says 0.5 — NC wins.
    assert abs(entry["grid_res"] - 0.25) < 0.01
    assert entry["_provenance"]["grid_res"] == "nc"


# ---------------------------------------------------------------------------
# Regression tests for 4 batch-fixed bugs (catalog-corruption, nested NC,
# partial-new variants, 1D station vars).
# ---------------------------------------------------------------------------


def test_register_refuses_overwrite_when_catalog_unparseable(tmp_path: Path):
    """Corrupted catalog must raise rather than silently overwrite with empty
    + new entry (which would delete all previously registered datasets).
    """
    import pytest

    catalog_path = tmp_path / "reference_catalog.yaml"
    # Write corrupted YAML
    catalog_path.write_text("not: valid: yaml: ::: garbage\n[broken")
    original = catalog_path.read_text(encoding="utf-8")

    scanned = ScannedDataset(
        name="DemoNew",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "."},
    )

    with pytest.raises(RuntimeError, match="Failed to load existing catalog"):
        register_scanned_dataset(scanned, catalog_path=catalog_path)

    # Original (corrupted) file untouched — user can recover manually
    assert catalog_path.read_text(encoding="utf-8") == original


def test_register_creates_backup_before_overwriting_existing_catalog(tmp_path: Path):
    """Each successful write must back up the previous catalog state to .bak,
    so a buggy rescan that overwrites hand-edited fields can be undone.
    """
    catalog_path = tmp_path / "reference_catalog.yaml"
    catalog_path.write_text("OldDataset_LowRes:\n  name: OldDataset_LowRes\n  category: Water\n  data_type: grid\n")

    scanned = ScannedDataset(
        name="NewDataset",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "."},
    )
    register_scanned_dataset(scanned, catalog_path=catalog_path)

    backup_path = Path(str(catalog_path) + ".bak")
    assert backup_path.exists(), f"Expected backup at {backup_path}"
    bak_data = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    assert "OldDataset_LowRes" in bak_data
    # New file has both
    new_data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert "OldDataset_LowRes" in new_data
    assert "NewDataset_LowRes" in new_data


def test_scan_uses_child_dir_when_nc_files_one_level_deep(tmp_path: Path):
    """When NC files live at dataset_dir/<single_child>/*.nc, both sub_dir
    AND tim_res detection must use the child path. Otherwise inspect_nc_file
    finds no files and returns empty descriptor (varname/prefix/suffix lost).
    """
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    nested_dir = ref_root / "Grid" / "LowRes" / "Water" / "Evapotranspiration" / "MyData" / "0p25deg-daily"
    nested_dir.mkdir(parents=True)

    # Daily data: 365 time steps, lat/lon present
    times = xr.date_range("2010-01-01", periods=365, freq="D", use_cftime=True)
    ds = xr.Dataset(
        {"ET_actual": (["time", "lat", "lon"], np.zeros((365, 4, 4), dtype=np.float32))},
        coords={"time": times, "lat": np.arange(4.0), "lon": np.arange(4.0)},
    )
    ds["ET_actual"].attrs["units"] = "mm/day"
    ds.to_netcdf(nested_dir / "ET_2010_daily.nc")

    groups = scan_reference_directory(ref_root)
    assert len(groups) == 1, f"Expected one group, got: {[g.base_name for g in groups]}"
    variant = groups[0].variants["LowRes"]

    sub_dir = variant.variables["Evapotranspiration"]
    assert "0p25deg-daily" in sub_dir, f"sub_dir must point to NC-bearing child directory, got: {sub_dir!r}"
    # tim_res detected from filename in the child (contains "daily")
    assert variant.tim_res == "Day", f"Expected 'Day', got: {variant.tim_res!r}"


def test_scan_skips_dataset_with_multiple_nc_bearing_children(tmp_path: Path, caplog):
    """When dataset_dir/<multi>/*.nc has multiple NC-bearing children
    (composite/multi-variant), the scanner must skip with a warning rather
    than merge into a single descriptor (which loses per-child metadata).
    """
    import logging

    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    base = ref_root / "Grid" / "LowRes" / "Crop" / "Crop_Yield" / "GDHY2019ver"
    for sub in ("maize", "soybean", "wheat"):
        sub_dir = base / sub
        sub_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {sub: (["lat", "lon"], np.zeros((4, 4), dtype=np.float32))},
            coords={"lat": np.arange(4.0), "lon": np.arange(4.0)},
        )
        ds.to_netcdf(sub_dir / f"{sub}_2010.nc")

    skipped = []
    with caplog.at_level(logging.WARNING, logger="openbench.data.registry.scanner"):
        groups = scan_reference_directory(ref_root, on_skip=skipped.append)

    assert groups == [], f"Multi-child dataset must be skipped, got: {[g.base_name for g in groups]}"
    assert skipped
    assert skipped[0].path == "Grid/LowRes/Crop/Crop_Yield/GDHY2019ver"
    assert skipped[0].reason == "ambiguous_nc_subdirectories"
    assert any("NC-bearing subdirectories" in rec.message for rec in caplog.records), (
        f"Expected ambiguity warning, got messages: {[r.message for r in caplog.records]}"
    )


def test_find_new_datasets_filters_already_registered_variants_from_group(tmp_path: Path, monkeypatch):
    """When a group has both existing (LowRes) and new (MidRes) variants,
    find_new_datasets must return ONLY the new variant. Otherwise the CLI
    re-registers existing variants and overwrites top-level descriptor
    fields (category, fulllist, etc.) the user may have hand-edited.
    """
    from openbench.data.registry import scanner as scanner_module
    from openbench.data.registry.scanner import (
        DatasetGroup,
        ScannedDataset,
        find_new_datasets,
    )

    group = DatasetGroup(base_name="Demo")
    group.variants["LowRes"] = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
    )
    group.variants["MidRes"] = ScannedDataset(
        name="Demo",
        resolution="MidRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
    )

    monkeypatch.setattr(scanner_module, "scan_reference_directory", lambda *a, **k: [group])

    # Demo_LowRes already registered, Demo_MidRes is new
    new_groups = find_new_datasets(tmp_path, existing_names={"Demo_LowRes"})

    assert len(new_groups) == 1
    new_group = new_groups[0]
    # Only MidRes should appear, NOT LowRes
    assert set(new_group.variants.keys()) == {"MidRes"}, (
        f"find_new_datasets must filter out already-registered variants, "
        f"got variants: {list(new_group.variants.keys())}"
    )


def test_inspect_nc_file_picks_1d_data_var_for_station(tmp_path: Path):
    """Single-station NC files often have 1D data variables like
    Qle_cor(time). The previous filter required >= 2 dimensions, which
    silently dropped the actual data variable and made the registered
    descriptor's varname default to the directory name (wrong NC key).
    """
    import netCDF4

    nc_file = tmp_path / "Stn123_2010.nc"
    with netCDF4.Dataset(nc_file, "w") as nc:
        nc.createDimension("time", 365)
        time_var = nc.createVariable("time", "f4", ("time",))
        time_var.units = "days since 2010-01-01"
        time_var[:] = list(range(365))
        # 1D station data variable — what should be detected
        qle = nc.createVariable("Qle_cor", "f4", ("time",))
        qle.units = "W/m2"
        # Station metadata as scalar (lat/lon often stored this way per-file)
        lat = nc.createVariable("lat", "f4", ())
        lat.units = "degrees_north"
        lon = nc.createVariable("lon", "f4", ())
        lon.units = "degrees_east"

    from openbench.data.registry.scanner import _inspect_nc_file

    info = _inspect_nc_file(tmp_path)

    assert info.get("detected_data_type") == "stn", f"Expected stn detection, got: {info.get('detected_data_type')!r}"
    assert info.get("varname") == "Qle_cor", (
        f"Expected Qle_cor as data var, got: {info.get('varname')!r}. 1D station variables must not be filtered out."
    )
    assert info.get("varunit") == "W/m2"


def test_inspect_nc_file_ignores_masked_time_values_without_warning(tmp_path: Path):
    """Unset time values should not leak netCDF masked-value warnings."""
    import warnings

    import netCDF4

    nc_file = tmp_path / "Stn123_2010.nc"
    with netCDF4.Dataset(nc_file, "w") as nc:
        nc.createDimension("time", 2)
        time_var = nc.createVariable("time", "f4", ("time",))
        time_var.units = "days since 2010-01-01"
        qle = nc.createVariable("Qle_cor", "f4", ("time",))
        qle.units = "W/m2"

    from openbench.data.registry.scanner import _inspect_nc_file

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = _inspect_nc_file(tmp_path)

    masked_warnings = [warning for warning in caught if "masked element" in str(warning.message)]
    assert masked_warnings == []
    assert "detected_tim_res" not in info
    assert info.get("varname") == "Qle_cor"


# ---------------------------------------------------------------------------
# Regression tests for batch 2 (rescan preservation, data_groupby detection,
# tim_res honest provenance, write-cache invalidation, year-regex tightening).
# ---------------------------------------------------------------------------


def test_rescan_preserves_user_edited_descriptor_fields(tmp_path: Path):
    """Description / category / years that the user hand-edits in the catalog
    must NOT be silently overwritten when the scanner re-registers the same
    dataset (e.g., when a new resolution variant is added to the same group).
    """
    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Evapotranspiration" / "Demo"
    nc_dir.mkdir(parents=True)
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {"ET": (["time", "lat", "lon"], np.zeros((12, 4, 4), dtype=np.float32))},
        coords={"time": np.arange(12), "lat": np.arange(4.0), "lon": np.arange(4.0)},
    )
    ds.to_netcdf(nc_dir / "ET_2004_2005.nc")

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )

    # First registration: descriptor entirely from scan
    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    # User edits these fields by hand:
    catalog["Demo_LowRes"]["description"] = "Custom user description for Demo"
    catalog["Demo_LowRes"]["category"] = "Energy"  # user re-categorized
    catalog["Demo_LowRes"]["years"] = [2000, 2010]  # user fixed years

    # Re-register (same group; existing_descriptor passed in like batch path does)
    existing = catalog["Demo_LowRes"]
    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    new_entry = catalog["Demo_LowRes"]

    assert new_entry["description"] == "Custom user description for Demo"
    assert new_entry["category"] == "Energy"
    assert new_entry["years"] == [2000, 2010]


def test_rescan_reinspects_nc_for_technical_provenance_and_keeps_user_fields(tmp_path: Path):
    """Rescan must preserve user-owned variable fields without downgrading
    NC-derived tim_res/grid_res provenance.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Runoff" / "DailyDemo"
    nc_dir.mkdir(parents=True)
    times = pd.date_range("2001-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {"ro": (["time", "lat", "lon"], np.zeros((3, 2, 2), dtype=np.float32))},
        coords={"time": times, "lat": [0.0, 0.25], "lon": [0.0, 0.25]},
    )
    ds.to_netcdf(nc_dir / "ro_2001_daily.nc")

    scanned = ScannedDataset(
        name="DailyDemo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Water/Runoff/DailyDemo"},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    existing = catalog["DailyDemo_LowRes"]
    existing["variables"]["Runoff"]["varname"] = "user_ro"
    existing["variables"]["Runoff"]["prefix"] = "user_prefix_"

    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    rescanned = catalog["DailyDemo_LowRes"]

    assert rescanned["tim_res"] == "Day"
    assert rescanned["_provenance"]["tim_res"] == "nc"
    assert abs(rescanned["grid_res"] - 0.25) < 0.01
    assert rescanned["_provenance"]["grid_res"] == "nc"
    assert rescanned["variables"]["Runoff"]["varname"] == "user_ro"
    assert rescanned["variables"]["Runoff"]["prefix"] == "user_prefix_"


def test_rescan_keeps_years_field_order_stable(tmp_path: Path):
    """When existing years are preserved, rescan should not move the field to
    the end of the YAML descriptor.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Runoff" / "YearOrderDemo"
    nc_dir.mkdir(parents=True)
    times = pd.date_range("2010-01-01", periods=2, freq="D")
    ds = xr.Dataset(
        {"ro": (["time", "lat", "lon"], np.zeros((2, 2, 2), dtype=np.float32))},
        coords={"time": times, "lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "ro_2010.nc")

    scanned = ScannedDataset(
        name="YearOrderDemo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Water/Runoff/YearOrderDemo"},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    existing = catalog["YearOrderDemo_LowRes"]

    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    keys = list(catalog["YearOrderDemo_LowRes"].keys())

    assert keys.index("years") < keys.index("_provenance")


def test_station_multi_file_scan_does_not_write_single_station_prefix_suffix(tmp_path: Path):
    """One-file-per-station datasets use fulllist paths; a prefix from the
    alphabetically first station file is not a dataset-wide pattern.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Carbon" / "CH4_Flux" / "MyStn"
    nc_dir.mkdir(parents=True)
    for site in ("AU_Tum", "DE_Tha", "US_Ha1"):
        ds = xr.Dataset(
            {
                "ch4": (["time"], np.zeros(2, dtype=np.float32)),
                "lat": ([], 1.0),
                "lon": ([], 2.0),
            },
            coords={"time": pd.date_range("2010-01-01", periods=2, freq="D")},
            attrs={"station_id": site},
        )
        ds.to_netcdf(nc_dir / f"{site}_2010_2014.nc")

    scanned = ScannedDataset(
        name="MyStn",
        resolution="Station",
        category="Carbon",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"CH4_Flux": "Carbon/CH4_Flux/MyStn"},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    variable = catalog["MyStn"]["variables"]["CH4_Flux"]

    assert "prefix" not in variable
    assert "suffix" not in variable


def test_station_multi_file_rescan_drops_existing_single_station_prefix_suffix(tmp_path: Path):
    """Rescan should clean up prefix/suffix values produced by the old
    first-station-file inference for one-file-per-station datasets.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Carbon" / "CH4_Flux" / "MyStn"
    nc_dir.mkdir(parents=True)
    for site in ("AU_Tum", "DE_Tha", "US_Ha1"):
        ds = xr.Dataset(
            {
                "ch4": (["time"], np.zeros(2, dtype=np.float32)),
                "lat": ([], 1.0),
                "lon": ([], 2.0),
            },
            coords={"time": pd.date_range("2010-01-01", periods=2, freq="D")},
            attrs={"station_id": site},
        )
        ds.to_netcdf(nc_dir / f"{site}_2010_2014.nc")

    scanned = ScannedDataset(
        name="MyStn",
        resolution="Station",
        category="Carbon",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"CH4_Flux": "Carbon/CH4_Flux/MyStn"},
    )
    existing = {
        "variables": {
            "CH4_Flux": {
                "varname": "ch4",
                "varunit": "umol m-2 s-1",
                "prefix": "AU_Tum_",
                "suffix": "_2014",
            }
        }
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    variable = catalog["MyStn"]["variables"]["CH4_Flux"]

    assert variable["varname"] == "ch4"
    assert variable["varunit"] == "umol m-2 s-1"
    assert "prefix" not in variable
    assert "suffix" not in variable


def test_rescan_preserves_station_matching_config(tmp_path: Path):
    """Manual station matching configuration is user-owned catalog metadata and
    must survive scan/register refreshes.
    """
    from openbench.data.registry.scanner import _register_to_dict

    scanned = ScannedDataset(
        name="RiverStn",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Streamflow": "Water/Streamflow/RiverStn"},
    )
    existing = {
        "station_matching": {
            "method": "direct",
            "dataset_file": "stations.nc",
            "station_id_var": "site_id",
            "lon_var": "xlon",
            "lat_var": "ylat",
            "discharge_var": "Q",
            "time_var": "time",
            "min_uparea": 10.0,
            "max_uparea": 5000.0,
        },
        "variables": {"Streamflow": {"varname": "Q", "varunit": "m3 s-1"}},
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)

    assert catalog["RiverStn"]["station_matching"] == existing["station_matching"]


def test_rescan_preserves_existing_variable_extension_fields(tmp_path: Path):
    """Variable-level user metadata such as fallbacks and station filters should
    not be dropped while refreshing scanner-derived fields.
    """
    from openbench.data.registry.scanner import _register_to_dict

    scanned = ScannedDataset(
        name="RiverGrid",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Streamflow": "Water/Streamflow/RiverGrid"},
    )
    existing_var = {
        "varname": "Q",
        "varunit": "m3 s-1",
        "prefix": "Q_",
        "suffix": "_daily",
        "fallbacks": [{"varname": "streamflow", "varunit": "m3 s-1"}],
        "fulllist": "list/subset.csv",
        "max_uparea": 5000000.0,
        "min_uparea": 1000.0,
        "prefix_fallback": ["Q_alt_"],
    }
    existing = {"variables": {"Streamflow": existing_var}}

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    variable = catalog["RiverGrid_LowRes"]["variables"]["Streamflow"]

    for key, value in existing_var.items():
        assert variable[key] == value
    assert variable["sub_dir"] == "Water/Streamflow/RiverGrid"


def test_env_var_fulllist_is_preserved_on_rescan(tmp_path: Path, monkeypatch):
    """Existing fulllist paths may themselves use OPENBENCH_REF_ROOT and should
    be expanded before deciding whether the file exists.
    """
    from openbench.data.registry.scanner import _register_to_dict

    ref_root = tmp_path / "Reference"
    list_path = ref_root / "Station" / "Water" / "Streamflow" / "RiverStn" / "list" / "subset.csv"
    list_path.parent.mkdir(parents=True)
    list_path.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nS1,2001,2002,1,2,S1.nc\n")
    monkeypatch.setenv("OPENBENCH_REF_ROOT", str(ref_root))

    scanned = ScannedDataset(
        name="RiverStn",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir="${OPENBENCH_REF_ROOT}/Station",
        variables={"Streamflow": "Water/Streamflow/RiverStn"},
    )
    existing = {
        "fulllist": "${OPENBENCH_REF_ROOT}/Station/Water/Streamflow/RiverStn/list/subset.csv",
        "root_dir": "${OPENBENCH_REF_ROOT}/Station",
        "variables": {"Streamflow": {"varname": "Q", "varunit": "m3 s-1"}},
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)

    assert catalog["RiverStn"]["fulllist"] == existing["fulllist"]


def test_rescan_preserves_existing_profile_variables_missing_from_scan(tmp_path: Path, monkeypatch):
    """Profile application must not delete existing variable mappings that
    remain declared in the profile but are not rediscovered by the walker.
    """
    import openbench.data.registry.scanner as scanner
    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(
        scanner,
        "get_reference_profile",
        lambda _name: {
            "variables": {
                "Present": {"varname": "present", "varunit": "1"},
                "LegacyOnly": {"varname": "legacy", "varunit": "kg m-2 s-1"},
            }
        },
    )

    scanned = ScannedDataset(
        name="Profiled",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Present": ""},
    )
    existing = {
        "variables": {
            "LegacyOnly": {
                "sub_dir": "Water/LegacyOnly/Profiled",
                "varname": "legacy_old",
                "varunit": "kg m-2 s-1",
                "prefix": "legacy_",
            }
        }
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)

    legacy = catalog["Profiled_LowRes"]["variables"]["LegacyOnly"]
    assert legacy["sub_dir"] == "Water/LegacyOnly/Profiled"
    assert legacy["varname"] == "legacy_old"
    assert legacy["prefix"] == "legacy_"


def test_rescan_preserves_existing_fulllist_when_file_exists(tmp_path: Path):
    """Station fulllist auto-generation must be skipped when an existing
    fulllist points to a real file. Previously it always regenerated,
    overwriting user-curated station subset CSVs.
    """
    from openbench.data.registry.scanner import _register_to_dict

    # Create a fake NC + fake user-curated fulllist
    nc_dir = tmp_path / "Carbon" / "CH4_Flux" / "FluxStn"
    nc_dir.mkdir(parents=True)
    import netCDF4

    with netCDF4.Dataset(nc_dir / "fluxstn_2010.nc", "w") as nc:
        nc.createDimension("time", 12)
        v = nc.createVariable("CH4_flux", "f4", ("time",))
        v.units = "g/m2/year"
        lat = nc.createVariable("lat", "f4", ())
        lat.units = "degrees_north"
        lon = nc.createVariable("lon", "f4", ())
        lon.units = "degrees_east"

    user_fulllist = tmp_path / "user_curated_stations.csv"
    user_fulllist.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nfluxstn,2010,2010,11.0,47.0,\n")

    scanned = ScannedDataset(
        name="FluxStn",
        resolution="Station",
        category="Carbon",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"CH4_Flux": "Carbon/CH4_Flux/FluxStn"},
    )

    catalog: dict = {}
    existing = {"fulllist": str(user_fulllist), "variables": {}}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)

    assert catalog["FluxStn"]["fulllist"] == str(user_fulllist), (
        "Existing fulllist pointing to an extant file must be preserved, "
        f"not regenerated. Got: {catalog['FluxStn']['fulllist']}"
    )
    # Verify file content unchanged (not rewritten)
    assert user_fulllist.read_text(encoding="utf-8").startswith("ID,SYEAR,EYEAR,LON,LAT,DIR\n")


def test_data_groupby_detects_monthly_files(tmp_path: Path):
    """Files like ET_2010_01.nc / ET_2010_02.nc / ... must be classified as
    Month, not Year. The previous implementation only output Year or Single.
    """
    from openbench.data.registry.scanner import ScannedDataset, _build_base_descriptor

    nc_dir = tmp_path / "var_dir"
    nc_dir.mkdir()
    for year in (2010, 2011):
        for month in (1, 2, 3):
            (nc_dir / f"ET_{year}_{month:02d}.nc").write_text("")

    scanned = ScannedDataset(
        name="MonthlyData",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "var_dir"},
    )
    desc = _build_base_descriptor(scanned, prov={})
    assert desc["data_groupby"] == "Month", f"Got: {desc['data_groupby']}"


def test_data_groupby_detects_daily_files(tmp_path: Path):
    """Files like ET_20100101.nc / ET_20100102.nc must be classified as Day."""
    from openbench.data.registry.scanner import ScannedDataset, _build_base_descriptor

    nc_dir = tmp_path / "var_dir"
    nc_dir.mkdir()
    for day in (1, 2, 3, 4, 5):
        (nc_dir / f"ET_201001{day:02d}.nc").write_text("")

    scanned = ScannedDataset(
        name="DailyData",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "var_dir"},
    )
    desc = _build_base_descriptor(scanned, prov={})
    assert desc["data_groupby"] == "Day", f"Got: {desc['data_groupby']}"


def test_tim_res_provenance_default_when_no_evidence(tmp_path: Path):
    """When neither filename nor NC content has time-resolution evidence,
    descriptor.tim_res should fall to "Month" with provenance="default",
    NOT provenance="scan" which would falsely claim directory-inferred.
    """
    from openbench.data.registry.scanner import ScannedDataset, _register_to_dict

    # NC with time variable but no time intervals to infer freq from
    nc_dir = tmp_path / "Water" / "Evapotranspiration" / "Mystery"
    nc_dir.mkdir(parents=True)
    import netCDF4

    # Single time step → can't compute _diff; tim_res from NC is None
    with netCDF4.Dataset(nc_dir / "et_unknown.nc", "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("lat", 4)
        nc.createDimension("lon", 4)
        et = nc.createVariable("ET", "f4", ("time", "lat", "lon"))
        et.units = "mm"

    scanned = ScannedDataset(
        name="Mystery",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Mystery"},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    entry = catalog["Mystery_LowRes"]

    assert entry["tim_res"] == "Month", f"Should fall back to Month default, got: {entry['tim_res']}"
    assert entry["_provenance"]["tim_res"] == "default", (
        f"Provenance must be 'default' (no evidence), not "
        f"{entry['_provenance']['tim_res']!r}. Lying about scan-confirmed "
        f"would mislead 'openbench check' output."
    )


def test_register_invalidates_registry_singleton_cache(tmp_path: Path, monkeypatch):
    """register_scanned_dataset[s_batch] must clear the registry singleton
    cache so subsequent get_registry() reads see the newly written entry.
    Long-lived processes (GUI, Jupyter) hit this without the fix.
    """
    from openbench.data.registry import manager as mgr_mod
    from openbench.data.registry.scanner import register_scanned_dataset

    # Set a sentinel cache value so we can detect when it's cleared
    mgr_mod._REGISTRY_CACHE = "SENTINEL_NOT_NONE"

    catalog_path = tmp_path / "reference_catalog.yaml"
    scanned = ScannedDataset(
        name="CacheTest",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "."},
    )
    register_scanned_dataset(scanned, catalog_path=catalog_path)

    assert mgr_mod._REGISTRY_CACHE is None, (
        "Write API must invalidate the singleton cache so get_registry() "
        "re-loads from the updated YAML; cache was not cleared."
    )


def test_year_regex_excludes_version_prefix(tmp_path: Path):
    """Filenames like 'ET_v2010_GLEAM.nc' should NOT extract 2010 as year.
    The leading 'v' marks version, not date. Previously year extraction was
    naive '\\d{4}' which polluted years with version markers.
    """
    import netCDF4

    from openbench.data.registry.scanner import _inspect_nc_file

    # Build minimal 2D NC so _inspect_nc_file has data to work with
    with netCDF4.Dataset(tmp_path / "ET_v2010_GLEAM.nc", "w") as nc:
        nc.createDimension("lat", 4)
        nc.createDimension("lon", 4)
        v = nc.createVariable("ET", "f4", ("lat", "lon"))
        v.units = "mm"

    info = _inspect_nc_file(tmp_path)
    # No real year in the filename → syear / eyear must NOT be set
    assert "syear" not in info, f"v2010 should NOT be detected as year, got syear={info.get('syear')}"
    assert "eyear" not in info
    # And prefix should be the full stem (no year split)
    assert info.get("prefix") == "ET_v2010_GLEAM"
    assert info.get("suffix") == ""


def test_year_regex_accepts_undelimited_year_after_variable_prefix(tmp_path: Path):
    """Common files like ro2001.nc / ro2002.nc must split at the year.

    The version-marker fix must not require an underscore before the year:
    processing searches prefix + year + suffix, so prefix="ro2001" would later
    look for ro20012001.nc and fail.
    """
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    for year in (2001, 2002):
        ds = xr.Dataset(
            {"ro": (["time", "lat", "lon"], np.zeros((1, 2, 2), dtype=np.float32))},
            coords={
                "time": np.array([f"{year}-01-01"], dtype="datetime64[ns]"),
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            },
        )
        ds.to_netcdf(tmp_path / f"ro{year}.nc")

    info = _inspect_nc_file(tmp_path)
    assert info["prefix"] == "ro"
    assert info.get("suffix", "") == ""
    assert info["syear"] == 2001
    assert info["eyear"] == 2002


def test_monthly_prefix_suffix_uses_full_date_token(tmp_path: Path):
    """Monthly files must not bake the first month into suffix.

    ET_2010_01.nc / ET_2010_02.nc should yield prefix="ET_", suffix="",
    so processing's prefix+year wildcard can pick up all months.
    """
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    for month in (1, 2):
        ds = xr.Dataset(
            {"ET": (["time", "lat", "lon"], np.zeros((1, 2, 2), dtype=np.float32))},
            coords={
                "time": np.array([f"2010-{month:02d}-01"], dtype="datetime64[ns]"),
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            },
        )
        ds.to_netcdf(tmp_path / f"ET_2010_{month:02d}.nc")

    info = _inspect_nc_file(tmp_path)
    assert info["prefix"] == "ET_"
    assert info.get("suffix", "") == ""
    assert info["syear"] == 2010
    assert info["eyear"] == 2010


def test_prefix_suffix_prefers_output_month_token_over_experiment_year_range(tmp_path: Path):
    """CLM-style files may contain both experiment year ranges and output months."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    for month in (1, 2):
        ds = xr.Dataset(
            {"QFLX_EVAP_TOT": (["time", "lat", "lon"], np.zeros((1, 2, 2), dtype=np.float32))},
            coords={
                "time": np.array([month - 1]),
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            },
        )
        ds["time"].attrs["units"] = "days since 1901-01-01 00:00:00"
        ds["time"].attrs["calendar"] = "noleap"
        ds.to_netcdf(tmp_path / f"IHistClm50BgcCropGSWP_CWD-1901-2014.clm2.h0.2002-{month:02d}.nc")

    info = _inspect_nc_file(tmp_path)

    assert info["prefix"] == "IHistClm50BgcCropGSWP_CWD-1901-2014.clm2.h0."
    assert info.get("suffix", "") == ""
    assert info["syear"] == 2002
    assert info["eyear"] == 2002


def test_reference_profile_matches_registry_name_and_applies_metadata(monkeypatch, tmp_path: Path):
    """Variant-specific profiles (Demo_LowRes) should apply during scan registration."""
    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "Demo_LowRes": {
                "description": "variant profile description",
                "category": "Energy",
                "tim_res": "Day",
                "data_groupby": "single",
                "variables": {
                    "Runoff": {"varname": "Q", "varunit": "m3 s-1"},
                },
            },
        },
    )
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "."},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    entry = catalog["Demo_LowRes"]

    assert entry["description"] == "variant profile description"
    assert entry["category"] == "Energy"
    assert entry["tim_res"] == "Day"
    assert entry["data_groupby"] == "single"
    assert entry["variables"]["Runoff"]["varname"] == "Q"
    assert entry["variables"]["Runoff"]["varunit"] == "m3 s-1"


def test_reference_profile_does_not_overwrite_existing_variable_edits(
    monkeypatch,
    tmp_path: Path,
):
    """Profile fields should fill missing variable metadata, not overwrite user edits."""
    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "Demo_LowRes": {
                "variables": {
                    "Runoff": {
                        "varname": "profile_Q",
                        "varunit": "profile_unit",
                        "prefix": "profile_",
                        "suffix": "_profile",
                        "max_uparea": 1000.0,
                    },
                },
            },
        },
    )
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "."},
    )
    existing = {
        "variables": {
            "Runoff": {
                "varname": "user_Q",
                "varunit": "user_unit",
                "sub_dir": "Water/UserCorrected/Demo",
                "prefix": "user_",
            }
        }
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    variable = catalog["Demo_LowRes"]["variables"]["Runoff"]

    assert variable["varname"] == "user_Q"
    assert variable["varunit"] == "user_unit"
    assert variable["sub_dir"] == "Water/UserCorrected/Demo"
    assert variable["prefix"] == "user_"
    assert variable["suffix"] == "_profile"
    assert variable["max_uparea"] == 1000.0


def test_reference_profile_fallback_preserves_existing_renamed_variable_fields(
    monkeypatch,
    tmp_path: Path,
):
    """When profile variable keys rename scanned directory keys, rescan must
    not drop existing locator fields like sub_dir/prefix.
    """
    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "Demo": {
                "variables": {
                    "Canonical_Runoff": {
                        "varname": "profile_Q",
                        "varunit": "profile_unit",
                        "suffix": "_profile",
                    },
                },
            },
        },
    )
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Raw_Runoff_Dir": "Water/Raw_Runoff_Dir/Demo"},
    )
    existing = {
        "variables": {
            "Canonical_Runoff": {
                "varname": "user_Q",
                "varunit": "user_unit",
                "sub_dir": "Water/Raw_Runoff_Dir/Demo",
                "prefix": "runoff_",
                "_nc_tim_res": "Day",
            },
        },
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    variables = catalog["Demo_LowRes"]["variables"]
    variable = variables["Canonical_Runoff"]

    assert "Raw_Runoff_Dir" not in variables
    assert variable["varname"] == "user_Q"
    assert variable["varunit"] == "user_unit"
    assert variable["sub_dir"] == "Water/Raw_Runoff_Dir/Demo"
    assert variable["prefix"] == "runoff_"
    assert variable["suffix"] == "_profile"
    assert "_nc_tim_res" not in variable


def test_register_to_dict_strips_nc_temporary_fields_from_all_variables(
    monkeypatch,
    tmp_path: Path,
):
    """NC detection scratch keys must not leak into persisted variable YAML."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})
    for sub_dir, nc_var in (("Water/VarA/Demo", "a"), ("Water/VarB/Demo", "b")):
        nc_dir = tmp_path / sub_dir
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {nc_var: (["time", "lat", "lon"], np.zeros((12, 2, 2)))},
            coords={
                "time": pd.date_range("2000-01-01", periods=12, freq="MS"),
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            },
        )
        ds[nc_var].attrs["units"] = "unit"
        ds.to_netcdf(nc_dir / f"{nc_var}_2000.nc")

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={
            "VarA": "Water/VarA/Demo",
            "VarB": "Water/VarB/Demo",
        },
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    for variable in catalog["Demo_LowRes"]["variables"].values():
        assert not any(key.startswith("_") for key in variable)


def test_register_to_dict_omits_empty_scanned_sub_dir(tmp_path: Path):
    """An empty scanned sub_dir is schema-equivalent to omission and should
    not create noisy rescan diffs.
    """
    from openbench.data.registry.scanner import _register_to_dict

    scanned = ScannedDataset(
        name="Demo",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Streamflow": ""},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    assert "sub_dir" not in catalog["Demo"]["variables"]["Streamflow"]


def test_reference_profile_fulllist_is_applied_for_station(monkeypatch, tmp_path: Path):
    """register-profile --fulllist must survive rescan into the descriptor."""
    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "DemoStn": {
                "tim_res": "Day",
                "fulllist": "list/subset.csv",
                "variables": {
                    "Runoff": {"varname": "Q", "varunit": "m3 s-1"},
                },
            },
        },
    )
    scanned = ScannedDataset(
        name="DemoStn",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Runoff": "."},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    assert catalog["DemoStn"]["fulllist"] == "list/subset.csv"


def test_reference_profile_variable_layout_fields_are_applied(monkeypatch, tmp_path: Path):
    """Profiles must be able to define layout fields, not only varname/unit."""
    from openbench.data.registry.scanner import _register_to_dict

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "ProfiledLayout": {
                "variables": {
                    "Evapotranspiration": {
                        "varname": "E",
                        "varunit": "mm day-1",
                        "sub_dir": "Composite/GLEAM_v4.2/E",
                        "prefix": "E_",
                        "suffix": "_daily",
                        "fallbacks": ["Latent_Heat"],
                    },
                },
            },
        },
    )
    scanned = ScannedDataset(
        name="ProfiledLayout",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "."},
    )

    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    variable = catalog["ProfiledLayout_LowRes"]["variables"]["Evapotranspiration"]

    assert variable["varname"] == "E"
    assert variable["varunit"] == "mm day-1"
    assert variable["sub_dir"] == "Composite/GLEAM_v4.2/E"
    assert variable["prefix"] == "E_"
    assert variable["suffix"] == "_daily"
    assert variable["fallbacks"] == ["Latent_Heat"]


def test_existing_root_relative_fulllist_is_preserved_on_rescan(tmp_path: Path):
    """Existing fulllist paths may be root_dir-relative, matching CLI docs."""
    from openbench.data.registry.scanner import _register_to_dict

    custom_list = tmp_path / "list" / "subset.csv"
    custom_list.parent.mkdir()
    custom_list.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nS1,2001,2002,1,2,S1.nc\n")

    scanned = ScannedDataset(
        name="DemoStn",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Runoff": "."},
    )
    existing = {
        "fulllist": "list/subset.csv",
        "root_dir": str(tmp_path),
        "variables": {"Runoff": {"varname": "Q", "varunit": "m3 s-1"}},
    }

    catalog: dict = {}
    _register_to_dict(scanned, catalog, existing_descriptor=existing)

    assert catalog["DemoStn"]["fulllist"] == "list/subset.csv"


def test_native_nc_spacing_wins_over_resolution_bucket(tmp_path: Path):
    """LowRes/MidRes registry variants keep native NC spacing in grid_res."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Runoff" / "NativeFine"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {"ro": (["time", "lat", "lon"], np.zeros((2, 3, 3)))},
        coords={
            "time": np.array(["2001-01-01", "2001-02-01"], dtype="datetime64[D]"),
            "lat": [0.0, 0.25, 0.5],
            "lon": [100.0, 100.25, 100.5],
        },
    )
    ds.to_netcdf(nc_dir / "NativeFine_2001.nc")

    scanned = ScannedDataset(
        name="NativeFine",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Water/Runoff/NativeFine"},
    )
    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    assert catalog["NativeFine_LowRes"]["grid_res"] == 0.25
    assert catalog["NativeFine_LowRes"]["_provenance"]["grid_res"] == "nc"


def test_custom_filter_station_does_not_persist_generated_registry_fulllist(tmp_path: Path, monkeypatch):
    """Datasets with custom station filters should not get package-local
    generated fulllist paths persisted into the catalog.
    """
    import openbench.data.registry.scanner as scanner
    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "dataset"
    nc_dir.mkdir()
    placeholder = nc_dir / "placeholder.nc"
    placeholder.write_text("not inspected in this test")

    def fake_glob_nc(path):
        return [placeholder] if Path(path) == nc_dir else []

    def fake_generate_station_list(_nc_dir, output_csv):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nx,2000,2001,0,0,\n")

    monkeypatch.setattr(scanner, "_glob_nc", fake_glob_nc)
    monkeypatch.setattr(scanner, "generate_station_list", fake_generate_station_list)

    scanned = ScannedDataset(
        name="CH4_FluxnetANN",
        resolution="Station",
        category="Carbon",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Methane": "dataset"},
    )
    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    assert "fulllist" not in catalog["CH4_FluxnetANN"]


def test_generate_station_list_handles_scalar_lat_lon(tmp_path: Path):
    """Single-station NC files commonly store lat/lon as scalar variables."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import generate_station_list

    ds = xr.Dataset(
        {
            "Q": (["time"], np.array([1.0, 2.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": np.array(["2001-01-01", "2002-01-01"], dtype="datetime64[ns]")},
    )
    ds.to_netcdf(tmp_path / "S1.nc")

    out = generate_station_list(tmp_path, tmp_path / "stations.csv")
    rows = pd.read_csv(out)

    assert rows.loc[0, "ID"] == "S1"
    assert rows.loc[0, "LAT"] == 2.0
    assert rows.loc[0, "LON"] == 1.0


def test_generate_station_list_preserves_existing_csv_on_write_failure(tmp_path: Path, monkeypatch):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import generate_station_list

    ds = xr.Dataset(
        {
            "Q": (["time"], np.array([1.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": np.array(["2001-01-01"], dtype="datetime64[ns]")},
    )
    ds.to_netcdf(tmp_path / "S1.nc")
    output = tmp_path / "stations.csv"
    original = "ID,SYEAR,EYEAR,LON,LAT,DIR\nold,1999,1999,0,0,old.nc\n"
    output.write_text(original, encoding="utf-8")

    def fail_to_csv(self, path, *args, **kwargs):
        Path(path).write_text("partial", encoding="utf-8")
        raise OSError("simulated csv failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="simulated csv failure"):
        generate_station_list(tmp_path, output)

    assert output.read_text(encoding="utf-8") == original


def test_catalog_write_lock_fails_closed_when_lock_file_cannot_be_created(tmp_path: Path, monkeypatch):
    from openbench.data.registry import scanner

    original_touch = Path.touch

    def fail_lock_touch(self, *args, **kwargs):
        if str(self).endswith(".lock"):
            raise OSError("lock unavailable")
        return original_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_lock_touch)

    with pytest.raises(RuntimeError, match="failed to acquire catalog lock"):
        with scanner._catalog_write_lock(tmp_path / "catalog.yaml"):
            raise AssertionError("lock body should not run")


def test_parse_single_station_file_decodes_char_array_station_id(tmp_path: Path):
    """Classic NetCDF S1 char arrays should decode to a station ID."""
    import netCDF4
    import numpy as np

    from openbench.data.registry.scanner import _parse_single_station_file

    nc_file = tmp_path / "fallback_2015_2019.nc"
    with netCDF4.Dataset(nc_file, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("nchar", 6)
        station = nc.createVariable("station_id", "S1", ("nchar",))
        station[:] = np.array(list("US_HRV"), dtype="S1")
        lat = nc.createVariable("lat", "f4")
        lon = nc.createVariable("lon", "f4")
        lat.assignValue(42.0)
        lon.assignValue(-72.0)

    row = _parse_single_station_file(nc_file)

    assert row is not None
    assert row[0] == "US_HRV"


def test_parse_single_station_file_decodes_scalar_string_station_id(tmp_path: Path):
    """Scalar vlen string station IDs should not fall back to the filename."""
    import netCDF4

    from openbench.data.registry.scanner import _parse_single_station_file

    nc_file = tmp_path / "filename_fallback_2015_2019.nc"
    with netCDF4.Dataset(nc_file, "w") as nc:
        station = nc.createVariable("station_id", str)
        station[0] = "US_HRV"
        lat = nc.createVariable("lat", "f4")
        lon = nc.createVariable("lon", "f4")
        lat.assignValue(42.0)
        lon.assignValue(-72.0)

    row = _parse_single_station_file(nc_file)

    assert row is not None
    assert row[0] == "US_HRV"


def test_parse_single_station_file_rejects_vector_lat_lon(tmp_path: Path):
    """Merged station files may use non-standard station dimension names."""
    import netCDF4

    from openbench.data.registry.scanner import _parse_single_station_file

    nc_file = tmp_path / "merged_locations.nc"
    with netCDF4.Dataset(nc_file, "w") as nc:
        nc.createDimension("location", 2)
        nc.createDimension("time", 1)
        lat = nc.createVariable("lat", "f4", ("location",))
        lon = nc.createVariable("lon", "f4", ("location",))
        lat[:] = [10.0, 20.0]
        lon[:] = [30.0, 40.0]
        q = nc.createVariable("q", "f4", ("time", "location"))
        q[:] = [[1.0, 2.0]]

    assert _parse_single_station_file(nc_file) is None


def test_parse_single_station_file_ignores_version_year_tokens(tmp_path: Path):
    """Station list year parsing should match scanner date-token rules."""
    import netCDF4

    from openbench.data.registry.scanner import _parse_single_station_file

    nc_file = tmp_path / "site_v2010_data_2015_2019.nc"
    with netCDF4.Dataset(nc_file, "w") as nc:
        nc.station_id = "SITE"
        lat = nc.createVariable("lat", "f4")
        lon = nc.createVariable("lon", "f4")
        lat.assignValue(42.0)
        lon.assignValue(-72.0)

    row = _parse_single_station_file(nc_file)

    assert row is not None
    assert row[1] == 2015
    assert row[2] == 2019


def test_station_list_generation_failure_logs_warning(tmp_path: Path, caplog):
    """Registration should surface station fulllist generation failures."""
    import logging

    import netCDF4

    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "dataset"
    nc_dir.mkdir()
    with netCDF4.Dataset(nc_dir / "bad_station_2010.nc", "w") as nc:
        nc.createDimension("time", 1)
        q = nc.createVariable("Q", "f4", ("time",))
        q[:] = [1.0]

    scanned = ScannedDataset(
        name="BadStation",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Streamflow": "dataset"},
    )
    catalog: dict = {}

    with caplog.at_level(logging.WARNING):
        _register_to_dict(scanned, catalog)

    assert "Failed to generate station list for BadStation" in caplog.text
    assert "Could not extract station info" in caplog.text
    assert "fulllist" not in catalog["BadStation"]


# ---------------------------------------------------------------------------
# Regression tests for batch 3 (concurrency lock, edge-case detections,
# dry-run preview).
# ---------------------------------------------------------------------------


def test_concurrent_register_does_not_lose_writes_under_lock(tmp_path: Path):
    """Two concurrent register calls must both land in the catalog.

    Without the flock, the second writer reads pre-A state, computes its
    write, then overwrites the file losing A's contribution. With flock,
    B blocks until A's read-modify-write completes, then B reads A's
    state and appends.
    """
    import threading

    catalog_path = tmp_path / "reference_catalog.yaml"
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(name: str):
        try:
            scanned = ScannedDataset(
                name=name,
                resolution="LowRes",
                category="Water",
                data_type="grid",
                root_dir=str(tmp_path),
                variables={"Evapotranspiration": "."},
            )
            barrier.wait(timeout=5)  # both threads start at the same moment
            register_scanned_dataset(scanned, catalog_path=catalog_path)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("ConcurrentA",))
    t2 = threading.Thread(target=worker, args=("ConcurrentB",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Worker errors: {errors}"

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    # BOTH datasets must be present — no silent overwrite
    assert "ConcurrentA_LowRes" in catalog, f"ConcurrentA missing from catalog (race condition): {list(catalog.keys())}"
    assert "ConcurrentB_LowRes" in catalog, f"ConcurrentB missing from catalog (race condition): {list(catalog.keys())}"


def test_data_type_detection_classifies_profile_data_as_grid(tmp_path: Path):
    """1D profile-style data (lat>1, lon=1) should detect as grid, not None.

    Previously returned None and caller defaulted grid. Making it explicit
    avoids ambiguity in callers that branch on the detected value.
    """
    import netCDF4

    from openbench.data.registry.scanner import _detect_data_type_from_nc

    nc_file = tmp_path / "profile.nc"
    with netCDF4.Dataset(nc_file, "w") as nc:
        nc.createDimension("time", 12)
        nc.createDimension("lat", 90)  # multi-element zonal axis
        nc.createDimension("lon", 1)  # single longitude
        v = nc.createVariable("temp", "f4", ("time", "lat", "lon"))
        v.units = "K"

    assert _detect_data_type_from_nc(nc_file) == "grid"


def test_tim_res_detection_rejects_half_hourly_interval(tmp_path: Path):
    """30-min interval (1800s) doesn't match any valid tim_res value
    (Hour/3Hour/6Hour/Day/...). Must return None instead of bucketing
    into Hour, which would make downstream evaluation aggregate wrong.
    """
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import _inspect_nc_file

    # 48 timesteps × 30min = 1 day of half-hourly data
    times = np.arange(0, 86400, 1800, dtype="float64")  # 48 values, 1800s apart
    ds = xr.Dataset(
        {"flux": (["time", "lat", "lon"], np.zeros((48, 4, 4), dtype=np.float32))},
        coords={"time": times, "lat": np.arange(4.0), "lon": np.arange(4.0)},
    )
    ds["time"].attrs["units"] = "seconds since 2010-01-01"
    ds.to_netcdf(tmp_path / "halfhourly_2010.nc")

    info = _inspect_nc_file(tmp_path)
    # 1800s doesn't match any valid bucket — must NOT silently become "Hour"
    assert info.get("detected_tim_res") != "Hour", (
        f"Half-hourly (1800s) must not bucket as Hour, got: {info.get('detected_tim_res')!r}"
    )


def test_cli_register_creates_backup_before_overwriting_catalog(tmp_path: Path, monkeypatch):
    """openbench ref register must back up the previous catalog state, like
    scan does. Previously used bare _atomic_yaml_write which left no recovery
    path if a register call overwrote a hand-edited entry by mistake.
    """
    import openbench.cli.data as cli_data
    import openbench.data.registry.manager as mgr_mod

    catalog_path = tmp_path / "reference_catalog.yaml"
    catalog_path.write_text(
        "OldDataset:\n  name: OldDataset\n  category: Water\n  description: Hand-edited description\n"
    )

    # Force the writable catalog path to our tmp location
    monkeypatch.setattr(
        mgr_mod,
        "get_writable_reference_catalog_path",
        lambda: catalog_path,
    )
    # Silence click output
    monkeypatch.setattr(cli_data.click, "echo", lambda *a, **k: None)
    monkeypatch.setattr(cli_data.click, "secho", lambda *a, **k: None)

    cli_data.register.callback(
        name="NewDataset",
        root_dir=str(tmp_path),
        data_type="grid",
        tim_res="Month",
        grid_res=0.5,
        category="Water",
        years=(2010, 2020),
        fulllist=None,
        variable=("Evapotranspiration:ET:mm/day",),
        fallback=(),
    )

    backup_path = Path(str(catalog_path) + ".bak")
    assert backup_path.exists(), f"Expected .bak at {backup_path}"
    bak_data = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    assert "OldDataset" in bak_data, "Backup must contain pre-register state"
    # New catalog has both entries (OldDataset preserved)
    new_data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert "OldDataset" in new_data
    assert "NewDataset" in new_data


def test_cli_register_refuses_overwrite_when_catalog_unparseable(tmp_path: Path, monkeypatch):
    """openbench ref register on a corrupted catalog must raise rather than
    silently overwrite. Same hardening as scan path.
    """
    import pytest

    import openbench.cli.data as cli_data
    import openbench.data.registry.manager as mgr_mod

    catalog_path = tmp_path / "reference_catalog.yaml"
    catalog_path.write_text("not: valid: ::: garbage\n[broken")
    original = catalog_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        mgr_mod,
        "get_writable_reference_catalog_path",
        lambda: catalog_path,
    )
    monkeypatch.setattr(cli_data.click, "echo", lambda *a, **k: None)
    monkeypatch.setattr(cli_data.click, "secho", lambda *a, **k: None)

    with pytest.raises(cli_data.click.ClickException, match="Failed to load existing catalog"):
        cli_data.register.callback(
            name="NewDataset",
            root_dir=str(tmp_path),
            data_type="grid",
            tim_res="Month",
            grid_res=0.5,
            category="Water",
            years=(2010, 2020),
            fulllist=None,
            variable=("Evapotranspiration:ET:mm/day",),
            fallback=(),
        )

    # Original (corrupted) file untouched
    assert catalog_path.read_text(encoding="utf-8") == original


def test_cli_model_register_creates_backup_before_overwriting(tmp_path: Path, monkeypatch):
    """openbench model register must back up the previous catalog state, like
    openbench ref register does. Previously used bare _atomic_yaml_write
    which left no recovery path.
    """
    import openbench.cli.model as cli_model
    import openbench.data.registry.manager as mgr_mod

    catalog_path = tmp_path / "model_catalog.yaml"
    catalog_path.write_text(
        "OldModel:\n"
        "  name: OldModel\n"
        "  description: Hand-edited\n"
        "  variables:\n"
        "    Evapotranspiration:\n"
        "      varname: ET\n"
    )

    monkeypatch.setattr(
        mgr_mod,
        "get_writable_model_catalog_path",
        lambda: catalog_path,
    )
    monkeypatch.setattr(cli_model.click, "echo", lambda *a, **k: None)
    monkeypatch.setattr(cli_model.click, "secho", lambda *a, **k: None)

    cli_model.register.callback(
        name="NewModel",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        description=None,
        variable=("Latent_Heat:LE:W m-2",),
        fallback=(),
        append_only=False,
    )

    backup_path = Path(str(catalog_path) + ".bak")
    assert backup_path.exists(), f"Expected .bak at {backup_path}"
    bak_data = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    assert "OldModel" in bak_data
    new_data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert "OldModel" in new_data
    assert "NewModel" in new_data


def test_cli_model_register_refuses_overwrite_when_catalog_unparseable(tmp_path: Path, monkeypatch):
    """openbench model register on a corrupted model_catalog.yaml must raise
    rather than silently overwrite. Same hardening as the ref register path.
    """
    import pytest

    import openbench.cli.model as cli_model
    import openbench.data.registry.manager as mgr_mod

    catalog_path = tmp_path / "model_catalog.yaml"
    catalog_path.write_text("not: valid: ::: garbage\n[broken")
    original = catalog_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        mgr_mod,
        "get_writable_model_catalog_path",
        lambda: catalog_path,
    )
    monkeypatch.setattr(cli_model.click, "echo", lambda *a, **k: None)
    monkeypatch.setattr(cli_model.click, "secho", lambda *a, **k: None)

    with pytest.raises(cli_model.click.ClickException, match="Failed to load existing catalog"):
        cli_model.register.callback(
            name="NewModel",
            data_type="grid",
            grid_res=0.5,
            tim_res="Month",
            description=None,
            variable=("Latent_Heat:LE:W m-2",),
            fallback=(),
            append_only=False,
        )

    assert catalog_path.read_text(encoding="utf-8") == original


def test_cli_model_remove_var_creates_backup(tmp_path: Path, monkeypatch):
    """openbench model remove-var must back up before deleting a variable."""
    import openbench.cli.model as cli_model
    import openbench.data.registry.manager as mgr_mod

    catalog_path = tmp_path / "model_catalog.yaml"
    catalog_path.write_text(
        "DemoModel:\n"
        "  name: DemoModel\n"
        "  variables:\n"
        "    Evapotranspiration:\n"
        "      varname: ET\n"
        "    Snow_Depth:\n"
        "      varname: SD\n"
    )

    monkeypatch.setattr(
        mgr_mod,
        "get_writable_model_catalog_path",
        lambda: catalog_path,
    )
    monkeypatch.setattr(cli_model.click, "echo", lambda *a, **k: None)
    monkeypatch.setattr(cli_model.click, "secho", lambda *a, **k: None)

    cli_model.remove_var.callback(name="DemoModel", variable_name="Snow_Depth")

    backup_path = Path(str(catalog_path) + ".bak")
    assert backup_path.exists()
    bak_data = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    # Backup retains BOTH variables (pre-remove state)
    assert "Snow_Depth" in bak_data["DemoModel"]["variables"]
    # Current file has only ET
    new_data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert "Snow_Depth" not in new_data["DemoModel"]["variables"]
    assert "Evapotranspiration" in new_data["DemoModel"]["variables"]


def test_rescan_preserves_user_edited_timezone(tmp_path: Path):
    """timezone must be preserved across rescans. Stage 1 always writes 0
    (UTC default); without preserve, rescan resets a user's hand-edited
    non-zero offset (e.g., -8 for Pacific local-time station data).
    """
    from openbench.data.registry.scanner import _register_to_dict

    nc_dir = tmp_path / "Water" / "Evapotranspiration" / "Demo"
    nc_dir.mkdir(parents=True)
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {"ET": (["time", "lat", "lon"], np.zeros((12, 4, 4), dtype=np.float32))},
        coords={"time": np.arange(12), "lat": np.arange(4.0), "lon": np.arange(4.0)},
    )
    ds.to_netcdf(nc_dir / "ET_2004_2005.nc")

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )

    # First registration → catalog has timezone=0
    catalog: dict = {}
    _register_to_dict(scanned, catalog)
    assert catalog["Demo_LowRes"]["timezone"] == 0  # default

    # User hand-edits to -8 (Pacific Standard Time)
    catalog["Demo_LowRes"]["timezone"] = -8

    # Re-register: user's -8 must survive
    existing = catalog["Demo_LowRes"]
    _register_to_dict(scanned, catalog, existing_descriptor=existing)

    assert catalog["Demo_LowRes"]["timezone"] == -8, (
        "User-edited timezone lost on rescan. Stage 6 must preserve "
        "timezone explicitly (key-presence check, not truthy check, since "
        "timezone=0 is a legitimate value)."
    )


def test_stn_scan_skips_multi_year_subdirs_as_ambiguous(tmp_path: Path, caplog):
    """Station dataset organized as MyStn/2010/, MyStn/2011/, MyStn/2012/
    used to silently accumulate NC counts but record only the last child as
    nc_dir. Now correctly detected as ambiguous and skipped with a warning.
    """
    import logging

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    base = ref_root / "Station" / "Carbon" / "CH4_Flux" / "MyStn"
    for year in (2010, 2011, 2012):
        ydir = base / str(year)
        ydir.mkdir(parents=True)
        for stn in (1, 2):
            (ydir / f"stn_{stn:03d}.nc").write_text("")

    skipped = []
    with caplog.at_level(logging.WARNING, logger="openbench.data.registry.scanner"):
        groups = scan_reference_directory(ref_root, on_skip=skipped.append)

    assert groups == [], f"Multi-year stn dataset must be skipped (ambiguous), got: {[g.base_name for g in groups]}"
    assert skipped
    assert skipped[0].path == "Station/Carbon/CH4_Flux/MyStn"
    assert skipped[0].reason == "ambiguous_nc_subdirectories"
    assert any("multiple NC-bearing subdirectories" in rec.message for rec in caplog.records), (
        f"Expected ambiguity warning, got: {[r.message for r in caplog.records]}"
    )


def test_stn_scan_records_nc_path_when_data_in_subdir(tmp_path: Path):
    """Station dataset MyStn/data/*.nc (data is a subdir, not the dataset
    name) should record variables[var_name] pointing to MyStn/data so
    finalize can find the NCs. Previously recorded only MyStn/ and the
    +/dataset heuristic happened to match here, but if the subdir was
    something else (e.g., 'subset/'), fulllist would be silently skipped.
    """
    import netCDF4

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Carbon" / "CH4_Flux" / "MyStn" / "subset"
    nc_dir.mkdir(parents=True)
    with netCDF4.Dataset(nc_dir / "stn001.nc", "w") as nc:
        nc.createDimension("time", 12)
        v = nc.createVariable("CH4_flux", "f4", ("time",))
        v.units = "g/m2/year"
        lat = nc.createVariable("lat", "f4", ())
        lat.units = "degrees_north"
        lon = nc.createVariable("lon", "f4", ())
        lon.units = "degrees_east"

    groups = scan_reference_directory(ref_root)
    assert len(groups) == 1, f"Expected one group, got: {[g.base_name for g in groups]}"
    variant = groups[0].variants["Station"]
    sub_dir = variant.variables["CH4_Flux"]
    assert sub_dir.endswith("MyStn/subset"), f"Expected sub_dir to point to actual NC location, got: {sub_dir!r}"


def test_stn_composite_layout_still_recognized(tmp_path: Path):
    """Backward compat: Station/Composite/FLUXNET_PLUMBER2/dataset/*.nc
    should still be treated as composite — dataset_name = FLUXNET_PLUMBER2
    (one level up), variables key = dataset_name with empty value (relies
    on reference_profiles.yaml to fill variable mappings).
    """
    import netCDF4

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Composite" / "FLUXNET_PLUMBER2" / "dataset"
    nc_dir.mkdir(parents=True)
    with netCDF4.Dataset(nc_dir / "stn001.nc", "w") as nc:
        nc.createDimension("time", 12)
        v = nc.createVariable("Qle", "f4", ("time",))
        v.units = "W/m2"
        lat = nc.createVariable("lat", "f4", ())
        lat.units = "degrees_north"
        lon = nc.createVariable("lon", "f4", ())
        lon.units = "degrees_east"

    groups = scan_reference_directory(ref_root)
    groups_by_name = {g.base_name: g for g in groups}
    assert "FLUXNET_PLUMBER2" in groups_by_name

    variant = groups_by_name["FLUXNET_PLUMBER2"].variants["Station"]
    # Composite: variables key is dataset_name (not var_name "FLUXNET_PLUMBER2")
    # — wait, var_name = FLUXNET_PLUMBER2 because Composite/FLUXNET_PLUMBER2/
    # in this layout iter has var_dir = FLUXNET_PLUMBER2.
    assert "FLUXNET_PLUMBER2" in variant.variables
    assert variant.variables["FLUXNET_PLUMBER2"] == ""
    # ds_root points to the literal "dataset" subdir directly (NCs there)
    assert variant.root_dir.endswith("FLUXNET_PLUMBER2/dataset")


def test_station_direct_nc_layout_uses_reference_profile(monkeypatch, tmp_path: Path):
    """Station/<category>/<dataset>/*.nc is a valid flat station layout."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "FlatStation": {
                "description": "flat station dataset",
                "tim_res": "Month",
                "data_groupby": "single",
                "variables": {
                    "Surface_Downward_SW_Radiation": {
                        "varname": "rsds",
                        "varunit": "W m-2",
                    },
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Heat" / "FlatStation"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {
            "rsds": (["time"], np.array([1.0, 2.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": np.array(["2010-01-01", "2010-02-01"], dtype="datetime64[ns]")},
    )
    ds["rsds"].attrs["units"] = "W m-2"
    ds.to_netcdf(nc_dir / "FlatStation_monthly.nc")

    groups = scan_reference_directory(ref_root)
    assert [g.base_name for g in groups] == ["FlatStation"]
    variant = groups[0].variants["Station"]
    assert variant.root_dir.endswith("Station/Heat/FlatStation")
    assert variant.variables == {"FlatStation": ""}

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["FlatStation"]

    assert entry["description"] == "flat station dataset"
    assert entry["variables"] == {
        "Surface_Downward_SW_Radiation": {
            "varname": "rsds",
            "varunit": "W m-2",
        }
    }


def test_profile_scan_station_direct_can_read_nested_nc_dir(monkeypatch, tmp_path: Path):
    """Named station profiles may keep root_dir at dataset root but inspect dataset/*.nc."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "NamedStation": {
                "scan": {
                    "layout": "station_direct",
                    "root_sub_dir": "Station/Water/Evapotranspiration/RawName",
                    "nc_sub_dir": "dataset",
                },
                "fulllist": "list/stations.csv",
                "tim_res": "Day",
                "data_groupby": "single",
                "variables": {
                    "Evapotranspiration": {
                        "sub_dir": "",
                        "varname": "et",
                        "varunit": "mm day-1",
                    },
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Water" / "Evapotranspiration" / "RawName" / "dataset"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {
            "et": (["time"], np.array([1.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": np.array(["2010-01-01"], dtype="datetime64[ns]")},
    )
    ds.to_netcdf(nc_dir / "station.nc")

    groups = scan_reference_directory(ref_root)
    assert [g.base_name for g in groups] == ["NamedStation"]
    variant = groups[0].variants["Station"]
    assert variant.root_dir.endswith("Station/Water/Evapotranspiration/RawName")
    assert variant.variables == {"Evapotranspiration": "dataset"}

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["NamedStation"]
    assert entry["fulllist"] == "list/stations.csv"
    assert entry["variables"]["Evapotranspiration"]["sub_dir"] == ""


def test_profile_scan_station_shared_files_creates_named_dataset(monkeypatch, tmp_path: Path):
    """Profiles can split shared StreamFlow folders into named datasets."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "GRDC_Daily": {
                "scan": {
                    "layout": "station_shared_files",
                    "root_sub_dir": "Station/Water/StreamFlow/Daily",
                    "file_glob": "GRDC_*.nc",
                },
                "description": "GRDC daily streamflow",
                "tim_res": "Day",
                "data_groupby": "single",
                "station_matching": {
                    "method": "cama_allocation",
                    "dataset_file": "GRDC_station.nc",
                    "station_id_var": "station",
                },
                "variables": {
                    "Streamflow": {"varname": "discharge", "varunit": "m3 s-1"},
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Water" / "StreamFlow" / "Daily"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {
            "discharge": (["time"], np.array([1.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": np.array(["2010-01-01"], dtype="datetime64[ns]")},
    )
    ds.to_netcdf(nc_dir / "GRDC_station.nc")
    ds.to_netcdf(nc_dir / "Other_station.nc")

    groups = scan_reference_directory(ref_root)
    groups_by_name = {g.base_name: g for g in groups}
    assert set(groups_by_name) == {"GRDC_Daily"}
    variant = groups_by_name["GRDC_Daily"].variants["Station"]
    assert variant.root_dir.endswith("Station/Water/StreamFlow/Daily")
    assert variant.variables == {"Streamflow": ""}

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["GRDC_Daily"]

    assert entry["description"] == "GRDC daily streamflow"
    assert entry["tim_res"] == "Day"
    assert entry["station_matching"]["dataset_file"] == "GRDC_station.nc"
    assert "fulllist" not in entry
    assert entry["variables"]["Streamflow"]["varname"] == "discharge"


def test_grid_composite_dataset_layout_uses_reference_profile(monkeypatch, tmp_path: Path):
    """Grid/Composite/<dataset>/dataset/*.nc should be scanned as the dataset
    name and then receive profile variable mappings during registration.
    """
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "GridComposite": {
                "description": "profiled grid composite",
                "tim_res": "Day",
                "data_groupby": "single",
                "variables": {
                    "Net_Radiation": {"varname": "Rn", "varunit": "W m-2"},
                    "Latent_Heat": {"varname": "LE", "varunit": "W m-2"},
                },
            }
        },
    )

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "GridComposite" / "dataset"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {
            "Rn": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32)),
            "LE": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32)),
        },
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "GridComposite_2010.nc")

    groups = scan_reference_directory(ref_root)
    assert [g.base_name for g in groups] == ["GridComposite"]
    variant = groups[0].variants["LowRes"]
    assert variant.root_dir.endswith("GridComposite/dataset")
    assert variant.variables == {"GridComposite": ""}

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["GridComposite_LowRes"]

    assert entry["description"] == "profiled grid composite"
    assert entry["tim_res"] == "Day"
    assert entry["data_groupby"] == "single"
    assert entry["variables"] == {
        "Net_Radiation": {"varname": "Rn", "varunit": "W m-2"},
        "Latent_Heat": {"varname": "LE", "varunit": "W m-2"},
    }


def test_grid_composite_multi_child_root_reports_profile_skip(monkeypatch, tmp_path: Path):
    """Composite/<dataset>/<child> layouts should require a profile."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    monkeypatch.setattr(scanner_module, "_REFERENCE_PROFILES", {})

    ref_root = tmp_path / "Reference"
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

    skipped = []
    groups = scan_reference_directory(ref_root, on_skip=skipped.append)

    assert groups == []
    assert len(skipped) == 1
    assert skipped[0].path == "Grid/LowRes/Composite/BadComposite"
    assert skipped[0].reason == "ambiguous_nc_subdirectories"


def test_profile_scan_grid_composite_children_aggregates_child_dirs(monkeypatch, tmp_path: Path):
    """Profiles can aggregate Composite/<source>/<child> dirs into one dataset."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "GLEAM_Profiled": {
                "scan": {
                    "layout": "grid_composite_children",
                    "root_sub_dir": "Grid/LowRes/Composite/GLEAM_v4.2",
                },
                "description": "profiled GLEAM composite",
                "tim_res": "Day",
                "data_groupby": "single",
                "variables": {
                    "Evapotranspiration": {
                        "child": "E",
                        "varname": "E",
                        "varunit": "mm day-1",
                    },
                    "Bare_Soil_Evaporation": {
                        "child": "Eb",
                        "varname": "Eb",
                        "varunit": "mm day-1",
                    },
                    "Open_Water_Evaporation": {
                        "child": "Ew",
                        "sub_dir": "Water/Open_Water_Evaporation/GLEAM4.2a",
                        "varname": "Ew",
                        "varunit": "mm day-1",
                    },
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    for child in ("E", "Eb"):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "GLEAM_v4.2" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {child: (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{child}_2010.nc")
    alias_dir = ref_root / "Grid" / "LowRes" / "Water" / "Open_Water_Evaporation" / "GLEAM4.2a"
    alias_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {"Ew": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(alias_dir / "Ew_2010.nc")

    groups = scan_reference_directory(ref_root)
    assert [g.base_name for g in groups] == ["GLEAM_Profiled"]
    variant = groups[0].variants["LowRes"]
    assert variant.root_dir.endswith("Grid/LowRes")
    assert variant.variables == {
        "Evapotranspiration": "Composite/GLEAM_v4.2/E",
        "Bare_Soil_Evaporation": "Composite/GLEAM_v4.2/Eb",
        "Open_Water_Evaporation": "Water/Open_Water_Evaporation/GLEAM4.2a",
    }

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["GLEAM_Profiled_LowRes"]

    assert entry["description"] == "profiled GLEAM composite"
    assert entry["variables"]["Evapotranspiration"]["sub_dir"] == "Composite/GLEAM_v4.2/E"
    assert entry["variables"]["Bare_Soil_Evaporation"]["sub_dir"] == "Composite/GLEAM_v4.2/Eb"
    assert entry["variables"]["Open_Water_Evaporation"]["sub_dir"] == ("Water/Open_Water_Evaporation/GLEAM4.2a")


def test_profile_scan_grid_composite_files_reports_mixed_resolution_skip(monkeypatch, tmp_path: Path):
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "MixedComposite": {
                "scan": {
                    "layout": "grid_composite_files",
                    "root_sub_dir": "Grid/LowRes/Composite/MixedComposite",
                },
                "variables": {
                    "Latent_Heat": {
                        "root_sub_dir": "Grid/LowRes/Composite/MixedComposite/land",
                        "file_glob": "*land*.nc",
                        "varname": "LE",
                        "varunit": "W m-2",
                    },
                    "Streamflow": {
                        "root_sub_dir": "Grid/MidRes/Composite/MixedComposite/cama",
                        "file_glob": "*cama*.nc",
                        "varname": "discharge",
                        "varunit": "m3 s-1",
                    },
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    for res, child, glob_name, step in (
        ("LowRes", "land", "land", 0.5),
        ("MidRes", "cama", "cama", 0.25),
    ):
        nc_dir = ref_root / "Grid" / res / "Composite" / "MixedComposite" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {glob_name: (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, step], "lon": [0.0, step]},
        )
        ds.to_netcdf(nc_dir / f"{glob_name}_2010.nc")

    skipped = []
    groups = scan_reference_directory(ref_root, on_skip=skipped.append)

    assert skipped
    assert skipped[0].path == "Grid/MidRes/Composite/MixedComposite/cama"
    assert skipped[0].reason == "mixed_grid_resolutions_in_profile"
    assert groups == []


def test_profile_scan_grid_nested_root_registers_ambiguous_children(monkeypatch, tmp_path: Path):
    """Nested-root profiles must name a concrete NC-bearing sub_dir."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "GDHY2019ver": {
                "scan": {
                    "layout": "grid_nested_root",
                    "root_sub_dir": "Grid/LowRes/Anth/Crop/GDHY2019ver",
                },
                "category": "Urban",
                "variables": {
                    "Crop": {"varname": "Crop", "varunit": ""},
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    for crop in ("maize", "rice"):
        nc_dir = ref_root / "Grid" / "LowRes" / "Anth" / "Crop" / "GDHY2019ver" / crop
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {"Crop": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{crop}_2010.nc")

    skipped = []
    groups = scan_reference_directory(ref_root, on_skip=skipped.append)
    assert groups == []
    assert skipped
    assert skipped[0].path == "Grid/LowRes/Anth/Crop/GDHY2019ver"
    assert skipped[0].reason == "grid_nested_root_requires_concrete_subdir"


def test_profile_scan_grid_nested_root_uses_concrete_subdir(monkeypatch, tmp_path: Path):
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "DeepRunnable": {
                "scan": {
                    "layout": "grid_nested_root",
                    "root_sub_dir": "Grid/LowRes/Water/Runoff/DeepRunnable",
                },
                "variables": {
                    "Runoff": {
                        "sub_dir": "Water/Runoff/DeepRunnable/a/b/c",
                        "varname": "ro",
                        "varunit": "mm day-1",
                    },
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Grid" / "LowRes" / "Water" / "Runoff" / "DeepRunnable" / "a" / "b" / "c"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {"ro": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "ro_2010.nc")

    groups = scan_reference_directory(ref_root)
    variant = groups[0].variants["LowRes"]
    assert variant.variables == {"Runoff": "Water/Runoff/DeepRunnable/a/b/c"}

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["DeepRunnable_LowRes"]
    assert entry["variables"]["Runoff"]["sub_dir"] == "Water/Runoff/DeepRunnable/a/b/c"


def test_profile_scan_ignore_consumes_raw_composite_tree(monkeypatch, tmp_path: Path):
    """Profiles can mark raw composite sources as scanner-only ignored roots."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "_IgnoreFluxcomRaw": {
                "scan": {
                    "layout": "ignore",
                    "root_sub_dir": "Grid/LowRes/Composite/FLUXCOM",
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    for child in ("Carbon", "Energy"):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "FLUXCOM" / child
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {"x": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{child}_2010.nc")

    groups = scan_reference_directory(ref_root)
    assert groups == []


def test_scan_uses_grandchild_dir_when_nc_files_two_levels_deep(tmp_path: Path):
    """Verify end-to-end that 3-level NC discovery wires through to a
    correct registration: sub_dir AND tim_res come from the level-2 dir,
    so _inspect_nc_file actually finds the NC file (varname/years populated).
    """
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    ref_root = tmp_path / "Reference"
    grandchild = ref_root / "Grid" / "MidRes" / "Water" / "Evapotranspiration" / "DeepData" / "0p25deg" / "daily"
    grandchild.mkdir(parents=True)

    times = xr.date_range("2010-01-01", periods=365, freq="D", use_cftime=True)
    ds = xr.Dataset(
        {"ET_actual": (["time", "lat", "lon"], np.zeros((365, 4, 4), dtype=np.float32))},
        coords={"time": times, "lat": np.arange(4.0), "lon": np.arange(4.0)},
    )
    ds["ET_actual"].attrs["units"] = "mm/day"
    ds.to_netcdf(grandchild / "ET_2010.nc")

    groups = scan_reference_directory(ref_root)
    assert len(groups) == 1, f"Expected one group, got: {[g.base_name for g in groups]}"
    variant = groups[0].variants["MidRes"]

    # sub_dir points to level-2 (grandchild)
    sub_dir = variant.variables["Evapotranspiration"]
    assert sub_dir.endswith("DeepData/0p25deg/daily"), f"Expected level-2 path, got: {sub_dir!r}"
    # tim_res detected from filename keyword in grandchild
    assert variant.tim_res == "Day"

    # Register and verify _inspect_nc_file finds the NC at level 2
    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    entry = catalog["DeepData_MidRes"]
    assert entry["variables"]["Evapotranspiration"]["varname"] == "ET_actual", (
        f"NC inspection at level 2 should populate varname, got: {entry['variables']['Evapotranspiration']}"
    )


def test_scan_root_dir_uses_openbench_ref_root_token_when_matching_env(tmp_path: Path, monkeypatch):
    """Catalog paths scanned under OPENBENCH_REF_ROOT should remain portable."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Grid" / "LowRes" / "Water" / "Runoff" / "PortableDemo"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {"ro": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "ro_2010.nc")
    monkeypatch.setenv("OPENBENCH_REF_ROOT", str(ref_root))

    groups = scan_reference_directory(ref_root)
    variant = groups[0].variants["LowRes"]

    assert variant.root_dir == "${OPENBENCH_REF_ROOT}/Grid/LowRes"


def test_scan_root_dir_uses_persisted_openbench_ref_root_token_when_env_unset(
    tmp_path: Path,
    monkeypatch,
):
    """After ref scan saves a root, scanner paths should remain portable."""
    import numpy as np
    import xarray as xr
    import yaml

    from openbench.data.registry.scanner import scan_reference_directory

    home = tmp_path / "home"
    settings_dir = home / ".openbench"
    settings_dir.mkdir(parents=True)
    ref_root = tmp_path / "Reference"
    (settings_dir / "settings.yaml").write_text(yaml.safe_dump({"reference_root": str(ref_root)}))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    nc_dir = ref_root / "Grid" / "LowRes" / "Water" / "Runoff" / "PortableDemo"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {"ro": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "ro_2010.nc")

    groups = scan_reference_directory(ref_root)
    variant = groups[0].variants["LowRes"]

    assert variant.root_dir == "${OPENBENCH_REF_ROOT}/Grid/LowRes"


def test_reference_scan_skips_generated_derived_dirs(tmp_path: Path):
    """Generated derived folders beside source datasets should not register as references."""
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import scan_reference_directory

    ref_root = tmp_path / "Reference"
    source_dir = ref_root / "Grid" / "LowRes" / "Water" / "Runoff" / "Demo"
    derived_dir = ref_root / "Grid" / "LowRes" / "Water" / "Runoff" / "Demo_derived"
    source_dir.mkdir(parents=True)
    derived_dir.mkdir(parents=True)

    ds = xr.Dataset(
        {"ro": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(source_dir / "ro_2010.nc")
    ds.to_netcdf(derived_dir / "ro_derived_2010.nc")

    groups = scan_reference_directory(ref_root)

    assert [group.base_name for group in groups] == ["Demo"]


def test_profile_scan_unknown_layout_warns(monkeypatch, tmp_path: Path, caplog):
    import logging

    from openbench.data.registry.scanner import scan_reference_directory

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "TypoProfile": {
                "scan": {
                    "layout": "staion_direct",
                    "root_sub_dir": "Station/Water/Bad",
                },
                "variables": {"Runoff": {"varname": "ro", "varunit": ""}},
            }
        },
    )
    ref_root = tmp_path / "Reference"
    ref_root.mkdir()

    with caplog.at_level(logging.WARNING, logger="openbench.data.registry.scanner"):
        scan_reference_directory(ref_root)

    assert "Unknown scan layout 'staion_direct' in profile 'TypoProfile'" in caplog.text


def test_station_generated_fulllist_uses_portable_reference_root(
    monkeypatch,
    tmp_path: Path,
):
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import register_scanned_datasets_batch

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Water" / "Streamflow" / "DemoStation"
    nc_dir.mkdir(parents=True)
    ds = xr.Dataset(
        {
            "q": (["time"], np.array([1.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": np.array(["2010-01-01"], dtype="datetime64[ns]")},
    )
    ds.to_netcdf(nc_dir / "S01_2010.nc")
    monkeypatch.setenv("OPENBENCH_REF_ROOT", str(ref_root))

    scanned = ScannedDataset(
        name="DemoStation",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir="${OPENBENCH_REF_ROOT}/Station/Water/Streamflow/DemoStation",
        variables={"Streamflow": ""},
    )
    catalog_path = tmp_path / "reference_catalog.yaml"

    register_scanned_datasets_batch([scanned], catalog_path=catalog_path)

    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["DemoStation"]
    assert entry["fulllist"] == "${OPENBENCH_REF_ROOT}/station_lists/DemoStation.csv"
    assert (ref_root / "station_lists" / "DemoStation.csv").exists()


def test_profile_file_glob_is_used_for_registration_inspection(monkeypatch, tmp_path: Path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import (
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "SharedDaily": {
                "scan": {
                    "layout": "station_shared_files",
                    "root_sub_dir": "Station/Water/Streamflow/Shared",
                    "file_glob": "Z_target_daily.nc",
                },
                "variables": {
                    "Streamflow": {"varname": "q_daily", "varunit": "m3 s-1"},
                },
            },
        },
    )

    ref_root = tmp_path / "Reference"
    nc_dir = ref_root / "Station" / "Water" / "Streamflow" / "Shared"
    nc_dir.mkdir(parents=True)
    monthly = xr.Dataset(
        {
            "q_monthly": (["time"], np.array([1.0, 2.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": pd.date_range("2010-01-01", periods=2, freq="MS")},
    )
    daily = xr.Dataset(
        {
            "q_daily": (["time"], np.array([1.0, 2.0], dtype=np.float32)),
            "lat": ((), 2.0),
            "lon": ((), 1.0),
        },
        coords={"time": pd.date_range("2010-01-01", periods=2, freq="D")},
    )
    monthly.to_netcdf(nc_dir / "A_other_monthly.nc")
    daily.to_netcdf(nc_dir / "Z_target_daily.nc")

    groups = scan_reference_directory(ref_root)
    variant = groups[0].variants["Station"]
    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch([variant], catalog_path=catalog_path)

    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["SharedDaily"]
    assert entry["tim_res"] == "Day"
    assert entry["_provenance"]["tim_res"] == "nc"
    assert entry["variables"]["Streamflow"]["varname"] == "q_daily"


def test_profile_placeholder_variables_do_not_prompt_before_profile_replacement(
    monkeypatch,
    tmp_path: Path,
):
    import numpy as np
    import xarray as xr

    from openbench.data.registry.scanner import register_scanned_datasets_batch

    monkeypatch.setattr(
        scanner_module,
        "_REFERENCE_PROFILES",
        {
            "GridComposite": {
                "variables": {
                    "Net_Radiation": {"varname": "Rn", "varunit": "W m-2"},
                    "Latent_Heat": {"varname": "LE", "varunit": "W m-2"},
                }
            }
        },
    )

    nc_dir = tmp_path / "dataset"
    nc_dir.mkdir()
    ds = xr.Dataset(
        {
            "Rn": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32)),
            "LE": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32)),
        },
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    ds.to_netcdf(nc_dir / "GridComposite_2010.nc")
    scanned = ScannedDataset(
        name="GridComposite",
        resolution="LowRes",
        category="Other",
        data_type="grid",
        root_dir=str(nc_dir),
        variables={"GridComposite": ""},
    )

    def fail_prompt(*_args):
        raise RuntimeError("prompted")

    catalog_path = tmp_path / "reference_catalog.yaml"
    register_scanned_datasets_batch(
        [scanned],
        catalog_path=catalog_path,
        on_multi_var=fail_prompt,
    )

    entry = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["GridComposite_LowRes"]
    assert set(entry["variables"]) == {"Net_Radiation", "Latent_Heat"}


def test_generate_station_list_uses_merged_parser_for_char_matrix_station_ids(tmp_path: Path):
    import netCDF4
    import numpy as np
    import pandas as pd

    from openbench.data.registry.scanner import generate_station_list

    nc_file = tmp_path / "merged.nc"
    with netCDF4.Dataset(nc_file, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("station", 2)
        nc.createDimension("nchar", 3)
        nc.createDimension("time", 1)
        station = nc.createVariable("station_id", "S1", ("station", "nchar"))
        station[:] = np.array([list("S01"), list("S02")], dtype="S1")
        lat = nc.createVariable("lat", "f4", ("station",))
        lon = nc.createVariable("lon", "f4", ("station",))
        lat[:] = [10.0, 20.0]
        lon[:] = [30.0, 40.0]
        q = nc.createVariable("q", "f4", ("time", "station"))
        q[:] = [[1.0, 2.0]]

    out = generate_station_list(tmp_path, tmp_path / "stations.csv")
    rows = pd.read_csv(out)

    assert rows["ID"].tolist() == ["S01", "S02"]
    assert rows["LAT"].tolist() == [10.0, 20.0]
    assert rows["LON"].tolist() == [30.0, 40.0]


def test_overlay_diff_records_deleted_bundled_fields_and_variables():
    from openbench.data.registry.scanner import _descriptor_overlay_diff, _merge_descriptor_overlay

    base = {
        "name": "Demo_LowRes",
        "data_type": "grid",
        "grid_res": 0.5,
        "variables": {
            "A": {"varname": "a", "varunit": ""},
            "B": {"varname": "b", "varunit": ""},
        },
    }
    descriptor = {
        "name": "Demo_LowRes",
        "data_type": "stn",
        "variables": {
            "A": {"varname": "a", "varunit": ""},
        },
    }

    overlay = _descriptor_overlay_diff(base, descriptor)
    assert overlay["data_type"] == "stn"
    assert overlay["grid_res"] is None
    assert overlay["variables"]["B"] is None
    assert _merge_descriptor_overlay(base, overlay) == descriptor


def test_cli_scan_dry_run_does_not_write_catalog(tmp_path: Path, monkeypatch):
    """openbench ref scan --dry-run lists what would be registered but
    does not modify the catalog file (or any registry state).
    """
    import openbench.cli.data as cli_data
    import openbench.data.registry.scanner as scanner_module
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset

    catalog_path = tmp_path / "reference_catalog.yaml"
    # Pre-existing catalog content — must remain untouched after dry-run
    catalog_path.write_text("ExistingDataset:\n  category: Water\n")
    original = catalog_path.read_text(encoding="utf-8")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)

    group = DatasetGroup(base_name="DryDemo")
    group.variants["LowRes"] = ScannedDataset(
        name="DryDemo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
    )

    register_called = {"flag": False}

    def fake_find_new_datasets(ref_root, on_progress=None, on_skip=None):
        return [group]

    def fake_register_batch(*args, **kwargs):
        register_called["flag"] = True

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register_batch)
    monkeypatch.setattr(cli_data.click, "echo", lambda *a, **k: None)
    monkeypatch.setattr(cli_data.click, "secho", lambda *a, **k: None)

    cli_data.scan.callback(str(tmp_path), auto=True, dry_run=True)

    assert not register_called["flag"], "Dry-run must NOT call register_scanned_datasets_batch"
    # Original catalog content unchanged
    assert catalog_path.read_text(encoding="utf-8") == original
    assert not (home / ".openbench" / "settings.yaml").exists()


def test_gui_multi_variable_selector_returns_chosen_nc_name(monkeypatch):
    """GUI scan registration should ask which NC variable to use."""
    pytest.importorskip("PySide6")
    from openbench.gui.dialogs import data_discovery

    def fake_get_item(parent, title, label, items, current, editable):
        assert "Multiple variables" in title
        assert "Runoff" in label
        assert "q_alt" in items[1]
        return items[1], True

    monkeypatch.setattr(data_discovery.QInputDialog, "getItem", fake_get_item)

    chosen = data_discovery.choose_nc_variable(
        None,
        "Runoff",
        "Water/Runoff/Demo",
        [
            {"name": "q", "unit": "m3 s-1", "dims": ["time"], "long_name": ""},
            {"name": "q_alt", "unit": "m3 s-1", "dims": ["time"], "long_name": "alternate"},
        ],
    )

    assert chosen == "q_alt"


def test_register_scanned_dataset_uses_union_years_across_variables(tmp_path: Path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    for dirname, year in (("var_a", 1990), ("var_b", 2005)):
        data_dir = tmp_path / dirname
        data_dir.mkdir()
        ds = xr.Dataset(
            {dirname: (["time", "lat", "lon"], np.zeros((1, 1, 1), dtype=np.float32))},
            coords={"time": pd.date_range(f"{year}-01-01", periods=1), "lat": [0.0], "lon": [0.0]},
        )
        ds.to_netcdf(data_dir / f"{dirname}_{year}.nc")

    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"VarA": "var_a", "VarB": "var_b"},
    )

    catalog = tmp_path / "reference_catalog.yaml"
    register_scanned_dataset(scanned, catalog_path=catalog)

    data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    assert data["Demo_LowRes"]["years"] == [1990, 2005]


def test_parse_merged_station_file_uses_first_non_time_dimension(tmp_path: Path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry.scanner import _parse_merged_station_file

    path = tmp_path / "stations.nc"
    ds = xr.Dataset(
        {
            "lat": ("station", np.array([10.0, 20.0])),
            "lon": ("station", np.array([100.0, 110.0])),
            "station_id": ("station", np.array(["A", "B"], dtype=object)),
            "extra_bounds": ("bounds", np.array([0, 1, 2])),
        },
        coords={
            "time": pd.date_range("2001-01-01", periods=1),
            "station": [0, 1],
            "bounds": [0, 1, 2],
        },
    )
    ds.to_netcdf(path)

    rows = _parse_merged_station_file(path, tmp_path)

    assert [row[0] for row in rows] == ["A", "B"]


def test_build_variables_uses_injected_remote_inspection(tmp_path):
    """Remote-scanned datasets ship NC inspection results computed on the remote host."""
    from openbench.data.registry.scanner import ScannedDataset, _build_variables

    scanned = ScannedDataset(
        name="RemoteSet",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path / "definitely-missing"),
        variables={"Runoff": "Runoff/RemoteSet"},
        nc_inspections={
            "Runoff": {
                "varname": "ro",
                "varunit": "mm/day",
                "all_data_vars": [
                    {"name": "ro", "unit": "mm/day"},
                    {"name": "ro2", "unit": "mm/d"},
                ],
                "nc_file_count": 3,
                "detected_grid_res": 0.5,
            }
        },
    )
    chooser_calls = []

    def on_multi_var(var_name, sub_dir, all_vars):
        chooser_calls.append((var_name, sub_dir, [v["name"] for v in all_vars]))
        return "ro2"

    descriptor = {}
    variables = _build_variables(scanned, descriptor, None, on_multi_var)

    assert chooser_calls == [("Runoff", "Runoff/RemoteSet", ["ro", "ro2"])]
    assert variables["Runoff"]["varname"] == "ro2"
    assert variables["Runoff"]["varunit"] == "mm/d"
    assert variables["Runoff"]["_nc_grid_res"] == 0.5


def test_detect_data_groupby_prefers_injected_value(tmp_path):
    from openbench.data.registry.scanner import ScannedDataset, _detect_data_groupby

    scanned = ScannedDataset(
        name="RemoteSet",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path / "missing"),
        variables={"Runoff": "sub"},
        detected_data_groupby="Month",
    )

    assert _detect_data_groupby(scanned) == "Month"


def test_finalize_descriptor_uses_remote_fulllist_for_remote_station_dataset(tmp_path):
    """Remote-scanned stn datasets carry a fulllist generated on the remote host."""
    from openbench.data.registry.scanner import ScannedDataset, _finalize_descriptor

    scanned = ScannedDataset(
        name="RemoteStations",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path / "missing"),
        variables={"Streamflow": "Q"},
        remote_fulllist="/remote/home/.openbench/station_lists/RemoteStations.csv",
    )
    descriptor = {"data_type": "stn"}

    _finalize_descriptor(scanned, descriptor, prov={})

    assert descriptor["fulllist"] == "/remote/home/.openbench/station_lists/RemoteStations.csv"


def test_build_variables_prefers_remote_inspection_over_coincidental_local_dir(tmp_path):
    """A remote root_dir that happens to exist locally must not shadow shipped results."""
    from openbench.data.registry.scanner import ScannedDataset, _build_variables

    (tmp_path / "Runoff" / "RemoteSet").mkdir(parents=True)  # exists locally, but empty

    scanned = ScannedDataset(
        name="RemoteSet",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Runoff": "Runoff/RemoteSet"},
        nc_inspections={
            "Runoff": {
                "varname": "ro",
                "varunit": "mm/day",
                "all_data_vars": [{"name": "ro", "unit": "mm/day"}],
                "nc_file_count": 1,
            }
        },
    )

    variables = _build_variables(scanned, {}, None, None)

    assert variables["Runoff"]["varname"] == "ro"
    assert variables["Runoff"]["varunit"] == "mm/day"


def test_finalize_descriptor_prefers_remote_fulllist_over_local_regeneration(tmp_path):
    from openbench.data.registry.scanner import ScannedDataset, _finalize_descriptor

    nc_dir = tmp_path / "Q"
    nc_dir.mkdir(parents=True)
    (nc_dir / "station1.nc").write_bytes(b"not a real nc")  # local glob would match

    scanned = ScannedDataset(
        name="RemoteStations",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir=str(tmp_path),
        variables={"Streamflow": "Q"},
        remote_fulllist="/remote/home/.openbench/station_lists/RemoteStations.csv",
    )
    descriptor = {"data_type": "stn"}

    _finalize_descriptor(scanned, descriptor, prov={})

    assert descriptor["fulllist"] == "/remote/home/.openbench/station_lists/RemoteStations.csv"


def test_resolve_station_nc_dir_shared_helper(tmp_path):
    """The remote scanner and _finalize_descriptor share one nc_dir selector."""
    from openbench.data.registry.scanner import resolve_station_nc_dir

    # root/sub holds no nc; root/sub/dataset does -> dataset wins
    dataset = tmp_path / "Q" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "s1.nc").write_bytes(b"x")

    nc_dir = resolve_station_nc_dir(str(tmp_path), {"Streamflow": "Q"})

    assert nc_dir == dataset
