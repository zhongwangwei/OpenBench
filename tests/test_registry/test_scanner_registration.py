"""Tests for registration of scanned datasets."""

from pathlib import Path
from types import SimpleNamespace

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

    text = catalog.read_text()
    assert "1980" not in text
    assert "2023" not in text
    assert "years:" not in text


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

    data = yaml.safe_load(catalog.read_text())
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

    data = yaml.safe_load(catalog.read_text())
    variable = data["Demo_LowRes"]["variables"]["Evapotranspiration"]

    assert variable["varname"] == "Evapotranspiration"
    assert variable["varunit"] == ""
    assert "prefix" not in variable
    assert "suffix" not in variable


def test_cli_scan_prefers_base_name_existing_descriptor_before_registry_name(
    monkeypatch,
    tmp_path: Path,
):
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

    def fake_find_new_datasets(ref_root, on_progress=None):
        return [group]

    def fake_register_batch(datasets, on_multi_var=None, on_progress=None, catalog_path=None):
        for ds in datasets:
            captured["scanned"] = ds.registry_name
        return tmp_path / "reference_catalog.yaml"

    monkeypatch.setattr(scanner_module, "find_new_datasets", fake_find_new_datasets)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register_batch)
    monkeypatch.setattr(cli_data.click, "echo", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_data.click, "secho", lambda *args, **kwargs: None)

    cli_data.scan.callback(str(tmp_path), auto=True)

    assert captured["scanned"] == "Demo_LowRes"


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

    data = yaml.safe_load(catalog_path.read_text())
    var = data["Demo_LowRes"]["variables"]["Evapotranspiration"]

    # Hand-edited fields should be preserved (not overwritten by scanner)
    assert var["varname"] == "my_custom_et"
    assert var["varunit"] == "mm/day"
    assert var["prefix"] == "my_prefix_"
    assert var["suffix"] == "_my_suffix"


def test_register_profile_uses_writable_registry_dir_and_clears_caches(
    monkeypatch,
    tmp_path: Path,
):
    writable_registry = tmp_path / "user-registry"
    writable_registry.mkdir()
    fake_package_cli = tmp_path / "pkg" / "cli" / "data.py"
    fake_package_cli.parent.mkdir(parents=True)
    fake_package_registry = fake_package_cli.parent.parent / "data" / "registry"
    fake_package_registry.mkdir(parents=True)

    registry_cleared = []
    profile_cleared = []

    monkeypatch.setattr(cli_data, "__file__", str(fake_package_cli))
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_registry_dir",
        lambda: writable_registry,
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

    assert (writable_registry / "reference_profiles.yaml").exists()
    assert not (fake_package_registry / "reference_profiles.yaml").exists()
    saved = yaml.safe_load((writable_registry / "reference_profiles.yaml").read_text())
    assert saved["DemoProfile"]["variables"]["Runoff"]["varname"] == "ro"
    assert registry_cleared == [True]
    assert profile_cleared == [True]


def test_reference_profiles_load_from_writable_registry_dir(monkeypatch, tmp_path: Path):
    writable_registry = tmp_path / "user-registry"
    writable_registry.mkdir()
    fake_scanner_file = tmp_path / "pkg" / "registry" / "scanner.py"
    fake_scanner_file.parent.mkdir(parents=True)

    (writable_registry / "reference_profiles.yaml").write_text(
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

    monkeypatch.setattr(scanner_module, "__file__", str(fake_scanner_file))
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_registry_dir",
        lambda: writable_registry,
    )
    scanner_module._REFERENCE_PROFILES = None

    profile = scanner_module.get_reference_profile("DemoProfile_LowRes")

    assert profile is not None
    assert profile["variables"]["Runoff"]["varname"] == "ro"


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
    data = yaml.safe_load(catalog_path.read_text())
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
    ds = xr.Dataset({
        "ET": (["time", "y", "x"], np.zeros((12, 1, 1))),
        "lat": ([], 47.0),
        "lon": ([], 11.0),
    })
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
    ds = xr.Dataset({
        "H": (["time", "lat", "lon"], np.zeros((12, 180, 360))),
    }, coords={
        "lat": np.linspace(-89.5, 89.5, 180),
        "lon": np.linspace(-179.5, 179.5, 360),
    })
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
        ds = xr.Dataset({"ro": (["time", "lat", "lon"], np.zeros((12, 10, 20)))},
                        coords={"lat": np.arange(10), "lon": np.arange(20)})
        ds.to_netcdf(nc_dir / f"ro_{year}_test.nc4")

    variant = ScannedDataset(
        name="TestNC4", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
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


def test_register_to_dict_nc_grid_res_overrides_resolution_map(tmp_path: Path):
    """NC-detected grid_res should win over RESOLUTION_MAP default."""
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

    # NC says 0.25, RESOLUTION_MAP["LowRes"] says 0.5 — NC wins
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
    original = catalog_path.read_text()

    scanned = ScannedDataset(
        name="DemoNew", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
        variables={"Evapotranspiration": "."},
    )

    with pytest.raises(RuntimeError, match="Failed to load existing catalog"):
        register_scanned_dataset(scanned, catalog_path=catalog_path)

    # Original (corrupted) file untouched — user can recover manually
    assert catalog_path.read_text() == original


def test_register_creates_backup_before_overwriting_existing_catalog(tmp_path: Path):
    """Each successful write must back up the previous catalog state to .bak,
    so a buggy rescan that overwrites hand-edited fields can be undone.
    """
    catalog_path = tmp_path / "reference_catalog.yaml"
    catalog_path.write_text(
        "OldDataset_LowRes:\n"
        "  name: OldDataset_LowRes\n"
        "  category: Water\n"
        "  data_type: grid\n"
    )

    scanned = ScannedDataset(
        name="NewDataset", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
        variables={"Evapotranspiration": "."},
    )
    register_scanned_dataset(scanned, catalog_path=catalog_path)

    backup_path = Path(str(catalog_path) + ".bak")
    assert backup_path.exists(), f"Expected backup at {backup_path}"
    bak_data = yaml.safe_load(backup_path.read_text())
    assert "OldDataset_LowRes" in bak_data
    # New file has both
    new_data = yaml.safe_load(catalog_path.read_text())
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
    nested_dir = (
        ref_root / "Grid" / "LowRes" / "Water" / "Evapotranspiration"
        / "MyData" / "0p25deg-daily"
    )
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
    assert "0p25deg-daily" in sub_dir, (
        f"sub_dir must point to NC-bearing child directory, got: {sub_dir!r}"
    )
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

    with caplog.at_level(logging.WARNING, logger="openbench.data.registry.scanner"):
        groups = scan_reference_directory(ref_root)

    assert groups == [], f"Multi-child dataset must be skipped, got: {[g.base_name for g in groups]}"
    assert any(
        "NC-bearing subdirectories" in rec.message for rec in caplog.records
    ), f"Expected ambiguity warning, got messages: {[r.message for r in caplog.records]}"


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
        name="Demo", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
    )
    group.variants["MidRes"] = ScannedDataset(
        name="Demo", resolution="MidRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
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

    assert info.get("detected_data_type") == "stn", (
        f"Expected stn detection, got: {info.get('detected_data_type')!r}"
    )
    assert info.get("varname") == "Qle_cor", (
        f"Expected Qle_cor as data var, got: {info.get('varname')!r}. "
        "1D station variables must not be filtered out."
    )
    assert info.get("varunit") == "W/m2"


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
        name="Demo", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )

    # First registration: descriptor entirely from scan
    catalog: dict = {}
    _register_to_dict(scanned, catalog)

    # User edits these fields by hand:
    catalog["Demo_LowRes"]["description"] = "Custom user description for Demo"
    catalog["Demo_LowRes"]["category"] = "Energy"   # user re-categorized
    catalog["Demo_LowRes"]["years"] = [2000, 2010]  # user fixed years

    # Re-register (same group; existing_descriptor passed in like batch path does)
    existing = catalog["Demo_LowRes"]
    _register_to_dict(scanned, catalog, existing_descriptor=existing)
    new_entry = catalog["Demo_LowRes"]

    assert new_entry["description"] == "Custom user description for Demo"
    assert new_entry["category"] == "Energy"
    assert new_entry["years"] == [2000, 2010]


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
        lat = nc.createVariable("lat", "f4", ()); lat.units = "degrees_north"
        lon = nc.createVariable("lon", "f4", ()); lon.units = "degrees_east"

    user_fulllist = tmp_path / "user_curated_stations.csv"
    user_fulllist.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nfluxstn,2010,2010,11.0,47.0,\n")

    scanned = ScannedDataset(
        name="FluxStn", resolution="Station", category="Carbon",
        data_type="stn", root_dir=str(tmp_path),
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
    assert user_fulllist.read_text().startswith("ID,SYEAR,EYEAR,LON,LAT,DIR\n")


def test_data_groupby_detects_monthly_files(tmp_path: Path):
    """Files like ET_2010_01.nc / ET_2010_02.nc / ... must be classified as
    Month, not Year. The previous implementation only output Year or Single.
    """
    from openbench.data.registry.scanner import _build_base_descriptor, ScannedDataset

    nc_dir = tmp_path / "var_dir"
    nc_dir.mkdir()
    for year in (2010, 2011):
        for month in (1, 2, 3):
            (nc_dir / f"ET_{year}_{month:02d}.nc").write_text("")

    scanned = ScannedDataset(
        name="MonthlyData", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
        variables={"Evapotranspiration": "var_dir"},
    )
    desc = _build_base_descriptor(scanned, prov={})
    assert desc["data_groupby"] == "Month", f"Got: {desc['data_groupby']}"


def test_data_groupby_detects_daily_files(tmp_path: Path):
    """Files like ET_20100101.nc / ET_20100102.nc must be classified as Day."""
    from openbench.data.registry.scanner import _build_base_descriptor, ScannedDataset

    nc_dir = tmp_path / "var_dir"
    nc_dir.mkdir()
    for day in (1, 2, 3, 4, 5):
        (nc_dir / f"ET_201001{day:02d}.nc").write_text("")

    scanned = ScannedDataset(
        name="DailyData", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
        variables={"Evapotranspiration": "var_dir"},
    )
    desc = _build_base_descriptor(scanned, prov={})
    assert desc["data_groupby"] == "Day", f"Got: {desc['data_groupby']}"


def test_tim_res_provenance_default_when_no_evidence(tmp_path: Path):
    """When neither filename nor NC content has time-resolution evidence,
    descriptor.tim_res should fall to "Month" with provenance="default",
    NOT provenance="scan" which would falsely claim directory-inferred.
    """
    from openbench.data.registry.scanner import _register_to_dict, ScannedDataset

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
        name="Mystery", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
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
        name="CacheTest", resolution="LowRes", category="Water",
        data_type="grid", root_dir=str(tmp_path),
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
    from openbench.data.registry.scanner import _inspect_nc_file
    import netCDF4

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
