"""The annual soil products are separate from the legacy SMAP snow products."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from openbench.data.file_lookup import missing_data_files, year_matches
from openbench.data.registry import scanner
from openbench.data.registry.manager import RegistryManager

REGISTRY = Path(scanner.__file__).parent
VARIABLES = {
    "Surface_Soil_Moisture": ("sm_surface", "m3 m-3", "Water"),
    "Root_Zone_Soil_Moisture": ("sm_rootzone", "m3 m-3", "Water"),
    "Surface_Soil_Temperature": ("soil_temp_layer1", "K", "Meteo"),
    "Root_Zone_Soil_Temperature": ("soil_temp_rootzone_0_1m_estimate", "K", "Meteo"),
}
RESOLUTIONS = [("MidRes", 0.25, "Day"), ("LowRes", 0.5, "Month")]


def assert_variables(variables):
    assert set(variables) == set(VARIABLES)
    for logical, (native, unit, category) in VARIABLES.items():
        entry = variables[logical]
        assert entry["varname"] == native
        assert entry["varunit"] == unit
        assert entry["sub_dir"] == f"{category}/{logical}/SMAP_L4_Soil"
        assert entry["prefix"] == f"SMAP_L4_{native}_"
        assert entry.get("suffix", "") == ""
        assert f"{entry['prefix']}2016{entry.get('suffix', '')}.nc" == f"SMAP_L4_{native}_2016.nc"


def assert_description(description):
    for text in (
        "0.1",
        "daily",
        "user-approved",
        "nominal 0–1 m",
        "p=0",
        "not a native NASA field",
        "0.0988",
        "0.1952",
        "0.3859",
        "0.3201",
        "2016-06-13",
        "date retained",
    ):
        assert text in description


@pytest.mark.parametrize("resolution,grid,frequency", RESOLUTIONS)
def test_soil_catalog_and_profiles(resolution, grid, frequency, tmp_path):
    name = f"SMAP_L4_Soil_{resolution}"
    catalog = yaml.safe_load((REGISTRY / "reference_catalog.yaml").read_text(encoding="utf-8"))
    profiles = yaml.safe_load((REGISTRY / "reference_profiles.yaml").read_text(encoding="utf-8"))
    for entry in (catalog[name], profiles[name]):
        assert entry["tim_res"] == frequency
        assert entry["data_groupby"] == "Year"
        assert entry["years"] == [2016, 2024]
        assert_description(entry["description"])
    assert_variables(catalog[name]["variables"])
    assert profiles[name]["variables"] == {
        logical: {"varname": native, "varunit": unit} for logical, (native, unit, _) in VARIABLES.items()
    }
    assert catalog[name]["root_dir"] == f"${{OPENBENCH_REF_ROOT}}/Grid/{resolution}"
    assert catalog[name]["grid_res"] == grid
    registry = RegistryManager(user_dir=tmp_path)
    ref = registry.get_reference(name)
    assert ref.name == name
    for logical, (native, unit, _) in VARIABLES.items():
        assert ref.variables[logical].varname == native
        assert ref.variables[logical].varunit == unit
        assert name in {r.name for r in registry.references_for_variable(logical)}


def test_resolution_selection_and_legacy_snow(tmp_path):
    registry = RegistryManager(user_dir=tmp_path)
    assert set(registry.get_resolution_variants("SMAP_L4_Soil")) == {"LowRes", "MidRes"}
    assert registry.get_reference("SMAP_L4_Soil", sim_tim_res="Day", sim_grid_res=0.5).name.endswith("MidRes")
    assert registry.get_reference("SMAP_L4_Soil", sim_tim_res="Month", sim_grid_res=0.5).name.endswith("LowRes")
    for resolution, _, frequency in RESOLUTIONS:
        snow = registry.get_reference(f"SMAP_L4_{resolution}")
        assert snow.data_groupby == ("Month" if resolution == "MidRes" else "Single")
        assert snow.tim_res == frequency
        assert set(snow.variables) == {"Snow_Depth"}
        var = snow.variables["Snow_Depth"]
        assert (var.varname, var.varunit, var.sub_dir) == ("snow_depth", "m", "Water/Snow_Depth/SMAP_L4")
        assert var.prefix == ("SMAP_L4_snow_depth_" if resolution == "MidRes" else "SMAP_L4_snow_depth")
        assert var.suffix == ""
    profile = scanner.get_reference_profile("SMAP_L4")
    assert profile["data_groupby"] == "Single"
    assert profile["variables"] == {"Snow_Depth": {"varname": "snow_depth", "varunit": "m"}}


def test_real_scan_merges_four_folders_and_registers_annual_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", None)
    root = tmp_path / "Reference"
    for resolution, grid, frequency in RESOLUTIONS:
        for logical, (native, unit, category) in VARIABLES.items():
            folder = root / "Grid" / resolution / category / logical / "SMAP_L4_Soil"
            folder.mkdir(parents=True)
            for year in (2016, 2024):
                time = pd.date_range(f"{year}-01-01", periods=3, freq="D" if frequency == "Day" else "MS")
                ds = xr.Dataset(
                    {
                        native: (
                            ("time", "lat", "lon"),
                            np.full((3, 2, 2), 0.3 if unit != "K" else 280, dtype="f4"),
                            {"units": unit},
                        )
                    },
                    coords={"time": time, "lat": [0, grid], "lon": [0, grid]},
                )
                ds.to_netcdf(folder / f"SMAP_L4_{native}_{year}.nc")
    groups = scanner.scan_reference_directory(root, max_workers=1)
    assert len(groups) == 1
    group = groups[0]
    assert group.base_name == "SMAP_L4_Soil"
    assert set(group.variants) == {"MidRes", "LowRes"}
    for scanned in group.variants.values():
        assert set(scanned.variables) == set(VARIABLES)
    output = tmp_path / "registered.yaml"
    scanner.register_scanned_datasets_batch(list(group.variants.values()), catalog_path=output)
    registered = yaml.safe_load(output.read_text(encoding="utf-8"))
    for resolution, grid, frequency in RESOLUTIONS:
        entry = registered[f"SMAP_L4_Soil_{resolution}"]
        assert entry["tim_res"] == frequency
        assert entry["grid_res"] == grid
        assert entry["data_groupby"] == "Year"
        assert entry["years"] == [2016, 2024]
        assert_variables(entry["variables"])
        assert_description(entry["description"])


def test_legacy_midres_snow_month_files_are_found_and_rescan_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", None)
    root = tmp_path / "Reference"
    folder = root / "Grid/MidRes/Water/Snow_Depth/SMAP_L4"
    folder.mkdir(parents=True)
    expected = []
    for month in ("201601", "201602", "202401"):
        path = folder / f"SMAP_L4_snow_depth_{month}.nc"
        xr.Dataset(
            {"snow_depth": (("time", "lat", "lon"), np.zeros((3, 2, 2)), {"units": "m"})},
            coords={
                "time": pd.date_range(f"{month[:4]}-{month[4:]}-01", periods=3, freq="D"),
                "lat": [0, 0.25],
                "lon": [0, 0.25],
            },
        ).to_netcdf(path)
        expected.append(str(path))
    ref = RegistryManager(user_dir=tmp_path / "registry").get_reference("SMAP_L4_MidRes")
    var = ref.variables["Snow_Depth"]
    assert missing_data_files(str(folder), var.prefix, var.suffix, ref.data_groupby, [2016, 2024]) == []
    assert year_matches(str(folder), var.prefix, var.suffix, [2016, 2024]) == {2016: expected[:2], 2024: expected[2:]}
    groups = scanner.scan_reference_directory(root, max_workers=1)
    assert len(groups) == 1
    scanned = groups[0].variants["MidRes"]
    output = tmp_path / "registered.yaml"
    scanner.register_scanned_datasets_batch([scanned], catalog_path=output)
    entry = yaml.safe_load(output.read_text(encoding="utf-8"))["SMAP_L4_MidRes"]
    assert entry["tim_res"] == "Day"
    assert entry["data_groupby"] == "Month"
    assert entry["variables"]["Snow_Depth"]["prefix"] == var.prefix
    assert entry["variables"]["Snow_Depth"].get("suffix", "") == ""
