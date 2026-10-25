"""Catalog paths follow the retained standardized reference layout."""

from pathlib import Path

import pytest
import yaml

from openbench.data.registry import manager as registry_manager


def _catalog(filename="reference_catalog.yaml"):
    return yaml.safe_load(Path(registry_manager.__file__).with_name(filename).read_text())


@pytest.mark.parametrize("version", ["4.2a", "4.3a", "4.3b"])
@pytest.mark.parametrize("resolution", ["LowRes", "MidRes"])
def test_gleam_transpiration_uses_one_canonical_mapping(version, resolution):
    variables = _catalog()[f"GLEAM_v{version}_{resolution}"]["variables"]
    assert "Transpiration" not in variables
    assert variables["Canopy_Transpiration"]["sub_dir"] == f"Water/Canopy_Transpiration/GLEAM_v{version}"
    assert variables["Canopy_Transpiration"]["varname"] == "Et"
    assert variables["Canopy_Transpiration"]["varunit"] == "mm day-1"


@pytest.mark.parametrize(
    "dataset,variable,path",
    [
        ("FLUXCOM-X-BASE_LowRes", "Canopy_Transpiration", "Water/Canopy_Transpiration/FLUXCOM-X-BASE"),
        ("FLUXCOM-X-BASE_MidRes", "Canopy_Transpiration", "Water/Canopy_Transpiration/FLUXCOM-X-BASE"),
        (
            "Transpiration_Li_etal2024_LowRes",
            "Canopy_Transpiration",
            "Water/Canopy_Transpiration/Transpiration_Li_etal2024",
        ),
        ("ERA5LAND_MidRes", "Surface_Upward_LW_Radiation", "Heat/Surface_Upward_LW_Radiation/ERA5LAND"),
        ("globsnow_v3_MidRes", "Snow_Water_Equivalent", "Water/Snow_Water_Equivalent/globsnow_v3"),
        ("MSWEP_0p25_MidRes", "Precipitation", "Water/Precipitation/MSWEP"),
        ("Rn-Xu2022_MidRes", "Net_Radiation", "Heat/Net_Radiation/Rn-Xu2022"),
        ("Rn-Xu2022_0p25_MidRes", "Net_Radiation", "Heat/Net_Radiation/Rn-Xu2022"),
        ("ET_Xu_etal_2025_MidRes", "Evapotranspiration", "Water/Evapotranspiration/ET_Xu_etal_2025_0p25"),
        ("LGS_Reco_LowRes", "Ecosystem_Respiration", "Bio/Ecosystem_Respiration/LGS_Reco"),
        ("LGS_Reco_MidRes", "Ecosystem_Respiration", "Bio/Ecosystem_Respiration/LGS_Reco"),
    ],
)
def test_retained_reference_paths(dataset, variable, path):
    assert _catalog()[dataset]["variables"][variable]["sub_dir"].rstrip("/") == path


@pytest.mark.parametrize("resolution,variable_count", [("LowRes", 7), ("MidRes", 8)])
def test_mswx_retains_native_fields_when_redundant_folder_suffix_is_removed(resolution, variable_count):
    variables = _catalog()[f"MSWX_0p25_{resolution}"]["variables"]
    assert len(variables) == variable_count
    for mapping in variables.values():
        assert mapping["sub_dir"].endswith("/MSWX")
    assert variables["Surface_Air_Temperature"]["varname"] == "air_temperature"
    assert variables["Surface_Air_Temperature"]["varunit"] == "degree_Celsius"
    if resolution == "MidRes":
        assert variables["Precipitation"]["varunit"] == "mm 3h-1"


def test_rn_xu_uses_daily_samples_in_monthly_files_and_flat_scan_layout():
    catalog = _catalog()
    for name in ["Rn-Xu2022_MidRes", "Rn-Xu2022_0p25_MidRes"]:
        assert catalog[name]["tim_res"] == "Day"
        assert catalog[name]["data_groupby"] == "Month"
    scan = _catalog("reference_profiles.yaml")["Rn-Xu2022_0p25"]["scan"]
    assert scan["root_sub_dir"] == "Grid/MidRes/Heat/Net_Radiation/Rn-Xu2022"
    assert scan["nc_sub_dir"] == "."


def test_era5land_daily_minimum_accepts_native_t2m_without_dropping_mn2t(tmp_path):
    registry = registry_manager.RegistryManager(user_dir=tmp_path)
    mapping = registry.get_reference("ERA5LAND_LowRes").variables["Diurnal_Min_Temperature"]
    assert mapping.varname == "mn2t"
    assert mapping.varunit == "K"
    assert mapping.suffix == "_daymin_2m_temperature_1D_0p25"
    assert [(fb.varname, fb.varunit, fb.convert) for fb in mapping.fallbacks or []] == [("t2m", "K", "")]
    profile = _catalog("reference_profiles.yaml")["ERA5LAND"]["variables"]["Diurnal_Min_Temperature"]
    assert profile["varname"] == "mn2t"
    assert profile["fallbacks"] == [{"varname": "t2m", "varunit": "K"}]
