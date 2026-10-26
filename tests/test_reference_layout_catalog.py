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


@pytest.mark.parametrize(
    "name,years",
    [
        ("LGS_Reco_LowRes", [1985, 2015]),
        ("LGS_Reco_MidRes", [1985, 2015]),
        ("GIEMS_v2_LowRes", [1992, 2015]),
        ("FLUXCOM-X-BASE_LowRes", [2001, 2020]),
        ("FLUXCOM-X-BASE_MidRes", [2001, 2020]),
    ],
)
def test_reference_period_corrections_survive_fresh_scanning(name, years, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from openbench.data.registry import scanner

    catalog = _catalog()
    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference(name)
    assert ref.years == years
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", None)
    family, resolution = name.rsplit("_", 1)
    assert scanner.get_reference_profile(family)["years"] == years
    root = tmp_path / "Reference"
    resolution_root = root / "Grid" / resolution
    sample_years = [years[0], years[1]]
    if family == "LGS_Reco":
        sample_years.insert(0, 1982)
    elif family == "FLUXCOM-X-BASE":
        sample_years.append(2021)
    grid = ref.grid_res
    for logical, mapping in ref.variables.items():
        directory = resolution_root / mapping.sub_dir
        directory.mkdir(parents=True)
        for year in sample_years:
            time = pd.date_range(f"{year}-01-01", periods=3, freq="D" if ref.tim_res == "Day" else "MS")
            filename = (
                f"GIEMS_v2_2020_{mapping.varname}_15min_{year}.nc"
                if family == "GIEMS_v2"
                else f"{mapping.prefix}{year}{mapping.suffix or ''}.nc"
            )
            xr.Dataset(
                {mapping.varname: (("time", "lat", "lon"), np.ones((3, 2, 2)), {"units": mapping.varunit})},
                coords={"time": time, "lat": [0, grid], "lon": [0, grid]},
            ).to_netcdf(directory / filename)
    groups = scanner.scan_reference_directory(root, max_workers=1)
    assert len(groups) == 1
    scanned = groups[0].variants[resolution]
    output = tmp_path / "registered.yaml"
    scanner.register_scanned_datasets_batch([scanned], catalog_path=output)
    registered = yaml.safe_load(output.read_text())[name]
    assert registered["years"] == years
    for logical in catalog[name]["variables"]:
        assert registered["variables"][logical]["prefix"] == ref.variables[logical].prefix
        assert registered["variables"][logical].get("suffix", "") == (ref.variables[logical].suffix or "")


@pytest.mark.parametrize("logical,native", [("Inundation_Area", "fldare"), ("Inundation_Fraction", "fldfrc")])
def test_giems_version_label_is_not_the_scientific_year(logical, native, tmp_path):
    from openbench.data.file_lookup import year_matches

    mapping = (
        registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("GIEMS_v2_LowRes").variables[logical]
    )
    expected = {}
    for year in range(1992, 2016):
        path = tmp_path / f"GIEMS_v2_2020_{native}_15min_{year}.nc"
        path.touch()
        expected[year] = [str(path)]
    assert mapping.prefix == f"GIEMS_v2_2020_{native}_15min_"
    assert (mapping.suffix or "") == ""
    assert year_matches(str(tmp_path), mapping.prefix, mapping.suffix or "", range(1992, 2016)) == expected


@pytest.mark.parametrize(
    "resolution,variable,years",
    [
        ("LowRes", "Latent_Heat", [1980, 2022]),
        ("LowRes", "Net_Radiation", [1970, 2020]),
        ("LowRes", "Sensible_Heat", [1970, 2020]),
        ("LowRes", "Surface_Albedo", [1979, 2020]),
        ("LowRes", "Surface_Net_LW_Radiation", [1970, 2020]),
        ("LowRes", "Surface_Net_SW_Radiation", [1970, 2020]),
        ("LowRes", "Surface_Upward_LW_Radiation", [1979, 2020]),
        ("LowRes", "Surface_Upward_SW_Radiation", [1979, 2020]),
        ("LowRes", "Surface_Specific_Humidity", [1980, 2022]),
        ("MidRes", "Latent_Heat", [1970, 2022]),
        ("MidRes", "Net_Radiation", [1970, 2020]),
        ("MidRes", "Sensible_Heat", [1970, 2020]),
        ("MidRes", "Surface_Albedo", [1979, 2020]),
        ("MidRes", "Surface_Net_LW_Radiation", [1970, 2020]),
        ("MidRes", "Surface_Net_SW_Radiation", [1970, 2020]),
        ("MidRes", "Surface_Upward_LW_Radiation", [1979, 2020]),
        ("MidRes", "Surface_Upward_SW_Radiation", [1979, 2020]),
        ("MidRes", "Surface_Downward_LW_Radiation", [1950, 2022]),
        ("MidRes", "Surface_Downward_SW_Radiation", [1950, 2022]),
    ],
)
def test_era5land_variable_periods_preserve_actual_coverage(resolution, variable, years, tmp_path):
    from openbench.data.registry.schema import reference_years

    name = f"ERA5LAND_{resolution}"
    raw = _catalog()[name]
    assert raw["years"] == [1950, 2022]
    assert raw["variables"][variable].get("years", raw["years"]) == years
    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference(name)
    assert reference_years(ref, ref.variables[variable]) == years


def test_retired_globsnow_backup_is_not_advertised(tmp_path):
    registry = registry_manager.RegistryManager(user_dir=tmp_path / "user")
    assert registry.get_reference("globsnow_v3-bk_MidRes") is None
    assert registry.get_reference("globsnow_v3_MidRes") is not None
    assert registry.get_reference("globsnow_v3_LowRes") is not None


def test_cru_precipitation_uses_processed_daily_rate(tmp_path):
    registry = registry_manager.RegistryManager(user_dir=tmp_path / "user")
    ref = registry.get_reference("CRU_TS_4.08_LowRes")
    mapping = ref.variables["Precipitation"]
    assert mapping.sub_dir == "Water/Precipitation/CRU_TS_4.08"
    assert mapping.prefix == "pre"
    assert mapping.suffix in (None, "")
    assert mapping.varname == "pre"
    assert mapping.varunit == "mm day-1"
    assert mapping.years == [1901, 2023]


def test_era5land_lowres_wind_uses_half_degree_monthly_product(tmp_path):
    from openbench.data.registry.schema import reference_years

    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("ERA5LAND_LowRes")
    mapping = ref.variables["Surface_Wind_Speed"]
    assert ref.grid_res == 0.5
    assert mapping.sub_dir == "Meteo/Surface_Wind_Speed/ERA5LAND"
    assert mapping.prefix == "ERA5LAND_"
    assert mapping.suffix == "_ws_monmean_0p5"
    assert mapping.varname == "ws"
    assert mapping.varunit == "m s-1"
    assert reference_years(ref, mapping) == [1950, 2022]


def test_era5land_midres_et_is_already_processed(tmp_path):
    from openbench.data.registry.schema import reference_years

    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("ERA5LAND_MidRes")
    mapping = ref.variables["Evapotranspiration"]
    assert mapping.sub_dir == "Water/Evapotranspiration/ERA5LAND"
    assert mapping.prefix == "ERA5LAND_"
    assert mapping.suffix == "_ET_1D_0p25"
    assert mapping.varname == "ET"
    assert mapping.varunit == "mm day-1"
    assert not mapping.compute
    assert reference_years(ref, mapping) == [1970, 2022]


def test_grfr_high_resolution_uses_existing_tenth_degree_daily_product(tmp_path):
    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("GRFR_HigRes")
    mapping = ref.variables["Total_Runoff"]
    assert ref.grid_res == 0.1
    assert ref.tim_res == "Day"
    assert mapping.sub_dir == "Water/Total_Runoff/GRFR/0p1_daily"
    assert mapping.prefix == "RUNOFF_remap_"
    assert mapping.varname == "ro"
    assert mapping.varunit == "mm day-1"
    assert ref.years == [1979, 2019]


def test_gpcc_lowres_uses_native_half_degree_v2022_daily_rate(tmp_path):
    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("GPCC_LowRes")
    mapping = ref.variables["Precipitation"]
    assert ref.grid_res == 0.5
    assert ref.tim_res == "Month"
    assert ref.data_groupby.lower() == "single"
    assert ref.years == [1891, 2020]
    assert mapping.sub_dir == "Water/Precipitation/GPCC"
    assert mapping.prefix == "precip.mon.total.0.5x0.5.v2022_mm_day"
    assert mapping.varname == "precip"
    assert mapping.varunit == "mm day-1"
    assert not mapping.compute


def test_rn_highres_uses_daily_mean_radiation_units(tmp_path):
    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("Rn-Xu2022_HigRes")
    mapping = ref.variables["Net_Radiation"]
    assert ref.grid_res == 0.05
    assert ref.tim_res == "Day"
    assert mapping.varname == "Rn"
    assert mapping.varunit == "W m-2"
    assert mapping.sub_dir == "Heat/Net_Radiation/Rn-Xu2022/netcdf"
    assert not mapping.compute


@pytest.mark.parametrize("variable", ["Evapotranspiration", "Canopy_Transpiration"])
def test_fluxcom_x_base_lowres_water_uses_repaired_half_degree_files(variable, tmp_path):
    raw = _catalog()["FLUXCOM-X-BASE_LowRes"]
    assert raw["variables"][variable]["suffix"] == "_050_monthly"
    assert "grid_res" not in raw["variables"][variable]
    ref = registry_manager.RegistryManager(user_dir=tmp_path / "user").get_reference("FLUXCOM-X-BASE_LowRes")
    assert ref.grid_res == 0.5
    assert ref.variables[variable].suffix == "_050_monthly"
