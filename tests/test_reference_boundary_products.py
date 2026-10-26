"""References with distinct point-depth, monthly-count and area-basis meanings."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from openbench.data.file_lookup import select_data_files
from openbench.data.registry import manager, scanner
from openbench.data.registry.schema import reference_years

DEPTHS = (5, 10, 15, 20, 40)
UPCH4 = "UpCH4_v0.4_WAD2M_LowRes"
WITHDRAWALS = {
    "Domestic_Water_Withdrawal": "dom",
    "Manufacturing_Water_Withdrawal": "manu",
    "Irrigation_Water_Withdrawal": "irr",
    "Thermoelectric_Water_Withdrawal": "ele",
}
POINTS = {f"Soil_Temperature_{d}cm" for d in DEPTHS}


def _fixture(path, native, units, frequency="MS", extra=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {v: (("time", "lat", "lon"), np.ones((12, 2, 2), dtype="float32")) for v in (native, *extra)},
        coords={"time": pd.date_range("2001-01-01", periods=12, freq=frequency), "lat": [0, 0.25], "lon": [0, 0.25]},
    )
    ds[native].attrs["units"] = units
    ds.to_netcdf(path)


def test_bundled_semantic_boundaries(tmp_path):
    registry = manager.RegistryManager(user_dir=tmp_path)
    cru = registry.get_reference("CRU_TS_4.08_LowRes")
    frost = cru.variables["Ground_Frost_Frequency"]
    assert (frost.varname, frost.varunit, frost.prefix) == ("frs", "days", "frs")
    assert reference_years(cru, frost) == [1901, 2023]
    assert (cru.tim_res, cru.data_groupby) == ("Month", "Single")
    homts = registry.get_reference("HOMTS_MidRes")
    assert set(homts.variables) == POINTS | {"Surface_Soil_Temperature"}
    assert (homts.tim_res, homts.data_groupby, homts.years) == ("Day", "Single", [1960, 2017])
    for depth in DEPTHS:
        item = homts.variables[f"Soil_Temperature_{depth}cm"]
        assert (item.varname, item.varunit) == ("TS", "degC")
        assert item.prefix == f"HOM_TS{depth}cm_daily_025x025_1960_2017"
        assert item.sub_dir == "Meteo/Root_Zone_Soil_Temperature/HOMTS"
    up = registry.get_reference(UPCH4)
    assert set(up.variables) == {"Wetland_Methane_Emission"}
    assert (up.tim_res, up.data_groupby, up.years) == ("Month", "Single", [2001, 2018])
    item = up.variables["Wetland_Methane_Emission"]
    assert (item.varname, item.varunit) == ("mean_ch4", "mg CH4 m-2 day-1")
    assert not item.compute
    assert "total grid-cell area" in up.description


def test_explicit_scans_and_runtime_selection(tmp_path, monkeypatch):
    profiles = yaml.safe_load(Path(scanner.__file__).with_name("reference_profiles.yaml").read_text())
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", profiles)
    registry = manager.RegistryManager(user_dir=tmp_path / "user")
    root = tmp_path / "Reference"
    expected = {}
    for name, resolution in (("HOMTS_MidRes", "MidRes"), (UPCH4, "LowRes"), ("HSWUD_v8_MidRes", "MidRes")):
        ref = registry.get_reference(name)
        for logical, mapping in ref.variables.items():
            path = root / "Grid" / resolution / mapping.sub_dir / (mapping.prefix + ".nc")
            _fixture(
                path,
                mapping.varname,
                mapping.varunit,
                "D" if name == "HOMTS_MidRes" else "MS",
                ("sd_ch4", "var_ch4") if name == UPCH4 else (),
            )
            expected[name, logical] = path
    # The alternate 0cm copy and the six mixed-area/uncertainty raw products are not references.
    _fixture(
        root / "Grid/MidRes/Meteo/Root_Zone_Soil_Temperature/HOMTS/HOM_TS0cm_daily_025x025_1960_2017.nc",
        "TS",
        "Degrees C",
        "D",
    )
    for index in range(6):
        _fixture(
            root / f"Grid/LowRes/Bio/Wetland_Methane_Emission/UpCH4/raw_{index}.nc",
            "mean_ch4",
            "gC m-2 day-1",
            extra=("sd_ch4", "var_ch4"),
        )
    variants = [variant for group in scanner.scan_reference_directory(root) for variant in group.variants.values()]
    assert {v.registry_name for v in variants} == {"HOMTS_MidRes", UPCH4, "HSWUD_v8_MidRes"}
    catalog = tmp_path / "catalog.yaml"
    scanner.register_scanned_datasets_batch(variants, catalog_path=catalog)
    result = yaml.safe_load(catalog.read_text())
    for name, raw in result.items():
        ref = manager._build_reference(raw)
        assert set(ref.variables) == set(registry.get_reference(name).variables)
        for logical, mapping in ref.variables.items():
            path = expected[name, logical]
            found, computed = select_data_files(
                str(path.parent), mapping.prefix, mapping.suffix or "", None, candidate_varnames=[mapping.varname]
            )
            assert [Path(p) for p in found] == [path]
            assert not computed


def test_gui_selection_and_no_unverified_default_model_aliases():
    source = Path(scanner.__file__).parents[2] / "gui/pages/page_evaluation.py"
    tree = ast.parse(source.read_text())
    items = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EVALUATION_ITEMS" for t in n.targets)
    )
    assert set(WITHDRAWALS) <= set(items["Human Water Withdrawal"])
    assert POINTS <= set(items["Energy Cycle"])
    assert "Ground_Frost_Frequency" in items["Atmospheric"]
    assert "Wetland_Methane_Emission" in items["Carbon Cycle"]
    assert "Methane" in items["Carbon Cycle"]
    models = yaml.safe_load(Path(scanner.__file__).with_name("model_catalog.yaml").read_text())
    for name, model in models.items():
        variables = model.get("variables", {})
        assert not (set(WITHDRAWALS) | {"Ground_Frost_Frequency", "Wetland_Methane_Emission"}) & set(variables)
        if name == "CoLM2024":
            assert POINTS <= set(variables)
            for logical in POINTS:
                assert variables[logical]["varname"] == logical
                assert "ds['f_t_soisno']" in variables[logical]["compute"]
        else:
            assert not POINTS & set(variables)


def test_withdrawal_period_units_and_no_irrigation_alias(tmp_path):
    ref = manager.RegistryManager(user_dir=tmp_path).get_reference("HSWUD_v8_MidRes")
    assert (ref.tim_res, ref.data_groupby, ref.years) == ("Month", "Single", [1965, 2022])
    assert set(ref.variables) == set(WITHDRAWALS)
    assert ref.grid_res is None  # Native latitude/longitude spacings are unequal.
    assert "unequal spacings" in ref.description
    for logical, native in WITHDRAWALS.items():
        mapping = ref.variables[logical]
        assert (mapping.varname, mapping.varunit) == (native, "mm day-1")
        assert mapping.prefix == f"HSWUD_v8_{native}_mm_day"
        assert mapping.sub_dir == "Anth/Water_Withdrawal/HSWUD_v8"
        assert not mapping.compute
    with pytest.raises(KeyError):
        _ = ref.variables["Total_Irrigation_Amount"]


def test_upch4_methane_mass_conversion():
    from openbench.data.unit import UnitProcessing

    value, units = UnitProcessing.convert_unit(16.043, "mg CH4 m-2 day-1")
    assert value == pytest.approx(0.012011)
    assert units == "gc m-2 day-1"


def test_cru_frost_scan_monthly_count(tmp_path, monkeypatch):
    profiles = yaml.safe_load(Path(scanner.__file__).with_name("reference_profiles.yaml").read_text())
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", profiles)
    root = tmp_path / "Reference"
    directory = root / "Grid/LowRes/Meteo/Ground_Frost_Frequency/CRU_TS_4.08"
    _fixture(directory / "frs.nc", "frs", "days", extra=("stn",))
    groups = scanner.scan_reference_directory(root)
    assert len(groups) == 1
    variant = groups[0].variants["LowRes"]
    assert variant.registry_name == "CRU_TS_4.08_LowRes"
    assert set(variant.variables) == {"Ground_Frost_Frequency"}
    target = tmp_path / "cru.yaml"
    scanner.register_scanned_datasets_batch([variant], catalog_path=target)
    ref = manager._build_reference(yaml.safe_load(target.read_text())[variant.registry_name])
    mapping = ref.variables["Ground_Frost_Frequency"]
    assert (mapping.varname, mapping.varunit) == ("frs", "days")
    assert reference_years(ref, mapping) == [1901, 2023]
    found, computed = select_data_files(
        str(directory), mapping.prefix, mapping.suffix or "", None, candidate_varnames=["frs"]
    )
    assert [Path(p) for p in found] == [directory / "frs.nc"]
    assert not computed


def test_no_cross_definition_alias_or_default_model_mapping(tmp_path):
    from openbench.util.names import canonical_variable_name

    logicals = POINTS | set(WITHDRAWALS) | {"Ground_Frost_Frequency", "Wetland_Methane_Emission"}
    model = manager.RegistryManager(user_dir=tmp_path).get_model("CoLM2024")
    assert model is not None
    for logical in logicals:
        assert canonical_variable_name(logical) == logical
        if logical in POINTS:
            mapping = model.variables[logical]
            assert mapping.varname == logical and mapping.varunit == "K"
            assert mapping.compute and mapping.fallbacks is None
        else:
            with pytest.raises(KeyError):
                _ = model.variables[logical]


def test_unselected_legacy_sources_do_not_become_references(tmp_path, monkeypatch):
    profiles = yaml.safe_load(Path(scanner.__file__).with_name("reference_profiles.yaml").read_text())
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", profiles)
    root = tmp_path / "Reference"
    _fixture(
        root / "Grid/MidRes/Meteo/Root_Zone_Soil_Temperature/HOMTS/HOM_TS0cm_daily_025x025_1960_2017.nc",
        "TS",
        "Degrees C",
        "D",
    )
    _fixture(root / "Grid/MidRes/Anth/Water_Use/HSWUD/irr.nc", "irr", "10^8 m3")
    _fixture(root / "Grid/LowRes/Bio/Wetland_Methane_Emission/UpCH4/raw.nc", "mean_ch4", "gC m-2 day-1")
    assert scanner.scan_reference_directory(root) == []
