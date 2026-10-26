"""Keep the RS+METEO ensemble median distinct from other FLUXCOM products."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from openbench.data.file_lookup import select_data_files
from openbench.data.registry import manager, scanner
from openbench.data.registry.schema import reference_years

NAME = "FLUXCOM_RS_METEO_Ensemble_LowRes"
FIELDS = {
    "Gross_Primary_Productivity": ("Carbon", "GPP", "FP-ALL", 2010),
    "Ecosystem_Respiration": ("Carbon", "TER", "FP-ALL", 2010),
    "Net_Ecosystem_Exchange": ("Carbon", "NEE", "FP-NONE", 2010),
    "Sensible_Heat": ("Energy", "H", "EBC-ALL", 2013),
    "Latent_Heat": ("Energy", "LE", "EBC-ALL", 2013),
    "Net_Radiation": ("Energy", "Rn", "EBC-NONE", 2013),
}


def _prefix(native, choice):
    return f"{native}.RS_METEO.{choice}.MLM-ALL.METEO-ALL.720_360.monthly."


@pytest.mark.parametrize("logical,fields", FIELDS.items())
def test_bundled_ensemble_binding_is_explicit(tmp_path, logical, fields):
    ref = manager.RegistryManager(user_dir=tmp_path).get_reference(NAME)
    assert ref is not None
    folder, native, choice, end = fields
    mapping = ref.variables[logical]
    assert ref.grid_res == 0.5
    assert ref.tim_res == "Month"
    assert ref.data_groupby == "Year"
    assert ref.years == [2001, 2013]
    assert "median" in ref.description
    assert mapping.varname == native
    assert mapping.varunit == ("gC m-2 d-1" if folder == "Carbon" else "MJ m-2 d-1")
    assert mapping.sub_dir == f"Composite/FLUXCOM/{folder}"
    assert mapping.prefix == _prefix(native, choice)
    assert reference_years(ref, mapping) == [2001, end]


def test_raw_ensemble_scan_register_and_runtime_selection(tmp_path, monkeypatch):
    profiles = yaml.safe_load(Path(scanner.__file__).with_name("reference_profiles.yaml").read_text())
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", profiles)
    root = tmp_path / "Reference"
    expected_files = {}
    for logical, (folder, native, choice, end) in FIELDS.items():
        directory = root / "Grid/LowRes/Composite/FLUXCOM" / folder
        directory.mkdir(parents=True, exist_ok=True)
        for year in (2001, end):
            data = np.ones((12, 2, 2), dtype="float32")
            ds = xr.Dataset(
                {field: (("time", "lat", "lon"), data) for field in (native, native + "_mad", native + "_n")},
                coords={
                    "time": pd.date_range(f"{year}-01-01", periods=12, freq="MS"),
                    "lat": [0, 0.5],
                    "lon": [0, 0.5],
                },
                attrs={"Comment": "Ensemble median"},
            )
            ds[native].attrs["units"] = "gC m-2 d-1" if folder == "Carbon" else "MJ m-2 d-1"
            filename = directory / f"{_prefix(native, choice)}{year}.nc"
            ds.to_netcdf(filename)
            expected_files[logical, year] = filename
            # Same native field from a different forcing must not enter this ensemble.
            ds.to_netcdf(directory / filename.name.replace("METEO-ALL", "METEO-GSWP3"))

    groups = scanner.scan_reference_directory(root)
    assert len(groups) == 1
    variant = groups[0].variants["LowRes"]
    assert variant.registry_name == NAME
    assert variant.file_count == 12
    assert set(variant.variables) == set(FIELDS)
    catalog_path = tmp_path / "reference_catalog.yaml"
    scanner.register_scanned_datasets_batch([variant], catalog_path=catalog_path)
    raw = yaml.safe_load(catalog_path.read_text())[NAME]
    ref = manager._build_reference(raw)
    assert set(ref.variables) == set(FIELDS)
    for logical, (_, native, _, end) in FIELDS.items():
        mapping = ref.variables[logical]
        assert mapping.varname == native
        assert reference_years(ref, mapping) == [2001, end]
        directory = root / "Grid/LowRes" / mapping.sub_dir
        for year in (2001, end):
            selected, used_compute = select_data_files(
                str(directory), mapping.prefix, mapping.suffix or "", year, candidate_varnames=[mapping.varname]
            )
            assert [Path(path) for path in selected] == [expected_files[logical, year]]
            assert not used_compute


def test_ensemble_does_not_replace_existing_fluxcom_products(tmp_path):
    registry = manager.RegistryManager(user_dir=tmp_path)
    legacy = registry.get_reference("FLUXCOM_LowRes")
    assert legacy.variables["Latent_Heat"].sub_dir == "Heat/Latent_Heat/FLUXCOM"
    assert legacy.variables["Latent_Heat"].varname == "le"
    assert registry.get_reference("FLUXCOM-X-BASE_LowRes") is not None
    assert set(registry.get_reference(NAME).variables) == set(FIELDS)
