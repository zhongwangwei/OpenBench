"""Tests for registration of scanned datasets."""

from pathlib import Path

import yaml

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
