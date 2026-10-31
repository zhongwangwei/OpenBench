"""Imported station products keep variable-specific files and source IDs."""

import ast
import os
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from openbench.config.adapter import build_legacy_namelists, reference_data_dir
from openbench.config.schema import EvaluationConfig, OpenBenchConfig, ProjectConfig, ReferenceConfig
from openbench.data.processing import StationDatasetProcessing
from openbench.data.registry import manager, scanner
from openbench.util.station_ids import read_station_csv

SOURCE = "OBRefGen_Filled_LE_daily"
VARIABLES = ("Ground_Heat", "Latent_Heat", "Net_Radiation")


def _fixture(tmp_path, monkeypatch):
    root = tmp_path / "Reference"
    monkeypatch.setenv("OPENBENCH_REF_ROOT", str(root))
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path / "home"))
    registry = manager.RegistryManager(user_dir=tmp_path / "registry")
    monkeypatch.setattr(manager, "get_registry", lambda: registry)
    reference = registry.get_reference(SOURCE)
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="station_import", output_dir=str(tmp_path / "output"), years=[2001, 2001]),
        evaluation=EvaluationConfig(variables=list(VARIABLES)),
        reference=ReferenceConfig(sources={name: SOURCE for name in VARIABLES}),
        simulation={},
    )
    expected = {}
    for index, name in enumerate(VARIABLES, start=1):
        mapping = reference.variables[name]
        _, directory = reference_data_dir(cfg, reference, mapping, None)
        path = Path(directory) / (mapping.prefix + ".nc")
        path.parent.mkdir(parents=True, exist_ok=True)
        values = [[index * 10.0], [index * 10.0 + 1]]
        xr.Dataset(
            {mapping.varname: (("station", "time"), values, {"units": mapping.varunit})},
            coords={
                "station": [0, 1],
                "source_station_id": ("station", ["0001", "0002"]),
                "lat": ("station", [42.0, 43.0]),
                "lon": ("station", [-72.0, -73.0]),
                "time": pd.date_range("2001-01-01", periods=1),
            },
        ).to_netcdf(path)
        station_list = Path(os.path.expandvars(mapping.fulllist))
        station_list.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ID": ["0001", "0002"],
                "SYEAR": [2001, 2001],
                "EYEAR": [2001, 2001],
                "LON": [-72.0, -73.0],
                "LAT": [42.0, 43.0],
                "DIR": [path.name, path.name],
            }
        ).to_csv(station_list, index=False)
        expected[name] = (path, station_list, values[1][0])
    assert len({row[0] for row in expected.values()}) == len(VARIABLES)
    assert len({row[1] for row in expected.values()}) == len(VARIABLES)
    return root, registry, cfg, expected


def test_imported_variable_lists_reach_runtime_without_crossing_files(tmp_path, monkeypatch):
    _, _, cfg, expected = _fixture(tmp_path, monkeypatch)
    _, references, _ = build_legacy_namelists(cfg)

    for name, (path, station_list, expected_value) in expected.items():
        section = references[name]
        assert Path(section[f"{SOURCE}_fulllist"]) == station_list
        assert section[f"{SOURCE}_prefix"] == path.stem
        rows = read_station_csv(station_list)
        assert rows["ID"].tolist() == ["0001", "0002"]
        assert len(rows) == 2
        selected_path = Path(section[f"{SOURCE}_dir"]) / rows.iloc[1]["DIR"]
        assert selected_path == path
        processor = StationDatasetProcessing.__new__(StationDatasetProcessing)
        processor.ref_varname = section[f"{SOURCE}_varname"]
        with xr.open_dataset(selected_path) as dataset:
            selected = processor._select_merged_station_data(dataset, rows.iloc[1], "ref")
            assert selected[processor.ref_varname].item() == expected_value


def test_profile_rescan_keeps_each_variable_station_list(tmp_path, monkeypatch):
    root, registry, _, expected = _fixture(tmp_path, monkeypatch)
    for path, _, _ in expected.values():
        path.with_name("._" + path.name).write_bytes(b"AppleDouble metadata, not a NetCDF dataset")
    profiles = yaml.safe_load(Path(scanner.__file__).with_name("reference_profiles.yaml").read_text())
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", profiles)
    combined_list_calls = []

    def forbid_combined_list(*args, **kwargs):
        combined_list_calls.append((args, kwargs))
        raise AssertionError("Variable-specific lists must not be replaced by a combined station list")

    monkeypatch.setattr(scanner, "generate_station_list", forbid_combined_list)
    variants = [variant for group in scanner.scan_reference_directory(root) for variant in group.variants.values()]
    assert {variant.registry_name for variant in variants} == {SOURCE}
    assert variants[0].file_count == len(VARIABLES)
    catalog = tmp_path / "rescanned.yaml"
    scanner.register_scanned_datasets_batch(variants, catalog_path=catalog)
    # A second registration must preserve the same variable-specific bindings.
    scanner.register_scanned_datasets_batch(variants, catalog_path=catalog)
    assert not combined_list_calls
    raw = yaml.safe_load(catalog.read_text())[SOURCE]
    assert not raw.get("fulllist")
    reference = manager._build_reference(raw)
    for name, (path, station_list, _) in expected.items():
        mapping = reference.variables[name]
        assert mapping.fulllist == registry.get_reference(SOURCE).variables[name].fulllist
        assert Path(os.path.expandvars(mapping.fulllist)) == station_list
        assert mapping.prefix == path.stem
        assert len(read_station_csv(station_list)) == 2


def test_new_reference_catalog_excludes_unqualified_records_and_runoff(tmp_path):
    registry = manager.RegistryManager(user_dir=tmp_path / "registry")
    references = [reference for reference in registry.list_references() if reference.name.startswith("OBRefGen_")]
    assert references
    forbidden_sources = ("GLEAM_History", "NPP_Rodal2025", "GLAST")
    forbidden_variables = {
        "Streamflow",
        "Runoff",
        "Total_Runoff",
        "Surface_Runoff",
        "Subsurface_Runoff",
        "Dam_Inflow",
        "Dam_Outflow",
    }
    for reference in references:
        assert "/Station/OBRefGen" not in reference.root_dir
        parts = Path(reference.root_dir).parts
        assert parts[parts.index("Station") + 1] in {"Water", "Heat", "Bio", "Lake", "Anth", "Composite", "Meteo"}
        for mapping in reference.variables.values():
            assert "/Station/station_lists/openbench_station_new/" in mapping.fulllist
        assert not any(source in reference.name for source in forbidden_sources)
        assert not forbidden_variables.intersection(reference.variables)


def test_category_scan_ignores_received_record_directories(tmp_path, monkeypatch):
    root, _, _, _ = _fixture(tmp_path, monkeypatch)
    profiles = yaml.safe_load(Path(scanner.__file__).with_name("reference_profiles.yaml").read_text())
    ignored = [
        profile["scan"]["root_sub_dir"] for name, profile in profiles.items() if name.startswith("OBRefGen_Received_")
    ]
    assert len(ignored) == 4
    for relative in ignored:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"et": (("station", "time"), [[1.0]])},
            coords={"station": ["site"], "time": pd.date_range("2001-01-01", periods=1)},
        ).to_netcdf(directory / "received.nc")
    monkeypatch.setattr(scanner, "_REFERENCE_PROFILES", profiles)
    variants = [variant for group in scanner.scan_reference_directory(root) for variant in group.variants.values()]
    assert {variant.registry_name for variant in variants} == {SOURCE}


def test_imported_station_variables_are_available_in_gui_selectors():
    gui = Path(manager.__file__).parents[2] / "gui"
    choices = {}
    for relative, name in (
        ("pages/page_evaluation.py", "EVALUATION_ITEMS"),
        ("widgets/model_definition_editor.py", "EVALUATION_VARIABLES"),
    ):
        tree = ast.parse((gui / relative).read_text())
        choices[name] = ast.literal_eval(
            next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            )
        )

    categories = choices["EVALUATION_ITEMS"]
    atmospheric = {
        "Sea_Level_Pressure",
        "Surface_Air_Temperature",
        "Surface_Dew_Point_Temperature",
        "Surface_Pressure",
        "Surface_Wind_Speed",
        "Surface_Friction_Velocity",
        "Vapor_Pressure_Deficit",
    }
    water_bodies = {"Water_Surface_Elevation", "Lake_Water_Area", "Lake_Water_Level"}
    assert atmospheric <= set(categories["Atmospheric"])
    assert "Water_Table_Depth" in categories["Water Cycle"]
    assert water_bodies <= set(categories["Water Bodies"])
    model_variables = choices["EVALUATION_VARIABLES"]
    assert atmospheric | water_bodies | {"Water_Table_Depth"} <= set(model_variables)
    assert len(model_variables) == len(set(model_variables))
    all_items = [item for items in categories.values() for item in items]
    assert len(all_items) == len(set(all_items))
