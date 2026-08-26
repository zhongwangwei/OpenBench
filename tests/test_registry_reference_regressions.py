from pathlib import Path

import pytest
import yaml

from openbench.data.registry import manager as registry_manager
from openbench.data.registry.manager import RegistryManager, _auto_resolve_variant
from openbench.data.registry.schema import (
    ReferenceDataset,
    StationMatchingConfig,
    VariableMapping,
)


def _ref(name, root_dir=None, grid_res=0.5):
    return ReferenceDataset(
        name=name,
        description="demo",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Year",
        timezone=0,
        years=[2000, 2001],
        variables={"Runoff": VariableMapping(varname="runoff", varunit="mm")},
        grid_res=grid_res,
        root_dir=root_dir,
    )


def test_save_reference_rejects_case_insensitive_duplicate(monkeypatch, tmp_path: Path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    catalog.write_text(yaml.safe_dump({"DemoRef": _ref("DemoRef").to_dict()}), encoding="utf-8")
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)

    with pytest.raises(ValueError, match="conflicts with existing catalog entry"):
        RegistryManager(user_dir=tmp_path).save_reference("demoref", _ref("demoref"))


def test_user_reference_catalog_malformed_fails_closed(tmp_path: Path):
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "reference_catalog.yaml").write_text("bad: [", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to read user reference catalog"):
        RegistryManager(user_dir=tmp_path)


def test_auto_resolve_does_not_switch_variants_by_catalog_root_dir(tmp_path: Path):
    missing_preferred = _ref("Demo_LowRes", root_dir=str(tmp_path / "missing"), grid_res=0.5)
    existing_worse = _ref("Demo_MidRes", root_dir=str(tmp_path), grid_res=1.0)

    picked, reason = _auto_resolve_variant(
        {"LowRes": missing_preferred, "MidRes": existing_worse},
        sim_tim_res="Month",
        sim_grid_res=0.5,
    )

    assert picked is missing_preferred
    assert "switched" not in reason


def test_matching_nc_files_honors_uppercase_explicit_glob(tmp_path: Path):
    from openbench.data.registry.scanner import _matching_nc_files

    upper = tmp_path / "CASE.NC4"
    upper.write_text("placeholder", encoding="utf-8")

    assert _matching_nc_files(tmp_path, "*.nc4") == [upper]


def test_registry_page_reference_edit_preserves_hidden_descriptor_fields():
    from openbench.gui.pages.page_registry import _merge_reference_editor_dataset

    existing = ReferenceDataset(
        name="Demo",
        description="old",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Year",
        timezone=0,
        years=[1999, 2000],
        variables={
            "Runoff": VariableMapping(
                varname="old_q",
                varunit="mm",
                fulllist="stations.csv",
                max_uparea=1000.0,
                min_uparea=10.0,
                compute="a + b",
                prefix_fallback=["alt_"],
            )
        },
        fulllist="dataset.csv",
        station_matching=StationMatchingConfig(dataset_file="stations.nc"),
        _provenance={"tim_res": "scan"},
    )
    edited = ReferenceDataset(
        name="Demo",
        description="new",
        category="Water",
        data_type="grid",
        tim_res="Day",
        data_groupby="Month",
        timezone=8,
        years=[],
        variables={"runoff": VariableMapping(varname="new_q", varunit="kg", prefix="p")},
        grid_res=0.25,
        root_dir="/new/root",
    )

    merged = _merge_reference_editor_dataset(existing, edited)

    assert merged.description == "new"
    assert merged.years == [1999, 2000]
    assert merged.fulllist == "dataset.csv"
    assert merged.station_matching is existing.station_matching
    assert merged._provenance == {"tim_res": "scan"}
    var = merged.variables["runoff"]
    assert var.varname == "new_q"
    assert var.fulllist == "stations.csv"
    assert var.max_uparea == 1000.0
    assert var.compute == "a + b"
    assert var.prefix_fallback == ["alt_"]


def test_user_reference_catalog_invalid_entry_fails_closed(tmp_path: Path):
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "reference_catalog.yaml").write_text(
        yaml.safe_dump({"Broken": {"name": "Broken", "tim_res": "Month"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Failed to merge user reference"):
        RegistryManager(user_dir=tmp_path)


def test_save_reference_rejects_case_variant_of_bundled_reference(monkeypatch, tmp_path: Path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)

    bundled_name = "CLARA_3_LowRes"
    bundled = RegistryManager(user_dir=tmp_path).get_reference(bundled_name)
    assert bundled is not None

    with pytest.raises(ValueError, match=bundled_name):
        RegistryManager(user_dir=tmp_path).save_reference(bundled_name.lower(), bundled)


def test_save_reference_allows_exact_case_bundled_overlay(monkeypatch, tmp_path: Path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)

    bundled_name = "CLARA_3_LowRes"
    bundled = RegistryManager(user_dir=tmp_path).get_reference(bundled_name)
    assert bundled is not None

    RegistryManager(user_dir=tmp_path).save_reference(bundled_name, bundled)

    assert bundled_name in yaml.safe_load(catalog.read_text(encoding="utf-8"))


def test_registry_page_refresh_surfaces_registry_load_failure(monkeypatch):
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    messages = []

    class FakeList:
        def clear(self):
            messages.append("cleared")

        def addItem(self, _item):  # pragma: no cover - must not continue to populate
            raise AssertionError("should not populate after registry load failure")

    class BrokenRegistry:
        def list_references(self):
            raise RuntimeError("bad catalog")

    monkeypatch.setattr(page_registry, "_get_registry", lambda: BrokenRegistry())
    monkeypatch.setattr(page_registry.QMessageBox, "critical", lambda *args: messages.append(args[2]))

    page = PageRegistry.__new__(PageRegistry)
    page.dataset_list = FakeList()

    PageRegistry._refresh_dataset_list(page)

    assert "cleared" in messages
    assert any("bad catalog" in str(message) for message in messages)


def test_matching_nc_files_preserves_path_glob_semantics_case_insensitive(tmp_path: Path):
    from openbench.data.registry.scanner import _matching_nc_files

    nested = tmp_path / "sub"
    nested.mkdir()
    direct = tmp_path / "ROOT.NC4"
    child = nested / "CASE.NC4"
    direct.write_text("placeholder", encoding="utf-8")
    child.write_text("placeholder", encoding="utf-8")

    lower = tmp_path / "lower.nc4"
    lower.write_text("placeholder", encoding="utf-8")

    assert _matching_nc_files(tmp_path, "*.nc4") == [direct, lower]
    assert _matching_nc_files(tmp_path, "*.NC4") == [direct, lower]
    assert _matching_nc_files(tmp_path, "sub/*.nc4") == [child]
    assert _matching_nc_files(tmp_path, "**/*.nc4") == [direct, lower, child]

def test_user_reference_mapping_entry_uses_key_as_missing_name(tmp_path: Path):
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "reference_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "Daily": {
                    "description": "daily source",
                    "category": "Water",
                    "data_type": "grid",
                    "tim_res": "Day",
                    "data_groupby": "Year",
                    "timezone": 0,
                    "variables": {"Runoff": {"varname": "q", "varunit": "mm"}},
                }
            }
        ),
        encoding="utf-8",
    )

    ref = RegistryManager(user_dir=tmp_path).get_reference("Daily")

    assert ref is not None
    assert ref.name == "Daily"


def test_user_reference_dir_mapping_entry_uses_key_as_missing_name(tmp_path: Path):
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "custom.yaml").write_text(
        yaml.safe_dump(
            {
                "MSWEP_MidRes": {
                    "description": "mapped source",
                    "category": "Water",
                    "data_type": "grid",
                    "tim_res": "Month",
                    "data_groupby": "Year",
                    "timezone": 0,
                    "variables": {"Precipitation": {"varname": "pr", "varunit": "mm"}},
                }
            }
        ),
        encoding="utf-8",
    )

    ref = RegistryManager(user_dir=tmp_path).get_reference("MSWEP_MidRes")

    assert ref is not None
    assert ref.name == "MSWEP_MidRes"
