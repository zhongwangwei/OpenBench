from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("root", [None, "/tmp/review-reference-root"])
def test_save_reference_allows_exact_case_bundled_overlay(monkeypatch, tmp_path: Path, root):
    if root is not None:
        monkeypatch.setenv("OPENBENCH_REF_ROOT", root)
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)

    bundled_name = "CLARA_3_LowRes"
    bundled = RegistryManager(user_dir=tmp_path).get_reference(bundled_name)
    assert bundled is not None

    edited = replace(bundled, description="edited in the GUI")
    RegistryManager(user_dir=tmp_path).save_reference(bundled_name, edited)

    # Only the edited field reaches the overlay; the rest keeps following bundled.
    assert yaml.safe_load(catalog.read_text(encoding="utf-8")) == {bundled_name: {"description": "edited in the GUI"}}


def test_reference_save_preserves_explicit_fixed_path(monkeypatch, tmp_path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    name = "CLARA_3_LowRes"
    pinned = "/tmp/pinned/Grid/LowRes"
    catalog.write_text(yaml.safe_dump({name: {"root_dir": pinned}}))
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)
    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/tmp/pinned")
    manager = RegistryManager(user_dir=tmp_path)
    manager.save_reference(name, replace(manager.get_reference(name), description="edited"))
    assert yaml.safe_load(catalog.read_text())[name]["root_dir"] == pinned
    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/tmp/moved")
    assert RegistryManager(user_dir=tmp_path).get_reference(name).root_dir == pinned


def test_model_save_snapshot_preserves_clear_and_delete_on_reload(monkeypatch, tmp_path):
    catalog = tmp_path / "models" / "model_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_model_catalog_path", lambda: catalog)
    manager = RegistryManager(user_dir=tmp_path)
    original = manager.get_model("CoLM2024")
    variables = {key: value for key, value in original.variables.items() if key != "Snow_Depth"}
    variables["Suspended_Sediment_Concentration"] = VariableMapping("native_ssc", "mg L-1")
    edited = replace(original, time_offset=None, grid_res=None, variables=variables)
    manager.save_model("CoLM2024", edited)
    reloaded = RegistryManager(user_dir=tmp_path).get_model("CoLM2024")
    assert reloaded == edited
    assert reloaded.variables["Suspended_Sediment_Concentration"].compute is None
    assert "Snow_Depth" not in reloaded.variables
    assert reloaded.time_offset is None


def test_model_editor_preserves_hidden_fields_and_explicit_clears():
    from openbench.data.registry.schema import ModelProfile
    from openbench.gui.pages.page_registry import _merge_model_editor_profile

    original = ModelProfile(
        "Demo",
        "old",
        tim_res="Day",
        time_offset=None,
        variables={"Soil_Evaporation": VariableMapping("old", "mm", prefix="p_", sub_dir="land", compute="a+b")},
    )
    edited = ModelProfile("Demo", "new", variables={"Bare_Soil_Evaporation": VariableMapping("native", "mm")})
    merged = _merge_model_editor_profile(original, edited)
    assert merged.tim_res == "Day"
    assert merged.time_offset is None
    mapping = merged.variables["Bare_Soil_Evaporation"]
    assert mapping.prefix == "p_"
    assert mapping.sub_dir == "land"
    assert mapping.compute is None


def test_model_variable_null_is_a_deletion_tombstone():
    original = RegistryManager().get_model("CLM5")
    merged = registry_manager._deep_merge_model(original, {"variables": {"Snow_Depth": None}})
    assert "Snow_Depth" not in merged.variables


def test_reference_save_keeps_explicit_null_path(monkeypatch, tmp_path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    catalog.write_text(yaml.safe_dump({"CN05.1_MidRes": {"root_dir": None}}))
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)
    manager = RegistryManager(user_dir=tmp_path)
    ref = manager.get_reference("CN05.1_MidRes")
    assert ref.root_dir is None
    manager.save_reference(ref.name, replace(ref, description="edited"))
    assert yaml.safe_load(catalog.read_text())[ref.name]["root_dir"] is None
    assert RegistryManager(user_dir=tmp_path).get_reference(ref.name).root_dir is None


def test_reference_save_can_clear_inherited_path(monkeypatch, tmp_path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)
    manager = RegistryManager(user_dir=tmp_path)
    ref = manager.get_reference("CN05.1_MidRes")
    manager.save_reference(ref.name, replace(ref, root_dir=None))
    assert RegistryManager(user_dir=tmp_path).get_reference(ref.name).root_dir is None


def test_model_save_preserves_partial_complete_offsets(monkeypatch, tmp_path):
    catalog = tmp_path / "models" / "model_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_model_catalog_path", lambda: catalog)
    manager = RegistryManager(user_dir=tmp_path)
    edited = replace(manager.get_model("BCC_AVIM"), time_offset={"Day": "0"})
    manager.save_model(edited.name, edited)
    assert RegistryManager(user_dir=tmp_path).get_model(edited.name) == edited


def test_model_snapshot_removes_nested_offset_fields():
    from openbench.data.registry.schema import ModelProfile

    bundled = ModelProfile("Demo", "", time_offset={"Day": {"default": "-1 days", "Runoff": "-2 days"}})
    edited = replace(bundled, time_offset={"Day": {"default": "0"}})
    overlay = registry_manager.model_snapshot_overlay(edited, bundled)
    assert registry_manager._deep_merge_model(bundled, overlay) == edited


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


def _registry_list(names):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidget, QListWidgetItem

    widget = QListWidget()
    for name in names:
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        widget.addItem(item)
    return widget


def test_new_model_does_not_inherit_the_previously_selected_profile(qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QLineEdit, QTableWidget, QTableWidgetItem

    from openbench.data.registry.schema import ModelProfile
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    class FakeRegistry:
        def __init__(self):
            self.saved = []

        def get_model(self, name):
            if name == "CLM5":
                return ModelProfile(name="CLM5", description="bundled", time_offset={"Day": "-1 days"})
            return None

        def save_model(self, name, profile):
            self.saved.append((name, profile))

    class Silent:
        information = warning = critical = staticmethod(lambda *args, **kwargs: None)

    registry = FakeRegistry()
    monkeypatch.setattr(page_registry, "QMessageBox", Silent)
    page = PageRegistry.__new__(PageRegistry)
    page.model_list = _registry_list(["CLM5", "VIC5"])
    page.model_name, page.model_desc, page.model_grid_res = QLineEdit(), QLineEdit(), QLineEdit()
    page.model_data_type = QComboBox()
    page.model_data_type.addItems(["grid", "stn"])
    page.model_var_table = QTableWidget(0, 4)
    page._registry = lambda: registry
    page._clear_registry_cache = lambda remote=False: None
    page._refresh_model_list = lambda: None

    page.model_list.setCurrentRow(0)  # the user looked at CLM5 first
    page._new_model()
    assert page.model_list.currentItem() is None
    page.model_name.setText("BrandNew")
    page.model_var_table.insertRow(0)
    for column, text in enumerate(["Latent_Heat", "lh", "W m-2", ""]):
        page.model_var_table.setItem(0, column, QTableWidgetItem(text))
    page._model_save()

    [(name, profile)] = registry.saved
    assert name == "BrandNew"
    assert profile.time_offset is None
    assert profile.description == ""


def test_new_dataset_forgets_the_previously_selected_reference(qapp):
    from PySide6.QtWidgets import QComboBox, QLineEdit, QTableWidget

    from openbench.gui.pages.page_registry import PageRegistry

    page = PageRegistry.__new__(PageRegistry)
    page.dataset_list = _registry_list(["FLUXNET_PLUMBER2", "CN05.1_MidRes"])
    for field in ("ds_name", "ds_desc", "ds_grid_res", "ds_root_dir", "ds_timezone"):
        setattr(page, field, QLineEdit())
    for field in ("ds_category", "ds_data_type", "ds_tim_res", "ds_data_groupby"):
        setattr(page, field, QComboBox())
    page.ds_var_table = QTableWidget(0, 4)

    page.dataset_list.setCurrentRow(0)
    page._new_dataset()

    assert page.dataset_list.currentItem() is None
    assert page.dataset_list.selectedItems() == []


@pytest.mark.parametrize("answer", ["no", "yes"])
def test_new_model_with_an_existing_name_asks_before_replacing(qapp, monkeypatch, answer):
    from PySide6.QtWidgets import QComboBox, QLineEdit, QTableWidget, QTableWidgetItem

    from openbench.data.registry.schema import ModelProfile
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    saved, asked = [], []

    class FakeRegistry:
        def get_model(self, name):
            return ModelProfile(name="CLM5", description="bundled") if name == "CLM5" else None

        def save_model(self, name, profile):
            saved.append(name)

    class Dialogs:
        Yes, No = 1, 2
        information = warning = critical = staticmethod(lambda *args, **kwargs: None)

        @staticmethod
        def question(*args, **kwargs):
            asked.append(args[2])
            return Dialogs.Yes if answer == "yes" else Dialogs.No

    monkeypatch.setattr(page_registry, "QMessageBox", Dialogs)
    page = PageRegistry.__new__(PageRegistry)
    page.model_list = _registry_list(["CLM5"])
    page.model_name, page.model_desc, page.model_grid_res = QLineEdit(), QLineEdit(), QLineEdit()
    page.model_data_type = QComboBox()
    page.model_data_type.addItems(["grid", "stn"])
    page.model_var_table = QTableWidget(0, 4)
    page._registry = lambda: FakeRegistry()
    page._clear_registry_cache = lambda remote=False: None
    page._refresh_model_list = lambda: None

    page._new_model()
    page.model_name.setText("CLM5")
    page.model_var_table.insertRow(0)
    for column, text in enumerate(["Latent_Heat", "lh", "W m-2", ""]):
        page.model_var_table.setItem(0, column, QTableWidgetItem(text))
    page._model_save()

    assert asked and "already exists" in asked[0]
    assert saved == ([] if answer == "no" else ["CLM5"])


def test_dataset_save_merges_onto_the_resolution_variant_that_was_loaded(qapp, monkeypatch):
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    page = PageRegistry.__new__(PageRegistry)
    page.dataset_list = _registry_list(["GRFR_HigRes"])
    page.dataset_list.item(0).setData(page_registry.Qt.UserRole + 1, "GRFR")
    page._ds_group_map = {"GRFR": [("GRFR_HigRes", "HigRes", None), ("GRFR_LowRes", "LowRes", None)]}
    loaded = []

    class Registry:
        def get_reference(self, name):
            return SimpleNamespace(name=name)

    page._registry = lambda: Registry()
    page._populate_dataset_editor = lambda ref: loaded.append(ref.name)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QInputDialog.getItem", staticmethod(lambda *args, **kwargs: ("GRFR_LowRes", True))
    )

    page._on_dataset_selected(0)

    assert loaded == ["GRFR_LowRes"]
    assert page._editing_dataset_name == "GRFR_LowRes"


def test_cn05_units_are_declared():
    from openbench.data.unit import UnitProcessing

    variables = RegistryManager().get_reference("CN05.1_MidRes").variables
    # Celsius converts to K like the simulations; mm/day is already the base unit.
    assert UnitProcessing.base_unit(variables["Surface_Air_Temperature"].varunit, "Surface_Air_Temperature") == "k"
    assert UnitProcessing.base_unit(variables["Precipitation"].varunit, "Precipitation") == "mm day-1"


def test_overlay_keeping_the_old_empty_cn05_unit_is_flagged(tmp_path):
    from openbench.data.registry import overlay_audit as oa

    overlay = tmp_path / "references" / "reference_catalog.yaml"
    overlay.parent.mkdir()
    overlay.write_text(yaml.safe_dump({"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"varunit": ""}}}}))

    hints = oa.overlay_hints(tmp_path)

    assert any(hint.startswith("CN05.1_MidRes:") and "older bundled defaults" in hint for hint in hints)


def test_reference_save_removes_bundled_variables_left_out(monkeypatch, tmp_path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)
    manager = RegistryManager(user_dir=tmp_path)
    ref = manager.get_reference("CN05.1_MidRes")

    precipitation_only = {"Precipitation": ref.variables["Precipitation"]}
    manager.save_reference(ref.name, replace(ref, variables=precipitation_only))

    assert yaml.safe_load(catalog.read_text())[ref.name] == {"variables": {"Surface_Air_Temperature": None}}
    assert sorted(RegistryManager(user_dir=tmp_path).get_reference(ref.name).variables) == ["Precipitation"]

    # Saving it back with the variable restores the bundled entry.
    manager = RegistryManager(user_dir=tmp_path)
    manager.save_reference(ref.name, ref)
    assert ref.name not in (yaml.safe_load(catalog.read_text()) or {})
    reloaded = RegistryManager(user_dir=tmp_path).get_reference(ref.name)
    assert reloaded.variables["Surface_Air_Temperature"] == ref.variables["Surface_Air_Temperature"]
