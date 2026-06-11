from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from openbench.gui.main_window import MainWindow  # noqa: E402
from openbench.gui.pages.page_ref_data import PageRefData  # noqa: E402


class _FakeController:
    def __init__(self, config):
        self.config = config
        self.updated = []

    def update_section(self, name, value):
        self.config[name] = value
        self.updated.append((name, value))


def test_reference_resolution_picker_allows_lower_frequency_variants(monkeypatch):
    """Users must be able to choose LowRes/Monthly even when MidRes/Daily exists."""

    refs = {
        "ERA5LAND_LowRes": SimpleNamespace(
            name="ERA5LAND_LowRes",
            data_type="grid",
            tim_res="Month",
            grid_res=0.25,
            variables={},
        ),
        "ERA5LAND_MidRes": SimpleNamespace(
            name="ERA5LAND_MidRes",
            data_type="grid",
            tim_res="Day",
            grid_res=0.25,
            variables={},
        ),
    }

    class FakeRegistry:
        def get_reference(self, name):
            return refs.get(name)

    captured = {}

    class FakeResolutionPickerDialog:
        def __init__(self, base_name, variants, compatible=None, parent=None):
            captured["base_name"] = base_name
            captured["variants"] = variants
            captured["compatible"] = compatible

        def exec(self):
            return True

        def selected_resolution(self):
            return "LowRes"

    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: FakeRegistry())
    monkeypatch.setattr(
        "openbench.gui.dialogs.data_discovery.ResolutionPickerDialog",
        FakeResolutionPickerDialog,
    )

    page = PageRefData.__new__(PageRefData)

    selected = page._pick_resolution(
        {"group": "ERA5LAND", "variants": ["ERA5LAND_LowRes", "ERA5LAND_MidRes"]},
        "Surface_Air_Temperature",
    )

    assert selected == "ERA5LAND_LowRes"
    assert captured["base_name"] == "ERA5LAND"
    assert set(captured["variants"]) == {"LowRes", "MidRes"}
    assert captured["compatible"] is None


def test_available_simulation_variables_do_not_expand_user_evaluation_selection():
    controller = _FakeController(
        {
            "evaluation_items": {
                "Surface_Air_Temperature": True,
                "Precipitation": True,
            }
        }
    )
    window = MainWindow.__new__(MainWindow)
    window.controller = controller

    window._on_available_variables_changed(
        [
            "Surface_Air_Temperature",
            "Precipitation",
            "Latent_Heat",
            "Sensible_Heat",
        ]
    )

    assert window._available_variables == [
        "Surface_Air_Temperature",
        "Precipitation",
        "Latent_Heat",
        "Sensible_Heat",
    ]
    assert controller.config["evaluation_items"] == {
        "Surface_Air_Temperature": True,
        "Precipitation": True,
    }
    assert controller.updated == []


def test_sim_data_does_not_create_variable_mappings_when_evaluation_selection_is_empty():
    from openbench.gui.pages import page_sim_data
    from tests.gui_fakes import FakeLineEdit as FakeText

    class FakeCombo:
        def __init__(self, value):
            self.value = value

        def currentText(self):
            return self.value

    controller = _FakeController({"evaluation_items": {}, "sim_data": {"general": {}}})
    page = SimpleNamespace(
        controller=controller,
        get_selected_cases=lambda: [{"label": "CaseA", "model": "CoLM2024", "nc_dir": "/sim", "prefix": "hist_"}],
        _prefix_input=FakeText(""),
        _data_type_combo=FakeCombo("grid"),
        _grid_res_input=FakeText("0.5"),
        _tim_res_combo=FakeCombo("Month"),
        _data_groupby_combo=FakeCombo("Month"),
        _suffix_input=FakeText(".nc"),
        _root_input=FakeText("/sim"),
        _get_available_variables=lambda: {"Runoff", "Latent_Heat"},
    )

    page_sim_data.PageSimData.save_to_config(page)

    assert controller.config["sim_data"]["general"] == {}
