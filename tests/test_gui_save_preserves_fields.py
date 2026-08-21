from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from openbench.gui.pages.page_general import PageGeneral  # noqa: E402
from openbench.gui.pages.page_ref_data import PageRefData  # noqa: E402
from openbench.gui.pages.page_sim_data import PageSimData  # noqa: E402
from tests.gui_fakes import FakeLineEdit as FakeText  # noqa: E402


class FakeCombo:
    def __init__(self, value, data=None):
        self.value = value
        self.data = data

    def currentText(self):
        return self.value

    def currentData(self):
        return self.data


class FakeLoadCombo:
    def __init__(self, items):
        self.items = list(items)
        self.index = 0

    def blockSignals(self, _blocked):
        pass

    def count(self):
        return len(self.items)

    def itemData(self, index):
        return self.items[index][1]

    def setCurrentIndex(self, index):
        self.index = index

    def addItem(self, label, data):
        self.items.append((label, data))


class FakeSpin:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeCheck:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class FakePath:
    def __init__(self, path):
        self._path = path

    def path(self):
        return self._path


class FakeController:
    def __init__(self, config):
        self.config = config
        self.updated = []
        self.synced = False

    def update_section(self, section, data):
        self.config[section] = data
        self.updated.append((section, data))

    def sync_namelists(self):
        self.synced = True


def test_sim_data_save_preserves_unknown_top_level_metadata():
    controller = FakeController(
        {
            "evaluation_items": {"Runoff": True},
            "sim_data": {
                "_manual_note": "keep me",
                "_schema_version": 2,
                "general": {"Runoff_sim_source": ["OldCase"]},
            },
        }
    )
    page = PageSimData.__new__(PageSimData)
    page.controller = controller
    page.get_selected_cases = lambda: [
        {"label": "CaseA", "model": "CoLM2024", "nc_dir": "/sim/CaseA", "prefix": "case_a_"}
    ]
    page._prefix_input = FakeText("")
    page._data_type_combo = FakeCombo("grid")
    page._grid_res_input = FakeText("0.5")
    page._tim_res_combo = FakeCombo("Month")
    page._data_groupby_combo = FakeCombo("Year")
    page._suffix_input = FakeText(".nc")
    page._root_input = FakeText("/sim")

    page.save_to_config()

    assert controller.config["sim_data"]["_manual_note"] == "keep me"
    assert controller.config["sim_data"]["_schema_version"] == 2
    assert controller.config["sim_data"]["general"] == {"Runoff_sim_source": ["CaseA"]}


def test_sim_data_save_removes_cleared_variable_pattern_overrides():
    controller = FakeController(
        {
            "evaluation_items": {"Albedo": True},
            "sim_data": {
                "general": {"Albedo_sim_source": ["TE"]},
                "source_configs": {
                    "TE": {
                        "general": {"model_namelist": "TE", "root_dir": "/sim/TE"},
                        "variables": {"Albedo": {"prefix": "old_"}},
                    }
                },
            },
        }
    )
    page = PageSimData.__new__(PageSimData)
    page.controller = controller
    page.get_selected_cases = lambda: [
        {"label": "TE", "model": "TE", "nc_dir": "/sim/TE", "prefix": "", "suffix": "", "variables": {}}
    ]
    page._prefix_input = FakeText("")
    page._data_type_combo = FakeCombo("grid")
    page._grid_res_input = FakeText("0.5")
    page._tim_res_combo = FakeCombo("Month")
    page._data_groupby_combo = FakeCombo("month")
    page._suffix_input = FakeText("")
    page._root_input = FakeText("/sim")

    page.save_to_config()

    assert "variables" not in controller.config["sim_data"]["source_configs"]["TE"]


def test_sim_data_save_uses_detected_case_metadata_instead_of_auto_override_labels():
    controller = FakeController({"evaluation_items": {"Latent_Heat": True}, "sim_data": {}})
    page = PageSimData.__new__(PageSimData)
    page.controller = controller
    page.get_selected_cases = lambda: [
        {
            "label": "StationCase",
            "model": "CoLM2024",
            "nc_dir": "/sim/StationCase",
            "prefix": "",
            "suffix": "",
            "variables": {},
            "data_type": "stn",
            "grid_res": None,
            "tim_res": "Day",
            "data_groupby": "Single",
            "fulllist": "/output/stations.csv",
        }
    ]
    page._prefix_input = FakeText("")
    page._data_type_combo = FakeCombo("Auto (per case)", "")
    page._grid_res_input = FakeText("")
    page._tim_res_combo = FakeCombo("Auto (per case)", "")
    page._data_groupby_combo = FakeCombo("Auto (per case)", "")
    page._suffix_input = FakeText("")
    page._root_input = FakeText("/sim")

    page.save_to_config()

    general = controller.config["sim_data"]["source_configs"]["StationCase"]["general"]
    assert general["data_type"] == "stn"
    assert general["tim_res"] == "Day"
    assert general["data_groupby"] == "Single"
    assert general["fulllist"] == "/output/stations.csv"


def test_sim_data_save_keeps_unchecked_scan_rows_out_of_runtime_sources():
    controller = FakeController({"evaluation_items": {"Runoff": True}, "sim_data": {}})
    page = PageSimData.__new__(PageSimData)
    page.controller = controller
    page.get_selected_cases = lambda: [
        {"label": "CaseA", "model": "CLM5", "nc_dir": "/sim/CaseA", "prefix": "hist_", "variables": {}}
    ]
    page._cases = [
        {
            "label": "CaseA",
            "nc_dir": "/sim/CaseA",
            "prefix_input": FakeText("hist_"),
            "suffix_input": FakeText(""),
            "checkbox": FakeCheck(True),
            "model_combo": FakeCombo("CLM5", "CLM5"),
            "variable_overrides": {},
        },
        {
            "label": "CaseB",
            "nc_dir": "/sim/CaseB",
            "prefix_input": FakeText("case_b_"),
            "suffix_input": FakeText(".nc"),
            "checkbox": FakeCheck(False),
            "model_combo": FakeCombo("CoLM2024", "CoLM2024"),
            "variable_overrides": {"Runoff": {"varname": "q"}},
            "multi_stream": True,
        },
    ]
    page._prefix_input = FakeText("")
    page._data_type_combo = FakeCombo("grid")
    page._grid_res_input = FakeText("0.5")
    page._tim_res_combo = FakeCombo("Month")
    page._data_groupby_combo = FakeCombo("Year")
    page._suffix_input = FakeText("")
    page._root_input = FakeText("/sim")

    page.save_to_config()

    sim_data = controller.config["sim_data"]
    assert list(sim_data["source_configs"]) == ["CaseA"]
    assert sim_data["general"]["Runoff_sim_source"] == ["CaseA"]
    assert [(case["label"], case["checked"]) for case in sim_data["_scanned_cases"]] == [
        ("CaseA", True),
        ("CaseB", False),
    ]


def test_sim_data_load_restores_unchecked_scan_rows():
    sim_data = {
        "_scan_root": "/sim",
        "_scanned_cases": [
            {"label": "CaseA", "nc_dir": "/sim/CaseA", "prefix": "hist_", "checked": True, "model": "CLM5"},
            {
                "label": "CaseB",
                "nc_dir": "/sim/CaseB",
                "prefix": "case_b_",
                "suffix": ".nc",
                "checked": False,
                "model": "CoLM2024",
                "variables": {"Runoff": {"varname": "q"}},
                "multi_stream": True,
            },
        ],
    }
    page = PageSimData.__new__(PageSimData)
    page.controller = FakeController({"sim_data": sim_data})
    page._root_input = FakeText("")
    page._cases = []
    page._clear_cases = lambda: page._cases.clear()
    restored = []

    def add_case(label, nc_dir, prefix, **kwargs):
        restored.append((label, nc_dir, prefix, kwargs))
        page._cases.append({})

    page._add_case_row = add_case
    page._settings_group = type("Settings", (), {"setVisible": lambda self, visible: None})()

    page.load_from_config()

    assert [case[0] for case in restored] == ["CaseA", "CaseB"]
    assert restored[1][3]["checked"] is False
    assert restored[1][3]["variable_overrides"] == {"Runoff": {"varname": "q"}}
    assert page._cases[1]["case_pattern_edited"] is False


def test_ref_data_save_migrates_legacy_scan_root_without_export_override():
    controller = FakeController(
        {
            "general": {"basedir": "/out"},
            "ref_data": {
                "_schema_version": 2,
                "general": {
                    "data_root": "/old/ref",
                    "strict_reference": True,
                    "Runoff_ref_source": "OldRef",
                },
            },
        }
    )
    page = PageRefData.__new__(PageRefData)
    page.controller = controller
    page.data_root_input = FakeText("/new/ref")
    page._source_configs = {
        "Runoff": {
            "NewRef": {
                "def_nml_path": "/defs/NewRef.yaml",
                "general": {"root_dir": "/new/ref/NewRef"},
                "varname": "q",
            }
        }
    }

    page.save_to_config()

    ref_data = controller.config["ref_data"]
    assert ref_data["_schema_version"] == 2
    assert ref_data["general"]["strict_reference"] is True
    assert "data_root" not in ref_data["general"]
    assert ref_data["_scan_root"] == "/new/ref"
    assert ref_data["general"]["Runoff_ref_source"] == "NewRef"
    assert ref_data["source_configs"]["Runoff::NewRef"]["_var_name"] == "Runoff"
    assert controller.synced is True


def test_ref_data_save_preserves_explicit_runtime_override_separately_from_scan_root():
    controller = FakeController(
        {
            "general": {"basedir": "/out"},
            "ref_data": {
                "_data_root_explicit": True,
                "general": {
                    "data_root": "/runtime/override",
                    "Runoff_ref_source": "Ref",
                },
            },
        }
    )
    page = PageRefData.__new__(PageRefData)
    page.controller = controller
    page.data_root_input = FakeText("/scan/root")
    page._source_configs = {
        "Runoff": {
            "Ref": {
                "general": {"root_dir": "/registered/root"},
                "varname": "q",
            }
        }
    }

    page.save_to_config()

    ref_data = controller.config["ref_data"]
    assert ref_data["general"]["data_root"] == "/runtime/override"
    assert ref_data["_data_root_explicit"] is True
    assert ref_data["_scan_root"] == "/scan/root"


def test_ref_data_load_uses_registry_when_generated_namelist_is_missing(monkeypatch):
    from types import SimpleNamespace

    import openbench.data.registry.manager as manager_module

    ref = SimpleNamespace(
        root_dir="${OPENBENCH_REF_ROOT}/Grid/MidRes/Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2000, 2020],
        grid_res=0.25,
        fulllist="",
        variables={
            "Runoff": SimpleNamespace(
                varname="ro",
                varunit="mm day-1",
                prefix="runoff_",
                suffix=".nc",
                sub_dir="Runoff/Demo",
            )
        },
    )
    monkeypatch.setattr(
        manager_module,
        "get_registry",
        lambda: SimpleNamespace(get_reference=lambda name: ref if name == "Demo" else None),
    )

    controller = FakeController(
        {
            "evaluation_items": {"Runoff": True},
            "ref_data": {
                "general": {"Runoff_ref_source": "Demo"},
                "def_nml": {"Demo": "/missing/Demo.yaml"},
            },
        }
    )
    controller.storage = object()
    controller.project_root = ""
    page = PageRefData.__new__(PageRefData)
    page.controller = controller
    page.data_root_input = FakeText("")
    page._source_configs = {}
    page._var_combos = {"Runoff": FakeLoadCombo([("Choose", None), ("Demo", "Demo")])}
    page._var_advanced_fields = {
        "Runoff": {key: FakeText("") for key in ("varname", "varunit", "prefix", "suffix", "sub_dir")}
    }
    page._rebuild_variable_groups = lambda: None

    page.load_from_config()

    source = page._source_configs["Runoff"]["Demo"]
    assert source["general"]["root_dir"] == "${OPENBENCH_REF_ROOT}/Grid/MidRes/Water"
    assert source["varname"] == "ro"
    assert source["def_nml_path"] == "/missing/Demo.yaml"


def test_general_save_preserves_runtime_local_openbench_path():
    controller = FakeController(
        {
            "general": {
                "basename": "old",
                "basedir": "/old",
                "execution_mode": "local",
                "python_path": "/usr/bin/python",
                "conda_env": "base",
                "local_openbench_path": "/repo/openbench",
                "remote": {"host": "example"},
            }
        }
    )
    page = PageGeneral.__new__(PageGeneral)
    page.controller = controller
    page.basename_input = FakeText("case")
    page.basedir_input = FakePath("/out")
    page.syear_spin = FakeSpin(2000)
    page.eyear_spin = FakeSpin(2020)
    page.min_year_spin = FakeSpin(1.0)
    page.min_lat_spin = FakeSpin(-90.0)
    page.max_lat_spin = FakeSpin(90.0)
    page.min_lon_spin = FakeSpin(-180.0)
    page.max_lon_spin = FakeSpin(180.0)
    page.tim_res_combo = FakeCombo("month")
    page.grid_res_spin = FakeSpin(1.0)
    page.timezone_spin = FakeSpin(0.0)
    page.time_alignment_combo = FakeCombo("intersection", "intersection")
    page.cb_evaluation = FakeCheck(True)
    page.cb_comparison = FakeCheck(True)
    page.cb_statistics = FakeCheck(False)
    page.cb_debug = FakeCheck(False)
    page.cb_report = FakeCheck(True)
    page.cb_only_drawing = FakeCheck(False)
    page.cb_igbp = FakeCheck(True)
    page.cb_pft = FakeCheck(True)
    page.cb_climate = FakeCheck(True)
    page.cb_unified_mask = FakeCheck(True)
    page.num_cores_spin = FakeSpin(4)
    page.weight_combo = FakeCombo("None")

    page.save_to_config()

    general = controller.config["general"]
    assert general["local_openbench_path"] == "/repo/openbench"
    assert general["remote"] == {"host": "example"}


def test_general_save_includes_visible_performance_settings():
    controller = FakeController(
        {
            "general": {
                "basename": "old",
                "basedir": "/old",
                "execution_mode": "local",
            }
        }
    )
    page = PageGeneral.__new__(PageGeneral)
    page.controller = controller
    page.basename_input = FakeText("case")
    page.basedir_input = FakePath("/out")
    page.syear_spin = FakeSpin(2000)
    page.eyear_spin = FakeSpin(2020)
    page.min_year_spin = FakeSpin(1.0)
    page.min_lat_spin = FakeSpin(-90.0)
    page.max_lat_spin = FakeSpin(90.0)
    page.min_lon_spin = FakeSpin(-180.0)
    page.max_lon_spin = FakeSpin(180.0)
    page.tim_res_combo = FakeCombo("month")
    page.grid_res_spin = FakeSpin(1.0)
    page.timezone_spin = FakeSpin(0.0)
    page.time_alignment_combo = FakeCombo("intersection", "intersection")
    page.cb_evaluation = FakeCheck(True)
    page.cb_comparison = FakeCheck(True)
    page.cb_statistics = FakeCheck(False)
    page.cb_debug = FakeCheck(False)
    page.cb_report = FakeCheck(True)
    page.cb_only_drawing = FakeCheck(False)
    page.cb_igbp = FakeCheck(True)
    page.cb_pft = FakeCheck(True)
    page.cb_climate = FakeCheck(True)
    page.cb_unified_mask = FakeCheck(True)
    page.num_cores_spin = FakeSpin(4)
    page.weight_combo = FakeCombo("None")
    page.cb_netcdf_compression = FakeCheck(True)
    page.netcdf_compression_level_spin = FakeSpin(1)
    page.mfdataset_batch_mode_combo = FakeCombo("Fixed batch size", "fixed")
    page.mfdataset_batch_size_spin = FakeSpin(25)
    page.mfdataset_auto_min_files_spin = FakeSpin(150)
    page.mfdataset_auto_max_size_spin = FakeSpin(80)
    page.mfdataset_auto_memory_fraction_spin = FakeSpin(0.5)
    page.cb_dask_enabled = FakeCheck(True)
    page.dask_workers_spin = FakeSpin(3)
    page.dask_threads_spin = FakeSpin(2)
    page.cb_dask_processes = FakeCheck(False)
    page.dask_memory_limit_input = FakeText("2GB")

    page.save_to_config()

    general = controller.config["general"]
    assert general["io"] == {
        "netcdf_compression": True,
        "netcdf_compression_level": 1,
        "mfdataset_batch_size": 25,
        "mfdataset_auto_batch_min_files": 150,
        "mfdataset_auto_batch_max_size": 80,
        "mfdataset_auto_batch_memory_fraction": 0.5,
    }
    assert general["dask"] == {
        "enabled": True,
        "n_workers": 3,
        "threads_per_worker": 2,
        "processes": False,
        "memory_limit": "2GB",
    }


def test_runtime_page_startup_autoloads_instead_of_clearing_cached_settings():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "openbench" / "gui" / "pages" / "page_runtime.py"
    ).read_text(encoding="utf-8")

    assert "self._auto_load_settings()" in source
    assert "self._clear_cached_settings_file()" not in source
