import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


def test_ref_data_save_preserves_general_metadata_and_top_level_metadata():
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
    assert ref_data["general"]["data_root"] == "/new/ref"
    assert ref_data["general"]["Runoff_ref_source"] == "NewRef"
    assert ref_data["source_configs"]["Runoff::NewRef"]["_var_name"] == "Runoff"
    assert controller.synced is True


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
