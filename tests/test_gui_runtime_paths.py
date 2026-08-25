from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from openbench.gui.controller import WizardController  # noqa: E402
from openbench.gui.pages.page_general import PageGeneral  # noqa: E402
from openbench.gui.pages.page_runtime import PageRuntime, _LocalInstallWorker  # noqa: E402
from openbench.gui.widgets.checkbox_group import CheckboxGroup  # noqa: E402


def _runtime_page(qapp, monkeypatch):
    monkeypatch.setattr(PageRuntime, "_detect_python", lambda self: None)
    monkeypatch.setattr(PageRuntime, "_auto_load_settings", lambda self: None)
    page = PageRuntime(WizardController())
    page._auto_save_settings = lambda: None
    return page


def test_runtime_python_combo_uses_item_data_for_paths_with_spaces(qapp, monkeypatch):
    page = _runtime_page(qapp, monkeypatch)
    python_path = r"C:\Users\Jane Doe\miniconda3\python.exe"

    page.python_combo.clear()
    page.python_combo.addItem(f"{python_path} (miniconda3)", python_path)
    page.python_combo.setCurrentIndex(0)
    page.conda_combo.setCurrentIndex(0)

    page.save_to_config()

    assert page.controller.config["general"]["python_path"] == python_path
    assert page._collect_runtime_settings()["python_path"] == python_path


def test_runtime_typed_legacy_python_display_suffix_preserves_spaced_path(qapp, monkeypatch):
    page = _runtime_page(qapp, monkeypatch)
    python_path = "/tmp/Open Bench Env/bin/python"

    page.python_combo.clear()
    page.python_combo.setCurrentText(f"{python_path} (PATH)")
    page.conda_combo.setCurrentIndex(0)

    page.save_to_config()

    assert page.controller.config["general"]["python_path"] == python_path


def test_runtime_conda_selection_controls_saved_python_path(qapp, monkeypatch, tmp_path):
    page = _runtime_page(qapp, monkeypatch)
    env_path = tmp_path / "conda envs" / "openbench"

    page.conda_combo.addItem("openbench", str(env_path))
    page.conda_combo.setCurrentIndex(1)

    expected = str(env_path / "bin" / "python")
    assert page.controller.config["general"]["python_path"] == expected
    assert page.python_combo.currentData() == expected


def test_runtime_remote_mode_saves_remote_cpu_count(qapp, monkeypatch):
    page = _runtime_page(qapp, monkeypatch)

    page.radio_remote.blockSignals(True)
    page.radio_remote.setChecked(True)
    page.radio_local.setChecked(False)
    page.radio_remote.blockSignals(False)
    page.remote_config_widget.num_cores_spin.setValue(12)
    page.num_cores_spin.setValue(3)

    page.save_to_config()

    assert page.controller.config["general"]["num_cores"] == 12
    assert page._collect_runtime_settings()["num_cores"] == 12


def test_runtime_remote_mode_cannot_continue_without_execution_target(qapp, monkeypatch):
    page = _runtime_page(qapp, monkeypatch)
    page.radio_remote.setChecked(True)
    page.remote_config_widget.is_connected = lambda: False
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.pages.page_runtime.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    assert page.validate() is False
    assert warnings and warnings[-1][1] == "Remote Connection Required"


def test_switching_local_flushes_storage_before_widget_disconnect():
    events = []
    page = PageRuntime.__new__(PageRuntime)
    page.radio_local = SimpleNamespace(isChecked=lambda: True)
    page.parallel_group = SimpleNamespace(show=lambda: None)
    page.local_env_group = SimpleNamespace(show=lambda: None)
    page.cpu_available_label = SimpleNamespace(setText=lambda text: None)
    page.remote_config_widget = SimpleNamespace(
        hide=lambda: None,
        get_ssh_manager=lambda: object(),
        disconnect=lambda: events.append("disconnect"),
        reset_to_defaults=lambda: events.append("reset"),
    )
    page._switch_to_local_storage = lambda: events.append("flush-and-switch")
    page._on_config_changed = lambda: events.append("save")

    page._on_execution_mode_changed(True)

    assert events == ["flush-and-switch", "disconnect", "reset", "save"]


def test_local_install_commands_include_editable_install_with_selected_python(qapp, monkeypatch, tmp_path):
    page = _runtime_page(qapp, monkeypatch)
    install_path = tmp_path / "Open Bench"
    python_path = tmp_path / "Python Env" / "bin" / "python"

    page.python_combo.clear()
    page.python_combo.addItem(f"{python_path} (selected)", str(python_path))
    page.python_combo.setCurrentIndex(0)

    commands, starting = page._build_local_install_commands(str(install_path), "https://example.test/repo.git", False)

    assert starting == "Cloning from https://example.test/repo.git..."
    assert commands == [
        ["git", "clone", "--progress", "https://example.test/repo.git", str(install_path)],
        [str(python_path), "-m", "pip", "install", "-e", str(install_path)],
    ]


def test_local_install_worker_returns_failure_when_editable_install_fails(qapp, monkeypatch):
    import subprocess

    calls = []

    class FakeProcess:
        def __init__(self, cmd, returncode):
            self.cmd = cmd
            self.stdout = [f"ran {cmd[0]}\n"]
            self.returncode = returncode
            self.terminated = False

        def wait(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

    returncodes = iter([0, 23])

    def fake_popen(cmd, **_kwargs):
        calls.append(cmd)
        return FakeProcess(cmd, next(returncodes))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    worker = _LocalInstallWorker([["git", "pull"], ["python", "-m", "pip", "install", "-e", "."]])
    results = []
    worker.finished_with_result.connect(results.append)

    worker.run()

    assert calls == [["git", "pull"], ["python", "-m", "pip", "install", "-e", "."]]
    assert results == [23]


def test_general_load_from_config_does_not_write_stale_defaults(qapp, tmp_path):
    controller = WizardController()
    page = PageGeneral(controller)
    controller.config = {
        "general": {
            "basename": "loaded",
            "basedir": str(tmp_path),
            "syear": 1999,
            "eyear": 2001,
            "min_year": 2.5,
            "min_lat": -10.0,
            "max_lat": 50.0,
            "min_lon": 20.0,
            "max_lon": 130.0,
            "compare_tim_res": "day",
            "compare_grid_res": 0.25,
            "compare_tzone": 8.0,
            "time_alignment": "strict",
            "evaluation": True,
            "comparison": False,
            "statistics": True,
            "debug_mode": True,
            "generate_report": False,
            "only_drawing": True,
            "IGBP_groupby": False,
            "PFT_groupby": False,
            "Climate_zone_groupby": False,
            "unified_mask": False,
            "num_cores": 7,
            "weight": "area",
            "io": {"netcdf_compression": True, "mfdataset_batch_size": 0},
            "dask": {
                "enabled": True,
                "n_workers": 2,
                "threads_per_worker": 3,
                "processes": False,
                "memory_limit": "1GB",
            },
        },
        "evaluation_items": {},
        "ref_data": {"general": {}, "def_nml": {}},
        "sim_data": {"general": {}, "def_nml": {}},
        "metrics": {},
        "scores": {},
        "comparisons": {},
        "statistics": {},
    }
    before = deepcopy(controller.config["general"])
    sync_calls = []
    controller.sync_namelists = lambda *args, **kwargs: sync_calls.append((args, kwargs))

    page.load_from_config()

    assert controller.config["general"] == before
    assert sync_calls == []
    assert page.weight_combo.currentText() == "area"
    assert page.num_cores_spin.value() == 7


def test_checkbox_group_set_selection_clears_items_not_in_selection(qapp):
    group = CheckboxGroup({"Group": ["a", "b", "c"]})
    group.set_selection({"a": True, "b": True, "c": True})

    group.set_selection({"b": True})

    assert group.get_selection() == {"a": False, "b": True, "c": False}
