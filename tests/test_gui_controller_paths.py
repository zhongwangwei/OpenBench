import os

import pytest

pytest.importorskip("PySide6")

from openbench.gui.controller import WizardController  # noqa: E402
from openbench.remote.storage import LocalStorage, RemoteStorage  # noqa: E402
from tests.gui_fakes import FakeLineEdit  # noqa: E402


def _controller(config, storage=None, project_root=""):
    controller = WizardController.__new__(WizardController)
    controller._config = config
    controller._storage = storage
    controller._project_root = project_root
    return controller


def test_remote_default_output_dir_uses_remote_openbench_path_not_local_root():
    controller = _controller(
        {
            "general": {
                "basename": "demo",
                "basedir": "./output",
                "remote": {"openbench_path": "/remote/openbench"},
            }
        },
        storage=RemoteStorage("/remote/openbench", sync_engine=object()),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == "/remote/openbench/output/demo"


def test_remote_relative_output_dir_uses_remote_storage_root_when_openbench_path_missing():
    controller = _controller(
        {"general": {"basename": "demo", "basedir": "runs"}},
        storage=RemoteStorage("/remote/project", sync_engine=object()),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == "/remote/project/runs/demo"


def test_local_relative_output_dir_uses_selected_relative_basedir():
    controller = _controller(
        {"general": {"basename": "demo", "basedir": "runs"}},
        storage=LocalStorage("/local/source/tree"),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == os.path.join("/local/source/tree", "runs", "demo")


class FakeBrowseSSH:
    is_connected = True

    def __init__(self, existing_dirs=(), home="/home/me"):
        self._existing = set(existing_dirs)
        self._home = home
        self.commands = []

    def execute(self, command, timeout=30):
        import shlex

        self.commands.append(command)
        if command.startswith("for p in "):
            # Emulate the compound probe: first listed candidate that exists
            # wins; "$HOME" expands to the remote home.
            listed = command[len("for p in ") :].split(";", 1)[0]
            for candidate in shlex.split(listed):
                candidate = candidate.replace("$HOME", self._home)
                if candidate in self._existing:
                    return candidate, "", 0
            return "/", "", 0
        if command.startswith("test -d "):
            path = command[len("test -d ") :].strip().strip("'\"")
            return "", "", (0 if path in self._existing else 1)
        return "", "", 1

    def _get_home_dir(self):
        return self._home


def _remote_controller(config=None):
    return _controller(
        config or {"general": {"remote": {"openbench_path": "/remote/openbench"}}},
        storage=RemoteStorage("/remote/openbench", sync_engine=object()),
    )


def _fail_local_dialog(*args, **kwargs):
    raise AssertionError("used local file dialog")


def test_pick_remote_path_returns_emitted_path(qapp, monkeypatch):
    from PySide6.QtCore import QTimer, Signal
    from PySide6.QtWidgets import QWidget

    from openbench.gui import path_utils
    from openbench.gui.widgets import remote_config

    created = []

    class StubBrowser(QWidget):
        file_selected = Signal(str)

        def __init__(self, ssh_manager, start_path, parent=None, select_dirs=False):
            super().__init__(parent)
            created.append({"start": start_path, "select_dirs": select_dirs})
            QTimer.singleShot(0, lambda: self.file_selected.emit("/remote/chosen"))

    monkeypatch.setattr(remote_config, "RemoteFileBrowser", StubBrowser)

    result = path_utils.pick_remote_path(object(), None, "Pick", "/start", select_dirs=False)

    assert result == "/remote/chosen"
    assert created == [{"start": "/start", "select_dirs": False}]


def test_pick_remote_path_returns_empty_when_dialog_cancelled(qapp, monkeypatch):
    from PySide6.QtCore import QTimer, Signal
    from PySide6.QtWidgets import QWidget

    from openbench.gui import path_utils
    from openbench.gui.widgets import remote_config

    class StubBrowser(QWidget):
        file_selected = Signal(str)

        def __init__(self, ssh_manager, start_path, parent=None, select_dirs=False):
            super().__init__(parent)
            QTimer.singleShot(0, lambda: self.parent().reject())

    monkeypatch.setattr(remote_config, "RemoteFileBrowser", StubBrowser)

    assert path_utils.pick_remote_path(object(), None, "Pick", "/start") == ""


def test_remote_start_path_keeps_existing_remote_current_path():
    from openbench.gui import path_utils

    ssh = FakeBrowseSSH(existing_dirs=["/remote/data"])

    start = path_utils._resolve_remote_start_path(_remote_controller(), ssh, "/remote/data")

    assert start == "/remote/data"
    # All candidates are probed in ONE compound round trip.
    assert len(ssh.commands) == 1
    assert ssh.commands[0].startswith("for p in ")


def test_remote_start_path_falls_back_when_current_path_is_stale():
    from openbench.gui import path_utils

    ssh = FakeBrowseSSH(existing_dirs=["/remote/openbench"])

    start = path_utils._resolve_remote_start_path(_remote_controller(), ssh, "C:\\old\\local\\path")

    assert start == "/remote/openbench"
    assert len(ssh.commands) == 1


def test_remote_start_path_skips_stale_openbench_path():
    from openbench.gui import path_utils

    ssh = FakeBrowseSSH(existing_dirs=["/home/me"])

    start = path_utils._resolve_remote_start_path(_remote_controller(), ssh, "")

    assert start == "/home/me"
    assert len(ssh.commands) == 1


def test_remote_start_path_falls_back_to_root_when_everything_is_stale():
    from openbench.gui import path_utils

    ssh = FakeBrowseSSH()

    start = path_utils._resolve_remote_start_path(_remote_controller(), ssh, "")

    assert start == "/"
    assert len(ssh.commands) == 1


def test_remote_directory_exists_propagates_connection_loss():
    from openbench.gui import path_utils
    from openbench.remote.ssh import SSHConnectionError

    class DeadSSH:
        def execute(self, command, timeout=30):
            raise SSHConnectionError("session dropped")

    # 'Connection lost' must not be folded into 'directory does not exist' —
    # callers turn that into a misleading 'not found' message.
    with pytest.raises(SSHConnectionError):
        path_utils._remote_directory_exists(DeadSSH(), "/remote/data")


def test_remote_start_path_survives_null_remote_config_section():
    from openbench.gui import path_utils

    controller = _remote_controller({"general": {"remote": None}})

    start = path_utils._resolve_remote_start_path(controller, FakeBrowseSSH(existing_dirs=["/home/me"]), "")

    assert start == "/home/me"


def test_controller_remote_settings_tolerates_null_yaml_sections():
    assert _controller({"general": {"remote": None}}).remote_settings() == {}
    assert _controller({"general": None}).remote_settings() == {}
    assert _controller({}).remote_settings() == {}
    assert _controller({"general": {"remote": {"openbench_path": "/r"}}}).remote_settings() == {"openbench_path": "/r"}


def test_remote_output_dir_survives_null_remote_section():
    controller = _controller(
        {"general": {"basename": "demo", "basedir": "runs", "remote": None}},
        storage=RemoteStorage("/remote/project", sync_engine=object()),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == "/remote/project/runs/demo"


def test_browse_remote_directory_warns_with_reconnect_guidance_when_disconnected(monkeypatch):
    from openbench.gui import path_utils

    warnings = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    controller = _remote_controller()
    controller._ssh_manager = None

    assert path_utils.browse_remote_directory(controller, None, "Pick") == ""
    assert warnings and "Runtime Environment" in warnings[0][1]


def test_browse_directory_routes_remote_mode_to_remote_browser(monkeypatch):
    from openbench.gui import path_utils

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getExistingDirectory", _fail_local_dialog)
    monkeypatch.setattr(path_utils, "browse_remote_directory", lambda *args, **kwargs: "/remote/picked")

    assert path_utils.browse_directory(_remote_controller(), None, "Pick", "") == "/remote/picked"


def test_browse_directory_routes_local_mode_to_local_dialog(monkeypatch):
    from openbench.gui import path_utils

    controller = _controller({"general": {}}, storage=LocalStorage("/local/tree"))
    captured = {}

    def fake_dialog(parent, title, directory="", *args, **kwargs):
        captured["args"] = (title, directory)
        return "/local/picked"

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getExistingDirectory", fake_dialog)

    assert path_utils.browse_directory(controller, None, "Pick", "/local/current") == "/local/picked"
    assert captured["args"] == ("Pick", "/local/current")


def _page_browse_routes_through_shared_helper(monkeypatch, page_module_name, page_class_name, input_attr, method_name):
    import importlib

    page_module = importlib.import_module(f"openbench.gui.pages.{page_module_name}")
    page_class = getattr(page_module, page_class_name)

    controller = _remote_controller()
    controller._ssh_manager = None
    page = page_class.__new__(page_class)
    page.controller = controller
    setattr(page, input_attr, FakeLineEdit())
    getattr(page, input_attr).setText("/old/value")

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getExistingDirectory", _fail_local_dialog)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *args: None)
    captured = {}

    def fake_browse(controller, parent, title, current_path=""):
        captured["args"] = (controller, title, current_path)
        return "/remote/picked"

    monkeypatch.setattr(page_module, "browse_directory", fake_browse)

    getattr(page, method_name)()

    return controller, captured, getattr(page, input_attr).text()


def test_remote_sim_browse_routes_through_shared_browse_directory(monkeypatch):
    controller, captured, text = _page_browse_routes_through_shared_helper(
        monkeypatch, "page_sim_data", "PageSimData", "_root_input", "_browse_root"
    )

    assert captured.get("args") == (controller, "Select Simulation Root Directory", "/old/value")
    assert text == "/remote/picked"


def test_remote_ref_browse_routes_through_shared_browse_directory(monkeypatch):
    controller, captured, text = _page_browse_routes_through_shared_helper(
        monkeypatch, "page_ref_data", "PageRefData", "data_root_input", "_browse_data_root"
    )

    assert captured.get("args") == (controller, "Select Reference Data Root", "/old/value")
    assert text == "/remote/picked"


def test_registry_root_browse_routes_through_shared_browse_directory(monkeypatch):
    controller, captured, text = _page_browse_routes_through_shared_helper(
        monkeypatch, "page_registry", "PageRegistry", "ds_root_dir", "_browse_ds_root"
    )

    assert captured.get("args") == (controller, "Select root_dir", "/old/value")
    assert text == "/remote/picked"


def test_general_output_browse_routes_through_shared_browse_directory(monkeypatch):
    from openbench.gui.pages import page_general
    from openbench.gui.pages.page_general import PageGeneral

    controller = _remote_controller()
    controller._ssh_manager = None
    page = PageGeneral.__new__(PageGeneral)
    page.controller = controller

    class FakePathSelector:
        def __init__(self):
            self.value = "/old/output"

        def path(self):
            return self.value

        def set_path(self, value):
            self.value = value

    page.basedir_input = FakePathSelector()

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getExistingDirectory", _fail_local_dialog)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *args: None)
    captured = {}

    def fake_browse(controller, parent, title, current_path=""):
        captured["args"] = (controller, title, current_path)
        return "/remote/output"

    monkeypatch.setattr(page_general, "browse_directory", fake_browse)

    page._browse_output_directory()

    assert captured.get("args") == (controller, "Select Output Directory", "/old/output")
    assert page.basedir_input.path() == "/remote/output"
