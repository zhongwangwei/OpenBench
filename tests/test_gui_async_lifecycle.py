import pytest

from tests.gui_fakes import FakeButton  # noqa: E402

pytest.importorskip("PySide6")

from openbench.gui.controller import WizardController  # noqa: E402
from openbench.gui.main_window import MainWindow  # noqa: E402
from openbench.gui.pages.page_preview import PagePreview  # noqa: E402
from openbench.gui.widgets.validation_dialog import ValidationWorker  # noqa: E402


class FakeController:
    current_page = "run_monitor"

    def prev_page(self):
        return "preview"

    def next_page(self):
        return None


class RunningRunner:
    def isRunning(self):
        return True


def test_main_window_disables_navigation_while_runner_is_active():
    window = MainWindow.__new__(MainWindow)
    window.controller = FakeController()
    window.btn_back = FakeButton()
    window.btn_next = FakeButton()
    window.btn_rerun = FakeButton()
    window.btn_load = FakeButton()
    window.btn_new = FakeButton()
    window.pages = {"run_monitor": type("RunPage", (), {"_runner": RunningRunner()})()}

    window._update_buttons()

    assert window.btn_back.enabled is False
    assert window.btn_next.enabled is False
    assert window.btn_rerun.enabled is False
    assert window.btn_rerun.visible is True
    assert window.btn_load.enabled is False
    assert window.btn_new.enabled is False


def test_main_window_rejects_load_and_new_while_runner_is_active(monkeypatch):
    warnings = []
    window = MainWindow.__new__(MainWindow)
    window._runner_is_active = lambda: True
    monkeypatch.setattr(
        "openbench.gui.main_window.QMessageBox.warning",
        lambda *_args: warnings.append(True),
    )

    window._on_load_clicked()
    window._on_new_clicked()

    assert warnings == [True, True]


def test_next_saves_current_page_before_navigation():
    events = []

    class Controller:
        current_page = "sim_data"

        def go_next(self):
            events.append("next")
            return True

    class Page:
        def validate(self):
            events.append("validate")
            return True

        def save_to_config(self):
            events.append("save")

    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window.pages = {"sim_data": Page()}
    window._runner_is_active = lambda: False
    window._save_current_page = lambda trigger_sync=False: window.pages["sim_data"].save_to_config()

    window._on_next_clicked()

    assert events == ["validate", "save", "next"]


def test_preview_ignores_duplicate_run_request_while_export_in_progress():
    preview = PagePreview.__new__(PagePreview)
    preview._export_in_progress = True
    calls = []
    preview._export_and_run_once = lambda: calls.append("run") or True

    assert preview.export_and_run() is False
    assert calls == []


def test_validation_worker_uses_snapshot_not_live_config_dicts():
    seen = {}

    class Validator:
        def validate_all(self, sources, general_config, progress_callback):
            seen["sources"] = sources
            seen["general"] = general_config
            return object()

    sources = {"Runoff::RefA": {"general": {"root_dir": "/old"}}}
    general = {"basedir": "/old-out"}
    worker = ValidationWorker(Validator(), sources, general)

    sources["Runoff::RefA"]["general"]["root_dir"] = "/new"
    general["basedir"] = "/new-out"
    worker.run()

    assert seen["sources"]["Runoff::RefA"]["general"]["root_dir"] == "/old"
    assert seen["general"]["basedir"] == "/old-out"


def test_progress_parser_comparison_stage_uses_specific_increment():
    from openbench.gui.progress_parser import parse_progress_line

    constants = {
        "PROGRESS_INIT": 5,
        "PROGRESS_WORK": 80,
        "PROGRESS_MAX": 95,
        "PROGRESS_INCREMENT": 2,
    }
    state = {
        "completed_eval_tasks": set(),
        "completed_groupby_tasks": set(),
        "completed_comparison_tasks": set(),
        "total_tasks": 0,
        "num_comparisons": 0,
        "num_variables": 0,
    }

    progress, _var, stage = parse_progress_line("Comparison", 5, state, constants)

    assert stage == "Comparison"
    assert progress == 7


def test_save_current_page_restores_auto_sync_after_save_failure():
    class Controller:
        current_page = "general"
        auto_sync_enabled = True

    class Page:
        def save_to_config(self):
            raise RuntimeError("boom")

    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window.pages = {"general": Page()}

    with pytest.raises(RuntimeError, match="boom"):
        window._save_current_page(trigger_sync=False)

    assert window.controller.auto_sync_enabled is True


def test_setup_local_storage_flushes_stops_and_disconnects_remote_storage(tmp_path):
    from openbench.remote.storage import RemoteStorage

    events = []

    class Sync:
        def __init__(self):
            self._on_status_changed = lambda *args: None
            self._ssh = None

        def sync_all(self):
            events.append("sync_all")
            return True

        def stop_background_sync(self):
            events.append("stop")

    class SSH:
        def disconnect(self):
            events.append("disconnect")

    class Controller:
        storage = RemoteStorage("/remote/project", Sync())
        ssh_manager = SSH()
        project_root = ""

    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window._sync_status = None

    assert window.setup_local_storage(str(tmp_path)) is True

    assert events == ["stop", "sync_all", "disconnect"]
    assert window.controller.ssh_manager is None
    assert window.controller.project_root == str(tmp_path)


def test_setup_local_storage_preserves_remote_storage_when_flush_fails(monkeypatch, tmp_path):
    from openbench.remote.storage import RemoteStorage

    events = []

    class Sync:
        _on_status_changed = None
        _ssh = None

        def sync_all(self):
            events.append("sync_all")
            return False

        def stop_background_sync(self):
            events.append("stop")

        def start_background_sync(self):
            events.append("start")

    class SSH:
        def disconnect(self):
            events.append("disconnect")

    remote_storage = RemoteStorage("/remote/project", Sync())

    class Controller:
        storage = remote_storage
        ssh_manager = SSH()
        project_root = "/old"

    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window._sync_status = None
    monkeypatch.setattr("openbench.gui.main_window.QMessageBox.warning", lambda *args: None)

    assert window.setup_local_storage(str(tmp_path)) is False
    assert events == ["stop", "sync_all", "start"]
    assert window.controller.storage is remote_storage
    assert window.controller.project_root == "/old"
    assert window.controller.ssh_manager is not None


def test_setup_remote_storage_stops_previous_sync_before_replacing(monkeypatch):
    from openbench.gui import main_window
    from openbench.remote.storage import RemoteStorage

    events = []

    class OldSync:
        def sync_all(self):
            events.append("old-sync-all")
            return True

        def stop_background_sync(self):
            events.append("old-stop")

    class NewSync:
        def __init__(self, ssh, root):
            events.append(("new", ssh, root))
            self._on_status_changed = None

        def start_background_sync(self):
            events.append("new-start")

        def get_overall_status(self):
            from openbench.remote.sync import SyncStatus

            return SyncStatus.SYNCED

        def get_pending_count(self):
            return 0

        def retry_errors(self):
            pass

    class Controller:
        storage = RemoteStorage("/old", OldSync())
        ssh_manager = object()

    ssh = object()
    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window._sync_status = None
    window._nav_bar_layout = None
    sync_widget = type(
        "W",
        (),
        {"retry_clicked": type("S", (), {"connect": lambda *a: None})(), "set_status": lambda *a: None},
    )
    monkeypatch.setattr(main_window, "SyncStatusWidget", lambda parent: sync_widget())
    monkeypatch.setattr("openbench.remote.sync.SyncEngine", NewSync)

    window.setup_remote_storage(ssh, "/new")

    assert events[:2] == ["old-stop", "old-sync-all"]
    assert events[2:] == [("new", ssh, "/new"), "new-start"]


def test_setup_remote_storage_disconnects_replaced_ssh_manager(monkeypatch):
    from openbench.remote.storage import RemoteStorage

    events = []

    class OldSSH:
        def disconnect(self):
            events.append("old-disconnect")

    class OldSync:
        _ssh = OldSSH()
        _on_status_changed = None

        def sync_all(self):
            return True

        def stop_background_sync(self):
            pass

    class NewSync:
        def __init__(self, ssh, root):
            self._on_status_changed = None

        def start_background_sync(self):
            pass

    class Controller:
        storage = RemoteStorage("/old", OldSync())
        ssh_manager = object()

    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window._sync_status = None
    window._nav_bar_layout = None
    window._setup_sync_status = lambda sync_engine: None
    monkeypatch.setattr("openbench.remote.sync.SyncEngine", NewSync)

    window.setup_remote_storage(object(), "/new")

    assert events == ["old-disconnect"]


def test_find_remote_project_root_accepts_v3_source_marker(monkeypatch):
    from openbench.gui import main_window

    commands = []

    class SSH:
        is_connected = True

    def fake_execute(_ssh, command, timeout=None):
        commands.append(command)
        return "", "", 0

    window = MainWindow.__new__(MainWindow)
    window._get_remote_ssh_manager = lambda: SSH()
    monkeypatch.setattr(main_window, "execute_responsive", fake_execute)

    assert window._find_remote_project_root("/remote/OpenBench/output/case") == "/remote/OpenBench/output/case"
    assert "src/openbench/cli/main.py" in commands[0]
    assert "openbench/openbench.py" not in commands[0]


def test_close_waits_for_remote_download_worker_to_stop(monkeypatch):
    class DownloadWorker:
        stopped = False

        def isRunning(self):
            return True

        def stop(self):
            self.stopped = True

        def wait(self, timeout):
            return False

    class Event:
        ignored = False

        def ignore(self):
            self.ignored = True

    worker = DownloadWorker()
    window = MainWindow.__new__(MainWindow)
    window.pages = {"run_monitor": type("RunPage", (), {"_runner": None, "_download_worker": worker})()}
    event = Event()
    monkeypatch.setattr("openbench.gui.main_window.QMessageBox.warning", lambda *args: None)

    window.closeEvent(event)

    assert worker.stopped is True
    assert event.ignored is True


def test_load_local_config_rehomes_storage_to_loaded_project(monkeypatch, tmp_path):
    from openbench.remote.storage import LocalStorage

    project = tmp_path / "loaded_project"
    project.mkdir()
    config = tmp_path / "external" / "case.yaml"
    config.parent.mkdir()
    config.write_text("general:\n  basename: loaded\n  basedir: ./output\n", encoding="utf-8")

    window = MainWindow.__new__(MainWindow)
    window.controller = WizardController()
    window.controller.storage = LocalStorage(str(tmp_path / "old_project"))
    window.pages = {}
    window._find_project_root = lambda _start: str(project)
    window._validate_loaded_paths = lambda _config: None
    monkeypatch.setattr("openbench.gui.main_window.QMessageBox.information", lambda *args: None)

    window._load_config_file(str(config))

    assert window.controller.project_root == str(project)
    assert isinstance(window.controller.storage, LocalStorage)
    assert window.controller.storage.project_dir == str(project)
    assert window.controller.get_output_dir().startswith(str(project))
