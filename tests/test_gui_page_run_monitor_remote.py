import pytest

pytest.importorskip("PySide6")

from openbench.gui.pages.page_run_monitor import PageRunMonitor, RemoteFolderDownloadWorker  # noqa: E402


class RaisingController:
    def parent(self):
        raise RuntimeError("main window gone")


class FakeSSH:
    is_connected = True

    def __init__(self, exc=None):
        self.exc = exc
        self.commands = []

    def execute(self, command, timeout=30, should_abort=None):
        self.commands.append(command)
        if self.exc:
            raise self.exc
        return "", "", 1


def _page(controller=None):
    page = PageRunMonitor.__new__(PageRunMonitor)
    page.controller = controller
    return page


def test_get_ssh_manager_records_diagnostic_when_lookup_raises():
    page = _page(RaisingController())

    assert page._get_ssh_manager() is None
    assert page._last_ssh_manager_error == "main window gone"


def test_open_remote_output_reports_directory_probe_exception(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    page = _page()
    page._get_ssh_manager = lambda: FakeSSH(exc=RuntimeError("network down"))

    page._open_remote_output("/remote/output")

    assert warnings == [
        (
            "Remote Output Error",
            "Failed to check remote output directory:\n/remote/output\n\nError: network down",
        )
    ]


def test_open_remote_output_not_connected_includes_lookup_diagnostic(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    page = _page(RaisingController())

    page._open_remote_output("/remote/output")

    assert warnings == [
        (
            "Not Connected",
            "SSH connection is not available.\n\nRemote output directory:\n/remote/output\n\nDetails: main window gone",
        )
    ]


def test_remote_download_relpath_rejects_paths_outside_remote_dir():
    page = _page()

    assert page._remote_download_relpath("/remote/output/a/b.nc", "/remote/output") == "a/b.nc"
    assert page._remote_download_relpath("/remote/output2/evil.nc", "/remote/output") is None
    assert page._remote_download_relpath("/etc/passwd", "/remote/output") is None


def test_on_finished_reports_stopped_without_failed_warning(monkeypatch):
    page = _page()
    page.dashboard = type("Dashboard", (), {"stop_monitoring": lambda self: None})()
    page._refresh_parent_navigation = lambda: None
    infos = []
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QMessageBox.information",
        lambda parent, title, message: infos.append((title, message)),
    )
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )

    page._on_finished(False, "Stopped by user")

    assert infos == [("Stopped", "Evaluation stopped by user.")]
    assert warnings == []


class FakeSftp:
    def __init__(self):
        self.downloads = []

    def get(self, remote, local, callback=None):
        if callback is not None:
            callback(1, 1)
        self.downloads.append((remote, local))


class ListingSSH(FakeSSH):
    def __init__(self):
        super().__init__()
        self.sftp = FakeSftp()

    def execute(self, command, timeout=30, should_abort=None):
        self.commands.append(command)
        return "/remote/output/a.nc\n/remote/output/sub/b.nc\n", "", 0

    def open_sftp(self):
        return self.sftp


def test_remote_folder_download_worker_downloads_in_background_target(tmp_path, qapp):
    ssh = ListingSSH()
    worker = RemoteFolderDownloadWorker(ssh, "/remote/output", str(tmp_path / "output"))
    finished = []
    worker.finished_signal.connect(
        lambda success, canceled, message, target: finished.append((success, canceled, message, target))
    )

    worker.run()

    assert finished == [(True, False, "Download complete", str(tmp_path / "output"))]
    assert len(ssh.sftp.downloads) == 2
    assert (tmp_path / "output" / "a.nc").parent.exists()


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class FakeDownloadWorker:
    created = []

    def __init__(self, ssh_manager, remote_dir, local_target):
        self.ssh_manager = ssh_manager
        self.remote_dir = remote_dir
        self.local_target = local_target
        self.progress_updated = FakeSignal()
        self.finished_signal = FakeSignal()
        self.finished = FakeSignal()
        self.stop_count = 0
        self.started = False
        FakeDownloadWorker.created.append(self)

    def isRunning(self):
        return self.started

    def stop(self):
        self.stop_count += 1

    def deleteLater(self):
        pass

    def start(self):
        self.started = True


class FakeProgressDialog:
    canceled = FakeSignal()

    def __init__(self, *args, **kwargs):
        self.closed = False

    def setWindowTitle(self, title):
        self.title = title

    def setWindowModality(self, modality):
        self.modality = modality

    def setMinimumDuration(self, duration):
        self.minimum_duration = duration

    def setValue(self, value):
        self.value = value

    def setMaximum(self, value):
        self.maximum = value

    def setLabelText(self, value):
        self.label = value

    def show(self):
        self.shown = True

    def close(self):
        self.closed = True


class FakeParentDialog:
    def __init__(self):
        self.finished = FakeSignal()
        self.destroyed = FakeSignal()


def test_remote_download_rejects_second_running_worker(monkeypatch):
    page = _page()
    page._download_worker = type("RunningWorker", (), {"isRunning": lambda self: True})()
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: pytest.fail("directory prompt should not open while a download is running"),
    )

    page._download_remote_folder(ListingSSH(), "/remote/output", FakeParentDialog())

    assert warnings == [("Download in Progress", "A remote folder download is already running.")]


def test_remote_download_stops_when_parent_output_dialog_closes(monkeypatch, tmp_path):
    from openbench.gui.pages import page_run_monitor as run_monitor_module

    FakeDownloadWorker.created = []
    parent = FakeParentDialog()
    page = _page()
    monkeypatch.setattr(
        "openbench.gui.pages.page_run_monitor.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(run_monitor_module, "RemoteFolderDownloadWorker", FakeDownloadWorker)
    monkeypatch.setattr("openbench.gui.pages.page_run_monitor.QProgressDialog", FakeProgressDialog)

    page._download_remote_folder(ListingSSH(), "/remote/output", parent)

    worker = FakeDownloadWorker.created[-1]
    parent.finished.emit(0)
    parent.destroyed.emit()

    assert worker.stop_count == 2


def test_open_output_uses_saved_run_output_even_if_config_changes(monkeypatch):
    class Controller:
        storage = object()

        def __init__(self):
            self.paths = ["/first/output", "/changed/output"]

        def get_output_dir(self):
            return self.paths.pop(0)

    page = _page(Controller())
    page._last_run_output_dir = "/saved/output"
    page._last_run_is_remote = True
    opened = []
    page._open_remote_output = lambda output_dir: opened.append(output_dir)

    page._open_output()

    assert opened == ["/saved/output"]
