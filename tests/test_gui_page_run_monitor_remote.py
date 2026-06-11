import pytest

pytest.importorskip("PySide6")

from openbench.gui.pages.page_run_monitor import PageRunMonitor  # noqa: E402


class RaisingController:
    def parent(self):
        raise RuntimeError("main window gone")


class FakeSSH:
    is_connected = True

    def __init__(self, exc=None):
        self.exc = exc
        self.commands = []

    def execute(self, command, timeout=30):
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
