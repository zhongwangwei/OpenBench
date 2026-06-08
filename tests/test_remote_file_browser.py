import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from openbench.gui.widgets.remote_config import RemoteFileBrowser  # noqa: E402


class FakeSSH:
    def __init__(self, responses):
        self.responses = responses
        self.commands = []

    def execute(self, command, timeout=30):
        self.commands.append(command)
        for needle, response in self.responses.items():
            if needle in command:
                if isinstance(response, Exception):
                    raise response
                return response
        return "", "", 1


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def _first_browser_item(browser, text_part):
    for index in range(browser.file_list.count()):
        item = browser.file_list.item(index)
        if text_part in item.text():
            return item
    raise AssertionError(f"No browser item containing {text_part!r}")


def test_symlink_readlink_failure_warns_and_does_not_emit(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.widgets.remote_config.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    ssh = FakeSSH(
        {
            "cd /remote && pwd -P": ("/remote\n", "", 0),
            "ls -la /remote": (
                "total 1\nlrwxrwxrwx 1 user group 11 Jan 1 00:00 model.yaml -> missing.yaml\n",
                "",
                0,
            ),
            "test -d /remote/model.yaml": ("", "", 1),
            "readlink -f /remote/model.yaml": ("", "broken link", 1),
        }
    )
    browser = RemoteFileBrowser(ssh, "/remote")
    emitted = []
    browser.file_selected.connect(emitted.append)

    browser._on_item_double_clicked(_first_browser_item(browser, "model.yaml"))

    assert emitted == []
    assert warnings == [("Broken Link", "Failed to resolve remote symlink:\n/remote/model.yaml")]


def test_load_directory_failure_warns_and_keeps_previous_path(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.widgets.remote_config.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    ssh = FakeSSH(
        {
            "cd /remote && pwd -P": ("/remote\n", "", 0),
            "ls -la /remote": ("total 0\n", "", 0),
            "cd /missing && pwd -P": ("", "", 1),
            "ls -la /missing": ("", "permission denied", 2),
        }
    )
    browser = RemoteFileBrowser(ssh, "/remote")

    browser._load_directory("/missing")

    assert browser._current_path == "/remote"
    assert browser.path_input.text() == "/remote"
    assert warnings == [("Remote Browser", "Failed to list remote directory:\n/missing")]
