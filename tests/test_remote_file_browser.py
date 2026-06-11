import pytest

PySide6 = pytest.importorskip("PySide6")

from openbench.gui.widgets.remote_config import RemoteFileBrowser  # noqa: E402
from tests.gui_fakes import FakeLineEdit  # noqa: E402


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


def _first_browser_item(browser, text_part):
    for index in range(browser.file_list.count()):
        item = browser.file_list.item(index)
        if text_part in item.text():
            return item
    raise AssertionError(f"No browser item containing {text_part!r}")


SECTION = "__OPENBENCH_SECTION__"


def listing_response(resolved, ls_output, find_output=""):
    """Build the single-round-trip listing reply: resolved path, ls, find sections."""
    return f"{resolved}\n{SECTION}\n{ls_output}\n{SECTION}\n{find_output}", "", 0


def _silence_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.widgets.remote_config.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    return warnings


def test_symlink_readlink_failure_warns_and_does_not_emit(qapp, monkeypatch):
    warnings = _silence_warnings(monkeypatch)
    ssh = FakeSSH(
        {
            "cd /remote ": listing_response(
                "/remote",
                "total 1\nlrwxrwxrwx 1 user group 11 Jan 1 00:00 model.yaml -> missing.yaml",
                "/remote",
            ),
            "readlink -f /remote/model.yaml": ("", "broken link", 1),
        }
    )
    browser = RemoteFileBrowser(ssh, "/remote")
    emitted = []
    browser.file_selected.connect(emitted.append)

    browser._on_item_double_clicked(_first_browser_item(browser, "model.yaml"))

    assert emitted == []
    assert warnings == [("Broken Link", "Failed to resolve remote symlink:\n/remote/model.yaml")]


def test_load_directory_uses_a_single_ssh_round_trip(qapp, monkeypatch):
    warnings = _silence_warnings(monkeypatch)
    ssh = FakeSSH({"cd /remote ": listing_response("/remote", "total 0")})

    browser = RemoteFileBrowser(ssh, "/remote")

    assert len(ssh.commands) == 1
    assert warnings == []
    assert browser.file_list.count() == 1  # the ".." parent entry rendered


def test_symlink_file_double_click_in_dir_mode_selects_current_directory(qapp, monkeypatch):
    """In select_dirs mode a symlink to a file must not be returned as the chosen directory."""
    _silence_warnings(monkeypatch)
    ssh = FakeSSH(
        {
            "cd /remote ": listing_response(
                "/remote",
                "total 1\nlrwxrwxrwx 1 user group 11 Jan 1 00:00 data.nc -> real_data.nc",
                "/remote",
            ),
            "readlink -f /remote/data.nc": ("/real/real_data.nc\n", "", 0),
        }
    )
    browser = RemoteFileBrowser(ssh, "/remote", select_dirs=True)
    emitted = []
    browser.file_selected.connect(emitted.append)

    browser._on_item_double_clicked(_first_browser_item(browser, "data.nc"))

    assert emitted == ["/remote"]


def test_parent_entry_select_emits_normalized_parent_path(qapp, monkeypatch):
    _silence_warnings(monkeypatch)
    ssh = FakeSSH({"cd /remote/sub ": listing_response("/remote/sub", "total 0")})
    browser = RemoteFileBrowser(ssh, "/remote/sub", select_dirs=True)
    emitted = []
    browser.file_selected.connect(emitted.append)

    browser.file_list.setCurrentItem(_first_browser_item(browser, ".."))
    browser._on_select()

    assert emitted == ["/remote"]


def test_call_responsive_keeps_event_loop_alive(qapp):
    import time

    from PySide6.QtCore import QTimer

    from openbench.gui.widgets._ssh_worker import call_responsive

    fired = []
    QTimer.singleShot(20, lambda: fired.append(True))

    result = call_responsive(lambda: (time.sleep(0.05), "done")[1])

    assert result == "done"
    assert fired == [True]


def test_execute_responsive_returns_execute_tuple_and_reraises(qapp):
    from openbench.gui.widgets._ssh_worker import execute_responsive

    class OkSSH:
        def execute(self, command, timeout=None):
            return "out", "err", 0

    assert execute_responsive(OkSSH(), "echo hi", timeout=5) == ("out", "err", 0)

    class BoomSSH:
        def execute(self, command, timeout=None):
            raise OSError("link down")

    with pytest.raises(OSError, match="link down"):
        execute_responsive(BoomSSH(), "echo hi", timeout=5)


def test_load_directory_runs_ssh_off_the_gui_thread(qapp, monkeypatch):
    from PySide6.QtCore import QThread

    _silence_warnings(monkeypatch)
    threads = []

    class ThreadRecordingSSH(FakeSSH):
        def execute(self, command, timeout=30):
            threads.append(QThread.currentThread())
            return super().execute(command, timeout=timeout)

    ssh = ThreadRecordingSSH({"cd /remote ": listing_response("/remote", "total 0")})
    RemoteFileBrowser(ssh, "/remote")

    assert threads
    assert all(thread != qapp.thread() for thread in threads)


def test_symlink_double_click_is_guarded_while_loading(qapp, monkeypatch):
    """The readlink branch spins a nested event loop; queued double-clicks must not re-enter."""
    _silence_warnings(monkeypatch)
    ssh = FakeSSH(
        {
            "cd /remote ": listing_response(
                "/remote",
                "total 1\nlrwxrwxrwx 1 user group 11 Jan 1 00:00 model.yaml -> real.yaml",
                "/remote",
            ),
            "readlink -f /remote/model.yaml": ("/real/real.yaml\n", "", 0),
        }
    )
    browser = RemoteFileBrowser(ssh, "/remote")
    emitted = []
    browser.file_selected.connect(emitted.append)

    ssh.commands.clear()
    browser._loading = True
    browser._on_item_double_clicked(_first_browser_item(browser, "model.yaml"))

    assert emitted == []
    assert ssh.commands == []


def test_symlink_fallback_probes_in_one_round_trip(qapp, monkeypatch):
    """When the bulk find fails, symlink targets are probed with ONE compound
    command instead of one test -d round trip per symlink."""
    from PySide6.QtCore import Qt

    _silence_warnings(monkeypatch)
    listing = (
        "total 2\n"
        "lrwxrwxrwx 1 u g 1 Jan 1 00:00 dirlink -> target_dir\n"
        "lrwxrwxrwx 1 u g 1 Jan 1 00:00 filelink -> target_file"
    )
    ssh = FakeSSH(
        {
            # find section empty => the bulk lookup failed
            "cd /remote ": (f"/remote\n{SECTION}\n{listing}\n{SECTION}\n", "", 0),
            "for p in ": ("/remote/dirlink\n", "", 0),
        }
    )

    browser = RemoteFileBrowser(ssh, "/remote")

    assert sum(1 for command in ssh.commands if "test -d" in command) == 0
    assert sum(1 for command in ssh.commands if command.startswith("for p in ")) == 1
    assert _first_browser_item(browser, "dirlink").data(Qt.UserRole)["is_dir"] is True
    assert _first_browser_item(browser, "filelink").data(Qt.UserRole)["is_dir"] is False


def test_load_directory_ignores_reentrant_calls(qapp, monkeypatch):
    _silence_warnings(monkeypatch)
    ssh = FakeSSH({"cd /remote ": listing_response("/remote", "total 0")})
    browser = RemoteFileBrowser(ssh, "/remote")

    ssh.commands.clear()
    browser._loading = True
    browser._load_directory("/elsewhere")

    assert ssh.commands == []


class FakeConnectedSSH:
    is_connected = True

    def _get_home_dir(self):
        return "/home/me"


def test_browse_python_routes_through_shared_remote_picker(qapp, monkeypatch):
    from openbench.gui.widgets.remote_config import RemoteConfigWidget

    widget = RemoteConfigWidget.__new__(RemoteConfigWidget)
    widget._ssh_manager = FakeConnectedSSH()

    class FakeCombo:
        def __init__(self):
            self.items = []
            self.current = ""

        def findText(self, text):
            return self.items.index(text) if text in self.items else -1

        def addItem(self, text):
            self.items.append(text)

        def setCurrentText(self, text):
            self.current = text

    widget.python_combo = FakeCombo()
    captured = {}

    def fake_pick(ssh_manager, parent, title, start_path, select_dirs=True):
        captured["args"] = (title, start_path, select_dirs)
        return "/usr/bin/python3"

    monkeypatch.setattr("openbench.gui.path_utils.pick_remote_path", fake_pick)

    widget._browse_python()

    assert captured.get("args") == ("Select Python on Remote Server", "/home/me", False)
    assert widget.python_combo.current == "/usr/bin/python3"
    assert widget.python_combo.items == ["/usr/bin/python3"]


def test_browse_openbench_routes_through_shared_remote_picker(qapp, monkeypatch):
    from openbench.gui.widgets.remote_config import RemoteConfigWidget

    widget = RemoteConfigWidget.__new__(RemoteConfigWidget)
    widget._ssh_manager = FakeConnectedSSH()
    widget.openbench_input = FakeLineEdit()
    captured = {}

    def fake_pick(ssh_manager, parent, title, start_path, select_dirs=True):
        captured["args"] = (title, start_path, select_dirs)
        return "/opt/openbench"

    monkeypatch.setattr("openbench.gui.path_utils.pick_remote_path", fake_pick)

    widget._browse_openbench()

    assert captured.get("args") == ("Select OpenBench Directory on Remote Server", "/home/me", True)
    assert widget.openbench_input.value == "/opt/openbench"


def test_safe_remote_path_expands_leading_tilde():
    import shlex

    from openbench.gui.widgets.remote_config import _safe_remote_path

    assert _safe_remote_path("~/data") == '"$HOME"/data'
    assert _safe_remote_path("~") == '"$HOME"'
    assert _safe_remote_path("~/dir with space") == '"$HOME"' + shlex.quote("/dir with space")
    # Only a LEADING tilde is home expansion; elsewhere it is a literal char.
    assert _safe_remote_path("/data/~backup") == shlex.quote("/data/~backup")


def test_remote_directory_exists_expands_tilde(monkeypatch):
    from openbench.gui.path_utils import _remote_directory_exists

    class SSH:
        def __init__(self):
            self.commands = []

        def execute(self, command, timeout=30):
            self.commands.append(command)
            return "", "", 0

    ssh = SSH()

    assert _remote_directory_exists(ssh, "~/OpenBenchData") is True
    assert ssh.commands == ['test -d "$HOME"/OpenBenchData']


def test_load_directory_ignores_login_banner_in_resolved_path(qapp, monkeypatch):
    """rc-file/motd noise printed before pwd output must not become the current path."""
    _silence_warnings(monkeypatch)
    stdout = f"Welcome to Cluster X\n*** maintenance friday ***\n/remote\n{SECTION}\ntotal 0\n{SECTION}\n"
    ssh = FakeSSH({"cd /remote ": (stdout, "", 0)})

    browser = RemoteFileBrowser(ssh, "/remote")

    assert browser._current_path == "/remote"
    assert browser.path_input.text() == "/remote"


def test_load_directory_failure_warns_and_keeps_previous_path(qapp, monkeypatch):
    warnings = _silence_warnings(monkeypatch)
    ssh = FakeSSH(
        {
            "cd /remote ": listing_response("/remote", "total 0"),
            "cd /missing ": ("", "permission denied", 21),
        }
    )
    browser = RemoteFileBrowser(ssh, "/remote")

    browser._load_directory("/missing")

    assert browser._current_path == "/remote"
    assert browser.path_input.text() == "/remote"
    assert warnings == [("Remote Browser", "Failed to list remote directory:\n/missing")]
