import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QDialog, QMessageBox  # noqa: E402

from openbench.gui.widgets._ssh_worker import SshExecuteWorker  # noqa: E402
from openbench.gui.widgets.remote_config import RemoteConfigWidget  # noqa: E402


class FakeSSHStream:
    def __init__(self):
        self.stream_calls = []
        self.execute_calls = []

    def execute_stream(self, command, callback=None, total_timeout=None):
        self.stream_calls.append((command, total_timeout))
        for line in ["line 1\n", "line 2\n"]:
            if callback:
                callback(line)
            yield line
        return 7

    def execute(self, command, timeout=None):
        self.execute_calls.append((command, timeout))
        return "fallback\n", "", 0


def test_ssh_execute_worker_streams_when_execute_stream_is_available(qapp):
    ssh = FakeSSHStream()
    worker = SshExecuteWorker(ssh, "long command", timeout=900)
    lines = []
    finished = []

    worker.line.connect(lines.append)
    worker.finished_with_result.connect(lambda exit_code, stdout, stderr: finished.append((exit_code, stdout, stderr)))

    worker.run()

    assert ssh.stream_calls == [("long command", 900)]
    assert ssh.execute_calls == []
    # Line emissions may be batched, but the concatenated text is intact.
    assert "".join(lines) == "line 1\nline 2\n"
    assert finished == [(7, "line 1\nline 2\n", "")]


def test_ssh_execute_worker_batches_line_emissions(qapp):
    class ManyLinesSSH:
        def execute_stream(self, command, total_timeout=None):
            for index in range(1000):
                yield f"line {index}\n"
            return 0

    worker = SshExecuteWorker(ManyLinesSSH(), "cmd", timeout=10)
    emissions = []
    finished = []
    worker.line.connect(emissions.append)
    worker.finished_with_result.connect(lambda *args: finished.append(args))

    worker.run()

    assert finished and finished[0][0] == 0
    assert "".join(emissions) == "".join(f"line {index}\n" for index in range(1000))
    # One queued cross-thread signal per output line floods the GUI event
    # loop on a 10k-line install log; emissions must be batched.
    assert len(emissions) <= 25


def test_ssh_execute_worker_streaming_honors_interruption(qapp):
    import time

    class EndlessSSH:
        def __init__(self):
            self.closed = False

        def execute_stream(self, command, total_timeout=None):
            try:
                for _ in range(300):  # bounded so a failing run self-terminates
                    time.sleep(0.01)
                    yield "tick\n"
            finally:
                self.closed = True
            return 0

    ssh = EndlessSSH()
    worker = SshExecuteWorker(ssh, "endless", timeout=None)
    worker.start()
    time.sleep(0.05)

    worker.requestInterruption()

    assert worker.wait(2000), "worker did not stop promptly after requestInterruption()"
    assert ssh.closed


def test_streaming_worker_flushes_buffered_lines_on_failure(qapp):
    class FailingSSH:
        def execute_stream(self, command, total_timeout=None, should_abort=None):
            yield "important error context\n"
            raise RuntimeError("link down")

    worker = SshExecuteWorker(FailingSSH(), "cmd", timeout=5)
    lines = []
    failures = []
    worker.line.connect(lines.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert failures and "link down" in failures[0]
    # The buffered tail of the log is exactly what explains the failure.
    assert "important error context" in "".join(lines)


def test_safe_disconnect_tolerates_unconnected_signals_under_warning_errors(qapp):
    """PySide6 warns (SystemError under -W error) when disconnecting a signal
    with no receivers; cleanup must shrug that off while still detaching
    connected slots."""
    import warnings

    from PySide6.QtCore import QObject, Signal

    from openbench.gui.widgets._task_worker import safe_disconnect

    class Worker(QObject):
        connected = Signal(str)
        unconnected = Signal(str)

    received = []
    worker = Worker()
    worker.connected.connect(received.append)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the strictest caller environment
        safe_disconnect(worker.connected, worker.unconnected)

    worker.connected.emit("late")
    worker.unconnected.emit("late")
    assert received == []


def test_conda_create_task_aborts_between_steps_when_interrupted():
    from openbench.gui.widgets.remote_config import _build_conda_create_task

    commands = []

    class SSH:
        def execute(self, command, timeout=None):
            commands.append(command)
            return "ok", "", 0

        def detect_conda_envs(self):
            raise AssertionError("env detection must not run after interruption")

    flags = iter([False, True])  # survive the removal check, interrupt before create
    task = _build_conda_create_task(SSH(), "conda", "openbench", env_exists=True, interrupted=lambda: next(flags))

    result = task()

    assert result["exit_code"] != 0
    assert result["envs"] == []
    assert len(commands) == 1  # removal ran, creation was skipped
    assert "env remove" in commands[0]


def test_remote_config_main_thread_ssh_inventory():
    from pathlib import Path

    text = Path("src/openbench/gui/widgets/remote_config.py").read_text(encoding="utf-8")
    # The SSH handshake now runs via call_responsive (host-key prompt is
    # marshalled back to the GUI thread), so no repaint band-aids remain.
    assert text.count("QApplication.processEvents()") == 0
    # No widget method calls execute directly anymore; the only raw executes
    # live in _build_conda_create_task, which runs on a CallableWorker thread.
    assert text.count("self._ssh_manager.execute(") == 0
    assert text.count("self._ssh_manager.detect_conda_envs()") == 0
    assert text.count("self._ssh_manager.detect_python_interpreters()") == 0


def test_gui_modules_have_no_main_thread_ssh_band_aids():
    """processEvents pumps and raw SSH executes are banned outside known worker contexts."""
    from pathlib import Path

    gui_root = Path("src/openbench/gui")
    # _ssh_worker only MENTIONS the pattern in its docstring (it is the
    # replacement for it).
    process_events_allowlist = {"_ssh_worker.py"}
    # Raw executes allowed only where the call already runs off the GUI
    # thread or is the responsive layer itself.
    raw_execute_allowlist = {
        "_ssh_worker.py",
        "remote_config.py",
        "remote_runner.py",
        "remote_python.py",
        # RemoteFolderDownloadWorker owns its execute() call on a QThread; the
        # GUI slot only starts the worker and receives signals.
        "page_run_monitor.py",
    }

    offenders = []
    for path in sorted(gui_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name not in process_events_allowlist and "QApplication.processEvents()" in text:
            offenders.append(f"{path.name}: QApplication.processEvents()")
        if path.name not in raw_execute_allowlist:
            for line in text.splitlines():
                if ("ssh_manager.execute(" in line or "._ssh.execute(" in line) and "execute_responsive" not in line:
                    offenders.append(f"{path.name}: {line.strip()[:80]}")

    assert offenders == []


def test_env_actions_are_blocked_while_handshake_runs(qapp):
    """is_connected turns true mid-auth; env/install entry points must not
    open channels on a transport that is still authenticating."""
    widget = RemoteConfigWidget.__new__(RemoteConfigWidget)
    widget._handshake_active = True

    # Without a guard-first implementation these would blow up on the
    # missing widget attributes (none are set on this bare instance).
    widget._detect_python()
    widget._refresh_conda()
    widget._create_conda_env()
    widget._install_openbench()


def test_node_connect_is_blocked_while_main_handshake_runs(qapp):
    from types import SimpleNamespace

    calls = []
    widget = RemoteConfigWidget.__new__(RemoteConfigWidget)
    widget._ssh_manager = SimpleNamespace(is_connected=True, connect_with_jump=lambda **kwargs: calls.append(kwargs))
    widget._handshake_active = True  # main-server handshake in flight

    widget._confirm_node_connection()

    assert calls == []


@pytest.mark.parametrize("node_auth", ["password", "key"])
def test_main_connect_rejects_missing_selected_compute_credentials(qapp, monkeypatch, node_auth):
    from openbench.gui.widgets import remote_config

    widget = RemoteConfigWidget()
    widget.host_input.setText("alice@login.example")
    widget.radio_password.setChecked(True)
    widget.password_input.setText("main-secret")
    widget.node_group.setChecked(True)
    widget.node_input.setText("node001")
    if node_auth == "password":
        widget.radio_node_password.setChecked(True)
        widget.node_password_input.clear()
    else:
        widget.radio_node_key.setChecked(True)
        widget.node_key_input.clear()

    warnings = []
    monkeypatch.setattr(remote_config.QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(remote_config, "SSHManager", lambda *args, **kwargs: pytest.fail("SSH fallback attempted"))

    widget._test_connection()

    assert warnings and warnings[-1][1] == "Authentication Required"


def test_conda_env_change_discards_stale_query_result(qapp, monkeypatch):
    from types import SimpleNamespace

    from openbench.gui.widgets import remote_config

    class FakeCombo:
        def __init__(self, text, data):
            self._text = text
            self._data = data

        def currentText(self):
            return self._text

        def itemData(self, index):
            return self._data

    class FakePythonCombo:
        def __init__(self):
            self.current = ""
            self.items = []

        def findText(self, text):
            return -1

        def addItem(self, text):
            self.items.append(text)

        def setCurrentText(self, text):
            self.current = text

    widget = RemoteConfigWidget.__new__(RemoteConfigWidget)
    widget._ssh_manager = SimpleNamespace(is_connected=True)
    widget.conda_combo = FakeCombo("envA", "/envs/A")
    widget.python_combo = FakePythonCombo()

    def fake_exec(ssh_manager, command, timeout=None, should_abort=None):
        # Simulate the user picking another env while this query is in flight.
        widget._conda_env_sync_seq += 1
        return "/envs/A/bin/python\n", "", 0

    monkeypatch.setattr(remote_config, "execute_responsive", fake_exec)

    widget._on_conda_env_changed(1)

    # The superseded query must not apply its (now stale) result.
    assert widget.python_combo.current == ""


def test_create_conda_env_uses_guarded_dialog_and_blocks_reentry(qapp, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtCore import QObject, Signal

    from openbench.gui.widgets import remote_config

    created = []

    class FakeCallable:
        def __init__(self, func, parent=None):
            class Signals(QObject):
                finished_with_result = Signal(object)
                failed = Signal(str)
                finished = Signal()

            self._signals = Signals()
            self.finished_with_result = self._signals.finished_with_result
            self.failed = self._signals.failed
            self.finished = self._signals.finished
            created.append(self)

        def start(self):
            pass

        def isRunning(self):
            return True

        def deleteLater(self):
            pass

        def requestInterruption(self):
            pass

    monkeypatch.setattr(remote_config, "CallableWorker", FakeCallable)
    monkeypatch.setattr(remote_config.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(remote_config.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget = RemoteConfigWidget()
    widget._ssh_manager = SimpleNamespace(is_connected=True, detect_conda_envs=lambda: [])
    widget.python_combo.setCurrentText("/opt/miniconda3/bin/python")

    widget._create_conda_env()

    assert len(created) == 1
    dialog = widget.findChild(remote_config._InstallProgressDialog)
    assert dialog is not None and dialog.isVisible()

    dialog.reject()  # Esc while conda create runs
    assert dialog.isVisible(), "Esc must not close the conda-create dialog mid-run"

    widget._create_conda_env()  # re-entry while a worker is active
    assert len(created) == 1

    created[0].finished_with_result.emit({"exit_code": 1, "output": "boom", "envs": []})
    dialog.reject()
    assert not dialog.isVisible()


def test_call_on_gui_thread_marshals_from_worker(qapp):
    from PySide6.QtCore import QThread

    from openbench.gui.widgets._ssh_worker import call_on_gui_thread, call_responsive

    records = []

    def gui_work():
        records.append(QThread.currentThread())
        return "verdict"

    result = call_responsive(lambda: call_on_gui_thread(gui_work))

    assert result == "verdict"
    assert records == [qapp.thread()]


def test_call_on_gui_thread_reraises_exceptions(qapp):
    from openbench.gui.widgets._ssh_worker import call_on_gui_thread, call_responsive

    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        call_responsive(lambda: call_on_gui_thread(boom))


def test_test_connection_runs_handshake_off_gui_thread_with_marshalled_prompt(qapp, monkeypatch):
    from PySide6.QtCore import QThread

    from openbench.gui.widgets import remote_config

    connect_threads = []
    prompt_threads = []

    class FakeManager:
        def __init__(self, host_key_callback=None):
            self._host_key_callback = host_key_callback
            self.is_connected = True

        def connect(self, host, password=None, key_file=None):
            connect_threads.append(QThread.currentThread())
            # The handshake consults the host-key prompt mid-connect.
            assert self._host_key_callback("example.com", "ssh-ed25519", "ab:cd") is True

        def execute(self, command, timeout=None):
            return "8\n", "", 0  # cpu-count probe after connect

    def fake_question(parent, title, message, *args, **kwargs):
        prompt_threads.append(QThread.currentThread())
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(remote_config, "SSHManager", FakeManager)
    monkeypatch.setattr(remote_config.QMessageBox, "question", staticmethod(fake_question))
    monkeypatch.setattr(remote_config.QMessageBox, "information", staticmethod(lambda *args, **kwargs: None))

    widget = RemoteConfigWidget()
    widget.host_input.setText("example.com")
    widget.password_input.setText("pw")

    widget._test_connection()

    assert connect_threads
    assert all(thread != qapp.thread() for thread in connect_threads)
    # The QMessageBox prompt must have run on the GUI thread.
    assert prompt_threads == [qapp.thread()]


def test_detect_python_runs_probes_off_the_gui_thread(qapp, monkeypatch):
    from PySide6.QtCore import QThread

    threads = []

    class ProbeSSH:
        is_connected = True
        last_detection_errors = ()

        def execute(self, command, timeout=None):
            threads.append(QThread.currentThread())
            return "/usr/bin/python3\n", "", 0

        def detect_python_interpreters(self):
            threads.append(QThread.currentThread())
            return ["/usr/bin/python3"]

    class FakeCombo:
        def __init__(self):
            self.items = []

        def clear(self):
            self.items.clear()

        def addItem(self, text):
            self.items.append(text)

    class FakeButton:
        def setEnabled(self, value):
            self.enabled = value

    monkeypatch.setattr("openbench.gui.widgets.remote_config.QMessageBox.information", lambda *args, **kwargs: None)
    monkeypatch.setattr("openbench.gui.widgets.remote_config.QMessageBox.warning", lambda *args, **kwargs: None)

    widget = RemoteConfigWidget.__new__(RemoteConfigWidget)
    widget._ssh_manager = ProbeSSH()
    widget.python_combo = FakeCombo()
    widget.btn_detect_python = FakeButton()

    widget._detect_python()

    assert threads
    assert all(thread != qapp.thread() for thread in threads)
    assert widget.python_combo.items == ["/usr/bin/python3"]


def test_enabled_compute_node_actions_require_confirmed_target(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    class SSH:
        is_connected = True
        is_jump_connected = True

        def __init__(self):
            self.detect_calls = 0
            self.disconnect_jump_calls = 0

        def detect_python_interpreters(self):
            self.detect_calls += 1
            return ["/usr/bin/python3"]

        def disconnect_jump(self):
            self.is_jump_connected = False
            self.disconnect_jump_calls += 1

    warnings = []
    monkeypatch.setattr(remote_config.QMessageBox, "warning", lambda *args: warnings.append(args))

    widget = RemoteConfigWidget()
    ssh = SSH()
    widget._ssh_manager = ssh
    widget.node_group.setChecked(True)
    widget.node_input.setText("node001")
    widget._confirmed_node_config = widget._current_node_config()

    widget.node_input.setText("node002")
    widget._detect_python()

    assert widget.is_connected() is False
    assert ssh.disconnect_jump_calls == 1
    assert ssh.detect_calls == 0
    assert warnings


def test_install_path_probe_requires_exit_zero_and_exact_sentinel(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    class SSH:
        is_connected = True

        def __init__(self):
            self.execute_calls = []

        def execute(self, command, timeout=None):
            self.execute_calls.append((command, timeout))
            if command == "which git":
                return "/usr/bin/git\n", "", 0
            if command.startswith("test -d /remote/OpenBench"):
                return "not exists\n", "", 1
            raise AssertionError(command)

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = SSH()
    widget.openbench_input.setText("/remote/OpenBench")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    widget._install_openbench()

    assert len(FakeWorker.created) == 1
    assert (
        FakeWorker.created[0].command
        == "git clone --progress git@github.com:zhongwangwei/OpenBench.git /remote/OpenBench 2>&1"
    )
    assert all(".git" not in command for command, _timeout in widget._ssh_manager.execute_calls)


def test_git_probe_requires_exit_zero_and_exact_sentinel(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    class SSH:
        is_connected = True

        def __init__(self):
            self.execute_calls = []

        def execute(self, command, timeout=None):
            self.execute_calls.append((command, timeout))
            if command == "which git":
                return "/usr/bin/git\n", "", 0
            if command.startswith("test -d /remote/OpenBench/.git"):
                return "not is_git\n", "", 1
            if command.startswith("test -d /remote/OpenBench"):
                return "exists\n", "", 0
            if command.startswith("rm -rf /remote/OpenBench"):
                return "", "", 0
            raise AssertionError(command)

    questions = []

    def fake_question(*args, **kwargs):
        questions.append(args[2])
        return QMessageBox.No

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = SSH()
    widget.openbench_input.setText("/remote/OpenBench")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QMessageBox, "question", fake_question)

    widget._install_openbench()

    assert questions and "is NOT a git repository" in questions[-1]
    assert FakeWorker.created == []


class GuardedInstallSSH:
    is_connected = True

    def __init__(self):
        self.execute_calls = []

    def execute(self, command, timeout=None):
        self.execute_calls.append((command, timeout))
        if command == "which git":
            return "/usr/bin/git\n", "", 0
        if command.startswith("test -d /remote/OpenBench/.git") or command.startswith(
            "test -d '/remote/OpenBench/.git'"
        ):
            return "is_git\n", "", 0
        if command.startswith("test -d /remote/OpenBench") or command.startswith("test -d '/remote/OpenBench'"):
            return "exists\n", "", 0
        if command.startswith("test -f /remote/OpenBench/requirements.yml") or command.startswith(
            "test -f '/remote/OpenBench/requirements.yml'"
        ):
            return "exists\n", "", 0
        raise AssertionError(f"long install command executed synchronously: {command}")


class FakeWorker:
    created = []

    def __init__(self, ssh_manager, command, timeout=None, parent=None):
        from PySide6.QtCore import QObject, Signal

        class Signals(QObject):
            line = Signal(str)
            finished_with_result = Signal(int, str, str)
            failed = Signal(str)
            finished = Signal()

        self.signals = Signals()
        self.line = self.signals.line
        self.finished_with_result = self.signals.finished_with_result
        self.failed = self.signals.failed
        self.finished = self.signals.finished
        self.ssh_manager = ssh_manager
        self.command = command
        self.timeout = timeout
        self.parent = parent
        self.started = False
        FakeWorker.created.append(self)

    def start(self):
        self.started = True

    def deleteLater(self):
        pass

    def isRunning(self):
        return self.started

    def requestInterruption(self):
        pass


def test_install_openbench_update_uses_ssh_execute_worker_not_blocking_execute(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = GuardedInstallSSH()
    widget.openbench_input.setText("/remote/OpenBench")
    widget.python_combo.setCurrentText("/opt/miniconda/bin/python")
    widget.conda_combo.setCurrentText("base")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    widget._install_openbench()

    assert len(FakeWorker.created) == 1
    worker = FakeWorker.created[0]
    assert worker.ssh_manager is widget._ssh_manager
    assert worker.command == "cd /remote/OpenBench && git pull --ff-only 2>&1"
    assert worker.timeout == 300
    assert worker.started is True
    assert all("git pull" not in command for command, _timeout in widget._ssh_manager.execute_calls)
    # A parented QThread is destroyed with the widget even while running
    # (Qt fatals with "QThread: Destroyed while thread is still running"),
    # which defeats the detach-on-close machinery. The worker must be
    # unparented like the conda-create worker.
    assert worker.parent is None


def test_install_progress_dialog_ignores_escape_while_worker_runs(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = GuardedInstallSSH()
    widget.openbench_input.setText("/remote/OpenBench")
    widget.python_combo.setCurrentText("/opt/miniconda/bin/python")
    widget.conda_combo.setCurrentText("base")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    widget._install_openbench()
    dialog = widget.findChild(remote_config._InstallProgressDialog)
    assert dialog is not None
    assert dialog.isVisible()

    dialog.reject()  # Esc while the 300-900s worker is still running

    assert dialog.isVisible(), "Esc must not close the progress dialog mid-install"

    FakeWorker.created[0].finished_with_result.emit(1, "", "")  # install finished (failed)
    dialog.reject()

    assert not dialog.isVisible()


def test_install_openbench_installs_package_with_pip_as_second_worker(qapp, monkeypatch):
    """The repo has no requirements.yml; deps come from pyproject via pip install -e."""
    from openbench.gui.widgets import remote_config

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = GuardedInstallSSH()
    widget.openbench_input.setText("/remote/OpenBench")
    widget.python_combo.setCurrentText("/opt/miniconda/bin/python")
    widget.conda_combo.setCurrentText("base")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    widget._install_openbench()
    FakeWorker.created[0].finished_with_result.emit(0, "git ok", "")

    assert len(FakeWorker.created) == 2
    pip_worker = FakeWorker.created[1]
    assert pip_worker.command == "/opt/miniconda/bin/python -m pip install -e /remote/OpenBench 2>&1"
    assert pip_worker.timeout == 900
    assert pip_worker.started is True
    assert all("pip install" not in command for command, _timeout in widget._ssh_manager.execute_calls)
    # The requirements.yml probe round-trip is gone too.
    assert all("test -f" not in command for command, _timeout in widget._ssh_manager.execute_calls)


def test_install_pip_step_expands_tilde_python_path(qapp, monkeypatch):
    """'~/miniconda3/...' is the common interpreter form; a shlex-quoted
    literal tilde makes the remote shell look for a '~' directory."""
    from openbench.gui.widgets import remote_config

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = GuardedInstallSSH()
    widget.openbench_input.setText("/remote/OpenBench")
    widget.python_combo.setCurrentText("~/miniconda3/envs/ob/bin/python")
    widget.conda_combo.setCurrentText("base")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    widget._install_openbench()
    FakeWorker.created[0].finished_with_result.emit(0, "git ok", "")

    pip_worker = FakeWorker.created[1]
    assert pip_worker.command == '"$HOME"/miniconda3/envs/ob/bin/python -m pip install -e /remote/OpenBench 2>&1'
    assert "'~/" not in pip_worker.command


def test_conda_create_commands_expand_tilde_conda_exe(qapp, monkeypatch):
    """A '~/miniconda3/...' interpreter derives a '~/miniconda3/bin/conda'
    exe; quoting it literally would break both conda commands remotely."""
    from PySide6.QtCore import QObject, Signal

    from openbench.gui.widgets import remote_config

    created = []

    class FakeCallable:
        def __init__(self, func, parent=None):
            class Signals(QObject):
                finished_with_result = Signal(object)
                failed = Signal(str)
                finished = Signal()

            self._signals = Signals()
            self.finished_with_result = self._signals.finished_with_result
            self.failed = self._signals.failed
            self.finished = self._signals.finished
            self.func = func
            created.append(self)

        def start(self):
            pass

        def isRunning(self):
            return True

        def deleteLater(self):
            pass

        def requestInterruption(self):
            pass

        def isInterruptionRequested(self):
            return False

    class RecordingSSH:
        is_connected = True

        def __init__(self):
            self.commands = []

        def detect_conda_envs(self):
            return []

        def execute(self, command, timeout=None):
            self.commands.append(command)
            return "ok", "", 0

    monkeypatch.setattr(remote_config, "CallableWorker", FakeCallable)
    monkeypatch.setattr(remote_config.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(remote_config.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget = RemoteConfigWidget()
    ssh = RecordingSSH()
    widget._ssh_manager = ssh
    widget.python_combo.setCurrentText("~/miniconda3/bin/python")

    widget._create_conda_env()

    assert created
    created[0].func()  # run the captured task against the recording manager

    assert ssh.commands
    assert ssh.commands[0].startswith('"$HOME"/miniconda3/bin/conda create')
    assert "'~/" not in ssh.commands[0]


def test_install_openbench_skips_dependency_step_without_python_env(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    FakeWorker.created.clear()
    widget = RemoteConfigWidget()
    widget._ssh_manager = GuardedInstallSSH()
    widget.openbench_input.setText("/remote/OpenBench")
    widget.python_combo.setCurrentText("")

    monkeypatch.setattr(remote_config, "SshExecuteWorker", FakeWorker, raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    widget._install_openbench()
    FakeWorker.created[0].finished_with_result.emit(0, "git ok", "")

    assert len(FakeWorker.created) == 1  # no pip worker without a configured interpreter
    # And no leftover requirements.yml probe either.
    assert all("test -f" not in command for command, _timeout in widget._ssh_manager.execute_calls)


def test_local_install_worker_stop_terminates_silent_process(qapp):
    """A hung git process (no output) must die on stop(), not on the next line."""
    import sys
    import time

    from openbench.gui.pages.page_runtime import _LocalInstallWorker

    worker = _LocalInstallWorker([sys.executable, "-c", "import time; time.sleep(60)"])
    worker.start()
    time.sleep(0.3)  # let Popen start and block on readline

    worker.stop()

    assert worker.wait(5000), "worker did not exit promptly after stop()"


def test_runtime_local_install_cleanup_disconnects_and_detaches(qapp):
    from PySide6.QtCore import QObject, Signal

    from openbench.gui.pages import page_runtime
    from openbench.gui.pages.page_runtime import PageRuntime

    class FakeWorker(QObject):
        line = Signal(str)
        finished_with_result = Signal(int)
        failed = Signal(str)
        finished = Signal()

        def __init__(self):
            super().__init__()
            self.stopped = False

        def isRunning(self):
            return True

        def stop(self):
            self.stopped = True

    received = []
    worker = FakeWorker()
    worker.line.connect(received.append)

    page = PageRuntime.__new__(PageRuntime)
    page._local_install_worker = worker

    page._cleanup_local_install_worker(detach=True)

    worker.line.emit("late output")
    assert received == []  # UI closures were disconnected
    assert worker.stopped
    assert worker in page_runtime._DETACHED_INSTALL_WORKERS
    assert page._local_install_worker is None
    page_runtime._DETACHED_INSTALL_WORKERS.remove(worker)


def test_remote_config_roundtrips_compute_node_key_without_saved_credentials(qapp):
    widget = RemoteConfigWidget()

    widget.set_config(
        {
            "host": "alice@login.example.org",
            "auth_type": "key",
            "key_file": "/keys/login",
            "use_jump": True,
            "jump_node": "node001",
            "jump_auth": "key",
            "node_key_file": "/keys/node",
            "num_cores": 12,
            "python_path": "/opt/python/bin/python",
            "openbench_path": "/remote/OpenBench",
        }
    )

    config = widget.get_config()
    assert widget.key_input.text() == "/keys/login"
    assert widget.node_group.isChecked() is True
    assert widget.node_input.text() == "node001"
    assert widget.radio_node_key.isChecked() is True
    assert widget.node_key_input.text() == "/keys/node"
    assert config["jump_auth"] == "key"
    assert config["node_key_file"] == "/keys/node"


def test_test_connection_passes_compute_node_key_to_jump_connect(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    calls = []

    class FakeManager:
        def __init__(self, host_key_callback=None):
            self.is_connected = True

        def connect(self, host, password=None, key_file=None):
            calls.append(("connect", host, password, key_file))

        def connect_with_jump(self, **kwargs):
            calls.append(("jump", kwargs))

        def execute(self, command, timeout=None):
            return "banner\n64\n", "", 0

    monkeypatch.setattr(remote_config, "SSHManager", FakeManager)
    monkeypatch.setattr(remote_config.QMessageBox, "information", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(remote_config.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))

    widget = RemoteConfigWidget()
    widget.host_input.setText("alice@login")
    widget.password_input.setText("login-pw")
    widget.node_group.setChecked(True)
    widget.node_input.setText("node001")
    widget.radio_node_key.setChecked(True)
    widget.node_key_input.setText("/keys/node")

    widget._test_connection()

    assert ("jump", {"main_host": "node001", "main_password": None, "main_key_file": "/keys/node"}) in calls


def test_save_current_credentials_includes_compute_node_secret(qapp):
    saved = []

    class Creds:
        def save_credential(self, **kwargs):
            saved.append(kwargs)

    widget = RemoteConfigWidget()
    widget._credential_manager = Creds()
    widget.host_input.setText("alice@login")
    widget.radio_key.setChecked(True)
    widget.key_input.setText("/keys/login")
    widget.node_group.setChecked(True)
    widget.node_input.setText("node001")
    widget.radio_node_key.setChecked(True)
    widget.node_key_input.setText("/keys/node")

    widget._save_current_credentials()

    assert saved == [
        {
            "host": "alice@login",
            "auth_type": "key",
            "password": None,
            "key_file": "/keys/login",
            "jump_node": "node001",
            "jump_auth": "key",
            "node_key_file": "/keys/node",
        }
    ]


def test_load_saved_credentials_restores_compute_node_key(qapp):
    class Creds:
        def get_credential(self, host):
            return {
                "auth_type": "password",
                "password": "login-pw",
                "jump_node": "node001",
                "jump_auth": "key",
                "node_key_file": "/keys/node",
            }

    widget = RemoteConfigWidget()
    widget._credential_manager = Creds()

    widget._load_saved_credentials("alice@login")

    assert widget.password_input.text() == "login-pw"
    assert widget.node_group.isChecked() is True
    assert widget.node_input.text() == "node001"
    assert widget.radio_node_key.isChecked() is True
    assert widget.node_key_input.text() == "/keys/node"


def test_reset_to_defaults_keeps_compute_node_auth_none(qapp):
    widget = RemoteConfigWidget()
    widget.radio_node_password.setChecked(True)

    widget.reset_to_defaults()

    assert widget.radio_node_none.isChecked() is True


def test_update_remote_cpu_count_ignores_banner_lines(qapp, monkeypatch):
    from openbench.gui.widgets import remote_config

    class SSH:
        is_connected = True

    def fake_execute(_ssh, command, timeout=None):
        return "Welcome to cluster\n64\n", "", 0

    monkeypatch.setattr(remote_config, "execute_responsive", fake_execute)
    widget = RemoteConfigWidget()
    widget._ssh_manager = SSH()

    widget._update_remote_cpu_count()

    assert widget.cpu_available_label.text() == "(Available on remote: 64)"
    assert widget.num_cores_spin.maximum() >= 128


def test_parse_ssh_config_splits_multiple_host_aliases(tmp_path, monkeypatch):
    from openbench.gui.widgets import remote_config

    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text(
        "Host login login-short *.internal\n"
        "  HostName login.example.org\n"
        "  User alice\n"
        "  Port 2222\n"
        "  IdentityFile ~/.ssh/id_login\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(remote_config.platform, "system", lambda: "Darwin")

    hosts = remote_config.parse_ssh_config()

    assert [host["name"] for host in hosts] == ["login", "login-short"]
    assert {host["hostname"] for host in hosts} == {"login.example.org"}
    assert {host["user"] for host in hosts} == {"alice"}
    assert {host["port"] for host in hosts} == {"2222"}
