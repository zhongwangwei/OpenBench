"""Unit tests for SSHManager.execute_stream chunk/decoding behavior."""

import os
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("paramiko")

from openbench.remote.ssh import SSHManager  # noqa: E402


class FakeChannel:
    """Feeds pre-cut byte chunks through the execute_stream recv loop."""

    def __init__(self, stdout_chunks, stderr_chunks=(), exit_code=0):
        self._stdout = list(stdout_chunks)
        self._stderr = list(stderr_chunks)
        self._exit = exit_code
        self._r, self._w = os.pipe()
        os.write(self._w, b"x")  # keep select() instantly ready

    def fileno(self):
        return self._r

    def exec_command(self, command):
        self.command = command

    def setblocking(self, flag):
        pass

    def exit_status_ready(self):
        return True

    def recv_ready(self):
        return bool(self._stdout)

    def recv_stderr_ready(self):
        return bool(self._stderr)

    def recv(self, _size):
        return self._stdout.pop(0)

    def recv_stderr(self, _size):
        return self._stderr.pop(0)

    def recv_exit_status(self):
        return self._exit

    def close(self):
        for fd in (self._r, self._w):
            try:
                os.close(fd)
            except OSError:
                pass


def _manager_with_channel(channel):
    manager = SSHManager.__new__(SSHManager)
    manager._state_lock = threading.Lock()
    client = SimpleNamespace(get_transport=lambda: SimpleNamespace(open_session=lambda: channel))
    manager.get_active_client = lambda: client
    return manager


def test_execute_stream_reassembles_multibyte_chars_across_chunks():
    payload = "温度数据\n".encode("utf-8")
    channel = FakeChannel([payload[:5], payload[5:]])  # cut mid-character

    lines = list(_manager_with_channel(channel).execute_stream("cmd"))

    assert lines == ["温度数据\n"]


def test_execute_stream_yields_only_complete_lines_across_chunks():
    channel = FakeChannel([b"first part ", b"of line\nsecond\n"])

    lines = list(_manager_with_channel(channel).execute_stream("cmd"))

    assert lines == ["first part of line\n", "second\n"]


def test_execute_stream_flushes_trailing_partial_line():
    channel = FakeChannel([b"no newline at end"])

    lines = list(_manager_with_channel(channel).execute_stream("cmd"))

    assert lines == ["no newline at end"]


def test_execute_stream_wraps_command_for_posix_sh():
    import shlex

    channel = FakeChannel([b"hi\n"])

    list(_manager_with_channel(channel).execute_stream("echo $X && true"))

    assert channel.command == "sh -c " + shlex.quote("echo $X && true")


def test_execute_wraps_command_for_posix_sh():
    import shlex

    channel = FakeChannel([b"hi\n"])

    class FakeExecClient:
        def __init__(self):
            self.commands = []

        def exec_command(self, command, timeout=None):
            self.commands.append(command)
            stdin = SimpleNamespace(close=lambda: None)
            stdout = SimpleNamespace(channel=channel, close=lambda: None)
            stderr = SimpleNamespace(close=lambda: None)
            return stdin, stdout, stderr

    client = FakeExecClient()
    manager = SSHManager.__new__(SSHManager)
    manager._state_lock = threading.Lock()
    manager._timeout = 30
    manager.get_active_client = lambda: client

    stdout, stderr, exit_code = manager.execute("echo $X && true")

    assert (stdout, exit_code) == ("hi\n", 0)
    assert client.commands == ["sh -c " + shlex.quote("echo $X && true")]


def test_execute_should_abort_stops_silent_command():
    from openbench.remote.ssh import SSHConnectionError

    class SilentChannel(FakeChannel):
        def __init__(self):
            super().__init__([])

        def exit_status_ready(self):
            return False

    channel = SilentChannel()

    class FakeExecClient:
        def exec_command(self, command, timeout=None):
            stdin = SimpleNamespace(close=lambda: None)
            stdout = SimpleNamespace(channel=channel, close=lambda: None)
            stderr = SimpleNamespace(close=lambda: None)
            return stdin, stdout, stderr

    manager = SSHManager.__new__(SSHManager)
    manager._state_lock = threading.Lock()
    manager._timeout = 30
    manager.get_active_client = lambda: FakeExecClient()

    probes = []

    def should_abort():
        probes.append(1)
        return len(probes) >= 2

    with pytest.raises(SSHConnectionError, match="abort"):
        manager.execute("hang", should_abort=should_abort)


def test_execute_stream_should_abort_stops_silent_command():
    from openbench.remote.ssh import SSHConnectionError

    class SilentChannel(FakeChannel):
        def __init__(self):
            super().__init__([])

        def exit_status_ready(self):
            return False

    probes = []

    def should_abort():
        probes.append(1)
        return len(probes) >= 2

    stream = _manager_with_channel(SilentChannel()).execute_stream("hang", should_abort=should_abort)

    with pytest.raises(SSHConnectionError, match="abort"):
        list(stream)


def test_ssh_execute_worker_passes_interruption_probe_to_stream():
    pytest.importorskip("PySide6")
    from openbench.gui.widgets._ssh_worker import SshExecuteWorker

    captured = {}

    class SSH:
        def execute_stream(self, command, total_timeout=None, should_abort=None):
            captured["should_abort"] = should_abort
            return iter(())

    worker = SshExecuteWorker(SSH(), "cmd", timeout=5)
    worker.run()

    assert callable(captured.get("should_abort"))


def test_execute_stream_returns_exit_code():
    channel = FakeChannel([b"done\n"], exit_code=3)
    stream = _manager_with_channel(channel).execute_stream("cmd")

    collected = []
    while True:
        try:
            collected.append(next(stream))
        except StopIteration as done:
            assert done.value == 3
            break

    assert collected == ["done\n"]
