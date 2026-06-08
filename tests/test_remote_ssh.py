import select as select_module

import pytest

from openbench.remote import ssh as ssh_module
from openbench.remote.ssh import SSHConnectionError, SSHManager


class FakeSSHClient:
    instances = []

    def __init__(self, *, raise_on_connect=False):
        self.raise_on_connect = raise_on_connect
        self.connect_kwargs = None
        self.closed = False
        self.policy = None
        FakeSSHClient.instances.append(self)

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs
        if self.raise_on_connect:
            raise RuntimeError("connect failed")

    def close(self):
        self.closed = True

    def get_transport(self):
        return FakeTransport()


class FakeTransport:
    def __init__(self):
        self.channel = FakeChannel()

    def is_active(self):
        return True

    def open_channel(self, *args, **kwargs):
        return self.channel


class FakeChannel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class StreamChannel:
    def __init__(self):
        self.closed = False
        self.stdout = [b"first\n", b"second\n"]

    def exec_command(self, command):
        self.command = command

    def setblocking(self, value):
        self.blocking = value

    def exit_status_ready(self):
        return not self.stdout

    def recv_ready(self):
        return bool(self.stdout)

    def recv_stderr_ready(self):
        return False

    def recv(self, size):
        return self.stdout.pop(0)

    def recv_stderr(self, size):
        return b""

    def recv_exit_status(self):
        return 0

    def close(self):
        self.closed = True

    def fileno(self):
        return 0


class StreamTransport:
    def __init__(self):
        self.channel = StreamChannel()

    def open_session(self):
        return self.channel


def test_connect_forwards_private_key_passphrase(monkeypatch):
    FakeSSHClient.instances = []
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", FakeSSHClient)

    manager = SSHManager(auto_add_host_keys=True)
    manager.connect("alice@example.org", key_file="/keys/id_rsa", passphrase="key-passphrase")

    kwargs = FakeSSHClient.instances[0].connect_kwargs
    assert kwargs["key_filename"] == "/keys/id_rsa"
    assert kwargs["passphrase"] == "key-passphrase"


def test_jump_connection_failure_closes_partial_channel(monkeypatch):
    main_client = FakeSSHClient()
    manager = SSHManager(auto_add_host_keys=True)
    manager._client = main_client
    manager._user = "alice"
    channel = main_client.get_transport().channel

    class RaisingSSHClient(FakeSSHClient):
        def __init__(self):
            super().__init__(raise_on_connect=True)

    transport = FakeTransport()
    channel = transport.channel
    main_client.get_transport = lambda: transport
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", RaisingSSHClient)

    with pytest.raises(SSHConnectionError, match="Jump connection failed"):
        manager.connect_with_jump("node001", main_password="secret")

    assert channel.closed is True
    assert manager._jump_client is None
    assert manager._jump_channel is None


def test_reconnect_closes_existing_jump_connection(monkeypatch):
    FakeSSHClient.instances = []
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", FakeSSHClient)
    manager = SSHManager(auto_add_host_keys=True)
    jump_client = FakeSSHClient()
    jump_channel = FakeChannel()
    manager._jump_client = jump_client
    manager._jump_channel = jump_channel

    manager.connect("alice@example.org", password="secret")

    assert jump_client.closed is True
    assert jump_channel.closed is True
    assert manager._jump_client is None
    assert manager._jump_channel is None


def test_detect_python_interpreters_records_suppressed_method_errors():
    manager = SSHManager(auto_add_host_keys=True)
    manager._user = "alice"
    calls = []

    def fake_execute(command, timeout=None):
        calls.append(command)
        if command == "echo $HOME":
            return "/home/alice\n", "", 0
        if "miniconda" in command:
            raise RuntimeError("ls failed")
        if "which python3" in command:
            return "/home/alice/miniconda3/bin/python\n", "", 0
        return "", "", 1

    manager.execute = fake_execute

    assert manager.detect_python_interpreters() == ["/home/alice/miniconda3/bin/python"]
    assert "Python discovery command failed" in manager.last_detection_errors[0]
    assert "ls failed" in manager.last_detection_errors[0]


class FakeSFTPDirs:
    def __init__(self):
        self.dirs = set()
        self.mkdir_calls = []

    def stat(self, path):
        if path not in self.dirs:
            raise FileNotFoundError(path)

    def mkdir(self, path):
        self.mkdir_calls.append(path)
        self.dirs.add(path)


def test_ensure_remote_dir_preserves_relative_paths():
    manager = SSHManager(auto_add_host_keys=True)
    sftp = FakeSFTPDirs()
    manager._get_sftp = lambda: sftp

    manager._ensure_remote_dir("relative/path")

    assert sftp.mkdir_calls == ["relative", "relative/path"]


def test_ensure_remote_dir_preserves_absolute_paths():
    manager = SSHManager(auto_add_host_keys=True)
    sftp = FakeSFTPDirs()
    manager._get_sftp = lambda: sftp

    manager._ensure_remote_dir("/remote/path")

    assert sftp.mkdir_calls == ["/remote", "/remote/path"]


def test_execute_stream_closes_channel_when_generator_is_closed(monkeypatch):
    manager = SSHManager(auto_add_host_keys=True)
    transport = StreamTransport()

    class Client:
        def get_transport(self):
            return transport

    manager._client = Client()
    monkeypatch.setattr(select_module, "select", lambda *args, **kwargs: ([], [], []))

    stream = manager.execute_stream("long command")
    assert next(stream) == "first\n"
    stream.close()

    assert transport.channel.closed is True
