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


def test_jump_connection_failure_disconnects_main_to_avoid_login_node_fallback(monkeypatch):
    main_client = FakeSSHClient()
    manager = SSHManager(auto_add_host_keys=True)
    manager._client = main_client
    manager._user = "alice"

    class RaisingSSHClient(FakeSSHClient):
        def __init__(self):
            super().__init__(raise_on_connect=True)

    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", RaisingSSHClient)

    with pytest.raises(SSHConnectionError, match="Jump connection failed"):
        manager.connect_with_jump("node001", main_password="secret")

    assert main_client.closed is True
    assert manager.get_active_client() is None


def test_disconnected_requested_compute_node_never_falls_back_to_login_node():
    manager = SSHManager(auto_add_host_keys=True)
    manager._client = FakeSSHClient()
    manager._jump_required = True

    assert manager.get_active_client() is None


def test_detect_conda_envs_uses_last_absolute_path_from_noisy_login_shell():
    manager = SSHManager(auto_add_host_keys=True)
    calls = []

    def fake_execute(command, timeout=None):
        calls.append(command)
        if command == "echo $HOME":
            return "/home/alice\n", "", 0
        if "ls -d" in command:
            return "", "", 1
        if "which conda" in command:
            return "Welcome\n/home/alice/miniconda3/bin/conda\n", "", 0
        if command == "/home/alice/miniconda3/bin/conda env list":
            return (
                "# conda environments:\n"
                "base * /home/alice/miniconda3\n"
                "openbench /home/alice/miniconda3/envs/openbench\n",
                "",
                0,
            )
        return "", "", 1

    manager.execute = fake_execute

    assert manager.detect_conda_envs() == [
        ("base", "/home/alice/miniconda3"),
        ("openbench", "/home/alice/miniconda3/envs/openbench"),
    ]


def test_open_sftp_proxy_serializes_cached_client_operations():
    import threading
    import time

    class UnsafeSFTP:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def put(self, local, remote):
            current = self.active + 1
            self.active = current
            self.max_active = max(self.max_active, current)
            time.sleep(0.01)
            self.active -= 1

    sftp = UnsafeSFTP()
    manager = SSHManager(auto_add_host_keys=True)
    manager._get_sftp = lambda: sftp
    proxy = manager.open_sftp()

    threads = [threading.Thread(target=proxy.put, args=(str(i), str(i))) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sftp.max_active == 1


def test_parse_host_string_supports_ipv6_forms():
    manager = SSHManager(auto_add_host_keys=True)

    assert manager._parse_host_string("alice@[2001:db8::1]:2222") == ("alice", "2001:db8::1", 2222)
    assert manager._parse_host_string("alice@2001:db8::1") == ("alice", "2001:db8::1", 22)
    assert manager._parse_host_string("alice@example.org:2200") == ("alice", "example.org", 2200)


def test_sftp_operations_expand_tilde_remote_paths(tmp_path):
    class SFTP(FakeSFTPDirs):
        def __init__(self):
            super().__init__()
            self.put_calls = []

        def put(self, local, remote):
            self.put_calls.append((local, remote))

    manager = SSHManager(auto_add_host_keys=True)
    sftp = SFTP()
    manager._get_sftp = lambda: sftp
    manager._get_home_dir = lambda: "/home/alice"
    local_file = tmp_path / "openbench.yaml"
    local_file.write_text("project: {}\n", encoding="utf-8")

    manager.upload_file(str(local_file), "~/OpenBench/output/openbench.yaml")

    assert sftp.mkdir_calls == ["/home", "/home/alice", "/home/alice/OpenBench", "/home/alice/OpenBench/output"]
    assert sftp.put_calls == [(str(local_file), "/home/alice/OpenBench/output/openbench.yaml")]


def test_download_file_expands_tilde_remote_path(tmp_path):
    class SFTP:
        def __init__(self):
            self.get_calls = []

        def get(self, remote, local):
            self.get_calls.append((remote, local))

    manager = SSHManager(auto_add_host_keys=True)
    sftp = SFTP()
    manager._get_sftp = lambda: sftp
    manager._get_home_dir = lambda: "/home/alice"
    local_file = tmp_path / "out" / "openbench.yaml"

    manager.download_file("~/OpenBench/output/openbench.yaml", str(local_file))

    assert sftp.get_calls == [("/home/alice/OpenBench/output/openbench.yaml", str(local_file))]
    assert local_file.parent.is_dir()


def test_active_target_identity_for_direct_connection(monkeypatch):
    FakeSSHClient.instances = []
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", FakeSSHClient)
    manager = SSHManager(auto_add_host_keys=True)

    manager.connect("alice@login.example:2200", password="secret")

    assert manager.get_active_target_identity() == ("direct", "alice", "login.example", 2200)


def test_active_target_identity_for_jump_and_same_reconnect(monkeypatch):
    FakeSSHClient.instances = []
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", FakeSSHClient)
    manager = SSHManager(auto_add_host_keys=True)
    manager.connect("alice@login.example", password="secret")

    manager.connect_with_jump("node001", main_password="secret")
    first = manager.get_active_target_identity()
    manager.disconnect_jump()

    assert manager.get_active_target_identity() is None
    manager.connect_with_jump("node001", main_password="secret")

    assert first == ("jump", "alice", "login.example", 22, "node001", 22)
    assert manager.get_active_target_identity() == first


def test_select_main_target_explicitly_restores_login_execution(monkeypatch):
    FakeSSHClient.instances = []
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", FakeSSHClient)
    manager = SSHManager(auto_add_host_keys=True)
    manager.connect("alice@login.example", password="secret")
    manager.connect_with_jump("node001", main_password="secret")

    manager.select_main_target()

    assert manager.get_active_target_identity() == ("direct", "alice", "login.example", 22)
    assert manager.get_active_client() is manager._client


def test_remote_home_query_ignores_banner_and_is_cached_per_target():
    manager = SSHManager(auto_add_host_keys=True)
    identity = ("direct", "alice", "login.example", 22)
    calls = []
    manager._user = "alice"
    manager.get_active_target_identity = lambda: identity

    def execute(command, timeout=None):
        calls.append((command, timeout))
        return "Welcome to the cluster\n/home/alice\n", "", 0

    manager.execute = execute

    assert manager._get_home_dir() == "/home/alice"
    assert manager._get_home_dir() == "/home/alice"
    assert calls == [("echo $HOME", 5)]
