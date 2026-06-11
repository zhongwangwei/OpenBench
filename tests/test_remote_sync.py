import shlex

import pytest

from openbench.remote.sync import SyncEngine, SyncStatus


class FakeSSH:
    def __init__(self):
        self.commands = []
        self.files = {}

    def execute(self, command, timeout=30):
        self.commands.append(command)
        return "", "", 0

    def open_sftp(self):
        return FakeSFTP(self)


class FakeRemoteFile:
    def __init__(self, ssh, path):
        self.ssh = ssh
        self.path = path
        self.parts = []

    def write(self, data):
        self.parts.append(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.ssh.files[self.path] = b"".join(self.parts)
        return False


class FakeSFTP:
    def __init__(self, ssh):
        self.ssh = ssh

    def open(self, path, mode):
        assert mode == "wb"
        return FakeRemoteFile(self.ssh, path)


def test_remote_path_rejects_escape_from_project_root():
    sync = SyncEngine(FakeSSH(), "/remote/project")

    with pytest.raises(ValueError, match="escapes remote project directory"):
        sync.read("../outside.txt")


def test_list_dir_quotes_remote_path_with_shell_metacharacters():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")
    path = "bad'; touch /tmp/openbench_pwn; echo '"

    sync.list_dir(path)

    # list_dir no longer redirects stderr to /dev/null — it surfaces the
    # stderr message when the call fails so callers can distinguish
    # "permission denied" / "no such directory" from a genuinely empty dir.
    remote_path = "/remote/project/" + path
    assert ssh.commands == [f"ls -1 {shlex.quote(remote_path)}"]


def test_glob_rejects_shell_metacharacters():
    sync = SyncEngine(FakeSSH(), "/remote/project")

    with pytest.raises(ValueError, match="unsafe glob pattern"):
        sync.glob("bad; touch /tmp/openbench_pwn")


def test_mark_synced_replaces_stale_pending_cache_without_remote_read():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("output/demo/openbench.yaml", "old")

    sync.mark_synced("output/demo/openbench.yaml", "new")

    assert sync.get_sync_status("output/demo/openbench.yaml") is SyncStatus.SYNCED
    assert sync.get_pending_count() == 0
    assert sync.read("output/demo/openbench.yaml") == "new"
    assert ssh.commands == []


def test_sync_file_uses_sftp_without_forcing_trailing_newline():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("notes.txt", "no trailing newline")

    assert sync._sync_file("notes.txt") is True

    assert ssh.commands[-1] == "mkdir -p /remote/project"
    assert ssh.files["/remote/project/notes.txt"] == b"no trailing newline"


def test_sync_file_uses_sftp_for_nul_bytes_and_large_content():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")
    content = ("x" * 140_000) + "\0tail"
    sync.write("notes.txt", content)

    assert sync._sync_file("notes.txt") is True

    assert ssh.files["/remote/project/notes.txt"] == content.encode("utf-8")


def test_remote_path_allows_relative_paths_when_project_root_is_filesystem_root():
    sync = SyncEngine(FakeSSH(), "/")

    assert sync._remote_path("notes.txt") == "/notes.txt"
    assert sync._remote_path("dir/notes.txt") == "/dir/notes.txt"


class FlakySSH(FakeSSH):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def execute(self, command, timeout=30):
        self.commands.append(command)
        return "", "", 0

    def open_sftp(self):
        if self.failures > 0:
            self.failures -= 1
            raise OSError("temporary")
        return super().open_sftp()


def test_sync_file_retries_transient_write_errors():
    ssh = FlakySSH(failures=2)
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("notes.txt", "content")

    assert sync._sync_file("notes.txt") is True
    assert sync.get_sync_status("notes.txt") is SyncStatus.SYNCED
    assert ssh.files["/remote/project/notes.txt"] == b"content"


class FailingMkdirSSH(FakeSSH):
    def execute(self, command, timeout=30):
        self.commands.append(command)
        if command.startswith("mkdir -p"):
            return "", "permission denied", 1
        return "", "", 0


def test_sync_file_fails_fast_when_remote_mkdir_fails():
    ssh = FailingMkdirSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("dir/notes.txt", "content")

    assert sync._sync_file("dir/notes.txt") is False
    assert sync.get_sync_status("dir/notes.txt") is SyncStatus.ERROR
    assert ssh.files == {}


def test_mkdir_reports_remote_failure():
    ssh = FailingMkdirSSH()
    sync = SyncEngine(ssh, "/remote/project")

    with pytest.raises(Exception, match="Create remote directory failed"):
        sync.mkdir("dir")


@pytest.mark.parametrize("path", ["", ".", "./"])
def test_delete_refuses_remote_project_root(path):
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")

    with pytest.raises(ValueError, match="Refusing to delete remote project root"):
        sync.delete(path)

    assert ssh.commands == []


def test_delete_refuses_normalized_remote_project_root():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")

    with pytest.raises(ValueError, match="Refusing to delete remote project root"):
        sync.delete("subdir/..")

    assert ssh.commands == []


def test_delete_uses_explicit_file_or_recursive_directory_command():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")

    sync.delete("dir/file.txt")

    quoted = shlex.quote("/remote/project/dir/file.txt")
    assert ssh.commands == [
        f"test -e {quoted}",
        f"if [ -d {quoted} ] && [ ! -L {quoted} ]; then rm -rf {quoted}; else rm -f {quoted}; fi",
    ]


def test_sync_engine_commands_expand_tilde_project_dir():
    """A '~/OpenBench' remote project dir must reach the shell as "$HOME",
    not a shlex-quoted literal tilde, across the SyncEngine command surface."""
    from openbench.remote.sync import SyncEngine

    commands = []

    class TildeSSH:
        is_connected = True

        def execute(self, command, timeout=None):
            commands.append(command)
            return "", "", 0

        def read_file(self, path):
            return ""

        def write_file(self, path, content):
            return None

    engine = SyncEngine(TildeSSH(), "~/OpenBench")
    engine.mkdir("nml")
    engine.exists("nml/main.yaml")
    engine.list_dir("nml")

    joined = "\n".join(commands)
    assert commands
    assert '"$HOME"/OpenBench' in joined
    assert "'~/" not in joined
