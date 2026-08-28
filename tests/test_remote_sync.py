import shlex
import threading

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


def test_glob_allows_relative_pattern_with_spaces():
    ssh = FakeSSH()
    sync = SyncEngine(ssh, "/remote/project")

    sync.glob("nml/my case/**/*.yaml")

    assert r"for f in nml/my\ case/**/*.yaml;" in ssh.commands[-1]


def test_exists_requires_exact_success_sentinel():
    class NoisyMissingSSH(FakeSSH):
        def execute(self, command, timeout=30):
            self.commands.append(command)
            return "not exists\n", "", 1

    sync = SyncEngine(NoisyMissingSSH(), "/remote/project")

    assert sync.exists("nml/main.yaml") is False


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


class CallbackRemoteFile(FakeRemoteFile):
    def write(self, data):
        super().write(data)
        callback = self.ssh.write_callback
        if callback is not None:
            self.ssh.write_callback = None
            callback()


class CallbackSFTP(FakeSFTP):
    def open(self, path, mode):
        assert mode == "wb"
        return CallbackRemoteFile(self.ssh, path)


class CallbackSSH(FakeSSH):
    def __init__(self):
        super().__init__()
        self.write_callback = None

    def execute(self, command, timeout=30):
        self.commands.append(command)
        if command.startswith("rm -f "):
            path = shlex.split(command)[2]
            self.files.pop(path, None)
        return "", "", 0

    def open_sftp(self):
        return CallbackSFTP(self)


class ReadCallbackRemoteFile:
    def __init__(self, ssh, path):
        self.ssh = ssh
        self.path = path

    def read(self):
        data = self.ssh.files[self.path]
        callback = self.ssh.read_callback
        if callback is not None:
            self.ssh.read_callback = None
            callback()
        return data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ReadCallbackSFTP:
    def __init__(self, ssh):
        self.ssh = ssh

    def open(self, path, mode):
        assert mode == "rb"
        return ReadCallbackRemoteFile(self.ssh, path)


class ReadCallbackSSH(FakeSSH):
    def __init__(self):
        super().__init__()
        self.read_callback = None

    def open_sftp(self):
        return ReadCallbackSFTP(self)


def test_read_does_not_overwrite_a_write_that_happens_during_remote_fetch():
    ssh = ReadCallbackSSH()
    ssh.files["/remote/project/notes.txt"] = b"old"
    sync = SyncEngine(ssh, "/remote/project")
    ssh.read_callback = lambda: sync.write("notes.txt", "new")

    assert sync.read("notes.txt") == "new"
    assert sync.get_sync_status("notes.txt") is SyncStatus.PENDING
    assert sync.get_pending_count() == 1


def test_read_rejects_remote_result_when_engine_freezes_during_fetch():
    ssh = ReadCallbackSSH()
    ssh.files["/remote/project/notes.txt"] = b"old"
    sync = SyncEngine(ssh, "/remote/project")
    ssh.read_callback = sync.freeze

    with pytest.raises(RuntimeError, match="sync engine is frozen"):
        sync.read("notes.txt")

    assert "notes.txt" not in sync._cache
    assert "notes.txt" not in sync._fetching


def test_waiting_read_rechecks_frozen_guard_before_returning_cache():
    sync = SyncEngine(FakeSSH(), "/remote/project")
    path = "notes.txt"
    entered = threading.Event()
    errors = []
    original_guard = sync._ensure_remote_io_allowed

    def guarded():
        original_guard()
        if threading.current_thread() is waiter:
            entered.set()

    def read():
        try:
            sync.read(path)
        except Exception as exc:
            errors.append(exc)

    sync._ensure_remote_io_allowed = guarded
    sync._fetching.add(path)
    waiter = threading.Thread(target=read)
    waiter.start()
    assert entered.wait(timeout=1)
    sync.freeze()
    with sync._lock:
        sync._cache[path] = "old"
        sync._fetching.discard(path)
        sync._fetch_cv.notify_all()
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert len(errors) == 1
    assert "sync engine is frozen" in str(errors[0])


def test_sync_file_keeps_new_write_pending_when_uploading_old_snapshot():
    ssh = CallbackSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("notes.txt", "old")
    ssh.write_callback = lambda: sync.write("notes.txt", "new")

    assert sync._sync_file("notes.txt") is False

    assert ssh.files["/remote/project/notes.txt"] == b"old"
    assert sync.read("notes.txt") == "new"
    assert sync.get_sync_status("notes.txt") is SyncStatus.PENDING
    assert sync.get_pending_count() == 1


def test_sync_file_does_not_retry_or_mark_error_after_new_write_replaces_failed_snapshot():
    class FailingRemoteFile(CallbackRemoteFile):
        def write(self, data):
            super().write(data)
            raise OSError("stale upload failed")

    class FailingSFTP(CallbackSFTP):
        def open(self, path, mode):
            self.ssh.open_count += 1
            return FailingRemoteFile(self.ssh, path)

    ssh = CallbackSSH()
    ssh.open_count = 0
    ssh.open_sftp = lambda: FailingSFTP(ssh)
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("notes.txt", "old")
    ssh.write_callback = lambda: sync.write("notes.txt", "new")

    assert sync._sync_file("notes.txt") is False

    assert ssh.open_count == 1
    assert sync.read("notes.txt") == "new"
    assert sync.get_sync_status("notes.txt") is SyncStatus.PENDING
    assert sync.get_error_files() == {}


def test_sync_file_deletes_old_snapshot_when_file_deleted_during_upload():
    ssh = CallbackSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("notes.txt", "old")
    ssh.write_callback = lambda: sync.delete("notes.txt")

    assert sync._sync_file("notes.txt") is True

    assert "/remote/project/notes.txt" not in ssh.files
    assert sync.get_pending_count() == 0



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


def test_write_clears_stale_error_for_replaced_content():
    ssh = FailingMkdirSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("dir/notes.txt", "old")
    assert sync._sync_file("dir/notes.txt") is False
    assert sync.get_sync_status("dir/notes.txt") is SyncStatus.ERROR
    assert "dir/notes.txt" in sync.get_error_files()

    sync.write("dir/notes.txt", "new")

    assert sync.get_sync_status("dir/notes.txt") is SyncStatus.PENDING
    assert sync.get_error_files() == {}
    assert sync.get_overall_status() is SyncStatus.PENDING


def test_delete_clears_stale_error_after_remote_delete_succeeds():
    ssh = FailingMkdirSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("dir/notes.txt", "old")
    assert sync._sync_file("dir/notes.txt") is False
    assert sync.get_sync_status("dir/notes.txt") is SyncStatus.ERROR
    assert "dir/notes.txt" in sync.get_error_files()

    sync._ssh = FakeSSH()
    sync.delete("dir/notes.txt")

    assert sync.get_sync_status("dir/notes.txt") is SyncStatus.SYNCED
    assert sync.get_error_files() == {}
    assert sync.get_overall_status() is SyncStatus.SYNCED


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
        f"if [ -d {quoted} ] && [ ! -L {quoted} ]; then rm -rf {quoted}; else rm -f {quoted}; fi"
    ]


def test_delete_permission_failure_preserves_local_cache():
    class PermissionDeniedSSH(FakeSSH):
        def execute(self, command, timeout=30):
            self.commands.append(command)
            return "", "permission denied", 13

    ssh = PermissionDeniedSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.mark_synced("dir/file.txt", "cached")

    with pytest.raises(OSError, match="permission denied"):
        sync.delete("dir/file.txt")

    assert sync.read("dir/file.txt") == "cached"


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


def test_sync_engine_sftp_paths_expand_tilde_project_dir():
    """SFTP does not run through a shell, so SyncEngine must resolve
    '~/OpenBench' to an absolute remote path before read/write."""

    class RemoteFile:
        def __init__(self, sftp, path, mode):
            self.sftp = sftp
            self.path = path
            self.mode = mode
            self.parts = []

        def read(self):
            return b"cached: true\n"

        def write(self, data):
            self.parts.append(data)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None and "w" in self.mode:
                self.sftp.files[self.path] = b"".join(self.parts)
            return False

    class SFTP:
        def __init__(self):
            self.opens = []
            self.files = {}

        def open(self, path, mode):
            self.opens.append((path, mode))
            return RemoteFile(self, path, mode)

    class HomeSSH:
        def __init__(self):
            self.commands = []
            self.sftp = SFTP()

        def _get_home_dir(self):
            return "/home/openbench"

        def execute(self, command, timeout=None):
            self.commands.append(command)
            return "", "", 0

        def open_sftp(self):
            return self.sftp

    ssh = HomeSSH()
    sync = SyncEngine(ssh, "~/OpenBench")

    assert sync.read("nml/main.yaml") == "cached: true\n"
    sync.write("nml/ref.yaml", "ref: true\n")
    assert sync._sync_file("nml/ref.yaml") is True

    assert ssh.sftp.opens == [
        ("/home/openbench/OpenBench/nml/main.yaml", "rb"),
        ("/home/openbench/OpenBench/nml/ref.yaml", "wb"),
    ]
    assert all(not path.startswith("~") for path, _mode in ssh.sftp.opens)


def test_glob_raises_remote_diagnostics_on_nonzero_exit():
    class FailingGlobSSH(FakeSSH):
        def execute(self, command, timeout=30):
            self.commands.append(command)
            return "", "permission denied", 13

    sync = SyncEngine(FailingGlobSSH(), "/remote/project")

    with pytest.raises(IOError, match="permission denied"):
        sync.glob("nml/**/*.yaml")


def test_stop_background_sync_reports_thread_that_did_not_exit():
    class StuckThread:
        def join(self, timeout=None):
            assert timeout == 5

        def is_alive(self):
            return True

    sync = SyncEngine(FakeSSH(), "/remote/project")
    sync._sync_thread = StuckThread()

    assert sync.stop_background_sync() is False


class IdentityRemoteFile:
    def __init__(self, ssh, path, mode):
        self.ssh = ssh
        self.path = path
        self.mode = mode
        self.parts = []

    def read(self):
        return self.ssh.files[self.path]

    def write(self, data):
        self.parts.append(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and "w" in self.mode:
            self.ssh.files[self.path] = b"".join(self.parts)
        return False


class IdentitySFTP:
    def __init__(self, ssh):
        self.ssh = ssh
        self.opens = []

    def open(self, path, mode):
        self.opens.append((path, mode))
        return IdentityRemoteFile(self.ssh, path, mode)


class IdentitySSH(FakeSSH):
    def __init__(self, identity):
        super().__init__()
        self.identity = identity
        self.sftp = IdentitySFTP(self)

    def get_active_target_identity(self):
        return self.identity

    def open_sftp(self):
        return self.sftp


def test_sync_engine_refuses_cache_miss_read_after_target_switch_without_b_io():
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    b.files["/remote/project/nml/main.yaml"] = b"bad: true\n"
    sync = SyncEngine(a, "/remote/project")
    sync._ssh = b

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        sync.read("nml/main.yaml")

    assert b.commands == []
    assert b.sftp.opens == []


def test_sync_engine_refuses_cached_read_and_exists_after_target_switch_without_b_io():
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    sync = SyncEngine(a, "/remote/project")
    sync.mark_synced("nml/main.yaml", "safe: true\n")
    sync._ssh = b

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        sync.read("nml/main.yaml")
    with pytest.raises(RuntimeError, match="remote target identity changed"):
        sync.exists("nml/main.yaml")

    assert b.commands == []
    assert b.sftp.opens == []


def test_sync_engine_refuses_cached_read_while_frozen():
    sync = SyncEngine(FakeSSH(), "/remote/project")
    sync.mark_synced("nml/main.yaml", "safe: true\n")
    assert sync.freeze_if_synced() is True

    with pytest.raises(RuntimeError, match="sync engine is frozen"):
        sync.read("nml/main.yaml")


def test_sync_engine_refuses_sync_after_target_switch_without_retry_or_b_io():
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    sync = SyncEngine(a, "/remote/project")
    sync.write("nml/main.yaml", "safe: true\n")
    sync._ssh = b

    assert sync._sync_file("nml/main.yaml") is False

    assert sync.get_sync_status("nml/main.yaml") is SyncStatus.ERROR
    assert "remote target identity changed" in sync.get_error_files()["nml/main.yaml"]
    assert b.commands == []
    assert b.sftp.opens == []


class SwitchDuringWriteRemoteFile(IdentityRemoteFile):
    def write(self, data):
        super().write(data)
        self.ssh.identity = ("direct", "alice", "login-b", 22)


class SwitchDuringWriteSFTP(IdentitySFTP):
    def open(self, path, mode):
        self.opens.append((path, mode))
        return SwitchDuringWriteRemoteFile(self.ssh, path, mode)


def test_sync_file_rechecks_target_after_sftp_write_before_marking_synced():
    ssh = IdentitySSH(("direct", "alice", "login-a", 22))
    ssh.sftp = SwitchDuringWriteSFTP(ssh)
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("nml/main.yaml", "safe: true\n")

    assert sync._sync_file("nml/main.yaml") is False

    assert sync.get_sync_status("nml/main.yaml") is SyncStatus.ERROR
    assert "remote target identity changed" in sync.get_error_files()["nml/main.yaml"]
    assert sync.get_pending_count() == 1


def test_sync_engine_refuses_write_after_target_switch():
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    sync = SyncEngine(a, "/remote/project")
    sync._ssh = b

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        sync.write("nml/main.yaml", "bad: true\n")

    assert sync.get_pending_count() == 0


def test_sync_engine_refuses_mark_synced_after_target_switch():
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    sync = SyncEngine(a, "/remote/project")
    sync._ssh = b

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        sync.mark_synced("nml/main.yaml", "bad: true\n")

    assert sync.get_pending_count() == 0


def test_sync_engine_refuses_list_dir_after_target_switch_without_b_command():
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    sync = SyncEngine(a, "/remote/project")
    sync._ssh = b

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        sync.list_dir("nml")

    assert b.commands == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda sync: sync.exists("nml/main.yaml"),
        lambda sync: sync.glob("nml/*.yaml"),
        lambda sync: sync.mkdir("nml/new"),
        lambda sync: sync.delete("nml/main.yaml"),
    ],
)
def test_sync_engine_refuses_other_remote_io_after_target_switch(operation):
    a = IdentitySSH(("direct", "alice", "login-a", 22))
    b = IdentitySSH(("direct", "alice", "login-b", 22))
    sync = SyncEngine(a, "/remote/project")
    sync._ssh = b

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        operation(sync)

    assert b.commands == []
    assert b.sftp.opens == []


class SwitchingExecuteSSH(IdentitySSH):
    def __init__(self):
        super().__init__(("direct", "alice", "login-a", 22))

    def execute(self, command, timeout=30):
        self.commands.append(command)
        self.identity = ("direct", "alice", "login-b", 22)
        if command.startswith("ls -1"):
            return "main.yaml\n", "", 0
        if "test -e" in command:
            return "exists\n", "", 0
        if "bash -c" in command:
            return "nml/main.yaml\n", "", 0
        return "", "", 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda sync: sync.exists("nml/main.yaml"),
        lambda sync: sync.list_dir("nml"),
        lambda sync: sync.glob("nml/*.yaml"),
        lambda sync: sync.mkdir("nml/new"),
    ],
)
def test_remote_queries_recheck_target_after_execute_before_returning(operation):
    ssh = SwitchingExecuteSSH()
    sync = SyncEngine(ssh, "/remote/project")

    with pytest.raises(RuntimeError, match="remote target identity changed"):
        operation(sync)

    assert ssh.commands


def test_delete_rechecks_frozen_after_execute_before_clearing_cache():
    class FreezingDeleteSSH(FakeSSH):
        def execute(inner_self, command, timeout=30):
            inner_self.commands.append(command)
            sync.freeze()
            return "", "", 0

    ssh = FreezingDeleteSSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.mark_synced("nml/main.yaml", "safe: true\n")

    with pytest.raises(RuntimeError, match="sync engine is frozen"):
        sync.delete("nml/main.yaml")

    sync.thaw()
    assert sync.read("nml/main.yaml") == "safe: true\n"


def test_freeze_if_synced_atomically_blocks_write_until_thawed():
    sync = SyncEngine(FakeSSH(), "/remote/project")
    sync.write("nml/main.yaml", "pending: true\n")

    assert sync.freeze_if_synced() is False
    sync.mark_synced("nml/main.yaml", "pending: false\n")
    assert sync.freeze_if_synced() is True
    with pytest.raises(RuntimeError, match="sync engine is frozen"):
        sync.write("nml/other.yaml", "nope\n")

    sync.thaw()
    sync.write("nml/other.yaml", "ok\n")
    assert sync.get_sync_status("nml/other.yaml") is SyncStatus.PENDING


def test_frozen_engine_with_no_pending_changes_is_already_synced():
    sync = SyncEngine(FakeSSH(), "/remote/project")

    assert sync.freeze_if_synced() is True
    assert sync.sync_all() is True


def test_rebind_same_identity_allows_pending_flush():
    identity = ("jump", "alice", "login", 22, "node001", 22)
    a1 = IdentitySSH(identity)
    a2 = IdentitySSH(identity)
    sync = SyncEngine(a1, "/remote/project")
    sync.write("nml/main.yaml", "ok: true\n")
    assert sync.freeze_if_synced() is False
    sync.freeze()

    sync.rebind_ssh(a2)

    assert sync.sync_all() is True
    assert a2.files["/remote/project/nml/main.yaml"] == b"ok: true\n"


def test_bound_target_reports_offline_without_losing_identity():
    identity = ("jump", "alice", "login", 22, "node001", 22)
    ssh = IdentitySSH(identity)
    sync = SyncEngine(ssh, "/remote/project")

    assert sync.is_bound_target_active() is True
    ssh.identity = None

    assert sync.is_bound_target_active() is False
