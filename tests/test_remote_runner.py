import os

import pytest

pytest.importorskip("PySide6")

from openbench.gui.remote_runner import RemoteRunner  # noqa: E402


class FakeSSH:
    is_connected = True

    def __init__(self, fail_upload_names=None):
        self.uploads = []
        self.fail_upload_names = set(fail_upload_names or [])

    def upload_file(self, local_path, remote_path):
        self.uploads.append((local_path, remote_path))
        if os.path.basename(local_path) in self.fail_upload_names:
            raise RuntimeError(f"upload failed for {os.path.basename(local_path)}")

    def execute(self, command, timeout=30):
        return "", "", 0


def _runner(config_path, ssh):
    runner = RemoteRunner(
        str(config_path),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
    )
    runner._remote_temp_dir = "/tmp/openbench_test"
    return runner


def test_upload_config_fails_when_related_yaml_upload_fails(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("include: related.yaml\n", encoding="utf-8")
    related = tmp_path / "related.yaml"
    related.write_text("x: 1\n", encoding="utf-8")
    ssh = FakeSSH(fail_upload_names={"related.yaml"})
    runner = _runner(config, ssh)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    assert runner._upload_config() is False

    assert finished == [(False, "Failed to upload config file: upload failed for related.yaml")]
    assert ssh.uploads == [
        (str(config), "/tmp/openbench_test/main.yaml"),
        (str(related), "/tmp/openbench_test/related.yaml"),
    ]


def test_upload_config_uploads_related_files_successfully(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("include: related.yaml\n", encoding="utf-8")
    (tmp_path / "related.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored\n", encoding="utf-8")
    ssh = FakeSSH()
    runner = _runner(config, ssh)

    assert runner._upload_config() is True

    assert ssh.uploads == [
        (str(config), "/tmp/openbench_test/main.yaml"),
        (str(tmp_path / "related.yaml"), "/tmp/openbench_test/related.yaml"),
    ]


class ExecuteSSH(FakeSSH):
    def __init__(self, execute_response):
        super().__init__()
        self.execute_response = execute_response
        self.commands = []

    def execute(self, command, timeout=30):
        self.commands.append(command)
        return self.execute_response


def test_create_remote_temp_dir_uses_mktemp_unique_path(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("/tmp/openbench_wizard_abcd1234\n", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
    )

    assert runner._create_remote_temp_dir() is True

    assert runner._remote_temp_dir == "/tmp/openbench_wizard_abcd1234"
    assert ssh.commands == ["mktemp -d /tmp/openbench_wizard_XXXXXXXXXX"]


def test_kill_remote_process_matches_current_config_not_all_openbench_runs(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "", 0))
    runner = _runner(config, ssh)
    runner._remote_config_path = "/tmp/openbench_wizard_abcd1234/openbench.yaml"

    runner._kill_remote_process()

    assert len(ssh.commands) == 1
    assert "pkill -f --" in ssh.commands[0]
    assert "openbench_wizard_abcd1234" in ssh.commands[0]
    assert "python.*-m openbench run" in ssh.commands[0]
    assert ssh.commands[0] != "pkill -f 'python.*-m openbench run' || true"


def test_cleanup_remote_reports_nonzero_rm_failure(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "permission denied", 1))
    runner = _runner(config, ssh)
    logs = []
    runner.log_message.connect(logs.append)

    runner._cleanup_remote()

    assert logs == ["Warning: Could not clean up remote directory /tmp/openbench_test: permission denied"]


class StreamSSH(FakeSSH):
    def __init__(self, lines=(), exit_code=0, exc=None):
        super().__init__()
        self.lines = list(lines)
        self.exit_code = exit_code
        self.exc = exc
        self.stream_command = None

    def execute_stream(self, command):
        self.stream_command = command
        if self.exc:
            raise self.exc
        yield from self.lines
        return self.exit_code


def test_execute_remote_openbench_nonzero_exit_includes_log_tail(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = StreamSSH(lines=[f"line {i}\n" for i in range(1, 8)], exit_code=3)
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"

    success, message = runner._execute_remote_openbench()

    assert success is False
    assert "Remote OpenBench exited with code 3" in message
    assert "Recent output:" in message
    assert "line 3" in message
    assert "line 7" in message
    assert "line 1" not in message


def test_execute_remote_openbench_preserves_partial_marker_outside_tail(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = StreamSSH(
        lines=[
            "Running evaluation: case\n",
            "✗ Evaluation completed with errors\n",
            *[f"detail {i}\n" for i in range(1, 8)],
        ],
        exit_code=1,
    )
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"

    success, message = runner._execute_remote_openbench()

    assert success is False
    assert "Evaluation completed with errors" in message
    assert "Remote OpenBench exited with code 1" in message


def test_execute_remote_openbench_stream_exception_includes_command_context(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = StreamSSH(exc=RuntimeError("stream broke"))
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"

    success, message = runner._execute_remote_openbench()

    assert success is False
    assert "Execution error while running remote command" in message
    assert "stream broke" in message
    assert "python3 -u -m openbench run" in message


class CloseFailStream:
    def __iter__(self):
        return self

    def __next__(self):
        return "running\n"

    def close(self):
        raise RuntimeError("close failed")


class CloseFailSSH(FakeSSH):
    def execute_stream(self, command):
        return CloseFailStream()

    def execute(self, command, timeout=30):
        return "", "", 0


def test_execute_remote_openbench_logs_stream_close_failure_on_stop(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    runner = _runner(config, CloseFailSSH())
    runner._remote_config_path = "/remote/main.yaml"
    runner.stop()
    logs = []
    runner.log_message.connect(logs.append)

    success, message = runner._execute_remote_openbench()

    assert success is False
    assert message == "Stopped by user"
    assert "Warning: could not close remote output stream: close failed" in logs
    assert "Sent kill signal to remote process" in logs


def test_remote_runner_progress_parser_ignores_exception_source_names(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    runner = _runner(config, FakeSSH())
    runner._current_ref = "GLEAM"
    runner._current_sim = "CoLM"

    progress, var, stage = runner._parse_progress("ReferenceError: variable missing; simulation traceback", 42)

    assert progress == 42
    assert var == ""
    assert stage == ""
    assert runner._current_ref == "GLEAM"
    assert runner._current_sim == "CoLM"
