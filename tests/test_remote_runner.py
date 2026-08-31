import base64
import os

import pytest

pytest.importorskip("PySide6")

from openbench.gui.remote_runner import RemoteRunner, build_remote_run_command  # noqa: E402


class FakeSSH:
    is_connected = True

    def __init__(self, fail_upload_names=None):
        self.uploads = []
        self.commands = []
        self.fail_upload_names = set(fail_upload_names or [])

    def upload_file(self, local_path, remote_path):
        self.uploads.append((local_path, remote_path))
        if os.path.basename(local_path) in self.fail_upload_names:
            raise RuntimeError(f"upload failed for {os.path.basename(local_path)}")

    def execute(self, command, timeout=30, should_abort=None):
        self.commands.append(command)
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
    config.write_text("include: !include related.yaml\n", encoding="utf-8")
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
    config.write_text("include: !include related.yaml\n", encoding="utf-8")
    (tmp_path / "related.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "secret.json").write_text('{"token": "ignored"}\n', encoding="utf-8")
    (tmp_path / "unused.yaml").write_text("ignored: true\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored\n", encoding="utf-8")
    ssh = FakeSSH()
    runner = _runner(config, ssh)

    assert runner._upload_config() is True

    assert ssh.uploads == [
        (str(config), "/tmp/openbench_test/main.yaml"),
        (str(tmp_path / "related.yaml"), "/tmp/openbench_test/related.yaml"),
    ]


def test_upload_config_preserves_nested_include_paths(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("include: !include nml/related.yaml\n", encoding="utf-8")
    nml = tmp_path / "nml"
    nml.mkdir()
    related = nml / "related.yaml"
    related.write_text("x: 1\n", encoding="utf-8")
    ssh = FakeSSH()
    runner = _runner(config, ssh)

    assert runner._upload_config() is True

    assert ssh.commands == ["mkdir -p /tmp/openbench_test/nml"]
    assert ssh.uploads == [
        (str(config), "/tmp/openbench_test/main.yaml"),
        (str(related.resolve()), "/tmp/openbench_test/nml/related.yaml"),
    ]


def test_upload_config_rejects_include_outside_config_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_INCLUDE_ROOTS", str(tmp_path))
    outside = tmp_path / "shared.yaml"
    outside.write_text("x: 1\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "main.yaml"
    config.write_text("include: !include ../shared.yaml\n", encoding="utf-8")
    ssh = FakeSSH()
    runner = _runner(config, ssh)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    assert runner._upload_config() is False

    assert ssh.uploads == [(str(config), "/tmp/openbench_test/main.yaml")]
    assert finished and "outside the upload root" in finished[0][1]


class ExecuteSSH(FakeSSH):
    def __init__(self, execute_response):
        super().__init__()
        self.execute_response = execute_response
        self.commands = []

    def execute(self, command, timeout=30, should_abort=None):
        self.commands.append(command)
        return self.execute_response


class ExecuteKwargsSSH(FakeSSH):
    def __init__(self, execute_response):
        super().__init__()
        self.execute_response = execute_response
        self.execute_calls = []

    def execute(self, command, timeout=30, should_abort=None):
        self.commands.append(command)
        self.execute_calls.append({"command": command, "timeout": timeout, "should_abort": should_abort})
        return self.execute_response


def test_create_remote_temp_dir_uses_mktemp_unique_path(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("__OPENBENCH_TMP__=/tmp/openbench_wizard_abcd1234\n", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
    )

    assert runner._create_remote_temp_dir() is True

    assert runner._remote_temp_dir == "/tmp/openbench_wizard_abcd1234"
    expected = "tmp=$(mktemp -d /tmp/openbench_wizard_XXXXXXXXXX) && printf '__OPENBENCH_TMP__=%s\\n' \"$tmp\""
    assert ssh.commands == [expected]


def test_create_remote_temp_dir_ignores_login_banner_paths(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("Welcome\n/home/alice\n__OPENBENCH_TMP__=/tmp/openbench_wizard_abcd1234\n", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
    )

    assert runner._create_remote_temp_dir() is True

    assert runner._remote_temp_dir == "/tmp/openbench_wizard_abcd1234"


def test_create_remote_temp_dir_ignores_later_matching_unmarked_path(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(
        (
            "__OPENBENCH_TMP__=/tmp/openbench_wizard_abcd1234\n/tmp/openbench_wizard_other9999\n",
            "",
            0,
        )
    )
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
    )

    assert runner._create_remote_temp_dir() is True

    assert runner._remote_temp_dir == "/tmp/openbench_wizard_abcd1234"


def test_create_remote_temp_dir_rejects_no_absolute_path(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("Welcome only\n", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
    )
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    assert runner._create_remote_temp_dir() is False

    assert runner._remote_temp_dir == ""
    assert finished == [
        (False, "Failed to create remote temp directory: mktemp returned no safe /tmp/openbench_wizard_* path")
    ]


def test_remote_run_command_expands_tilde_config_path():
    command = build_remote_run_command(
        "~/miniconda3/bin/python",
        "~/OpenBench",
        "~/OpenBench/output/case/openbench.yaml",
        "",
    )

    assert '"$HOME"/OpenBench/output/case/openbench.yaml' in command
    assert "'~/OpenBench/output/case/openbench.yaml'" not in command


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
    assert "python.*-m openbench (check|run)" in ssh.commands[0]
    assert ssh.commands[0] != "pkill -f 'python.*-m openbench (check|run)' || true"


def test_kill_remote_process_matches_tilde_config_as_expanded_home(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "", 0))
    runner = _runner(config, ssh)
    runner._remote_config_path = "~/OpenBench/output/case/openbench.yaml"

    runner._kill_remote_process()

    assert len(ssh.commands) == 1
    assert "OpenBench/output/case/openbench\\.yaml" in ssh.commands[0]
    assert "/[^[:space:]]+/OpenBench/output/case/openbench\\.yaml" in ssh.commands[0]


def test_kill_remote_process_terminates_captured_process_group(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "", 0))
    runner = _runner(config, ssh)
    runner._remote_process_group = 4321

    runner._kill_remote_process()

    assert len(ssh.commands) == 1
    assert "kill -TERM -4321" in ssh.commands[0]
    assert "kill -KILL -4321" in ssh.commands[0]
    assert "pkill" not in ssh.commands[0]


def test_cleanup_remote_reports_nonzero_rm_failure(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "permission denied", 1))
    runner = _runner(config, ssh)
    runner._remote_temp_dir = "/tmp/openbench_wizard_abcd1234"
    logs = []
    runner.log_message.connect(logs.append)

    runner._cleanup_remote()

    assert logs == ["Warning: Could not clean up remote directory /tmp/openbench_wizard_abcd1234: permission denied"]


def test_cleanup_remote_refuses_untrusted_remote_path(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "", 0))
    runner = _runner(config, ssh)
    runner._remote_temp_dir = "/home/alice"
    logs = []
    runner.log_message.connect(logs.append)

    runner._cleanup_remote()

    assert ssh.commands == []
    assert logs == ["Warning: Refusing to clean unsafe remote directory: /home/alice"]


class StreamSSH(FakeSSH):
    def __init__(self, lines=(), exit_code=0, exc=None):
        super().__init__()
        self.lines = list(lines)
        self.exit_code = exit_code
        self.exc = exc
        self.stream_command = None
        self.stream_kwargs = {}

    def execute_stream(self, command, **kwargs):
        self.stream_command = command
        self.stream_kwargs = kwargs
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


def test_execute_remote_openbench_passes_should_abort_to_stream(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = StreamSSH(
        lines=[
            "__OPENBENCH_PHASE__=run_started\n",
            "running\n",
            "__OPENBENCH_PHASE__=run_completed\n",
        ],
        exit_code=0,
    )
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"

    success, _message = runner._execute_remote_openbench()

    assert success is True
    assert callable(ssh.stream_kwargs.get("should_abort"))


def test_execute_remote_openbench_captures_hidden_process_group_marker(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = StreamSSH(
        lines=[
            "login banner\n",
            "__OPENBENCH_PGID__=4321\n",
            "__OPENBENCH_PHASE__=run_started\n",
            "running\n",
            "__OPENBENCH_PHASE__=run_completed\n",
        ],
        exit_code=0,
    )
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"
    logs = []
    runner.log_message.connect(logs.append)

    success, _message = runner._execute_remote_openbench()

    assert success is True
    assert runner._remote_process_group == 4321
    assert "running" in logs
    assert logs[-1] == "Remote evaluation process completed."
    assert "login banner" in logs
    assert all("__OPENBENCH_PGID__" not in line for line in logs)
    assert "setsid" in ssh.stream_command


def test_execute_remote_openbench_rejects_unconfirmed_zero_exit(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = StreamSSH(lines=["Config valid. Ready to run.\n"], exit_code=0)
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"

    success, message = runner._execute_remote_openbench()

    assert success is False
    assert message == "Remote command exited before the OpenBench evaluation started"


def test_remote_resource_sample_is_normalized_to_configured_cores(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("240.0 1.5\n", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench", "num_cores": 12},
        config_already_remote=True,
    )
    runner._remote_process_group = 4321
    samples = []
    runner.resource_updated.connect(lambda cpu, memory: samples.append((cpu, memory)))

    assert runner._sample_remote_resources() is True

    assert samples == [(20.0, 1.5)]
    assert "ps -eo pgid=,pcpu=,rss=" in ssh.commands[-1]
    assert "target=4321" in ssh.commands[-1]


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
    def execute_stream(self, command, **kwargs):
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


def test_remote_runner_stop_is_not_reported_as_failed(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    runner = RemoteRunner(
        str(config),
        FakeSSH(),
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
        config_already_remote=True,
    )
    runner._last_progress = 17
    runner._execute_remote_openbench = lambda: (False, "Stopped by user")
    progress = []
    finished = []
    runner.progress_updated.connect(progress.append)
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert finished[-1] == (False, "Stopped by user")
    assert progress[-1].status.value == "stopped"
    assert progress[-1].progress != runner.PROGRESS_MAX


def test_remote_runner_hard_failure_keeps_current_progress(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    runner = RemoteRunner(
        str(config),
        FakeSSH(),
        {"python_path": "python3", "openbench_path": "/remote/openbench"},
        config_already_remote=True,
    )
    runner._last_progress = 17
    runner._execute_remote_openbench = lambda: (False, "boom")
    progress = []
    runner.progress_updated.connect(progress.append)

    runner.run()

    assert progress[-1].status.value == "failed"
    assert progress[-1].progress != runner.PROGRESS_MAX


def test_remote_runner_applies_remote_num_cores_before_execution(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench", "num_cores": 12},
        config_already_remote=True,
    )
    runner._remote_config_path = "/remote/main.yaml"
    logs = []
    runner.log_message.connect(logs.append)

    assert runner._apply_remote_num_cores_override() is True

    assert ssh.commands
    payload = ssh.commands[0].split()[2]
    assert 'project["num_cores"] = 12' in base64.b64decode(payload).decode()
    assert "Using 12 CPU cores on remote server." in logs


def test_remote_runner_num_cores_patch_expands_tilde_and_is_abortable(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteKwargsSSH(("", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "~/OpenBench", "num_cores": 12},
        config_already_remote=True,
    )
    runner._remote_config_path = "~/OpenBench/output/case/openbench.yaml"

    assert runner._apply_remote_num_cores_override() is True

    payload = ssh.commands[0].split()[2]
    script = base64.b64decode(payload).decode()
    assert ".expanduser()" in script
    assert '"~/OpenBench/output/case/openbench.yaml"' in script
    assert callable(ssh.execute_calls[0]["should_abort"])


def test_remote_runner_stop_during_num_cores_patch_does_not_start_openbench(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteKwargsSSH(("", "", 0))
    runner = RemoteRunner(
        str(config),
        ssh,
        {"python_path": "python3", "openbench_path": "/remote/openbench", "num_cores": 12},
        config_already_remote=True,
    )
    runner._remote_config_path = "/remote/main.yaml"

    def execute_then_stop(command, timeout=30, should_abort=None):
        ssh.commands.append(command)
        ssh.execute_calls.append({"command": command, "timeout": timeout, "should_abort": should_abort})
        runner.stop()
        return "", "", 0

    ssh.execute = execute_then_stop
    executed = []
    runner._execute_remote_openbench = lambda: executed.append(True) or (True, "Completed")

    runner.run()

    assert executed == []


def test_remote_runner_stop_does_not_synchronously_execute_ssh_kill(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    ssh = ExecuteSSH(("", "", 0))
    runner = _runner(config, ssh)
    runner._remote_config_path = "/remote/main.yaml"

    runner.stop()

    assert ssh.commands == []


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
