import pytest

pytest.importorskip("PySide6")

from openbench.gui.runner import EvaluationRunner, RunnerStatus  # noqa: E402


class FakeStdout:
    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        return ""

    def read(self):
        return ""


class FakeProcess:
    def __init__(self, lines, return_code):
        self.stdout = FakeStdout(lines)
        self.return_code = return_code
        self.killed = False
        self.terminated = False

    def poll(self):
        return self.return_code if not self.lines_left() else None

    def lines_left(self):
        return bool(self.stdout.lines)

    def wait(self):
        return self.return_code

    def kill(self):
        self.killed = True

    def terminate(self):
        self.terminated = True


def _runner(tmp_path, monkeypatch, process):
    config = tmp_path / "openbench.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    runner = EvaluationRunner(str(config), python_path="/fake/python")
    monkeypatch.setattr(runner, "_find_python_interpreter", lambda: "/fake/python")
    processes = iter([FakeProcess(["check passed\n"], 0), process])
    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", lambda *args, **kwargs: next(processes))
    return runner


def test_local_runner_nonzero_exit_includes_recent_output_tail(tmp_path, monkeypatch):
    process = FakeProcess([f"line {i}\n" for i in range(1, 8)], 2)
    runner = _runner(tmp_path, monkeypatch, process)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert finished[-1][0] is False
    message = finished[-1][1]
    assert "Process exited with code 2" in message
    assert "Recent output:" in message
    assert "line 3" in message
    assert "line 7" in message
    assert "line 1" not in message


def test_local_runner_partial_exit_emits_partial_status(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            "Running evaluation: case\n",
            "✓ Runoff completed\n",
            "✗ Evaluation completed with errors\n",
            "  - [evaluation] ET failed\n",
            "  - detail 1\n",
            "  - detail 2\n",
            "  - detail 3\n",
            "  - detail 4\n",
            "  - detail 5\n",
            "  - detail 6\n",
        ],
        1,
    )
    runner = _runner(tmp_path, monkeypatch, process)
    progress = []
    finished = []
    runner.progress_updated.connect(lambda update: progress.append(update))
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert finished[-1][0] is False
    assert "Process exited with code 1" in finished[-1][1]
    assert "Evaluation completed with errors" in finished[-1][1]
    assert progress[-1].status is RunnerStatus.PARTIAL
    assert progress[-1].current_task == "Partial"


def test_local_runner_popen_exception_includes_command_context(tmp_path, monkeypatch):
    config = tmp_path / "openbench.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    runner = EvaluationRunner(str(config), python_path="/fake/python")
    monkeypatch.setattr(runner, "_find_python_interpreter", lambda: "/fake/python")

    def raise_popen(*args, **kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", raise_popen)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert finished[-1][0] is False
    assert "Local execution error: spawn failed" in finished[-1][1]
    assert "Command: /fake/python -m openbench check" in finished[-1][1]


def test_local_runner_checks_before_run(tmp_path, monkeypatch):
    config = tmp_path / "openbench.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    runner = EvaluationRunner(str(config), python_path="/fake/python")
    monkeypatch.setattr(runner, "_find_python_interpreter", lambda: "/fake/python")
    commands = []
    processes = iter([FakeProcess(["Ready to run\n"], 0), FakeProcess(["Evaluation complete\n"], 0)])

    def fake_popen(command, **kwargs):
        commands.append(command)
        return next(processes)

    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", fake_popen)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert [command[3] for command in commands] == ["check", "run"]
    assert finished[-1][0] is True


def test_local_runner_resets_progress_between_check_and_run(tmp_path, monkeypatch):
    config = tmp_path / "openbench.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    runner = EvaluationRunner(str(config), python_path="/fake/python")
    runner.set_task_counts(1, 1, 1, 1, 0, 0, 0)
    monkeypatch.setattr(runner, "_find_python_interpreter", lambda: "/fake/python")
    processes = iter([FakeProcess(["Config validation\n"], 0), FakeProcess(["Processing Runoff\n"], 0)])
    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", lambda *args, **kwargs: next(processes))
    progress = []
    runner.progress_updated.connect(progress.append)

    runner.run()

    run_start = next(item for item in progress if item.message == "Starting OpenBench evaluation...")
    assert run_start.progress == 0


def test_local_runner_counts_groupby_without_comparison(tmp_path):
    runner = EvaluationRunner(str(tmp_path / "openbench.yaml"), python_path="/fake/python")

    runner.set_task_counts(
        num_variables=2,
        num_ref_sources=1,
        num_sim_sources=1,
        num_metrics=1,
        num_scores=1,
        num_groupby=1,
        num_comparisons=0,
        do_evaluation=True,
        do_comparison=False,
        do_statistics=False,
    )

    assert runner._total_tasks == 3


def test_remote_runner_counts_groupby_without_comparison(tmp_path):
    from openbench.gui.remote_runner import RemoteRunner

    runner = RemoteRunner(str(tmp_path / "openbench.yaml"), object(), {}, config_already_remote=True)

    runner.set_task_counts(2, 1, 1, 1, 1, 1, 0, do_evaluation=True, do_comparison=False)

    assert runner._total_tasks == 3


def test_local_runner_does_not_run_when_check_fails(tmp_path, monkeypatch):
    config = tmp_path / "openbench.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    runner = EvaluationRunner(str(config), python_path="/fake/python")
    monkeypatch.setattr(runner, "_find_python_interpreter", lambda: "/fake/python")
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return FakeProcess(
            [
                "Config validation:\n",
                "  ✓ YAML syntax valid\n",
                "  ✗ Reference years do not overlap project years\n",
                "Reference data (1 sources):\n",
                "Simulation data (1 models):\n",
                "Options:\n",
                "  Time alignment: intersection\n",
                "✗ Config has errors. Please fix and re-check.\n",
            ],
            1,
        )

    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", fake_popen)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert [command[3] for command in commands] == ["check"]
    assert finished[-1][0] is False
    assert "Configuration check failed" in finished[-1][1]
    assert "years do not overlap" in finished[-1][1]


def test_local_runner_stop_after_check_does_not_start_run(tmp_path, monkeypatch):
    config = tmp_path / "openbench.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    runner = EvaluationRunner(str(config), python_path="/fake/python")
    monkeypatch.setattr(runner, "_find_python_interpreter", lambda: "/fake/python")
    commands = []

    class StopAfterCheck(FakeProcess):
        def wait(self):
            with runner._stop_lock:
                runner._stop_requested = True
            return super().wait()

    def fake_popen(command, **kwargs):
        commands.append(command)
        return StopAfterCheck(["Ready to run\n"], 0)

    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", fake_popen)
    finished = []
    runner.finished_signal.connect(lambda success, message: finished.append((success, message)))

    runner.run()

    assert [command[3] for command in commands] == ["check"]
    assert finished[-1] == (False, "Stopped by user")
