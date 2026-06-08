import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    monkeypatch.setattr("openbench.gui.runner.subprocess.Popen", lambda *args, **kwargs: process)
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
    assert "Command: /fake/python -m openbench run" in finished[-1][1]
