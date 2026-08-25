import sys
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QLabel, QProgressBar  # noqa: E402

from openbench.gui.widgets.progress_dashboard import ProgressDashboard, TaskStatus  # noqa: E402
from openbench.gui.widgets.validation_dialog import ValidationProgressDialog  # noqa: E402


def test_set_progress_clamps_bar_and_label(qapp):
    dashboard = ProgressDashboard()

    dashboard.set_progress(125)
    assert dashboard.progress_bar.value() == 1000
    assert dashboard.progress_label.text() == "100%"

    dashboard.set_progress(-7)
    assert dashboard.progress_bar.value() == 0
    assert dashboard.progress_label.text() == "0%"


def test_running_task_status_does_not_roll_back_numeric_progress(qapp):
    dashboard = ProgressDashboard()
    dashboard.set_tasks(["tas - Evaluation"])
    dashboard.set_progress(42)

    dashboard.update_task_status("tas - Evaluation", TaskStatus.RUNNING)

    assert dashboard.progress_bar.value() == 420
    assert dashboard.progress_label.text() == "42%"


def test_completed_task_status_can_advance_progress_when_higher(qapp):
    dashboard = ProgressDashboard()
    dashboard.set_tasks(["tas - Evaluation", "tas - Comparison"])
    dashboard.set_progress(10)

    dashboard.update_task_status("tas - Evaluation", TaskStatus.COMPLETED)

    assert dashboard.progress_bar.value() == 500
    assert dashboard.progress_label.text() == "50%"


def test_set_progress_preserves_decimal_precision_for_large_runs(qapp):
    dashboard = ProgressDashboard()

    dashboard.set_progress(5.09)

    assert dashboard.progress_bar.value() == 51
    assert dashboard.progress_label.text() == "5.1%"


def test_resource_usage_updates_from_psutil(qapp, monkeypatch):
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda: 37.9,
        virtual_memory=lambda: SimpleNamespace(percent=62.1),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    dashboard = ProgressDashboard()

    dashboard._update_resource_usage()

    assert dashboard.cpu_label.text() == "37%"
    assert dashboard.mem_label.text() == "62%"


def test_resource_usage_updates_from_local_process_tree(qapp, monkeypatch):
    class FakeProcess:
        def __init__(self, pid, cpu, rss, children=()):
            self.pid = pid
            self._cpu = cpu
            self._rss = rss
            self._children = list(children)

        def children(self, recursive=True):
            return self._children

        def cpu_percent(self, interval=None):
            return self._cpu

        def memory_info(self):
            return SimpleNamespace(rss=self._rss)

    child = FakeProcess(2, 7.6, 25)
    root = FakeProcess(1, 12.4, 75, [child])
    fake_psutil = SimpleNamespace(
        Process=lambda pid: root if pid == 1 else child,
        cpu_count=lambda: 4,
        virtual_memory=lambda: SimpleNamespace(total=1000),
        NoSuchProcess=RuntimeError,
        AccessDenied=PermissionError,
        ZombieProcess=ChildProcessError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    dashboard = ProgressDashboard()
    dashboard.monitor_process_tree(lambda: 1)

    dashboard._update_resource_usage()

    assert dashboard.cpu_label.text() == "5%"
    assert dashboard.mem_label.text() == "10%"


def test_remote_resource_mode_is_explicitly_unavailable(qapp, monkeypatch):
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda: 99,
        virtual_memory=lambda: SimpleNamespace(percent=99),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    dashboard = ProgressDashboard()
    dashboard.show_resource_unavailable("Resources (remote not monitored)")

    dashboard._update_resource_usage()

    assert dashboard.resource_group.title() == "Resources (remote not monitored)"
    assert dashboard.cpu_label.text() == "N/A"
    assert dashboard.mem_label.text() == "N/A"


def test_resource_usage_shows_unavailable_when_psutil_is_missing(qapp, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    dashboard = ProgressDashboard()

    dashboard._update_resource_usage()

    assert dashboard.cpu_label.text() == "N/A"
    assert dashboard.mem_label.text() == "N/A"


def _validation_progress_probe():
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    return SimpleNamespace(
        _closing=False,
        progress_bar=progress_bar,
        progress_label=QLabel(),
        current_label=QLabel(),
    )


def test_validation_progress_clamps_out_of_range_counts(qapp):
    dialog = _validation_progress_probe()

    ValidationProgressDialog._on_progress(dialog, 3, 2, "tas", "source")
    assert dialog.progress_bar.value() == 100
    assert dialog.progress_label.text() == "2/2"

    ValidationProgressDialog._on_progress(dialog, -1, 2, "tas", "source")
    assert dialog.progress_bar.value() == 0
    assert dialog.progress_label.text() == "0/2"


def test_validation_cancel_requests_cooperative_stop_without_waiting(qapp):
    class Worker:
        cancelled = False

        def isRunning(self):
            return True

        def cancel(self):
            self.cancelled = True

    worker = Worker()
    dialog = SimpleNamespace(
        _worker=worker,
        _cancel_requested=False,
        _closing=False,
        cancel_btn=SimpleNamespace(setEnabled=lambda enabled: setattr(dialog, "cancel_enabled", enabled)),
        progress_label=QLabel(),
    )

    ValidationProgressDialog._on_cancel(dialog)

    assert worker.cancelled is True
    assert dialog._cancel_requested is True
    assert dialog.cancel_enabled is False
    assert dialog.progress_label.text() == "Cancelling..."
