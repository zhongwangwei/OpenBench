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
    assert dashboard.progress_bar.value() == 100
    assert dashboard.progress_label.text() == "100%"

    dashboard.set_progress(-7)
    assert dashboard.progress_bar.value() == 0
    assert dashboard.progress_label.text() == "0%"


def test_running_task_status_does_not_roll_back_numeric_progress(qapp):
    dashboard = ProgressDashboard()
    dashboard.set_tasks(["tas - Evaluation"])
    dashboard.set_progress(42)

    dashboard.update_task_status("tas - Evaluation", TaskStatus.RUNNING)

    assert dashboard.progress_bar.value() == 42
    assert dashboard.progress_label.text() == "42%"


def test_completed_task_status_can_advance_progress_when_higher(qapp):
    dashboard = ProgressDashboard()
    dashboard.set_tasks(["tas - Evaluation", "tas - Comparison"])
    dashboard.set_progress(10)

    dashboard.update_task_status("tas - Evaluation", TaskStatus.COMPLETED)

    assert dashboard.progress_bar.value() == 50
    assert dashboard.progress_label.text() == "50%"


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
