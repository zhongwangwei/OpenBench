# -*- coding: utf-8 -*-
"""
Progress dashboard widget for displaying evaluation progress.
"""

from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QGroupBox,
)
from PySide6.QtCore import Signal, QTimer


class TaskStatus(Enum):
    """Task status enum."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Task information."""

    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""


class ProgressDashboard(QWidget):
    """Dashboard widget for displaying evaluation progress."""

    stop_requested = Signal()
    open_output_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: List[TaskInfo] = []
        self._progress_value = 0.0
        self._resource_mode = "system"
        self._resource_pid_provider: Optional[Callable[[], int | None]] = None
        self._setup_ui()

        # Timer for updating resource usage
        self._resource_timer = QTimer(self)
        self._resource_timer.timeout.connect(self._update_resource_usage)

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # === Progress & Resources in one row ===
        top_row = QHBoxLayout()
        top_row.setSpacing(15)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(5)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)
        progress_layout.addWidget(self.progress_bar)

        # Progress info
        progress_info = QHBoxLayout()
        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        progress_info.addWidget(self.progress_label)
        progress_info.addStretch()
        self.eta_label = QLabel("ETA: --")
        self.eta_label.setStyleSheet("color: #666;")
        progress_info.addWidget(self.eta_label)
        progress_layout.addLayout(progress_info)

        top_row.addWidget(progress_group, 3)

        # Resource usage section
        self.resource_group = QGroupBox("Resources")
        resource_layout = QHBoxLayout(self.resource_group)
        resource_layout.setSpacing(15)

        # CPU
        resource_layout.addWidget(QLabel("CPU:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximum(100)
        self.cpu_bar.setFixedWidth(80)
        resource_layout.addWidget(self.cpu_bar)
        self.cpu_label = QLabel("0%")
        self.cpu_label.setFixedWidth(65)
        resource_layout.addWidget(self.cpu_label)

        # Memory
        resource_layout.addWidget(QLabel("Mem:"))
        self.mem_bar = QProgressBar()
        self.mem_bar.setMaximum(100)
        self.mem_bar.setFixedWidth(80)
        resource_layout.addWidget(self.mem_bar)
        self.mem_label = QLabel("0%")
        self.mem_label.setFixedWidth(65)
        resource_layout.addWidget(self.mem_label)

        top_row.addWidget(self.resource_group, 2)

        layout.addLayout(top_row)

        # === Log Output (expanded to fill available space) ===
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 5, 5, 5)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(10000)  # Increased from 1000 to handle longer runs
        self.log_output.setMinimumHeight(200)  # Ensure minimum visible area
        log_layout.addWidget(self.log_output, 1)

        # Give log_group higher stretch factor to maximize vertical space
        layout.addWidget(log_group, 3)

        # === Control Buttons ===
        btn_layout = QHBoxLayout()

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()

        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.clicked.connect(self.open_output_requested.emit)
        btn_layout.addWidget(self.btn_open_output)

        layout.addLayout(btn_layout)

    def set_tasks(self, tasks: List[str]):
        """Set the task list."""
        self._tasks = [TaskInfo(name=t) for t in tasks]

    def update_task_status(self, task_name: str, status: TaskStatus, message: str = ""):
        """Update status of a specific task."""
        for task in self._tasks:
            if task.name == task_name:
                task.status = status
                task.message = message
                break
        self._update_overall_progress()

    def set_progress(self, value: float, eta_seconds: Optional[int] = None):
        """Set overall progress with one-decimal precision for large task sets."""
        value = max(0.0, min(100.0, float(value)))
        self._progress_value = value
        self.progress_bar.setValue(int(round(value * 10)))
        if abs(value - round(value)) < 0.05:
            label = f"{int(round(value))}%"
        else:
            label = f"{value:.1f}%"
        self.progress_label.setText(label)

        if eta_seconds is not None:
            if eta_seconds < 60:
                eta_str = f"{eta_seconds} seconds"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds // 60} minutes"
            else:
                eta_str = f"{eta_seconds // 3600}h {(eta_seconds % 3600) // 60}m"
            self.eta_label.setText(f"Estimated time: {eta_str}")
        else:
            self.eta_label.setText("Estimated time: --")

    def append_log(self, message: str):
        """Append message to log output."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    def _update_overall_progress(self):
        """Update overall progress based on task statuses."""
        if not self._tasks:
            return

        completed = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        total = len(self._tasks)
        progress = completed / total * 100
        if progress >= self._progress_value:
            self.set_progress(progress)

    def monitor_system_resources(self):
        """Show host-wide resources (legacy/default standalone behavior)."""
        self._resource_mode = "system"
        self._resource_pid_provider = None
        self.resource_group.setTitle("Resources")

    def monitor_process_tree(self, pid_provider: Callable[[], int | None]):
        """Show resources used by the local OpenBench subprocess tree."""
        self._resource_mode = "process"
        self._resource_pid_provider = pid_provider
        self.resource_group.setTitle("Resources (OpenBench local)")

    def show_resource_unavailable(self, title: str = "Resources (remote N/A)"):
        """Make resource labels explicit when the GUI is not measuring the run host."""
        self._resource_mode = "unavailable"
        self._resource_pid_provider = None
        self.resource_group.setTitle(title)
        self._set_resource_labels_unavailable()

    def _set_resource_labels_unavailable(self):
        self.cpu_bar.setValue(0)
        self.mem_bar.setValue(0)
        self.cpu_label.setText("N/A")
        self.mem_label.setText("N/A")

    @staticmethod
    def _psutil_process_errors(psutil):
        return tuple(
            exc
            for exc in (
                getattr(psutil, "NoSuchProcess", None),
                getattr(psutil, "AccessDenied", None),
                getattr(psutil, "ZombieProcess", None),
            )
            if exc is not None
        ) or (Exception,)

    def _update_process_tree_usage(self, psutil):
        provider = self._resource_pid_provider
        pid = provider() if provider is not None else None
        if not pid:
            self._set_resource_labels_unavailable()
            return

        errors = self._psutil_process_errors(psutil)
        try:
            root = psutil.Process(pid)
            processes = [root] + list(root.children(recursive=True))
        except errors:
            self._set_resource_labels_unavailable()
            return

        cpu_percent = 0.0
        rss_bytes = 0
        for process in processes:
            try:
                cpu_percent += float(process.cpu_percent(interval=None))
                rss_bytes += int(process.memory_info().rss)
            except errors + (OSError, AttributeError, TypeError, ValueError):
                continue

        # psutil's per-process percentage can exceed 100 on multi-core hosts.
        # Normalize it to a whole-machine 0..100 scale to match the progress bar.
        try:
            logical_cpus = max(1, int(psutil.cpu_count() or 1))
        except (AttributeError, TypeError, ValueError):
            logical_cpus = 1
        cpu_percent /= logical_cpus

        try:
            total_memory = int(psutil.virtual_memory().total)
            mem_percent = (rss_bytes / total_memory * 100) if total_memory > 0 else 0.0
        except (OSError, AttributeError, TypeError, ValueError):
            mem_percent = 0.0

        cpu_value = max(0, min(100, int(round(cpu_percent))))
        mem_value = max(0, min(100, int(round(mem_percent))))
        self.cpu_bar.setValue(cpu_value)
        self.mem_bar.setValue(mem_value)
        self.cpu_label.setText(f"{cpu_value}%")
        self.mem_label.setText(f"{mem_value}%")

    def _update_resource_usage(self):
        """Update resource usage display."""
        if self._resource_mode == "unavailable":
            self._set_resource_labels_unavailable()
            return
        try:
            import psutil

            if self._resource_mode == "process":
                self._update_process_tree_usage(psutil)
                return

            cpu_percent = int(psutil.cpu_percent())
            self.cpu_bar.setValue(cpu_percent)
            self.cpu_label.setText(f"{cpu_percent}%")

            mem = psutil.virtual_memory()
            mem_percent = int(mem.percent)
            self.mem_bar.setValue(mem_percent)
            self.mem_label.setText(f"{mem_percent}%")
        except (ImportError, AttributeError, OSError):
            self._set_resource_labels_unavailable()

    def start_monitoring(self):
        """Start resource monitoring."""
        self._resource_timer.start(1000)  # Update every second

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._resource_timer.stop()

    def reset(self):
        """Reset dashboard to initial state."""
        self._tasks.clear()
        self.log_output.clear()
        self.monitor_system_resources()
        self.set_progress(0)
        self.cpu_bar.setValue(0)
        self.mem_bar.setValue(0)
        self.cpu_label.setText("0%")
        self.mem_label.setText("0%")
