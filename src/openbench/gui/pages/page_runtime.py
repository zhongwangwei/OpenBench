# -*- coding: utf-8 -*-
"""
Runtime Environment page for configuring local or remote execution.
"""

import logging
import os
import sys
import shutil
import yaml

from PySide6.QtWidgets import (
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLineEdit,
)

from PySide6.QtCore import QThread, Signal

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets.remote_config import RemoteConfigWidget, _InstallProgressDialog

_DETACHED_INSTALL_WORKERS = []


class _LocalInstallWorker(QThread):
    """Stream a local git install/update subprocess off the GUI thread."""

    line = Signal(str)
    finished_with_result = Signal(int)
    failed = Signal(str)

    def __init__(self, cmd, parent=None):
        super().__init__(parent)
        self._cmd = cmd
        self._process = None

    def stop(self) -> None:
        """Terminate the subprocess directly; a silent process never reaches
        the per-line interruption check (readline blocks until output)."""
        self.requestInterruption()
        process = self._process
        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass

    def run(self) -> None:  # pragma: no cover - exercised via GUI integration
        import subprocess

        try:
            self._process = subprocess.Popen(
                self._cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in self._process.stdout:
                if self.isInterruptionRequested():
                    self._process.terminate()
                    break
                self.line.emit(line.rstrip())
            self._process.wait()
            self.finished_with_result.emit(self._process.returncode)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._process = None


from openbench.gui.widgets.no_scroll_widgets import NoScrollSpinBox, NoScrollComboBox
from openbench.gui.path_utils import is_cross_platform_path

logger = logging.getLogger(__name__)


def get_default_runtime_settings_path() -> str:
    """Get the default path for runtime settings file."""
    config_dir = os.path.join(os.path.expanduser("~"), ".openbench_wizard")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "runtime_settings.yaml")


class PageRuntime(BasePage):
    """Runtime Environment configuration page."""

    PAGE_ID = "runtime"
    PAGE_TITLE = "Runtime Environment"
    PAGE_SUBTITLE = "Configure where OpenBench will run - locally on this machine or on a remote server."

    def _setup_content(self):
        """Setup page content."""
        # === Execution Mode ===
        mode_group = QGroupBox("Execution Mode")
        mode_layout = QFormLayout(mode_group)
        mode_layout.setSpacing(12)

        mode_buttons = QHBoxLayout()
        mode_buttons.setSpacing(20)
        self.execution_mode_group = QButtonGroup(self)
        self.radio_local = QRadioButton("Local")
        self.radio_remote = QRadioButton("Remote Server")
        self.radio_local.setChecked(True)
        self.execution_mode_group.addButton(self.radio_local)
        self.execution_mode_group.addButton(self.radio_remote)
        self.radio_local.toggled.connect(self._on_execution_mode_changed)
        mode_buttons.addWidget(self.radio_local)
        mode_buttons.addWidget(self.radio_remote)
        mode_buttons.addStretch()

        # Save/Load Settings buttons
        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_save_settings.setToolTip("Save runtime settings to a file")
        self.btn_save_settings.clicked.connect(self._save_runtime_settings)
        mode_buttons.addWidget(self.btn_save_settings)

        self.btn_load_settings = QPushButton("Load Settings")
        self.btn_load_settings.setToolTip("Load runtime settings from a file")
        self.btn_load_settings.clicked.connect(self._load_runtime_settings)
        mode_buttons.addWidget(self.btn_load_settings)

        self.btn_reset_settings = QPushButton("Reset")
        self.btn_reset_settings.setToolTip("Clear cached settings and reset to defaults")
        self.btn_reset_settings.clicked.connect(self._reset_cached_settings)
        mode_buttons.addWidget(self.btn_reset_settings)

        mode_layout.addRow("Mode:", mode_buttons)

        self.content_layout.addWidget(mode_group)

        # === Parallel Processing (always visible) ===
        self.parallel_group = QGroupBox("Parallel Processing")
        parallel_layout = QFormLayout(self.parallel_group)
        parallel_layout.setSpacing(12)

        # Number of CPU cores
        cores_layout = QHBoxLayout()
        self.num_cores_spin = NoScrollSpinBox()
        self.num_cores_spin.setRange(1, 128)
        self.num_cores_spin.setValue(min(4, os.cpu_count() or 4))
        self.num_cores_spin.setMinimumWidth(80)
        self.num_cores_spin.setToolTip("Number of CPU cores to use for parallel processing")
        self.num_cores_spin.valueChanged.connect(self._on_config_changed)
        cores_layout.addWidget(self.num_cores_spin)
        self.cpu_available_label = QLabel(f"(Available: {os.cpu_count() or 'N/A'})")
        cores_layout.addWidget(self.cpu_available_label)
        cores_layout.addStretch()
        parallel_layout.addRow("CPU Cores:", cores_layout)

        self.content_layout.addWidget(self.parallel_group)

        # === Local Python Environment ===
        self.local_env_group = QGroupBox("Local Python Environment")
        local_layout = QFormLayout(self.local_env_group)
        local_layout.setSpacing(12)

        # Conda environment with Refresh button (first, so user selects env before Python)
        conda_layout = QHBoxLayout()
        conda_layout.setSpacing(8)
        self.conda_combo = NoScrollComboBox()
        self.conda_combo.addItem("(Not using conda environment)")
        self.conda_combo.currentTextChanged.connect(self._on_conda_changed)
        conda_layout.addWidget(self.conda_combo, 1)

        self.btn_refresh_conda = QPushButton("Refresh")
        self.btn_refresh_conda.setFixedWidth(60)
        self.btn_refresh_conda.setToolTip("Refresh conda environments")
        self.btn_refresh_conda.clicked.connect(self._refresh_conda)
        conda_layout.addWidget(self.btn_refresh_conda)

        local_layout.addRow("Conda Env:", conda_layout)

        # Python path with Detect and Browse buttons
        python_layout = QHBoxLayout()
        python_layout.setSpacing(8)
        self.python_combo = NoScrollComboBox()
        self.python_combo.setEditable(True)
        self.python_combo.setMinimumWidth(300)
        self.python_combo.currentTextChanged.connect(self._on_python_changed)
        python_layout.addWidget(self.python_combo, 1)

        self.btn_detect_python = QPushButton("Detect")
        self.btn_detect_python.setFixedWidth(60)
        self.btn_detect_python.setToolTip("Auto-detect Python interpreters")
        self.btn_detect_python.clicked.connect(self._detect_python)
        python_layout.addWidget(self.btn_detect_python)

        self.btn_browse_python = QPushButton("Browse")
        self.btn_browse_python.setFixedWidth(60)
        self.btn_browse_python.setToolTip("Browse for Python interpreter")
        self.btn_browse_python.clicked.connect(self._browse_python)
        python_layout.addWidget(self.btn_browse_python)

        local_layout.addRow("Python:", python_layout)

        # OpenBench path — auto-detected for local mode, hidden by default.
        # Kept as a hidden field so save/load config and remote mode still work.
        self.local_openbench_input = QLineEdit()
        self.local_openbench_input.setPlaceholderText("Path to OpenBench installation directory")
        self.local_openbench_input.textChanged.connect(self._on_config_changed)
        self.local_openbench_input.setVisible(False)

        self.btn_browse_openbench = QPushButton("Browse")
        self.btn_browse_openbench.setFixedWidth(60)
        self.btn_browse_openbench.setToolTip("Browse for OpenBench installation directory")
        self.btn_browse_openbench.clicked.connect(self._browse_openbench)
        self.btn_browse_openbench.setVisible(False)

        self.btn_install_openbench = QPushButton("Install")
        self.btn_install_openbench.setFixedWidth(60)
        self.btn_install_openbench.setToolTip("Install OpenBench from GitHub")
        self.btn_install_openbench.clicked.connect(self._install_openbench)
        self.btn_install_openbench.setVisible(False)

        # Auto-fill with current OpenBench root
        from openbench.gui.path_utils import get_openbench_root

        self.local_openbench_input.setText(get_openbench_root())

        self.content_layout.addWidget(self.local_env_group)

        # === Remote Configuration ===
        self.remote_config_widget = RemoteConfigWidget()
        self.remote_config_widget.config_changed.connect(self._on_config_changed)
        self.remote_config_widget.connection_status_changed.connect(self._on_connection_status_changed)
        self.remote_config_widget.hide()  # Hidden by default (Local mode)
        self.content_layout.addWidget(self.remote_config_widget)

        # Auto-detect Python on startup
        self._detect_python()

        # Restore the previous runtime settings after widgets are initialized.
        # The settings file is written by _auto_save_settings() on user changes;
        # clearing it here made _auto_load_settings() unreachable in practice.
        self._auto_load_settings()

    def _on_execution_mode_changed(self, checked: bool):
        """Handle execution mode change."""
        if self.radio_local.isChecked():
            self.parallel_group.show()
            self.local_env_group.show()
            self.remote_config_widget.hide()
            # Reset CPU available label to local
            self.cpu_available_label.setText(f"(Available: {os.cpu_count() or 'N/A'})")

            # Disconnect remote connection when switching to local mode
            if self.remote_config_widget.is_connected():
                self.remote_config_widget.disconnect()

            # Clear remote config when switching to local
            self.remote_config_widget.reset_to_defaults()

            # Switch to local storage
            self._switch_to_local_storage()
        else:
            self.parallel_group.hide()  # Parallel Processing is inside RemoteConfigWidget
            self.local_env_group.hide()
            self.remote_config_widget.show()

            # Clear local config when switching to remote
            self.local_openbench_input.clear()

        self._on_config_changed()

    def _on_config_changed(self):
        """Handle any configuration change."""
        self.save_to_config()
        # Auto-save to default path for next startup
        self._auto_save_settings()

    def _on_connection_status_changed(self, connected: bool):
        """Handle SSH connection status change."""
        if connected:
            # Update controller's ssh_manager when connected
            self.controller.ssh_manager = self.remote_config_widget.get_ssh_manager()
            logger.debug("SSH manager set on controller")

            # Automatically switch to remote mode when connected
            if not self.radio_remote.isChecked():
                self.radio_remote.blockSignals(True)
                self.radio_remote.setChecked(True)
                self.radio_local.setChecked(False)
                self.radio_remote.blockSignals(False)
                # Update UI visibility
                self.parallel_group.hide()
                self.local_env_group.hide()
                self.remote_config_widget.show()

            # Switch to remote storage
            self._switch_to_remote_storage()
        else:
            # Clear ssh_manager when disconnected
            self.controller.ssh_manager = None
            logger.debug("SSH manager cleared from controller")

    def _switch_to_local_storage(self):
        """Switch to local storage mode."""
        from openbench.gui.path_utils import get_openbench_root

        main_window = self._get_main_window()
        if main_window and hasattr(main_window, "setup_local_storage"):
            # Use local OpenBench path if configured, otherwise fall back to auto-detect
            project_dir = self.local_openbench_input.text().strip()
            if not project_dir or not os.path.isdir(project_dir):
                project_dir = get_openbench_root()
            main_window.setup_local_storage(project_dir)
            logger.info(f"Switched to local storage mode: {project_dir}")

    def _switch_to_remote_storage(self):
        """Switch to remote storage mode."""
        main_window = self._get_main_window()
        if main_window and hasattr(main_window, "setup_remote_storage"):
            ssh_manager = self.remote_config_widget.get_ssh_manager()
            remote_config = self.remote_config_widget.get_config()
            # Use openbench_path as the remote project directory
            remote_project_dir = remote_config.get("openbench_path", "")
            if ssh_manager and remote_project_dir:
                main_window.setup_remote_storage(ssh_manager, remote_project_dir)
                logger.info(f"Switched to remote storage mode: {remote_project_dir}")
            elif ssh_manager:
                # If no project dir configured, use default ~/OpenBench
                default_dir = "~/OpenBench"
                main_window.setup_remote_storage(ssh_manager, default_dir)
                logger.info(f"Switched to remote storage mode with default: {default_dir}")

    def _get_main_window(self):
        """Get the main window instance."""
        # Use Qt's window() method to get the top-level window
        window = self.window()
        if window and window.__class__.__name__ == "MainWindow":
            return window
        return None

    def _on_python_changed(self, text):
        """Handle Python path change."""
        self._refresh_conda()
        self._on_config_changed()

    def _on_conda_changed(self, text):
        """Handle conda environment change."""
        self._on_config_changed()

    def _detect_python(self):
        """Auto-detect available Python interpreters."""
        detected = []
        is_windows = sys.platform == "win32"
        user_home = os.path.expanduser("~")

        # PRIORITY 1: Check active conda environment (CONDA_PREFIX)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            if is_windows:
                conda_python = os.path.join(conda_prefix, "python.exe")
            else:
                conda_python = os.path.join(conda_prefix, "bin", "python")
            if os.path.exists(conda_python):
                detected.append(f"{conda_python} (active conda)")

        # PRIORITY 2: Check common conda/miniforge locations
        if is_windows:
            conda_paths = [
                (os.path.join(user_home, "anaconda3", "python.exe"), "anaconda3"),
                (os.path.join(user_home, "miniconda3", "python.exe"), "miniconda3"),
                (os.path.join(user_home, "miniforge3", "python.exe"), "miniforge3"),
            ]
        else:
            conda_paths = [
                (os.path.join(user_home, "miniforge3", "bin", "python"), "miniforge3"),
                (os.path.join(user_home, "miniconda3", "bin", "python"), "miniconda3"),
                (os.path.join(user_home, "anaconda3", "bin", "python"), "anaconda3"),
                ("/opt/homebrew/bin/python3", "homebrew"),
                ("/usr/local/bin/python3", "local"),
            ]

        for path, label in conda_paths:
            if os.path.exists(path) and path not in [d.split(" ")[0] for d in detected]:
                detected.append(f"{path} ({label})")

        # PRIORITY 3: Check PATH
        python_names = ["python3", "python"] if not is_windows else ["python", "python3"]
        for name in python_names:
            path = shutil.which(name)
            if path and path not in [d.split(" ")[0] for d in detected]:
                if path == "/usr/bin/python3":
                    detected.append(f"{path} (system)")
                else:
                    detected.append(f"{path} (PATH)")

        # Update combo box
        current_text = self.python_combo.currentText()
        self.python_combo.blockSignals(True)
        self.python_combo.clear()

        for item in detected:
            self.python_combo.addItem(item)

        # Restore previous selection if valid
        if current_text:
            idx = self.python_combo.findText(current_text)
            if idx >= 0:
                self.python_combo.setCurrentIndex(idx)

        self.python_combo.blockSignals(False)

        # Also refresh conda environments
        self._refresh_conda()

    def _browse_python(self):
        """Open file dialog to select Python interpreter."""
        from PySide6.QtWidgets import QFileDialog

        if sys.platform == "win32":
            filter_str = "Python (python.exe);;All Files (*)"
        else:
            filter_str = "All Files (*)"

        path, _ = QFileDialog.getOpenFileName(self, "Select Python Interpreter", os.path.expanduser("~"), filter_str)

        if path:
            self.python_combo.setCurrentText(path)

    def _browse_openbench(self):
        """Browse for OpenBench installation directory."""
        current_path = self.local_openbench_input.text().strip()
        start_dir = current_path if current_path and os.path.isdir(current_path) else os.path.expanduser("~")

        path = QFileDialog.getExistingDirectory(self, "Select OpenBench Installation Directory", start_dir)

        if path:
            # Verify it's a valid OpenBench installation. v3 markers:
            # editable layout `src/openbench/cli/main.py` or a
            # `pyproject.toml` declaring the openbench package. The
            # pre-v3 marker `openbench/openbench.py` was retired and
            # would falsely reject every valid v3 checkout.
            from openbench.gui.path_utils import looks_like_openbench_root

            if looks_like_openbench_root(path):
                self.local_openbench_input.setText(path)
            else:
                reply = QMessageBox.question(
                    self,
                    "Not a Valid OpenBench Directory",
                    "The selected directory does not look like an OpenBench v3 "
                    "installation.\nExpected one of:\n"
                    "  • src/openbench/cli/main.py (editable install / repo)\n"
                    '  • pyproject.toml declaring `name = "openbench"`\n\n'
                    f"Selected: {path}\n\n"
                    "Use this path anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.local_openbench_input.setText(path)

    def _install_openbench(self):
        """Install OpenBench from GitHub."""
        from PySide6.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QTextEdit,
            QPushButton,
            QLabel,
            QRadioButton,
            QButtonGroup,
        )

        # Get installation path from input field
        install_path = self.local_openbench_input.text().strip()
        if not install_path:
            # Set default path if empty
            install_path = os.path.join(os.path.expanduser("~"), "OpenBench")
            self.local_openbench_input.setText(install_path)

        # Check if git is available
        git_path = shutil.which("git")
        if not git_path:
            QMessageBox.critical(
                self,
                "Git Not Found",
                "Git is not installed or not in PATH.\n\n"
                "Please install Git first:\n"
                "• macOS: brew install git\n"
                "• Ubuntu: sudo apt install git\n"
                "• Windows: https://git-scm.com/download/win",
            )
            return

        # Check if path already exists
        is_update = False
        if os.path.exists(install_path):
            git_dir = os.path.join(install_path, ".git")
            if os.path.exists(git_dir):
                reply = QMessageBox.question(
                    self,
                    "OpenBench Already Exists",
                    f"OpenBench already exists at:\n{install_path}\n\nWould you like to update it (git pull)?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply != QMessageBox.Yes:
                    return
                is_update = True
            else:
                QMessageBox.warning(
                    self,
                    "Directory Exists",
                    f"Directory already exists but is not a Git repository:\n{install_path}\n\n"
                    "Please choose a different path or remove the existing directory.",
                )
                return

        # Protocol selection dialog (consistent with remote mode)
        repo_url = "https://github.com/zhongwangwei/OpenBench.git"
        if not is_update:
            source_dialog = QDialog(self)
            source_dialog.setWindowTitle("Select Protocol")
            source_layout = QVBoxLayout(source_dialog)

            source_layout.addWidget(QLabel("Source: GitHub (github.com/zhongwangwei/OpenBench)"))
            source_layout.addWidget(QLabel("\nChoose protocol:"))

            protocol_group = QButtonGroup(source_dialog)
            radio_https = QRadioButton("HTTPS (https://github.com - Recommended)")
            radio_ssh = QRadioButton("SSH (git@github.com - If SSH key configured)")
            radio_https.setChecked(True)
            protocol_group.addButton(radio_https)
            protocol_group.addButton(radio_ssh)
            source_layout.addWidget(radio_https)
            source_layout.addWidget(radio_ssh)

            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_ok = QPushButton("OK")
            btn_ok.clicked.connect(source_dialog.accept)
            btn_cancel = QPushButton("Cancel")
            btn_cancel.clicked.connect(source_dialog.reject)
            btn_layout.addWidget(btn_ok)
            btn_layout.addWidget(btn_cancel)
            source_layout.addLayout(btn_layout)

            if source_dialog.exec() != QDialog.Accepted:
                return

            if radio_ssh.isChecked():
                repo_url = "git@github.com:zhongwangwei/OpenBench.git"

        # Build the git command up front so failures surface before the dialog.
        if is_update:
            cmd = ["git", "-C", install_path, "pull", "--ff-only"]
            starting = "Running git pull..."
        else:
            parent_dir = os.path.dirname(install_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            cmd = ["git", "clone", "--progress", repo_url, install_path]
            starting = f"Cloning from {repo_url}..."

        # Progress dialog (Esc/close stays blocked until the worker finishes).
        progress_dialog = _InstallProgressDialog(self)
        progress_dialog.setWindowTitle("Installing OpenBench" if not is_update else "Updating OpenBench")
        progress_dialog.resize(600, 400)
        progress_layout = QVBoxLayout(progress_dialog)

        status_label = QLabel(starting)
        progress_layout.addWidget(status_label)

        output_text = QTextEdit()
        output_text.setReadOnly(True)
        output_text.setStyleSheet("font-family: monospace;")
        progress_layout.addWidget(output_text)

        close_btn = QPushButton("Close")
        close_btn.setEnabled(False)
        close_btn.clicked.connect(progress_dialog.accept)
        progress_layout.addWidget(close_btn)

        output_text.append(f"$ {' '.join(cmd)}\n")
        self.btn_install_openbench.setEnabled(False)

        def finish(returncode: int):
            if returncode == 0:
                status_label.setText("✓ Installation successful!" if not is_update else "✓ Update successful!")
                status_label.setStyleSheet("color: green; font-weight: bold;")
                output_text.append("\n\nOpenBench installed successfully!")
            else:
                status_label.setText("✗ Installation failed!" if not is_update else "✗ Update failed!")
                status_label.setStyleSheet("color: red; font-weight: bold;")
            self.btn_install_openbench.setEnabled(True)
            close_btn.setEnabled(True)
            progress_dialog.allow_close = True
            self._local_install_worker = None

        def fail(message: str):
            output_text.append(f"\nError: {message}")
            status_label.setText("✗ Error occurred!")
            status_label.setStyleSheet("color: red; font-weight: bold;")
            self.btn_install_openbench.setEnabled(True)
            close_btn.setEnabled(True)
            progress_dialog.allow_close = True
            self._local_install_worker = None

        worker = _LocalInstallWorker(cmd)
        self._local_install_worker = worker
        if not getattr(self, "_install_destroy_hooked", False):
            # Embedded pages don't get closeEvent on app quit; destroyed does fire.
            self.destroyed.connect(lambda *_: self._cleanup_local_install_worker(detach=True))
            self._install_destroy_hooked = True
        worker.line.connect(lambda text: output_text.append(text))
        worker.finished_with_result.connect(finish)
        worker.failed.connect(fail)
        worker.finished.connect(worker.deleteLater)
        worker.start()

        progress_dialog.finished.connect(progress_dialog.deleteLater)
        progress_dialog.show()
        progress_dialog.exec()

    def _cleanup_local_install_worker(self, detach: bool = False):
        """Disconnect a running local install worker from this page's UI."""
        worker = getattr(self, "_local_install_worker", None)
        if worker is not None:
            from openbench.gui.widgets._task_worker import safe_disconnect

            safe_disconnect(worker.line, worker.finished_with_result, worker.failed)
            if worker.isRunning():
                worker.stop()
                if detach:
                    from openbench.gui.widgets._task_worker import detach_worker

                    detach_worker(worker, _DETACHED_INSTALL_WORKERS)
        self._local_install_worker = None

    def closeEvent(self, event):
        self._cleanup_local_install_worker(detach=True)
        super().closeEvent(event)

    def _refresh_conda(self):
        """Refresh the list of available conda environments."""

        current_python = self.python_combo.currentText().split(" ")[0]
        envs = self._get_conda_envs(current_python)

        current_env = self.conda_combo.currentText()
        self.conda_combo.blockSignals(True)
        self.conda_combo.clear()
        self.conda_combo.addItem("(Not using conda environment)")

        for env_name, env_path in envs:
            self.conda_combo.addItem(env_name, env_path)

        # Restore previous selection
        if current_env:
            idx = self.conda_combo.findText(current_env)
            if idx >= 0:
                self.conda_combo.setCurrentIndex(idx)

        self.conda_combo.blockSignals(False)

    def _get_conda_envs(self, python_path: str) -> list:
        """Get list of conda environments."""
        import subprocess
        import json

        envs = []
        if not python_path:
            return envs

        # Determine conda base path from Python path
        python_dir = os.path.dirname(python_path)
        if sys.platform == "win32":
            if "envs" in python_dir:
                conda_base = python_dir.split("envs")[0].rstrip(os.sep)
            else:
                conda_base = python_dir
            conda_exe = os.path.join(conda_base, "Scripts", "conda.exe")
            if not os.path.exists(conda_exe):
                conda_exe = os.path.join(conda_base, "condabin", "conda.bat")
        else:
            if "envs" in python_dir:
                conda_base = python_dir.split("envs")[0].rstrip(os.sep)
            else:
                conda_base = os.path.dirname(python_dir)
            conda_exe = os.path.join(conda_base, "bin", "conda")

        if not os.path.exists(conda_exe):
            user_home = os.path.expanduser("~")
            for base in ["miniforge3", "miniconda3", "anaconda3"]:
                if sys.platform == "win32":
                    test_exe = os.path.join(user_home, base, "Scripts", "conda.exe")
                else:
                    test_exe = os.path.join(user_home, base, "bin", "conda")
                if os.path.exists(test_exe):
                    conda_exe = test_exe
                    conda_base = os.path.join(user_home, base)
                    break

        if not os.path.exists(conda_exe):
            return envs

        try:
            result = subprocess.run([conda_exe, "env", "list", "--json"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for env_path in data.get("envs", []):
                    env_name = os.path.basename(env_path)
                    if env_name == conda_base.split(os.sep)[-1]:
                        env_name = "base"
                    envs.append((env_name, env_path))
        except Exception:
            pass

        return envs

    def load_from_config(self):
        """Load settings from controller config."""
        config = self.controller.config
        general = config.get("general", {})

        # Load execution mode
        execution_mode = general.get("execution_mode", "local")
        if execution_mode == "remote":
            self.radio_remote.setChecked(True)
            self.parallel_group.hide()
            self.local_env_group.hide()
            self.remote_config_widget.show()
        else:
            self.radio_local.setChecked(True)
            self.parallel_group.show()
            self.local_env_group.show()
            self.remote_config_widget.hide()

        # Load num_cores (for local mode)
        self.num_cores_spin.setValue(general.get("num_cores", 4))

        # Load Python path
        python_path = general.get("python_path", "")
        if python_path:
            self.python_combo.blockSignals(True)
            found = False
            for i in range(self.python_combo.count()):
                if self.python_combo.itemText(i).startswith(python_path):
                    self.python_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.python_combo.setCurrentText(python_path)
            self.python_combo.blockSignals(False)

        # Load conda environment
        conda_env = general.get("conda_env", "")
        if conda_env:
            self.conda_combo.blockSignals(True)
            idx = self.conda_combo.findText(conda_env)
            if idx >= 0:
                self.conda_combo.setCurrentIndex(idx)
            self.conda_combo.blockSignals(False)

        # Load local OpenBench path
        local_openbench_path = general.get("local_openbench_path", "")
        if local_openbench_path:
            self.local_openbench_input.setText(local_openbench_path)

        # Load remote config
        remote_config = general.get("remote") or {}
        if remote_config:
            self.remote_config_widget.set_config(remote_config)

    def save_to_config(self):
        """Save settings to controller config."""
        config = self.controller.config
        if "general" not in config:
            config["general"] = {}
        general = config["general"]

        # Save execution mode
        general["execution_mode"] = "local" if self.radio_local.isChecked() else "remote"

        # Save num_cores
        general["num_cores"] = self.num_cores_spin.value()

        # Save Python path, conda env, and OpenBench path for local mode
        general["python_path"] = self.python_combo.currentText().split(" ")[0]
        general["conda_env"] = self.conda_combo.currentText() if self.conda_combo.currentIndex() > 0 else ""
        general["local_openbench_path"] = self.local_openbench_input.text().strip()

        # Save remote config if in remote mode
        if self.radio_remote.isChecked():
            remote_config = self.remote_config_widget.get_config()
            general["remote"] = remote_config

    def validate(self) -> bool:
        """Validate the page configuration."""
        if self.radio_remote.isChecked():
            # Check if remote server is configured
            if not self.remote_config_widget.is_connected():
                from PySide6.QtWidgets import QMessageBox

                reply = QMessageBox.question(
                    self,
                    "Not Connected",
                    "You haven't connected to the remote server yet.\n\nDo you want to continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                return reply == QMessageBox.Yes
        return True

    def get_remote_config_widget(self):
        """Get the remote config widget for external access."""
        return self.remote_config_widget

    def is_remote_mode(self) -> bool:
        """Check if remote execution mode is selected."""
        return self.radio_remote.isChecked()

    def _save_runtime_settings(self):
        """Save runtime settings to a file."""
        default_path = get_default_runtime_settings_path()

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Runtime Settings", default_path, "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if not file_path:
            return

        try:
            settings = self._collect_runtime_settings()
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            QMessageBox.information(self, "Settings Saved", f"Runtime settings saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings:\n{str(e)}")

    def _load_runtime_settings(self):
        """Load runtime settings from a file."""
        default_path = get_default_runtime_settings_path()
        start_dir = os.path.dirname(default_path) if os.path.exists(default_path) else os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Runtime Settings", start_dir, "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}

            self._apply_runtime_settings(settings)

            QMessageBox.information(self, "Settings Loaded", f"Runtime settings loaded from:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load settings:\n{str(e)}")

    def _collect_runtime_settings(self) -> dict:
        """Collect current runtime settings into a dictionary."""
        settings = {
            "execution_mode": "local" if self.radio_local.isChecked() else "remote",
            "num_cores": self.num_cores_spin.value(),
            "python_path": self.python_combo.currentText().split(" ")[0],
            "conda_env": self.conda_combo.currentText() if self.conda_combo.currentIndex() > 0 else "",
            "local_openbench_path": self.local_openbench_input.text().strip(),
        }

        # Include remote config if in remote mode
        if self.radio_remote.isChecked():
            settings["remote"] = self.remote_config_widget.get_config()

        return settings

    def _apply_runtime_settings(self, settings: dict):
        """Apply runtime settings from a dictionary."""
        # Apply execution mode
        execution_mode = settings.get("execution_mode", "local")
        if execution_mode == "remote":
            self.radio_remote.setChecked(True)
        else:
            self.radio_local.setChecked(True)

        # Apply num_cores
        self.num_cores_spin.setValue(settings.get("num_cores", 4))

        # Apply Python path
        python_path = settings.get("python_path", "")
        if python_path:
            self.python_combo.blockSignals(True)
            found = False
            for i in range(self.python_combo.count()):
                if self.python_combo.itemText(i).startswith(python_path):
                    self.python_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.python_combo.setCurrentText(python_path)
            self.python_combo.blockSignals(False)

        # Apply conda environment
        conda_env = settings.get("conda_env", "")
        if conda_env:
            self.conda_combo.blockSignals(True)
            idx = self.conda_combo.findText(conda_env)
            if idx >= 0:
                self.conda_combo.setCurrentIndex(idx)
            self.conda_combo.blockSignals(False)

        # Apply local OpenBench path
        local_openbench_path = settings.get("local_openbench_path", "")
        if local_openbench_path:
            self.local_openbench_input.setText(local_openbench_path)

        # Apply remote config
        remote_config = settings.get("remote") or {}
        if remote_config:
            self.remote_config_widget.set_config(remote_config)

        # Save to controller config
        self.save_to_config()

    def _auto_load_settings(self):
        """Try to auto-load settings from default path on startup."""
        default_path = get_default_runtime_settings_path()
        if os.path.exists(default_path):
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f) or {}

                # Clean up cross-platform paths that won't work on current system
                self._clean_cross_platform_paths(settings)

                self._apply_runtime_settings(settings)
            except Exception:
                pass  # Silently ignore errors on auto-load

    def _clean_cross_platform_paths(self, settings: dict):
        """Remove paths that are from a different platform."""
        # Check local_openbench_path
        local_path = settings.get("local_openbench_path", "")
        if local_path and is_cross_platform_path(local_path):
            logger.info(f"Clearing cross-platform path: {local_path}")
            settings["local_openbench_path"] = ""

        # Check remote config paths
        remote = settings.get("remote") or {}
        if remote:
            # openbench_path in remote config should be Unix-style (for remote server)
            # so we don't need to clean it - it's expected to be a Linux path

            # But python_path might be mixed
            python_path = remote.get("python_path", "")
            if python_path and is_cross_platform_path(python_path):
                # For remote, python paths should be Unix-style
                # Only clear if it looks like a local Windows path being used incorrectly
                pass  # Remote paths are expected to be Unix-style

    def _clear_cached_settings_file(self):
        """Clear the cached settings file on startup."""
        try:
            default_path = get_default_runtime_settings_path()
            if os.path.exists(default_path):
                os.remove(default_path)
                logger.info(f"Cleared cached settings on startup: {default_path}")
        except Exception as e:
            logger.debug(f"Could not clear cached settings: {e}")

    def _auto_save_settings(self):
        """Auto-save settings to default path for next startup."""
        try:
            default_path = get_default_runtime_settings_path()
            settings = self._collect_runtime_settings()
            with open(default_path, "w", encoding="utf-8") as f:
                yaml.dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception:
            pass  # Silently ignore errors on auto-save

    def _reset_cached_settings(self):
        """Reset cached settings to defaults and clear the cache file."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "This will clear all cached runtime settings and reset to defaults.\n\n"
            "This is useful when switching between different systems (Windows/Linux/Mac) "
            "or when you want to start fresh.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            # Delete the cached settings file
            default_path = get_default_runtime_settings_path()
            if os.path.exists(default_path):
                os.remove(default_path)
                logger.info(f"Removed cached settings: {default_path}")

            # Reset UI to defaults
            self.radio_local.setChecked(True)
            self.num_cores_spin.setValue(min(4, os.cpu_count() or 4))
            self.python_combo.setCurrentIndex(0)
            self.conda_combo.setCurrentIndex(0)
            self.local_openbench_input.clear()
            self.remote_config_widget.reset_to_defaults()

            # Also reset the controller config
            if self.controller:
                self.controller.reset()

            QMessageBox.information(
                self,
                "Settings Reset",
                f"Cached settings have been cleared.\n\nCache location: {os.path.dirname(default_path)}",
            )

        except Exception as e:
            logger.error(f"Failed to reset settings: {e}")
            QMessageBox.warning(self, "Reset Failed", f"Failed to reset settings: {e}")
