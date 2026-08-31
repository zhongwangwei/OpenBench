# -*- coding: utf-8 -*-
"""
Run and Monitor page with progress dashboard.

Supports both local and remote execution modes:
- Local: Uses EvaluationRunner to run OpenBench on the local machine
- Remote: Uses RemoteRunner to execute OpenBench on a remote server via SSH
"""

import logging
import os
import subprocess
import platform
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog

from openbench.gui.remote_python import quote_remote_path
from openbench.remote.ssh import expand_remote_home
from openbench.gui.widgets._ssh_worker import execute_responsive
from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import ProgressDashboard, TaskStatus
from openbench.gui.runner import EvaluationRunner
from openbench.gui.remote_runner import RemoteRunner

logger = logging.getLogger(__name__)


def _source_list(value) -> list:
    if isinstance(value, (list, tuple, set)):
        return [source for source in value if source]
    return [value] if value else []


def _ssh_target_identity(ssh_manager):
    getter = getattr(ssh_manager, "get_active_target_identity", None)
    if not callable(getter):
        return None
    identity = getter()
    return tuple(identity) if identity is not None else None


def count_evaluation_tasks(config: dict, selected_variables: list[str]) -> int:
    """Count the per-variable ref/sim pairs that the runner will actually build."""
    ref_data = config.get("ref_data", {}) or {}
    sim_data = config.get("sim_data", {}) or {}
    ref_general = ref_data.get("general", {}) or {}
    sim_general = sim_data.get("general", {}) or {}
    ref_configs = ref_data.get("source_configs", {}) or {}
    sim_configs = sim_data.get("source_configs", {}) or {}
    fallback_refs = [name for name, path in (ref_data.get("def_nml", {}) or {}).items() if path]
    fallback_sims = [name for name, path in (sim_data.get("def_nml", {}) or {}).items() if path]

    total = 0
    for variable in selected_variables:
        refs = _source_list(ref_general.get(f"{variable}_ref_source"))
        if not refs:
            refs = [key.split("::", 1)[1] for key in ref_configs if key.startswith(f"{variable}::")]
        if not refs:
            refs = fallback_refs

        sims = _source_list(sim_general.get(f"{variable}_sim_source"))
        if not sims:
            sims = list(sim_configs) or fallback_sims

        total += len(set(refs)) * len(set(sims))
    return total


class _RemoteDownloadCanceled(Exception):
    """Internal sentinel for user-canceled remote folder downloads."""


_REMOTE_FILE_LIST_MARKER = "__OPENBENCH_FILE_LIST__"


class RemoteFolderDownloadWorker(QThread):
    """Download a remote output folder without blocking the GUI thread."""

    progress_updated = Signal(int, int, str)
    finished_signal = Signal(bool, bool, str, str)  # success, canceled, message, local_target

    def __init__(self, ssh_manager, remote_dir: str, local_target: str, target_identity=None, parent=None):
        super().__init__(parent)
        self._ssh_manager = ssh_manager
        self._remote_dir = remote_dir
        self._local_target = local_target
        self._target_identity = tuple(target_identity) if target_identity is not None else None
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _raise_if_canceled(self):
        if self._stop_requested:
            raise _RemoteDownloadCanceled()

    def _ensure_target_active(self):
        if self._target_identity is not None and _ssh_target_identity(self._ssh_manager) != self._target_identity:
            raise RuntimeError("remote target identity changed")

    def run(self):
        staging_target: Path | None = None
        try:
            self._raise_if_canceled()
            self._ensure_target_active()
            self._remote_dir = expand_remote_home(self._ssh_manager, self._remote_dir).replace("\\", "/")
            final_target = Path(self._local_target).expanduser()
            final_target.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(final_target) and not final_target.is_dir():
                raise RuntimeError(f"Local download target is not a directory: {final_target}")
            staging_target = Path(
                tempfile.mkdtemp(prefix=f".{final_target.name}.download.", dir=str(final_target.parent))
            )
            self._ensure_target_active()
            stdout, stderr, exit_code = self._ssh_manager.execute(
                f"printf '%s\\0' {_REMOTE_FILE_LIST_MARKER}; "
                f"find {quote_remote_path(self._remote_dir)} -type f -print0",
                timeout=60,
                should_abort=lambda: self._stop_requested,
            )
            self._ensure_target_active()
            if exit_code != 0:
                self.finished_signal.emit(False, False, f"Failed to list remote files:\n{stderr}", self._local_target)
                return

            marker = f"{_REMOTE_FILE_LIST_MARKER}\0"
            if marker not in stdout:
                raise RuntimeError("Remote file listing did not return its expected marker")
            files = [remote_file for remote_file in stdout.split(marker, 1)[1].split("\0") if remote_file]
            if not files:
                self.finished_signal.emit(True, False, "No files found in the remote directory.", self._local_target)
                return

            total_files = len(files)
            self._ensure_target_active()
            sftp = self._ssh_manager.open_sftp()
            self._ensure_target_active()
            for index, remote_file in enumerate(files):
                self._raise_if_canceled()
                self._ensure_target_active()
                rel_path = PageRunMonitor._remote_download_relpath(remote_file, self._remote_dir)
                if rel_path is None:
                    raise ValueError(f"Remote file is outside the requested directory: {remote_file}")
                local_file = os.path.join(str(staging_target), rel_path)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                self.progress_updated.emit(index, total_files, rel_path)

                def _copy_progress(_transferred, _total):
                    self._ensure_target_active()
                    self._raise_if_canceled()

                try:
                    sftp.get(remote_file, local_file, callback=_copy_progress)
                except TypeError:
                    # Some tests/fakes implement a minimal Paramiko-like get()
                    # without callback support. Cancellation remains checked
                    # between files in that compatibility path.
                    sftp.get(remote_file, local_file)
                self._ensure_target_active()
                self.progress_updated.emit(index + 1, total_files, rel_path)

            self._ensure_target_active()
            self._raise_if_canceled()
            from openbench.cli.sim import _replace_directory_preserving_old_on_failure

            _replace_directory_preserving_old_on_failure(staging_target, final_target)
            staging_target = None
            self.finished_signal.emit(True, False, "Download complete", self._local_target)
        except _RemoteDownloadCanceled:
            self.finished_signal.emit(False, True, "Download canceled", self._local_target)
        except Exception as exc:
            if self._stop_requested:
                self.finished_signal.emit(False, True, "Download canceled", self._local_target)
            else:
                self.finished_signal.emit(False, False, f"Failed to download files:\n{exc}", self._local_target)
        finally:
            if staging_target is not None:
                shutil.rmtree(staging_target, ignore_errors=True)


class PageRunMonitor(BasePage):
    """Run and Monitor page."""

    PAGE_ID = "run_monitor"
    PAGE_TITLE = "Run & Monitor"
    PAGE_SUBTITLE = "Monitor evaluation progress"

    def __init__(self, controller, parent=None):
        self._runner = None
        self._download_worker = None
        self._last_ssh_manager_error = ""
        self._last_run_output_dir = None
        self._last_run_is_remote = None
        self._last_run_target_identity = None
        super().__init__(controller, parent)
        # Remove the trailing stretch added by BasePage so dashboard can expand
        self._remove_trailing_stretch()

    def _remove_trailing_stretch(self):
        """Remove the trailing stretch from content_layout to allow dashboard to expand."""
        count = self.content_layout.count()
        if count > 0:
            item = self.content_layout.itemAt(count - 1)
            if item and item.spacerItem():
                self.content_layout.takeAt(count - 1)

    def _setup_content(self):
        """Setup page content."""
        self.dashboard = ProgressDashboard()
        self.dashboard.stop_requested.connect(self._on_stop)
        self.dashboard.open_output_requested.connect(self._open_output)

        # Add with stretch factor 1 to fill available space
        self.content_layout.addWidget(self.dashboard, 1)

    def start_run(self, config_path: str):
        """Start evaluation run.

        Supports both local and remote execution based on execution_mode in config.
        - Local mode: Uses EvaluationRunner to run OpenBench locally
        - Remote mode: Uses RemoteRunner to execute via SSH on a remote server

        Args:
            config_path: Path to the exported OpenBench configuration file
        """
        if self._runner and self._runner.isRunning():
            QMessageBox.warning(self, "Already Running", "An evaluation is already running. Stop it first.")
            return

        # Build task list from config
        config = self.controller.config
        eval_items = config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]
        general = config.get("general", {})

        from openbench.remote.storage import RemoteStorage

        is_remote = general.get("execution_mode") == "remote"
        if is_remote and not isinstance(self.controller.storage, RemoteStorage):
            QMessageBox.critical(
                self,
                "Remote Connection Required",
                "Remote execution is selected, but no remote project connection is active.",
            )
            return
        run_output_dir = self.controller.get_output_dir()

        tasks = []
        for item in selected:
            tasks.append(f"{item} - Evaluation")
            if general.get("comparison"):
                tasks.append(f"{item} - Comparison")
            if general.get("statistics"):
                tasks.append(f"{item} - Statistics")

        self.dashboard.reset()
        self.dashboard.set_tasks(tasks)
        self.dashboard.start_monitoring()

        # Mark first task as running
        if tasks:
            self.dashboard.update_task_status(tasks[0], TaskStatus.RUNNING)

        # Calculate task counts for accurate progress
        num_variables = len(selected)

        # Count reference sources. The new scan-based ref page stores
        # per-(variable, source) entries in `source_configs` keyed
        # "<variable>::<source>"; legacy `def_nml` is no longer populated
        # by PageRefData and counting it gave 0 → progress bar stuck at
        # 95% for multi-source runs.
        ref_data = config.get("ref_data", {})
        ref_source_configs = ref_data.get("source_configs", {})
        if ref_source_configs:
            num_ref_sources = len({k.split("::", 1)[1] for k in ref_source_configs if "::" in k})
        else:
            ref_def_nml = ref_data.get("def_nml", {})
            num_ref_sources = len([k for k, v in ref_def_nml.items() if v])

        # Count simulation sources. Same situation: PageSimData writes
        # source_configs and leaves def_nml empty.
        sim_data = config.get("sim_data", {})
        sim_source_configs = sim_data.get("source_configs", {})
        if sim_source_configs:
            num_sim_sources = len(sim_source_configs)
        else:
            sim_def_nml = sim_data.get("def_nml", {})
            num_sim_sources = len([k for k, v in sim_def_nml.items() if v])

        # Count metrics and scores
        metrics = config.get("metrics", {})
        num_metrics = len([k for k, v in metrics.items() if v])

        scores = config.get("scores", {})
        num_scores = len([k for k, v in scores.items() if v])

        # Count groupby types
        num_groupby = 0
        if general.get("IGBP_groupby"):
            num_groupby += 1
        if general.get("PFT_groupby"):
            num_groupby += 1
        if general.get("Climate_zone_groupby"):
            num_groupby += 1

        # Count comparisons
        comparisons = config.get("comparisons", {})
        num_comparisons = len([k for k, v in comparisons.items() if v])

        # Count statistics independently from comparison tasks.
        statistics = config.get("statistics", {})
        num_statistics = len([k for k, v in statistics.items() if v])

        if is_remote:
            # Remote execution mode
            self._runner = self._create_remote_runner(config_path, general)
            if self._runner is None:
                # Error creating remote runner - message already shown
                self.dashboard.stop_monitoring()
                self._refresh_parent_navigation()
                return
            self.dashboard.monitor_remote_resources()
        else:
            # Local execution mode (default)
            python_path = general.get("python_path", "")
            self._runner = EvaluationRunner(config_path, python_path, self)
            self.dashboard.monitor_process_tree(lambda: self._runner.get_process_pid() if self._runner else None)

        self._last_run_output_dir = run_output_dir
        self._last_run_is_remote = is_remote
        self._last_run_target_identity = (
            _ssh_target_identity(getattr(self._runner, "_ssh_manager", None)) if is_remote else None
        )

        # Configure task counts for accurate progress tracking
        self._runner.set_task_counts(
            num_variables=num_variables,
            num_ref_sources=num_ref_sources,
            num_sim_sources=num_sim_sources,
            num_metrics=num_metrics,
            num_scores=num_scores,
            num_groupby=num_groupby,
            num_comparisons=num_comparisons,
            do_evaluation=general.get("evaluation", True),
            do_comparison=general.get("comparison", False),
            do_statistics=general.get("statistics", False),
            num_evaluation_tasks=count_evaluation_tasks(config, selected),
            num_statistics=num_statistics,
        )

        # Connect signals - same interface for both runners
        self._runner.progress_updated.connect(self._on_progress)
        self._runner.log_message.connect(self._on_log)
        if is_remote:
            self._runner.resource_updated.connect(self.dashboard.update_remote_resource_usage)
        self._runner.finished_signal.connect(self._on_finished)
        self._runner.finished.connect(self._refresh_parent_navigation)
        self._runner.start()
        self._refresh_parent_navigation()

    def _create_remote_runner(self, config_path: str, general: dict):
        """Create a RemoteRunner for SSH-based execution.

        Args:
            config_path: Path to the exported OpenBench configuration file
            general: General configuration dictionary

        Returns:
            RemoteRunner instance, or None if SSH connection not available
        """
        # Get remote configuration
        remote_config = general.get("remote") or {}

        if not remote_config:
            QMessageBox.critical(
                self,
                "Remote Configuration Missing",
                "Remote execution is enabled but no remote server is configured.\n\n"
                "Please configure a remote server in Runtime Environment.",
            )
            return None

        # Get SSH manager from the remote config widget via controller
        # The controller has access to pages through the main window
        ssh_manager = self._get_ssh_manager()

        if ssh_manager is None or not ssh_manager.is_connected:
            QMessageBox.critical(
                self,
                "SSH Connection Required",
                "Remote execution requires an active SSH connection.\n\n"
                "Please:\n"
                "1. Go to Runtime Environment\n"
                "2. Configure the remote server\n"
                "3. Click 'Test' to establish the connection\n"
                "4. Then return here and click Run again",
            )
            return None

        # Validate remote configuration
        python_path = remote_config.get("python_path", "")
        openbench_path = remote_config.get("openbench_path", "")

        if not python_path:
            QMessageBox.critical(
                self,
                "Remote Python Not Configured",
                "Remote Python path is not configured.\n\n"
                "Please configure the Python path in Runtime Environment > Remote Python Environment.",
            )
            return None

        if not openbench_path:
            QMessageBox.critical(
                self,
                "Remote OpenBench Not Configured",
                "Remote OpenBench path is not configured.\n\n"
                "Please configure the OpenBench path in Runtime Environment > Remote Python Environment.",
            )
            return None

        # Create and return the RemoteRunner
        # config_already_remote=True because page_preview already uploaded the config
        return RemoteRunner(config_path, ssh_manager, remote_config, self, config_already_remote=True)

    def _get_ssh_manager(self):
        """Get the SSH manager from the Runtime Environment page.

        The SSH manager is held by the RemoteConfigWidget in the Runtime Environment page.
        We access it through the controller's parent (MainWindow) which holds all pages.

        Returns:
            SSHManager instance if available and connected, None otherwise
        """
        self._last_ssh_manager_error = ""
        try:
            # Access the main window through the controller's parent
            # The controller is created by MainWindow with 'self' as parent
            main_window = self.controller.parent()
            if main_window is None:
                self._last_ssh_manager_error = "main window is not available"
                return None

            # Get the pages dictionary from the main window
            pages = getattr(main_window, "pages", None)
            if pages is None:
                self._last_ssh_manager_error = "main window has no pages registry"
                return None

            # Get the runtime page which contains the remote config widget
            runtime_page = pages.get("runtime")
            if runtime_page is None:
                self._last_ssh_manager_error = "runtime page is not available"
                return None

            # The remote config widget is a child of the runtime page
            remote_config_widget = getattr(runtime_page, "remote_config_widget", None)
            if remote_config_widget is None:
                self._last_ssh_manager_error = "remote config widget is not available"
                return None

            if not remote_config_widget.is_connected():
                self._last_ssh_manager_error = "configured remote execution target is not connected"
                return None

            # Get the SSH manager from the remote config widget
            return remote_config_widget.get_ssh_manager()
        except Exception as exc:
            self._last_ssh_manager_error = str(exc)
            logger.warning("Failed to retrieve SSH manager: %s", exc)
            return None

    def _on_progress(self, progress):
        """Handle progress update."""
        self.dashboard.set_progress(progress.progress)

        # Update task statuses based on progress
        if progress.current_variable and progress.current_stage:
            task_name = f"{progress.current_variable} - {progress.current_stage}"
            self.dashboard.update_task_status(task_name, TaskStatus.RUNNING)

    def _on_log(self, message: str):
        """Handle log message."""
        self.dashboard.append_log(message)

    def _on_finished(self, success: bool, message: str):
        """Handle run completion."""
        self.dashboard.stop_monitoring()
        self._refresh_parent_navigation()

        if success:
            self.dashboard.set_progress(100)
            QMessageBox.information(self, "Complete", message)
        else:
            if message == "Stopped by user":
                QMessageBox.information(self, "Stopped", "Evaluation stopped by user.")
                return
            # Check if it's an OpenBench not found error
            if "Could not find OpenBench" in message:
                self._prompt_openbench_location()
            elif "Evaluation completed with errors" in message:
                QMessageBox.warning(self, "Completed with Errors", message)
            else:
                QMessageBox.warning(self, "Failed", message)

    def _refresh_parent_navigation(self):
        """Refresh main-window navigation buttons after runner state changes."""
        main_window = self.controller.parent() if self.controller is not None else None
        if main_window is not None and hasattr(main_window, "_update_navigation"):
            main_window._update_navigation()

    def _prompt_openbench_location(self):
        """Prompt user to select OpenBench directory."""
        reply = QMessageBox.question(
            self,
            "OpenBench Not Found",
            "Could not find the OpenBench directory automatically.\n\nWould you like to select it manually?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select OpenBench Directory", os.path.expanduser("~"), QFileDialog.ShowDirsOnly
            )

            if dir_path:
                # Verify it's a valid v3 OpenBench directory using the
                # shared marker helper (editable layout or pyproject.toml
                # declaring the openbench package). The previous check
                # required `openbench/openbench.py`, a v2 layout that no
                # longer exists in v3 — so every valid v3 directory was
                # being rejected with no "Use anyway" fallback.
                from openbench.gui.path_utils import looks_like_openbench_root

                if looks_like_openbench_root(dir_path):
                    # Save the path
                    self._save_openbench_path(dir_path)
                    QMessageBox.information(
                        self, "Success", f"OpenBench directory saved:\n{dir_path}\n\nPlease click Run again."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Directory",
                        "The selected directory does not look like an OpenBench v3 "
                        "installation (no src/openbench/cli/main.py and no "
                        "pyproject.toml declaring the openbench package):\n"
                        f"{dir_path}",
                    )

    def _save_openbench_path(self, path: str):
        """Save OpenBench directory path."""
        try:
            home_dir = os.path.expanduser("~")
            config_dir = os.path.join(home_dir, ".openbench_wizard")
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "config.txt")
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(path)
        except Exception as e:
            print(f"Warning: Could not save OpenBench path: {e}")

    def _on_stop(self):
        """Handle stop request."""
        # Skip the confirm dialog (and the redundant stop()/kill round-trip)
        # when the runner has already finished naturally — `self._runner` is
        # still set but `isRunning()` is False after the worker exited.
        if self._runner and self._runner.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Stop", "Are you sure you want to stop the evaluation?", QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._runner.stop()
                self.dashboard.stop_monitoring()

    def _open_output(self):
        """Open output directory."""
        output_dir = getattr(self, "_last_run_output_dir", None) or self.controller.get_output_dir()

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        last_run_is_remote = getattr(self, "_last_run_is_remote", None)
        is_remote = (
            last_run_is_remote if last_run_is_remote is not None else isinstance(self.controller.storage, RemoteStorage)
        )

        if is_remote:
            # In remote mode, open remote file browser
            self._open_remote_output(output_dir)
            return

        if os.path.exists(output_dir):
            # Cross-platform open folder with error handling
            try:
                if platform.system() == "Windows":
                    os.startfile(output_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", output_dir], check=False)
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir], check=False)
            except Exception as e:
                QMessageBox.warning(self, "Open Error", f"Could not open directory:\n{output_dir}\n\nError: {str(e)}")
        else:
            QMessageBox.warning(self, "Directory Not Found", f"Output directory does not exist:\n{output_dir}")

    def _ensure_saved_remote_target_active(self, ssh_manager) -> bool:
        saved_identity = getattr(self, "_last_run_target_identity", None)
        if saved_identity is None:
            return True
        if _ssh_target_identity(ssh_manager) == tuple(saved_identity):
            return True
        QMessageBox.warning(
            self,
            "Remote Target Changed",
            "The saved run output belongs to a different remote target. Reconnect to that target or run again.",
        )
        return False

    def _open_remote_output(self, output_dir: str):
        """Open remote output directory in file browser with download option."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
        from openbench.gui.widgets.remote_config import RemoteFileBrowser

        ssh_manager = self._get_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            message = f"SSH connection is not available.\n\nRemote output directory:\n{output_dir}"
            detail = getattr(self, "_last_ssh_manager_error", "")
            if detail:
                message += f"\n\nDetails: {detail}"
            QMessageBox.warning(self, "Not Connected", message)
            return

        if not self._ensure_saved_remote_target_active(ssh_manager):
            return

        output_dir = expand_remote_home(ssh_manager, output_dir).replace("\\", "/")

        # Check if directory exists on remote
        try:
            stdout, stderr, exit_code = execute_responsive(
                ssh_manager, f"test -d {quote_remote_path(output_dir)} && echo 'exists'", timeout=10
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Remote Output Error",
                f"Failed to check remote output directory:\n{output_dir}\n\nError: {exc}",
            )
            return
        if exit_code != 0 or "exists" not in stdout:
            QMessageBox.warning(self, "Directory Not Found", f"Remote output directory does not exist:\n{output_dir}")
            return

        # Create dialog with remote file browser
        dialog = QDialog(self)
        dialog.setWindowTitle("Remote Output Directory")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(dialog)

        # Remote file browser
        browser = RemoteFileBrowser(ssh_manager=ssh_manager, start_path=output_dir, parent=dialog, select_dirs=True)
        layout.addWidget(browser, 1)

        # Button layout
        btn_layout = QHBoxLayout()

        # Download folder button
        btn_download_all = QPushButton("Download Folder to Local...")
        btn_download_all.setToolTip("Download the entire output folder to local machine")
        btn_download_all.clicked.connect(lambda: self._download_remote_folder(ssh_manager, output_dir, dialog))
        btn_layout.addWidget(btn_download_all)

        btn_layout.addStretch()

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        dialog.exec()

    @staticmethod
    def _remote_download_relpath(remote_file: str, remote_dir: str) -> str | None:
        """Return a safe relative path for a remote download entry.

        `find <remote_dir> -type f` should only return files inside
        `remote_dir`, but validate the invariant before joining with a local
        destination so malformed remote output cannot escape `local_target`.
        """
        import posixpath

        remote_dir_norm = posixpath.normpath(remote_dir)
        remote_file_norm = posixpath.normpath(remote_file)
        if remote_file_norm == remote_dir_norm:
            return None
        if not remote_file_norm.startswith(remote_dir_norm.rstrip("/") + "/"):
            return None
        rel_path = posixpath.relpath(remote_file_norm, remote_dir_norm)
        if rel_path in ("", ".") or rel_path.startswith("../") or rel_path == ".." or posixpath.isabs(rel_path):
            return None
        return rel_path

    def _download_remote_folder(self, ssh_manager, remote_dir: str, parent_dialog):
        """Download entire remote folder to local."""
        from PySide6.QtCore import Qt

        existing_worker = getattr(self, "_download_worker", None)
        if existing_worker is not None and existing_worker.isRunning():
            QMessageBox.warning(parent_dialog, "Download in Progress", "A remote folder download is already running.")
            return
        if not self._ensure_saved_remote_target_active(ssh_manager):
            return

        # Ask user where to save
        local_dir = QFileDialog.getExistingDirectory(
            parent_dialog, "Select Local Directory to Save Output", os.path.expanduser("~"), QFileDialog.ShowDirsOnly
        )

        if not local_dir:
            return

        # Get folder name from remote path
        folder_name = os.path.basename(remote_dir.rstrip("/"))
        local_target = os.path.join(local_dir, folder_name)
        if os.path.lexists(local_target):
            reply = QMessageBox.question(
                parent_dialog,
                "Replace Existing Download",
                f"A local folder already exists:\n{local_target}\n\n"
                "Replace it after the new download completes? The existing folder is preserved if the download fails or is canceled.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        progress = QProgressDialog("Downloading files from remote server...", "Cancel", 0, 100, parent_dialog)
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        worker = RemoteFolderDownloadWorker(
            ssh_manager, remote_dir, local_target, getattr(self, "_last_run_target_identity", None)
        )
        self._download_worker = worker
        progress.canceled.connect(worker.stop)
        worker.finished.connect(worker.deleteLater)

        def _stop_worker_on_dialog_close(*_args):
            if worker.isRunning():
                worker.stop()

        for signal_name in ("finished", "destroyed"):
            signal = getattr(parent_dialog, signal_name, None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(_stop_worker_on_dialog_close)

        def _on_progress(done: int, total: int, rel_path: str):
            import shiboken6

            if not shiboken6.isValid(progress):
                return
            progress.setMaximum(max(1, total))
            progress.setLabelText(f"Downloading: {rel_path}")
            progress.setValue(done)

        def _on_finished(success: bool, canceled: bool, message: str, target: str):
            if getattr(self, "_download_worker", None) is worker:
                self._download_worker = None
            import shiboken6

            if not shiboken6.isValid(progress) or not shiboken6.isValid(parent_dialog):
                return
            progress.close()
            if canceled:
                QMessageBox.information(parent_dialog, "Download Canceled", "Remote folder download was canceled.")
                return
            if not success:
                QMessageBox.warning(parent_dialog, "Download Error", message)
                return
            if message.startswith("No files"):
                QMessageBox.information(parent_dialog, "Empty Directory", message)
                return
            QMessageBox.information(parent_dialog, "Download Complete", f"Successfully downloaded files to:\n{target}")
            try:
                if platform.system() == "Darwin":
                    subprocess.run(["open", target], check=False)
                elif platform.system() == "Windows":
                    os.startfile(target)
                else:
                    subprocess.run(["xdg-open", target], check=False)
            except Exception as e:
                logger.debug("Failed to open downloaded folder: %s", e)

        worker.progress_updated.connect(_on_progress)
        worker.finished_signal.connect(_on_finished)
        worker.start()

    def load_from_config(self):
        """Called when page becomes visible."""
        pass  # Dashboard state is managed separately
