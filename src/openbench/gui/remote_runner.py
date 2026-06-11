# -*- coding: utf-8 -*-
"""
Remote runner for executing OpenBench evaluations on remote servers via SSH.

This module mirrors the EvaluationRunner interface but executes commands
remotely using SSHManager for file transfer and command execution.
"""

import os
import re
import shlex
import threading
from collections import deque
from typing import Dict, Any

from PySide6.QtCore import QThread, Signal

from openbench.remote.ssh import SSHManager, SSHConnectionError
from openbench.gui.runner import RunnerStatus, RunnerProgress, _looks_like_partial_completion


class RemoteRunner(QThread):
    """Thread for running OpenBench evaluation on a remote server.

    This class provides the same interface as EvaluationRunner but executes
    the evaluation on a remote server via SSH. It handles:
    - Creating a temporary directory on the remote server
    - Uploading config files via SFTP
    - Executing OpenBench on the remote server
    - Streaming logs back in real-time
    - Handling completion and errors
    """

    # Progress calculation constants (same as EvaluationRunner)
    PROGRESS_INIT = 5  # Reserve 5% for initialization
    PROGRESS_WORK = 90  # 90% for actual work (5% to 95%)
    PROGRESS_MAX = 95  # Cap at 95% until completion confirmed
    PROGRESS_INCREMENT = 0.5  # Slow increment when no task info available

    # Signals - same interface as EvaluationRunner
    progress_updated = Signal(object)  # RunnerProgress
    log_message = Signal(str)
    finished_signal = Signal(bool, str)  # success, message

    def __init__(
        self,
        config_path: str,
        ssh_manager: SSHManager,
        remote_config: Dict[str, Any],
        parent=None,
        config_already_remote: bool = False,
    ):
        """Initialize the remote runner.

        Args:
            config_path: Path to the OpenBench config file
                - If config_already_remote=True, this is the remote path
                - If config_already_remote=False, this is the local path to upload
            ssh_manager: Connected SSHManager instance
            remote_config: Remote configuration dictionary containing:
                - python_path: Path to Python interpreter on remote server
                - conda_env: Conda environment name (optional)
                - openbench_path: Path to OpenBench installation on remote server
            parent: Parent QObject
            config_already_remote: If True, config_path is already on remote server
        """
        super().__init__(parent)
        self.config_path = config_path
        self._ssh_manager = ssh_manager
        self._remote_config = remote_config
        self._config_already_remote = config_already_remote
        self._stop_requested = False
        self._stop_lock = threading.Lock()

        # Remote paths
        self._remote_temp_dir = ""
        self._remote_config_path = config_path if config_already_remote else ""

        # Progress tracking (same as EvaluationRunner)
        self._total_tasks = 0
        self._completed_tasks = 0
        self._current_variable = ""
        self._current_ref = ""
        self._current_sim = ""

        # Task counts for detailed progress
        self._num_variables = 0
        self._num_ref_sources = 0
        self._num_sim_sources = 0
        self._num_metrics = 0
        self._num_scores = 0
        self._num_groupby = 0
        self._num_comparisons = 0
        self._do_evaluation = True
        self._do_comparison = False
        self._do_statistics = False

        # Track completed items to avoid double counting
        self._completed_eval_tasks = set()
        self._completed_groupby_tasks = set()
        self._completed_comparison_tasks = set()

    def run(self):
        """Run the evaluation on the remote server."""
        try:
            self._emit_progress(
                RunnerStatus.RUNNING, 0, "Initializing", "", "Starting", "Preparing remote execution..."
            )
            self.log_message.emit("Starting remote OpenBench evaluation...")

            # Validate SSH connection
            if not self._ssh_manager or not self._ssh_manager.is_connected:
                error_msg = "SSH connection not established. Please connect to the remote server first."
                self.finished_signal.emit(False, error_msg)
                return

            # Validate remote configuration
            python_path = self._remote_config.get("python_path", "")
            openbench_path = self._remote_config.get("openbench_path", "")

            if not python_path:
                error_msg = "Remote Python path not configured. Please configure in General Settings."
                self.finished_signal.emit(False, error_msg)
                return

            if not openbench_path:
                error_msg = "Remote OpenBench path not configured. Please configure in General Settings."
                self.finished_signal.emit(False, error_msg)
                return

            # Check for stop request
            if self._is_stop_requested():
                self._handle_stop()
                return

            # Skip upload steps if config is already on remote
            if self._config_already_remote:
                self.log_message.emit(f"Using remote config: {self._remote_config_path}")
            else:
                # Step 1: Create remote temp directory
                self._emit_progress(
                    RunnerStatus.RUNNING, 2, "Setup", "", "Creating directory", "Creating remote temporary directory..."
                )
                self.log_message.emit("Creating remote temporary directory...")

                if not self._create_remote_temp_dir():
                    return

                # Check for stop request
                if self._is_stop_requested():
                    self._handle_stop()
                    return

                # Step 2: Upload config file
                self._emit_progress(
                    RunnerStatus.RUNNING, 4, "Upload", "", "Uploading config", "Uploading configuration file..."
                )
                self.log_message.emit("Uploading configuration file...")

                if not self._upload_config():
                    return

            # Check for stop request
            if self._is_stop_requested():
                self._handle_stop()
                return

            # Step 3: Execute OpenBench on remote server
            self._emit_progress(
                RunnerStatus.RUNNING, self.PROGRESS_INIT, "Executing", "", "Running", "Starting OpenBench execution..."
            )

            success, message = self._execute_remote_openbench()

            if success:
                self._emit_progress(
                    RunnerStatus.COMPLETED, 100, "Complete", "", "", "Evaluation completed successfully"
                )
                self.finished_signal.emit(True, "Evaluation completed successfully")
            else:
                if _looks_like_partial_completion([message]):
                    self._emit_progress(RunnerStatus.PARTIAL, self.PROGRESS_MAX, "Partial", "", "", message)
                else:
                    self._emit_progress(RunnerStatus.FAILED, self.PROGRESS_MAX, "Failed", "", "", message)
                self.finished_signal.emit(False, message)

        except SSHConnectionError as e:
            error_msg = f"SSH connection error: {e}"
            self._emit_progress(RunnerStatus.FAILED, 0, "Error", "", "", error_msg)
            self.finished_signal.emit(False, error_msg)

        except Exception as e:
            error_msg = f"Remote execution error: {e}"
            self._emit_progress(RunnerStatus.FAILED, 0, "Error", "", "", error_msg)
            self.finished_signal.emit(False, error_msg)

        finally:
            # Cleanup remote temp directory
            self._cleanup_remote()

    def _is_stop_requested(self) -> bool:
        """Thread-safe check for stop request."""
        with self._stop_lock:
            return self._stop_requested

    def _handle_stop(self):
        """Handle stop request."""
        self._emit_progress(RunnerStatus.STOPPED, 0, "Stopped", "", "", "Evaluation stopped by user")
        self.finished_signal.emit(False, "Stopped by user")

    def _create_remote_temp_dir(self) -> bool:
        """Create a temporary directory on the remote server.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a truly unique temp directory.  A second-level timestamp
            # collides for concurrent GUI users sharing one HPC account and
            # cleanup can then remove another run's staging directory.
            stdout, stderr, exit_code = self._ssh_manager.execute(
                "mktemp -d /tmp/openbench_wizard_XXXXXXXXXX",
                timeout=30,
            )

            if exit_code != 0:
                error_msg = f"Failed to create remote temp directory: {stderr}"
                self.log_message.emit(error_msg)
                self.finished_signal.emit(False, error_msg)
                return False

            self._remote_temp_dir = stdout.strip()
            if not self._remote_temp_dir:
                error_msg = "Failed to create remote temp directory: mktemp returned an empty path"
                self.log_message.emit(error_msg)
                self.finished_signal.emit(False, error_msg)
                return False

            self.log_message.emit(f"Created remote directory: {self._remote_temp_dir}")
            return True

        except Exception as e:
            error_msg = f"Failed to create remote temp directory: {e}"
            self.log_message.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
            return False

    def _upload_config(self) -> bool:
        """Upload the config file to the remote server.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get config filename
            config_filename = os.path.basename(self.config_path)
            self._remote_config_path = f"{self._remote_temp_dir}/{config_filename}"

            # Upload the config file
            self._ssh_manager.upload_file(self.config_path, self._remote_config_path)
            self.log_message.emit(f"Uploaded config to: {self._remote_config_path}")

            # Also upload any additional files in the same directory
            # (e.g., included YAML files or data mappings)
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                self._upload_related_files(config_dir)

            return True

        except Exception as e:
            error_msg = f"Failed to upload config file: {e}"
            self.log_message.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
            return False

    def _upload_related_files(self, config_dir: str) -> None:
        """Upload related YAML/JSON files from the config directory.

        Raises:
            OSError: If the config directory cannot be scanned.
            Exception: If any related file upload fails.
        """
        config_path = os.path.abspath(self.config_path)
        related_files = []
        for filename in sorted(os.listdir(config_dir)):
            if filename.endswith((".yaml", ".yml", ".json")):
                local_path = os.path.abspath(os.path.join(config_dir, filename))
                if os.path.isfile(local_path) and local_path != config_path:
                    related_files.append((filename, local_path))

        # Surface the scope of the implicit "upload every sibling YAML/JSON"
        # behavior — if the user is running from a shared configs/ folder
        # with dozens of unrelated projects, this would silently upload all
        # of them. A proper include-graph parse is a larger change; for
        # now, warn loudly when the count looks abnormal.
        if len(related_files) > 10:
            self.log_message.emit(
                f"Warning: uploading {len(related_files)} YAML/JSON files alongside "
                f"{os.path.basename(self.config_path)} from {config_dir}. "
                "Consider isolating the active run's config into its own directory "
                "to avoid uploading unrelated project files."
            )

        for filename, local_path in related_files:
            remote_path = f"{self._remote_temp_dir}/{filename}"
            self._ssh_manager.upload_file(local_path, remote_path)
            self.log_message.emit(f"Uploaded: {filename}")

    @staticmethod
    def _format_remote_failure(message: str, output_tail) -> str:
        """Append recent remote output to a failure message when available."""
        tail = [line for line in output_tail if line]
        if not tail:
            return message
        return f"{message}\n\nRecent output:\n" + "\n".join(tail)

    @staticmethod
    def _format_command_context(message: str, command: str) -> str:
        """Append the remote command to unexpected execution errors."""
        return f"{message}\n\nCommand: {command}"

    def _execute_remote_openbench(self) -> tuple:
        """Execute OpenBench on the remote server.

        Returns:
            Tuple of (success: bool, message: str)
        """
        import shlex

        python_path = self._remote_config.get("python_path", "python3")
        conda_env = self._remote_config.get("conda_env", "")
        openbench_path = self._remote_config.get("openbench_path", "")

        # Quote each user-supplied component so paths with spaces are
        # respected and shell metacharacters in config values cannot
        # inject extra commands.
        q_python = shlex.quote(python_path)
        q_openbench = shlex.quote(openbench_path)
        q_config = shlex.quote(self._remote_config_path)

        # v3 entry point is the installed openbench package; the legacy
        # path (openbench/openbench.py) was removed during the v3.0 repo
        # restructuring and no longer exists, so the remote process must
        # invoke the module directly via "python -m openbench run".
        invocation = f"PYTHONUNBUFFERED=1 {q_python} -u -m openbench run {q_config}"

        # Build the command with unbuffered output for real-time logging.
        # PYTHONUNBUFFERED=1 ensures output is not buffered.
        from openbench.gui.remote_python import wrap_with_conda_env

        cmd = wrap_with_conda_env(
            f"cd {q_openbench} && {invocation}",
            python_path=python_path,
            conda_env=conda_env,
        )

        self.log_message.emit(f"Executing: {cmd}")

        # Execute and stream output
        try:
            progress = self.PROGRESS_INIT
            output_tail = deque(maxlen=5)
            saw_partial_completion = False

            # `execute_stream` yields output lines and `return`s the exit
            # code. We need to capture the StopIteration.value to know
            # whether the remote process succeeded; iterating with `for`
            # discards the return value, so drive the generator manually.
            stream = self._ssh_manager.execute_stream(cmd)
            exit_code = 0
            stopped_by_user = False
            try:
                while True:
                    line = next(stream)
                    if self._is_stop_requested():
                        # Close the generator to signal we don't want
                        # more output, then attempt remote kill.
                        try:
                            stream.close()
                        except Exception as exc:
                            self.log_message.emit(f"Warning: could not close remote output stream: {exc}")
                        self._kill_remote_process()
                        stopped_by_user = True
                        break

                    line = line.rstrip("\n\r")
                    if line:
                        output_tail.append(line)
                        saw_partial_completion = saw_partial_completion or _looks_like_partial_completion([line])
                        self.log_message.emit(line)
                        progress, var, stage = self._parse_progress(line, progress)
                        self._emit_progress(
                            RunnerStatus.RUNNING,
                            progress,
                            f"{var} - {stage}" if var else "Processing",
                            var,
                            stage,
                            line,
                        )
            except StopIteration as stop:
                # Generator finished naturally — its return value is the
                # remote process exit code (paramiko channel.recv_exit_status).
                if isinstance(stop.value, int):
                    exit_code = stop.value
                else:
                    exit_code = 1
                    self.log_message.emit(
                        "Warning: remote process did not report a numeric exit code; treating as failure."
                    )

            if stopped_by_user:
                return (False, "Stopped by user")
            if exit_code == 0:
                return (True, "Completed")
            message = self._format_remote_failure(f"Remote OpenBench exited with code {exit_code}", output_tail)
            if saw_partial_completion and not _looks_like_partial_completion([message]):
                message = "Evaluation completed with errors\n" + message
            return (False, message)

        except SSHConnectionError as e:
            return (False, self._format_command_context(f"SSH error while running remote command: {e}", cmd))
        except Exception as e:
            return (False, self._format_command_context(f"Execution error while running remote command: {e}", cmd))

    def _kill_remote_process(self):
        """Attempt to kill the remote OpenBench process."""
        try:
            if not self._remote_config_path:
                self.log_message.emit("Warning: No remote config path available; skipping remote process kill")
                return
            # Match only this run's uploaded config path, not every OpenBench
            # run owned by the same shared HPC account.
            pattern = f"python.*-m openbench run .*{re.escape(self._remote_config_path)}"
            self._ssh_manager.execute(
                f"pkill -f -- {shlex.quote(pattern)} || true",
                timeout=10,
            )
            self.log_message.emit("Sent kill signal to remote process")
        except Exception as e:
            self.log_message.emit(f"Warning: Could not kill remote process: {e}")

    def _cleanup_remote(self):
        """Clean up the remote temporary directory."""
        # Only cleanup if we created a temp directory (not if config was already remote)
        if self._remote_temp_dir and not self._config_already_remote:
            try:
                quoted_dir = shlex.quote(self._remote_temp_dir)
                stdout, stderr, exit_code = self._ssh_manager.execute(f"rm -rf {quoted_dir}", timeout=30)
                if exit_code == 0:
                    self.log_message.emit(f"Cleaned up remote directory: {self._remote_temp_dir}")
                else:
                    detail = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
                    self.log_message.emit(
                        f"Warning: Could not clean up remote directory {self._remote_temp_dir}: {detail}"
                    )
            except Exception as e:
                self.log_message.emit(f"Warning: Could not clean up remote directory {self._remote_temp_dir}: {e}")

    def _parse_progress(self, line: str, current_progress: float) -> tuple:
        """Parse progress from log line (delegates to shared parser)."""
        from openbench.gui.progress_parser import parse_progress_line

        state = {
            "current_variable": self._current_variable,
            "current_ref": self._current_ref,
            "current_sim": self._current_sim,
            "completed_eval_tasks": self._completed_eval_tasks,
            "completed_groupby_tasks": self._completed_groupby_tasks,
            "completed_comparison_tasks": self._completed_comparison_tasks,
            "total_tasks": self._total_tasks,
            "num_comparisons": self._num_comparisons,
            "num_variables": self._num_variables,
        }
        constants = {
            "PROGRESS_INIT": self.PROGRESS_INIT,
            "PROGRESS_WORK": self.PROGRESS_WORK,
            "PROGRESS_MAX": self.PROGRESS_MAX,
            "PROGRESS_INCREMENT": self.PROGRESS_INCREMENT,
        }
        progress, var, stage = parse_progress_line(line, current_progress, state, constants)
        self._current_variable = state["current_variable"]
        self._current_ref = state["current_ref"]
        self._current_sim = state["current_sim"]
        return progress, var, stage

    def set_total_variables(self, count: int):
        """Set the total number of variables to process (legacy method)."""
        self._num_variables = count

    def set_task_counts(
        self,
        num_variables: int,
        num_ref_sources: int,
        num_sim_sources: int,
        num_metrics: int,
        num_scores: int,
        num_groupby: int,
        num_comparisons: int,
        do_evaluation: bool = True,
        do_comparison: bool = False,
        do_statistics: bool = False,
    ):
        """Set detailed task counts for accurate progress calculation.

        This method mirrors EvaluationRunner.set_task_counts() for consistency.
        """
        self._num_variables = num_variables
        self._num_ref_sources = max(1, num_ref_sources)
        self._num_sim_sources = max(1, num_sim_sources)
        self._num_metrics = num_metrics
        self._num_scores = num_scores
        self._num_groupby = num_groupby
        self._num_comparisons = num_comparisons
        self._do_evaluation = do_evaluation
        self._do_comparison = do_comparison
        self._do_statistics = do_statistics

        # Calculate total tasks
        self._total_tasks = 0

        if do_evaluation:
            self._total_tasks += num_variables * self._num_ref_sources * self._num_sim_sources

        if do_comparison:
            if num_comparisons > 0:
                self._total_tasks += num_comparisons
            if num_groupby > 0:
                metric_score_count = max(1, num_metrics + num_scores)
                self._total_tasks += num_variables * num_groupby * metric_score_count

        if do_statistics and num_comparisons > 0:
            self._total_tasks += num_comparisons

        self._total_tasks = max(1, self._total_tasks)

        # Reset completion tracking
        self._completed_tasks = 0
        self._completed_eval_tasks = set()
        self._completed_groupby_tasks = set()
        self._completed_comparison_tasks = set()

    def _emit_progress(self, status: RunnerStatus, progress: float, task: str, variable: str, stage: str, message: str):
        """Emit progress signal."""
        self.progress_updated.emit(
            RunnerProgress(
                status=status,
                progress=progress,
                current_task=task,
                current_variable=variable,
                current_stage=stage,
                message=message,
            )
        )

    def stop(self):
        """Request stop (thread-safe)."""
        with self._stop_lock:
            self._stop_requested = True
        self._kill_remote_process()
