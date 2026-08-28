# -*- coding: utf-8 -*-
"""
Remote runner for executing OpenBench evaluations on remote servers via SSH.

This module mirrors the EvaluationRunner interface but executes commands
remotely using SSHManager for file transfer and command execution.
"""

import os
import json
import re
import shlex
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import QThread, Signal

from openbench.config.loader import collect_include_files
from openbench.remote.ssh import SSHManager, SSHConnectionError
from openbench.gui.runner import RunnerStatus, RunnerProgress, _looks_like_partial_completion


_REMOTE_PGID_PREFIX = "__OPENBENCH_PGID__="


def build_remote_run_command(python_path: str, openbench_path: str, config_path: str, conda_env: str) -> str:
    """Build the remote evaluation invocation with tilde-safe path quoting.

    PYTHONUNBUFFERED=1 keeps output unbuffered for real-time logging. The
    existing CLI check gates the v3 run entry point so invalid configs never
    start an evaluation.
    """
    from openbench.gui.remote_python import quote_remote_path, wrap_with_conda_env

    q_python = quote_remote_path(python_path or "python3")
    q_openbench = quote_remote_path(openbench_path)
    q_config = quote_remote_path(config_path)
    prefix = f"PYTHONUNBUFFERED=1 OPENBENCH_GUI_PROGRESS=1 {q_python} -u -m openbench"
    invocation = f"{prefix} check {q_config} && {prefix} run {q_config}"
    return wrap_with_conda_env(
        f"cd {q_openbench} && {invocation}",
        python_path=python_path,
        conda_env=conda_env,
    )


def _is_safe_remote_temp_dir(path: str) -> bool:
    """Only accept the single /tmp dir shape emitted by our mktemp template."""
    return bool(re.fullmatch(r"/tmp/openbench_wizard_[A-Za-z0-9_-]+", path or ""))


def _marked_remote_temp_dir(stdout: str) -> str:
    prefix = "__OPENBENCH_TMP__="
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            path = line[len(prefix) :].strip()
            return path if _is_safe_remote_temp_dir(path) else ""
    return ""


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
        self._remote_process_group: int | None = None

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
        self._num_statistics = 0
        self._last_progress = 0.0

        # Track completed items to avoid double counting
        self._started_preprocess_tasks = set()
        self._completed_preprocess_tasks = set()
        self._completed_eval_tasks = set()
        self._completed_groupby_tasks = set()
        self._completed_comparison_tasks = set()
        self._completed_statistics_tasks = set()

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

            if not self._apply_remote_num_cores_override():
                return

            # A stop request can arrive while the num_cores patch command is
            # running. Do not start OpenBench after a successful patch if the
            # user has already canceled the run.
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
                if message == "Stopped by user":
                    self._emit_progress(
                        RunnerStatus.STOPPED, self._last_progress, "Stopped", "", "", "Evaluation stopped by user"
                    )
                elif _looks_like_partial_completion([message]):
                    self._emit_progress(RunnerStatus.PARTIAL, self._last_progress, "Partial", "", "", message)
                else:
                    self._emit_progress(RunnerStatus.FAILED, self._last_progress, "Failed", "", "", message)
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
        self._emit_progress(RunnerStatus.STOPPED, self._last_progress, "Stopped", "", "", "Evaluation stopped by user")
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
                "tmp=$(mktemp -d /tmp/openbench_wizard_XXXXXXXXXX) && printf '__OPENBENCH_TMP__=%s\\n' \"$tmp\"",
                timeout=30,
            )

            if exit_code != 0:
                error_msg = f"Failed to create remote temp directory: {stderr}"
                self.log_message.emit(error_msg)
                self.finished_signal.emit(False, error_msg)
                return False

            self._remote_temp_dir = _marked_remote_temp_dir(stdout)
            if not self._remote_temp_dir:
                error_msg = "Failed to create remote temp directory: mktemp returned no safe /tmp/openbench_wizard_* path"
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

            # Also upload the YAML files this config explicitly includes.
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
        """Upload files explicitly referenced by YAML ``!include`` tags.

        Raises:
            Exception: If include collection or upload fails.
        """
        config_root = Path(config_dir).resolve()
        config_path = Path(self.config_path).resolve()

        for local_path in collect_include_files(config_path):
            try:
                relative = local_path.relative_to(config_root)
            except ValueError as exc:
                raise ValueError(
                    f"Included config file is outside the upload root: {local_path} (root: {config_root})"
                ) from exc

            relative_posix = relative.as_posix()
            remote_path = f"{self._remote_temp_dir}/{relative_posix}"
            remote_dir = os.path.dirname(remote_path)
            if remote_dir and remote_dir != self._remote_temp_dir:
                _stdout, stderr, exit_code = self._ssh_manager.execute(
                    f"mkdir -p {shlex.quote(remote_dir)}",
                    timeout=30,
                )
                if exit_code != 0:
                    raise RuntimeError(f"Failed to create remote include directory {remote_dir}: {stderr}")
            self._ssh_manager.upload_file(str(local_path), remote_path)
            self.log_message.emit(f"Uploaded include: {relative_posix}")

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

    def _apply_remote_num_cores_override(self) -> bool:
        """Persist the remote runtime core count into the uploaded config."""
        raw_num_cores = self._remote_config.get("num_cores")
        if raw_num_cores in (None, ""):
            return True
        try:
            num_cores = int(raw_num_cores)
        except (TypeError, ValueError):
            self.finished_signal.emit(False, f"Invalid remote CPU core count: {raw_num_cores!r}")
            return False
        if num_cores <= 0:
            self.finished_signal.emit(False, f"Invalid remote CPU core count: {num_cores}")
            return False
        if not self._remote_config_path:
            return True

        try:
            from openbench.gui.remote_python import build_remote_python_command

            script = f"""
import pathlib
import yaml

path = pathlib.Path({json.dumps(self._remote_config_path)}).expanduser()
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {{}}
if not isinstance(data, dict):
    raise TypeError("OpenBench config root must be a mapping")
project = data.setdefault("project", {{}})
if not isinstance(project, dict):
    raise TypeError("OpenBench config project section must be a mapping")
project["num_cores"] = {num_cores}
general = data.get("general")
if isinstance(general, dict):
    general["num_cores"] = {num_cores}
path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
"""
            command = build_remote_python_command(
                script,
                python_path=self._remote_config.get("python_path", "python3"),
                conda_env=self._remote_config.get("conda_env", ""),
            )
            stdout, stderr, exit_code = self._ssh_manager.execute(
                command,
                timeout=30,
                should_abort=self._is_stop_requested,
            )
            if exit_code != 0:
                detail = stderr or stdout or "unknown error"
                self.finished_signal.emit(False, f"Failed to apply remote CPU core count to config: {detail}")
                return False
            self.log_message.emit(f"Using {num_cores} CPU cores on remote server.")
            return True
        except SSHConnectionError as exc:
            if self._is_stop_requested():
                self._handle_stop()
            else:
                self.finished_signal.emit(False, f"Failed to apply remote CPU core count to config: {exc}")
            return False
        except Exception as exc:
            self.finished_signal.emit(False, f"Failed to apply remote CPU core count to config: {exc}")
            return False

    def _execute_remote_openbench(self) -> tuple:
        """Execute OpenBench on the remote server.

        Returns:
            Tuple of (success: bool, message: str)
        """
        cmd = build_remote_run_command(
            python_path=self._remote_config.get("python_path", "python3"),
            openbench_path=self._remote_config.get("openbench_path", ""),
            config_path=self._remote_config_path,
            conda_env=self._remote_config.get("conda_env", ""),
        )

        self.log_message.emit(f"Executing: {cmd}")

        # Run the CLI in its own process group when `setsid` is available so
        # Stop can terminate ProcessPool/Dask descendants, not only the parent
        # `python -m openbench` process. The foreground wrapper preserves the
        # SSH channel's normal exit-code and streaming behavior.
        grouped_inner = (
            f"printf '{_REMOTE_PGID_PREFIX}%s\\n' \"$$\"; "
            f"exec sh -c {shlex.quote(cmd)}"
        )
        stream_cmd = (
            "if command -v setsid >/dev/null 2>&1; then "
            f"exec setsid sh -c {shlex.quote(grouped_inner)}; "
            f"else exec sh -c {shlex.quote(cmd)}; fi"
        )

        # Execute and stream output
        try:
            progress = self.PROGRESS_INIT
            output_tail = deque(maxlen=5)
            saw_partial_completion = False

            # `execute_stream` yields output lines and `return`s the exit
            # code. We need to capture the StopIteration.value to know
            # whether the remote process succeeded; iterating with `for`
            # discards the return value, so drive the generator manually.
            stream = self._ssh_manager.execute_stream(stream_cmd, should_abort=self._is_stop_requested)
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
                        if self._remote_process_group is None and line.startswith(_REMOTE_PGID_PREFIX):
                            value = line[len(_REMOTE_PGID_PREFIX) :].strip()
                            if value.isdigit() and int(value) > 1:
                                self._remote_process_group = int(value)
                                continue
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
            if self._is_stop_requested():
                self._kill_remote_process()
                return (False, "Stopped by user")
            return (False, self._format_command_context(f"SSH error while running remote command: {e}", cmd))
        except Exception as e:
            return (False, self._format_command_context(f"Execution error while running remote command: {e}", cmd))

    def _kill_remote_process(self):
        """Attempt to kill the remote OpenBench process."""
        try:
            if self._remote_process_group is not None:
                pgid = self._remote_process_group
                self._ssh_manager.execute(
                    f"kill -TERM -{pgid} 2>/dev/null || true; "
                    f"sleep 1; kill -KILL -{pgid} 2>/dev/null || true",
                    timeout=10,
                )
                self.log_message.emit("Sent kill signal to remote process group")
                return
            if not self._remote_config_path:
                self.log_message.emit("Warning: No remote config path available; skipping remote process kill")
                return
            # Match only this run's uploaded config path, not every OpenBench
            # run owned by the same shared HPC account.
            config_pattern = re.escape(self._remote_config_path)
            if self._remote_config_path.startswith("~/"):
                # The run command expands ~/ to the remote home directory before
                # it reaches Python argv; match either the literal or expanded
                # spelling so Stop still finds runs launched from a ~/ project.
                suffix = re.escape(self._remote_config_path[2:])
                config_pattern = f"({config_pattern}|/[^[:space:]]+/{suffix})"
            pattern = f"python.*-m openbench (check|run) .*{config_pattern}"
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
            if not _is_safe_remote_temp_dir(self._remote_temp_dir):
                self.log_message.emit(f"Warning: Refusing to clean unsafe remote directory: {self._remote_temp_dir}")
                return
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
            "started_preprocess_tasks": self._started_preprocess_tasks,
            "completed_preprocess_tasks": self._completed_preprocess_tasks,
            "completed_eval_tasks": self._completed_eval_tasks,
            "completed_groupby_tasks": self._completed_groupby_tasks,
            "completed_comparison_tasks": self._completed_comparison_tasks,
            "completed_statistics_tasks": self._completed_statistics_tasks,
            "total_tasks": self._total_tasks,
            "num_comparisons": self._num_comparisons,
            "num_statistics": self._num_statistics,
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
        num_evaluation_tasks: int | None = None,
        num_statistics: int = 0,
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
        self._num_statistics = max(0, int(num_statistics or 0))

        # Calculate total tasks
        self._total_tasks = 0

        if do_evaluation:
            if num_evaluation_tasks is None:
                num_evaluation_tasks = num_variables * self._num_ref_sources * self._num_sim_sources
            self._total_tasks += max(0, int(num_evaluation_tasks))

        if do_comparison and num_comparisons > 0:
            self._total_tasks += num_comparisons

        if num_groupby > 0:
            self._total_tasks += num_groupby

        if do_statistics:
            self._total_tasks += self._num_statistics

        self._total_tasks = max(1, self._total_tasks)

        # Reset completion tracking
        self._completed_tasks = 0
        self._started_preprocess_tasks = set()
        self._completed_preprocess_tasks = set()
        self._completed_eval_tasks = set()
        self._completed_groupby_tasks = set()
        self._completed_comparison_tasks = set()
        self._completed_statistics_tasks = set()

    def _emit_progress(self, status: RunnerStatus, progress: float, task: str, variable: str, stage: str, message: str):
        """Emit progress signal."""
        self._last_progress = progress
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
