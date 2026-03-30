# -*- coding: utf-8 -*-
"""
Remote runner for executing OpenBench evaluations on remote servers via SSH.

This module mirrors the EvaluationRunner interface but executes commands
remotely using SSHManager for file transfer and command execution.
No Qt dependency — uses plain threading.Thread and callback pattern.
"""

import os
import re
import shlex
import threading
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from openbench.remote.ssh import SSHManager, SSHConnectionError


class RunnerStatus(Enum):
    """Runner status enum."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class RunnerProgress:
    """Progress information."""

    status: RunnerStatus
    progress: float  # 0-100
    current_task: str
    current_variable: str
    current_stage: str
    message: str


class RemoteRunner(threading.Thread):
    """Thread for running OpenBench evaluation on a remote server.

    This class provides the same interface as EvaluationRunner but executes
    the evaluation on a remote server via SSH. It handles:
    - Creating a temporary directory on the remote server
    - Uploading config files via SFTP
    - Executing OpenBench on the remote server
    - Streaming logs back in real-time
    - Handling completion and errors

    Callbacks replace Qt signals:
        _on_progress(RunnerProgress)  -- progress updates
        _on_log(str)                  -- log messages
        _on_finished(bool, str)       -- completion (success, message)
    """

    # Progress calculation constants (same as EvaluationRunner)
    PROGRESS_INIT = 5  # Reserve 5% for initialization
    PROGRESS_WORK = 90  # 90% for actual work (5% to 95%)
    PROGRESS_MAX = 95  # Cap at 95% until completion confirmed
    PROGRESS_INCREMENT = 0.5  # Slow increment when no task info available

    def __init__(
        self,
        config_path: str,
        ssh_manager: SSHManager,
        remote_config: Dict[str, Any],
        config_already_remote: bool = False,
    ):
        """Initialize the remote runner.

        Args:
            config_path: Path to the OpenBench config file.
                - If config_already_remote=True, this is the remote path.
                - If config_already_remote=False, this is the local path to upload.
            ssh_manager: Connected SSHManager instance.
            remote_config: Remote configuration dictionary containing:
                - python_path: Path to Python interpreter on remote server
                - conda_env: Conda environment name (optional)
                - openbench_path: Path to OpenBench installation on remote server
            config_already_remote: If True, config_path is already on remote server.
        """
        super().__init__(daemon=True)
        self.config_path = config_path
        self._ssh_manager = ssh_manager
        self._remote_config = remote_config
        self._config_already_remote = config_already_remote
        self._stop_requested = False
        self._stop_lock = threading.Lock()

        # Callbacks (replace Qt signals)
        self._on_progress: Optional[Callable[[RunnerProgress], None]] = None
        self._on_log: Optional[Callable[[str], None]] = None
        self._on_finished: Optional[Callable[[bool, str], None]] = None

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
            if self._on_log:
                self._on_log("Starting remote OpenBench evaluation...")

            # Validate SSH connection
            if not self._ssh_manager or not self._ssh_manager.is_connected:
                error_msg = "SSH connection not established. Please connect to the remote server first."
                if self._on_finished:
                    self._on_finished(False, error_msg)
                return

            # Validate remote configuration
            python_path = self._remote_config.get("python_path", "")
            openbench_path = self._remote_config.get("openbench_path", "")

            if not python_path:
                error_msg = "Remote Python path not configured. Please configure in General Settings."
                if self._on_finished:
                    self._on_finished(False, error_msg)
                return

            if not openbench_path:
                error_msg = "Remote OpenBench path not configured. Please configure in General Settings."
                if self._on_finished:
                    self._on_finished(False, error_msg)
                return

            # Check for stop request
            if self._is_stop_requested():
                self._handle_stop()
                return

            # Skip upload steps if config is already on remote
            if self._config_already_remote:
                if self._on_log:
                    self._on_log(f"Using remote config: {self._remote_config_path}")
            else:
                # Step 1: Create remote temp directory
                self._emit_progress(
                    RunnerStatus.RUNNING, 2, "Setup", "", "Creating directory", "Creating remote temporary directory..."
                )
                if self._on_log:
                    self._on_log("Creating remote temporary directory...")

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
                if self._on_log:
                    self._on_log("Uploading configuration file...")

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
                if self._on_finished:
                    self._on_finished(True, "Evaluation completed successfully")
            else:
                self._emit_progress(RunnerStatus.FAILED, self.PROGRESS_MAX, "Failed", "", "", message)
                if self._on_finished:
                    self._on_finished(False, message)

        except SSHConnectionError as e:
            error_msg = f"SSH connection error: {e}"
            self._emit_progress(RunnerStatus.FAILED, 0, "Error", "", "", error_msg)
            if self._on_finished:
                self._on_finished(False, error_msg)

        except Exception as e:
            error_msg = f"Remote execution error: {e}"
            self._emit_progress(RunnerStatus.FAILED, 0, "Error", "", "", error_msg)
            if self._on_finished:
                self._on_finished(False, error_msg)

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
        if self._on_finished:
            self._on_finished(False, "Stopped by user")

    def _create_remote_temp_dir(self) -> bool:
        """Create a temporary directory on the remote server.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create temp directory under /tmp with unique name
            timestamp = int(time.time())
            temp_name = f"openbench_wizard_{timestamp}"
            self._remote_temp_dir = f"/tmp/{temp_name}"

            quoted_dir = shlex.quote(self._remote_temp_dir)
            stdout, stderr, exit_code = self._ssh_manager.execute(f"mkdir -p {quoted_dir}", timeout=30)

            if exit_code != 0:
                error_msg = f"Failed to create remote temp directory: {stderr}"
                if self._on_log:
                    self._on_log(error_msg)
                if self._on_finished:
                    self._on_finished(False, error_msg)
                return False

            if self._on_log:
                self._on_log(f"Created remote directory: {self._remote_temp_dir}")
            return True

        except Exception as e:
            error_msg = f"Failed to create remote temp directory: {e}"
            if self._on_log:
                self._on_log(error_msg)
            if self._on_finished:
                self._on_finished(False, error_msg)
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
            if self._on_log:
                self._on_log(f"Uploaded config to: {self._remote_config_path}")

            # Also upload any additional files in the same directory
            # (e.g., included YAML files or data mappings)
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                self._upload_related_files(config_dir)

            return True

        except Exception as e:
            error_msg = f"Failed to upload config file: {e}"
            if self._on_log:
                self._on_log(error_msg)
            if self._on_finished:
                self._on_finished(False, error_msg)
            return False

    def _upload_related_files(self, config_dir: str):
        """Upload related YAML/JSON files from the config directory.

        Args:
            config_dir: Local directory containing config files
        """
        try:
            # Upload any additional .yaml/.yml/.json files in the config directory
            for filename in os.listdir(config_dir):
                if filename.endswith((".yaml", ".yml", ".json")):
                    local_path = os.path.join(config_dir, filename)
                    if os.path.isfile(local_path) and local_path != self.config_path:
                        remote_path = f"{self._remote_temp_dir}/{filename}"
                        try:
                            self._ssh_manager.upload_file(local_path, remote_path)
                            if self._on_log:
                                self._on_log(f"Uploaded: {filename}")
                        except Exception as e:
                            if self._on_log:
                                self._on_log(f"Warning: Could not upload {filename}: {e}")
        except Exception as e:
            if self._on_log:
                self._on_log(f"Warning: Could not scan config directory: {e}")

    def _execute_remote_openbench(self) -> tuple:
        """Execute OpenBench on the remote server.

        Returns:
            Tuple of (success: bool, message: str)
        """
        python_path = self._remote_config.get("python_path", "python3")
        conda_env = self._remote_config.get("conda_env", "")
        openbench_path = self._remote_config.get("openbench_path", "")

        # Build the OpenBench script path
        openbench_script = f"{openbench_path}/openbench/openbench.py"

        # Build the command with unbuffered output for real-time logging
        # PYTHONUNBUFFERED=1 ensures output is not buffered
        if conda_env:
            # Derive conda base from python path (e.g., /path/to/miniconda3/bin/python -> /path/to/miniconda3)
            # This works for paths like: .../miniconda3/bin/python or .../miniconda3/envs/myenv/bin/python
            conda_base_match = re.search(r"(.*?/(?:miniconda|miniforge|anaconda|mambaforge)[^/]*)", python_path)
            if conda_base_match:
                conda_base = conda_base_match.group(1)
                cmd = f"source {conda_base}/etc/profile.d/conda.sh && conda activate {conda_env} && cd {openbench_path} && PYTHONUNBUFFERED=1 {python_path} -u {openbench_script} {self._remote_config_path}"
            else:
                # Fallback: try using bash login shell to get conda in PATH
                cmd = f"bash -l -c 'conda activate {conda_env} && cd {openbench_path} && PYTHONUNBUFFERED=1 {python_path} -u {openbench_script} {self._remote_config_path}'"
        else:
            cmd = f"cd {openbench_path} && PYTHONUNBUFFERED=1 {python_path} -u {openbench_script} {self._remote_config_path}"

        if self._on_log:
            self._on_log(f"Executing: {cmd}")

        # Execute and stream output
        try:
            progress = self.PROGRESS_INIT

            # Use execute_stream to get real-time output
            for line in self._ssh_manager.execute_stream(cmd):
                # Check for stop request
                if self._is_stop_requested():
                    # Try to kill remote process
                    self._kill_remote_process()
                    return (False, "Stopped by user")

                line = line.rstrip("\n\r")
                if line:
                    if self._on_log:
                        self._on_log(line)

                    # Parse progress from log
                    progress, var, stage = self._parse_progress(line, progress)
                    self._emit_progress(
                        RunnerStatus.RUNNING, progress, f"{var} - {stage}" if var else "Processing", var, stage, line
                    )

            return (True, "Completed")

        except SSHConnectionError as e:
            return (False, f"SSH error: {e}")
        except Exception as e:
            return (False, f"Execution error: {e}")

    def _kill_remote_process(self):
        """Attempt to kill the remote OpenBench process."""
        try:
            # Try to find and kill the Python process running openbench
            self._ssh_manager.execute("pkill -f 'openbench.py' || true", timeout=10)
            if self._on_log:
                self._on_log("Sent kill signal to remote process")
        except Exception as e:
            if self._on_log:
                self._on_log(f"Warning: Could not kill remote process: {e}")

    def _cleanup_remote(self):
        """Clean up the remote temporary directory."""
        # Only cleanup if we created a temp directory (not if config was already remote)
        if self._remote_temp_dir and not self._config_already_remote:
            try:
                quoted_dir = shlex.quote(self._remote_temp_dir)
                self._ssh_manager.execute(f"rm -rf {quoted_dir}", timeout=30)
                if self._on_log:
                    self._on_log(f"Cleaned up remote directory: {self._remote_temp_dir}")
            except Exception as e:
                if self._on_log:
                    self._on_log(f"Warning: Could not clean up remote directory: {e}")

    def _parse_progress(self, line: str, current_progress: float) -> tuple:
        """Parse progress from log line with detailed task tracking.

        This method mirrors EvaluationRunner._parse_progress() for consistency.

        Args:
            line: Log line to parse
            current_progress: Current progress value

        Returns:
            Tuple of (progress, variable, stage)
        """
        var = self._current_variable
        stage = ""

        line_lower = line.lower()

        # Detect variable being processed
        if "processing" in line_lower or "evaluating" in line_lower:
            for keyword in ["Processing", "Evaluating", "processing", "evaluating"]:
                if keyword in line:
                    parts = line.split(keyword)
                    if len(parts) > 1:
                        remaining = parts[1].strip()
                        if remaining:
                            var_name = remaining.split()[0].strip(".:,")
                            if var_name and len(var_name) > 2:
                                self._current_variable = var_name
                                var = var_name
                    break

        # Detect reference/simulation source being processed
        if "ref_source" in line_lower or "reference" in line_lower or " ref:" in line_lower:
            if " ref:" in line:
                match = re.search(r"[-\s]ref:\s*(\S+)", line)
                if match:
                    self._current_ref = match.group(1).strip(",:")
            else:
                parts = line.split(":")
                if len(parts) > 1:
                    self._current_ref = parts[-1].strip().split()[0] if parts[-1].strip() else ""

        if "sim_source" in line_lower or "simulation" in line_lower or " sim:" in line_lower:
            if " sim:" in line:
                match = re.search(r"[-\s]sim:\s*(\S+)", line)
                if match:
                    self._current_sim = match.group(1).strip(",:")
            else:
                parts = line.split(":")
                if len(parts) > 1:
                    self._current_sim = parts[-1].strip().split()[0] if parts[-1].strip() else ""

        # Detect stage
        if "evaluation" in line_lower and "item" not in line_lower:
            stage = "Evaluation"
        elif "comparison" in line_lower or "groupby" in line_lower:
            stage = "Comparison"
            if "done running" in line_lower and "comparison" in line_lower:
                match = re.search(r"done running\s+(\w+)\s+comparison", line_lower)
                if match:
                    comp_name = match.group(1)
                    if comp_name not in self._completed_comparison_tasks:
                        self._completed_comparison_tasks.add(comp_name)
        elif "statistic" in line_lower:
            stage = "Statistics"

        # Detect task completions
        task_completed = False

        if stage == "Evaluation" and ("completed" in line_lower or "finished" in line_lower or "done" in line_lower):
            task_key = (self._current_variable, self._current_ref, self._current_sim)
            if task_key not in self._completed_eval_tasks and self._current_variable:
                self._completed_eval_tasks.add(task_key)
                task_completed = True

        # Groupby task completion
        for groupby_type in ["igbp", "pft", "climate", "landcover"]:
            if groupby_type in line_lower and (
                "completed" in line_lower or "finished" in line_lower or "done" in line_lower
            ):
                task_key = (self._current_variable, groupby_type)
                if task_key not in self._completed_groupby_tasks:
                    self._completed_groupby_tasks.add(task_key)
                    task_completed = True

        if stage == "Statistics" and ("completed" in line_lower or "finished" in line_lower):
            comp_name = self._current_variable or "comparison"
            if comp_name not in self._completed_comparison_tasks:
                self._completed_comparison_tasks.add(comp_name)
                task_completed = True

        # Calculate progress
        if self._total_tasks > 0:
            total_completed = (
                len(self._completed_eval_tasks)
                + len(self._completed_groupby_tasks)
                + len(self._completed_comparison_tasks)
            )
            task_progress = (total_completed / max(1, self._total_tasks)) * self.PROGRESS_WORK
            current_progress = min(self.PROGRESS_INIT + task_progress, self.PROGRESS_MAX)
        elif self._num_comparisons > 0 and len(self._completed_comparison_tasks) > 0:
            comparison_progress = (
                len(self._completed_comparison_tasks) / max(1, self._num_comparisons)
            ) * self.PROGRESS_WORK
            current_progress = min(self.PROGRESS_INIT + comparison_progress, self.PROGRESS_MAX)
        elif self._num_variables > 0:
            completed_vars = len(set(t[0] for t in self._completed_eval_tasks if t[0]))
            variable_progress = (completed_vars / max(1, self._num_variables)) * self.PROGRESS_WORK
            current_progress = min(self.PROGRESS_INIT + variable_progress, self.PROGRESS_MAX)
        else:
            if task_completed or stage or "complete" in line_lower or "done" in line_lower:
                current_progress = min(current_progress + self.PROGRESS_INCREMENT * 2, self.PROGRESS_MAX)
            elif stage == "Comparison":
                current_progress = min(current_progress + self.PROGRESS_INCREMENT, self.PROGRESS_MAX)

        return current_progress, var, stage

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
        """Invoke progress callback."""
        if self._on_progress:
            self._on_progress(
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
