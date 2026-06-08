# -*- coding: utf-8 -*-
"""
OpenBench evaluation runner with progress tracking.
"""

import os
import sys
import subprocess
import threading
from collections import deque
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Signal, QThread


class RunnerStatus(Enum):
    """Runner status enum."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    STOPPED = "stopped"


def _looks_like_partial_completion(lines) -> bool:
    """Return True when CLI output represents a partial OpenBench run.

    The CLI exits non-zero for both hard failures and partial runs so that
    automation notices errors.  The GUI should still surface the runner-level
    status distinctly instead of flattening "some tasks failed after others
    succeeded" into an ordinary process failure.
    """
    return any("Evaluation completed with errors" in str(line) for line in lines)


@dataclass
class RunnerProgress:
    """Progress information."""

    status: RunnerStatus
    progress: float  # 0-100
    current_task: str
    current_variable: str
    current_stage: str
    message: str


class EvaluationRunner(QThread):
    """Thread for running OpenBench evaluation."""

    # Progress calculation constants
    PROGRESS_INIT = 5  # Reserve 5% for initialization
    PROGRESS_WORK = 90  # 90% for actual work (5% to 95%)
    PROGRESS_MAX = 95  # Cap at 95% until completion confirmed
    PROGRESS_INCREMENT = 0.5  # Slow increment when no task info available

    progress_updated = Signal(object)  # RunnerProgress
    log_message = Signal(str)
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, config_path: str, python_path: str = "", parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.python_path = python_path  # User-configured Python path
        self._stop_requested = False
        self._stop_lock = threading.Lock()  # Lock for thread-safe stop flag access
        self._process: Optional[subprocess.Popen] = None

        # Cleanup runs on the Qt event loop when the QThread reports finished,
        # not from Python's GC thread. Replaces the old __del__ pattern which
        # could fire during interpreter shutdown (psutil module unloaded) and
        # carried a pid-reuse risk if the subprocess had already exited.
        self.finished.connect(self._cleanup_process)

        # Progress tracking
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
        self._num_groupby = 0  # IGBP, PFT, Climate zone
        self._num_comparisons = 0
        self._do_evaluation = True
        self._do_comparison = False
        self._do_statistics = False

        # Track completed items to avoid double counting
        self._completed_eval_tasks = set()  # (var, ref, sim) tuples
        self._completed_groupby_tasks = set()  # (var, groupby_type) tuples
        self._completed_comparison_tasks = set()  # comparison names

    @staticmethod
    def _format_failure_with_tail(message: str, output_tail) -> str:
        """Append recent subprocess output to a failure message when available."""
        tail = [line for line in output_tail if line]
        if not tail:
            return message
        return f"{message}\n\nRecent output:\n" + "\n".join(tail)

    @staticmethod
    def _format_command_context(message: str, cmd: list[str] | None) -> str:
        """Append local command context to unexpected runner errors."""
        if not cmd:
            return message
        return f"{message}\n\nCommand: {' '.join(cmd)}"

    def _cleanup_process(self):
        """Safely cleanup the subprocess. Idempotent; called from the
        Qt event loop via the finished signal and from run()'s finally
        block. Does nothing if the subprocess has already exited."""
        if self._process is not None:
            try:
                if self._process.poll() is None:  # Process still running
                    self._kill_process_tree()
            except Exception as exc:
                self.log_message.emit(f"Warning: error during subprocess cleanup: {exc}")

    def run(self):
        """Run the evaluation."""
        cmd = None
        output_tail = deque(maxlen=5)
        saw_partial_completion = False
        try:
            self._emit_progress(
                RunnerStatus.RUNNING, 0, "Initializing", "", "Starting", "Starting OpenBench evaluation..."
            )
            self.log_message.emit("Starting OpenBench evaluation...")

            # Find Python interpreter (not the bundled executable)
            python_exe = self._find_python_interpreter()

            if not python_exe:
                error_msg = (
                    "Could not find a suitable Python interpreter.\n\n"
                    "Please configure the Python path in General Settings:\n"
                    "1. Go to 'General Settings' page\n"
                    "2. In 'Runtime Environment' section, select or browse for Python\n"
                    "3. Choose a Python with numpy, scipy, and other required packages\n\n"
                    "Note: System Python (/usr/bin/python3) is not used as it typically lacks required packages."
                )
                self.finished_signal.emit(False, error_msg)
                return

            cmd = [python_exe, "-m", "openbench", "run", self.config_path]

            self.log_message.emit(f"Running: {' '.join(cmd)}")

            # Determine project root from the config file location
            project_root = os.path.dirname(os.path.abspath(self.config_path))

            self.log_message.emit(f"Working directory: {project_root}")

            # Start process with clean environment (avoid PyInstaller conflicts)
            env = os.environ.copy()
            # Remove PyInstaller-specific environment variables that can cause conflicts
            for var in [
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
                "DYLD_FALLBACK_LIBRARY_PATH",
                "_MEIPASS",
                "_MEIPASS2",
                "TCL_LIBRARY",
                "TK_LIBRARY",
            ]:
                env.pop(var, None)

            # Force UTF-8 encoding for Python subprocess (fixes Unicode errors on Windows)
            env["PYTHONIOENCODING"] = "utf-8"
            # Disable output buffering to ensure real-time log display
            env["PYTHONUNBUFFERED"] = "1"

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=project_root,
                env=env,
                encoding="utf-8",
                errors="replace",  # Replace unencodable characters instead of raising error
            )

            # Read output in real-time
            progress = 0

            # Verify process started successfully
            if self._process is None or self._process.stdout is None:
                self.finished_signal.emit(False, "Failed to start OpenBench process")
                return

            while True:
                # Thread-safe check for stop request
                with self._stop_lock:
                    stop_requested = self._stop_requested

                if stop_requested:
                    # Kill the process and all children
                    self._kill_process_tree()
                    self._emit_progress(RunnerStatus.STOPPED, progress, "Stopped", "", "", "Evaluation stopped by user")
                    self.finished_signal.emit(False, "Stopped by user")
                    return

                line = self._process.stdout.readline()
                if not line:
                    # No more output - check if process is still running
                    if self._process.poll() is not None:
                        break
                    # Process still running, just no output yet - continue
                    continue

                line = line.strip()
                if line:
                    output_tail.append(line)
                    saw_partial_completion = saw_partial_completion or _looks_like_partial_completion([line])
                    self.log_message.emit(line)

                    # Parse progress from log
                    progress, var, stage = self._parse_progress(line, progress)
                    self._emit_progress(
                        RunnerStatus.RUNNING, progress, f"{var} - {stage}" if var else "Processing", var, stage, line
                    )

            # Read any remaining buffered output after process terminates
            remaining_output = self._process.stdout.read()
            if remaining_output:
                for line in remaining_output.strip().split("\n"):
                    line = line.strip()
                    if line:
                        output_tail.append(line)
                        saw_partial_completion = saw_partial_completion or _looks_like_partial_completion([line])
                        self.log_message.emit(line)
                        progress, var, stage = self._parse_progress(line, progress)

            # Check result
            return_code = self._process.wait()

            if return_code == 0:
                self._emit_progress(
                    RunnerStatus.COMPLETED, 100, "Complete", "", "", "Evaluation completed successfully"
                )
                self.finished_signal.emit(True, "Evaluation completed successfully")
            else:
                message = self._format_failure_with_tail(f"Process exited with code {return_code}", output_tail)
                if saw_partial_completion:
                    if not _looks_like_partial_completion([message]):
                        message = "Evaluation completed with errors\n" + message
                    self._emit_progress(RunnerStatus.PARTIAL, progress, "Partial", "", "", message)
                else:
                    self._emit_progress(RunnerStatus.FAILED, progress, "Failed", "", "", message)
                self.finished_signal.emit(False, message)

        except Exception as e:
            error_msg = self._format_command_context(f"Local execution error: {e}", cmd)
            self._emit_progress(RunnerStatus.FAILED, 0, "Error", "", "", error_msg)
            self.finished_signal.emit(False, error_msg)
        finally:
            # Ensure subprocess is reaped even if run() exits via early
            # return or an exception path. Prevents orphan processes
            # that the old __del__ tried to clean up unsafely.
            self._cleanup_process()

    def _find_python_interpreter(self) -> Optional[str]:
        """Find a Python interpreter to run OpenBench.

        Returns the Python path or None if no suitable interpreter found.
        User should configure Python path in General settings if auto-detection fails.
        """
        import shutil

        is_windows = sys.platform == "win32"

        # PRIORITY 0: Use user-configured Python path (from General settings)
        if self.python_path and os.path.exists(self.python_path):
            self.log_message.emit(f"Using Python (configured): {self.python_path}")
            return self.python_path

        # Check if sys.executable is a real Python interpreter (not bundled app)
        if sys.executable and "python" in sys.executable.lower():
            # Verify it's not the bundled executable
            if os.path.basename(sys.executable).lower() not in ("openbench_wizard.exe", "openbench_wizard"):
                self.log_message.emit(f"Using Python: {sys.executable}")
                return sys.executable

        # PRIORITY 1: Check active conda environment (CONDA_PREFIX)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            if is_windows:
                conda_python = os.path.join(conda_prefix, "python.exe")
            else:
                conda_python = os.path.join(conda_prefix, "bin", "python")
            if os.path.exists(conda_python):
                self.log_message.emit(f"Using Python (conda): {conda_python}")
                return conda_python

        # PRIORITY 2: Check common conda/miniforge locations BEFORE system Python
        user_home = os.path.expanduser("~")
        if is_windows:
            conda_paths = [
                os.path.join(user_home, "anaconda3", "python.exe"),
                os.path.join(user_home, "miniconda3", "python.exe"),
                os.path.join(user_home, "miniforge3", "python.exe"),
                os.path.join(user_home, "Anaconda3", "python.exe"),
                os.path.join(user_home, "Miniconda3", "python.exe"),
            ]
        else:
            conda_paths = [
                os.path.join(user_home, "miniforge3", "bin", "python"),
                os.path.join(user_home, "miniconda3", "bin", "python"),
                os.path.join(user_home, "anaconda3", "bin", "python"),
                "/opt/homebrew/bin/python3",
                "/usr/local/bin/python3",
            ]

        for path in conda_paths:
            if os.path.exists(path):
                self.log_message.emit(f"Using Python: {path}")
                return path

        # PRIORITY 3: Check PATH (skip system Python /usr/bin/python3)
        if is_windows:
            python_names = ["python", "python3", "py"]
        else:
            python_names = ["python3", "python", "python3.11", "python3.10", "python3.12"]

        for name in python_names:
            path = shutil.which(name)
            if path:
                # Skip system Python on macOS/Linux (usually missing packages)
                if path == "/usr/bin/python3" or path == "/usr/bin/python":
                    continue
                self.log_message.emit(f"Using Python: {path}")
                return path

        # PRIORITY 4: Windows standard Python installations only
        if is_windows:
            common_paths = [
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python311", "python.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python310", "python.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python312", "python.exe"),
                r"C:\Python311\python.exe",
                r"C:\Python310\python.exe",
                r"C:\Python312\python.exe",
            ]

            for path in common_paths:
                if path and os.path.exists(path):
                    self.log_message.emit(f"Using Python: {path}")
                    return path

        # No suitable Python found - user needs to configure in General settings
        return None

    # NOTE: Deleted four v2 helper methods that searched the filesystem
    # for `openbench/openbench.py`:
    #   _find_openbench_script, _get_config_file_path,
    #   _save_openbench_path, _load_openbench_path
    # In v3 the runner invokes the installed module via
    # `python -m openbench run` (see `run()` above) — there is no
    # standalone script to locate. Removing these methods also removes
    # the misleading "OpenBench script not found" log lines that v3
    # users would never have hit.

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
        """
        Set detailed task counts for accurate progress calculation.

        Total tasks formula:
        - Evaluation: variables × ref_sources × sim_sources
        - Groupby: variables × groupby_count × (metrics + scores)
        - Comparisons: num_comparisons

        Args:
            num_variables: Number of evaluation variables (e.g., GPP, ET, etc.)
            num_ref_sources: Number of reference data sources
            num_sim_sources: Number of simulation data sources
            num_metrics: Number of metrics (RMSE, bias, etc.)
            num_scores: Number of scores
            num_groupby: Number of groupby types (IGBP, PFT, Climate zone)
            num_comparisons: Number of comparison tasks
            do_evaluation: Whether evaluation is enabled
            do_comparison: Whether comparison is enabled
            do_statistics: Whether statistics is enabled
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
            # Each variable × each ref source × each sim source
            self._total_tasks += num_variables * self._num_ref_sources * self._num_sim_sources

        if do_comparison:
            # Each comparison type is one task
            if num_comparisons > 0:
                self._total_tasks += num_comparisons
            # Also count groupby tasks if enabled
            if num_groupby > 0:
                metric_score_count = max(1, num_metrics + num_scores)
                self._total_tasks += num_variables * num_groupby * metric_score_count

        if do_statistics and num_comparisons > 0:
            self._total_tasks += num_comparisons

        # Ensure at least 1 task
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

    def _kill_process_tree(self):
        """Kill the process and all its children."""
        if not self._process:
            return

        try:
            # Try to kill child processes first (more thorough termination)
            import psutil

            try:
                parent = psutil.Process(self._process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
        except ImportError:
            # psutil not installed → only the direct subprocess will be
            # killed below; any grandchildren spawned by OpenBench are
            # orphaned. Surface this so packaging / install-doc readers
            # know psutil should be a hard dep for the GUI runner.
            self.log_message.emit(
                "Warning: psutil not installed — cannot recursively kill child "
                "processes. Install psutil to ensure clean shutdown."
            )

        # Kill the main process (SIGKILL on Unix, TerminateProcess on Windows)
        try:
            self._process.kill()
        except Exception as kill_exc:
            # Fallback to terminate if kill fails
            try:
                self._process.terminate()
                self.log_message.emit(f"Warning: process.kill() failed; used terminate(): {kill_exc}")
            except Exception as terminate_exc:
                self.log_message.emit(
                    f"Warning: could not stop OpenBench process: kill failed ({kill_exc}); "
                    f"terminate failed ({terminate_exc})"
                )

    def stop(self):
        """Request stop (thread-safe)."""
        with self._stop_lock:
            self._stop_requested = True
        self._kill_process_tree()
