# -*- coding: utf-8 -*-
"""
OpenBench evaluation runner with progress tracking.
"""

import os
import re
import sys
import subprocess
import threading
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QObject, Signal, QThread


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


class EvaluationRunner(QThread):
    """Thread for running OpenBench evaluation."""

    # Progress calculation constants
    PROGRESS_INIT = 5       # Reserve 5% for initialization
    PROGRESS_WORK = 90      # 90% for actual work (5% to 95%)
    PROGRESS_MAX = 95       # Cap at 95% until completion confirmed
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

    def __del__(self):
        """Cleanup subprocess on thread destruction to prevent orphaned processes."""
        self._cleanup_process()

    def _cleanup_process(self):
        """Safely cleanup the subprocess."""
        if self._process is not None:
            try:
                if self._process.poll() is None:  # Process still running
                    self._kill_process_tree()
            except Exception:
                pass  # Ignore errors during cleanup

    def run(self):
        """Run the evaluation."""
        try:
            self._emit_progress(
                RunnerStatus.RUNNING,
                0,
                "Initializing",
                "",
                "Starting",
                "Starting OpenBench evaluation..."
            )
            self.log_message.emit("Starting OpenBench evaluation...")

            # Build command
            # Assumes openbench is in PYTHONPATH or installed
            openbench_path = self._find_openbench_script()

            if not openbench_path:
                error_msg = (
                    "Could not find OpenBench script (openbench/openbench.py).\n\n"
                    "Please ensure OpenBench is installed in one of these locations:\n"
                    "• ~/Desktop/OpenBench/\n"
                    "• ~/Documents/OpenBench/\n"
                    "• ~/OpenBench/\n"
                    "• Or set the path in Run Monitor page"
                )
                self.finished_signal.emit(False, error_msg)
                return

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

            cmd = [
                python_exe,
                openbench_path,
                self.config_path
            ]

            self.log_message.emit(f"Running: {' '.join(cmd)}")

            # Determine project root from the OpenBench script location
            # openbench_path is like: /path/to/OpenBench/openbench/openbench.py
            # project_root should be: /path/to/OpenBench
            project_root = os.path.dirname(os.path.dirname(openbench_path))

            self.log_message.emit(f"Working directory: {project_root}")

            # Start process with clean environment (avoid PyInstaller conflicts)
            env = os.environ.copy()
            # Remove PyInstaller-specific environment variables that can cause conflicts
            for var in ['LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH',
                        '_MEIPASS', '_MEIPASS2', 'TCL_LIBRARY', 'TK_LIBRARY']:
                env.pop(var, None)

            # Force UTF-8 encoding for Python subprocess (fixes Unicode errors on Windows)
            env['PYTHONIOENCODING'] = 'utf-8'
            # Disable output buffering to ensure real-time log display
            env['PYTHONUNBUFFERED'] = '1'

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=project_root,
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace unencodable characters instead of raising error
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
                    self._emit_progress(
                        RunnerStatus.STOPPED, progress,
                        "Stopped", "", "", "Evaluation stopped by user"
                    )
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
                    self.log_message.emit(line)

                    # Parse progress from log
                    progress, var, stage = self._parse_progress(line, progress)
                    self._emit_progress(
                        RunnerStatus.RUNNING,
                        progress,
                        f"{var} - {stage}" if var else "Processing",
                        var,
                        stage,
                        line
                    )

            # Read any remaining buffered output after process terminates
            remaining_output = self._process.stdout.read()
            if remaining_output:
                for line in remaining_output.strip().split('\n'):
                    line = line.strip()
                    if line:
                        self.log_message.emit(line)
                        progress, var, stage = self._parse_progress(line, progress)

            # Check result
            return_code = self._process.wait()

            if return_code == 0:
                self._emit_progress(
                    RunnerStatus.COMPLETED, 100,
                    "Complete", "", "", "Evaluation completed successfully"
                )
                self.finished_signal.emit(True, "Evaluation completed successfully")
            else:
                self._emit_progress(
                    RunnerStatus.FAILED, progress,
                    "Failed", "", "", f"Process exited with code {return_code}"
                )
                self.finished_signal.emit(False, f"Process exited with code {return_code}")

        except Exception as e:
            self._emit_progress(
                RunnerStatus.FAILED, 0,
                "Error", "", "", str(e)
            )
            self.finished_signal.emit(False, str(e))

    def _find_python_interpreter(self) -> Optional[str]:
        """Find a Python interpreter to run OpenBench.

        Returns the Python path or None if no suitable interpreter found.
        User should configure Python path in General settings if auto-detection fails.
        """
        import shutil

        is_windows = sys.platform == 'win32'

        # PRIORITY 0: Use user-configured Python path (from General settings)
        if self.python_path and os.path.exists(self.python_path):
            self.log_message.emit(f"Using Python (configured): {self.python_path}")
            return self.python_path

        # Check if sys.executable is a real Python interpreter (not bundled app)
        if sys.executable and 'python' in sys.executable.lower():
            # Verify it's not the bundled executable
            if os.path.basename(sys.executable).lower() not in ('openbench_wizard.exe', 'openbench_wizard'):
                self.log_message.emit(f"Using Python: {sys.executable}")
                return sys.executable

        # PRIORITY 1: Check active conda environment (CONDA_PREFIX)
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            if is_windows:
                conda_python = os.path.join(conda_prefix, 'python.exe')
            else:
                conda_python = os.path.join(conda_prefix, 'bin', 'python')
            if os.path.exists(conda_python):
                self.log_message.emit(f"Using Python (conda): {conda_python}")
                return conda_python

        # PRIORITY 2: Check common conda/miniforge locations BEFORE system Python
        user_home = os.path.expanduser('~')
        if is_windows:
            conda_paths = [
                os.path.join(user_home, 'anaconda3', 'python.exe'),
                os.path.join(user_home, 'miniconda3', 'python.exe'),
                os.path.join(user_home, 'miniforge3', 'python.exe'),
                os.path.join(user_home, 'Anaconda3', 'python.exe'),
                os.path.join(user_home, 'Miniconda3', 'python.exe'),
            ]
        else:
            conda_paths = [
                os.path.join(user_home, 'miniforge3', 'bin', 'python'),
                os.path.join(user_home, 'miniconda3', 'bin', 'python'),
                os.path.join(user_home, 'anaconda3', 'bin', 'python'),
                '/opt/homebrew/bin/python3',
                '/usr/local/bin/python3',
            ]

        for path in conda_paths:
            if os.path.exists(path):
                self.log_message.emit(f"Using Python: {path}")
                return path

        # PRIORITY 3: Check PATH (skip system Python /usr/bin/python3)
        if is_windows:
            python_names = ['python', 'python3', 'py']
        else:
            python_names = ['python3', 'python', 'python3.11', 'python3.10', 'python3.12']

        for name in python_names:
            path = shutil.which(name)
            if path:
                # Skip system Python on macOS/Linux (usually missing packages)
                if path == '/usr/bin/python3' or path == '/usr/bin/python':
                    continue
                self.log_message.emit(f"Using Python: {path}")
                return path

        # PRIORITY 4: Windows standard Python installations only
        if is_windows:
            common_paths = [
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python', 'Python311', 'python.exe'),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python', 'Python310', 'python.exe'),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python', 'Python312', 'python.exe'),
                r'C:\Python311\python.exe',
                r'C:\Python310\python.exe',
                r'C:\Python312\python.exe',
            ]

            for path in common_paths:
                if path and os.path.exists(path):
                    self.log_message.emit(f"Using Python: {path}")
                    return path

        # No suitable Python found - user needs to configure in General settings
        return None

    def _find_openbench_script(self) -> Optional[str]:
        """Find the OpenBench main script."""
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        home_dir = os.path.expanduser("~")

        # Look for openbench.py in common locations
        possible_paths = [
            # Relative to config file (nml/nml-yaml -> project root -> openbench)
            os.path.join(config_dir, "..", "..", "openbench", "openbench.py"),
            os.path.join(config_dir, "..", "..", "..", "openbench", "openbench.py"),
            os.path.join(config_dir, "..", "openbench", "openbench.py"),
            # Relative to current working directory
            os.path.join(os.getcwd(), "openbench", "openbench.py"),
            # Common user directories
            os.path.join(home_dir, "Desktop", "OpenBench", "openbench", "openbench.py"),
            os.path.join(home_dir, "Documents", "OpenBench", "openbench", "openbench.py"),
            os.path.join(home_dir, "OpenBench", "openbench", "openbench.py"),
            # Check if executable is in OpenBench directory
            os.path.join(os.path.dirname(sys.executable), "..", "openbench", "openbench.py"),
            os.path.join(os.path.dirname(sys.executable), "openbench", "openbench.py"),
        ]

        # Also search in parent directories of config file
        parent = os.path.dirname(config_dir)
        for _ in range(5):  # Search up to 5 levels up
            check_path = os.path.join(parent, "openbench", "openbench.py")
            if check_path not in possible_paths:
                possible_paths.append(check_path)
            parent = os.path.dirname(parent)

        self.log_message.emit("Looking for OpenBench script...")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            self.log_message.emit(f"  Checking: {abs_path}")
            if os.path.exists(abs_path):
                self.log_message.emit(f"  Found: {abs_path}")
                # Save found path for future reference
                self._save_openbench_path(os.path.dirname(os.path.dirname(abs_path)))
                return abs_path

        # Try to load saved path
        saved_path = self._load_openbench_path()
        if saved_path:
            script_path = os.path.join(saved_path, "openbench", "openbench.py")
            if os.path.exists(script_path):
                self.log_message.emit(f"  Using saved path: {script_path}")
                return script_path

        self.log_message.emit("OpenBench script not found in any expected location")
        return None

    def _get_config_file_path(self) -> str:
        """Get path to wizard config file."""
        home_dir = os.path.expanduser("~")
        config_dir = os.path.join(home_dir, ".openbench_wizard")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "config.txt")

    def _save_openbench_path(self, path: str):
        """Save OpenBench directory path for future use."""
        try:
            config_file = self._get_config_file_path()
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(path)
        except Exception as e:
            self.log_message.emit(f"Warning: Could not save OpenBench path: {e}")

    def _load_openbench_path(self) -> Optional[str]:
        """Load saved OpenBench directory path."""
        try:
            config_file = self._get_config_file_path()
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    path = f.read().strip()
                    if os.path.exists(path):
                        return path
        except Exception:
            pass
        return None

    def _parse_progress(self, line: str, current_progress: float) -> tuple:
        """Parse progress from log line with detailed task tracking."""
        var = self._current_variable
        stage = ""

        line_lower = line.lower()

        # Detect variable being processed
        # Patterns: "Processing Variable_Name", "Evaluating Variable_Name"
        if "processing" in line_lower or "evaluating" in line_lower:
            for keyword in ["Processing", "Evaluating", "processing", "evaluating"]:
                if keyword in line:
                    parts = line.split(keyword)
                    if len(parts) > 1:
                        remaining = parts[1].strip()
                        if remaining:
                            var_name = remaining.split()[0].strip('.:,')
                            if var_name and len(var_name) > 2:
                                self._current_variable = var_name
                                var = var_name
                        break

        # Detect reference/simulation source being processed
        # Patterns: "ref_source: FLUXNET", "ref: FLUXNET", "sim_source: CLM5", "sim: 01_case"
        if "ref_source" in line_lower or "reference" in line_lower or " ref:" in line_lower or "- ref:" in line_lower:
            # Try to extract ref source from patterns like "- ref: SOURCE_NAME"
            if " ref:" in line or "- ref:" in line:
                match = re.search(r'[-\s]ref:\s*(\S+)', line)
                if match:
                    self._current_ref = match.group(1).strip(',:')
            else:
                parts = line.split(":")
                if len(parts) > 1:
                    self._current_ref = parts[-1].strip().split()[0] if parts[-1].strip() else ""
        if "sim_source" in line_lower or "simulation" in line_lower or " sim:" in line_lower or "- sim:" in line_lower:
            # Try to extract sim source from patterns like "- sim: SOURCE_NAME"
            if " sim:" in line or "- sim:" in line:
                match = re.search(r'[-\s]sim:\s*(\S+)', line)
                if match:
                    self._current_sim = match.group(1).strip(',:')
            else:
                parts = line.split(":")
                if len(parts) > 1:
                    self._current_sim = parts[-1].strip().split()[0] if parts[-1].strip() else ""

        # Detect stage
        if "evaluation" in line_lower and "item" not in line_lower:
            stage = "Evaluation"
        elif "comparison" in line_lower or "groupby" in line_lower:
            stage = "Comparison"
            # Track comparison task progress
            # Pattern: "Processing X comparison" starts a comparison
            # Pattern: "Done running X comparison" completes a comparison
            if "done running" in line_lower and "comparison" in line_lower:
                # Extract comparison name from "Done running X comparison"
                match = re.search(r'done running\s+(\w+)\s+comparison', line_lower)
                if match:
                    comp_name = match.group(1)
                    if comp_name not in self._completed_comparison_tasks:
                        self._completed_comparison_tasks.add(comp_name)
        elif "statistic" in line_lower:
            stage = "Statistics"

        # Detect task completions
        task_completed = False

        # Evaluation task completion
        # Patterns: "Evaluation completed for X", "finished evaluating X"
        if stage == "Evaluation" and ("completed" in line_lower or "finished" in line_lower or "done" in line_lower):
            task_key = (self._current_variable, self._current_ref, self._current_sim)
            if task_key not in self._completed_eval_tasks and self._current_variable:
                self._completed_eval_tasks.add(task_key)
                task_completed = True

        # Groupby task completion
        # Patterns: "IGBP groupby completed", "PFT analysis done"
        for groupby_type in ["igbp", "pft", "climate", "landcover"]:
            if groupby_type in line_lower and ("completed" in line_lower or "finished" in line_lower or "done" in line_lower):
                task_key = (self._current_variable, groupby_type)
                if task_key not in self._completed_groupby_tasks:
                    self._completed_groupby_tasks.add(task_key)
                    task_completed = True

        # Comparison/Statistics task completion
        if stage == "Statistics" and ("completed" in line_lower or "finished" in line_lower):
            # Try to extract comparison name
            comp_name = self._current_variable or "comparison"
            if comp_name not in self._completed_comparison_tasks:
                self._completed_comparison_tasks.add(comp_name)
                task_completed = True

        # Calculate progress using class constants
        if self._total_tasks > 0:
            # Count completed tasks from all categories
            total_completed = (
                len(self._completed_eval_tasks) +
                len(self._completed_groupby_tasks) +
                len(self._completed_comparison_tasks)
            )
            # Use max(1, ...) for extra safety against division by zero
            task_progress = (total_completed / max(1, self._total_tasks)) * self.PROGRESS_WORK
            current_progress = min(self.PROGRESS_INIT + task_progress, self.PROGRESS_MAX)
        elif self._num_comparisons > 0 and len(self._completed_comparison_tasks) > 0:
            # Fallback for comparison phase
            comparison_progress = (len(self._completed_comparison_tasks) / max(1, self._num_comparisons)) * self.PROGRESS_WORK
            current_progress = min(self.PROGRESS_INIT + comparison_progress, self.PROGRESS_MAX)
        elif self._num_variables > 0:
            # Fallback to simple variable counting
            completed_vars = len(set(t[0] for t in self._completed_eval_tasks if t[0]))
            # Use max(1, ...) for extra safety against division by zero
            variable_progress = (completed_vars / max(1, self._num_variables)) * self.PROGRESS_WORK
            current_progress = min(self.PROGRESS_INIT + variable_progress, self.PROGRESS_MAX)
        else:
            # Last resort: increment based on activity
            if task_completed or stage or "complete" in line_lower or "done" in line_lower:
                current_progress = min(current_progress + self.PROGRESS_INCREMENT * 2, self.PROGRESS_MAX)
            # Slow increment during comparison phase to show activity
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
        do_statistics: bool = False
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

    def _emit_progress(
        self,
        status: RunnerStatus,
        progress: float,
        task: str,
        variable: str,
        stage: str,
        message: str
    ):
        """Emit progress signal."""
        self.progress_updated.emit(RunnerProgress(
            status=status,
            progress=progress,
            current_task=task,
            current_variable=variable,
            current_stage=stage,
            message=message
        ))

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
            pass

        # Kill the main process (SIGKILL on Unix, TerminateProcess on Windows)
        try:
            self._process.kill()
        except Exception:
            # Fallback to terminate if kill fails
            try:
                self._process.terminate()
            except Exception:
                pass

    def stop(self):
        """Request stop (thread-safe)."""
        with self._stop_lock:
            self._stop_requested = True
        self._kill_process_tree()
