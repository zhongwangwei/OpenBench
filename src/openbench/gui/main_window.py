# -*- coding: utf-8 -*-
"""
Main window with sidebar navigation and page container.
"""

import logging
import os
import yaml

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QPushButton,
    QLabel,
    QFrame,
    QMessageBox,
    QFileDialog,
    QSplitter,
    QDialog,
)
from PySide6.QtCore import Qt

from openbench.gui.remote_python import quote_remote_path
from openbench.gui.widgets._ssh_worker import execute_responsive
from openbench.gui.controller import WizardController
from openbench.gui.widgets.remote_config import RemoteFileBrowser
from openbench.gui.widgets.sync_status import SyncStatusWidget
from openbench.remote.storage import LocalStorage, RemoteStorage
from openbench.gui.path_utils import (
    get_openbench_root,
    to_absolute_path,
    convert_paths_in_dict,
    validate_paths_in_dict,
)
from openbench.gui.pages import (
    PageGeneral,
    PageRegistry,
    PageEvaluation,
    PageMetrics,
    PageScores,
    PageComparisons,
    PageStatistics,
    PageRefData,
    PageSimData,
    PagePreview,
    PageRunMonitor,
    PageRuntime,
)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""

    # Maximum seconds to wait for an evaluation runner to clean up its
    # subprocess on window close before forcing the close anyway.
    _RUNNER_SHUTDOWN_TIMEOUT_MS = 5000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenBench NML Wizard")
        self.setMinimumSize(1200, 800)

        # Initialize controller
        self.controller = WizardController(self)

        # Set project_root on startup
        self.controller.project_root = get_openbench_root()

        # Sync status widget (created later if needed for remote mode)
        self._sync_status = None
        self._nav_bar_layout = None  # Reference to nav bar layout for sync status

        # Setup UI
        self._setup_ui()
        self._connect_signals()
        self._update_navigation()

        # Initialize with default local storage (project selection via Runtime Environment page)
        self._init_default_storage()

    def _setup_ui(self):
        """Initialize the user interface."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for sidebar and content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === Sidebar ===
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("QWidget#sidebar { background-color: #2d2d2d; }")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Title area
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #252525; padding: 20px;")
        title_layout = QVBoxLayout(title_frame)

        title_label = QLabel("OpenBench")
        title_label.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: bold;")
        title_layout.addWidget(title_label)

        subtitle_label = QLabel("NML Configuration Wizard")
        subtitle_label.setStyleSheet("color: #888888; font-size: 12px;")
        title_layout.addWidget(subtitle_label)

        sidebar_layout.addWidget(title_frame)

        # Navigation list
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav_sidebar")
        self.nav_list.setFocusPolicy(Qt.NoFocus)
        sidebar_layout.addWidget(self.nav_list)

        # Sidebar buttons
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background-color: #252525; padding: 10px;")
        btn_layout = QVBoxLayout(btn_frame)

        self.btn_load = QPushButton("Load Config...")
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: none;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        btn_layout.addWidget(self.btn_load)

        self.btn_new = QPushButton("New Config")
        self.btn_new.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: none;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        btn_layout.addWidget(self.btn_new)

        sidebar_layout.addWidget(btn_frame)

        splitter.addWidget(sidebar)

        # === Content Area ===
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Page stack
        self.page_stack = QStackedWidget()
        self._setup_pages()
        content_layout.addWidget(self.page_stack, 1)

        # Bottom navigation bar
        nav_bar = QFrame()
        nav_bar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-top: 1px solid #e0e0e0;
            }
        """)
        self._nav_bar_layout = QHBoxLayout(nav_bar)
        self._nav_bar_layout.setContentsMargins(20, 15, 20, 15)

        self.btn_back = QPushButton("Back")
        self.btn_back.setProperty("secondary", True)
        self.btn_back.setMinimumWidth(100)
        self._nav_bar_layout.addWidget(self.btn_back)

        self._nav_bar_layout.addStretch()

        # Page indicator
        self.page_indicator = QLabel("Step 1 of 10")
        self.page_indicator.setStyleSheet("color: #666666;")
        self._nav_bar_layout.addWidget(self.page_indicator)

        self._nav_bar_layout.addStretch()

        # Rerun button (only visible on run_monitor page)
        self.btn_rerun = QPushButton("Rerun")
        self.btn_rerun.setMinimumWidth(100)
        self.btn_rerun.setVisible(False)
        self._nav_bar_layout.addWidget(self.btn_rerun)

        self.btn_next = QPushButton("Next")
        self.btn_next.setMinimumWidth(100)
        self._nav_bar_layout.addWidget(self.btn_next)

        content_layout.addWidget(nav_bar)

        splitter.addWidget(content)

        # Set splitter sizes (sidebar: 220px, content: rest)
        splitter.setSizes([220, 980])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

    def _setup_pages(self):
        """Create and add all pages to the stack."""
        self.pages = {}

        page_classes = {
            "general": PageGeneral,
            "registry": PageRegistry,
            "evaluation_items": PageEvaluation,
            "ref_data": PageRefData,
            "sim_data": PageSimData,
            "metrics": PageMetrics,
            "scores": PageScores,
            "comparisons": PageComparisons,
            "statistics": PageStatistics,
            "runtime": PageRuntime,
            "preview": PagePreview,
            "run_monitor": PageRunMonitor,
        }

        for page_id, page_class in page_classes.items():
            page = page_class(self.controller)
            self.pages[page_id] = page
            self.page_stack.addWidget(page)

        # Connect preview page run signal to monitor page
        if "preview" in self.pages and "run_monitor" in self.pages:
            self.pages["preview"].run_requested.connect(self.pages["run_monitor"].start_run)

        # Connect sim page → propagate available variables to ref/eval pages
        if "sim_data" in self.pages:
            self.pages["sim_data"].available_variables_changed.connect(self._on_available_variables_changed)

    def _connect_signals(self):
        """Connect signals to slots."""
        # Navigation buttons
        self.btn_back.clicked.connect(self._on_back_clicked)
        self.btn_next.clicked.connect(self._on_next_clicked)
        self.btn_rerun.clicked.connect(self._on_rerun_clicked)
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_new.clicked.connect(self._on_new_clicked)

        # Sidebar navigation
        self.nav_list.currentRowChanged.connect(self._on_nav_selected)

        # Controller signals
        self.controller.page_changed.connect(self._on_page_changed)
        self.controller.pages_visibility_changed.connect(self._update_navigation)

    def _update_navigation(self):
        """Update sidebar navigation based on visible pages."""
        current = self.controller.current_page
        visible_pages = self.controller.get_visible_pages()

        self.nav_list.blockSignals(True)
        self.nav_list.clear()

        for page_id in self.controller.ALL_PAGES:
            item = QListWidgetItem(self.controller.get_page_name(page_id))
            item.setData(Qt.UserRole, page_id)

            if page_id not in visible_pages or (self._runner_is_active() and page_id != current):
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setForeground(Qt.gray)

            self.nav_list.addItem(item)

            if page_id == current:
                self.nav_list.setCurrentItem(item)

        self.nav_list.blockSignals(False)
        self._update_buttons()
        self._update_page_indicator()

    def closeEvent(self, event):
        """Stop any running evaluation before closing.

        Without this, closing the window while an EvaluationRunner is
        active leaves an orphan OpenBench subprocess running. We ask the
        runner to stop, wait a bounded time for it to terminate, then
        accept the close. If the user cancels via the confirmation
        dialog the close is rejected.
        """
        runner = None
        run_monitor = self.pages.get("run_monitor") if hasattr(self, "pages") else None
        if run_monitor is not None:
            runner = getattr(run_monitor, "_runner", None)

        if runner is not None and hasattr(runner, "isRunning") and runner.isRunning():
            reply = QMessageBox.question(
                self,
                "Evaluation Running",
                "An evaluation is still running. Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

            try:
                if hasattr(runner, "stop"):
                    runner.stop()
                # QThread.wait expects ms; runner.stop sets a flag and the
                # subprocess loop in run() should exit on the next poll.
                if hasattr(runner, "wait"):
                    runner.wait(self._RUNNER_SHUTDOWN_TIMEOUT_MS)
            except Exception as e:
                logger.warning("Error during runner shutdown on close: %s", e)

        super().closeEvent(event)

    def _runner_is_active(self) -> bool:
        """Return True while the run monitor owns a live runner thread."""
        run_monitor = self.pages.get("run_monitor") if hasattr(self, "pages") else None
        runner = getattr(run_monitor, "_runner", None) if run_monitor is not None else None
        return bool(runner is not None and hasattr(runner, "isRunning") and runner.isRunning())

    def _update_buttons(self):
        """Update Back/Next button states."""
        runner_active = self._runner_is_active()
        self.btn_back.setEnabled((self.controller.prev_page() is not None) and not runner_active)

        # Show Rerun button only on run_monitor page
        self.btn_rerun.setVisible(self.controller.current_page == "run_monitor")
        self.btn_rerun.setEnabled(not runner_active)

        next_page = self.controller.next_page()
        if next_page is None:
            self.btn_next.setText("Finish")
        elif self.controller.current_page == "preview":
            self.btn_next.setText("Run")
        else:
            self.btn_next.setText("Next")
        self.btn_next.setEnabled(not runner_active)

    def _update_page_indicator(self):
        """Update the step indicator."""
        visible = self.controller.get_visible_pages()
        current = self.controller.current_page
        try:
            idx = visible.index(current) + 1
            self.page_indicator.setText(f"Step {idx} of {len(visible)}")
        except ValueError:
            self.page_indicator.setText("")

    def _on_available_variables_changed(self, variables: list):
        """Handle sim page reporting which variables are available from model profiles."""
        # Cache as a MainWindow attribute. The previous implementation wrote
        # this into a fake `_internal` section of the controller config dict,
        # which then leaked into any saved YAML and was never actually read
        # back from there (page_sim_data recomputes via _get_available_variables).
        self._available_variables = list(variables)

        # Update evaluation_items: auto-select newly available variables
        eval_items = dict(self.controller.config.get("evaluation_items", {}))
        # Add new variables (default checked), keep existing selections
        for var in variables:
            if var not in eval_items:
                eval_items[var] = True
        # Remove variables no longer available
        for var in list(eval_items.keys()):
            if var not in variables:
                del eval_items[var]
        self.controller.update_section("evaluation_items", eval_items)

    def _on_nav_selected(self, row: int):
        """Handle sidebar navigation selection.

        Sidebar navigation is always free — no validation required.
        Validation only triggers on Next button (forward progression).
        """
        item = self.nav_list.item(row)
        if item and item.flags() & Qt.ItemIsEnabled:
            page_id = item.data(Qt.UserRole)

            if page_id == self.controller.current_page:
                return

            if self._runner_is_active():
                QMessageBox.warning(
                    self,
                    "Evaluation Running",
                    "Stop the running evaluation before leaving the Run & Monitor page.",
                )
                self._restore_nav_selection()
                return

            # Save current page before switching (without triggering sync)
            self._save_current_page(trigger_sync=False)

            self.controller.go_to_page(page_id)

    def _restore_nav_selection(self):
        """Restore sidebar selection to current page after failed validation."""
        current_page_id = self.controller.current_page
        self.nav_list.blockSignals(True)
        for i in range(self.nav_list.count()):
            item = self.nav_list.item(i)
            if item and item.data(Qt.UserRole) == current_page_id:
                self.nav_list.setCurrentItem(item)
                break
        self.nav_list.blockSignals(False)

    def _save_current_page(self, trigger_sync: bool = True):
        """Save the current page's data to config."""
        current_page = self.pages.get(self.controller.current_page)
        if current_page and hasattr(current_page, "save_to_config"):
            # Temporarily disable auto sync if requested
            if not trigger_sync:
                old_auto_sync = self.controller.auto_sync_enabled
                self.controller.auto_sync_enabled = False

            current_page.save_to_config()

            if not trigger_sync:
                self.controller.auto_sync_enabled = old_auto_sync

    def _on_page_changed(self, page_id: str):
        """Handle page change."""
        if page_id in self.pages:
            self.page_stack.setCurrentWidget(self.pages[page_id])
            self._update_navigation()

    def _on_back_clicked(self):
        """Handle Back button click."""
        if self._runner_is_active():
            QMessageBox.warning(
                self,
                "Evaluation Running",
                "Stop the running evaluation before going back.",
            )
            return
        # Save current page before going back (without triggering sync)
        self._save_current_page(trigger_sync=False)
        self.controller.go_prev()

    def _on_next_clicked(self):
        """Handle Next button click."""
        if self._runner_is_active():
            QMessageBox.warning(
                self,
                "Evaluation Running",
                "Stop the running evaluation before navigating or starting another run.",
            )
            return

        # Validate current page before proceeding
        current_page = self.pages.get(self.controller.current_page)
        if current_page and callable(getattr(current_page, "validate", None)):
            if not current_page.validate():
                return

        # Special handling for Preview page - trigger export and run
        if self.controller.current_page == "preview":
            preview_page = self.pages.get("preview")
            if preview_page:
                preview_page.export_and_run()
            return

        if not self.controller.go_next():
            # At the end - ask for confirmation before closing
            reply = QMessageBox.question(
                self,
                "Exit",
                "Are you sure you want to exit the wizard?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.close()

    def _on_rerun_clicked(self):
        """Handle Rerun button click - re-export and run."""
        if self._runner_is_active():
            QMessageBox.warning(
                self,
                "Evaluation Running",
                "Stop the running evaluation before rerunning.",
            )
            return
        preview_page = self.pages.get("preview")
        if preview_page:
            preview_page.export_and_run()

    def _on_load_clicked(self):
        """Handle Load Config button click."""
        # Check if using remote storage (not config, since storage type is authoritative)
        if isinstance(self.controller.storage, RemoteStorage):
            file_path = self._browse_remote_config_file()
        else:
            # Use OpenBench root as default directory
            start_dir = self._get_local_openbench_path()
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Configuration", start_dir, "YAML Files (*.yaml *.yml);;All Files (*)"
            )
        if file_path:
            self._load_config_file(file_path)

    def _get_local_openbench_path(self) -> str:
        """Get local OpenBench path from config or runtime settings."""
        # Try from controller config first
        general = self.controller.config.get("general", {})
        local_path = general.get("local_openbench_path", "")
        if local_path and os.path.isdir(local_path):
            return local_path

        # Try from runtime settings file
        try:
            settings_path = os.path.join(os.path.expanduser("~"), ".openbench_wizard", "runtime_settings.yaml")
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f) or {}
                    saved_path = settings.get("local_openbench_path", "")
                    if saved_path and os.path.isdir(saved_path):
                        return saved_path
        except Exception:
            pass

        # Fall back to project_root or detected OpenBench root
        if self.controller.project_root:
            return self.controller.project_root
        return get_openbench_root()

    def _get_remote_ssh_manager(self):
        """Get SSH manager from the runtime page."""
        if "runtime" in self.pages:
            runtime_page = self.pages["runtime"]
            if hasattr(runtime_page, "remote_config_widget"):
                return runtime_page.remote_config_widget.get_ssh_manager()
        return None

    def _browse_remote_config_file(self) -> str:
        """Browse remote server for config file."""
        ssh_manager = self._get_remote_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            QMessageBox.warning(
                self, "Not Connected", "Please connect to the remote server first in the Runtime Environment page."
            )
            return ""

        # Start in the configured OpenBench output directory when it exists
        # on the remote host; _resolve_remote_start_path validates each
        # candidate and falls back (openbench_path -> remote home -> /), so a
        # stale path cannot strand the browser on a failed listing.
        from openbench.gui.path_utils import _resolve_remote_start_path

        openbench_path = self.controller.remote_settings().get("openbench_path", "")
        candidate = f"{openbench_path.rstrip('/')}/output" if openbench_path else ""
        start_path = _resolve_remote_start_path(self.controller, ssh_manager, candidate)

        # Create dialog with remote file browser
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Configuration from Remote Server")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        browser = RemoteFileBrowser(ssh_manager, start_path, dialog, select_dirs=False)
        layout.addWidget(browser)

        selected_path = [None]

        def on_path_selected(path):
            if path.endswith(".yaml") or path.endswith(".yml"):
                selected_path[0] = path
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Invalid File", "Please select a YAML file (.yaml or .yml)")

        browser.file_selected.connect(on_path_selected)
        dialog.exec()

        return selected_path[0] or ""

    def _load_config_file(self, file_path: str):
        """Load configuration from a YAML file."""
        try:
            # Check if using remote storage
            is_remote = isinstance(self.controller.storage, RemoteStorage)

            if is_remote:
                loaded_config = self._load_remote_yaml_file(file_path)
                if loaded_config is None:
                    return
                config_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
                base_dir = config_dir.rsplit("/", 1)[0] if "/" in config_dir else ""
                project_root = self._find_remote_project_root(config_dir)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded_config = yaml.safe_load(f) or {}
                config_dir = os.path.dirname(os.path.abspath(file_path))
                base_dir = os.path.dirname(config_dir)
                project_root = self._find_project_root(config_dir)

            self.controller.project_root = project_root

            # Start with default config
            new_config = self.controller._default_config()

            # Check if this is a main config file (has reference_nml and simulation_nml)
            general = loaded_config.get("general", {})

            if self.controller._config_manager.is_unified_config(loaded_config):
                # v3 openbench.yaml uses top-level project/evaluation/reference/
                # simulation sections. Convert it back to the GUI's legacy-shaped
                # internal dict so pages can render the loaded values.
                new_config.update(self.controller._config_manager.unified_to_gui_config(loaded_config))
            elif "reference_nml" in general or "simulation_nml" in general:
                # This is a main config file
                self._load_main_config(loaded_config, new_config, config_dir, base_dir)
            elif any(key.endswith("_ref_source") for key in general.keys()):
                # This is a ref config file
                new_config["ref_data"] = loaded_config
                self._extract_evaluation_items_from_ref(loaded_config, new_config)
            elif any(key.endswith("_sim_source") for key in general.keys()):
                # This is a sim config file
                new_config["sim_data"] = loaded_config
                self._extract_evaluation_items_from_sim(loaded_config, new_config)
            else:
                # Unknown format, try to load as-is
                new_config.update(loaded_config)

            # Preserve runtime settings when in remote mode
            if is_remote:
                # Keep current runtime settings (execution_mode, SSH connection, remote config, etc.)
                current_runtime = self.controller.config.get("general", {})
                for key in ("execution_mode", "num_cores", "python_path", "conda_env", "remote"):
                    if key in current_runtime:
                        new_config["general"][key] = current_runtime[key]

            # Update the controller with the new config
            self.controller.config = new_config

            # Navigate to the first page (skip runtime in remote mode)
            self.controller.go_to_page("general")

            # Refresh all pages (skip runtime page in remote mode to preserve SSH connection)
            for page_id, page in self.pages.items():
                if is_remote and page_id == "runtime":
                    continue  # Don't refresh runtime page in remote mode
                if hasattr(page, "load_from_config"):
                    page.load_from_config()

            QMessageBox.information(
                self, "Configuration Loaded", f"Successfully loaded configuration from:\n{file_path}"
            )

        except FileNotFoundError:
            QMessageBox.critical(self, "File Not Found", f"Configuration file not found:\n{file_path}")
        except PermissionError:
            QMessageBox.critical(self, "Permission Denied", f"Cannot read file (permission denied):\n{file_path}")
        except yaml.YAMLError as e:
            QMessageBox.critical(
                self, "YAML Parse Error", f"Invalid YAML format in file:\n{file_path}\n\nDetails: {str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load configuration:\n{file_path}\n\nError: {str(e)}")

    def _load_main_config(self, loaded_config: dict, new_config: dict, config_dir: str, base_dir: str):
        """Load a main config file and its referenced ref/sim configs."""
        general = loaded_config.get("general", {})
        project_root = self.controller.project_root

        # Check if using remote storage
        is_remote = isinstance(self.controller.storage, RemoteStorage)

        # Copy general settings and convert paths to absolute
        for key, value in general.items():
            if key not in ("reference_nml", "simulation_nml", "statistics_nml", "figure_nml"):
                # Convert path fields to absolute (only for local mode)
                if key in ("basedir",) and value and not is_remote:
                    value = to_absolute_path(value, project_root)
                new_config["general"][key] = value

        # Copy other sections
        for section in ("evaluation_items", "metrics", "scores", "comparisons", "statistics"):
            if section in loaded_config:
                new_config[section] = loaded_config[section]

        # Load reference NML if specified
        ref_nml_path = general.get("reference_nml", "")
        if ref_nml_path:
            if is_remote:
                ref_full_path = self._resolve_remote_path(ref_nml_path, config_dir, base_dir)
                if ref_full_path:
                    ref_config = self._load_remote_yaml_file(ref_full_path)
                    if ref_config:
                        new_config["ref_data"] = ref_config
            else:
                ref_full_path = self._resolve_path(ref_nml_path, config_dir, base_dir)
                if ref_full_path and os.path.exists(ref_full_path):
                    try:
                        with open(ref_full_path, "r", encoding="utf-8") as f:
                            ref_config = yaml.safe_load(f) or {}
                        # Convert all paths in ref_config to absolute
                        ref_config = convert_paths_in_dict(ref_config, project_root)
                        new_config["ref_data"] = ref_config
                    except Exception as e:
                        print(f"Warning: Failed to load reference NML: {e}")

        # Load simulation NML if specified
        sim_nml_path = general.get("simulation_nml", "")
        if sim_nml_path:
            if is_remote:
                sim_full_path = self._resolve_remote_path(sim_nml_path, config_dir, base_dir)
                if sim_full_path:
                    sim_config = self._load_remote_yaml_file(sim_full_path)
                    if sim_config:
                        new_config["sim_data"] = sim_config
            else:
                sim_full_path = self._resolve_path(sim_nml_path, config_dir, base_dir)
                if sim_full_path and os.path.exists(sim_full_path):
                    try:
                        with open(sim_full_path, "r", encoding="utf-8") as f:
                            sim_config = yaml.safe_load(f) or {}
                        # Convert all paths in sim_config to absolute
                        sim_config = convert_paths_in_dict(sim_config, project_root)
                        new_config["sim_data"] = sim_config
                    except Exception as e:
                        print(f"Warning: Failed to load simulation NML: {e}")

        # Validate all paths and warn user about missing ones (skip for remote mode)
        if not is_remote:
            self._validate_loaded_paths(new_config)

    def _resolve_path(self, path: str, config_dir: str, base_dir: str) -> str:
        """Resolve a path that might be relative to different base directories."""
        if os.path.isabs(path):
            return path

        # Try relative to base_dir first (for paths like ./nml/nml-yaml/...)
        if path.startswith("./"):
            relative_path = path[2:]

            # Try from current working directory first
            full_path = os.path.normpath(os.path.join(os.getcwd(), relative_path))
            if os.path.exists(full_path):
                return full_path

            # Try from project root (parent of nml directory, i.e., OpenBench)
            project_root = os.path.dirname(base_dir)
            full_path = os.path.normpath(os.path.join(project_root, relative_path))
            if os.path.exists(full_path):
                return full_path

            # Try from base_dir
            full_path = os.path.normpath(os.path.join(base_dir, relative_path))
            if os.path.exists(full_path):
                return full_path

            # Try from config_dir
            full_path = os.path.normpath(os.path.join(config_dir, relative_path))
            if os.path.exists(full_path):
                return full_path

        # Try relative to config_dir
        full_path = os.path.normpath(os.path.join(config_dir, path))
        if os.path.exists(full_path):
            return full_path

        # Try relative to base_dir
        full_path = os.path.normpath(os.path.join(base_dir, path))
        if os.path.exists(full_path):
            return full_path

        # Return the path as-is if nothing works
        return path

    def _resolve_remote_path(self, path: str, config_dir: str, base_dir: str) -> str:
        """Resolve a path that might be relative on a remote server."""
        ssh_manager = self._get_remote_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            return path

        # If absolute, check if exists and return
        if path.startswith("/"):
            stdout, stderr, exit_code = execute_responsive(
                ssh_manager, f"test -e {quote_remote_path(path)} && echo 'exists'", timeout=10
            )
            if exit_code == 0 and "exists" in stdout:
                return path
            return path

        # Helper to check if path exists on remote
        def remote_exists(check_path):
            stdout, stderr, exit_code = execute_responsive(
                ssh_manager, f"test -e {quote_remote_path(check_path)} && echo 'exists'", timeout=10
            )
            return exit_code == 0 and "exists" in stdout

        # Try relative to base_dir first (for paths like ./nml/nml-yaml/...)
        if path.startswith("./"):
            relative_path = path[2:]

            # Try from project root
            project_root = base_dir.rsplit("/", 1)[0] if "/" in base_dir else base_dir
            full_path = f"{project_root}/{relative_path}"
            if remote_exists(full_path):
                return full_path

            # Try from base_dir
            full_path = f"{base_dir}/{relative_path}"
            if remote_exists(full_path):
                return full_path

            # Try from config_dir
            full_path = f"{config_dir}/{relative_path}"
            if remote_exists(full_path):
                return full_path

        # Try relative to config_dir
        full_path = f"{config_dir}/{path}"
        if remote_exists(full_path):
            return full_path

        # Try relative to base_dir
        full_path = f"{base_dir}/{path}"
        if remote_exists(full_path):
            return full_path

        # Return the path as-is if nothing works
        return path

    def _extract_evaluation_items_from_ref(self, ref_config: dict, new_config: dict):
        """Extract evaluation items from ref config source keys."""
        general = ref_config.get("general", {})
        for key in general.keys():
            if key.endswith("_ref_source"):
                var_name = key.replace("_ref_source", "")
                new_config["evaluation_items"][var_name] = True

    def _extract_evaluation_items_from_sim(self, sim_config: dict, new_config: dict):
        """Extract evaluation items from sim config source keys."""
        general = sim_config.get("general", {})
        for key in general.keys():
            if key.endswith("_sim_source"):
                var_name = key.replace("_sim_source", "")
                new_config["evaluation_items"][var_name] = True

    def _find_project_root(self, start_dir: str) -> str:
        """Find the OpenBench project root directory."""
        # Use the shared strict v3-root marker check; the old loose "has
        # openbench/ or nml/ subdir" test matched any random folder that
        # happened to contain an `nml/` directory and silently misconfigured
        # downstream paths.
        from openbench.gui.path_utils import looks_like_openbench_root

        current = start_dir
        for _ in range(10):  # Max 10 levels up
            if looks_like_openbench_root(current):
                return current
            parent = os.path.dirname(current)
            if parent == current:  # Reached filesystem root
                break
            current = parent
        return start_dir  # Fallback to start directory

    def _load_remote_yaml_file(self, file_path: str) -> dict:
        """Load YAML content from a remote file via SSH."""
        ssh_manager = self._get_remote_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            QMessageBox.warning(self, "Not Connected", "Cannot load remote file: SSH connection not available.")
            return None

        try:
            stdout, stderr, exit_code = execute_responsive(
                ssh_manager, f"cat {quote_remote_path(file_path)}", timeout=30
            )
            if exit_code != 0:
                QMessageBox.critical(self, "Error", f"Failed to read remote file:\n{file_path}\n\nError: {stderr}")
                return None
            return yaml.safe_load(stdout) or {}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load remote YAML file:\n{file_path}\n\nError: {str(e)}")
            return None

    def _find_remote_project_root(self, start_dir: str) -> str:
        """Find the OpenBench project root directory on remote server."""
        ssh_manager = self._get_remote_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            return start_dir

        current = start_dir
        for _ in range(10):  # Max 10 levels up
            # Check if this looks like OpenBench root
            stdout, stderr, exit_code = execute_responsive(
                ssh_manager,
                f"ls -d {quote_remote_path(current + '/openbench')} {quote_remote_path(current + '/nml')} 2>/dev/null | head -1",
                timeout=10,
            )
            if exit_code == 0 and stdout.strip():
                return current

            # Go up one directory
            parent = current.rsplit("/", 1)[0] if "/" in current else ""
            if not parent or parent == current:
                break
            current = parent

        return start_dir  # Fallback to start directory

    def _convert_to_absolute_path(self, path: str, project_root: str) -> str:
        """Convert a relative path to absolute path based on project root."""
        if not path:
            return path
        if os.path.isabs(path):
            return path
        if path.startswith("./"):
            path = path[2:]
        return os.path.normpath(os.path.join(project_root, path))

    def _validate_path(self, path: str, path_type: str = "file") -> bool:
        """Validate if a path exists. Returns True if valid."""
        if not path:
            return True  # Empty paths are OK
        if path_type == "file":
            return os.path.isfile(path)
        elif path_type == "directory":
            return os.path.isdir(path)
        return os.path.exists(path)

    def _prompt_for_missing_path(self, description: str, path_type: str = "file") -> str:
        """Prompt user to select a path when it's missing."""
        msg = "The following path was not found:\n\nPlease select the correct location."
        QMessageBox.warning(self, "Path Not Found", msg)

        if path_type == "file":
            new_path, _ = QFileDialog.getOpenFileName(self, f"Select {description}", "", "All Files (*)")
        else:
            new_path = QFileDialog.getExistingDirectory(self, f"Select {description}")
        return new_path or ""

    def _validate_loaded_paths(self, config: dict):
        """Validate loaded paths and show warnings for missing ones."""
        errors = validate_paths_in_dict(config)

        if errors:
            # Build error message
            error_lines = []
            for key, path, error in errors[:10]:  # Show max 10 errors
                error_lines.append(f"  {key}: {path}")

            if len(errors) > 10:
                error_lines.append(f"  ... and {len(errors) - 10} more")

            msg = (
                "Some paths in the configuration could not be found:\n\n"
                + "\n".join(error_lines)
                + "\n\nPlease update the paths in the respective data pages."
            )
            QMessageBox.warning(self, "Path Validation Warning", msg)

    def _on_new_clicked(self):
        """Handle New Config button click."""
        reply = QMessageBox.question(
            self,
            "New Configuration",
            "Create a new configuration? Any unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.controller.reset()
            # Set project_root for new configs
            self.controller.project_root = get_openbench_root()

    def _init_default_storage(self):
        """Initialize with default local storage.

        Uses the OpenBench root directory as the project directory.
        Remote mode can be configured via the Runtime Environment page.
        """
        # Always use auto-detection for fresh start (no cached paths)
        project_root = get_openbench_root()

        self.controller.project_root = project_root
        self.controller.storage = LocalStorage(project_root)

        # Don't load existing project config - start fresh each time

    def setup_remote_storage(self, ssh_manager, remote_project_dir: str):
        """Setup remote storage when user connects via Runtime Environment page.

        Called by page_runtime when remote mode is configured.

        Args:
            ssh_manager: Connected SSH manager
            remote_project_dir: Remote project directory path
        """
        from openbench.remote.sync import SyncEngine

        # Create sync engine and remote storage
        sync_engine = SyncEngine(ssh_manager, remote_project_dir)
        self.controller.storage = RemoteStorage(remote_project_dir, sync_engine)
        self.controller.ssh_manager = ssh_manager

        # Setup sync status widget
        self._setup_sync_status(sync_engine)

        # Start background sync
        sync_engine.start_background_sync()

    def setup_local_storage(self, project_dir: str):
        """Setup local storage when user switches to local mode.

        Called by page_runtime when local mode is configured.

        Args:
            project_dir: Local project directory path
        """
        # Stop and cleanup old sync engine if exists (from previous remote mode)
        old_storage = self.controller.storage
        if old_storage and hasattr(old_storage, "_sync_engine"):
            sync_engine = old_storage._sync_engine
            if sync_engine:
                # Clear the callback first to prevent crashes
                sync_engine._on_status_changed = None
                # Stop background sync thread
                sync_engine.stop_background_sync()

        self.controller.storage = LocalStorage(project_dir)
        self.controller.project_root = project_dir

        # Remove sync status widget if exists
        if self._sync_status:
            self._sync_status.cleanup()
            self._sync_status.setParent(None)
            self._sync_status.deleteLater()
            self._sync_status = None

    def _setup_sync_status(self, sync_engine):
        """Setup sync status widget for remote mode."""
        # Remove old sync status if exists
        if self._sync_status:
            self._sync_status.cleanup()
            self._sync_status.setParent(None)
            self._sync_status.deleteLater()

        self._sync_status = SyncStatusWidget(self)

        # Insert sync status widget before the Rerun button in nav bar
        # Find the index of Rerun button (it's after the second stretch)
        if self._nav_bar_layout:
            # Insert before the Rerun button (index 4: back, stretch, indicator, stretch, [sync], rerun, next)
            self._nav_bar_layout.insertWidget(4, self._sync_status)

        # Connect to sync engine status changes using thread-safe signal
        # The callback is called from background sync thread, so we use a signal
        # with QueuedConnection to safely update UI in the main thread
        # Use a weak reference to avoid crashes when widget is deleted
        import weakref

        widget_ref = weakref.ref(self._sync_status)
        main_window_ref = weakref.ref(self)

        def on_status_changed(path, status):
            # Get the widget via weak reference - returns None if deleted
            widget = widget_ref()
            main_window = main_window_ref()
            if widget is None or main_window is None:
                # Widget or main window was garbage collected, clear callback
                try:
                    sync_engine._on_status_changed = None
                except Exception:
                    pass
                return
            # Additional check: ensure this is still the current sync status widget
            if main_window._sync_status is not widget:
                return
            try:
                overall = sync_engine.get_overall_status()
                pending = sync_engine.get_pending_count()
                # Emit signal instead of direct call - thread-safe
                widget.status_update_requested.emit(overall, pending)
            except RuntimeError:
                # Widget was deleted, clear callback
                sync_engine._on_status_changed = None

        sync_engine._on_status_changed = on_status_changed
        self._sync_status.retry_clicked.connect(sync_engine.retry_errors)

        # Set initial status
        overall = sync_engine.get_overall_status()
        pending = sync_engine.get_pending_count()
        self._sync_status.set_status(overall, pending)

    def _try_load_project_config(self):
        """Try to load existing project config if available."""
        storage = self.controller.storage
        if not storage:
            return

        # Look for main config in nml/
        try:
            files = storage.glob("nml/main-*.yaml")
            if files:
                # Load first main config found
                config_path = files[0]
                content = storage.read_file(config_path)
                loaded_config = yaml.safe_load(content) or {}

                # Extract basename from filename (main-{basename}.yaml)
                filename = os.path.basename(config_path)
                if filename.startswith("main-") and filename.endswith(".yaml"):
                    basename = filename[5:-5]  # Remove "main-" prefix and ".yaml" suffix
                else:
                    basename = "config"

                # Start with default config and merge loaded config
                new_config = self.controller._default_config()

                # Copy general settings
                general = loaded_config.get("general", {})
                for key, value in general.items():
                    if key not in ("reference_nml", "simulation_nml", "statistics_nml", "figure_nml"):
                        new_config["general"][key] = value

                # Ensure basename is set
                new_config["general"]["basename"] = basename

                # Copy other sections if present
                for section in ("evaluation_items", "metrics", "scores", "comparisons", "statistics"):
                    if section in loaded_config:
                        new_config[section] = loaded_config[section]

                # Try to load ref and sim configs
                ref_path = f"nml/ref-{basename}.yaml"
                if storage.exists(ref_path):
                    try:
                        ref_content = storage.read_file(ref_path)
                        ref_config = yaml.safe_load(ref_content) or {}
                        new_config["ref_data"] = ref_config
                    except Exception as e:
                        logger.warning(f"Failed to load ref config: {e}")

                sim_path = f"nml/sim-{basename}.yaml"
                if storage.exists(sim_path):
                    try:
                        sim_content = storage.read_file(sim_path)
                        sim_config = yaml.safe_load(sim_content) or {}
                        new_config["sim_data"] = sim_config
                    except Exception as e:
                        logger.warning(f"Failed to load sim config: {e}")

                # Update the controller with the loaded config
                self.controller.config = new_config

                # Navigate to the general page (skip runtime)
                self.controller.go_to_page("general")

                # Refresh all pages
                for page_id, page in self.pages.items():
                    if hasattr(page, "load_from_config"):
                        page.load_from_config()

                logger.info(f"Loaded project config: {config_path}")
        except Exception as e:
            logger.debug(f"No existing config found or failed to load: {e}")
