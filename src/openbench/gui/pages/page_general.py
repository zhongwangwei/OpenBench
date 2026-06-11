# -*- coding: utf-8 -*-
"""
General settings page.
"""

import logging
import shlex

from PySide6.QtWidgets import (
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
)
from openbench.gui.widgets._ssh_worker import execute_responsive
from openbench.gui.widgets.no_scroll_widgets import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox

from openbench.gui.pages.base_page import BasePage
from openbench.gui.path_utils import browse_directory
from openbench.gui.widgets import PathSelector

logger = logging.getLogger(__name__)


class PageGeneral(BasePage):
    """General configuration page."""

    PAGE_ID = "general"
    PAGE_TITLE = "General Settings"
    PAGE_SUBTITLE = "Configure basic project settings and evaluation options"

    def _setup_content(self):
        """Setup page content."""
        # === Project Info ===
        project_group = QGroupBox("Project Information")
        project_layout = QFormLayout(project_group)

        # Output directory first
        self.basedir_input = PathSelector(mode="directory", placeholder="Output directory")
        self.basedir_input.path_changed.connect(self._on_basedir_changed)
        self.basedir_input.set_custom_browse_handler(self._browse_output_directory)
        project_layout.addRow("Output Directory:", self.basedir_input)

        # Project name with confirm button
        name_layout = QHBoxLayout()
        self.basename_input = QLineEdit()
        self.basename_input.setPlaceholderText("Project name (e.g., Initial_test)")
        self.basename_input.textChanged.connect(self._on_project_name_changed)
        name_layout.addWidget(self.basename_input)

        self.btn_confirm_name = QPushButton("Confirm")
        self.btn_confirm_name.setFixedWidth(80)
        self.btn_confirm_name.clicked.connect(self._on_confirm_project)
        name_layout.addWidget(self.btn_confirm_name)

        project_layout.addRow("Project Name:", name_layout)

        self.content_layout.addWidget(project_group)

        # === Spatial-Temporal Settings ===
        st_group = QGroupBox("Spatial-Temporal Settings")
        st_layout = QGridLayout(st_group)

        # Year range
        self.year_range_label = QLabel("Year Range:")
        st_layout.addWidget(self.year_range_label, 0, 0)
        self.syear_spin = NoScrollSpinBox()
        self.syear_spin.setRange(1900, 2100)
        self.syear_spin.setValue(2000)
        self.syear_spin.valueChanged.connect(self._on_year_range_changed)
        st_layout.addWidget(self.syear_spin, 0, 1)
        self.year_range_to_label = QLabel("to")
        st_layout.addWidget(self.year_range_to_label, 0, 2)
        self.eyear_spin = NoScrollSpinBox()
        self.eyear_spin.setRange(1900, 2100)
        self.eyear_spin.setValue(2020)
        self.eyear_spin.valueChanged.connect(self._on_year_range_changed)
        st_layout.addWidget(self.eyear_spin, 0, 3)

        # Minimum year threshold
        st_layout.addWidget(QLabel("Min Year Threshold:"), 1, 0)
        self.min_year_spin = NoScrollDoubleSpinBox()
        self.min_year_spin.setRange(0.0, 100.0)
        self.min_year_spin.setValue(1.0)
        self.min_year_spin.setSingleStep(0.5)
        self.min_year_spin.setToolTip("Minimum number of years of valid data required")
        st_layout.addWidget(self.min_year_spin, 1, 1)

        # Latitude range
        st_layout.addWidget(QLabel("Latitude Range:"), 2, 0)
        self.min_lat_spin = NoScrollDoubleSpinBox()
        self.min_lat_spin.setRange(-90.0, 90.0)
        self.min_lat_spin.setValue(-90.0)
        st_layout.addWidget(self.min_lat_spin, 2, 1)
        st_layout.addWidget(QLabel("to"), 2, 2)
        self.max_lat_spin = NoScrollDoubleSpinBox()
        self.max_lat_spin.setRange(-90.0, 90.0)
        self.max_lat_spin.setValue(90.0)
        st_layout.addWidget(self.max_lat_spin, 2, 3)

        # Longitude range
        st_layout.addWidget(QLabel("Longitude Range:"), 3, 0)
        self.min_lon_spin = NoScrollDoubleSpinBox()
        self.min_lon_spin.setRange(-180.0, 180.0)
        self.min_lon_spin.setValue(-180.0)
        st_layout.addWidget(self.min_lon_spin, 3, 1)
        st_layout.addWidget(QLabel("to"), 3, 2)
        self.max_lon_spin = NoScrollDoubleSpinBox()
        self.max_lon_spin.setRange(-180.0, 180.0)
        self.max_lon_spin.setValue(180.0)
        st_layout.addWidget(self.max_lon_spin, 3, 3)

        # Resolution
        st_layout.addWidget(QLabel("Time Resolution:"), 4, 0)
        self.tim_res_combo = NoScrollComboBox()
        self.tim_res_combo.addItems(["month", "day", "hour", "year", "Climatology-month", "Climatology-year"])
        st_layout.addWidget(self.tim_res_combo, 4, 1)

        st_layout.addWidget(QLabel("Grid Resolution:"), 4, 2)
        self.grid_res_spin = NoScrollDoubleSpinBox()
        self.grid_res_spin.setRange(0.01, 10.0)
        self.grid_res_spin.setValue(2.0)
        self.grid_res_spin.setSingleStep(0.1)
        self.grid_res_spin.setSuffix("°")
        st_layout.addWidget(self.grid_res_spin, 4, 3)

        # Time alignment strategy
        st_layout.addWidget(QLabel("Time Alignment:"), 5, 0)
        self.time_alignment_combo = NoScrollComboBox()
        self.time_alignment_combo.addItem("intersection — All models' common time range", "intersection")
        self.time_alignment_combo.addItem("per_pair — Each sim-ref pair uses its own overlap", "per_pair")
        self.time_alignment_combo.addItem("strict — All data must cover config years exactly", "strict")
        self.time_alignment_combo.setToolTip(
            "How to handle different time periods across simulations:\n\n"
            "intersection: Use only the common time range across ALL models (strictest, ensures comparability)\n"
            "per_pair: Each simulation-reference pair uses its own time overlap (maximizes data use)\n"
            "strict: All data must fully cover the configured year range (error if any gaps)"
        )
        st_layout.addWidget(self.time_alignment_combo, 5, 1, 1, 3)

        # Timezone
        st_layout.addWidget(QLabel("Timezone:"), 6, 0)
        self.timezone_spin = NoScrollDoubleSpinBox()
        self.timezone_spin.setRange(-12.0, 14.0)
        self.timezone_spin.setValue(0.0)
        self.timezone_spin.setSingleStep(0.5)
        st_layout.addWidget(self.timezone_spin, 6, 1)

        # Weight
        st_layout.addWidget(QLabel("Weight:"), 6, 2)
        self.weight_combo = NoScrollComboBox()
        self.weight_combo.addItems(["None", "area", "mass"])
        self.weight_combo.setToolTip("Weight method for spatial averaging (None, area-weighted, or mass-weighted)")
        st_layout.addWidget(self.weight_combo, 6, 3)

        self.content_layout.addWidget(st_group)

        # === Feature Toggles ===
        toggle_group = QGroupBox("Feature Toggles")
        toggle_layout = QGridLayout(toggle_group)

        self.cb_evaluation = QCheckBox("Evaluation")
        self.cb_evaluation.setChecked(True)
        self.cb_evaluation.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_evaluation, 0, 0)

        self.cb_comparison = QCheckBox("Comparison")
        self.cb_comparison.setChecked(True)
        self.cb_comparison.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_comparison, 0, 1)

        self.cb_statistics = QCheckBox("Statistics")
        self.cb_statistics.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_statistics, 0, 2)

        self.cb_debug = QCheckBox("Debug Mode")
        self.cb_debug.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_debug, 1, 0)

        self.cb_report = QCheckBox("Generate Report")
        self.cb_report.setChecked(True)
        self.cb_report.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_report, 1, 1)

        self.cb_only_drawing = QCheckBox("Only Drawing")
        self.cb_only_drawing.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_only_drawing, 1, 2)

        self.cb_unified_mask = QCheckBox("Unified Mask")
        self.cb_unified_mask.setChecked(True)
        self.cb_unified_mask.stateChanged.connect(self._on_toggle_changed)
        toggle_layout.addWidget(self.cb_unified_mask, 2, 0)

        self.content_layout.addWidget(toggle_group)

        # === Groupby Options ===
        groupby_group = QGroupBox("Groupby Options")
        groupby_layout = QHBoxLayout(groupby_group)

        self.cb_igbp = QCheckBox("IGBP Groupby")
        self.cb_igbp.setChecked(True)
        groupby_layout.addWidget(self.cb_igbp)

        self.cb_pft = QCheckBox("PFT Groupby")
        self.cb_pft.setChecked(True)
        groupby_layout.addWidget(self.cb_pft)

        self.cb_climate = QCheckBox("Climate Zone Groupby")
        self.cb_climate.setChecked(True)
        groupby_layout.addWidget(self.cb_climate)

        groupby_layout.addStretch()

        self.content_layout.addWidget(groupby_group)

        # === Performance Settings ===
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QGridLayout(performance_group)

        self.cb_netcdf_compression = QCheckBox("Compress NetCDF outputs")
        self.cb_netcdf_compression.setToolTip("Enable zlib compression for numeric NetCDF outputs.")
        self.cb_netcdf_compression.stateChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.cb_netcdf_compression, 0, 0)

        performance_layout.addWidget(QLabel("Compression Level:"), 0, 1)
        self.netcdf_compression_level_spin = NoScrollSpinBox()
        self.netcdf_compression_level_spin.setRange(0, 9)
        self.netcdf_compression_level_spin.setValue(1)
        self.netcdf_compression_level_spin.setToolTip("zlib level 0-9; level 1 is the recommended fast default.")
        self.netcdf_compression_level_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.netcdf_compression_level_spin, 0, 2)

        performance_layout.addWidget(QLabel("Multi-file Combine:"), 1, 0)
        self.mfdataset_batch_mode_combo = NoScrollComboBox()
        self.mfdataset_batch_mode_combo.addItem("Auto planner", "auto")
        self.mfdataset_batch_mode_combo.addItem("Disabled", "disabled")
        self.mfdataset_batch_mode_combo.addItem("Fixed batch size", "fixed")
        self.mfdataset_batch_mode_combo.setToolTip(
            "Auto uses file count, total size, and available memory. Disabled maps to batch size 0."
        )
        self.mfdataset_batch_mode_combo.currentIndexChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.mfdataset_batch_mode_combo, 1, 1)

        self.mfdataset_batch_size_spin = NoScrollSpinBox()
        self.mfdataset_batch_size_spin.setRange(1, 10000)
        self.mfdataset_batch_size_spin.setValue(100)
        self.mfdataset_batch_size_spin.setToolTip("Used only when Multi-file Combine is set to Fixed batch size.")
        self.mfdataset_batch_size_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.mfdataset_batch_size_spin, 1, 2)

        performance_layout.addWidget(QLabel("Auto Min Files:"), 2, 0)
        self.mfdataset_auto_min_files_spin = NoScrollSpinBox()
        self.mfdataset_auto_min_files_spin.setRange(1, 1000000)
        self.mfdataset_auto_min_files_spin.setValue(200)
        self.mfdataset_auto_min_files_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.mfdataset_auto_min_files_spin, 2, 1)

        performance_layout.addWidget(QLabel("Auto Max Batch:"), 2, 2)
        self.mfdataset_auto_max_size_spin = NoScrollSpinBox()
        self.mfdataset_auto_max_size_spin.setRange(1, 10000)
        self.mfdataset_auto_max_size_spin.setValue(100)
        self.mfdataset_auto_max_size_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.mfdataset_auto_max_size_spin, 2, 3)

        performance_layout.addWidget(QLabel("Memory Fraction:"), 3, 0)
        self.mfdataset_auto_memory_fraction_spin = NoScrollDoubleSpinBox()
        self.mfdataset_auto_memory_fraction_spin.setRange(0.01, 1.0)
        self.mfdataset_auto_memory_fraction_spin.setSingleStep(0.05)
        self.mfdataset_auto_memory_fraction_spin.setValue(0.25)
        self.mfdataset_auto_memory_fraction_spin.setToolTip("Fraction of available memory used to size auto batches.")
        self.mfdataset_auto_memory_fraction_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.mfdataset_auto_memory_fraction_spin, 3, 1)

        self.cb_dask_enabled = QCheckBox("Enable dask.distributed")
        self.cb_dask_enabled.setToolTip("Start a local dask.distributed cluster for xarray lazy workloads.")
        self.cb_dask_enabled.stateChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.cb_dask_enabled, 4, 0)

        performance_layout.addWidget(QLabel("Workers:"), 4, 1)
        self.dask_workers_spin = NoScrollSpinBox()
        self.dask_workers_spin.setRange(1, 128)
        self.dask_workers_spin.setValue(4)
        self.dask_workers_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.dask_workers_spin, 4, 2)

        performance_layout.addWidget(QLabel("Threads/Worker:"), 5, 0)
        self.dask_threads_spin = NoScrollSpinBox()
        self.dask_threads_spin.setRange(1, 64)
        self.dask_threads_spin.setValue(1)
        self.dask_threads_spin.valueChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.dask_threads_spin, 5, 1)

        self.cb_dask_processes = QCheckBox("Use processes")
        self.cb_dask_processes.setChecked(True)
        self.cb_dask_processes.stateChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.cb_dask_processes, 5, 2)

        performance_layout.addWidget(QLabel("Memory Limit:"), 6, 0)
        self.dask_memory_limit_input = QLineEdit()
        self.dask_memory_limit_input.setPlaceholderText("auto or e.g. 4GB")
        self.dask_memory_limit_input.setText("auto")
        self.dask_memory_limit_input.textChanged.connect(self._on_performance_changed)
        performance_layout.addWidget(self.dask_memory_limit_input, 6, 1, 1, 2)

        self.content_layout.addWidget(performance_group)
        self._update_performance_controls_state()

        # Note: Runtime Environment is now on a separate page (PageRuntime)
        # num_cores_spin kept for config compatibility
        self.num_cores_spin = NoScrollSpinBox()
        self.num_cores_spin.setRange(1, 128)
        self.num_cores_spin.setValue(4)
        self.num_cores_spin.hide()  # Hidden, but kept for config compatibility

    def _on_toggle_changed(self, state):
        """Handle feature toggle changes."""
        self.save_to_config()

    def _on_performance_changed(self, *_args):
        """Handle performance setting changes."""
        self._update_performance_controls_state()
        self.save_to_config()

    def _update_performance_controls_state(self):
        """Enable controls that are relevant to the selected performance modes."""
        if hasattr(self, "mfdataset_batch_mode_combo"):
            mode = self.mfdataset_batch_mode_combo.currentData() or "auto"
            self.mfdataset_batch_size_spin.setEnabled(mode == "fixed")
        if hasattr(self, "netcdf_compression_level_spin"):
            self.netcdf_compression_level_spin.setEnabled(self.cb_netcdf_compression.isChecked())
        if hasattr(self, "cb_dask_enabled"):
            enabled = self.cb_dask_enabled.isChecked()
            for widget in (
                self.dask_workers_spin,
                self.dask_threads_spin,
                self.cb_dask_processes,
                self.dask_memory_limit_input,
            ):
                widget.setEnabled(enabled)

    def _on_year_range_changed(self, value):
        """Handle year range changes - save and sync namelists."""
        self.save_to_config()
        self.controller.sync_namelists()

    def _has_per_var_time_range(self) -> bool:
        """Check if any source has per_var_time_range enabled."""
        # Check ref_data source_configs
        ref_source_configs = self.controller.config.get("ref_data", {}).get("source_configs", {})
        for source_config in ref_source_configs.values():
            general = source_config.get("general", {})
            if general.get("per_var_time_range", False):
                return True

        # Check sim_data source_configs
        sim_source_configs = self.controller.config.get("sim_data", {}).get("source_configs", {})
        for source_config in sim_source_configs.values():
            general = source_config.get("general", {})
            if general.get("per_var_time_range", False):
                return True

        return False

    def update_year_range_state(self):
        """Update Year Range tooltip based on per_var_time_range settings.

        Year Range is always editable. When per-variable time range is enabled
        on any source, these values are used as defaults but may be overridden.
        """
        has_per_var = self._has_per_var_time_range()

        if has_per_var:
            tooltip = "Some sources use per-variable time range. This value is used for sources without per-variable settings."
            self.syear_spin.setToolTip(tooltip)
            self.eyear_spin.setToolTip(tooltip)
        else:
            self.syear_spin.setToolTip("")
            self.eyear_spin.setToolTip("")

    def _browse_output_directory(self):
        """Handle output directory browse - uses remote browser if in remote mode."""
        from openbench.gui.widgets.path_selector import get_default_browse_path

        current = self.basedir_input.path() or get_default_browse_path()
        path = browse_directory(self.controller, self, "Select Output Directory", current)
        if path:
            self.basedir_input.set_path(path)

    def _get_remote_ssh_manager(self):
        """Get SSH manager from the runtime page."""
        # Access the main window to get the runtime page
        main_window = self.window()
        if hasattr(main_window, "pages") and "runtime" in main_window.pages:
            runtime_page = main_window.pages["runtime"]
            if hasattr(runtime_page, "remote_config_widget"):
                return runtime_page.remote_config_widget.get_ssh_manager()
        return None

    def _on_project_name_changed(self, text):
        """Handle project name changes.

        Note: Only saves to config without triggering sync_namelists.
        Directory creation happens only when Confirm button is clicked.
        """
        self._save_to_config_no_sync()

    def _on_basedir_changed(self, path):
        """Handle output directory changes.

        Note: Only saves to config without triggering sync_namelists.
        Directory creation happens only when Confirm button is clicked.
        """
        self._save_to_config_no_sync()

    def _on_confirm_project(self):
        """Handle confirm project button click."""
        import os
        import re

        basename = self.basename_input.text().strip()
        if not basename:
            QMessageBox.warning(self, "Error", "Please enter a project name.")
            return

        # Validate filesystem-safe characters
        # Allow letters, numbers, underscores, hyphens, and dots
        if not re.match(r"^[a-zA-Z0-9_.-]+$", basename):
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Project name can only contain:\n"
                "• Letters (a-z, A-Z)\n"
                "• Numbers (0-9)\n"
                "• Underscores (_)\n"
                "• Hyphens (-)\n"
                "• Dots (.)",
            )
            return

        # Check for reserved names on Windows
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        if basename.upper() in reserved_names:
            QMessageBox.warning(self, "Invalid Name", f"'{basename}' is a reserved system name.")
            return

        # Check if output directory is set
        basedir = self.basedir_input.path().strip()
        if not basedir:
            QMessageBox.warning(self, "Error", "Please select an output directory first.")
            return

        # Save config first so get_output_dir() can compute correctly
        self.save_to_config()

        # Use controller.get_output_dir() to get the proper output path (basedir/basename)
        output_dir = self.controller.get_output_dir()

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)

        if is_remote:
            # Create directories on remote server
            self._create_remote_project_folder(output_dir)
        else:
            # Create the output directory and nml subdirectories locally
            try:
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(output_dir, "nml", "sim"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "nml", "ref"), exist_ok=True)

                # Trigger namelist sync
                self.controller.sync_namelists()

                QMessageBox.information(self, "Project Created", f"Project folder created:\n{output_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create project folder:\n{str(e)}")

    def _create_remote_project_folder(self, output_dir: str):
        """Create project folder on remote server."""
        ssh_manager = self._get_remote_ssh_manager()
        if not ssh_manager:
            QMessageBox.warning(
                self, "Not Connected", "Please connect to the remote server first in the Runtime Environment page."
            )
            return

        try:
            # Create directories on remote server
            cmd = (
                f"mkdir -p {shlex.quote(output_dir)} "
                f"{shlex.quote(output_dir + '/nml/sim')} "
                f"{shlex.quote(output_dir + '/nml/ref')}"
            )
            stdout, stderr, exit_code = execute_responsive(ssh_manager, cmd, timeout=30)

            if exit_code == 0:
                QMessageBox.information(
                    self, "Project Created", f"Project folder created on remote server:\n{output_dir}"
                )
            else:
                QMessageBox.critical(
                    self, "Error", f"Failed to create project folder on remote server:\n{stderr or stdout}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project folder:\n{str(e)}")

    def load_from_config(self):
        """Load settings from controller config."""
        import os

        general = self.controller.config.get("general", {})
        basename = general.get("basename", "")

        # Block signals to prevent save_to_config from being called during load
        self.basename_input.blockSignals(True)
        self.basename_input.setText(basename)
        self.basename_input.blockSignals(False)

        # Get basedir and convert to absolute path (without appending project name)
        basedir = general.get("basedir", "")

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)

        if is_remote:
            # Remote mode: use remote OpenBench path for defaults
            remote_openbench = self.controller.remote_settings().get("openbench_path", "")

            if not basedir or basedir == "./output":
                # Set default to remote OpenBench/output
                if remote_openbench:
                    basedir = f"{remote_openbench.rstrip('/')}/output"
                else:
                    basedir = "./output"
            elif not basedir.startswith("/"):
                # Convert relative path to absolute using remote root
                if basedir.startswith("./"):
                    basedir = basedir[2:]
                if remote_openbench:
                    basedir = f"{remote_openbench.rstrip('/')}/{basedir}"
            # Normalize slashes for remote paths
            basedir = basedir.replace("\\", "/")
        else:
            # Local mode: use local OpenBench root
            openbench_root = self._get_openbench_root()

            if not basedir or basedir == "./output":
                # Set default to OpenBench/output (without project name)
                basedir = os.path.join(openbench_root, "output")
            elif not os.path.isabs(basedir):
                # Convert relative path to absolute
                if basedir.startswith("./"):
                    basedir = basedir[2:]
                basedir = os.path.normpath(os.path.join(openbench_root, basedir))

        # Set path without emitting signal to prevent save_to_config loop
        self.basedir_input.set_path(basedir, emit_signal=False)

        # Skip local path validation in remote mode (remote paths won't exist locally)
        self.basedir_input.set_skip_validation(is_remote)

        self.syear_spin.setValue(general.get("syear", 2000))
        self.eyear_spin.setValue(general.get("eyear", 2020))
        self.min_year_spin.setValue(general.get("min_year", 1.0))
        self.min_lat_spin.setValue(general.get("min_lat", -90.0))
        self.max_lat_spin.setValue(general.get("max_lat", 90.0))
        self.min_lon_spin.setValue(general.get("min_lon", -180.0))
        self.max_lon_spin.setValue(general.get("max_lon", 180.0))

        tim_res = general.get("compare_tim_res", "month")
        idx = self.tim_res_combo.findText(tim_res)
        if idx >= 0:
            self.tim_res_combo.setCurrentIndex(idx)

        self.grid_res_spin.setValue(general.get("compare_grid_res", 2.0))
        self.timezone_spin.setValue(general.get("compare_tzone", 0.0))

        # Time alignment
        time_align = general.get("time_alignment", "intersection")
        for i in range(self.time_alignment_combo.count()):
            if self.time_alignment_combo.itemData(i) == time_align:
                self.time_alignment_combo.setCurrentIndex(i)
                break

        self.cb_evaluation.setChecked(general.get("evaluation", True))
        self.cb_comparison.setChecked(general.get("comparison", True))
        self.cb_statistics.setChecked(general.get("statistics", False))
        self.cb_debug.setChecked(general.get("debug_mode", False))
        self.cb_report.setChecked(general.get("generate_report", True))
        self.cb_only_drawing.setChecked(general.get("only_drawing", False))

        self.cb_igbp.setChecked(general.get("IGBP_groupby", True))
        self.cb_pft.setChecked(general.get("PFT_groupby", True))
        self.cb_climate.setChecked(general.get("Climate_zone_groupby", True))
        self.cb_unified_mask.setChecked(general.get("unified_mask", True))

        self.num_cores_spin.setValue(general.get("num_cores", 4))
        self._load_performance_settings(general)

        weight = general.get("weight", "none")
        if weight is None:
            weight = "none"
        # Map lowercase to display text
        weight_map = {"none": "None", "area": "area", "mass": "mass"}
        display_weight = weight_map.get(str(weight).lower(), "None")
        idx = self.weight_combo.findText(display_weight)
        if idx >= 0:
            self.weight_combo.setCurrentIndex(idx)

        # Note: Runtime Environment settings (execution_mode, remote config, python_path, conda_env)
        # are now handled by PageRuntime

        # Update Year Range state based on per_var_time_range settings
        self.update_year_range_state()

    def _set_combo_by_data(self, combo, value):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _load_performance_settings(self, general):
        """Load project.io and project.dask settings into visible controls."""
        if not hasattr(self, "cb_netcdf_compression"):
            return
        io_config = general.get("io", {}) or {}
        dask_config = general.get("dask", {}) or {}

        widgets = [
            self.cb_netcdf_compression,
            self.netcdf_compression_level_spin,
            self.mfdataset_batch_mode_combo,
            self.mfdataset_batch_size_spin,
            self.mfdataset_auto_min_files_spin,
            self.mfdataset_auto_max_size_spin,
            self.mfdataset_auto_memory_fraction_spin,
            self.cb_dask_enabled,
            self.dask_workers_spin,
            self.dask_threads_spin,
            self.cb_dask_processes,
            self.dask_memory_limit_input,
        ]
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self.cb_netcdf_compression.setChecked(bool(io_config.get("netcdf_compression", False)))
            self.netcdf_compression_level_spin.setValue(int(io_config.get("netcdf_compression_level", 1)))

            batch_size = io_config.get("mfdataset_batch_size", None)
            if batch_size == 0:
                self._set_combo_by_data(self.mfdataset_batch_mode_combo, "disabled")
            elif batch_size is None or str(batch_size).lower() == "auto":
                self._set_combo_by_data(self.mfdataset_batch_mode_combo, "auto")
            else:
                self._set_combo_by_data(self.mfdataset_batch_mode_combo, "fixed")
                self.mfdataset_batch_size_spin.setValue(int(batch_size))

            self.mfdataset_auto_min_files_spin.setValue(int(io_config.get("mfdataset_auto_batch_min_files", 200)))
            self.mfdataset_auto_max_size_spin.setValue(int(io_config.get("mfdataset_auto_batch_max_size", 100)))
            self.mfdataset_auto_memory_fraction_spin.setValue(
                float(io_config.get("mfdataset_auto_batch_memory_fraction", 0.25))
            )

            self.cb_dask_enabled.setChecked(bool(dask_config.get("enabled", False)))
            self.dask_workers_spin.setValue(int(dask_config.get("n_workers") or 4))
            self.dask_threads_spin.setValue(int(dask_config.get("threads_per_worker") or 1))
            self.cb_dask_processes.setChecked(bool(dask_config.get("processes", True)))
            self.dask_memory_limit_input.setText(str(dask_config.get("memory_limit", "auto") or "auto"))
        finally:
            for widget, state in zip(widgets, previous):
                widget.blockSignals(state)
        self._update_performance_controls_state()

    def _collect_io_config(self, existing_general):
        """Collect visible project.io controls, preserving old config in tests/legacy pages."""
        if not hasattr(self, "cb_netcdf_compression"):
            return existing_general.get("io", {})

        io_config = {}
        if self.cb_netcdf_compression.isChecked():
            io_config["netcdf_compression"] = True
            io_config["netcdf_compression_level"] = self.netcdf_compression_level_spin.value()

        batch_mode = self.mfdataset_batch_mode_combo.currentData() or "auto"
        if batch_mode == "disabled":
            io_config["mfdataset_batch_size"] = 0
        elif batch_mode == "fixed":
            io_config["mfdataset_batch_size"] = self.mfdataset_batch_size_spin.value()

        min_files = self.mfdataset_auto_min_files_spin.value()
        max_size = self.mfdataset_auto_max_size_spin.value()
        memory_fraction = self.mfdataset_auto_memory_fraction_spin.value()
        if min_files != 200:
            io_config["mfdataset_auto_batch_min_files"] = min_files
        if max_size != 100:
            io_config["mfdataset_auto_batch_max_size"] = max_size
        if abs(memory_fraction - 0.25) > 1e-9:
            io_config["mfdataset_auto_batch_memory_fraction"] = memory_fraction

        return io_config

    def _collect_dask_config(self, existing_general):
        """Collect visible project.dask controls, preserving old config in tests/legacy pages."""
        if not hasattr(self, "cb_dask_enabled"):
            return existing_general.get("dask", {})
        if not self.cb_dask_enabled.isChecked():
            return {}
        return {
            "enabled": True,
            "n_workers": self.dask_workers_spin.value(),
            "threads_per_worker": self.dask_threads_spin.value(),
            "processes": self.cb_dask_processes.isChecked(),
            "memory_limit": self.dask_memory_limit_input.text().strip() or "auto",
        }

    def _save_to_config_no_sync(self):
        """Save settings to controller config WITHOUT triggering sync_namelists.

        Used for intermediate saves (like typing project name) where we don't
        want to create directories yet.
        """
        import os

        new_basename = self.basename_input.text().strip()
        new_basedir = self.basedir_input.path().strip()

        # If basedir ends with basename, use the parent directory as basedir
        # This prevents path duplication like /path/F58/F58
        if new_basename and new_basedir:
            if os.path.basename(new_basedir.rstrip(os.sep)) == new_basename:
                new_basedir = os.path.dirname(new_basedir.rstrip(os.sep))

        # Get existing general config to preserve runtime settings
        existing_general = self.controller.config.get("general", {})

        general = {
            "basename": new_basename,
            "basedir": new_basedir,
            "syear": self.syear_spin.value(),
            "eyear": self.eyear_spin.value(),
            "min_year": self.min_year_spin.value(),
            "min_lat": self.min_lat_spin.value(),
            "max_lat": self.max_lat_spin.value(),
            "min_lon": self.min_lon_spin.value(),
            "max_lon": self.max_lon_spin.value(),
            "compare_tim_res": self.tim_res_combo.currentText(),
            "compare_grid_res": self.grid_res_spin.value(),
            "compare_tzone": self.timezone_spin.value(),
            "time_alignment": self.time_alignment_combo.currentData() or "intersection",
            "evaluation": self.cb_evaluation.isChecked(),
            "comparison": self.cb_comparison.isChecked(),
            "statistics": self.cb_statistics.isChecked(),
            "debug_mode": self.cb_debug.isChecked(),
            "generate_report": self.cb_report.isChecked(),
            "only_drawing": self.cb_only_drawing.isChecked(),
            "IGBP_groupby": self.cb_igbp.isChecked(),
            "PFT_groupby": self.cb_pft.isChecked(),
            "Climate_zone_groupby": self.cb_climate.isChecked(),
            "unified_mask": self.cb_unified_mask.isChecked(),
            "num_cores": self.num_cores_spin.value(),
            "weight": self.weight_combo.currentText().lower(),
            # Preserve runtime settings from PageRuntime
            "execution_mode": existing_general.get("execution_mode", "local"),
            "python_path": existing_general.get("python_path", ""),
            "conda_env": existing_general.get("conda_env", ""),
            "local_openbench_path": existing_general.get("local_openbench_path", ""),
        }

        io_config = self._collect_io_config(existing_general)
        if io_config:
            general["io"] = io_config
        dask_config = self._collect_dask_config(existing_general)
        if dask_config:
            general["dask"] = dask_config

        # Preserve remote config if exists
        if "remote" in existing_general:
            general["remote"] = existing_general["remote"]

        self.controller.update_section("general", general)

    def save_to_config(self):
        """Save settings to controller config."""

        # Check if basename or basedir changed (affects output directory)
        old_general = self.controller.config.get("general", {})
        old_basename = old_general.get("basename", "")
        old_basedir = old_general.get("basedir", "")

        # Save config first
        self._save_to_config_no_sync()

        new_basename = self.basename_input.text().strip()
        new_basedir = self.basedir_input.path().strip()

        # Trigger namelist sync if output location changed
        if new_basename != old_basename or new_basedir != old_basedir:
            self.controller.sync_namelists()

    def validate(self) -> bool:
        """Validate page input."""
        from openbench.gui.validation import FieldValidator, ValidationManager

        errors = []
        manager = ValidationManager(self)

        # Project name required
        error = FieldValidator.required(
            self.basename_input.text().strip(),
            "basename",
            "Project name is required",
            page_id=self.PAGE_ID,
            widget=self.basename_input,
        )
        if error:
            errors.append(error)

        # Output directory required
        error = FieldValidator.required(
            self.basedir_input.path().strip(),
            "basedir",
            "Output directory is required",
            page_id=self.PAGE_ID,
            widget=self.basedir_input,
        )
        if error:
            errors.append(error)

        # Year range validation
        error = FieldValidator.min_max(
            self.syear_spin.value(),
            self.eyear_spin.value(),
            "year_range",
            "Start year cannot be greater than end year",
            page_id=self.PAGE_ID,
            widget=self.syear_spin,
        )
        if error:
            errors.append(error)

        # Latitude range validation
        error = FieldValidator.number_range(
            self.min_lat_spin.value(),
            -90.0,
            90.0,
            "min_lat",
            "Minimum latitude is invalid (must be -90 to 90)",
            page_id=self.PAGE_ID,
            widget=self.min_lat_spin,
        )
        if error:
            errors.append(error)

        error = FieldValidator.number_range(
            self.max_lat_spin.value(),
            -90.0,
            90.0,
            "max_lat",
            "Maximum latitude is invalid (must be -90 to 90)",
            page_id=self.PAGE_ID,
            widget=self.max_lat_spin,
        )
        if error:
            errors.append(error)

        error = FieldValidator.min_max(
            self.min_lat_spin.value(),
            self.max_lat_spin.value(),
            "lat_range",
            "Minimum latitude cannot be greater than maximum latitude",
            page_id=self.PAGE_ID,
            widget=self.min_lat_spin,
        )
        if error:
            errors.append(error)

        # Longitude range validation
        error = FieldValidator.number_range(
            self.min_lon_spin.value(),
            -180.0,
            180.0,
            "min_lon",
            "Minimum longitude is invalid (must be -180 to 180)",
            page_id=self.PAGE_ID,
            widget=self.min_lon_spin,
        )
        if error:
            errors.append(error)

        error = FieldValidator.number_range(
            self.max_lon_spin.value(),
            -180.0,
            180.0,
            "max_lon",
            "Maximum longitude is invalid (must be -180 to 180)",
            page_id=self.PAGE_ID,
            widget=self.max_lon_spin,
        )
        if error:
            errors.append(error)

        error = FieldValidator.min_max(
            self.min_lon_spin.value(),
            self.max_lon_spin.value(),
            "lon_range",
            "Minimum longitude cannot be greater than maximum longitude",
            page_id=self.PAGE_ID,
            widget=self.min_lon_spin,
        )
        if error:
            errors.append(error)

        # Show first error if any, allow user to skip
        if errors:
            if not manager.show_error_and_focus(errors[0]):
                return False

        self.save_to_config()
        return True
