# -*- coding: utf-8 -*-
"""
Dialog for editing data source configuration.
"""

import logging
from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit,
    QPushButton, QGroupBox, QRadioButton, QButtonGroup,
    QDialogButtonBox, QLabel, QMessageBox, QFileDialog,
    QCheckBox, QScrollArea, QWidget
)
from openbench.gui.widgets.no_scroll_widgets import (
    NoScrollDoubleSpinBox, NoScrollComboBox
)
from PySide6.QtCore import Qt

from openbench.gui.widgets.path_selector import PathSelector
from openbench.gui.widgets.remote_config import RemoteFileBrowser
from openbench.gui.path_utils import to_absolute_path, validate_path, get_openbench_root
from openbench.gui.validation import ValidationError

logger = logging.getLogger(__name__)


class DataSourceEditor(QDialog):
    """Dialog for editing data source configuration.

    For reference data:
        - Each variable has its own sub_dir, varname, prefix, suffix, varunit
        - These are stored per-variable in the YAML file

    For simulation data:
        - prefix/suffix are shared at the general level for all variables
        - All variables from the same case share the same naming pattern
    """

    def __init__(
        self,
        source_name: str = "",
        source_type: str = "ref",  # "ref" or "sim"
        var_name: str = "",  # Variable name (for ref data context)
        initial_data: Optional[Dict[str, Any]] = None,
        ssh_manager=None,  # SSH manager for remote browsing
        parent=None
    ):
        super().__init__(parent)
        self.source_name = source_name
        self.source_type = source_type
        self.var_name = var_name
        self.initial_data = initial_data or {}
        self._ssh_manager = ssh_manager

        # Build title with context
        if source_name and var_name:
            title = f"Configure {source_name} for {var_name.replace('_', ' ')}"
        elif source_name:
            title = f"Configure Data Source: {source_name}"
        elif var_name:
            title = f"New Data Source for {var_name.replace('_', ' ')}"
        else:
            title = "New Data Source"
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.setMaximumHeight(700)  # Limit height to ensure buttons are visible
        self.setModal(True)

        self._setup_ui()
        self._setup_remote_browsing()  # Setup remote browsing if ssh_manager provided
        self._on_data_type_changed()  # Initial show/hide
        self._load_data()

    def _is_remote_mode(self) -> bool:
        """Check if we're in remote mode (SSH manager is set)."""
        return self._ssh_manager is not None

    def _get_remote_openbench_root(self) -> str:
        """Get the remote OpenBench root path from the controller config."""
        try:
            # Try to get controller from parent widget chain
            parent = self.parent()
            while parent:
                if hasattr(parent, 'controller'):
                    controller = parent.controller
                    # Get remote OpenBench path from config
                    remote_config = controller.config.get("general", {}).get("remote", {})
                    remote_path = remote_config.get("openbench_path", "")
                    if remote_path:
                        return remote_path.rstrip('/').replace('\\', '/')
                    # Fallback to project_root if available
                    if hasattr(controller, 'project_root') and controller.project_root:
                        return controller.project_root.rstrip('/').replace('\\', '/')
                    break
                parent = parent.parent()
        except Exception as e:
            logger.debug("Failed to get remote OpenBench root: %s", e)
        return ""

    def _convert_path(self, path: str) -> str:
        """Convert relative path to absolute path.

        In remote mode, relative paths are converted using remote OpenBench root.
        In local mode, relative paths are converted using local openbench_root.
        """
        if not path:
            return path

        # Normalize slashes first
        path = path.replace('\\', '/')

        # Check if already absolute
        if path.startswith('/'):
            return path

        if self._is_remote_mode():
            # Remote mode: convert relative to absolute using remote OpenBench root
            remote_root = self._get_remote_openbench_root()
            if remote_root:
                # Handle ./ prefix
                if path.startswith('./'):
                    path = path[2:]
                return f"{remote_root}/{path}"
            # If no remote root available, keep path as-is
            return path
        else:
            # Local mode: convert to absolute path
            return to_absolute_path(path, get_openbench_root())

    def _setup_ui(self):
        """Setup dialog UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 10, 0)  # Right margin for scrollbar

        # === Load from File Button ===
        load_layout = QHBoxLayout()
        self.btn_load_file = QPushButton("Load from File...")
        self.btn_load_file.setToolTip("Load configuration from an existing YAML file")
        self.btn_load_file.clicked.connect(self._load_from_file)
        load_layout.addWidget(self.btn_load_file)
        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Source name (if new)
        if not self.source_name:
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Source Name:"))
            self.name_input = QLineEdit()
            self.name_input.setPlaceholderText("e.g., GLEAM4.2a_monthly")
            name_layout.addWidget(self.name_input)
            layout.addLayout(name_layout)

        # === Basic Settings ===
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout(basic_group)

        # Root directory
        self.root_dir = PathSelector(mode="directory", placeholder="Data root directory")
        basic_layout.addRow("Root Directory:", self.root_dir)

        # Data type
        type_layout = QHBoxLayout()
        self.type_group = QButtonGroup(self)
        self.radio_grid = QRadioButton("Grid")
        self.radio_station = QRadioButton("Station")
        self.type_group.addButton(self.radio_grid)
        self.type_group.addButton(self.radio_station)
        self.radio_grid.setChecked(True)
        type_layout.addWidget(self.radio_grid)
        type_layout.addWidget(self.radio_station)
        type_layout.addStretch()
        basic_layout.addRow("Data Type:", type_layout)

        # Data groupby
        self.groupby_combo = NoScrollComboBox()
        self.groupby_combo.addItems(["Year", "Month", "Day", "Single"])
        basic_layout.addRow("Data Groupby:", self.groupby_combo)

        # Station list file (for station data, optional)
        self.fulllist_label = QLabel("Station List:")
        self.fulllist = PathSelector(
            mode="file",
            filter="CSV Files (*.csv);;All Files (*)",
            placeholder="Station list CSV file (optional)"
        )
        basic_layout.addRow(self.fulllist_label, self.fulllist)

        # Connect radio buttons to show/hide station fields
        self.radio_grid.toggled.connect(self._on_data_type_changed)
        self.radio_station.toggled.connect(self._on_data_type_changed)

        # Model definition (for sim data) - add to basic settings
        if self.source_type == "sim":
            # Model definition with New and Show Variables buttons
            model_row = QHBoxLayout()
            self.model_nml = PathSelector(
                mode="file",
                filter="YAML Files (*.yaml)",
                placeholder="Model definition file"
            )
            model_row.addWidget(self.model_nml, 1)

            self.btn_new_model = QPushButton("New...")
            self.btn_new_model.setFixedWidth(60)
            self.btn_new_model.clicked.connect(self._create_new_model)
            model_row.addWidget(self.btn_new_model)

            self.btn_edit_model = QPushButton("Edit")
            self.btn_edit_model.setFixedWidth(50)
            self.btn_edit_model.setToolTip("Edit variables defined in the model file")
            self.btn_edit_model.clicked.connect(self._edit_model_definition)
            model_row.addWidget(self.btn_edit_model)

            self.model_def_label = QLabel("Model Definition:")
            basic_layout.addRow(self.model_def_label, model_row)

            # Connect model path change to auto-populate variable defaults
            self.model_nml.path_changed.connect(self._on_model_changed)

        layout.addWidget(basic_group)

        # === Time Settings ===
        time_group = QGroupBox("Time Settings")
        time_layout = QFormLayout(time_group)

        # Time resolution
        self.tim_res_combo = NoScrollComboBox()
        self.tim_res_combo.addItems(["Month", "Day", "Hour", "Year", "Climatology-month", "Climatology-year"])
        time_layout.addRow("Time Resolution:", self.tim_res_combo)

        # Per-variable time range checkbox
        self.cb_per_var_time_range = QCheckBox("Use per-variable time range")
        self.cb_per_var_time_range.setToolTip(
            "When enabled, this variable uses its own start/end year settings.\n"
            "When disabled, uses the Year Range from General Settings."
        )
        self.cb_per_var_time_range.stateChanged.connect(self._on_per_var_time_range_changed)
        time_layout.addRow("", self.cb_per_var_time_range)

        # Year range (use QLineEdit to allow empty values for station data)
        year_layout = QHBoxLayout()
        self.syear_input = QLineEdit()
        self.syear_input.setPlaceholderText("Start year (e.g., 2000)")
        self.syear_input.setFixedWidth(120)
        year_layout.addWidget(self.syear_input)
        self.year_range_to_label = QLabel("to")
        year_layout.addWidget(self.year_range_to_label)
        self.eyear_input = QLineEdit()
        self.eyear_input.setPlaceholderText("End year (e.g., 2020)")
        self.eyear_input.setFixedWidth(120)
        year_layout.addWidget(self.eyear_input)
        year_layout.addStretch()
        self.year_range_label = QLabel("Year Range:")
        time_layout.addRow(self.year_range_label, year_layout)

        # Initialize year range enabled state
        self._update_year_range_tooltip()

        # Timezone
        self.timezone_spin = NoScrollDoubleSpinBox()
        self.timezone_spin.setRange(-12.0, 14.0)
        self.timezone_spin.setValue(0.0)
        self.timezone_spin.setSingleStep(0.5)
        time_layout.addRow("Timezone Offset:", self.timezone_spin)

        # Grid resolution (for grid data, use QLineEdit to allow empty values)
        self.grid_res_label = QLabel("Grid Resolution:")
        self.grid_res_input = QLineEdit()
        self.grid_res_input.setPlaceholderText("e.g., 2.0")
        self.grid_res_input.setFixedWidth(120)
        time_layout.addRow(self.grid_res_label, self.grid_res_input)

        layout.addWidget(time_group)

        # === Variable Mapping (for ref data) or File Naming (for sim data) ===
        if self.source_type == "ref":
            # For reference data: variable-specific settings
            var_group_title = f"Variable Settings ({self.var_name.replace('_', ' ')})" if self.var_name else "Variable Settings"
            var_group = QGroupBox(var_group_title)
            var_layout = QFormLayout(var_group)

            # Sub directory (optional, relative to root_dir)
            self.sub_dir_input = QLineEdit()
            self.sub_dir_input.setPlaceholderText("Subdirectory (e.g., Latent_Heat/FLUXCOM)")
            var_layout.addRow("Sub Directory:", self.sub_dir_input)

            self.varname_input = QLineEdit()
            self.varname_input.setPlaceholderText("Variable name in file (e.g., E)")
            var_layout.addRow("Variable Name:", self.varname_input)

            self.varunit_input = QLineEdit()
            self.varunit_input.setPlaceholderText("e.g., mm month-1")
            var_layout.addRow("Variable Unit:", self.varunit_input)

            self.prefix_input = QLineEdit()
            self.prefix_input.setPlaceholderText("File prefix (e.g., E_)")
            var_layout.addRow("File Prefix:", self.prefix_input)

            self.suffix_input = QLineEdit()
            self.suffix_input.setPlaceholderText("File suffix (e.g., _GLEAM_v4.2a)")
            var_layout.addRow("File Suffix:", self.suffix_input)

            layout.addWidget(var_group)
        else:
            # For simulation data: variable settings with defaults from model definition
            var_group_title = f"Variable Settings ({self.var_name.replace('_', ' ')})" if self.var_name else "Variable Settings"
            var_group = QGroupBox(var_group_title)
            var_layout = QFormLayout(var_group)

            # Sub directory (optional, relative to root_dir)
            self.sub_dir_input = QLineEdit()
            self.sub_dir_input.setPlaceholderText("Subdirectory (optional)")
            var_layout.addRow("Sub Directory:", self.sub_dir_input)

            self.varname_input = QLineEdit()
            self.varname_input.setPlaceholderText("Variable name in file (from model definition)")
            var_layout.addRow("Variable Name:", self.varname_input)

            self.varunit_input = QLineEdit()
            self.varunit_input.setPlaceholderText("Variable unit (from model definition)")
            var_layout.addRow("Variable Unit:", self.varunit_input)

            # File naming (now in Variable Settings for sim data too)
            self.prefix_input = QLineEdit()
            self.prefix_input.setPlaceholderText("File prefix (e.g., Case01_hist_)")
            var_layout.addRow("File Prefix:", self.prefix_input)

            self.suffix_input = QLineEdit()
            self.suffix_input.setPlaceholderText("File suffix (optional)")
            var_layout.addRow("File Suffix:", self.suffix_input)

            layout.addWidget(var_group)

        # Finalize scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area, 1)  # Stretch factor 1 to fill space

        # === Dialog Buttons (outside scroll area) ===
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

    def _setup_remote_browsing(self):
        """Setup remote browsing for PathSelector widgets if ssh_manager is provided."""
        if not self._ssh_manager:
            return

        # Skip local path validation for remote paths (they won't exist locally)
        self.root_dir.set_skip_validation(True)
        self.fulllist.set_skip_validation(True)

        # Set custom browse handlers for PathSelector widgets
        self.root_dir.set_custom_browse_handler(
            lambda: self._browse_remote_path(self.root_dir, "directory")
        )
        self.fulllist.set_custom_browse_handler(
            lambda: self._browse_remote_path(self.fulllist, "file")
        )

        # Model definition (sim data only)
        if self.source_type == "sim" and hasattr(self, 'model_nml'):
            self.model_nml.set_skip_validation(True)
            self.model_nml.set_custom_browse_handler(
                lambda: self._browse_remote_path(self.model_nml, "file")
            )

    def _browse_remote_path(self, path_selector, mode: str):
        """Browse remote server for path."""
        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(
                self, "Not Connected",
                "Remote server is not connected."
            )
            return

        # Get start path
        current_path = path_selector.path()
        if current_path:
            start_path = current_path if mode == "directory" else current_path.rsplit('/', 1)[0]
        else:
            try:
                start_path = self._ssh_manager._get_home_dir()
            except Exception as e:
                logger.debug("Failed to get remote home directory: %s", e)
                start_path = "/"

        # Create dialog with remote file browser
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select {'Directory' if mode == 'directory' else 'File'} on Remote Server")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        browser = RemoteFileBrowser(
            self._ssh_manager, start_path, dialog,
            select_dirs=(mode == "directory")
        )
        layout.addWidget(browser)

        def on_path_selected(path):
            path_selector.set_path(path)
            dialog.accept()

        browser.file_selected.connect(on_path_selected)
        dialog.exec_()

    def _on_data_type_changed(self):
        """Show/hide fields based on data type selection."""
        is_station = self.radio_station.isChecked()
        # Show fulllist only for station data
        self.fulllist_label.setVisible(is_station)
        self.fulllist.setVisible(is_station)
        # Show grid_res only for grid data
        self.grid_res_label.setVisible(not is_station)
        self.grid_res_input.setVisible(not is_station)

    def _on_per_var_time_range_changed(self, state):
        """Handle per-variable time range checkbox change."""
        self._update_year_range_tooltip()

    def _update_year_range_tooltip(self):
        """Update year range tooltip based on per_var_time_range checkbox.

        Year range fields are always editable.
        When per-variable time range is enabled, values go to per-variable section in namelist.
        When disabled, values are ignored and General Settings values are used.
        """
        if not self.cb_per_var_time_range.isChecked():
            tooltip = "When 'Use per-variable time range' is unchecked, this value is ignored.\nGeneral Settings Year Range will be used instead."
            self.syear_input.setToolTip(tooltip)
            self.eyear_input.setToolTip(tooltip)
        else:
            self.syear_input.setToolTip("Start year for this variable (will be in namelist)")
            self.eyear_input.setToolTip("End year for this variable (will be in namelist)")

    def _load_data(self):
        """Load initial data into form.

        For ref data: prefix/suffix are variable-specific (at top level of data)
        For sim data: prefix/suffix are in general section (shared across variables)
        """
        data = self.initial_data
        if not data:
            return

        general = data.get("general", data)

        # Handle both "root_dir" (ref) and "dir" (sim) field names
        root_dir_value = general.get("root_dir") or general.get("dir", "")
        if root_dir_value:
            root_dir_value = self._convert_path(root_dir_value)
            self.root_dir.set_path(root_dir_value)

        if "data_type" in general:
            # Support both "stn" and "station" as station data type
            if general["data_type"] in ("stn", "station"):
                self.radio_station.setChecked(True)
            else:
                self.radio_grid.setChecked(True)
            self._on_data_type_changed()  # Update visibility

        if "data_groupby" in general:
            idx = self.groupby_combo.findText(general["data_groupby"], Qt.MatchFixedString)
            if idx >= 0:
                self.groupby_combo.setCurrentIndex(idx)

        if "tim_res" in general:
            idx = self.tim_res_combo.findText(general["tim_res"], Qt.MatchFixedString)
            if idx >= 0:
                self.tim_res_combo.setCurrentIndex(idx)

        # Load per-variable time range setting
        if "per_var_time_range" in general:
            self.cb_per_var_time_range.setChecked(general["per_var_time_range"])
            self._update_year_range_tooltip()

        # Load syear/eyear - check top level first (variable-specific), then general section
        if "syear" in data:
            self.syear_input.setText(str(data["syear"]))
        elif "syear" in general:
            self.syear_input.setText(str(general["syear"]))
        if "eyear" in data:
            self.eyear_input.setText(str(data["eyear"]))
        elif "eyear" in general:
            self.eyear_input.setText(str(general["eyear"]))
        if "timezone" in general:
            try:
                self.timezone_spin.setValue(float(general["timezone"]))
            except (ValueError, TypeError):
                pass
        if "grid_res" in general:
            self.grid_res_input.setText(str(general["grid_res"]))
        if "fulllist" in general:
            # Convert fulllist to absolute path when loading (only in local mode)
            fulllist = general["fulllist"]
            if fulllist:
                fulllist = self._convert_path(fulllist)
            self.fulllist.set_path(fulllist)

        # Load variable mapping fields (for both ref and sim)
        # Check top level first, then general section for backward compatibility
        if "sub_dir" in data:
            self.sub_dir_input.setText(str(data["sub_dir"]))
        if "varname" in data:
            self.varname_input.setText(str(data["varname"]))
        if "varunit" in data:
            self.varunit_input.setText(str(data["varunit"]))

        # Load prefix/suffix from top level or general section
        if "prefix" in data:
            self.prefix_input.setText(str(data["prefix"]))
        elif "prefix" in general:
            self.prefix_input.setText(str(general["prefix"]))
        if "suffix" in data:
            self.suffix_input.setText(str(data["suffix"]))
        elif "suffix" in general:
            self.suffix_input.setText(str(general["suffix"]))

        # Model definition for sim - convert to absolute path (only in local mode)
        if self.source_type == "sim" and "model_namelist" in general:
            model_nml = general["model_namelist"]
            if model_nml:
                model_nml = self._convert_path(model_nml)
            self.model_nml.set_path(model_nml)

    def _load_from_file(self):
        """Load configuration from an existing YAML file."""
        import os

        # Get file path from user
        file_path = self._prompt_for_yaml_file()
        if not file_path:
            return

        # Load and parse YAML content
        content = self._load_yaml_content(file_path)
        if content is None:
            return

        # Reset all form fields before loading new values
        self._reset_form_fields()

        # Update source name from filename
        if hasattr(self, 'name_input'):
            source_name = os.path.splitext(os.path.basename(file_path))[0]
            self.name_input.setText(source_name)

        # Populate form fields
        general = content.get("general", {})
        self._populate_general_settings(general)
        self._populate_variable_settings(content, general)

        QMessageBox.information(
            self, "Loaded",
            f"Configuration loaded from:\n{os.path.basename(file_path)}"
        )

    def _reset_form_fields(self):
        """Reset all form fields to default/empty values before loading new data."""
        # Reset path fields
        self.root_dir.set_path("")
        if hasattr(self, 'fulllist'):
            self.fulllist.set_path("")

        # Reset data type to default (grid)
        self.radio_grid.setChecked(True)
        self._on_data_type_changed()

        # Reset combo boxes to first item
        self.groupby_combo.setCurrentIndex(0)
        self.tim_res_combo.setCurrentIndex(0)

        # Reset text inputs
        self.syear_input.clear()
        self.eyear_input.clear()
        self.grid_res_input.clear()

        # Reset timezone
        self.timezone_spin.setValue(0.0)

        # Reset variable-specific fields
        if hasattr(self, 'sub_dir_input'):
            self.sub_dir_input.clear()
        if hasattr(self, 'varname_input'):
            self.varname_input.clear()
        if hasattr(self, 'varunit_input'):
            self.varunit_input.clear()
        if hasattr(self, 'prefix_input'):
            self.prefix_input.clear()
        if hasattr(self, 'suffix_input'):
            self.suffix_input.clear()

        # Reset sim-specific fields
        if self.source_type == "sim" and hasattr(self, 'model_nml'):
            self.model_nml.set_path("")

    def _get_local_openbench_path(self) -> str:
        """Get local OpenBench path from config or runtime settings."""
        import os
        import yaml as yaml_module

        # Try from controller if available
        if hasattr(self, '_controller') and self._controller:
            general = self._controller.config.get("general", {})
            local_path = general.get("local_openbench_path", "")
            if local_path and os.path.isdir(local_path):
                return local_path

        # Try from runtime settings file
        try:
            settings_path = os.path.join(
                os.path.expanduser("~"), ".openbench_wizard", "runtime_settings.yaml"
            )
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = yaml_module.safe_load(f) or {}
                    saved_path = settings.get("local_openbench_path", "")
                    if saved_path and os.path.isdir(saved_path):
                        return saved_path
        except Exception:
            pass

        # Fall back to detected OpenBench root
        return get_openbench_root()

    def _prompt_for_yaml_file(self) -> str:
        """Open file dialog to select a YAML file. Returns file path or empty string."""
        import os

        if self._ssh_manager and self._ssh_manager.is_connected:
            # Use remote file browser
            return self._prompt_for_remote_yaml_file()

        openbench_root = self._get_local_openbench_path()
        default_dir = os.path.join(openbench_root, "nml", "nml-yaml")

        if self.source_type == "ref":
            ref_dir = os.path.join(default_dir, "Ref_variables_definition_LowRes")
            if os.path.exists(ref_dir):
                default_dir = ref_dir
        else:
            user_dir = os.path.join(default_dir, "user")
            if os.path.exists(user_dir):
                default_dir = user_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration from YAML",
            default_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        return file_path

    def _prompt_for_remote_yaml_file(self) -> str:
        """Open remote file browser to select a YAML file."""
        # Get OpenBench path from remote config or use home
        try:
            home = self._ssh_manager._get_home_dir()
            # Try to find OpenBench nml directory
            stdout, stderr, exit_code = self._ssh_manager.execute(
                f"ls -d {home}/OpenBench/nml/nml-yaml 2>/dev/null || echo {home}",
                timeout=10
            )
            start_path = stdout.strip() if exit_code == 0 else home
        except Exception as e:
            logger.debug("Failed to get remote OpenBench path: %s", e)
            start_path = "/"

        # Create dialog with remote file browser
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Configuration from Remote YAML File")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        browser = RemoteFileBrowser(self._ssh_manager, start_path, dialog, select_dirs=False)
        layout.addWidget(browser)

        selected_path = [None]  # Use list to capture value from inner function

        def on_path_selected(path):
            # Only accept yaml/yml files
            if path.endswith('.yaml') or path.endswith('.yml'):
                selected_path[0] = path
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Invalid File", "Please select a YAML file (.yaml or .yml)")

        browser.file_selected.connect(on_path_selected)
        dialog.exec_()

        return selected_path[0] or ""

    def _load_yaml_content(self, file_path: str) -> dict:
        """Load and parse YAML file. Returns content dict or None on error."""
        import yaml

        if self._ssh_manager and self._ssh_manager.is_connected:
            # Load from remote file
            return self._load_remote_yaml_content(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            QMessageBox.warning(
                self, "Load Error",
                f"Failed to load file:\n{file_path}\n\nError: {str(e)}"
            )
            return None

    def _load_remote_yaml_content(self, file_path: str) -> dict:
        """Load and parse YAML file from remote server."""
        import yaml

        try:
            stdout, stderr, exit_code = self._ssh_manager.execute(
                f"cat '{file_path}'", timeout=30
            )
            if exit_code != 0:
                QMessageBox.warning(
                    self, "Load Error",
                    f"Failed to load remote file:\n{file_path}\n\nError: {stderr}"
                )
                return None
            return yaml.safe_load(stdout) or {}
        except Exception as e:
            QMessageBox.warning(
                self, "Load Error",
                f"Failed to load remote file:\n{file_path}\n\nError: {str(e)}"
            )
            return None

    def _populate_general_settings(self, general: dict):
        """Populate form fields from general section of config."""
        # Root directory
        root_dir = general.get("root_dir") or general.get("dir", "")
        if root_dir:
            root_dir = self._convert_path(root_dir)
            self.root_dir.set_path(root_dir)

        # Data type
        if "data_type" in general:
            if general["data_type"] in ("stn", "station"):
                self.radio_station.setChecked(True)
            else:
                self.radio_grid.setChecked(True)
            self._on_data_type_changed()

        # Data groupby
        if "data_groupby" in general:
            idx = self.groupby_combo.findText(general["data_groupby"], Qt.MatchFixedString)
            if idx >= 0:
                self.groupby_combo.setCurrentIndex(idx)

        # Time settings
        if "tim_res" in general:
            idx = self.tim_res_combo.findText(general["tim_res"], Qt.MatchFixedString)
            if idx >= 0:
                self.tim_res_combo.setCurrentIndex(idx)

        if "syear" in general:
            self.syear_input.setText(str(general["syear"]))
        if "eyear" in general:
            self.eyear_input.setText(str(general["eyear"]))
        if "timezone" in general:
            try:
                self.timezone_spin.setValue(float(general["timezone"]))
            except (ValueError, TypeError):
                pass
        if "grid_res" in general:
            self.grid_res_input.setText(str(general["grid_res"]))
        if "fulllist" in general and general["fulllist"]:
            self.fulllist.set_path(self._convert_path(general["fulllist"]))

    def _populate_variable_settings(self, content: dict, general: dict):
        """Populate variable-specific settings based on source type."""
        if self.source_type == "ref":
            self._populate_ref_variable_settings(content)
        else:
            self._populate_sim_variable_settings(content, general)

    def _populate_ref_variable_settings(self, content: dict):
        """Populate reference data variable-specific fields."""
        var_config = None
        if self.var_name and self.var_name in content:
            var_config = content[self.var_name]
        elif self.var_name:
            # Variable not found in file, show info
            available_vars = [k for k in content.keys() if k != "general"]
            if available_vars:
                QMessageBox.information(
                    self, "Variable Not Found",
                    f"Variable '{self.var_name}' not found in file.\n\n"
                    f"Available variables: {', '.join(available_vars[:10])}"
                    f"{'...' if len(available_vars) > 10 else ''}\n\n"
                    "General settings have been loaded."
                )

        if var_config:
            if "sub_dir" in var_config:
                self.sub_dir_input.setText(str(var_config["sub_dir"]))
            if "varname" in var_config:
                self.varname_input.setText(str(var_config["varname"]))
            if "varunit" in var_config:
                self.varunit_input.setText(str(var_config["varunit"]))
            if "prefix" in var_config:
                self.prefix_input.setText(str(var_config["prefix"]))
            if "suffix" in var_config:
                self.suffix_input.setText(str(var_config["suffix"]))
            # Load variable-specific syear/eyear (for per-variable time range)
            if "syear" in var_config:
                self.syear_input.setText(str(var_config["syear"]))
            if "eyear" in var_config:
                self.eyear_input.setText(str(var_config["eyear"]))

    def _populate_sim_variable_settings(self, content: dict, general: dict):
        """Populate simulation data variable-specific fields."""
        # For sim data: prefix/suffix from general
        if "prefix" in general:
            self.prefix_input.setText(str(general["prefix"]))
        if "suffix" in general:
            self.suffix_input.setText(str(general["suffix"]))

        # Load variable-specific fields from content (top level)
        # These might be saved per-variable in the wizard config
        if "sub_dir" in content:
            self.sub_dir_input.setText(str(content["sub_dir"]))
        if "varname" in content:
            self.varname_input.setText(str(content["varname"]))
        if "varunit" in content:
            self.varunit_input.setText(str(content["varunit"]))

        # Model namelist
        if "model_namelist" in general and general["model_namelist"]:
            model_path = self._convert_path(general["model_namelist"])
            self.model_nml.set_path(model_path)

    def _on_model_changed(self, path: str, force: bool = False):
        """Auto-populate varname and varunit from model definition when model is selected.

        Args:
            path: Path to the model definition file
            force: If True, update fields even if they already have values
        """
        if not path or not self.var_name:
            return

        import yaml
        import os

        content = None

        if self._is_remote_mode():
            # Load from remote server
            try:
                stdout, stderr, exit_code = self._ssh_manager.execute(
                    f"cat '{path}'", timeout=30
                )
                if exit_code == 0 and stdout.strip():
                    content = yaml.safe_load(stdout) or {}
            except Exception:
                return
        else:
            # Load from local file
            full_path = to_absolute_path(path, get_openbench_root())
            if not os.path.exists(full_path):
                return

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f) or {}
            except Exception:
                return

        if not content:
            return

        # Look up current variable in model definition
        if self.var_name in content:
            var_config = content[self.var_name]

            # Update fields (only if empty, unless force=True)
            if "varname" in var_config:
                if force or not self.varname_input.text():
                    self.varname_input.setText(str(var_config["varname"]))
            if "varunit" in var_config:
                if force or not self.varunit_input.text():
                    self.varunit_input.setText(str(var_config["varunit"]))

    def _edit_model_definition(self):
        """Edit the selected model definition file."""
        import yaml
        import os
        from openbench.gui.widgets.model_definition_editor import ModelDefinitionEditor

        model_path = self.model_nml.path()
        if not model_path:
            QMessageBox.information(
                self, "No Model Selected",
                "Please select a model definition file first."
            )
            return

        content = None

        if self._is_remote_mode():
            # Remote mode: load file from remote server
            try:
                stdout, stderr, exit_code = self._ssh_manager.execute(
                    f"cat '{model_path}'", timeout=30
                )
                if exit_code != 0:
                    QMessageBox.warning(
                        self, "Load Error",
                        f"Failed to load remote file:\n{model_path}\n\nError: {stderr}"
                    )
                    return
                content = yaml.safe_load(stdout) or {}
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error",
                    f"Failed to load remote file:\n{str(e)}"
                )
                return

            # Open editor with remote support
            dialog = ModelDefinitionEditor(
                initial_data=content,
                file_path=model_path,
                ssh_manager=self._ssh_manager,
                parent=self
            )
        else:
            # Local mode: resolve path and edit
            full_path = to_absolute_path(model_path, get_openbench_root())

            if not os.path.exists(full_path):
                QMessageBox.warning(
                    self, "File Not Found",
                    f"Model definition file not found:\n{full_path}"
                )
                return

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f) or {}
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error",
                    f"Failed to load model file:\n{str(e)}"
                )
                return

            # Open editor without remote support
            dialog = ModelDefinitionEditor(
                initial_data=content,
                file_path=full_path,
                parent=self
            )

        dialog.exec()
        # Always refresh after dialog closes (user may have saved)
        self._on_model_changed(model_path, force=True)

    def get_data(self) -> Dict[str, Any]:
        """Get form data as dictionary with absolute paths.

        For ref data: variable-specific fields (sub_dir, varname, varunit, prefix, suffix)
                     are stored at top level of returned dict
        For sim data: prefix/suffix are stored in general section

        In remote mode, paths are kept as-is (they are already remote paths).
        """
        is_station = self.radio_station.isChecked()

        # Convert root_dir to absolute path (only in local mode)
        root_dir = self.root_dir.path()
        if root_dir:
            root_dir = self._convert_path(root_dir)

        # Build general section
        # Use "dir" for sim data, "root_dir" for ref data
        general = {
            "data_type": "stn" if is_station else "grid",
            "data_groupby": self.groupby_combo.currentText(),
            "tim_res": self.tim_res_combo.currentText(),
            "timezone": self.timezone_spin.value(),
            "per_var_time_range": self.cb_per_var_time_range.isChecked(),
        }
        if self.source_type == "sim":
            general["dir"] = root_dir
        else:
            general["root_dir"] = root_dir

        # Handle year fields (preserve empty strings for station data)
        syear_text = self.syear_input.text().strip()
        eyear_text = self.eyear_input.text().strip()
        # Use try/except for robust int conversion (handles negative numbers too)
        if syear_text:
            try:
                general["syear"] = int(syear_text)
            except ValueError:
                general["syear"] = syear_text  # Keep as string if not a valid int
        else:
            general["syear"] = ""
        if eyear_text:
            try:
                general["eyear"] = int(eyear_text)
            except ValueError:
                general["eyear"] = eyear_text  # Keep as string if not a valid int
        else:
            general["eyear"] = ""

        # Handle grid_res (preserve empty strings)
        grid_res_text = self.grid_res_input.text().strip()
        if grid_res_text:
            try:
                general["grid_res"] = float(grid_res_text)
            except ValueError:
                general["grid_res"] = grid_res_text
        else:
            general["grid_res"] = ""

        # Add fulllist for station data (optional) - convert to absolute (only in local mode)
        if is_station:
            fulllist_path = self.fulllist.path()
            if fulllist_path:
                general["fulllist"] = self._convert_path(fulllist_path)

        data = {"general": general}

        # Variable-specific fields at top level (for both ref and sim)
        if self.sub_dir_input.text():
            data["sub_dir"] = self.sub_dir_input.text()
        if self.varname_input.text():
            data["varname"] = self.varname_input.text()
        if self.varunit_input.text():
            data["varunit"] = self.varunit_input.text()
        if self.prefix_input.text():
            data["prefix"] = self.prefix_input.text()
        if self.suffix_input.text():
            data["suffix"] = self.suffix_input.text()

        # Add model definition for sim - convert to absolute (only in local mode)
        if self.source_type == "sim":
            model_path = self.model_nml.path()
            if model_path:
                model_path = self._convert_path(model_path)
            data["general"]["model_namelist"] = model_path

        return data

    def accept(self):
        """Override accept to validate required fields and paths before closing."""
        from openbench.gui.validation import FieldValidator, ValidationManager

        errors = []
        manager = ValidationManager(self)

        # Validate source name (required for new sources)
        if hasattr(self, 'name_input'):
            error = FieldValidator.required(
                self.name_input.text().strip(),
                "source_name",
                "Source name is required",
                widget=self.name_input
            )
            if error:
                errors.append(error)

        # Validate root_dir (required)
        root_dir = self.root_dir.path()
        error = FieldValidator.required(
            root_dir,
            "root_dir",
            "Root directory is required",
            widget=self.root_dir
        )
        if error:
            errors.append(error)

        # Check varname - show warning if missing but allow continue
        self._varname_missing = not self.varname_input.text().strip()
        self._varunit_missing = not self.varunit_input.text().strip()

        # Grid type specific validations
        if self.radio_grid.isChecked():
            # Validate prefix/suffix (at least one required for grid type only)
            error = FieldValidator.at_least_one(
                [self.prefix_input.text().strip(), self.suffix_input.text().strip()],
                ["prefix", "suffix"],
                "At least one of file prefix or suffix is required",
                widget=self.prefix_input
            )
            if error:
                errors.append(error)
            # Grid resolution required
            grid_res = self.grid_res_input.text().strip()
            error = FieldValidator.required(
                grid_res,
                "grid_res",
                "Grid resolution is required for grid data type",
                widget=self.grid_res_input
            )
            if error:
                errors.append(error)

            # Year range required for grid type
            syear = self.syear_input.text().strip()
            eyear = self.eyear_input.text().strip()

            error = FieldValidator.required(
                syear,
                "syear",
                "Start year is required for grid data type",
                widget=self.syear_input
            )
            if error:
                errors.append(error)

            error = FieldValidator.required(
                eyear,
                "eyear",
                "End year is required for grid data type",
                widget=self.eyear_input
            )
            if error:
                errors.append(error)

            # Validate year range if both provided
            if syear and eyear:
                try:
                    syear_int = int(syear)
                    eyear_int = int(eyear)
                    error = FieldValidator.min_max(
                        syear_int,
                        eyear_int,
                        "year_range",
                        "Start year cannot be greater than end year",
                        widget=self.syear_input
                    )
                    if error:
                        errors.append(error)
                except ValueError:
                    pass  # Invalid number format handled elsewhere

        # Show first error if any
        if errors:
            manager.show_error_and_focus(errors[0])
            return

        # Path validation (skip in remote mode)
        if not self._is_remote_mode():
            if root_dir:
                root_dir = to_absolute_path(root_dir, get_openbench_root())
                is_valid, error_msg = validate_path(root_dir, "directory")
                if not is_valid:
                    error = ValidationError(
                        "root_dir",
                        f"Root directory does not exist: {root_dir}",
                        "",
                        self.root_dir
                    )
                    manager.show_error_and_focus(error)
                    return

            # Validate fulllist if station data
            if self.radio_station.isChecked():
                fulllist_path = self.fulllist.path()
                if fulllist_path:
                    fulllist_path = to_absolute_path(fulllist_path, get_openbench_root())
                    is_valid, error_msg = validate_path(fulllist_path, "file")
                    if not is_valid:
                        error = ValidationError(
                            "fulllist",
                            f"Station list file does not exist: {fulllist_path}",
                            "",
                            self.fulllist
                        )
                        manager.show_error_and_focus(error)
                        return

            # Validate model_namelist for sim data
            if self.source_type == "sim":
                model_path = self.model_nml.path()
                if model_path:
                    model_path = to_absolute_path(model_path, get_openbench_root())
                    is_valid, error_msg = validate_path(model_path, "file")
                    if not is_valid:
                        error = ValidationError(
                            "model_nml",
                            f"Model definition file does not exist: {model_path}",
                            "",
                            self.model_nml
                        )
                        manager.show_error_and_focus(error)
                        return

        # Show warning if varname or varunit is missing
        if self._varname_missing or self._varunit_missing:
            missing_items = []
            if self._varname_missing:
                missing_items.append("Variable Name")
            if self._varunit_missing:
                missing_items.append("Variable Unit")

            reply = QMessageBox.warning(
                self,
                "Missing Variable Information",
                f"{', '.join(missing_items)} not set.\n\n"
                f"Please confirm the variable name and unit are defined "
                f"in the filter configuration (e.g., CoLM_filter.py).\n\n"
                f"Click 'Yes' to continue, 'No' to go back and edit.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        super().accept()

    def get_source_name(self) -> str:
        """Get source name."""
        if hasattr(self, 'name_input'):
            return self.name_input.text()
        return self.source_name

    def _create_new_model(self):
        """Create a new model definition file."""
        from openbench.gui.widgets.model_definition_editor import ModelDefinitionEditor

        # Pass ssh_manager for remote mode support
        dialog = ModelDefinitionEditor(
            ssh_manager=self._ssh_manager,
            parent=self
        )
        if dialog.exec():
            file_path = dialog.get_saved_path()
            if file_path:
                self.model_nml.set_path(file_path)

    def _cleanup(self):
        """Clean up resources before dialog destruction.

        Clears custom browse handlers and signal connections to prevent
        crashes when reopening the dialog.
        """
        # Clear custom browse handlers (they hold lambdas that reference self)
        if hasattr(self, 'root_dir'):
            self.root_dir.set_custom_browse_handler(None)
        if hasattr(self, 'fulllist'):
            self.fulllist.set_custom_browse_handler(None)
        if self.source_type == "sim" and hasattr(self, 'model_nml'):
            self.model_nml.set_custom_browse_handler(None)
            # Disconnect model path change signal
            try:
                self.model_nml.path_changed.disconnect(self._on_model_changed)
            except (RuntimeError, TypeError):
                pass

        # Clear SSH manager reference
        self._ssh_manager = None

    def done(self, result):
        """Override done to ensure cleanup on dialog close."""
        self._cleanup()
        super().done(result)
