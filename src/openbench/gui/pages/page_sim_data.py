# -*- coding: utf-8 -*-
"""
Simulation Data configuration page.

For simulation data, prefix/suffix are shared across all variables at the general level.
This is different from reference data where each variable has its own prefix/suffix.
"""

import logging
import shlex
from typing import Dict, Any

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QListWidget, QLabel, QWidget, QMessageBox, QDialog
)

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import DataSourceEditor
from openbench.gui.path_utils import to_absolute_path, convert_paths_in_dict

logger = logging.getLogger(__name__)


def get_remote_ssh_manager(controller):
    """Get SSH manager from the controller if in remote mode.

    Args:
        controller: The WizardController instance

    Returns:
        SSHManager instance if in remote mode and connected, None otherwise
    """
    # Check storage type to determine if in remote mode
    from openbench.remote.storage import RemoteStorage
    if not isinstance(controller.storage, RemoteStorage):
        return None
    return controller.ssh_manager


class PageSimData(BasePage):
    """Simulation Data configuration page."""

    PAGE_ID = "sim_data"
    PAGE_TITLE = "Simulation Data"
    PAGE_SUBTITLE = "Configure simulation data sources for each evaluation variable"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        self.var_container = QWidget()
        self.var_layout = QVBoxLayout(self.var_container)
        self.var_layout.setContentsMargins(0, 0, 0, 0)
        self.var_layout.setSpacing(15)

        self.content_layout.addWidget(self.var_container)

        self._source_lists: Dict[str, QListWidget] = {}
        # For sim data: _source_configs[var_name][source_name] = {...}
        # prefix/suffix are stored in general section (shared)
        self._source_configs: Dict[str, Dict[str, Any]] = {}

        # Add validate button at bottom
        validate_layout = QHBoxLayout()
        validate_layout.addStretch()
        self.validate_btn = QPushButton("Validate Data")
        self.validate_btn.setToolTip("Check files, variable names, time and spatial ranges for all data sources")
        self.validate_btn.clicked.connect(self._validate_data)
        validate_layout.addWidget(self.validate_btn)
        self.content_layout.addLayout(validate_layout)

    def _rebuild_variable_groups(self):
        """Rebuild variable groups based on selected evaluation items."""
        while self.var_layout.count():
            child = self.var_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._source_lists.clear()

        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]

        if not selected:
            label = QLabel("No evaluation items selected. Please go back and select items.")
            label.setStyleSheet("color: #666; font-style: italic;")
            self.var_layout.addWidget(label)
            return

        for var_name in selected:
            group = QGroupBox(var_name.replace("_", " "))
            group_layout = QVBoxLayout(group)

            # Source list - use minimum height instead of maximum for better space usage
            source_list = QListWidget()
            source_list.setMinimumHeight(60)
            source_list.setProperty("var_name", var_name)
            self._source_lists[var_name] = source_list
            group_layout.addWidget(source_list, 1)  # stretch factor 1 to expand

            btn_layout = QHBoxLayout()

            btn_add = QPushButton("+ Add Source")
            btn_add.setProperty("secondary", True)
            btn_add.clicked.connect(lambda checked, v=var_name: self._add_source(v))
            btn_layout.addWidget(btn_add)

            btn_copy = QPushButton("Copy")
            btn_copy.setProperty("secondary", True)
            btn_copy.setToolTip("Copy selected source as a new source")
            btn_copy.clicked.connect(lambda checked, v=var_name: self._copy_source(v))
            btn_layout.addWidget(btn_copy)

            btn_edit = QPushButton("Edit")
            btn_edit.setProperty("secondary", True)
            btn_edit.clicked.connect(lambda checked, v=var_name: self._edit_source(v))
            btn_layout.addWidget(btn_edit)

            btn_remove = QPushButton("Remove")
            btn_remove.setProperty("secondary", True)
            btn_remove.clicked.connect(lambda checked, v=var_name: self._remove_source(v))
            btn_layout.addWidget(btn_remove)

            btn_layout.addStretch()
            group_layout.addLayout(btn_layout)

            self.var_layout.addWidget(group, 1)  # stretch factor 1 to expand

        # No addStretch() here - let groups expand to fill space

    def _add_source(self, var_name: str):
        """Add new data source."""
        ssh_manager = get_remote_ssh_manager(self.controller)
        dialog = DataSourceEditor(
            source_type="sim",
            var_name=var_name,  # Pass for context (shown in title)
            ssh_manager=ssh_manager,
            parent=self
        )
        if dialog.exec():
            source_name = dialog.get_source_name()
            if source_name:
                if var_name not in self._source_configs:
                    self._source_configs[var_name] = {}
                self._source_configs[var_name][source_name] = dialog.get_data()
                self._update_source_list(var_name)
                self.save_to_config()

    def _copy_source(self, var_name: str):
        """Copy selected data source as a new source."""
        import copy

        source_list = self._source_lists.get(var_name)
        if not source_list:
            return

        current = source_list.currentItem()
        if not current:
            QMessageBox.information(self, "Info", "Please select a source to copy.")
            return

        source_name = current.text()
        existing_data = self._source_configs.get(var_name, {}).get(source_name, {})

        # Deep copy the data to avoid modifying the original
        copied_data = copy.deepcopy(existing_data)
        # Remove def_nml_path so a new one will be generated
        copied_data.pop("def_nml_path", None)

        # Open dialog with copied data but no source name (user must enter new name)
        ssh_manager = get_remote_ssh_manager(self.controller)
        dialog = DataSourceEditor(
            source_type="sim",
            var_name=var_name,
            initial_data=copied_data,
            ssh_manager=ssh_manager,
            parent=self
        )
        if dialog.exec():
            new_source_name = dialog.get_source_name()
            if new_source_name:
                if new_source_name == source_name:
                    QMessageBox.warning(
                        self, "Error",
                        "New source name must be different from the original."
                    )
                    return
                if var_name not in self._source_configs:
                    self._source_configs[var_name] = {}
                self._source_configs[var_name][new_source_name] = dialog.get_data()
                self._update_source_list(var_name)
                self.save_to_config()

    def _edit_source(self, var_name: str):
        """Edit selected data source."""
        source_list = self._source_lists.get(var_name)
        if not source_list:
            return

        current = source_list.currentItem()
        if not current:
            QMessageBox.information(self, "Info", "Please select a source to edit.")
            return

        source_name = current.text()
        existing_data = self._source_configs.get(var_name, {}).get(source_name, {})

        ssh_manager = get_remote_ssh_manager(self.controller)
        dialog = DataSourceEditor(
            source_name=source_name,
            source_type="sim",
            var_name=var_name,  # Pass for context (shown in title)
            initial_data=existing_data,
            ssh_manager=ssh_manager,
            parent=self
        )
        if dialog.exec():
            self._source_configs[var_name][source_name] = dialog.get_data()
            self.save_to_config()

    def _remove_source(self, var_name: str):
        """Remove selected data source."""
        source_list = self._source_lists.get(var_name)
        if not source_list:
            return

        current = source_list.currentItem()
        if not current:
            QMessageBox.information(self, "Info", "Please select a source to remove.")
            return

        source_name = current.text()
        reply = QMessageBox.question(
            self, "Confirm",
            f"Remove source '{source_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if var_name in self._source_configs:
                self._source_configs[var_name].pop(source_name, None)
            self._update_source_list(var_name)
            self.save_to_config()

    def _update_source_list(self, var_name: str):
        """Update the source list widget."""
        source_list = self._source_lists.get(var_name)
        if not source_list:
            return

        source_list.clear()
        sources = self._source_configs.get(var_name, {})
        for source_name in sources.keys():
            source_list.addItem(source_name)

    def load_from_config(self):
        """Load from config.

        For sim data, prefix/suffix are in the general section of the source file,
        shared across all variables.
        Uses compound key "var_name::source_name" for source_configs.
        """
        import os
        import yaml

        # Clear existing source configs before reloading
        self._source_configs.clear()

        self._rebuild_variable_groups()

        sim_data = self.controller.config.get("sim_data", {})
        general_section = sim_data.get("general", {})
        def_nml = sim_data.get("def_nml", {})
        # saved_source_configs now uses compound key: "var_name::source_name"
        saved_source_configs = sim_data.get("source_configs", {})

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage
        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None

        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]

        for var_name in selected:
            key = f"{var_name}_sim_source"
            sources = general_section.get(key, [])
            if isinstance(sources, str):
                sources = [sources]

            self._source_configs[var_name] = {}
            for source_name in sources:
                # Use compound key for per-variable storage
                compound_key = f"{var_name}::{source_name}"

                # First check if we have saved source config (from previous edits)
                if compound_key in saved_source_configs:
                    saved_config = saved_source_configs[compound_key]
                    # Check if cached data has valid root_dir - if not, force re-read from def_nml
                    general = saved_config.get("general", {})
                    root_dir = general.get("root_dir") or general.get("dir", "")
                    if root_dir:
                        self._source_configs[var_name][source_name] = saved_config.copy()
                        self._update_source_list(var_name)
                        continue
                    # Cached data is incomplete, fall through to load from def_nml

                # Otherwise load from def_nml file
                def_nml_path = def_nml.get(source_name, "")
                source_data = {"def_nml_path": def_nml_path}

                # Try to load the actual def_nml file content
                if def_nml_path:
                    nml_content = None

                    if is_remote:
                        # In remote mode, only load from remote server
                        if ssh_manager and ssh_manager.is_connected:
                            remote_path = self._resolve_remote_def_nml_path(ssh_manager, def_nml_path)
                            nml_content = self._load_remote_nml_content(ssh_manager, remote_path)
                        # If not connected in remote mode, don't fall back to local - just skip loading
                    else:
                        # Local mode - load from local file
                        full_path = self._resolve_def_nml_path(def_nml_path)
                        if full_path and os.path.exists(full_path):
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    nml_content = yaml.safe_load(f) or {}
                            except Exception as e:
                                print(f"Warning: Failed to load def_nml file {full_path}: {e}")

                    if nml_content:
                        # For sim data, store the general section
                        # prefix/suffix are at general level
                        if "general" in nml_content:
                            source_data["general"] = nml_content["general"].copy()

                        # Load varname from model definition file
                        model_nml_path = nml_content.get("general", {}).get("model_namelist", "")
                        if model_nml_path:
                            varname = self._load_varname_from_model(
                                model_nml_path, var_name, is_remote, ssh_manager
                            )
                            if varname:
                                source_data["varname"] = varname

                self._source_configs[var_name][source_name] = source_data

            self._update_source_list(var_name)

        # Save loaded configs back to controller to ensure they're available for export
        if self._source_configs:
            self.save_to_config()

    def _load_remote_nml_content(self, ssh_manager, def_nml_path: str) -> dict:
        """Load NML content from remote server."""
        import yaml

        if not def_nml_path:
            return None

        try:
            stdout, stderr, exit_code = ssh_manager.execute(
                f"cat '{def_nml_path}'", timeout=30
            )
            if exit_code == 0 and stdout.strip():
                return yaml.safe_load(stdout) or {}
            else:
                print(f"Warning: Failed to read remote file {def_nml_path}: {stderr}")
        except Exception as e:
            print(f"Warning: Failed to load remote def_nml file {def_nml_path}: {e}")

        return None

    def _resolve_remote_def_nml_path(self, ssh_manager, def_nml_path: str) -> str:
        """Resolve def_nml path on remote server.

        Works the same way as _resolve_def_nml_path for local mode:
        to_absolute_path(def_nml_path, openbench_root)

        Handles both Unix and Windows local paths.
        """
        from openbench.gui.path_utils import to_posix_path

        if not def_nml_path:
            return ""

        # Get remote OpenBench path from config
        general = self.controller.config.get("general", {})
        remote_config = general.get("remote", {})
        remote_openbench_path = remote_config.get("openbench_path", "")

        if not remote_openbench_path:
            return def_nml_path

        # Convert to POSIX format (forward slashes)
        path = to_posix_path(def_nml_path)

        # Handle Windows absolute paths (e.g., C:/Users/...)
        if len(path) >= 2 and path[1] == ':':
            # This is a Windows local path - extract relative part if contains /nml/
            if '/nml/' in path:
                relative_path = 'nml/' + path.split('/nml/', 1)[1]
                return f"{remote_openbench_path.rstrip('/')}/{relative_path}"
            # Unknown Windows path - cannot convert to remote
            return path

        # Extract relative path from various formats
        relative_path = path

        # If path contains /nml/, extract from that point (handles local temp paths)
        if '/nml/' in path:
            relative_path = 'nml/' + path.split('/nml/', 1)[1]
        elif path.startswith('./'):
            relative_path = path[2:]
        elif path.startswith('/'):
            # Check if it's already a valid remote path
            if any(path.startswith(prefix) for prefix in ['/home/', '/share/', '/data/', '/work/', '/scratch/']):
                return path
            # Extract relative portion if contains /nml/
            if '/nml/' in path:
                relative_path = 'nml/' + path.split('/nml/', 1)[1]
            else:
                return path

        # Same as local: to_absolute_path(def_nml_path, openbench_root)
        return f"{remote_openbench_path.rstrip('/')}/{relative_path}"

    def _resolve_def_nml_path(self, def_nml_path: str) -> str:
        """Resolve def_nml path to YAML file."""
        import os

        if not def_nml_path:
            return ""

        # Get OpenBench root
        openbench_root = self._get_openbench_root()

        # Convert to absolute path
        full_path = to_absolute_path(def_nml_path, openbench_root)

        # If already absolute and exists, return it
        if os.path.exists(full_path):
            return full_path

        # Try converting .nml to .yaml
        yaml_path = full_path.replace("nml-Fortran", "nml-yaml").replace(".nml", ".yaml")
        if os.path.exists(yaml_path):
            return yaml_path

        return full_path  # Return even if doesn't exist, let validation catch it

    def _load_varname_from_model(
        self, model_path: str, var_name: str, is_remote: bool, ssh_manager
    ) -> str:
        """Load varname from model definition file for a specific variable.

        Args:
            model_path: Path to model definition file
            var_name: Variable name to look up (e.g., "Evapotranspiration")
            is_remote: Whether in remote mode
            ssh_manager: SSH manager for remote access

        Returns:
            Variable name from model definition, or empty string if not found
        """
        import os
        import yaml

        model_content = None

        if is_remote and ssh_manager and ssh_manager.is_connected:
            # Load from remote
            remote_path = self._resolve_remote_def_nml_path(ssh_manager, model_path)
            try:
                stdout, stderr, exit_code = ssh_manager.execute(
                    f"cat '{remote_path}'", timeout=30
                )
                if exit_code == 0 and stdout.strip():
                    model_content = yaml.safe_load(stdout) or {}
            except Exception as e:
                logger.debug("Failed to load remote model file %s: %s", remote_path, e)
        else:
            # Load from local
            full_path = self._resolve_def_nml_path(model_path)
            if full_path and os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        model_content = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.debug("Failed to load model file %s: %s", full_path, e)

        if model_content and var_name in model_content:
            var_config = model_content[var_name]
            if isinstance(var_config, dict):
                return var_config.get("varname", "")

        return ""

    def save_to_config(self):
        """Save to config.

        Uses compound key "var_name::source_name" for source_configs to preserve
        per-variable configurations even when the same source is used by multiple variables.
        """
        general = {}
        def_nml = {}
        source_configs = {}  # Store full source configurations with compound keys

        for var_name, sources in self._source_configs.items():
            if sources:
                key = f"{var_name}_sim_source"
                general[key] = list(sources.keys())

                for source_name, source_data in sources.items():
                    # Get def_nml_path if it exists, otherwise generate one
                    def_nml_path = source_data.get("def_nml_path", "")
                    if not def_nml_path:
                        # Will be generated during namelist sync
                        basedir = self.controller.config.get("general", {}).get("basedir", "./output")
                        def_nml_path = f"{basedir}/nml/sim/{source_name}.yaml"
                    def_nml[source_name] = def_nml_path

                    # Store with compound key to preserve per-variable configs
                    compound_key = f"{var_name}::{source_name}"
                    source_configs[compound_key] = source_data.copy()
                    # Also store var_name in the config for later retrieval
                    source_configs[compound_key]["_var_name"] = var_name

        sim_data = {
            "general": general,
            "def_nml": def_nml,
            "source_configs": source_configs  # Include full configs for sync
        }
        self.controller.update_section("sim_data", sim_data)

        # Trigger namelist sync
        self.controller.sync_namelists()

    def validate(self) -> bool:
        """Validate page input - ensure all evaluation items have data sources."""
        from openbench.gui.validation import ValidationError, ValidationManager

        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]

        manager = ValidationManager(self)

        for var_name in selected:
            sources = self._source_configs.get(var_name, {})
            if not sources:
                error = ValidationError(
                    field_name="data_source",
                    message=f"{var_name.replace('_', ' ')} is missing simulation data source configuration",
                    page_id=self.PAGE_ID,
                    context={"var_name": var_name}
                )
                if not manager.show_error_and_focus(error):
                    # Auto-open add source dialog
                    self._add_source(var_name)
                    return False

            # Validate each source has required fields
            for source_name, source_data in sources.items():
                # Check varname - can be at top level, in general, or in var_config
                general = source_data.get("general", {})
                var_config = source_data.get("var_config", {})
                varname = (
                    source_data.get("varname") or
                    var_config.get("varname") or
                    general.get("varname") or
                    ""
                )
                if not varname:
                    # Show warning and ask for confirmation
                    reply = QMessageBox.warning(
                        self,
                        "Variable Name Missing",
                        f"Variable name is not set for:\n\n"
                        f"Data source: {source_name}\n"
                        f"Variable: {var_name.replace('_', ' ')}\n\n"
                        f"Is this variable defined in the filter configuration?\n\n"
                        f"Click 'Yes' to continue, 'No' to edit the source.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        self._select_and_edit_source(var_name, source_name)
                        return False

                # Check prefix/suffix (only for grid data, not station data)
                # prefix/suffix can be at top level or in general section
                data_type = general.get("data_type", "grid")
                if data_type != "stn":
                    prefix = source_data.get("prefix") or general.get("prefix") or ""
                    suffix = source_data.get("suffix") or general.get("suffix") or ""
                    if not prefix and not suffix:
                        error = ValidationError(
                            field_name="prefix/suffix",
                            message=f"At least one of file prefix or suffix is required\n\nData source: {source_name}\nVariable: {var_name.replace('_', ' ')}",
                            page_id=self.PAGE_ID,
                            context={"var_name": var_name, "source_name": source_name}
                        )
                        if not manager.show_error_and_focus(error):
                            self._select_and_edit_source(var_name, source_name)
                            return False

                # Check root_dir
                root_dir = general.get("root_dir", "") or general.get("dir", "")
                if not root_dir:
                    error = ValidationError(
                        field_name="root_dir",
                        message=f"Root directory is required\n\nData source: {source_name}\nVariable: {var_name.replace('_', ' ')}",
                        page_id=self.PAGE_ID,
                        context={"var_name": var_name, "source_name": source_name}
                    )
                    if not manager.show_error_and_focus(error):
                        self._select_and_edit_source(var_name, source_name)
                        return False

        self.save_to_config()
        return True

    def _select_and_edit_source(self, var_name: str, source_name: str):
        """Select source in list and open edit dialog."""
        source_list = self._source_lists.get(var_name)
        if source_list:
            # Find and select the item
            for i in range(source_list.count()):
                if source_list.item(i).text() == source_name:
                    source_list.setCurrentRow(i)
                    break
            # Open edit dialog
            self._edit_source(var_name)

    def _validate_data(self):
        """Validate all configured data sources."""
        from openbench.gui.data_validator import DataValidator
        from openbench.gui.widgets.validation_dialog import (
            ValidationProgressDialog, ValidationResultsDialog
        )

        # Check if any sources configured
        if not self._source_configs:
            QMessageBox.information(
                self, "No Data", "No data sources configured. Please add a data source first."
            )
            return

        # Get general config
        general_config = self.controller.config.get("general", {})

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage
        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None

        if is_remote and not ssh_manager:
            QMessageBox.warning(
                self, "Not Connected", "Remote mode requires connecting to the server first."
            )
            return

        # Get remote config for remote mode
        remote_openbench_root = ""
        python_path = ""
        conda_env = ""
        if is_remote:
            remote_config = general_config.get("remote", {})
            remote_openbench_root = remote_config.get("openbench_path", "")
            python_path = remote_config.get("python_path", "")
            conda_env = remote_config.get("conda_env", "")

        # Create validator
        validator = DataValidator(
            is_remote=is_remote,
            ssh_manager=ssh_manager,
            remote_openbench_root=remote_openbench_root,
            python_path=python_path,
            conda_env=conda_env
        )

        # Show progress dialog
        progress_dialog = ValidationProgressDialog(
            validator,
            self._source_configs,
            general_config,
            parent=self
        )

        if progress_dialog.exec() == QDialog.Accepted:
            report = progress_dialog.get_report()
            if report:
                # Show results dialog
                results_dialog = ValidationResultsDialog(report, parent=self)
                results_dialog.exec()
