# -*- coding: utf-8 -*-
"""
Reference Data configuration page.

Data structure for _source_configs:
    _source_configs[var_name][source_name] = {
        "general": {...},           # Shared settings (root_dir, data_type, etc.)
        "var_config": {...},        # Variable-specific settings (sub_dir, varname, prefix, suffix, varunit)
    }

This allows the same source (e.g., GLEAM_v4.2a) to be used by multiple variables
with different per-variable configurations.
"""

import logging
import shlex
from typing import Dict, Any, List

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QScrollArea,
    QWidget,
    QFrame,
    QMessageBox,
    QDialog,
    QComboBox,
)
from PySide6.QtCore import Qt

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


class PageRefData(BasePage):
    """Reference Data configuration page."""

    PAGE_ID = "ref_data"
    PAGE_TITLE = "Reference Data"
    PAGE_SUBTITLE = "Configure reference data sources for each evaluation variable"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        # Container for variable groups
        self.var_container = QWidget()
        self.var_layout = QVBoxLayout(self.var_container)
        self.var_layout.setContentsMargins(0, 0, 0, 0)
        self.var_layout.setSpacing(15)

        self.content_layout.addWidget(self.var_container)

        # Store references to source lists
        self._source_lists: Dict[str, QListWidget] = {}
        # Structure: _source_configs[var_name][source_name] = {"general": {...}, "var_config": {...}}
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
        # Clear existing
        while self.var_layout.count():
            child = self.var_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._source_lists.clear()

        # Get selected evaluation items
        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]

        if not selected:
            label = QLabel("No evaluation items selected. Please go back and select items.")
            label.setStyleSheet("color: #666; font-style: italic;")
            self.var_layout.addWidget(label)
            return

        # Create group for each variable
        for var_name in selected:
            group = QGroupBox(var_name.replace("_", " "))
            group_layout = QVBoxLayout(group)

            # Source list - use minimum height instead of maximum for better space usage
            source_list = QListWidget()
            source_list.setMinimumHeight(60)
            source_list.setProperty("var_name", var_name)
            self._source_lists[var_name] = source_list
            group_layout.addWidget(source_list, 1)  # stretch factor 1 to expand

            # Buttons row 1: Registry quick-add
            registry_layout = QHBoxLayout()

            registry_combo = QComboBox()
            registry_combo.setMinimumWidth(250)
            registry_combo.setProperty("var_name", var_name)
            self._populate_registry_combo(registry_combo, var_name)
            registry_layout.addWidget(registry_combo, stretch=1)

            btn_add_registry = QPushButton("+ Add from Registry")
            btn_add_registry.setProperty("secondary", True)
            btn_add_registry.setToolTip("Add a known reference dataset from the built-in registry")
            btn_add_registry.clicked.connect(
                lambda checked, v=var_name, c=registry_combo: self._add_from_registry(v, c)
            )
            registry_layout.addWidget(btn_add_registry)
            group_layout.addLayout(registry_layout)

            # Buttons row 2: Manual add/edit/remove
            btn_layout = QHBoxLayout()

            btn_add = QPushButton("+ Add Custom")
            btn_add.setProperty("secondary", True)
            btn_add.setToolTip("Manually configure a custom data source")
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

    def _populate_registry_combo(self, combo, var_name):
        """Populate registry combo with available reference datasets for this variable.

        Datasets with multiple resolutions are grouped: selecting one opens
        a resolution picker dialog.
        """
        combo.clear()
        combo.addItem("-- Select from Registry --", None)

        try:
            from openbench.data.registry import RegistryManager

            mgr = RegistryManager()
            refs_with_var = mgr.references_for_variable(var_name)
            if refs_with_var:
                # Group by base name (strip _LowRes/_MidRes/_HigRes suffix)
                groups = {}
                for ref in refs_with_var:
                    base = ref.name
                    for suffix in ("_LowRes", "_MidRes", "_HigRes"):
                        if base.endswith(suffix):
                            base = base[: -len(suffix)]
                            break
                    if base not in groups:
                        groups[base] = []
                    groups[base].append(ref)

                for base_name in sorted(groups.keys()):
                    variants = groups[base_name]
                    if len(variants) == 1:
                        ref = variants[0]
                        label = f"{ref.name}  ({ref.data_type}, {ref.tim_res}, {ref.grid_res or 'stn'})"
                        combo.addItem(label, ref.name)
                    else:
                        # Multiple resolutions available — show as group
                        res_labels = []
                        for v in sorted(variants, key=lambda r: r.name):
                            for s in ("_LowRes", "_MidRes", "_HigRes"):
                                if v.name.endswith(s):
                                    res_labels.append(s[1:])
                                    break
                            else:
                                res_labels.append(v.data_type)
                        label = f"{base_name}  ({' / '.join(res_labels)})"
                        # Store the list of variant names for resolution picker
                        combo.addItem(label, {"group": base_name, "variants": [v.name for v in variants]})
            else:
                combo.addItem("(No registry datasets for this variable)", None)
        except ImportError:
            combo.addItem("(Registry not available)", None)

    def _add_from_registry(self, var_name: str, combo):
        """Add a reference source from the registry with pre-filled config."""
        combo_data = combo.currentData()
        if not combo_data:
            return

        # Handle multi-resolution group: open resolution picker
        if isinstance(combo_data, dict) and "group" in combo_data:
            source_name = self._pick_resolution(combo_data, var_name)
            if not source_name:
                return
        else:
            source_name = combo_data

        # Check for duplicate
        if var_name in self._source_configs and source_name in self._source_configs[var_name]:
            QMessageBox.information(self, "Duplicate", f"'{source_name}' is already added for this variable.")
            return

        try:
            from openbench.data.registry import RegistryManager

            mgr = RegistryManager()
            ref = mgr.get_reference(source_name)
            if ref is None:
                QMessageBox.warning(self, "Not Found", f"Dataset '{source_name}' not found in registry.")
                return

            # Build source_data from registry descriptor
            var_mapping = ref.variables.get(var_name, None)

            general = {
                "root_dir": ref.root_dir or "",
                "data_type": ref.data_type,
                "tim_res": ref.tim_res,
                "data_groupby": ref.data_groupby,
                "timezone": ref.timezone,
                "syear": ref.years[0] if ref.years else "",
                "eyear": ref.years[1] if len(ref.years) > 1 else "",
            }
            if ref.grid_res is not None:
                general["grid_res"] = ref.grid_res
            if ref.fulllist:
                general["fulllist"] = ref.fulllist

            source_data = {"general": general}

            if var_mapping:
                source_data["varname"] = var_mapping.varname
                source_data["varunit"] = var_mapping.varunit
                source_data["prefix"] = var_mapping.prefix
                source_data["suffix"] = var_mapping.suffix
                if var_mapping.sub_dir:
                    source_data["sub_dir"] = var_mapping.sub_dir
                    # If root_dir is empty but sub_dir exists, hint the user
                    if not general["root_dir"]:
                        general["root_dir"] = ""  # User needs to set data_root

            if var_name not in self._source_configs:
                self._source_configs[var_name] = {}
            self._source_configs[var_name][source_name] = source_data
            self._update_source_list(var_name)
            self.save_to_config()

            # Show info if root_dir is empty
            if not general.get("root_dir"):
                QMessageBox.information(
                    self,
                    "Set Data Path",
                    f"'{source_name}' added from registry.\n\n"
                    f"Please edit it to set the data root directory\n"
                    f"(where the reference data files are located).",
                )

        except ImportError:
            QMessageBox.warning(self, "Error", "Registry module not available.")

    def _pick_resolution(self, group_data, var_name):
        """Open resolution picker dialog for a multi-resolution dataset.

        Returns selected source_name or None if cancelled.
        """
        from openbench.data.registry import RegistryManager

        mgr = RegistryManager()
        base_name = group_data["group"]
        variant_names = group_data["variants"]

        # Build variants dict for the dialog
        variants = {}
        for vname in variant_names:
            ref = mgr.get_reference(vname)
            if ref:
                for suffix in ("_LowRes", "_MidRes", "_HigRes"):
                    if vname.endswith(suffix):
                        res_label = suffix[1:]
                        break
                else:
                    res_label = vname
                variants[res_label] = ref

        if not variants:
            return None

        # Check time resolution constraints using frequency hierarchy
        # Rule: only the highest-frequency variant is allowed
        from openbench.data.registry.scanner import _tim_res_rank

        max_rank = max(
            (_tim_res_rank(getattr(v, "tim_res", "")) for v in variants.values()),
            default=-1,
        )
        compatible = None
        if max_rank > 0:
            compatible = [
                res_label
                for res_label, ref in variants.items()
                if _tim_res_rank(getattr(ref, "tim_res", "")) >= max_rank
            ]

        from openbench.gui.dialogs.data_discovery import ResolutionPickerDialog

        dlg = ResolutionPickerDialog(base_name, variants, compatible=compatible, parent=self)
        if dlg.exec():
            res = dlg.selected_resolution()
            if res:
                # Map back to full registry name
                for vname in variant_names:
                    if vname.endswith(f"_{res}"):
                        return vname
                # Fallback: return first variant matching
                return variant_names[0]
        return None

    def _add_source(self, var_name: str):
        """Add new data source for variable."""
        ssh_manager = get_remote_ssh_manager(self.controller)
        dialog = DataSourceEditor(
            source_type="ref",
            var_name=var_name,  # Pass variable name for context
            ssh_manager=ssh_manager,
            parent=self,
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
            source_type="ref", var_name=var_name, initial_data=copied_data, ssh_manager=ssh_manager, parent=self
        )
        if dialog.exec():
            new_source_name = dialog.get_source_name()
            if new_source_name:
                if new_source_name == source_name:
                    QMessageBox.warning(self, "Error", "New source name must be different from the original.")
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
            source_type="ref",
            var_name=var_name,  # Pass variable name for context
            initial_data=existing_data,
            ssh_manager=ssh_manager,
            parent=self,
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
            self, "Confirm", f"Remove source '{source_name}'?", QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if var_name in self._source_configs:
                self._source_configs[var_name].pop(source_name, None)
            self._update_source_list(var_name)
            self.save_to_config()

    def _update_source_list(self, var_name: str):
        """Update the source list widget for a variable."""
        source_list = self._source_lists.get(var_name)
        if not source_list:
            return

        source_list.clear()
        sources = self._source_configs.get(var_name, {})
        for source_name in sources.keys():
            source_list.addItem(source_name)

    def load_from_config(self):
        """Load from config.

        Properly loads per-variable configurations from source files.
        Each variable gets its own copy of the config with variable-specific settings.
        """
        import os
        import yaml

        # Clear existing source configs before reloading
        self._source_configs.clear()

        self._rebuild_variable_groups()

        ref_data = self.controller.config.get("ref_data", {})
        general_section = ref_data.get("general", {})
        def_nml = ref_data.get("def_nml", {})
        # saved_source_configs now uses compound key: "var_name::source_name"
        saved_source_configs = ref_data.get("source_configs", {})

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None

        # Parse existing config into source configs
        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]

        for var_name in selected:
            key = f"{var_name}_ref_source"
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
                        # If not connected, don't fall back to local - just skip loading
                        # The paths in the config are already remote paths
                    else:
                        # Local mode - load from local file
                        full_path = self._resolve_def_nml_path(def_nml_path)
                        if full_path and os.path.exists(full_path):
                            try:
                                with open(full_path, "r", encoding="utf-8") as f:
                                    nml_content = yaml.safe_load(f) or {}
                            except Exception as e:
                                print(f"Warning: Failed to load def_nml file {full_path}: {e}")

                    if nml_content:
                        # Load general section
                        if "general" in nml_content:
                            source_data["general"] = nml_content["general"].copy()

                        # Load variable-specific settings (all fields from var section)
                        if var_name in nml_content:
                            var_config = nml_content[var_name]
                            # Store all var-specific fields at top level for DataSourceEditor
                            for field, value in var_config.items():
                                source_data[field] = value

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
            stdout, stderr, exit_code = ssh_manager.execute(f"cat '{def_nml_path}'", timeout=30)
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
        if len(path) >= 2 and path[1] == ":":
            # This is a Windows local path - extract relative part if contains /nml/
            if "/nml/" in path:
                relative_path = "nml/" + path.split("/nml/", 1)[1]
                return f"{remote_openbench_path.rstrip('/')}/{relative_path}"
            # Unknown Windows path - cannot convert to remote
            return path

        # Extract relative path from various formats
        relative_path = path

        # If path contains /nml/, extract from that point (handles local temp paths)
        if "/nml/" in path:
            relative_path = "nml/" + path.split("/nml/", 1)[1]
        elif path.startswith("./"):
            relative_path = path[2:]
        elif path.startswith("/"):
            # Check if it's already a valid remote path
            if any(path.startswith(prefix) for prefix in ["/home/", "/share/", "/data/", "/work/", "/scratch/"]):
                return path
            # Extract relative portion if contains /nml/
            if "/nml/" in path:
                relative_path = "nml/" + path.split("/nml/", 1)[1]
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
                key = f"{var_name}_ref_source"
                general[key] = list(sources.keys())

                for source_name, source_data in sources.items():
                    # Get def_nml_path if it exists, otherwise generate one
                    def_nml_path = source_data.get("def_nml_path", "")
                    if not def_nml_path:
                        # Will be generated during namelist sync
                        basedir = self.controller.config.get("general", {}).get("basedir", "./output")
                        def_nml_path = f"{basedir}/nml/ref/{source_name}.yaml"
                    def_nml[source_name] = def_nml_path

                    # Store with compound key to preserve per-variable configs
                    compound_key = f"{var_name}::{source_name}"
                    source_configs[compound_key] = source_data.copy()
                    # Also store var_name in the config for later retrieval
                    source_configs[compound_key]["_var_name"] = var_name

        ref_data = {
            "general": general,
            "def_nml": def_nml,
            "source_configs": source_configs,  # Include full configs for sync
        }
        self.controller.update_section("ref_data", ref_data)

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
                    message=f"{var_name.replace('_', ' ')} is missing reference data source configuration",
                    page_id=self.PAGE_ID,
                    context={"var_name": var_name},
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
                varname = source_data.get("varname") or var_config.get("varname") or general.get("varname") or ""
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
                        QMessageBox.No,
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
                            context={"var_name": var_name, "source_name": source_name},
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
                        context={"var_name": var_name, "source_name": source_name},
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
        from openbench.gui.widgets.validation_dialog import ValidationProgressDialog, ValidationResultsDialog

        # Check if any sources configured
        if not self._source_configs:
            QMessageBox.information(self, "No Data", "No data sources configured. Please add a data source first.")
            return

        # Get general config
        general_config = self.controller.config.get("general", {})

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None

        if is_remote and not ssh_manager:
            QMessageBox.warning(self, "Not Connected", "Remote mode requires connecting to the server first.")
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
            conda_env=conda_env,
        )

        # Show progress dialog
        progress_dialog = ValidationProgressDialog(validator, self._source_configs, general_config, parent=self)

        if progress_dialog.exec() == QDialog.Accepted:
            report = progress_dialog.get_report()
            if report:
                # Show results dialog
                results_dialog = ValidationResultsDialog(report, parent=self)
                results_dialog.exec()
