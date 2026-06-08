# -*- coding: utf-8 -*-
"""
Reference Data configuration page.

Each evaluation variable has exactly ONE reference source selected via a
registry combo box.  A collapsible "Advanced" section shows auto-filled
fields (varname, varunit, sub_dir, prefix, suffix) that the user can
override.

Internal data structure:
    _source_configs[var_name][source_name] = {
        "general": {...},           # Shared settings (root_dir, data_type, etc.)
        "varname": ..., "varunit": ..., "prefix": ..., "suffix": ..., "sub_dir": ...
    }
"""

import logging
import shlex
from typing import Dict, Any

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QLineEdit,
    QWidget,
    QMessageBox,
    QDialog,
    QComboBox,
    QToolButton,
    QProgressDialog,
)
from PySide6.QtCore import Qt

from openbench.gui.pages.base_page import BasePage
from openbench.gui.path_utils import to_absolute_path

logger = logging.getLogger(__name__)

_DETACHED_SCAN_WORKERS = []


from openbench.gui.path_utils import get_remote_ssh_manager


class PageRefData(BasePage):
    """Reference Data configuration page."""

    PAGE_ID = "ref_data"
    PAGE_TITLE = "Reference Data"
    PAGE_SUBTITLE = "Configure reference data sources for each evaluation variable"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        # === Data Root + Scan Controls ===
        scan_group = QGroupBox("Reference Data Root")
        scan_layout = QHBoxLayout(scan_group)

        self.data_root_input = QLineEdit()
        self.data_root_input.setPlaceholderText("Reference data root directory (e.g., /Volumes/work/Reference)")
        # Try to pre-fill from environment or common paths
        import os

        default_root = os.environ.get("OPENBENCH_DATA_ROOT", "")
        if not default_root:
            for p in ["/Volumes/work/Reference", os.path.expanduser("~/data/Reference")]:
                if os.path.isdir(p):
                    default_root = p
                    break
        self.data_root_input.setText(default_root)
        scan_layout.addWidget(self.data_root_input, stretch=1)

        self.btn_browse_root = QPushButton("Browse")
        self.btn_browse_root.clicked.connect(self._browse_data_root)
        scan_layout.addWidget(self.btn_browse_root)

        self.btn_scan = QPushButton("Scan for Datasets")
        self.btn_scan.setToolTip("Scan the data root directory for reference datasets and register new ones")
        self.btn_scan.clicked.connect(self._scan_data_root)
        scan_layout.addWidget(self.btn_scan)

        self.content_layout.addWidget(scan_group)

        # === Registry Info ===
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        ref_count = len(mgr.list_references())
        self.registry_label = QLabel(f"Registry: {ref_count} datasets available")
        self.content_layout.addWidget(self.registry_label)

        # Container for variable groups
        self.var_container = QWidget()
        self.var_layout = QVBoxLayout(self.var_container)
        self.var_layout.setContentsMargins(0, 0, 0, 0)
        self.var_layout.setSpacing(15)

        self.content_layout.addWidget(self.var_container)

        # Store references to per-variable combo boxes and advanced fields
        self._var_combos: Dict[str, QComboBox] = {}
        self._var_advanced_fields: Dict[str, Dict[str, QLineEdit]] = {}
        # Structure: _source_configs[var_name][source_name] = {"general": {...}, ...}
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
        """Rebuild variable groups based on selected evaluation items.

        Each variable gets a single combo box to select the registry dataset
        and a collapsible "Advanced" section with auto-filled fields.
        """
        # Clear existing
        while self.var_layout.count():
            child = self.var_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._var_combos.clear()
        self._var_advanced_fields.clear()

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

            # --- Dataset combo row ---
            combo_layout = QHBoxLayout()
            combo_label = QLabel("Dataset:")
            combo_layout.addWidget(combo_label)

            combo = QComboBox()
            combo.setMinimumWidth(300)
            combo.setProperty("var_name", var_name)
            self._populate_registry_combo(combo, var_name)
            combo.currentIndexChanged.connect(lambda _idx, v=var_name, c=combo: self._on_dataset_selected(v, c))
            combo_layout.addWidget(combo, stretch=1)
            self._var_combos[var_name] = combo

            gear_btn = QPushButton("⚙")
            gear_btn.setFixedWidth(30)
            gear_btn.setToolTip("Manage datasets in Data Registry")
            gear_btn.clicked.connect(lambda: self.controller.go_to_page("registry"))
            combo_layout.addWidget(gear_btn)

            group_layout.addLayout(combo_layout)

            # --- Collapsible Advanced section ---
            toggle_btn = QToolButton()
            toggle_btn.setText("Advanced")
            toggle_btn.setCheckable(True)
            toggle_btn.setChecked(False)
            toggle_btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
            toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            toggle_btn.setArrowType(Qt.RightArrow)

            advanced_widget = QWidget()
            advanced_form = QFormLayout(advanced_widget)
            advanced_form.setContentsMargins(20, 4, 0, 4)

            fields = {}
            for field_name in ("varname", "varunit", "sub_dir", "prefix", "suffix"):
                le = QLineEdit()
                le.setPlaceholderText("(auto-filled from registry)")
                le.setProperty("var_name", var_name)
                le.setProperty("field_name", field_name)
                le.editingFinished.connect(lambda v=var_name: self._on_advanced_field_edited(v))
                advanced_form.addRow(field_name + ":", le)
                fields[field_name] = le

            self._var_advanced_fields[var_name] = fields
            advanced_widget.setVisible(False)

            def _toggle_advanced(checked, btn=toggle_btn, widget=advanced_widget):
                widget.setVisible(checked)
                btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

            toggle_btn.toggled.connect(_toggle_advanced)

            group_layout.addWidget(toggle_btn)
            group_layout.addWidget(advanced_widget)

            self.var_layout.addWidget(group)

        self.var_layout.addStretch()

    def _browse_data_root(self):
        """Browse for reference data root directory."""
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Select Reference Data Root")
        if path:
            self.data_root_input.setText(path)

    def _scan_data_root(self):
        """Scan the data root for new reference datasets."""
        data_root = self.data_root_input.text().strip()
        if not data_root:
            QMessageBox.warning(self, "No Path", "Please enter or browse for the reference data root directory.")
            return

        # Refuse to silently scan a local path while the controller is wired
        # to a remote storage backend — the worker only knows the local FS,
        # so a remote path would either fail os.path.isdir or match an
        # unrelated local directory of the same name.
        try:
            from openbench.remote.storage import RemoteStorage

            if isinstance(getattr(self.controller, "storage", None), RemoteStorage):
                QMessageBox.warning(
                    self,
                    "Remote mode",
                    "Reference scanning currently only supports local paths.\n"
                    f"Path {data_root!r} would be checked on this machine, not the remote host.",
                )
                return
        except Exception:
            # Storage backend not importable in non-remote builds — proceed.
            pass

        import os

        if not os.path.isdir(data_root):
            QMessageBox.warning(self, "Invalid Path", f"Directory not found: {data_root}")
            return

        self.btn_scan.setEnabled(False)
        progress = QProgressDialog("Scanning reference datasets...", None, 0, 0, self)
        progress.setWindowTitle("Scanning")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)

        from openbench.gui.pages._scan_worker import FindDatasetsWorker

        worker = FindDatasetsWorker(data_root)
        self._scan_worker = worker
        self._scan_progress = progress
        worker.finished_with_result.connect(self._on_scan_data_root_finished)
        worker.failed.connect(self._on_scan_data_root_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress.show()

    def _detach_scan_worker(self, worker):
        """Keep an unparented QThread alive until Qt emits finished."""
        if worker is None:
            return
        _DETACHED_SCAN_WORKERS.append(worker)

        def _forget_worker():
            try:
                _DETACHED_SCAN_WORKERS.remove(worker)
            except ValueError:
                pass

        worker.finished.connect(_forget_worker)

    def _finish_scan_worker(self, cancel: bool = False):
        progress = getattr(self, "_scan_progress", None)
        if progress is not None:
            progress.close()
            progress.deleteLater()
        worker = getattr(self, "_scan_worker", None)
        if worker is not None:
            if cancel:
                for signal, slot in (
                    (worker.finished_with_result, self._on_scan_data_root_finished),
                    (worker.failed, self._on_scan_data_root_failed),
                ):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
                if worker.isRunning():
                    worker.requestInterruption()
                    worker.quit()
                    worker.wait(3000)
            if worker.isRunning():
                self._detach_scan_worker(worker)
        self._scan_progress = None
        self._scan_worker = None
        self.btn_scan.setEnabled(True)

    def closeEvent(self, event):
        self._finish_scan_worker(cancel=True)
        super().closeEvent(event)

    def _on_scan_data_root_failed(self, message: str):
        self._finish_scan_worker()
        QMessageBox.critical(self, "Scan Failed", f"Error scanning: {message}")
        logger.error("Data scan failed: %s", message)

    def _on_scan_data_root_finished(self, new_groups):
        self._finish_scan_worker()
        try:
            from openbench.data.registry.scanner import register_scanned_datasets_batch
            from openbench.gui.dialogs.data_discovery import DataDiscoveryDialog, choose_nc_variable

            if not new_groups:
                QMessageBox.information(
                    self, "Scan Complete", "No new datasets found. All datasets already registered."
                )
                return

            dlg = DataDiscoveryDialog(new_groups, parent=self)
            if dlg.exec():
                selected = dlg.get_selected()
                if not selected:
                    return

                variants = [variant for _base, _res, variant in selected]
                register_scanned_datasets_batch(
                    variants,
                    on_multi_var=lambda var_name, sub_dir, all_vars: choose_nc_variable(
                        self, var_name, sub_dir, all_vars
                    ),
                )
                registered = len(variants)

                # Refresh registry
                from openbench.data.registry.manager import clear_registry_cache, get_registry

                clear_registry_cache()
                mgr2 = get_registry()
                self.registry_label.setText(f"Registry: {len(mgr2.list_references())} datasets available")

                # Rebuild variable groups to pick up new registry entries
                self._rebuild_variable_groups()
                self.load_from_config()

                QMessageBox.information(
                    self,
                    "Scan Complete",
                    f"Registered {registered} new dataset(s).\nThey are now available in the dropdown menus below.",
                )

        except Exception as e:
            QMessageBox.critical(self, "Scan Failed", f"Error scanning: {e}")
            logger.exception("Data scan registration failed")

    def _populate_registry_combo(self, combo, var_name):
        """Populate registry combo with available reference datasets for this variable.

        Datasets with multiple resolutions are grouped: selecting one opens
        a resolution picker dialog.
        """
        combo.clear()
        combo.addItem("-- Select from Registry --", None)

        try:
            from openbench.data.registry.manager import get_registry

            mgr = get_registry()
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

    def _on_dataset_selected(self, var_name: str, combo: QComboBox):
        """Handle dataset selection from the combo box.

        For multi-resolution groups, opens the resolution picker.
        Looks up the dataset + variable from the registry, fills the
        advanced fields, and stores the result in ``_source_configs``.
        """
        combo_data = combo.currentData()
        if not combo_data:
            # Placeholder selected — clear source config for this variable
            self._source_configs.pop(var_name, None)
            self._clear_advanced_fields(var_name)
            self.save_to_config()
            return

        # Handle multi-resolution group: open resolution picker
        if isinstance(combo_data, dict) and "group" in combo_data:
            source_name = self._pick_resolution(combo_data, var_name)
            if not source_name:
                # User cancelled — revert combo to placeholder
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                return
            # After picker, select the resolved item in the combo if present,
            # otherwise just proceed with the source_name we got.
        else:
            source_name = combo_data

        try:
            from openbench.data.registry.manager import get_registry

            mgr = get_registry()
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

            # Store as the single source for this variable
            self._source_configs[var_name] = {source_name: source_data}
            self._fill_advanced_fields(var_name, source_data)
            self.save_to_config()

        except ImportError:
            QMessageBox.warning(self, "Error", "Registry module not available.")

    def _fill_advanced_fields(self, var_name: str, source_data: dict):
        """Populate the Advanced fields from *source_data*."""
        fields = self._var_advanced_fields.get(var_name, {})
        for key, line_edit in fields.items():
            value = source_data.get(key, "")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            line_edit.setText(str(value) if value else "")

    def _clear_advanced_fields(self, var_name: str):
        """Reset all Advanced fields for *var_name* to empty."""
        fields = self._var_advanced_fields.get(var_name, {})
        for line_edit in fields.values():
            line_edit.setText("")

    def _on_advanced_field_edited(self, var_name: str):
        """Persist manual overrides from the Advanced fields back to _source_configs."""
        sources = self._source_configs.get(var_name, {})
        if not sources:
            return
        # There is exactly one source per variable now
        source_name = next(iter(sources))
        source_data = sources[source_name]

        fields = self._var_advanced_fields.get(var_name, {})
        for key, line_edit in fields.items():
            text = line_edit.text().strip()
            if text:
                source_data[key] = text
            else:
                source_data.pop(key, None)

        self.save_to_config()

    def _pick_resolution(self, group_data, var_name):
        """Open resolution picker dialog for a multi-resolution dataset.

        Returns selected source_name or None if cancelled.
        """
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
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

    def load_from_config(self):
        """Load from config.

        For each variable, reads the single ``{var}_ref_source`` value, restores
        the combo selection, and fills the advanced fields.
        """
        import os
        import yaml

        # Clear existing source configs before reloading
        self._source_configs.clear()

        self._rebuild_variable_groups()

        ref_data = self.controller.config.get("ref_data", {})
        general_section = ref_data.get("general", {})
        def_nml = ref_data.get("def_nml", {})
        saved_source_configs = ref_data.get("source_configs", {})

        # Restore the data root text box from the persisted general
        # section so reloading a project doesn't lose the user's path.
        saved_data_root = general_section.get("data_root", "")
        if saved_data_root:
            self.data_root_input.setText(saved_data_root)

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None

        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]

        for var_name in selected:
            key = f"{var_name}_ref_source"
            raw = general_section.get(key, "")
            # Accept both string and single-element list for backward compat
            if isinstance(raw, list):
                source_name = raw[0] if raw else ""
            else:
                source_name = raw or ""

            if not source_name:
                continue

            # Try to restore from saved source_configs (compound key)
            compound_key = f"{var_name}::{source_name}"
            source_data = None

            if compound_key in saved_source_configs:
                saved = saved_source_configs[compound_key]
                general = saved.get("general", {})
                if general.get("root_dir") or general.get("dir"):
                    source_data = saved.copy()

            # Fall back to def_nml file
            if source_data is None:
                def_nml_path = def_nml.get(source_name, "")
                source_data = {"def_nml_path": def_nml_path}

                if def_nml_path:
                    nml_content = None

                    if is_remote:
                        if ssh_manager and ssh_manager.is_connected:
                            remote_path = self._resolve_remote_def_nml_path(ssh_manager, def_nml_path)
                            nml_content = self._load_remote_nml_content(ssh_manager, remote_path)
                    else:
                        full_path = self._resolve_def_nml_path(def_nml_path)
                        if full_path and os.path.exists(full_path):
                            try:
                                with open(full_path, "r", encoding="utf-8") as f:
                                    nml_content = yaml.safe_load(f) or {}
                            except Exception as e:
                                logger.warning("Failed to load def_nml file %s: %s", full_path, e)

                    if nml_content:
                        if "general" in nml_content:
                            source_data["general"] = nml_content["general"].copy()
                        if var_name in nml_content:
                            for field, value in nml_content[var_name].items():
                                source_data[field] = value

            self._source_configs[var_name] = {source_name: source_data}

            # Set the combo to the matching item (block signals to avoid re-trigger)
            combo = self._var_combos.get(var_name)
            if combo:
                combo.blockSignals(True)
                matched = False
                for i in range(combo.count()):
                    item_data = combo.itemData(i)
                    if item_data == source_name:
                        combo.setCurrentIndex(i)
                        matched = True
                        break
                if not matched:
                    # Source not in combo — add it as a custom entry
                    combo.addItem(source_name, source_name)
                    combo.setCurrentIndex(combo.count() - 1)
                combo.blockSignals(False)

            # Fill advanced fields
            self._fill_advanced_fields(var_name, source_data)

        # Persist loaded state back
        if self._source_configs:
            self.save_to_config()

    def _load_remote_nml_content(self, ssh_manager, def_nml_path: str) -> dict:
        """Load NML content from remote server."""
        import yaml

        if not def_nml_path:
            return None

        try:
            stdout, stderr, exit_code = ssh_manager.execute(f"cat {shlex.quote(def_nml_path)}", timeout=30)
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

        Each variable stores exactly one source name as a string in
        ``ref_data["general"]["{var}_ref_source"]``.
        """
        existing_ref_data = self.controller.config.get("ref_data", {})
        preserved = {
            key: value
            for key, value in existing_ref_data.items()
            if key not in {"general", "def_nml", "source_configs"}
        }
        existing_general = existing_ref_data.get("general", {})
        general = {
            key: value
            for key, value in existing_general.items()
            if key != "data_root" and not key.endswith("_ref_source")
        }
        def_nml = {}
        source_configs = {}

        for var_name, sources in self._source_configs.items():
            if sources:
                # Single source per variable — store the name as a plain string
                source_name = next(iter(sources))
                general[f"{var_name}_ref_source"] = source_name

                source_data = sources[source_name]
                def_nml_path = source_data.get("def_nml_path", "")
                if not def_nml_path:
                    basedir = self.controller.config.get("general", {}).get("basedir", "./output")
                    def_nml_path = f"{basedir}/nml/ref/{source_name}.yaml"
                def_nml[source_name] = def_nml_path

                compound_key = f"{var_name}::{source_name}"
                source_configs[compound_key] = source_data.copy()
                source_configs[compound_key]["_var_name"] = var_name

        # Persist the reference data root so it survives a reload. The
        # text box is the user's primary anchor for browsing/scanning;
        # losing it on save makes the page look blank on reopen.
        general["data_root"] = self.data_root_input.text().strip()

        ref_data = {
            **preserved,
            "general": general,
            "def_nml": def_nml,
            "source_configs": source_configs,
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
                    return False

            # Validate the single source
            for source_name, source_data in sources.items():
                general = source_data.get("general", {})
                var_config = source_data.get("var_config", {})
                varname = source_data.get("varname") or var_config.get("varname") or general.get("varname") or ""
                if not varname:
                    reply = QMessageBox.warning(
                        self,
                        "Variable Name Missing",
                        f"Variable name is not set for:\n\n"
                        f"Data source: {source_name}\n"
                        f"Variable: {var_name.replace('_', ' ')}\n\n"
                        f"Is this variable defined in the filter configuration?\n\n"
                        f"Click 'Yes' to continue, 'No' to go back and fix it.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if reply == QMessageBox.No:
                        return False

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
                            return False

                root_dir = general.get("root_dir", "") or general.get("dir", "")
                if not root_dir:
                    error = ValidationError(
                        field_name="root_dir",
                        message=f"Root directory is required\n\nData source: {source_name}\nVariable: {var_name.replace('_', ' ')}",
                        page_id=self.PAGE_ID,
                        context={"var_name": var_name, "source_name": source_name},
                    )
                    if not manager.show_error_and_focus(error):
                        return False

        self.save_to_config()
        return True

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
