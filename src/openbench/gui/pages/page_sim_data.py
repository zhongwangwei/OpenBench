# -*- coding: utf-8 -*-
"""
Simulation Data configuration page — scan-based workflow.

Users point at a simulation root directory, click Scan, and the page
discovers available case subdirectories.  Each case gets a checkbox
(to include/exclude) and its own model dropdown.  Shared settings
(data_type, grid_res, tim_res, etc.) are at the bottom.

The union of selected models' variable profiles determines which
variables are available for evaluation downstream.
"""

import logging
import os
import re
import shlex
from typing import Any, Dict, List, Set

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal

from openbench.gui.pages.base_page import BasePage

logger = logging.getLogger(__name__)


from openbench.gui.path_utils import get_remote_ssh_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_prefix(case_dir: str) -> str:
    for sub in (os.path.join(case_dir, "history"), case_dir):
        nc_files = [str(path) for path in _glob_nc_local(sub)]
        if nc_files:
            basename = os.path.basename(nc_files[0])
            match = re.match(r"^(.*?)(\d{4})", basename)
            return match.group(1) if match else ""
    return ""


def _find_nc_dir(case_dir: str) -> str:
    hist = os.path.join(case_dir, "history")
    if os.path.isdir(hist) and _glob_nc_local(hist):
        return hist
    if _glob_nc_local(case_dir):
        return case_dir
    return ""


def _glob_nc_local(directory: str):
    from openbench.data.coordinates import glob_nc

    return glob_nc(directory)


def _remote_is_dir(ssh_manager, path: str) -> bool:
    stdout, _, exit_code = ssh_manager.execute(
        f"test -d {shlex.quote(path)} && echo dir",
        timeout=10,
    )
    return exit_code == 0 and "dir" in stdout


def _remote_first_nc_file(ssh_manager, directory: str) -> str:
    quoted = shlex.quote(directory)
    cmd = (
        f"find {quoted} -maxdepth 1 -type f "
        r"\( -name '*.nc' -o -name '*.nc4' \) | sort | head -n 1"
    )
    stdout, _, exit_code = ssh_manager.execute(cmd, timeout=30)
    if exit_code != 0:
        return ""
    return stdout.strip().splitlines()[0] if stdout.strip() else ""


def _remote_find_nc_dir(ssh_manager, case_dir: str) -> str:
    hist = f"{case_dir.rstrip('/')}/history"
    if _remote_is_dir(ssh_manager, hist) and _remote_first_nc_file(ssh_manager, hist):
        return hist
    if _remote_first_nc_file(ssh_manager, case_dir):
        return case_dir
    return ""


def _remote_detect_prefix(ssh_manager, case_dir: str) -> str:
    for sub in (f"{case_dir.rstrip('/')}/history", case_dir):
        first_nc = _remote_first_nc_file(ssh_manager, sub)
        if first_nc:
            basename = os.path.basename(first_nc)
            match = re.match(r"^(.*?)(\d{4})", basename)
            return match.group(1) if match else ""
    return ""


def _remote_list_child_dirs(ssh_manager, root: str) -> list[str]:
    quoted = shlex.quote(root)
    stdout, _, exit_code = ssh_manager.execute(
        f"find {quoted} -mindepth 1 -maxdepth 1 -type d -print | sort",
        timeout=30,
    )
    if exit_code != 0:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _get_model_names() -> List[str]:
    """Return sorted list of registered model names."""
    try:
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        return sorted([m.name for m in mgr.list_models()])
    except Exception:
        return []


def _read_nc_varnames(nc_dir: str) -> List[str]:
    """Read variable names from the first NC file in a directory."""
    nc_files = [str(path) for path in _glob_nc_local(nc_dir)]
    if not nc_files:
        return []
    try:
        import xarray as xr

        with xr.open_dataset(nc_files[0]) as ds:
            return list(ds.data_vars)
    except Exception:
        return []


def _match_model(nc_vars: List[str]) -> List[tuple]:
    """Match NC variable names against registered model profiles.

    Returns list of (model_name, match_count, total_profile_vars, match_ratio)
    sorted by match_ratio descending. Only includes models with ratio > 0.
    """
    if not nc_vars:
        return []
    nc_set = set(nc_vars)
    results = []
    try:
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        for mp in mgr.list_models():
            if not mp.variables:
                continue
            # Collect all varnames (primary + fallbacks) for this model
            model_varnames = set()
            for vm in mp.variables.values():
                if isinstance(vm.varname, list):
                    model_varnames.update(vm.varname)
                elif vm.varname:
                    model_varnames.add(vm.varname)
                if vm.fallbacks:
                    for fb in vm.fallbacks:
                        model_varnames.add(fb.varname)
            overlap = nc_set & model_varnames
            if overlap:
                ratio = len(overlap) / len(model_varnames) if model_varnames else 0
                results.append((mp.name, len(overlap), len(model_varnames), ratio))
    except Exception:
        pass
    results.sort(key=lambda x: x[3], reverse=True)
    return results


def _get_model_variables(model_name: str) -> List[str]:
    """Return variable names supported by a model profile."""
    try:
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        mp = mgr.get_model(model_name)
        if mp and hasattr(mp, "variables"):
            return sorted(mp.variables.keys())
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class PageSimData(BasePage):
    """Simulation Data configuration page."""

    PAGE_ID = "sim_data"
    PAGE_TITLE = "Simulation Data"
    PAGE_SUBTITLE = "Scan a directory for simulation cases, assign models, and select cases to evaluate"
    CONTENT_EXPAND = True

    # Emitted when case selection or model assignment changes.
    # Carries the union of variable names from all selected models.
    available_variables_changed = Signal(list)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_content(self):
        # === Scan section ===
        scan_group = QGroupBox("Scan for Cases")
        scan_form = QFormLayout(scan_group)

        root_row = QHBoxLayout()
        self._root_input = QLineEdit()
        self._root_input.setPlaceholderText("Simulation root directory (e.g. /data/Simulation)")
        root_row.addWidget(self._root_input, 1)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._browse_root)
        root_row.addWidget(self._browse_btn)
        scan_form.addRow("Root directory:", root_row)

        btn_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.setToolTip("List subdirectories that contain NetCDF simulation output")
        self._scan_btn.clicked.connect(self._do_scan)
        btn_row.addWidget(self._scan_btn)
        btn_row.addStretch()
        scan_form.addRow("", btn_row)

        self.content_layout.addWidget(scan_group)

        # === Case list (scrollable) ===
        self._case_scroll = QScrollArea()
        self._case_scroll.setWidgetResizable(True)
        self._case_widget = QWidget()
        self._case_layout = QVBoxLayout(self._case_widget)
        self._case_layout.setContentsMargins(4, 4, 4, 4)
        self._case_layout.setSpacing(4)
        self._case_scroll.setWidget(self._case_widget)
        self.content_layout.addWidget(self._case_scroll, 1)

        # Per-case data: list of dicts with keys:
        #   checkbox, model_combo, label, nc_dir, auto_prefix
        self._cases: List[Dict[str, Any]] = []

        # Cached model names
        self._model_names: List[str] = _get_model_names()

        # === Shared settings ===
        self._settings_group = QGroupBox("Shared Case Settings")
        settings_form = QFormLayout(self._settings_group)

        self._data_type_combo = QComboBox()
        self._data_type_combo.addItems(["grid", "stn"])
        settings_form.addRow("data_type:", self._data_type_combo)

        self._grid_res_input = QLineEdit()
        self._grid_res_input.setPlaceholderText("e.g. 0.5")
        settings_form.addRow("grid_res:", self._grid_res_input)

        self._tim_res_combo = QComboBox()
        self._tim_res_combo.addItems(["Month", "Day", "Hour", "Year"])
        settings_form.addRow("tim_res:", self._tim_res_combo)

        self._data_groupby_combo = QComboBox()
        self._data_groupby_combo.addItems(["month", "Year", "day", "single"])
        settings_form.addRow("data_groupby:", self._data_groupby_combo)

        self._prefix_input = QLineEdit()
        self._prefix_input.setPlaceholderText("Per-case auto-detected (override here for all)")
        settings_form.addRow("prefix override:", self._prefix_input)

        self._suffix_input = QLineEdit()
        settings_form.addRow("suffix:", self._suffix_input)

        self._settings_group.setVisible(False)
        self.content_layout.addWidget(self._settings_group)

        # === Validate button ===
        validate_layout = QHBoxLayout()
        validate_layout.addStretch()
        self.validate_btn = QPushButton("Validate Data")
        self.validate_btn.setToolTip("Check that simulation files exist")
        self.validate_btn.clicked.connect(self._validate_data)
        validate_layout.addWidget(self.validate_btn)
        self.content_layout.addLayout(validate_layout)

        # Legacy compat
        self._source_configs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Browse & Scan
    # ------------------------------------------------------------------

    def _browse_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select Simulation Root Directory")
        if path:
            self._root_input.setText(path)

    def _do_scan(self):
        root = self._root_input.text().strip()

        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None
        if is_remote:
            if not root or not ssh_manager or not ssh_manager.is_connected or not _remote_is_dir(ssh_manager, root):
                QMessageBox.warning(self, "Invalid Path", "Please enter a valid remote simulation root directory.")
                return
        elif not root or not os.path.isdir(root):
            QMessageBox.warning(self, "Invalid Path", "Please enter a valid simulation root directory.")
            return

        self._clear_cases()

        # Scanning runs synchronously on the GUI thread (full QThread
        # refactor is a separate item); show a wait cursor and pump the
        # event loop between subdirectories so the window stays painted
        # and the cursor visible during multi-thousand-entry scans
        # rather than appearing frozen.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            discovered = []
            if is_remote:
                for full in _remote_list_child_dirs(ssh_manager, root):
                    label = os.path.basename(full.rstrip("/"))
                    nc_dir = _remote_find_nc_dir(ssh_manager, full)
                    if nc_dir:
                        prefix = _remote_detect_prefix(ssh_manager, full)
                        discovered.append((label, nc_dir, prefix))
                    QApplication.processEvents()
            else:
                try:
                    entries = sorted(os.listdir(root))
                except OSError as exc:
                    QMessageBox.critical(self, "Error", f"Cannot list directory:\n{exc}")
                    return

                for entry in entries:
                    full = os.path.join(root, entry)
                    if not os.path.isdir(full):
                        continue
                    nc_dir = _find_nc_dir(full)
                    if nc_dir:
                        prefix = _detect_prefix(full)
                        discovered.append((entry, nc_dir, prefix))
                    QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()

        if not discovered:
            QMessageBox.information(self, "No Cases Found", f"No subdirectories with NetCDF files under:\n{root}")
            return

        # Auto-detect model from the first case's NC file
        first_nc_dir = discovered[0][1]
        nc_vars = [] if is_remote else _read_nc_varnames(first_nc_dir)
        auto_model = ""
        match_info = ""

        if nc_vars:
            matches = _match_model(nc_vars)
            if matches and matches[0][3] >= 0.3:
                auto_model = matches[0][0]
            match_info = (
                "\n".join(f"{name}: {count}/{total} vars ({ratio:.0%})" for name, count, total, ratio in matches[:5])
                if matches
                else "No matching model profiles found."
            )

        # Refresh model names
        self._model_names = _get_model_names()

        # Show confirmation dialog
        from openbench.gui.dialogs.scan_confirm import ScanConfirmDialog

        dlg = ScanConfirmDialog(
            discovered=discovered,
            model_names=self._model_names,
            auto_model=auto_model,
            match_info=match_info,
            nc_var_count=len(nc_vars),
            parent=self,
        )
        # Wire "Register New Model" button to navigate to registry
        dlg.register_button.clicked.connect(lambda: (dlg.reject(), self.controller.go_to_page("registry")))

        if not dlg.exec():
            return

        confirmed = dlg.get_results()
        if not confirmed:
            return

        # Build per-case rows from confirmed results
        for case in confirmed:
            self._add_case_row(
                case["label"],
                case["nc_dir"],
                case["prefix"],
                checked=True,
                model_name=case["model"],
            )

        self._settings_group.setVisible(True)
        self._on_selection_changed()

    def _add_case_row(self, label: str, nc_dir: str, prefix: str, checked: bool = True, model_name: str = ""):
        """Add one case row: [checkbox] label  path  [model combo]"""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)

        cb = QCheckBox(label)
        cb.setChecked(checked)
        cb.toggled.connect(self._on_selection_changed)
        row_layout.addWidget(cb)

        path_label = QLabel(nc_dir)
        path_label.setStyleSheet("color: #888; font-size: 11px;")
        path_label.setToolTip(nc_dir)
        row_layout.addWidget(path_label, 1)

        prefix_label = QLabel(f"prefix: {prefix}")
        prefix_label.setStyleSheet("color: #aaa; font-size: 10px;")
        row_layout.addWidget(prefix_label)

        model_combo = QComboBox()
        model_combo.setMinimumWidth(150)
        for mn in self._model_names:
            model_combo.addItem(mn, mn)
        if model_name:
            idx = model_combo.findData(model_name)
            if idx >= 0:
                model_combo.setCurrentIndex(idx)
        model_combo.currentIndexChanged.connect(self._on_selection_changed)
        row_layout.addWidget(model_combo)

        gear_btn = QPushButton("⚙")
        gear_btn.setFixedWidth(30)
        gear_btn.setToolTip("Manage models in Data Registry")
        gear_btn.clicked.connect(lambda: self.controller.go_to_page("registry"))
        row_layout.addWidget(gear_btn)

        self._case_layout.addWidget(row)
        self._cases.append(
            {
                "checkbox": cb,
                "model_combo": model_combo,
                "label": label,
                "nc_dir": nc_dir,
                "auto_prefix": prefix,
                "row_widget": row,
            }
        )

    def _clear_cases(self):
        for case in self._cases:
            case["row_widget"].deleteLater()
        self._cases.clear()

    # ------------------------------------------------------------------
    # Selection changed → derive available variables
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        """Called when any checkbox or model combo changes."""
        self.save_to_config()
        # Emit available variables from selected models
        var_set = self._get_available_variables()
        self.available_variables_changed.emit(sorted(var_set))

    def _get_available_variables(self) -> Set[str]:
        """Union of variables from all selected cases' model profiles."""
        var_set: Set[str] = set()
        for case in self._cases:
            if not case["checkbox"].isChecked():
                continue
            model_name = case["model_combo"].currentData()
            if model_name:
                var_set.update(_get_model_variables(model_name))
        return var_set

    def get_selected_cases(self) -> List[Dict[str, Any]]:
        """Return list of selected case info dicts (for other pages)."""
        result = []
        prefix_override = self._prefix_input.text().strip()
        for case in self._cases:
            if case["checkbox"].isChecked():
                result.append(
                    {
                        "label": case["label"],
                        "nc_dir": case["nc_dir"],
                        "prefix": prefix_override or case["auto_prefix"],
                        "model": case["model_combo"].currentData() or "",
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_to_config(self):
        cases = self.get_selected_cases()
        case_labels = [c["label"] for c in cases]
        existing_sim_data = self.controller.config.get("sim_data", {})
        preserved = {
            key: value
            for key, value in existing_sim_data.items()
            if key not in {"general", "def_nml", "source_configs", "_scan_root", "_shared_settings"}
        }

        prefix_override = self._prefix_input.text().strip()

        existing_source_configs = existing_sim_data.get("source_configs", {}) or {}
        source_configs: Dict[str, Any] = {}
        for c in cases:
            existing_source = dict(existing_source_configs.get(c["label"], {}) or {})
            source_general = dict(existing_source.get("general", {}) or {})
            source_general.update(
                {
                    "model_namelist": c["model"],
                    "root_dir": c["nc_dir"],
                    "data_type": self._data_type_combo.currentText(),
                    "grid_res": self._grid_res_input.text().strip(),
                    "tim_res": self._tim_res_combo.currentText(),
                    "data_groupby": self._data_groupby_combo.currentText(),
                    "prefix": prefix_override or c["prefix"],
                    "suffix": self._suffix_input.text().strip(),
                }
            )
            existing_source["general"] = source_general
            source_configs[c["label"]] = existing_source

        # For every selected evaluation variable, all selected cases are sources
        eval_items = self.controller.config.get("evaluation_items", {})
        selected_vars = [k for k, v in eval_items.items() if v]
        # If no eval items selected yet, use the derived variable list
        if not selected_vars:
            selected_vars = sorted(self._get_available_variables())

        # Preserve user-set fields inside sim_data["general"] (e.g. `data_root`
        # or other non-*_sim_source keys) so they survive scan/checkbox saves.
        # Only the *_sim_source mappings are rewritten below.
        existing_inner_general = existing_sim_data.get("general", {}) or {}
        general: Dict[str, Any] = {k: v for k, v in existing_inner_general.items() if not k.endswith("_sim_source")}
        for var_name in selected_vars:
            general[f"{var_name}_sim_source"] = list(case_labels)

        sim_data = {
            **preserved,
            "general": general,
            "def_nml": existing_sim_data.get("def_nml", {}) or {},
            "source_configs": source_configs,
            "_scan_root": self._root_input.text().strip(),
            "_shared_settings": {
                "data_type": self._data_type_combo.currentText(),
                "grid_res": self._grid_res_input.text().strip(),
                "tim_res": self._tim_res_combo.currentText(),
                "data_groupby": self._data_groupby_combo.currentText(),
                "prefix": prefix_override,
                "suffix": self._suffix_input.text().strip(),
            },
        }

        self.controller.update_section("sim_data", sim_data)

    def load_from_config(self):
        sim_data = self.controller.config.get("sim_data", {})
        if not sim_data:
            return

        scan_root = sim_data.get("_scan_root", "")
        if scan_root:
            self._root_input.setText(scan_root)

        # Restore shared settings
        ss = sim_data.get("_shared_settings", {})
        if ss:
            if ss.get("data_type"):
                idx = self._data_type_combo.findText(ss["data_type"])
                if idx >= 0:
                    self._data_type_combo.setCurrentIndex(idx)
            if ss.get("grid_res"):
                self._grid_res_input.setText(str(ss["grid_res"]))
            if ss.get("tim_res"):
                idx = self._tim_res_combo.findText(ss["tim_res"])
                if idx >= 0:
                    self._tim_res_combo.setCurrentIndex(idx)
            if ss.get("data_groupby"):
                idx = self._data_groupby_combo.findText(ss["data_groupby"])
                if idx >= 0:
                    self._data_groupby_combo.setCurrentIndex(idx)
            if ss.get("prefix"):
                self._prefix_input.setText(ss["prefix"])
            if ss.get("suffix"):
                self._suffix_input.setText(ss["suffix"])

        # Restore cases from source_configs
        saved_configs = sim_data.get("source_configs", {})
        if not saved_configs:
            return

        # Determine which labels are selected
        general_section = sim_data.get("general", {})
        selected_labels = set()
        for key, val in general_section.items():
            if key.endswith("_sim_source"):
                if isinstance(val, list):
                    selected_labels.update(val)
                elif isinstance(val, str):
                    selected_labels.add(val)

        self._clear_cases()
        for label, cfg in saved_configs.items():
            gen = cfg.get("general", {})
            nc_dir = gen.get("root_dir", "")
            prefix = gen.get("prefix", "")
            model_name = gen.get("model_namelist", "")
            self._add_case_row(label, nc_dir, prefix, checked=(label in selected_labels), model_name=model_name)

        if saved_configs:
            self._settings_group.setVisible(True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        cases = self.get_selected_cases()
        if not cases:
            QMessageBox.warning(self, "No Cases", "Please scan and select at least one simulation case.")
            return False
        for c in cases:
            if not c["model"]:
                QMessageBox.warning(self, "No Model", f"Please select a model for case '{c['label']}'.")
                return False
        return True

    def _validate_data(self):
        cases = self.get_selected_cases()
        if not cases:
            QMessageBox.information(self, "Nothing to Validate", "No cases selected.")
            return
        # Quick check: verify NC directories exist
        issues = []
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None
        for c in cases:
            if is_remote:
                ok = ssh_manager and ssh_manager.is_connected and _remote_is_dir(ssh_manager, c["nc_dir"])
            else:
                ok = os.path.isdir(c["nc_dir"])
            if not ok:
                issues.append(f"{c['label']}: directory not found ({c['nc_dir']})")
        if issues:
            QMessageBox.warning(self, "Validation Issues", "\n".join(issues))
        else:
            QMessageBox.information(self, "Validation OK", f"All {len(cases)} case directories verified.")
