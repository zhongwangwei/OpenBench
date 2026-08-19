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
from pathlib import Path
from typing import Any, Dict, List, Set

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal

from openbench.gui.remote_python import quote_remote_path
from openbench.gui.widgets._ssh_worker import call_responsive, execute_responsive
from openbench.gui.pages.base_page import BasePage

logger = logging.getLogger(__name__)

SIM_TIM_RES_OPTIONS = [
    "Month",
    "Day",
    "Hour",
    "Year",
    "3Hour",
    "6Hour",
    "8Day",
    "climatology-month",
    "climatology-year",
]


from openbench.gui.path_utils import browse_directory, get_remote_ssh_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case_file_patterns(file_names: List[str]) -> tuple:
    """Return (prefix, suffix, multi_stream) for a case's NC file names.

    Uses the CLI scanner's date-token split so GUI and CLI derive identical
    patterns. A single file stream yields its (prefix, suffix) directly;
    multiple distinct streams (e.g. one file per variable like
    ``YEE2_JRA-55_alb_Mon_*.nc`` / ``YEE2_JRA-55_lai_Mon_*.nc``) yield the
    longest common stem prefix and are flagged ``multi_stream`` so
    per-variable overrides drive file lookup instead of one variable's
    pattern being applied to every variable.
    """
    from openbench.data.sim_scanner import _filename_pattern_for_file

    names = [os.path.basename(str(name)) for name in file_names if name]
    if not names:
        return "", "", False
    patterns = {_filename_pattern_for_file(Path(name)) for name in names}
    if len(patterns) == 1:
        prefix, suffix = next(iter(patterns))
        return prefix, suffix, False
    stems = [Path(name).stem for name in names]
    return os.path.commonprefix(stems), "", True


def _detect_case_pattern(case_dir: str) -> tuple:
    """Local case scan: (prefix, suffix, multi_stream) from all NC files."""
    for sub in (os.path.join(case_dir, "history"), case_dir):
        nc_files = [str(path) for path in _glob_nc_local(sub)]
        if nc_files:
            return _case_file_patterns(nc_files)
    return "", "", False


def _detect_prefix(case_dir: str) -> str:
    return _detect_case_pattern(case_dir)[0]


def _case_prefix_is_safe(prefix: str, suffix: str, overrides: Dict[str, Any]) -> bool:
    """Mirror cli/sim._case_prefix_is_safe_to_write for GUI-exported cases.

    When per-variable overrides reveal multiple file streams, a case-level
    prefix would silently apply one stream's pattern to every unmapped
    variable, so it must be dropped from the exported config.
    """
    if not overrides:
        return True
    seen_prefixes = {prefix or ""}
    seen_suffixes = {suffix or ""}
    for override in overrides.values():
        if not isinstance(override, dict):
            continue
        seen_prefixes.add(override.get("prefix", prefix) or "")
        seen_suffixes.add(override.get("suffix", suffix) or "")
    return len(seen_prefixes) <= 1 and len(seen_suffixes) <= 1


def _local_variable_overrides(nc_dir: str, model_name: str) -> Dict[str, Any]:
    """Per-variable file-pattern overrides via the CLI scanner (reads NC files)."""
    try:
        from openbench.data.registry.scanner import inspect_nc_file
        from openbench.data.sim_scanner import _infer_file_grouping, _infer_variable_file_overrides

        path = Path(nc_dir)
        info = inspect_nc_file(path)
        data_groupby, _years = _infer_file_grouping(path)
        return _infer_variable_file_overrides(
            path,
            model=model_name,
            default_grid_res=info.get("detected_grid_res"),
            default_tim_res=info.get("detected_tim_res"),
            default_data_type=info.get("detected_data_type"),
            default_data_groupby=data_groupby,
        )
    except Exception as exc:
        logger.warning("Could not infer per-variable overrides for %s: %s", nc_dir, exc)
        return {}


def _filename_variable_overrides(file_names: List[str], model_name: str) -> Dict[str, Any]:
    """Per-variable overrides from filenames only (no file IO; remote-safe)."""
    names = [os.path.basename(str(name)) for name in file_names if name]
    if not model_name or len(names) < 2:
        return {}
    from openbench.data.sim_scanner import _filename_pattern_for_file, _match_profile_variable_file

    if len({_filename_pattern_for_file(Path(name)) for name in names}) < 2:
        return {}
    try:
        from openbench.data.registry.manager import get_registry

        profile = get_registry().get_model(model_name)
    except Exception:
        profile = None
    if not profile:
        return {}
    paths = [Path(name) for name in names]
    overrides: Dict[str, Any] = {}
    for variable_name, mapping in profile.variables.items():
        matched = _match_profile_variable_file(paths, mapping)
        if matched is None:
            continue
        file_path, _candidate = matched
        prefix, suffix = _filename_pattern_for_file(file_path)
        override: Dict[str, Any] = {"prefix": prefix}
        if suffix:
            override["suffix"] = suffix
        overrides[variable_name] = override
    return overrides


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
    stdout, _, exit_code = execute_responsive(
        ssh_manager,
        f"test -d {quote_remote_path(path)} && echo dir",
        timeout=10,
    )
    return exit_code == 0 and "dir" in stdout


def _remote_list_nc_files(ssh_manager, directory: str) -> list[str]:
    quoted = quote_remote_path(directory)
    cmd = (
        f"find {quoted} -maxdepth 1 -type f "
        r"\( -name '*.nc' -o -name '*.nc4' \) | sort"
    )
    stdout, _, exit_code = execute_responsive(ssh_manager, cmd, timeout=30)
    if exit_code != 0:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _remote_first_nc_file(ssh_manager, directory: str) -> str:
    files = _remote_list_nc_files(ssh_manager, directory)
    return files[0] if files else ""


def _remote_find_nc_dir(ssh_manager, case_dir: str) -> str:
    hist = f"{case_dir.rstrip('/')}/history"
    if _remote_is_dir(ssh_manager, hist) and _remote_first_nc_file(ssh_manager, hist):
        return hist
    if _remote_first_nc_file(ssh_manager, case_dir):
        return case_dir
    return ""


def _remote_detect_case_pattern(ssh_manager, case_dir: str) -> tuple:
    """Remote case scan: (prefix, suffix, multi_stream, file_names)."""
    for sub in (f"{case_dir.rstrip('/')}/history", case_dir):
        files = _remote_list_nc_files(ssh_manager, sub)
        if files:
            names = [os.path.basename(name) for name in files]
            prefix, suffix, multi_stream = _case_file_patterns(names)
            return prefix, suffix, multi_stream, names
    return "", "", False, []


def _remote_detect_prefix(ssh_manager, case_dir: str) -> str:
    return _remote_detect_case_pattern(ssh_manager, case_dir)[0]


def _remote_list_dirs(ssh_manager, root: str, max_depth: int = 5) -> list[str]:
    quoted = quote_remote_path(root)
    stdout, _, exit_code = execute_responsive(
        ssh_manager,
        f"find {quoted} -mindepth 0 -maxdepth {int(max_depth)} -type d -print | sort",
        timeout=30,
    )
    if exit_code != 0:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _model_from_case_label(label: str, model_names: List[str]) -> str:
    """Match an exact case label to a registered model without guessing."""
    try:
        from openbench.data.registry.manager import canonical_model_key

        label_key = canonical_model_key(label)
        for name in model_names:
            if canonical_model_key(name) == label_key:
                return name
    except Exception:
        pass
    return ""


def _scan_local_cases(root: str) -> tuple[List[tuple], Dict[str, Dict[str, Any]]]:
    """Use the shared CLI scanner so GUI discovery follows the same rules."""
    from openbench.data.sim_scanner import scan_simulation_roots

    result = scan_simulation_roots([root], model_name="auto")
    discovered: List[tuple] = []
    case_meta: Dict[str, Dict[str, Any]] = {}
    for scanned in result.cases:
        nc_dir = str(scanned.root_dir)
        files = [os.path.basename(str(path)) for path in _glob_nc_local(nc_dir)]
        detected_prefix, detected_suffix, multi_stream = _case_file_patterns(files)
        prefix = detected_prefix if multi_stream else (scanned.prefix or detected_prefix)
        suffix = detected_suffix if multi_stream else (scanned.suffix or detected_suffix)
        overrides = dict(scanned.variable_overrides or {})
        multi_stream = multi_stream or not _case_prefix_is_safe(prefix, suffix, overrides)
        model = "" if scanned.model == "UNRESOLVED" else scanned.model
        discovered.append((scanned.label, nc_dir, prefix))
        case_meta[scanned.label] = {
            "files": files,
            "suffix": suffix,
            "multi_stream": multi_stream,
            "model": model,
            "variables": list(scanned.variables or []),
            "variable_overrides": overrides,
            "data_type": scanned.data_type,
            "grid_res": scanned.grid_res,
            "tim_res": scanned.tim_res,
            "data_groupby": scanned.data_groupby,
            "fulllist": str(scanned.fulllist) if scanned.fulllist else "",
            "station_layout": scanned.station_layout,
            "source_root": str(scanned.source_root) if scanned.source_root else "",
        }
    return discovered, case_meta


def _apply_variable_pattern_edit(overrides: Dict[str, Any], variable_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one variable-pattern edit while preserving other inferred fields."""
    current = overrides.get(variable_name, {}) if isinstance(overrides.get(variable_name, {}), dict) else {}
    edited_name = data.get("variable_name", "") or variable_name
    updated = dict(current)
    for key in ("varname", "varunit", "prefix", "suffix"):
        value = data.get(key, "")
        if value:
            updated[key] = value
        else:
            updated.pop(key, None)
    result = dict(overrides)
    result.pop(variable_name, None)
    if updated:
        result[edited_name] = updated
    return result


def _get_model_names() -> List[str]:
    """Return sorted list of registered model names."""
    try:
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        return sorted([m.name for m in mgr.list_models()])
    except Exception:
        return []


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


def _combo_value(combo) -> str:
    """Return a combo's data value, with text-only test/legacy fallback."""
    if combo is None:
        return ""
    current_data = getattr(combo, "currentData", None)
    if callable(current_data):
        value = current_data()
        if value is not None:
            return str(value).strip()
    current_text = getattr(combo, "currentText", None)
    return str(current_text()).strip() if callable(current_text) else ""


def _grid_res_value(value: Any) -> Any:
    """Keep blank/unresolved values, but export valid UI input numerically."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


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
        self._settings_group = QGroupBox("Optional Overrides for Selected Cases")
        settings_form = QFormLayout(self._settings_group)

        self._data_type_combo = QComboBox()
        self._data_type_combo.addItem("Auto (per case)", "")
        self._data_type_combo.addItem("grid", "grid")
        self._data_type_combo.addItem("stn", "stn")
        settings_form.addRow("data_type override:", self._data_type_combo)

        self._grid_res_input = QLineEdit()
        self._grid_res_input.setPlaceholderText("Auto per case (e.g. 0.5)")
        settings_form.addRow("grid_res override:", self._grid_res_input)

        self._tim_res_combo = QComboBox()
        self._tim_res_combo.addItem("Auto (per case)", "")
        for value in SIM_TIM_RES_OPTIONS:
            self._tim_res_combo.addItem(value, value)
        settings_form.addRow("tim_res override:", self._tim_res_combo)

        self._data_groupby_combo = QComboBox()
        self._data_groupby_combo.addItem("Auto (per case)", "")
        for value in ("month", "Year", "day", "single", "Single"):
            self._data_groupby_combo.addItem(value, value)
        settings_form.addRow("data_groupby override:", self._data_groupby_combo)

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
        path = browse_directory(
            self.controller, self, "Select Simulation Root Directory", self._root_input.text().strip()
        )
        if path:
            self._root_input.setText(path)

    def _do_scan(self):
        """Scan the simulation root (re-entrancy-guarded entry point).

        The remote branch keeps the event loop alive via execute_responsive,
        so a second click would re-enter mid-scan without the guard.
        """
        if getattr(self, "_scan_in_progress", False):
            return
        self._scan_in_progress = True
        self._scan_btn.setEnabled(False)
        try:
            self._do_scan_flow()
        finally:
            self._scan_in_progress = False
            self._scan_btn.setEnabled(True)

    def _do_scan_flow(self):
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

        # Remote SSH calls go through execute_responsive (worker thread +
        # live event loop), so the window stays painted without manual
        # event pumping; the wait cursor signals the ongoing scan.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            discovered = []
            # label → {files, suffix, multi_stream}; carried past the confirm
            # dialog so per-variable overrides can be derived for the chosen model.
            case_meta: Dict[str, Dict[str, Any]] = {}
            if is_remote:
                seen_nc_dirs = set()
                for full in _remote_list_dirs(ssh_manager, root):
                    nc_dir = _remote_find_nc_dir(ssh_manager, full)
                    normalized_nc_dir = nc_dir.rstrip("/")
                    if not nc_dir or normalized_nc_dir in seen_nc_dirs:
                        continue
                    seen_nc_dirs.add(normalized_nc_dir)
                    nc_leaf = normalized_nc_dir.rsplit("/", 1)[-1].lower()
                    label_dir = normalized_nc_dir.rsplit("/", 1)[0] if nc_leaf == "history" else full
                    label = label_dir.rstrip("/").rsplit("/", 1)[-1]
                    prefix, suffix, multi_stream, file_names = _remote_detect_case_pattern(ssh_manager, full)
                    discovered.append((label, nc_dir, prefix))
                    case_meta[label] = {
                        "files": file_names,
                        "suffix": suffix,
                        "multi_stream": multi_stream,
                        "data_type": None,
                        "grid_res": None,
                        "tim_res": None,
                        "data_groupby": None,
                        "fulllist": "",
                        "station_layout": None,
                        "source_root": root,
                    }
            else:
                try:
                    discovered, case_meta = call_responsive(lambda: _scan_local_cases(root))
                except Exception as exc:
                    QMessageBox.critical(self, "Error", f"Cannot scan directory:\n{exc}")
                    return
        finally:
            QApplication.restoreOverrideCursor()

        if not discovered:
            QMessageBox.information(self, "No Cases Found", f"No NetCDF simulation cases found under:\n{root}")
            return

        # Refresh model names
        self._model_names = _get_model_names()
        case_models = {}
        for label, _nc_dir, _prefix in discovered:
            meta = case_meta.get(label, {})
            model = meta.get("model", "")
            if not model and is_remote:
                model = _model_from_case_label(label, self._model_names)
            case_models[label] = model
        match_info = "\n".join(
            f"{label}: {case_models.get(label) or 'model unresolved'}" for label, _nc_dir, _prefix in discovered
        )
        nc_var_count = len({variable for meta in case_meta.values() for variable in meta.get("variables", [])})

        # Show confirmation dialog
        from openbench.gui.dialogs.scan_confirm import ScanConfirmDialog

        dlg = ScanConfirmDialog(
            discovered=discovered,
            model_names=self._model_names,
            auto_model="",
            match_info=match_info,
            nc_var_count=nc_var_count,
            case_models=case_models,
            parent=self,
        )
        # Wire "Register New Model" button to navigate to registry
        dlg.register_button.clicked.connect(lambda: (dlg.reject(), self.controller.go_to_page("registry")))

        if not dlg.exec():
            return

        confirmed = dlg.get_results()

        # Build per-case rows from confirmed results
        for case in confirmed:
            meta = case_meta.get(case["label"], {})
            if case["model"] == meta.get("model"):
                overrides = meta.get("variable_overrides") or {}
            else:
                overrides = self._compute_variable_overrides(
                    case["nc_dir"],
                    meta.get("files") or [],
                    case["model"],
                    is_remote,
                    meta.get("multi_stream", False),
                )
            self._add_case_row(
                case["label"],
                case["nc_dir"],
                case["prefix"],
                checked=case.get("checked", True),
                model_name=case["model"],
                suffix=meta.get("suffix", ""),
                files=meta.get("files") or [],
                variable_overrides=overrides,
                multi_stream=meta.get("multi_stream", False),
                scan_metadata=meta,
            )

        self._settings_group.setVisible(True)
        self._on_selection_changed()

    def _compute_variable_overrides(
        self, nc_dir: str, file_names: List[str], model_name: str, is_remote: bool, multi_stream: bool
    ) -> Dict[str, Any]:
        """Derive per-variable prefix/suffix overrides for a multi-stream case.

        Local cases use the CLI scanner (reads NC files, can also override
        varname/grid_res); remote cases fall back to filename-only matching.
        """
        if not multi_stream or not model_name:
            return {}
        if not is_remote and nc_dir and os.path.isdir(nc_dir):
            overrides = _local_variable_overrides(nc_dir, model_name)
            if overrides:
                return overrides
        return _filename_variable_overrides(file_names, model_name)

    def _add_case_row(
        self,
        label: str,
        nc_dir: str,
        prefix: str,
        checked: bool = True,
        model_name: str = "",
        suffix: str = "",
        files: List[str] = None,
        variable_overrides: Dict[str, Any] = None,
        multi_stream: bool = False,
        scan_metadata: Dict[str, Any] = None,
    ):
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

        metadata = dict(scan_metadata or {})
        detected = [
            str(value)
            for value in (metadata.get("data_type"), metadata.get("tim_res"), metadata.get("grid_res"))
            if value not in (None, "")
        ]
        metadata_label = QLabel(" / ".join(detected) if detected else "metadata unresolved")
        metadata_label.setStyleSheet("color: #777; font-size: 10px;")
        metadata_label.setToolTip("Detected data_type / tim_res / grid_res")
        row_layout.addWidget(metadata_label)

        prefix_input = QLineEdit(prefix)
        prefix_input.setPlaceholderText("prefix")
        prefix_input.setMaximumWidth(240)
        if multi_stream:
            prefix_input.setToolTip(
                "One file per variable detected — per-variable file patterns "
                "are exported instead of a single case prefix."
            )
        row_layout.addWidget(QLabel("prefix:"))
        row_layout.addWidget(prefix_input)

        suffix_input = QLineEdit(suffix)
        suffix_input.setPlaceholderText("suffix")
        suffix_input.setMaximumWidth(120)
        row_layout.addWidget(QLabel("suffix:"))
        row_layout.addWidget(suffix_input)

        model_combo = QComboBox()
        model_combo.setMinimumWidth(150)
        model_combo.addItem("Select model...", "")
        for mn in self._model_names:
            model_combo.addItem(mn, mn)
        if model_name:
            idx = model_combo.findData(model_name)
            if idx >= 0:
                model_combo.setCurrentIndex(idx)
        row_layout.addWidget(model_combo)

        patterns_btn = QPushButton("Variables...")
        patterns_btn.setToolTip("Edit per-variable varname, unit, prefix, and suffix")
        row_layout.addWidget(patterns_btn)

        gear_btn = QPushButton("⚙")
        gear_btn.setFixedWidth(30)
        gear_btn.setToolTip("Manage models in Data Registry")
        gear_btn.clicked.connect(lambda: self.controller.go_to_page("registry"))
        row_layout.addWidget(gear_btn)

        self._case_layout.addWidget(row)
        case = {
            "checkbox": cb,
            "model_combo": model_combo,
            "label": label,
            "nc_dir": nc_dir,
            "auto_prefix": prefix,
            "auto_suffix": suffix,
            "prefix_input": prefix_input,
            "suffix_input": suffix_input,
            "case_pattern_edited": False,
            "files": list(files or []),
            "variable_overrides": dict(variable_overrides or {}),
            "multi_stream": multi_stream,
            "scan_metadata": metadata,
            "row_widget": row,
        }
        prefix_input.textEdited.connect(lambda _text, c=case: self._on_case_pattern_changed(c))
        suffix_input.textEdited.connect(lambda _text, c=case: self._on_case_pattern_changed(c))
        model_combo.currentIndexChanged.connect(lambda _index, c=case: self._on_case_model_changed(c))
        patterns_btn.clicked.connect(lambda _checked=False, c=case: self._edit_variable_pattern(c))
        self._cases.append(case)

    def _on_case_pattern_changed(self, case: Dict[str, Any]):
        case["case_pattern_edited"] = True
        self._on_selection_changed()

    def _edit_variable_pattern(self, case: Dict[str, Any]):
        variables = sorted(
            set(_get_model_variables(case["model_combo"].currentData() or ""))
            | set((case.get("variable_overrides") or {}).keys())
        )
        if not variables:
            QMessageBox.information(self, "No Variables", "Select a model with registered variables first.")
            return
        variable_name, accepted = QInputDialog.getItem(self, "Edit Variable Pattern", "Variable:", variables, 0, False)
        if not accepted or not variable_name:
            return

        from openbench.gui.widgets.variable_editor import VariableEditorDialog

        overrides = case.get("variable_overrides") or {}
        current = overrides.get(variable_name, {}) if isinstance(overrides.get(variable_name, {}), dict) else {}
        dlg = VariableEditorDialog(
            mode="simulation",
            variable_name=variable_name,
            varname=current.get("varname", ""),
            varunit=current.get("varunit", ""),
            prefix=current.get("prefix", ""),
            suffix=current.get("suffix", ""),
            parent=self,
        )
        if not dlg.exec():
            return
        case["variable_overrides"] = _apply_variable_pattern_edit(overrides, variable_name, dlg.get_data())
        self._on_selection_changed()

    def _on_case_model_changed(self, case: Dict[str, Any]):
        """Recompute per-variable overrides when a case's model changes.

        Only possible when the scan captured the case's file list; rows
        restored from a saved config keep their stored overrides.
        """
        if case.get("files"):
            from openbench.remote.storage import RemoteStorage

            is_remote = isinstance(self.controller.storage, RemoteStorage)
            case["variable_overrides"] = self._compute_variable_overrides(
                case["nc_dir"],
                case["files"],
                case["model_combo"].currentData() or "",
                is_remote,
                case.get("multi_stream", False),
            )
        self._on_selection_changed()

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
        suffix_override = self._suffix_input.text().strip()
        data_type_override = _combo_value(getattr(self, "_data_type_combo", None))
        tim_res_override = _combo_value(getattr(self, "_tim_res_combo", None))
        data_groupby_override = _combo_value(getattr(self, "_data_groupby_combo", None))
        grid_res_override = getattr(self, "_grid_res_input", None)
        grid_res_override = grid_res_override.text().strip() if grid_res_override is not None else ""
        for case in self._cases:
            if not case["checkbox"].isChecked():
                continue
            overrides = case.get("variable_overrides") or {}
            metadata = case.get("scan_metadata") or {}
            case_prefix = case.get("prefix_input")
            case_suffix = case.get("suffix_input")
            case_prefix = case_prefix.text().strip() if case_prefix is not None else case["auto_prefix"]
            case_suffix = case_suffix.text().strip() if case_suffix is not None else case.get("auto_suffix", "")
            prefix = prefix_override or case_prefix
            suffix = suffix_override or case_suffix
            if (
                not prefix_override
                and not case.get("case_pattern_edited", False)
                and not _case_prefix_is_safe(case_prefix, suffix, overrides)
            ):
                # Multi-stream case (one file per variable): a case-level
                # prefix would force one stream's files onto every variable,
                # so only the per-variable overrides are exported.
                prefix = ""
                if not suffix_override:
                    suffix = ""
            data_type = data_type_override or metadata.get("data_type") or ""
            if data_type == "stn":
                prefix = ""
                suffix = ""
            result.append(
                {
                    "label": case["label"],
                    "nc_dir": case["nc_dir"],
                    "prefix": prefix,
                    "suffix": suffix,
                    "model": case["model_combo"].currentData() or "",
                    "variables": overrides,
                    "data_type": data_type,
                    "grid_res": _grid_res_value(grid_res_override if grid_res_override else metadata.get("grid_res")),
                    "tim_res": tim_res_override or metadata.get("tim_res") or "",
                    "data_groupby": data_groupby_override or metadata.get("data_groupby") or "",
                    "fulllist": (metadata.get("fulllist") or "") if data_type == "stn" else "",
                    "station_layout": metadata.get("station_layout") or "",
                    "source_root": metadata.get("source_root") or "",
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
            if key
            not in {
                "general",
                "def_nml",
                "source_configs",
                "_scan_root",
                "_scanned_cases",
                "_shared_settings",
            }
        }

        scanned_cases = existing_sim_data.get("_scanned_cases", [])
        if hasattr(self, "_cases"):
            scanned_cases = [
                {
                    "label": case["label"],
                    "nc_dir": case["nc_dir"],
                    "prefix": case["prefix_input"].text().strip(),
                    "suffix": case["suffix_input"].text().strip(),
                    "checked": case["checkbox"].isChecked(),
                    "model": case["model_combo"].currentData() or "",
                    "files": list(case.get("files") or []),
                    "variables": dict(case.get("variable_overrides") or {}),
                    "multi_stream": bool(case.get("multi_stream", False)),
                    "case_pattern_edited": bool(case.get("case_pattern_edited", False)),
                    "metadata": dict(case.get("scan_metadata") or {}),
                }
                for case in self._cases
            ]

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
                    "data_type": c.get("data_type") or _combo_value(self._data_type_combo),
                    "grid_res": (
                        c.get("grid_res") if c.get("grid_res") is not None else self._grid_res_input.text().strip()
                    ),
                    "tim_res": c.get("tim_res") or _combo_value(self._tim_res_combo),
                    "data_groupby": c.get("data_groupby") or _combo_value(self._data_groupby_combo),
                    "prefix": prefix_override or c["prefix"],
                    "suffix": c.get("suffix", self._suffix_input.text().strip()),
                }
            )
            if "fulllist" in c:
                if c["fulllist"]:
                    source_general["fulllist"] = c["fulllist"]
                else:
                    source_general.pop("fulllist", None)
            existing_source["general"] = source_general
            # Per-variable file-pattern overrides from the scan (one file per
            # variable layouts). Preserve overrides loaded from a config when
            # this row was restored without a fresh scan.
            if "variables" in c:
                if c["variables"]:
                    existing_source["variables"] = c["variables"]
                else:
                    existing_source.pop("variables", None)
            source_configs[c["label"]] = existing_source

        # For every selected evaluation variable, all selected cases are sources.
        # Do not fall back to all model-profile variables: Evaluation Variables
        # is the user's source of truth, and an empty selection must stay empty.
        eval_items = self.controller.config.get("evaluation_items", {})
        selected_vars = [k for k, v in eval_items.items() if v]

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
            "_scanned_cases": scanned_cases,
            "_shared_settings": {
                "data_type": _combo_value(self._data_type_combo),
                "grid_res": self._grid_res_input.text().strip(),
                "tim_res": _combo_value(self._tim_res_combo),
                "data_groupby": _combo_value(self._data_groupby_combo),
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
        if not scan_root:
            # Configs written by the CLI carry only per-case root_dir entries;
            # recover the scan root from their common parent so the user does
            # not have to re-pick the directory after loading a YAML.
            from openbench.gui.path_utils import infer_common_scan_root

            scan_root = infer_common_scan_root(
                [
                    (cfg.get("general", {}) or {}).get("root_dir", "")
                    for cfg in (sim_data.get("source_configs", {}) or {}).values()
                ]
            )
        if scan_root:
            self._root_input.setText(scan_root)

        saved_configs = sim_data.get("source_configs", {})
        scanned_cases = sim_data.get("_scanned_cases", [])

        # Restore only explicit shared overrides. CLI/unified configs carry
        # per-case metadata and must not be homogenized from the first case.
        ss = sim_data.get("_shared_settings", {})
        if ss:
            idx = self._data_type_combo.findData(str(ss.get("data_type", "")))
            self._data_type_combo.setCurrentIndex(max(idx, 0))
            if ss.get("grid_res"):
                self._grid_res_input.setText(str(ss["grid_res"]))
            idx = self._tim_res_combo.findData(str(ss.get("tim_res", "")))
            self._tim_res_combo.setCurrentIndex(max(idx, 0))
            idx = self._data_groupby_combo.findData(str(ss.get("data_groupby", "")))
            self._data_groupby_combo.setCurrentIndex(max(idx, 0))
            if ss.get("prefix"):
                self._prefix_input.setText(ss["prefix"])
            if ss.get("suffix"):
                self._suffix_input.setText(ss["suffix"])

        # Restore cases from source_configs
        if not saved_configs and not scanned_cases:
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
        if isinstance(scanned_cases, list) and scanned_cases:
            for saved_case in scanned_cases:
                if not isinstance(saved_case, dict) or not saved_case.get("label"):
                    continue
                label = saved_case["label"]
                cfg = saved_configs.get(label, {}) or {}
                gen = cfg.get("general", {}) or {}
                metadata = dict(saved_case.get("metadata") or {})
                for key in ("data_type", "grid_res", "tim_res", "data_groupby", "fulllist"):
                    if metadata.get(key) in (None, "") and gen.get(key) not in (None, ""):
                        metadata[key] = gen[key]
                overrides = saved_case.get("variables")
                if not isinstance(overrides, dict):
                    overrides = cfg.get("variables") if isinstance(cfg.get("variables"), dict) else {}
                self._add_case_row(
                    label,
                    saved_case.get("nc_dir", gen.get("root_dir", "")),
                    saved_case.get("prefix", gen.get("prefix", "")),
                    checked=bool(saved_case.get("checked", label in selected_labels)),
                    model_name=saved_case.get("model", gen.get("model_namelist", "")),
                    suffix=saved_case.get("suffix", gen.get("suffix", "")),
                    files=saved_case.get("files") or [],
                    variable_overrides=overrides,
                    multi_stream=bool(saved_case.get("multi_stream", False)),
                    scan_metadata=metadata,
                )
                self._cases[-1]["case_pattern_edited"] = bool(saved_case.get("case_pattern_edited", False))
            self._settings_group.setVisible(True)
            return

        for label, cfg in saved_configs.items():
            gen = cfg.get("general", {})
            nc_dir = gen.get("root_dir", "")
            prefix = gen.get("prefix", "")
            model_name = gen.get("model_namelist", "")
            overrides = cfg.get("variables") if isinstance(cfg.get("variables"), dict) else {}
            metadata = {
                key: gen.get(key)
                for key in ("data_type", "grid_res", "tim_res", "data_groupby", "fulllist")
                if gen.get(key) not in (None, "")
            }
            self._add_case_row(
                label,
                nc_dir,
                prefix,
                checked=(label in selected_labels),
                model_name=model_name,
                suffix=gen.get("suffix", ""),
                variable_overrides=overrides,
                multi_stream=not _case_prefix_is_safe(prefix, gen.get("suffix", ""), overrides),
                scan_metadata=metadata,
            )

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
            grid_res = c.get("grid_res")
            if grid_res is not None and not isinstance(grid_res, (int, float)):
                QMessageBox.warning(self, "Invalid Resolution", "grid_res must be a positive number.")
                return False
            if isinstance(grid_res, (int, float)) and grid_res <= 0:
                QMessageBox.warning(self, "Invalid Resolution", "grid_res must be a positive number.")
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
