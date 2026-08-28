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
import threading
from pathlib import Path
from typing import Any, Dict, List, Set

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal

from openbench.gui.localization import get_language_manager, translate_text
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


def _filename_variable_overrides(file_names: List[str], model_name: str, registry=None) -> Dict[str, Any]:
    """Per-variable overrides from filenames only (no file IO; remote-safe)."""
    names = [os.path.basename(str(name)) for name in file_names if name]
    if not model_name or len(names) < 2:
        return {}
    from openbench.data.sim_scanner import _filename_pattern_for_file, _match_profile_variable_file

    if len({_filename_pattern_for_file(Path(name)) for name in names}) < 2:
        return {}
    try:
        if registry is None:
            from openbench.data.registry.manager import get_registry

            registry = get_registry()
        profile = registry.get_model(model_name)
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
        r"\( -name '*.nc' -o -name '*.nc4' -o -name '*.NC' -o -name '*.NC4' \) | sort"
    )
    stdout, stderr, exit_code = execute_responsive(ssh_manager, cmd, timeout=30)
    if exit_code != 0:
        detail = (stderr or stdout or f"exit {exit_code}").strip()
        raise RuntimeError(f"Remote NetCDF scan failed for {directory}: {detail}")
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
        if not _remote_is_dir(ssh_manager, sub):
            continue
        files = _remote_list_nc_files(ssh_manager, sub)
        if files:
            names = [os.path.basename(name) for name in files]
            prefix, suffix, multi_stream = _case_file_patterns(names)
            return prefix, suffix, multi_stream, names
    return "", "", False, []


def _remote_detect_prefix(ssh_manager, case_dir: str) -> str:
    return _remote_detect_case_pattern(ssh_manager, case_dir)[0]


def scan_simulation_cases_remote(
    ssh_manager,
    root: str,
    *,
    python_path: str = "",
    conda_env: str = "",
    openbench_path: str = "",
    timeout: int = 900,
    should_abort=None,
) -> tuple[List[tuple], Dict[str, Dict[str, Any]]]:
    """Run the same simulation scanner on the remote host and rehydrate GUI rows."""
    import json

    from openbench.gui.remote_python import run_remote_python_json

    bootstrap = ""
    if openbench_path:
        remote_root = openbench_path.rstrip("/")
        bootstrap = (
            "import os\n"
            "import sys\n"
            f"for _path in ({json.dumps(remote_root)}, {json.dumps(remote_root + '/src')}):\n"
            "    _path = os.path.expanduser(_path)\n"
            "    if _path not in sys.path:\n"
            "        sys.path.insert(0, _path)\n"
        )

    script = f"""{bootstrap}
import dataclasses
import json
from hashlib import blake2s
from pathlib import Path

from openbench.data.coordinates import glob_nc
try:
    from openbench.data.sim_scanner import scan_simulation_roots
except ImportError as exc:
    raise RuntimeError("remote OpenBench checkout is missing simulation scanner: %s" % exc) from exc
try:
    from openbench.data.sim_scanner import materialize_station_cases
except ImportError as exc:
    materialize_station_cases = None
    station_materialize_error = "remote OpenBench checkout is missing station materializer: %s" % exc
else:
    station_materialize_error = ""

root = {json.dumps(root)}
result = scan_simulation_roots([root], model_name="auto")
if any(case.station_layout for case in result.cases):
    if materialize_station_cases is None:
        pass
    else:
        try:
            digest = blake2s(root.encode("utf-8"), digest_size=6).hexdigest()
            materialize_station_cases(result, Path.home() / ".openbench" / "sim_station_lists" / digest, num_workers=1)
        except Exception as exc:
            station_materialize_error = "station materialization failed: %s: %s" % (type(exc).__name__, exc)


def _case_files(case):
    for directory in (case.root_dir / "history", case.root_dir):
        files = glob_nc(directory)
        if files:
            return [path.name for path in files]
    return []


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


payload = []
for case in result.cases:
    data = dataclasses.asdict(case)
    data["files"] = _case_files(case)
    if case.station_layout and not case.fulllist and station_materialize_error:
        data["station_materialize_error"] = station_materialize_error
    payload.append(data)
print(json.dumps({{"cases": payload}}, default=_json_default))
"""
    payload = run_remote_python_json(
        ssh_manager,
        script,
        python_path=python_path,
        conda_env=conda_env,
        timeout=timeout,
        should_abort=should_abort,
    )
    return _rehydrate_simulation_cases(payload)


def _model_from_payload(item: Dict[str, Any]) -> str:
    model = "" if item.get("model") == "UNRESOLVED" else str(item.get("model") or "")
    return model


def _station_materialize_error(item: Dict[str, Any]) -> str:
    explicit = str(item.get("station_materialize_error") or "")
    if explicit:
        return explicit
    dropped = [str(name) for name in (item.get("station_dropped_sites") or []) if name]
    unresolved = {str(name) for name in (item.get("unresolved") or [])}
    if "station_partial" not in unresolved and not dropped:
        return ""
    if not dropped:
        return "station materialization was partial"
    shown = ", ".join(dropped[:5])
    if len(dropped) > 5:
        shown += f", ... (+{len(dropped) - 5} more)"
    return f"station materialization dropped {len(dropped)} site(s): {shown}"


def _rehydrate_simulation_cases(payload) -> tuple[List[tuple], Dict[str, Dict[str, Any]]]:
    raw_cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    discovered: List[tuple] = []
    case_meta: Dict[str, Dict[str, Any]] = {}
    for item in raw_cases or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "")
        nc_dir = str(item.get("root_dir") or "")
        if not label or not nc_dir:
            continue
        files = [os.path.basename(str(path)) for path in (item.get("files") or [])]
        prefix = str(item.get("prefix") or "")
        suffix = str(item.get("suffix") or "")
        overrides = item.get("variable_overrides") if isinstance(item.get("variable_overrides"), dict) else {}
        _, _, pattern_multi_stream = _case_file_patterns(files)
        multi_stream = pattern_multi_stream or not _case_prefix_is_safe(prefix, suffix, overrides)
        model = _model_from_payload(item)
        discovered.append((label, nc_dir, prefix))
        case_meta[label] = {
            "files": files,
            "suffix": suffix,
            "multi_stream": multi_stream,
            "model": model,
            "variables": list(item.get("variables") or []),
            "variable_overrides": overrides,
            "data_type": item.get("data_type"),
            "grid_res": item.get("grid_res"),
            "tim_res": item.get("tim_res"),
            "data_groupby": item.get("data_groupby"),
            "fulllist": str(item.get("fulllist") or ""),
            "station_layout": item.get("station_layout"),
            "station_dropped_sites": [str(name) for name in (item.get("station_dropped_sites") or [])],
            "unresolved": [str(name) for name in (item.get("unresolved") or [])],
            "station_materialize_error": _station_materialize_error(item),
            "source_root": str(item.get("source_root") or ""),
        }
    return discovered, case_meta


def _remote_list_dirs(ssh_manager, root: str, max_depth: int = 5) -> list[str]:
    quoted = quote_remote_path(root)
    stdout, stderr, exit_code = execute_responsive(
        ssh_manager,
        f"find {quoted} -mindepth 0 -maxdepth {int(max_depth)} -type d -print | sort",
        timeout=30,
    )
    if exit_code != 0:
        detail = (stderr or stdout or f"exit {exit_code}").strip()
        raise RuntimeError(f"Remote directory scan failed for {root}: {detail}")
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


def _get_model_names(registry=None) -> List[str]:
    """Return sorted list of registered model names."""
    try:
        if registry is None:
            from openbench.data.registry.manager import get_registry

            registry = get_registry()
        return sorted([m.name for m in registry.list_models()])
    except Exception:
        return []


def _get_model_variables(model_name: str, registry=None) -> List[str]:
    """Return variable names supported by a model profile."""
    try:
        if registry is None:
            from openbench.data.registry.manager import get_registry

            registry = get_registry()
        mp = registry.get_model(model_name)
        if mp and hasattr(mp, "variables"):
            return sorted(mp.variables.keys())
    except Exception:
        pass
    return []


def _remote_registry_is_offline(controller) -> bool:
    try:
        from openbench.gui.path_utils import get_remote_ssh_manager

        ssh = get_remote_ssh_manager(controller) if hasattr(controller, "storage") else getattr(controller, "ssh_manager", None)
    except Exception:
        # Only a provably disconnected target may use the offline-preservation
        # path.  A lookup failure is a real registry error, not evidence that
        # the target is offline.
        return False
    if ssh is None or not getattr(ssh, "is_connected", False):
        return True
    try:
        get_active_target_identity = getattr(ssh, "get_active_target_identity", None)
        return callable(get_active_target_identity) and get_active_target_identity() is None
    except Exception:
        return False


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

    def _registry(self):
        from openbench.gui.remote_registry import get_registry

        return get_registry(getattr(self, "controller", None))

    def _registry_model_names(self) -> List[str]:
        try:
            is_remote = getattr(getattr(self, "controller", None), "is_remote_mode", None)
            if callable(is_remote) and is_remote():
                return _get_model_names(PageSimData._registry(self))
            return _get_model_names()
        except Exception as exc:
            logger.warning("Could not load registry models: %s", exc)
            return []

    def _registry_model_variables(self, model_name: str) -> List[str]:
        try:
            is_remote = getattr(getattr(self, "controller", None), "is_remote_mode", None)
            if callable(is_remote) and is_remote():
                return _get_model_variables(model_name, PageSimData._registry(self))
            return _get_model_variables(model_name)
        except Exception as exc:
            logger.warning("Could not load registry model %s: %s", model_name, exc)
            return []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_content(self):
        # === Scan section ===
        scan_group = QGroupBox("Scan for Cases")
        scan_form = QFormLayout(scan_group)
        scan_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        scan_form.setLabelAlignment(Qt.AlignLeft)

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
        self._case_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._case_scroll.setMinimumHeight(240)
        self._case_widget = QWidget()
        self._case_layout = QVBoxLayout(self._case_widget)
        self._case_layout.setContentsMargins(4, 4, 4, 4)
        self._case_layout.setSpacing(4)
        self._case_layout.setAlignment(Qt.AlignTop)
        self._case_scroll.setWidget(self._case_widget)
        self.content_layout.addWidget(self._case_scroll, 1)

        # Per-case data: list of dicts with keys:
        #   checkbox, model_combo, label, nc_dir, auto_prefix
        self._cases: List[Dict[str, Any]] = []

        # Cached model names
        self._model_names: List[str] = PageSimData._registry_model_names(self)

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

        progress = None
        cancel_event = None
        if is_remote:
            progress = QProgressDialog("Scanning simulation cases...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Scanning")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            cancel_event = threading.Event()
            canceled = getattr(progress, "canceled", None)
            if canceled is not None and hasattr(canceled, "connect"):
                canceled.connect(cancel_event.set)
            progress.show()

        # Remote SSH calls go through execute_responsive (worker thread +
        # live event loop), so the window stays painted while the progress
        # dialog can cancel the SSH command.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            discovered = []
            # label → {files, suffix, multi_stream}; carried past the confirm
            # dialog so per-variable overrides can be derived for the chosen model.
            case_meta: Dict[str, Dict[str, Any]] = {}
            if is_remote:
                remote_settings = {}
                settings_fn = getattr(self.controller, "remote_settings", None)
                if callable(settings_fn):
                    remote_settings = settings_fn() or {}
                try:
                    discovered, case_meta = scan_simulation_cases_remote(
                        ssh_manager,
                        root,
                        python_path=remote_settings.get("python_path", ""),
                        conda_env=remote_settings.get("conda_env", ""),
                        openbench_path=remote_settings.get("openbench_path", ""),
                        should_abort=cancel_event.is_set if cancel_event is not None else None,
                    )
                except Exception as exc:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    QMessageBox.critical(self, "Error", f"Cannot scan remote simulation data:\n{exc}")
                    return
            else:
                try:
                    discovered, case_meta = call_responsive(lambda: _scan_local_cases(root))
                except Exception as exc:
                    QMessageBox.critical(self, "Error", f"Cannot scan directory:\n{exc}")
                    return
        finally:
            QApplication.restoreOverrideCursor()
            if progress is not None:
                progress.close()
                progress.deleteLater()

        if cancel_event is not None and cancel_event.is_set():
            return

        if not discovered:
            QMessageBox.information(self, "No Cases Found", f"No NetCDF simulation cases found under:\n{root}")
            return

        # Refresh model names
        self._model_names = PageSimData._registry_model_names(self)
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
        try:
            registry = PageSimData._registry(self)
        except Exception as exc:
            logger.warning("Could not load registry model %s: %s", model_name, exc)
            return {}
        return _filename_variable_overrides(file_names, model_name, registry)

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
        """Add a readable case card without squeezing paths and controls into one row."""
        row = QFrame()
        row.setFrameShape(QFrame.StyledPanel)
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(10, 8, 10, 8)
        row_layout.setSpacing(8)

        header_layout = QHBoxLayout()

        cb = QCheckBox(label)
        cb.setChecked(checked)
        cb.toggled.connect(self._on_selection_changed)
        header_layout.addWidget(cb)

        status_label = QLabel()
        header_layout.addWidget(status_label)
        header_layout.addStretch()

        model_button = QPushButton()
        header_layout.addWidget(model_button)

        gear_btn = QPushButton("Model Settings")
        gear_btn.setToolTip("Manage models in Data Registry")
        gear_btn.clicked.connect(lambda: self.controller.go_to_page("registry"))
        header_layout.addWidget(gear_btn)
        row_layout.addLayout(header_layout)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        path_input = QLineEdit(nc_dir)
        path_input.setReadOnly(True)
        path_input.setCursorPosition(0)
        path_input.setToolTip(nc_dir)
        path_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        path_layout.addWidget(path_input, 1)
        row_layout.addLayout(path_layout)

        metadata = dict(scan_metadata or {})
        metadata_layout = QHBoxLayout()
        metadata_layout.addWidget(QLabel("Data Type:"))
        metadata_layout.addWidget(QLabel(str(metadata.get("data_type") or "—")))
        metadata_layout.addSpacing(18)
        metadata_layout.addWidget(QLabel("Time Resolution:"))
        metadata_layout.addWidget(QLabel(str(metadata.get("tim_res") or "—")))
        metadata_layout.addSpacing(18)
        metadata_layout.addWidget(QLabel("Grid Resolution:"))
        grid_res = metadata.get("grid_res")
        metadata_layout.addWidget(QLabel(f"{grid_res:g}°" if isinstance(grid_res, (int, float)) else "—"))
        metadata_layout.addStretch()
        row_layout.addLayout(metadata_layout)

        pattern_toggle = QPushButton("File Matching (Advanced)")
        pattern_toggle.setCheckable(True)
        row_layout.addWidget(pattern_toggle, alignment=Qt.AlignLeft)

        pattern_widget = QWidget()
        pattern_layout = QHBoxLayout(pattern_widget)
        pattern_layout.setContentsMargins(0, 0, 0, 0)

        prefix_input = QLineEdit(prefix)
        prefix_input.setPlaceholderText("prefix")
        if multi_stream:
            prefix_input.setToolTip(
                "One file per variable detected — per-variable file patterns "
                "are exported instead of a single case prefix."
            )
        pattern_layout.addWidget(QLabel("Prefix:"))
        pattern_layout.addWidget(prefix_input, 1)

        suffix_input = QLineEdit(suffix)
        suffix_input.setPlaceholderText("suffix")
        pattern_layout.addWidget(QLabel("Suffix:"))
        pattern_layout.addWidget(suffix_input, 1)

        model_combo = QComboBox()
        model_combo.addItem("Select model...", "")
        for mn in self._model_names:
            model_combo.addItem(mn, mn)
        if model_name:
            idx = model_combo.findData(model_name)
            is_remote = getattr(self.controller, "is_remote_mode", None)
            if idx < 0 and callable(is_remote) and is_remote():
                model_combo.addItem(f"{model_name} (registry unavailable)", model_name)
                idx = model_combo.count() - 1
            if idx >= 0:
                model_combo.setCurrentIndex(idx)
        model_combo.hide()

        patterns_btn = QPushButton("Variables...")
        patterns_btn.setToolTip("Edit per-variable varname, unit, prefix, and suffix")
        pattern_layout.addWidget(patterns_btn)
        pattern_widget.hide()
        pattern_toggle.toggled.connect(pattern_widget.setVisible)
        row_layout.addWidget(pattern_widget)

        self._case_layout.addWidget(row)
        case = {
            "checkbox": cb,
            "model_combo": model_combo,
            "model_button": model_button,
            "status_label": status_label,
            "label": label,
            "nc_dir": nc_dir,
            "path_input": path_input,
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
            "pattern_widget": pattern_widget,
        }
        prefix_input.textEdited.connect(lambda _text, c=case: self._on_case_pattern_changed(c))
        suffix_input.textEdited.connect(lambda _text, c=case: self._on_case_pattern_changed(c))
        model_combo.currentIndexChanged.connect(lambda _index, c=case: self._on_case_model_changed(c))
        model_button.clicked.connect(lambda _checked=False, c=case: self._choose_case_model(c))
        patterns_btn.clicked.connect(lambda _checked=False, c=case: self._edit_variable_pattern(c))
        self._update_case_model_summary(case)
        self._cases.append(case)

    def _choose_case_model(self, case: Dict[str, Any]):
        language = get_language_manager().language
        if not self._model_names:
            QMessageBox.information(
                self,
                translate_text("No Models", language),
                translate_text("Register a model in Data Registry first.", language),
            )
            return
        combo = case["model_combo"]
        current = self._model_names.index(combo.currentData()) if combo.currentData() in self._model_names else 0
        model_name, accepted = QInputDialog.getItem(
            self,
            translate_text("Select Model", language),
            translate_text("Model:", language),
            self._model_names,
            current,
            False,
        )
        if accepted and model_name:
            combo.setCurrentIndex(combo.findData(model_name))

    def _update_case_model_summary(self, case: Dict[str, Any]):
        model_name = case["model_combo"].currentData() or ""
        if not model_name:
            case["status_label"].setText("Model required")
            case["status_label"].setStyleSheet("color: #b45309;")
            case["model_button"].setText("Select Model...")
        else:
            case["status_label"].setText(f"Model: {model_name}")
            case["status_label"].setStyleSheet("color: #15803d;")
            case["model_button"].setText("Change Model...")
        if case["row_widget"].isVisible():
            get_language_manager().apply(case["row_widget"])

    def _on_case_pattern_changed(self, case: Dict[str, Any]):
        case["case_pattern_edited"] = True
        self._on_selection_changed()

    def _edit_variable_pattern(self, case: Dict[str, Any]):
        variables = sorted(
            set(PageSimData._registry_model_variables(self, case["model_combo"].currentData() or ""))
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
        is_remote = getattr(self.controller, "is_remote_mode", None)
        dialog_kwargs = {"known_variables": variables} if callable(is_remote) and is_remote() else {}
        dlg = VariableEditorDialog(
            mode="simulation",
            variable_name=variable_name,
            varname=current.get("varname", ""),
            varunit=current.get("varunit", ""),
            prefix=current.get("prefix", ""),
            suffix=current.get("suffix", ""),
            parent=self,
            **dialog_kwargs,
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
        self._update_case_model_summary(case)
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
                var_set.update(PageSimData._registry_model_variables(self, model_name))
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
                    "station_materialize_error": metadata.get("station_materialize_error") or "",
                    "source_root": metadata.get("source_root") or "",
                }
            )
        return result

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_to_config(self):
        cases = self.get_selected_cases()
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

        # For every selected evaluation variable, only compatible selected cases are sources.
        # Do not fall back to all model-profile variables: Evaluation Variables
        # is the user's source of truth, and an empty selection must stay empty.
        eval_items = self.controller.config.get("evaluation_items", {})
        selected_vars = [k for k, v in eval_items.items() if v]

        # Preserve user-set fields inside sim_data["general"] (e.g. `data_root`
        # or other non-*_sim_source keys) so they survive scan/checkbox saves.
        # Only the *_sim_source mappings are rewritten below.
        existing_inner_general = existing_sim_data.get("general", {}) or {}
        general: Dict[str, Any] = {k: v for k, v in existing_inner_general.items() if not k.endswith("_sim_source")}
        model_vars_cache: Dict[str, Set[str]] = {}
        cases_by_label = {c["label"]: c for c in cases}
        is_remote_fn = getattr(getattr(self, "controller", None), "is_remote_mode", None)
        is_remote = callable(is_remote_fn) and bool(is_remote_fn())
        if not is_remote:
            try:
                from openbench.remote.storage import RemoteStorage

                is_remote = isinstance(self.controller.storage, RemoteStorage)
            except Exception:
                is_remote = False
        remote_registry = None
        remote_registry_unavailable = False
        if is_remote:
            try:
                remote_registry = PageSimData._registry(self)
            except Exception as exc:
                if _remote_registry_is_offline(self.controller):
                    remote_registry_unavailable = True
                else:
                    QMessageBox.critical(self, "Remote Registry Error", f"Failed to load remote registry:\n{exc}")
                    return
        for var_name in selected_vars:
            sources = []
            var_key = str(var_name).casefold()
            for c in cases:
                overrides = c.get("variables") if isinstance(c.get("variables"), dict) else {}
                if var_key in {str(name).casefold() for name in overrides}:
                    sources.append(c["label"])
                    continue
                model = c.get("model", "")
                if model not in model_vars_cache:
                    if not model or remote_registry_unavailable:
                        model_vars_cache[model] = set()
                    elif remote_registry is not None:
                        try:
                            profile = remote_registry.get_model(model)
                            variables = profile.variables.keys() if profile and hasattr(profile, "variables") else []
                            model_vars_cache[model] = {str(name).casefold() for name in variables}
                        except Exception as exc:
                            if _remote_registry_is_offline(self.controller):
                                remote_registry_unavailable = True
                                model_vars_cache[model] = set()
                            else:
                                QMessageBox.critical(
                                    self, "Remote Registry Error", f"Failed to read remote model registry:\n{exc}"
                                )
                                return
                    else:
                        model_vars_cache[model] = {
                            str(name).casefold() for name in PageSimData._registry_model_variables(self, model)
                        }
                if var_key in model_vars_cache[model]:
                    sources.append(c["label"])
            if remote_registry_unavailable:
                old_sources = existing_inner_general.get(f"{var_name}_sim_source", [])
                old_sources = [old_sources] if isinstance(old_sources, str) else list(old_sources or [])
                for label in old_sources:
                    case = cases_by_label.get(label)
                    if case and label not in sources and case.get("model"):
                        sources.append(label)
            general[f"{var_name}_sim_source"] = sources

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
        self._model_names = PageSimData._registry_model_names(self)
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
        issues = []
        from openbench.remote.storage import RemoteStorage
        from openbench.gui.data_validator import FilePathGenerator, LocalNetCDFValidator, RemoteNetCDFValidator

        is_remote = isinstance(self.controller.storage, RemoteStorage)
        ssh_manager = get_remote_ssh_manager(self.controller) if is_remote else None
        general = self.controller.config.get("general", {}) if getattr(self.controller, "config", None) else {}
        syear = int(general.get("syear", 2000))
        eyear = int(general.get("eyear", 2020))
        remote_openbench_root = ""
        remote_settings = getattr(self.controller, "remote_settings", None)
        if callable(remote_settings):
            remote_openbench_root = remote_settings().get("openbench_path", "")
        file_checker = RemoteNetCDFValidator(ssh_manager) if is_remote else LocalNetCDFValidator()
        for c in cases:
            if is_remote:
                if not ssh_manager or not ssh_manager.is_connected:
                    issues.append(f"{c['label']}: remote server is not connected")
                    continue
                if c.get("data_type") == "stn":
                    detail = c.get("station_materialize_error")
                    if detail:
                        issues.append(f"{c['label']}: {detail}")
                        continue
                    fulllist = c.get("fulllist", "")
                    if not fulllist:
                        issues.append(f"{c['label']}: station fulllist is missing")
                        continue
                    check = file_checker.check_file_exists(fulllist)
                    if not check.passed:
                        issues.append(f"{c['label']}: {check.message}")
                    continue
                try:
                    nc_dir = _remote_find_nc_dir(ssh_manager, c["nc_dir"])
                except Exception as exc:
                    issues.append(f"{c['label']}: {exc}")
                    continue
                if not nc_dir:
                    issues.append(f"{c['label']}: no NetCDF files found ({c['nc_dir']})")
                    continue
            else:
                if not os.path.isdir(c["nc_dir"]):
                    issues.append(f"{c['label']}: directory not found ({c['nc_dir']})")
                    continue
                nc_dir = _find_nc_dir(c["nc_dir"])
                if not nc_dir:
                    issues.append(f"{c['label']}: no NetCDF files found ({c['nc_dir']})")
                    continue

            variable_patterns = [
                (name, override)
                for name, override in (c.get("variables") or {}).items()
                if isinstance(override, dict) and ("prefix" in override or "suffix" in override)
            ]
            patterns = variable_patterns or [("", {})]
            for variable_name, override in patterns:
                path_gen = FilePathGenerator(
                    root_dir=nc_dir,
                    sub_dir="",
                    prefix=override.get("prefix", c.get("prefix", "")),
                    suffix=override.get("suffix", c.get("suffix", "")),
                    data_groupby=override.get("data_groupby") or c.get("data_groupby") or "Year",
                    syear=syear,
                    eyear=eyear,
                    is_remote=is_remote,
                    ssh_manager=ssh_manager,
                    remote_openbench_root=remote_openbench_root,
                )
                sample_paths = path_gen.get_sample_paths()
                issue_label = f"{c['label']} ({variable_name})" if variable_name else c["label"]
                if not sample_paths:
                    pattern = path_gen.describe_pattern()
                    message = getattr(path_gen, "last_error", None) or (
                        f"No files found matching pattern '{pattern}' in {path_gen._get_base_dir()}"
                    )
                    issues.append(f"{issue_label}: {message}")
                    continue

                file_checks = [file_checker.check_file_exists(path) for path in sample_paths]
                if not any(check.passed for check in file_checks):
                    pattern = path_gen.describe_pattern()
                    issues.append(
                        f"{issue_label}: No files found matching pattern '{pattern}' in {path_gen._get_base_dir()}"
                    )
                elif any(not check.passed for check in file_checks):
                    failed = next(check for check in file_checks if not check.passed)
                    issues.append(f"{issue_label}: {failed.message}")
        if issues:
            QMessageBox.warning(self, "Validation Issues", "\n".join(issues))
        else:
            QMessageBox.information(self, "Validation OK", f"All {len(cases)} case directories contain NetCDF files.")
