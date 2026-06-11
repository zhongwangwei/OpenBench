# -*- coding: utf-8 -*-
"""
Cross-platform path utilities for converting and validating paths.
"""

import os
import sys
import logging
from importlib.resources import files
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


_OPENBENCH_ROOT_MARKERS = [
    "/OpenBench/",
    "/openbench/",
]

_PROJECT_PATH_MARKERS = [
    "/output/",
]

# Legacy v2 layout fragments kept only for cross-platform conversion of
# existing saved paths. They are intentionally quarantined away from the
# active OpenBench-root markers so new v3 code does not treat them as
# install-layout defaults.
_LEGACY_V2_PATH_MARKERS = [
    "/nml/nml-yaml/",
    "/nml/nml-Fortran/",
    "/Mod_variables_definition/",
]


def is_cross_platform_path(path: str) -> bool:
    """
    Check if a path appears to be from a different platform.

    On Windows: returns True if path looks like a Unix path (starts with /)
    On Unix/Mac: returns True if path looks like a Windows path (has drive letter)

    Args:
        path: Path to check

    Returns:
        True if path appears to be from another platform
    """
    if not path:
        return False

    is_windows = sys.platform == "win32"

    # On Windows, check for Unix-style paths
    if is_windows:
        # Unix absolute path (starts with / but not UNC path //)
        if path.startswith("/") and not path.startswith("//"):
            return True

    # On Unix/Mac, check for Windows-style paths
    else:
        # Windows drive letter (e.g., C:\ or C:/)
        if len(path) >= 2 and path[1] == ":":
            return True
        # Windows backslash paths
        if "\\" in path and not path.startswith("/"):
            return True

    return False


def to_posix_path(path: str) -> str:
    """
    Convert a path to POSIX format (forward slashes).

    Use this for paths that will be used on remote Linux servers.

    Args:
        path: Path string with any separators

    Returns:
        Path with forward slashes only
    """
    if not path:
        return ""
    return path.replace("\\", "/")


def remote_join(*parts) -> str:
    """
    Join path components using forward slashes (for remote Linux paths).

    Use this instead of os.path.join() when constructing remote paths.

    Args:
        *parts: Path components to join

    Returns:
        Joined path with forward slashes
    """
    # Filter out empty parts and join with forward slashes
    clean_parts = []
    for i, part in enumerate(parts):
        if not part:
            continue
        part = str(part).replace("\\", "/")
        if i == 0:
            # Keep leading slash for absolute paths
            clean_parts.append(part.rstrip("/"))
        else:
            clean_parts.append(part.strip("/"))
    return "/".join(clean_parts)


def remote_dirname(path: str) -> str:
    """
    Get the directory name of a remote path.

    Use this instead of os.path.dirname() for remote paths.

    Args:
        path: Remote path

    Returns:
        Parent directory path
    """
    if not path:
        return ""
    path = to_posix_path(path).rstrip("/")
    if "/" not in path:
        return ""
    return "/".join(path.split("/")[:-1]) or "/"


def remote_basename(path: str) -> str:
    """
    Get the base name of a remote path.

    Use this instead of os.path.basename() for remote paths.

    Args:
        path: Remote path

    Returns:
        Base name (last component)
    """
    if not path:
        return ""
    path = to_posix_path(path).rstrip("/")
    return path.split("/")[-1]


def looks_like_openbench_root(root: str) -> bool:
    """Detect a valid OpenBench v3 root in any supported install layout.

    v3 is a pip-installable package; the v2 marker
    ``openbench/openbench.py`` no longer exists. We accept any of:

      * **editable / repo checkout** — ``src/openbench/cli/main.py``
        exists under ``root``;
      * **source tree with metadata** — ``pyproject.toml`` under
        ``root`` declares the ``colm-openbench`` distribution;
      * **wheel / pip install** — ``root`` is the package directory
        itself (i.e. it has ``cli/main.py`` and ``data/registry/``); or
        it is a parent directory containing such an
        ``openbench/`` subpackage (e.g. ``site-packages``).

    Without the wheel-layout cases, pip-install users would be forced
    to pick a directory that doesn't exist on their machine.

    Used by:
      - ``find_openbench_root`` below
      - the GUI's "browse for OpenBench directory" dialogs in
        page_runtime / page_run_monitor
      - ``config_manager._is_openbench_installation``
    """
    if not root or not os.path.isdir(root):
        return False

    # 1) Editable / repo checkout
    if os.path.exists(os.path.join(root, "src", "openbench", "cli", "main.py")):
        return True

    # 2) Source tree with pyproject.toml
    pyproject = os.path.join(root, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            with open(pyproject, "r", encoding="utf-8") as fh:
                head = fh.read(4096)
            if (
                'name = "colm-openbench"' in head
                or "name = 'colm-openbench'" in head
                or 'name = "openbench"' in head
                or "name = 'openbench'" in head
            ):
                return True
        except OSError:
            pass

    # 3) The package directory itself (wheel-installed `…/site-packages/openbench`)
    if os.path.exists(os.path.join(root, "cli", "main.py")) and os.path.isdir(os.path.join(root, "data", "registry")):
        return True

    # 4) A parent directory containing an installed `openbench/` subpackage
    #    (e.g. site-packages). We only require the same two markers under
    #    `openbench/` so a random folder named `openbench` doesn't match.
    sub = os.path.join(root, "openbench")
    if (
        os.path.isdir(sub)
        and os.path.exists(os.path.join(sub, "cli", "main.py"))
        and os.path.isdir(os.path.join(sub, "data", "registry"))
    ):
        return True

    return False


def get_openbench_root() -> str:
    """
    Find the OpenBench root directory.

    Resolution order (each candidate is validated by
    :func:`looks_like_openbench_root` before being accepted):

      1. ``~/.openbench_wizard/config.txt`` saved path
      2. Common user folders (Desktop / Documents / home / OpenBench)
      3. Current working directory
      4. The directory that *contains* the installed `openbench`
         package (so a `pip install`-only user always gets a sensible
         answer pointing at site-packages)

    Returns:
        Absolute path to a directory that passes the v3 marker check.
        If no candidate qualifies the function still returns CWD as a
        last-resort string (callers downstream emit warnings) — but
        `looks_like_openbench_root(returned_path)` is True in all
        normal install layouts, including wheel installs.
    """
    # 1) Saved path — but only if it's actually an OpenBench root.
    #    Earlier this trusted any existing path, which let a stale
    #    `~/.openbench_wizard/config.txt` pollute every relative path
    #    in the GUI (to_absolute_path, default output_dir, legacy
    #    nml sync, …) with no warning.
    try:
        home_dir = os.path.expanduser("~")
        config_file = os.path.join(home_dir, ".openbench_wizard", "config.txt")
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                saved = f.read().strip()
            if saved and looks_like_openbench_root(saved):
                return os.path.normpath(saved)
            if saved and os.path.exists(saved):
                # Path exists but isn't a v3 root — warn rather than
                # silently returning a misconfigured location.
                print(f"Warning: saved OpenBench path {saved!r} is not a valid v3 install root; ignoring.")
    except Exception as e:
        print(f"Warning: Could not load saved OpenBench path: {e}")

    # 1b) Honour an explicit OPENBENCH_ROOT env var (CI / HPC users set this
    # to point at a shared install rather than rely on the hard-coded list
    # below). Validated with the same v3-root check so a stale env var
    # doesn't silently misconfigure the GUI.
    env_root = os.environ.get("OPENBENCH_ROOT", "").strip()
    if env_root:
        if looks_like_openbench_root(env_root):
            return os.path.normpath(env_root)
        else:
            print(f"Warning: OPENBENCH_ROOT={env_root!r} is not a valid v3 install root; ignoring.")

    # 2) Search common locations using the shared v3 marker check.
    possible_roots = [
        os.path.join(os.path.expanduser("~"), "Desktop", "OpenBench"),
        os.path.join(os.path.expanduser("~"), "Documents", "OpenBench"),
        os.path.join(os.path.expanduser("~"), "OpenBench"),
    ]

    for root in possible_roots:
        if looks_like_openbench_root(root):
            return os.path.normpath(root)

    # 3) Current working directory if it qualifies.
    cwd = os.getcwd()
    if looks_like_openbench_root(cwd):
        return os.path.normpath(cwd)

    # 4) For pip-only filesystem installs, point at the directory that
    #    contains the installed `openbench` package (e.g.
    #    `.../site-packages`). Avoid package.__file__; in zipped wheels
    #    there is no stable filesystem parent to return, so this branch
    #    simply won't match and callers fall back to CWD.
    try:
        package_root = files("openbench")
        if isinstance(package_root, Path):
            installed_pkg_parent = str(package_root.parent)
            if looks_like_openbench_root(installed_pkg_parent):
                return os.path.normpath(installed_pkg_parent)
    except Exception:
        pass

    # 5) Last-resort fallback: CWD. `looks_like_openbench_root` will
    #    return False for callers that need to gate on it.
    return os.path.normpath(cwd)


def get_remote_ssh_manager(controller):
    """Get SSH manager from the controller if in remote mode.

    Returns:
        SSHManager instance if in remote mode and connected, None otherwise.
    """
    from openbench.remote.storage import RemoteStorage

    if not isinstance(controller.storage, RemoteStorage):
        return None
    return controller.ssh_manager


def _remote_directory_exists(ssh_manager, path: str) -> bool:
    """Check that a directory exists on the remote host."""
    from openbench.gui.widgets._ssh_worker import execute_responsive
    from openbench.gui.widgets.remote_config import _safe_remote_path

    try:
        quoted = _safe_remote_path(path)
        _, _, exit_code = execute_responsive(ssh_manager, f"test -d {quoted}", timeout=5)
    except Exception as exc:
        logger.debug("Remote directory check failed for %s: %s", path, exc)
        return False
    return exit_code == 0


def remote_home_dir(ssh_manager) -> str:
    """Best-effort remote home directory, falling back to '/'."""
    from openbench.gui.widgets._ssh_worker import call_responsive

    try:
        return call_responsive(ssh_manager._get_home_dir)
    except Exception as exc:
        logger.debug("Failed to get remote home directory: %s", exc)
        return "/"


def _resolve_remote_start_path(controller, ssh_manager, current_path: str = "") -> str:
    """Pick a start directory for the remote browser.

    Every candidate is validated on the remote host: a stale path (e.g. a
    local path left over from a local-mode session, or an outdated
    openbench_path) must not seed the browser — RemoteFileBrowser would keep
    it as its current path and the Select button could round-trip it back as
    the "chosen" remote directory.
    """
    if current_path and _remote_directory_exists(ssh_manager, current_path):
        return current_path
    openbench_path = controller.remote_settings().get("openbench_path", "")
    if openbench_path and _remote_directory_exists(ssh_manager, openbench_path):
        return openbench_path
    home = remote_home_dir(ssh_manager)
    if home and home != "/" and _remote_directory_exists(ssh_manager, home):
        return home
    return "/"


def pick_remote_path(ssh_manager, parent, title: str, start_path: str, select_dirs: bool = True) -> str:
    """Run the remote file browser dialog and return the chosen path ('' if cancelled)."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout

    from openbench.gui.widgets.remote_config import RemoteFileBrowser

    selected: list[str] = []
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(500, 400)
    layout = QVBoxLayout(dialog)
    browser = RemoteFileBrowser(ssh_manager, start_path, dialog, select_dirs=select_dirs)
    layout.addWidget(browser)

    def on_path_selected(path):
        selected.append(path)
        dialog.accept()

    browser.file_selected.connect(on_path_selected)
    dialog.exec()
    dialog.deleteLater()
    return selected[0] if selected else ""


def browse_remote_directory(controller, parent, title: str, current_path: str = "") -> str:
    """Return a remote directory selected through the SSH browser."""
    from PySide6.QtWidgets import QMessageBox

    ssh_manager = get_remote_ssh_manager(controller)
    if not ssh_manager or not getattr(ssh_manager, "is_connected", False):
        QMessageBox.warning(
            parent,
            "Not Connected",
            "Remote server is not connected.\nPlease connect on the Runtime Environment page first.",
        )
        return ""

    start_path = _resolve_remote_start_path(controller, ssh_manager, current_path)
    return pick_remote_path(ssh_manager, parent, title, start_path, select_dirs=True)


def remote_exec_context(controller, parent):
    """SSH manager + python/conda/openbench settings for remote execution.

    Returns ``{}`` in local mode, ``None`` (after warning the user) when
    remote mode is on but the server is not connected, and otherwise the
    kwargs shared by every remote scan/import entry point.
    """
    if not controller.is_remote_mode():
        return {}
    ssh_manager = get_remote_ssh_manager(controller)
    if not ssh_manager or not getattr(ssh_manager, "is_connected", False):
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(parent, "Not Connected", "Remote mode requires connecting to the server first.")
        return None
    remote_config = controller.remote_settings()
    return {
        "ssh_manager": ssh_manager,
        "python_path": remote_config.get("python_path", ""),
        "conda_env": remote_config.get("conda_env", ""),
        "openbench_path": remote_config.get("openbench_path", ""),
    }


def browse_directory(controller, parent, title: str, current_path: str = "") -> str:
    """Pick a directory with the dialog matching the controller's storage mode."""
    if controller.is_remote_mode():
        return browse_remote_directory(controller, parent, title, current_path)
    from PySide6.QtWidgets import QFileDialog

    return QFileDialog.getExistingDirectory(parent, title, current_path or "") or ""


def to_absolute_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Convert a path to absolute path, with cross-platform support.

    Args:
        path: Path to convert (can be relative or absolute, from any platform)
        base_dir: Base directory for relative paths (defaults to OpenBench root)

    Returns:
        Absolute path with normalized separators for current platform
    """
    if not path:
        return ""

    # Get base directory first (needed for cross-platform conversion)
    if base_dir is None:
        base_dir = get_openbench_root()
    else:
        base_dir = os.path.normpath(base_dir)

    # Try cross-platform conversion first (Linux path on Windows, or vice versa)
    converted_path = convert_cross_platform_path(path, base_dir)
    if converted_path != path:
        # Cross-platform conversion was applied
        return os.path.normpath(converted_path)

    # Normalize path separators for current platform
    path = normalize_path_separators(path)

    # If already absolute, just normalize and return
    if os.path.isabs(path):
        return os.path.normpath(path)

    # Handle relative paths starting with ./
    if path.startswith("./") or path.startswith(".\\"):
        path = path[2:]

    # Join with base directory and normalize
    return os.path.normpath(os.path.join(base_dir, path))


def normalize_path_separators(path: str) -> str:
    """
    Normalize path separators for the current platform.

    Args:
        path: Path string with potentially mixed separators

    Returns:
        Path with correct separators for current platform
    """
    if not path:
        return ""

    # Replace both types of separators with the current platform's separator
    if os.sep == "/":
        return path.replace("\\", "/")
    else:
        return path.replace("/", "\\")


def convert_cross_platform_path(path: str, openbench_root: Optional[str] = None) -> str:
    """
    Convert a path from another platform to the current platform.

    On Windows: converts Linux absolute paths (starting with /) to Windows paths
    On Linux/Mac: converts Windows absolute paths (with drive letter) to Unix paths

    The conversion works by:
    1. Detecting if path is from another platform
    2. Finding the relative portion within OpenBench structure
    3. Rebuilding with current OpenBench root

    Args:
        path: Path that may be from another platform
        openbench_root: Current OpenBench root directory

    Returns:
        Converted path for current platform, or original if conversion not possible
    """
    if not path:
        return ""

    if openbench_root is None:
        openbench_root = get_openbench_root()

    is_windows = sys.platform == "win32"

    # Detect Linux path on Windows (starts with / but no drive letter)
    if is_windows and path.startswith("/") and not path.startswith("//"):
        return _convert_linux_to_windows(path, openbench_root)

    # Detect Windows path on Linux/Mac (has drive letter like C:\ or C:/)
    if not is_windows and len(path) >= 2 and path[1] == ":":
        return _convert_windows_to_linux(path, openbench_root)

    return path


def _convert_linux_to_windows(linux_path: str, openbench_root: str) -> str:
    """Convert a Linux path to Windows path by finding relative portion."""
    # Normalize separators in the linux path for searching
    search_path = linux_path.replace("\\", "/")

    # Common markers that indicate OpenBench structure. Legacy v2
    # fragments are quarantined in a separate constant; they support
    # conversion of old saved paths, not active v3 root discovery.
    markers = _OPENBENCH_ROOT_MARKERS + _PROJECT_PATH_MARKERS + _LEGACY_V2_PATH_MARKERS

    for marker in markers:
        lower_search = search_path.lower()
        lower_marker = marker.lower()
        if lower_marker in lower_search:
            # Extract the relative path after the marker's parent
            if lower_marker == "/openbench/":
                # Get everything after OpenBench/
                idx = lower_search.find("/openbench/")
                relative = search_path[idx + len("/openbench/") :]
            else:
                # For other markers, find OpenBench root first
                idx = lower_search.find("/openbench/")
                if idx >= 0:
                    relative = search_path[idx + len("/openbench/") :]
                else:
                    # Just use the marker position
                    idx = lower_search.find(lower_marker)
                    relative = search_path[idx + 1 :]  # Skip leading /

            # Build Windows path - normalize all separators to backslash
            if relative:
                relative = relative.replace("/", "\\")
                result = os.path.join(openbench_root, relative)
                return result.replace("/", "\\")

    # If no marker found, just normalize separators
    return linux_path.replace("/", "\\")


def _convert_windows_to_linux(windows_path: str, openbench_root: str) -> str:
    """Convert a Windows path to Linux path by finding relative portion."""
    # Normalize separators
    search_path = windows_path.replace("\\", "/")

    # Common markers. See _convert_linux_to_windows for why legacy v2
    # markers are kept separate from root markers.
    markers = _OPENBENCH_ROOT_MARKERS + _PROJECT_PATH_MARKERS + _LEGACY_V2_PATH_MARKERS

    for marker in markers:
        lower_search = search_path.lower()
        lower_marker = marker.lower()
        if lower_marker in lower_search:
            if lower_marker == "/openbench/":
                idx = lower_search.find("/openbench/")
                relative = search_path[idx + len("/openbench/") :]
            else:
                idx = lower_search.find("/openbench/")
                if idx >= 0:
                    relative = search_path[idx + len("/openbench/") :]
                else:
                    idx = lower_search.find(lower_marker)
                    relative = search_path[idx + 1 :]

            # Build Linux path - normalize all separators to forward slash
            if relative:
                relative = relative.replace("\\", "/")
                result = os.path.join(openbench_root, relative)
                return result.replace("\\", "/")

    # If no marker found, just normalize separators
    return windows_path.replace("\\", "/")


def validate_path(path: str, path_type: str = "file", must_exist: bool = True) -> Tuple[bool, str]:
    """
    Validate a path exists and is the correct type.

    Args:
        path: Path to validate
        path_type: "file" or "directory"
        must_exist: If True, path must exist; if False, parent directory must exist

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return True, ""  # Empty paths are OK (optional fields)

    # Normalize path
    path = os.path.normpath(path)

    if must_exist:
        if not os.path.exists(path):
            return False, f"Path does not exist: {path}"

        if path_type == "file" and not os.path.isfile(path):
            return False, f"Path is not a file: {path}"

        if path_type == "directory" and not os.path.isdir(path):
            return False, f"Path is not a directory: {path}"
    else:
        # Check parent directory exists
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            return False, f"Parent directory does not exist: {parent}"

    return True, ""


def convert_paths_in_dict(
    data: dict,
    base_dir: Optional[str] = None,
    path_keys: Optional[list] = None,
    all_values_are_paths_keys: Optional[list] = None,
) -> dict:
    """
    Recursively convert all path values in a dictionary to absolute paths.

    Args:
        data: Dictionary containing paths
        base_dir: Base directory for relative paths
        path_keys: List of keys that contain paths (if None, uses default list)
        all_values_are_paths_keys: List of keys whose child dict has ALL values as paths
                                   (e.g., 'def_nml' where all values are path strings)

    Returns:
        Dictionary with converted paths
    """
    if path_keys is None:
        path_keys = [
            "root_dir",
            "basedir",
            "fulllist",
            "model_namelist",
            "reference_nml",
            "simulation_nml",
            "statistics_nml",
            "figure_nml",
            "def_nml_path",
            "data_path",
            "file_path",
            "output_dir",
        ]

    if all_values_are_paths_keys is None:
        all_values_are_paths_keys = ["def_nml"]

    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        # Special handling for sections where ALL values are paths (like def_nml)
        if key in all_values_are_paths_keys and isinstance(value, dict):
            result[key] = {
                k: to_absolute_path(v, base_dir) if isinstance(v, str) and v else v for k, v in value.items()
            }
        elif isinstance(value, dict):
            result[key] = convert_paths_in_dict(value, base_dir, path_keys, all_values_are_paths_keys)
        elif isinstance(value, list):
            result[key] = [
                convert_paths_in_dict(item, base_dir, path_keys, all_values_are_paths_keys)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        elif isinstance(value, str) and key in path_keys and value:
            result[key] = to_absolute_path(value, base_dir)
        else:
            result[key] = value

    return result


def validate_paths_in_dict(
    data: dict, path_keys: Optional[list] = None, all_values_are_paths_keys: Optional[list] = None
) -> list:
    """
    Validate all paths in a dictionary.

    Args:
        data: Dictionary containing paths
        path_keys: List of keys that contain paths
        all_values_are_paths_keys: List of keys whose child dict has ALL values as paths
                                   (e.g., 'def_nml' where all values are path strings)

    Returns:
        List of (key, path, error_message) tuples for invalid paths
    """
    if path_keys is None:
        path_keys = [
            "root_dir",
            "basedir",
            "fulllist",
            "model_namelist",
            "reference_nml",
            "simulation_nml",
            "statistics_nml",
            "figure_nml",
        ]

    if all_values_are_paths_keys is None:
        all_values_are_paths_keys = ["def_nml"]

    errors = []

    def _validate_recursive(d, prefix=""):
        if not isinstance(d, dict):
            return

        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Special handling for sections where ALL values are paths (like def_nml)
            if key in all_values_are_paths_keys and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and sub_value:
                        sub_full_key = f"{full_key}.{sub_key}"
                        is_valid, error = validate_path(sub_value, "file")
                        if not is_valid:
                            errors.append((sub_full_key, sub_value, error))
            elif isinstance(value, dict):
                _validate_recursive(value, full_key)
            elif isinstance(value, str) and key in path_keys and value:
                # Determine path type
                path_type = "directory" if key in ["root_dir", "basedir", "output_dir"] else "file"
                is_valid, error = validate_path(value, path_type)
                if not is_valid:
                    errors.append((full_key, value, error))

    _validate_recursive(data)
    return errors
