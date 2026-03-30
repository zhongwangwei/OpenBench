# -*- coding: utf-8 -*-
"""
Cross-platform path utilities for converting and validating paths.
"""

import os
import sys
from typing import Optional, Tuple


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

    is_windows = sys.platform == 'win32'

    # On Windows, check for Unix-style paths
    if is_windows:
        # Unix absolute path (starts with / but not UNC path //)
        if path.startswith('/') and not path.startswith('//'):
            return True

    # On Unix/Mac, check for Windows-style paths
    else:
        # Windows drive letter (e.g., C:\ or C:/)
        if len(path) >= 2 and path[1] == ':':
            return True
        # Windows backslash paths
        if '\\' in path and not path.startswith('/'):
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
    return path.replace('\\', '/')


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
        part = str(part).replace('\\', '/')
        if i == 0:
            # Keep leading slash for absolute paths
            clean_parts.append(part.rstrip('/'))
        else:
            clean_parts.append(part.strip('/'))
    return '/'.join(clean_parts)


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
    path = to_posix_path(path).rstrip('/')
    if '/' not in path:
        return ""
    return '/'.join(path.split('/')[:-1]) or '/'


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
    path = to_posix_path(path).rstrip('/')
    return path.split('/')[-1]


def get_openbench_root() -> str:
    """
    Find the OpenBench root directory.

    Returns:
        Absolute path to OpenBench root directory
    """
    # Try to load saved path first
    try:
        home_dir = os.path.expanduser("~")
        config_file = os.path.join(home_dir, ".openbench_wizard", "config.txt")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                path = f.read().strip()
                if path and os.path.exists(path):
                    return os.path.normpath(path)
    except Exception as e:
        print(f"Warning: Could not load saved OpenBench path: {e}")

    # Search common locations
    possible_roots = [
        os.path.join(os.path.expanduser("~"), "Desktop", "OpenBench"),
        os.path.join(os.path.expanduser("~"), "Documents", "OpenBench"),
        os.path.join(os.path.expanduser("~"), "OpenBench"),
    ]

    for root in possible_roots:
        if root and os.path.exists(os.path.join(root, "openbench", "openbench.py")):
            return os.path.normpath(root)

    # Fallback to current working directory
    return os.path.normpath(os.getcwd())


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
    if os.sep == '/':
        return path.replace('\\', '/')
    else:
        return path.replace('/', '\\')


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

    is_windows = sys.platform == 'win32'

    # Detect Linux path on Windows (starts with / but no drive letter)
    if is_windows and path.startswith('/') and not path.startswith('//'):
        return _convert_linux_to_windows(path, openbench_root)

    # Detect Windows path on Linux/Mac (has drive letter like C:\ or C:/)
    if not is_windows and len(path) >= 2 and path[1] == ':':
        return _convert_windows_to_linux(path, openbench_root)

    return path


def _convert_linux_to_windows(linux_path: str, openbench_root: str) -> str:
    """Convert a Linux path to Windows path by finding relative portion."""
    # Normalize separators in the linux path for searching
    search_path = linux_path.replace('\\', '/')

    # Common markers that indicate OpenBench structure
    markers = [
        '/OpenBench/',
        '/openbench/',
        '/nml/nml-yaml/',
        '/nml/nml-Fortran/',
        '/output/',
        '/Mod_variables_definition/',
    ]

    for marker in markers:
        lower_search = search_path.lower()
        lower_marker = marker.lower()
        if lower_marker in lower_search:
            # Extract the relative path after the marker's parent
            if lower_marker == '/openbench/':
                # Get everything after OpenBench/
                idx = lower_search.find('/openbench/')
                relative = search_path[idx + len('/openbench/'):]
            else:
                # For other markers, find OpenBench root first
                idx = lower_search.find('/openbench/')
                if idx >= 0:
                    relative = search_path[idx + len('/openbench/'):]
                else:
                    # Just use the marker position
                    idx = lower_search.find(lower_marker)
                    relative = search_path[idx + 1:]  # Skip leading /

            # Build Windows path - normalize all separators to backslash
            if relative:
                relative = relative.replace('/', '\\')
                result = os.path.join(openbench_root, relative)
                return result.replace('/', '\\')

    # If no marker found, just normalize separators
    return linux_path.replace('/', '\\')


def _convert_windows_to_linux(windows_path: str, openbench_root: str) -> str:
    """Convert a Windows path to Linux path by finding relative portion."""
    # Normalize separators
    search_path = windows_path.replace('\\', '/')

    # Common markers
    markers = [
        '/OpenBench/',
        '/openbench/',
        '/nml/nml-yaml/',
        '/nml/nml-Fortran/',
        '/output/',
        '/Mod_variables_definition/',
    ]

    for marker in markers:
        lower_search = search_path.lower()
        lower_marker = marker.lower()
        if lower_marker in lower_search:
            if lower_marker == '/openbench/':
                idx = lower_search.find('/openbench/')
                relative = search_path[idx + len('/openbench/'):]
            else:
                idx = lower_search.find('/openbench/')
                if idx >= 0:
                    relative = search_path[idx + len('/openbench/'):]
                else:
                    idx = lower_search.find(lower_marker)
                    relative = search_path[idx + 1:]

            # Build Linux path - normalize all separators to forward slash
            if relative:
                relative = relative.replace('\\', '/')
                result = os.path.join(openbench_root, relative)
                return result.replace('\\', '/')

    # If no marker found, just normalize separators
    return windows_path.replace('\\', '/')


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


def convert_paths_in_dict(data: dict, base_dir: Optional[str] = None, path_keys: Optional[list] = None,
                          all_values_are_paths_keys: Optional[list] = None) -> dict:
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
            "root_dir", "basedir", "fulllist", "model_namelist",
            "reference_nml", "simulation_nml", "statistics_nml", "figure_nml",
            "def_nml_path", "data_path", "file_path", "output_dir"
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
                k: to_absolute_path(v, base_dir) if isinstance(v, str) and v else v
                for k, v in value.items()
            }
        elif isinstance(value, dict):
            result[key] = convert_paths_in_dict(value, base_dir, path_keys, all_values_are_paths_keys)
        elif isinstance(value, list):
            result[key] = [
                convert_paths_in_dict(item, base_dir, path_keys, all_values_are_paths_keys) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str) and key in path_keys and value:
            result[key] = to_absolute_path(value, base_dir)
        else:
            result[key] = value

    return result


def validate_paths_in_dict(data: dict, path_keys: Optional[list] = None,
                          all_values_are_paths_keys: Optional[list] = None) -> list:
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
            "root_dir", "basedir", "fulllist", "model_namelist",
            "reference_nml", "simulation_nml", "statistics_nml", "figure_nml"
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
