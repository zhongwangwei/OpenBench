"""Auto-scan reference data directories and discover datasets.

Walks a reference data directory tree with the structure:
    <root>/Grid/LowRes/<category>/<variable>/<dataset>/
    <root>/Grid/MidRes/<category>/<variable>/<dataset>/
    <root>/Grid/HigRes/<category>/<variable>/<dataset>/
    <root>/Station/<category>/<variable>/<dataset>/

Groups datasets by base name across resolutions so the GUI can offer
resolution choices (e.g., GLEAM_v4.2a → LowRes / MidRes / HigRes).
"""

from __future__ import annotations

from fnmatch import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import yaml

from openbench.config.user_settings import resolve_home_dir, resolve_reference_root
from openbench.util.names import AmbiguousNameError
from openbench.util.netcdf import write_file_atomic

logger = logging.getLogger(__name__)

# NetCDF file discovery — supports .nc and .nc4
from openbench.data.coordinates import glob_nc as _glob_nc

from openbench.data.registry._filename_dates import (
    _classify_filename_date,
    _filename_split_match,
    _is_year_range_endpoint,
    _most_specific_date_matches,
)

classify_filename_date = _classify_filename_date
filename_split_match = _filename_split_match
is_year_range_endpoint = _is_year_range_endpoint
most_specific_date_matches = _most_specific_date_matches

DEFAULT_NC_DESCENT = 2

_REFERENCE_SCAN_EXCLUDE_DIRS = {
    "__pycache__",
    "backup",
    "cache",
    "debug",
    "derived",
    "figure",
    "figures",
    "log",
    "logs",
    "plot",
    "plots",
    "restart",
    "restarts",
    "rest",
    "tmp",
    "*_derived",
}


def _package_reference_profiles_path():
    """Return the package's bundled reference_profiles.yaml resource.

    The return value is an importlib.resources Traversable in production.
    Tests may monkey-patch this helper to return a pathlib.Path fixture.
    """
    return files("openbench.data.registry") / "reference_profiles.yaml"


def _load_package_reference_profiles() -> dict:
    """Load the bundled reference_profiles.yaml as a dict.

    Uses importlib.resources Traversable access so this works correctly
    under zipimport / PyInstaller bundles too. Returns ``{}`` if the
    file is missing or unparseable.

    Tests can monkey-patch :func:`_package_reference_profiles_path` to
    redirect the lookup; this function falls back to that path when
    the importlib.resources lookup is shadowed by a test fixture.
    """
    import yaml as _yaml

    # Allow tests to override via the helper above by checking whether
    # they've redirected to a custom on-disk location. If so, prefer
    # that explicit path so test fixtures (intentionally malformed
    # YAML, custom user-overlay scenarios, …) work as before.
    helper_path = _package_reference_profiles_path()
    try:
        # Compare to the real importlib.resources resource; if they differ,
        # the test has redirected and we should trust the redirect.
        real = files("openbench.data.registry") / "reference_profiles.yaml"
    except Exception:
        real = helper_path

    if helper_path != real:
        # Test redirected; use the redirected on-disk path directly.
        if not helper_path.is_file():
            return {}
        with helper_path.open("r", encoding="utf-8") as fh:
            return _yaml.safe_load(fh) or {}

    # Production path: zip-safe via importlib.resources.
    if not helper_path.is_file():
        return {}
    try:
        with helper_path.open("r", encoding="utf-8") as fh:
            return _yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def _count_nc(directory: Path) -> int:
    """Count NetCDF files (.nc + .nc4) in a directory."""
    return len(_glob_nc(directory))


def _atomic_yaml_write(path: Path, data: dict) -> None:
    """Write YAML atomically via tempfile + rename to prevent corruption."""
    import tempfile

    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_load_catalog(catalog_path: Path) -> dict:
    """Load existing catalog. Refuse to proceed (raise) if file is corrupted.

    Previous behavior silently swallowed YAML errors and returned {}; the
    next write would then overwrite the entire catalog with just the new
    entry, deleting all previously registered datasets.
    """
    if not catalog_path.exists():
        return {}
    try:
        with open(catalog_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(
            f"Failed to load existing catalog at {catalog_path}: {e}\n"
            f"Refusing to overwrite — this would silently delete all previously "
            f"registered datasets. Manually fix the YAML or restore from a backup. "
            f"If unrecoverable, delete the file before re-running scan/register."
        ) from e


def _load_builtin_catalog_for_user_overlay(catalog_path: Path) -> dict:
    """Return bundled reference catalog when writing the default user overlay.

    Scanned registrations are written to ``~/.openbench/references``. When a
    scan refreshes a dataset that already exists in the bundled catalog, the
    user file should contain only the meaningful overlay instead of a full copy
    of the bundled descriptor.
    """
    try:
        from openbench.data.registry import manager as registry_manager

        user_catalog = registry_manager.get_writable_reference_catalog_path()
        package_catalog = registry_manager.REGISTRY_DIR / "reference_catalog.yaml"
        if Path(catalog_path).resolve() != user_catalog.resolve():
            return {}
        if Path(catalog_path).resolve() == package_catalog.resolve():
            return {}
        if not package_catalog.exists():
            return {}
        return _safe_load_catalog(package_catalog)
    except Exception as e:
        logger.debug("Could not load bundled catalog for overlay diff: %s", e)
        return {}


def _merge_descriptor_overlay(base: Optional[dict], overlay: Optional[dict]) -> Optional[dict]:
    """Merge a raw user overlay dict onto a raw bundled descriptor dict."""
    if base is None:
        return overlay
    if not overlay:
        return dict(base)

    import copy

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key == "variables" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            variables = copy.deepcopy(merged["variables"])
            for var_name, var_data in value.items():
                if var_data is None:
                    variables.pop(var_name, None)
                else:
                    variables[var_name] = copy.deepcopy(var_data)
            merged["variables"] = variables
        elif value is None:
            merged.pop(key, None)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _descriptor_overlay_diff(base: dict, descriptor: dict) -> dict:
    """Return the minimal user overlay needed to reproduce descriptor over base."""
    overlay: dict = {}
    for key, value in descriptor.items():
        base_value = base.get(key)
        if key == "variables" and isinstance(value, dict) and isinstance(base_value, dict):
            variable_overlay = {
                var_name: var_data for var_name, var_data in value.items() if base_value.get(var_name) != var_data
            }
            for var_name in base_value:
                if var_name not in value:
                    variable_overlay[var_name] = None
            if variable_overlay:
                overlay[key] = variable_overlay
        elif isinstance(value, dict):
            if base_value != value:
                overlay[key] = value
        elif base_value != value:
            overlay[key] = value
    for key in base:
        if key not in descriptor:
            overlay[key] = None
    return overlay


def _store_catalog_descriptor(catalog: dict, name: str, descriptor: dict, base_descriptor: Optional[dict]) -> None:
    """Store either a full descriptor or a minimal overlay into catalog."""
    if base_descriptor is None:
        catalog[name] = descriptor
        return

    overlay = _descriptor_overlay_diff(base_descriptor, descriptor)
    if overlay:
        catalog[name] = overlay
    else:
        catalog.pop(name, None)


def _backup_then_write(catalog_path: Path, data: dict) -> Path | None:
    """Backup the previous catalog (if any) before atomic-writing the new one.

    Creates a single-slot backup at ``<catalog>.bak``. The backup is the
    catalog state immediately before this write — useful when a buggy
    rescan overwrites hand-edited fields and the user wants to recover.
    """
    backup_path: Path | None = None
    if catalog_path.exists():
        import shutil
        from datetime import datetime

        candidate_backup_path = Path(str(catalog_path) + ".bak")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        history_path = catalog_path.with_name(f"{catalog_path.name}.{timestamp}.bak")
        counter = 1
        while history_path.exists():
            history_path = catalog_path.with_name(f"{catalog_path.name}.{timestamp}-{counter}.bak")
            counter += 1
        try:
            shutil.copy2(catalog_path, candidate_backup_path)
            shutil.copy2(catalog_path, history_path)
            backup_path = candidate_backup_path
        except OSError as e:
            logger.warning("Could not create catalog backup at %s: %s", candidate_backup_path, e)
    _atomic_yaml_write(catalog_path, data)
    return backup_path


def _invalidate_registry_caches() -> None:
    """Invalidate the singleton RegistryManager and reference profile cache.

    Without this, long-lived processes (GUI sessions, Jupyter kernels, custom
    Python scripts) call register_scanned_dataset[s_batch] but subsequent
    get_registry() returns the stale pre-write singleton. The CLI manually
    clears caches after scan; the public write API should do it too so
    callers don't need to know about the cache.
    """
    try:
        from openbench.data.registry.manager import clear_registry_cache

        clear_registry_cache()
    except Exception as e:
        logger.debug("Could not clear registry cache: %s", e)
    clear_reference_profile_cache()


import contextlib


@contextlib.contextmanager
def _catalog_write_lock(catalog_path: Path):
    """Serialize concurrent writes to the same catalog via an exclusive flock.

    Without this, two scan processes both load the same pre-write state,
    each compute their own additions, and the second writer silently
    overwrites the first writer's new entries. Common in HPC shared-registry
    scenarios where multiple users scan their reference roots in parallel.

    Uses fcntl.flock on POSIX and msvcrt.locking on Windows. If the lock
    cannot be acquired, fail closed instead of writing without serialization.
    """
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(catalog_path) + ".lock")
    try:
        lock_path.touch(exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"failed to acquire catalog lock {lock_path}: {e}") from e

    lock_file = open(lock_path, "a+")
    have_lock = False
    release = None
    try:
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)

                def release():
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                def release():
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

            have_lock = True
        except (OSError, ImportError) as e:
            raise RuntimeError(f"failed to acquire catalog lock {lock_path}: {e}") from e
        yield
    finally:
        if have_lock and release is not None:
            try:
                release()
            except OSError:
                pass
        lock_file.close()


# Lazy-loaded reference profiles (like model_catalog for references)
_REFERENCE_PROFILES: dict | None = None


def clear_reference_profile_cache() -> None:
    """Clear cached reference profiles after profile updates."""
    global _REFERENCE_PROFILES
    _REFERENCE_PROFILES = None


def _load_reference_profiles() -> dict:
    """Load reference dataset profiles from reference_profiles.yaml."""
    global _REFERENCE_PROFILES
    if _REFERENCE_PROFILES is None:
        from openbench.data.registry.manager import (
            get_legacy_reference_profiles_path,
            get_writable_reference_profiles_path,
        )

        profiles: dict[str, Any] = {}

        # 1) Package-bundled profile (zip-safe via _load_package_*).
        try:
            loaded = _load_package_reference_profiles()
            if loaded:
                profiles = _deep_merge_profile_dicts(profiles, loaded)
        except Exception as e:
            package_profile = _package_reference_profiles_path()
            logger.warning(
                "Failed to load reference profiles from %s: %s",
                package_profile,
                e,
            )

        # 2) User-writable + legacy paths (filesystem-only).
        profile_paths = []
        writable_profile = get_writable_reference_profiles_path()
        legacy_profile = get_legacy_reference_profiles_path()
        if legacy_profile != writable_profile:
            profile_paths.append(legacy_profile)
        profile_paths.append(writable_profile)

        for profile_path in profile_paths:
            if not profile_path.exists():
                continue
            loaded = _safe_load_catalog(profile_path)
            profiles = _deep_merge_profile_dicts(profiles, loaded)

        _REFERENCE_PROFILES = profiles
    return _REFERENCE_PROFILES


def _deep_merge_profile_dicts(base: dict, overlay: dict) -> dict:
    """Deep-merge reference profile dictionaries with overlay values winning."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_profile_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def get_reference_profile(dataset_name: str) -> dict | None:
    """Look up a reference profile by exact name or base name (without resolution suffix).

    Matching order:
    1. Exact match: "GLEAM_v4.2a_LowRes" → profile["GLEAM_v4.2a_LowRes"]
    2. Base name:   "GLEAM_v4.2a_LowRes" → strip "_LowRes" → profile["GLEAM_v4.2a"]

    Returns profile dict or None.
    """
    profiles = _load_reference_profiles()
    # 1. Exact match
    if dataset_name in profiles:
        return profiles[dataset_name]
    # 1b. User-facing profile names are case-insensitive, but keep exact
    # spelling above as the winner when both exist.
    try:
        from openbench.util.names import get_mapping_key_case_insensitive

        profile_key = get_mapping_key_case_insensitive(profiles, dataset_name)
    except Exception:
        profile_key = None
    if profile_key is not None:
        return profiles[profile_key]
    # 2. Strip resolution suffix and try base name
    for suffix in ("_LowRes", "_MidRes", "_HigRes"):
        if dataset_name.casefold().endswith(suffix.casefold()):
            base = dataset_name[: -len(suffix)]
            if base in profiles:
                return profiles[base]
            try:
                profile_key = get_mapping_key_case_insensitive(profiles, base)
            except Exception:
                profile_key = None
            if profile_key is not None:
                return profiles[profile_key]
    return None


# Map directory names to categories
CATEGORY_MAP = {
    "Water": "Water",
    "Heat": "Energy",
    "Bio": "Carbon",
    "Meteo": "Meteorology",
    "Anth": "Urban",
    "Composite": "Other",
}

# Map resolution directory names to metadata
RESOLUTION_MAP = {
    "LowRes": {"label": "Low Resolution", "typical_grid_res": 0.5},
    "MidRes": {"label": "Mid Resolution", "typical_grid_res": 0.25},
    "HigRes": {"label": "High Resolution", "typical_grid_res": 0.1},
}

_SCANNER_TEMP_VAR_KEYS = {
    "_detected_data_type",
    "_nc_tim_res",
    "_nc_grid_res",
    "_climatology_candidate",
    "_inconsistent_data_type",
    "_inconsistent_tim_res",
    "_inconsistent_grid_res",
}


def _public_existing_var_fields(existing_var: dict) -> dict:
    """Copy persisted variable fields, dropping scanner scratch metadata."""
    return {key: value for key, value in existing_var.items() if not str(key).startswith("_")}


def _has_custom_station_filter(dataset_name: str) -> bool:
    """Return True when a station dataset is handled by custom filter code."""
    try:
        custom_dir = files("openbench.data.custom")
    except Exception:
        return False
    return (custom_dir / f"{dataset_name}_filter.py").is_file()


@dataclass
class ScannedDataset:
    """A discovered dataset with its location and resolution info."""

    name: str  # e.g., "GLEAM_v4.2a"
    resolution: str  # "LowRes", "MidRes", "HigRes", or "Station"
    category: str  # "Water", "Energy", etc.
    data_type: str  # "grid" or "stn"
    root_dir: str  # Full path to the resolution-level root (e.g., .../Grid/LowRes/Water)
    variables: dict[str, str] = field(default_factory=dict)  # var_name -> sub_dir path
    file_globs: dict[str, str | list[str]] = field(default_factory=dict)
    file_count: int = 0
    tim_res: str = ""  # Detected or empty
    # Remote-scan support: when root_dir is not visible on this machine,
    # registration consumes inspection results computed on the remote host.
    nc_inspections: dict[str, dict] = field(default_factory=dict)  # var_name -> _inspect_nc_file result
    detected_data_groupby: str = ""  # remote-computed _detect_data_groupby result
    remote_fulllist: str = ""  # station fulllist CSV generated on the remote host

    @property
    def registry_name(self) -> str:
        """Name for registry entry: 'GLEAM_v4.2a_LowRes'."""
        if self.data_type == "stn":
            return self.name
        return f"{self.name}_{self.resolution}"


@dataclass
class DatasetGroup:
    """A dataset that may exist at multiple resolutions."""

    base_name: str  # e.g., "GLEAM_v4.2a"
    variants: dict[str, ScannedDataset] = field(default_factory=dict)  # resolution -> ScannedDataset

    @property
    def available_resolutions(self) -> list[str]:
        return sorted(self.variants.keys())

    @property
    def category(self) -> str:
        for v in self.variants.values():
            return v.category
        return "Other"


@dataclass(frozen=True)
class ScanSkip:
    """A source-data folder that scanner saw but did not auto-register."""

    path: str
    reason: str
    hint: str = ""


def _scan_skip(ref_root: Path, path: Path, reason: str, hint: str = "") -> ScanSkip:
    try:
        display_path = path.relative_to(ref_root).as_posix()
    except ValueError:
        display_path = str(path)
    return ScanSkip(path=display_path, reason=reason, hint=hint)


def _notify_skip(on_skip, skip: ScanSkip) -> None:
    if on_skip:
        on_skip(skip)


def _skip_candidate_registry_names(skip: ScanSkip) -> set[str]:
    """Infer registry names that would own an unsupported scanned path."""
    parts = Path(skip.path).parts
    names: set[str] = set()
    if len(parts) >= 4 and parts[0] == "Grid":
        res_name = parts[1]
        dataset_name = parts[4] if len(parts) >= 5 else parts[3]
        if dataset_name:
            names.add(f"{dataset_name}_{res_name}")
    elif parts and parts[0] == "Station":
        if len(parts) >= 4:
            names.add(parts[3])
        elif len(parts) == 3:
            names.add(parts[2])
    return names


def _expand_path(path_value: str | Path) -> Path:
    """Expand environment variables and user markers in scanner path values."""
    return Path(os.path.expandvars(os.path.expanduser(str(path_value))))


def _portable_root_dir(path: Path) -> str:
    """Use OPENBENCH_REF_ROOT in scanned catalog paths when it exactly applies."""
    ref_root = resolve_reference_root()
    if not ref_root:
        return path.as_posix()

    expanded_root = _expand_path(ref_root)
    try:
        rel = path.resolve().relative_to(expanded_root.resolve())
    except (OSError, ValueError):
        return path.as_posix()

    if str(rel) == ".":
        return "${OPENBENCH_REF_ROOT}"
    return "${OPENBENCH_REF_ROOT}/" + rel.as_posix()


def _portable_path(path: Path) -> str:
    """Use OPENBENCH_REF_ROOT or OPENBENCH_HOME in generated catalog paths when possible."""
    ref_portable = _portable_root_dir(path)
    if ref_portable != path.as_posix():
        return ref_portable

    home_env = os.environ.get("OPENBENCH_HOME")
    if home_env:
        home_root = resolve_home_dir()
        try:
            rel = path.resolve().relative_to(home_root.resolve())
        except (OSError, ValueError):
            return path.as_posix()

        if str(rel) == ".":
            return "${OPENBENCH_HOME}"
        return "${OPENBENCH_HOME}/" + rel.as_posix()

    return path.as_posix()


def _station_list_dir_for_catalog(scanned: ScannedDataset, catalog_path: Path) -> Path:
    """Return the default directory for generated station fulllist CSV files.

    Keep generated station lists outside OPENBENCH_REF_ROOT so users can scan
    read-only shared reference roots without needing write permission there.
    OPENBENCH_HOME overrides the base directory via resolve_home_dir(); when it
    is unset, this falls back to the real user home directory.
    """
    return resolve_home_dir() / "station_lists"


def _profile_root(ref_root: Path, root_sub_dir: str, profile_name: str) -> Path | None:
    rel = Path(root_sub_dir)
    if rel.is_absolute():
        logger.warning(
            "Invalid absolute root_sub_dir %r in profile %r; paths must be relative to ref_root",
            root_sub_dir,
            profile_name,
        )
        return None
    root = ref_root / rel
    if not _is_within(root, ref_root):
        logger.warning(
            "Invalid root_sub_dir %r in profile %r; path escapes ref_root",
            root_sub_dir,
            profile_name,
        )
        return None
    return root


def _find_grid_composite_nc_dir(container: Path) -> tuple[Path | None, int, str]:
    """Find NC files for Grid/Composite/<dataset> layouts.

    Supported composite-specific layouts:
      - Composite/<dataset>/*.nc
      - Composite/<dataset>/dataset/*.nc
      - Composite/<dataset>/data/*.nc

    Other shapes fall back to the normal <category>/<variable>/<dataset>
    scanner path.
    """
    direct_count = _count_nc(container)
    if direct_count > 0:
        return (container, direct_count, "found")

    for child_name in ("dataset", "data"):
        child = container / child_name
        if child.is_dir():
            return _find_nc_dir_with_descent(child, max_descent=DEFAULT_NC_DESCENT)

    nc_children = []
    for child in _iter_dirs(container):
        _nc_dir, _nc_count, status = _find_nc_dir_with_descent(
            child,
            max_descent=DEFAULT_NC_DESCENT,
        )
        if status == "ambiguous":
            return (None, 0, "ambiguous")
        if status == "found":
            nc_children.append(child)
    if len(nc_children) > 1:
        return (None, 0, "ambiguous")

    return (None, 0, "missing")


def _composite_children_match_parent_variable(container: Path, parent_var_name: str) -> bool:
    """Heuristic for Composite/<variable>/<dataset> standard layouts.

    Composite raw source roots and standard Composite category layouts share the
    same depth. If every NC-bearing child exposes the parent directory name as a
    data variable, treat children as datasets under that variable.
    """
    found = False
    for child in _iter_dirs(container):
        nc_dir, _count, status = _find_nc_dir_with_descent(
            child,
            max_descent=DEFAULT_NC_DESCENT,
        )
        if status != "found" or nc_dir is None:
            continue
        found = True
        info = _inspect_nc_file(nc_dir)
        data_names = {var.get("name") for var in info.get("all_data_vars", [])}
        if parent_var_name not in data_names:
            return False
    return found


def _is_within(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or below parent."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _is_profile_consumed(path: Path, consumed_dirs: set[Path]) -> bool:
    """Return True if a profile-driven scan already owns this subtree."""
    return any(_is_within(path, consumed) for consumed in consumed_dirs)


def _add_scanned_variant(groups: dict[str, DatasetGroup], scanned: ScannedDataset) -> None:
    """Add a scanned variant to groups, preserving first profile-owned match."""
    if scanned.name not in groups:
        groups[scanned.name] = DatasetGroup(base_name=scanned.name)
    existing = groups[scanned.name].variants.get(scanned.resolution)
    if existing is not None:
        logger.warning(
            "Duplicate scanned dataset variant for %s/%s; keeping first root %s and ignoring %s",
            scanned.name,
            scanned.resolution,
            existing.root_dir,
            scanned.root_dir,
        )
        return
    groups[scanned.name].variants[scanned.resolution] = scanned


def _child_dir_case_insensitive(parent: Path, name: str) -> Path:
    """Return an exact child path or a unique child whose name differs only by case."""

    exact = parent / name
    if exact.exists():
        return exact
    try:
        matches = [child for child in parent.iterdir() if child.is_dir() and child.name.casefold() == name.casefold()]
    except OSError:
        return exact
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(str(match) for match in matches)
        logger.warning("Ambiguous directory name %r under %s: %s", name, parent, choices)
        raise AmbiguousNameError(f"directory '{name}' under {parent} is ambiguous ignoring case: {choices}")
    return exact


def _category_label(name: str) -> str:
    for key, value in CATEGORY_MAP.items():
        if key.casefold() == str(name).casefold():
            return value
    return name


def _resolution_label(name: str) -> str | None:
    for key in RESOLUTION_MAP:
        if key.casefold() == str(name).casefold():
            return key
    return None


def _profile_scan_specs():
    """Yield reference profiles that opt into scan-time layout handling."""
    for profile_name, profile in _load_reference_profiles().items():
        if not isinstance(profile, dict):
            continue
        scan = profile.get("scan")
        if isinstance(scan, dict) and scan.get("layout"):
            yield profile_name, profile, scan


def _profile_category_from_subdir(root_sub_dir: str, default: str = "Other") -> str:
    """Infer OpenBench category metadata from a profile root_sub_dir."""
    parts = Path(root_sub_dir).parts
    if len(parts) >= 2 and parts[0].casefold() == "station":
        return _category_label(parts[1])
    if len(parts) >= 3 and parts[0].casefold() == "grid":
        return _category_label(parts[2])
    return default


def _grid_resolution_from_subdir(root_sub_dir: str) -> str | None:
    parts = Path(root_sub_dir).parts
    if len(parts) >= 2 and parts[0].casefold() == "grid":
        return _resolution_label(parts[1])
    return None


def _matching_nc_files(directory: Path, pattern: str | list[str] | tuple[str, ...] | None) -> list[Path]:
    """Return NC files in directory matching one or more profile glob patterns."""
    if not directory.is_dir():
        return []

    patterns: list[str]
    if isinstance(pattern, (list, tuple)):
        patterns = [str(p) for p in pattern]
    elif pattern:
        patterns = [str(pattern)]
    else:
        patterns = ["*.nc", "*.nc4"]

    files: dict[Path, None] = {}
    for pat in patterns:
        for file_path in directory.glob(pat):
            if file_path.is_file() and file_path.suffix in {".nc", ".nc4"}:
                files[file_path] = None
    return sorted(files)


def _profile_var_names(profile: dict, fallback: str) -> list[str]:
    profile_vars = profile.get("variables", {})
    if isinstance(profile_vars, dict) and profile_vars:
        return list(profile_vars)
    return [fallback]


def _profile_dataset_name(profile_name: str, profile: dict) -> str:
    """Dataset name produced by a profile.

    The profile key remains the unique configuration name. ``dataset_name``
    lets split profiles such as MyComposite_LowRes/MyComposite_MidRes register
    as the same dataset with separate resolution variants.
    """
    value = profile.get("dataset_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return profile_name


def _scan_station_profile_layout(
    ref_root: Path,
    profile_name: str,
    profile: dict,
    scan: dict,
) -> tuple[ScannedDataset | None, set[Path]]:
    """Build a station ScannedDataset from a profile-owned directory."""
    root_sub_dir = scan.get("root_sub_dir")
    if not root_sub_dir:
        return None, set()

    dataset_root = _profile_root(ref_root, str(root_sub_dir), profile_name)
    if dataset_root is None:
        return None, set()
    nc_sub_dir = str(scan.get("nc_sub_dir") or "")
    nc_dir = dataset_root / nc_sub_dir if nc_sub_dir else dataset_root
    files = _matching_nc_files(nc_dir, scan.get("file_glob"))
    if not files:
        return None, set()

    variables = {vn: nc_sub_dir for vn in _profile_var_names(profile, profile_name)}
    file_glob = scan.get("file_glob")
    file_globs = {vn: file_glob for vn in variables} if file_glob else {}
    scanned = ScannedDataset(
        name=_profile_dataset_name(profile_name, profile),
        resolution="Station",
        category=profile.get("category") or _profile_category_from_subdir(root_sub_dir),
        data_type="stn",
        root_dir=_portable_root_dir(dataset_root),
        variables=variables,
        file_globs=file_globs,
        file_count=len(files),
        tim_res=profile.get("tim_res") or _detect_tim_res(nc_dir, file_glob=file_glob),
    )
    should_consume = scan.get("consume", True)
    consumed = (
        {dataset_root} if scan.get("layout") in {"station_direct", "station_shared_files"} and should_consume else set()
    )
    return scanned, consumed


def _scan_grid_composite_children_profile(
    ref_root: Path,
    profile_name: str,
    profile: dict,
    scan: dict,
) -> tuple[ScannedDataset | None, set[Path]]:
    """Aggregate Composite/<dataset>/<child> directories per profile variables."""
    root_sub_dir = scan.get("root_sub_dir")
    if not root_sub_dir:
        return None, set()

    res_name = _grid_resolution_from_subdir(root_sub_dir)
    if not res_name:
        return None, set()

    composite_root = _profile_root(ref_root, str(root_sub_dir), profile_name)
    if composite_root is None:
        return None, set()
    res_dir = ref_root / "Grid" / res_name
    profile_vars = profile.get("variables", {})
    if not isinstance(profile_vars, dict) or not profile_vars:
        return None, set()

    variables: dict[str, str] = {}
    file_globs: dict[str, str | list[str]] = {}
    consumed: set[Path] = {composite_root}
    file_count = 0
    detected_tim_res = ""
    for var_name, var_profile in profile_vars.items():
        if not isinstance(var_profile, dict):
            continue
        child = var_profile.get("child")
        if not child:
            continue

        child_dir = composite_root / str(child)
        nc_dir, nc_count, status = _find_nc_dir_with_descent(
            child_dir,
            max_descent=DEFAULT_NC_DESCENT,
        )
        if status != "found" and var_profile.get("sub_dir"):
            fallback_dir = res_dir / str(var_profile["sub_dir"])
            nc_dir, nc_count, status = _find_nc_dir_with_descent(
                fallback_dir,
                max_descent=DEFAULT_NC_DESCENT,
            )
            if status == "found":
                consumed.add(fallback_dir)
        if status != "found" or nc_dir is None:
            continue

        variables[var_name] = nc_dir.relative_to(res_dir).as_posix()
        file_glob = var_profile.get("file_glob") or scan.get("file_glob")
        if file_glob:
            file_globs[var_name] = file_glob
        file_count += nc_count
        if not detected_tim_res:
            detected_tim_res = _detect_tim_res(nc_dir, file_glob=file_glob)

    if not variables:
        return None, set()

    scanned = ScannedDataset(
        name=_profile_dataset_name(profile_name, profile),
        resolution=res_name,
        category=profile.get("category") or _profile_category_from_subdir(root_sub_dir),
        data_type="grid",
        root_dir=_portable_root_dir(res_dir),
        variables=variables,
        file_globs=file_globs,
        file_count=file_count,
        tim_res=profile.get("tim_res") or detected_tim_res,
    )
    return scanned, consumed


def _scan_grid_composite_files_profile(
    ref_root: Path,
    profile_name: str,
    profile: dict,
    scan: dict,
    on_skip=None,
) -> tuple[ScannedDataset | None, set[Path]]:
    """Aggregate profile variables from file patterns under composite folders."""
    default_root_sub_dir = scan.get("root_sub_dir")
    profile_vars = profile.get("variables", {})
    if not isinstance(profile_vars, dict) or not profile_vars:
        return None, set()

    res_name: str | None = None
    res_dir: Path | None = None
    variables: dict[str, str] = {}
    file_globs: dict[str, str | list[str]] = {}
    consumed: set[Path] = set()
    file_count = 0
    detected_tim_res = ""
    mixed_resolution = False

    for var_name, var_profile in profile_vars.items():
        if not isinstance(var_profile, dict):
            continue
        root_sub_dir = var_profile.get("root_sub_dir") or default_root_sub_dir
        if not root_sub_dir:
            continue
        var_res_name = _grid_resolution_from_subdir(root_sub_dir)
        if not var_res_name:
            continue
        if res_name is None:
            res_name = var_res_name
            res_dir = ref_root / "Grid" / res_name
        elif var_res_name != res_name:
            skip_path = ref_root / root_sub_dir
            consumed.add(skip_path)
            logger.warning(
                "Skipped profile '%s' variable '%s': mixed grid resolutions in one profile",
                profile_name,
                var_name,
            )
            _notify_skip(
                on_skip,
                _scan_skip(
                    ref_root,
                    skip_path,
                    "mixed_grid_resolutions_in_profile",
                    "Split this profile by resolution or register the variable manually.",
                ),
            )
            mixed_resolution = True
            continue

        nc_dir = _profile_root(ref_root, str(root_sub_dir), profile_name)
        if nc_dir is None:
            return None, consumed
        files = _matching_nc_files(nc_dir, var_profile.get("file_glob") or scan.get("file_glob"))
        if not files:
            continue

        variables[var_name] = nc_dir.relative_to(res_dir).as_posix()
        file_glob = var_profile.get("file_glob") or scan.get("file_glob")
        if file_glob:
            file_globs[var_name] = file_glob
        consumed.add(nc_dir)
        file_count += len(files)
        if not detected_tim_res:
            detected_tim_res = _detect_tim_res(nc_dir, file_glob=file_glob)

    if mixed_resolution:
        return None, consumed

    if not variables or res_name is None or res_dir is None:
        return None, set()

    if default_root_sub_dir:
        consumed.add(ref_root / default_root_sub_dir)

    scanned = ScannedDataset(
        name=_profile_dataset_name(profile_name, profile),
        resolution=res_name,
        category=profile.get("category") or _profile_category_from_subdir(default_root_sub_dir or ""),
        data_type="grid",
        root_dir=_portable_root_dir(res_dir),
        variables=variables,
        file_globs=file_globs,
        file_count=file_count,
        tim_res=profile.get("tim_res") or detected_tim_res,
    )
    return scanned, consumed


def _scan_grid_nested_root_profile(
    ref_root: Path,
    profile_name: str,
    profile: dict,
    scan: dict,
    on_skip=None,
) -> tuple[ScannedDataset | None, set[Path]]:
    """Register a grid profile only when variables point at runnable NC dirs."""
    root_sub_dir = scan.get("root_sub_dir")
    if not root_sub_dir:
        return None, set()

    res_name = _grid_resolution_from_subdir(root_sub_dir)
    if not res_name:
        return None, set()

    dataset_root = _profile_root(ref_root, str(root_sub_dir), profile_name)
    if dataset_root is None:
        return None, set()
    if not dataset_root.is_dir():
        return None, set()

    from openbench.data.coordinates import glob_nc as _deep_glob

    nc_files = _deep_glob(dataset_root, recursive=True)
    if not nc_files:
        return None, set()

    res_dir = ref_root / "Grid" / res_name
    default_sub_dir = dataset_root.relative_to(res_dir).as_posix()
    variables: dict[str, str] = {}
    file_globs: dict[str, str | list[str]] = {}
    for var_name, var_profile in (profile.get("variables") or {}).items():
        file_glob = None
        if isinstance(var_profile, dict):
            sub_dir = str(var_profile.get("sub_dir") or default_sub_dir)
            file_glob = var_profile.get("file_glob")
        else:
            sub_dir = default_sub_dir
        candidate = res_dir / sub_dir
        files = _matching_nc_files(candidate, file_glob)
        if not files:
            nc_dir, _count, status = _find_nc_dir_with_descent(
                candidate,
                max_descent=DEFAULT_NC_DESCENT,
            )
            if status == "found" and nc_dir is not None:
                sub_dir = nc_dir.relative_to(res_dir).as_posix()
                files = _matching_nc_files(nc_dir, file_glob)
        if not files:
            logger.warning(
                "Skipped profile '%s': grid_nested_root variable '%s' does not point "
                "to a concrete NC-bearing directory",
                profile_name,
                var_name,
            )
            _notify_skip(
                on_skip,
                _scan_skip(
                    ref_root,
                    dataset_root,
                    "grid_nested_root_requires_concrete_subdir",
                    "Set each profile variable sub_dir to the directory that directly contains NC files.",
                ),
            )
            return None, {dataset_root}
        variables[var_name] = sub_dir
        if file_glob:
            file_globs[var_name] = file_glob
    if not variables:
        nc_dir, _count, status = _find_nc_dir_with_descent(
            dataset_root,
            max_descent=DEFAULT_NC_DESCENT,
        )
        if status != "found" or nc_dir is None:
            _notify_skip(
                on_skip,
                _scan_skip(
                    ref_root,
                    dataset_root,
                    "grid_nested_root_requires_concrete_subdir",
                    "Set a profile variable sub_dir to the directory that directly contains NC files.",
                ),
            )
            return None, {dataset_root}
        variables[profile_name] = nc_dir.relative_to(res_dir).as_posix()

    scanned = ScannedDataset(
        name=_profile_dataset_name(profile_name, profile),
        resolution=res_name,
        category=profile.get("category") or _profile_category_from_subdir(root_sub_dir),
        data_type="grid",
        root_dir=_portable_root_dir(res_dir),
        variables=variables,
        file_globs=file_globs,
        file_count=len(nc_files),
        tim_res=profile.get("tim_res") or "",
    )
    return scanned, {dataset_root}


def _scan_grid_dataset_choice_profile(
    ref_root: Path,
    profile_name: str,
    profile: dict,
    scan: dict,
    on_skip=None,
) -> tuple[ScannedDataset | None, set[Path]]:
    """Register one chosen NC-bearing child from an ambiguous grid dataset."""
    root_sub_dir = scan.get("root_sub_dir")
    nc_sub_dir = scan.get("nc_sub_dir") or scan.get("choice_sub_dir")
    if not root_sub_dir or not nc_sub_dir:
        return None, set()
    res_name = _grid_resolution_from_subdir(root_sub_dir)
    if not res_name:
        return None, set()
    dataset_root = _profile_root(ref_root, str(root_sub_dir), profile_name)
    if dataset_root is None or not dataset_root.is_dir():
        return None, set()
    res_dir = ref_root / "Grid" / res_name
    nc_dir = dataset_root / str(nc_sub_dir)
    files = _matching_nc_files(nc_dir, scan.get("file_glob"))
    if not files:
        found_dir, _count, status = _find_nc_dir_with_descent(
            nc_dir,
            max_descent=DEFAULT_NC_DESCENT,
        )
        if status == "found" and found_dir is not None:
            nc_dir = found_dir
            files = _matching_nc_files(nc_dir, scan.get("file_glob"))
    if not files:
        _notify_skip(
            on_skip,
            _scan_skip(
                ref_root,
                dataset_root,
                "grid_dataset_choice_missing_nc",
                "The chosen profile nc_sub_dir does not contain NetCDF files.",
            ),
        )
        return None, {dataset_root}

    variables = {}
    file_globs = {}
    default_var = Path(root_sub_dir).parts[3] if len(Path(root_sub_dir).parts) >= 5 else profile_name
    for var_name, var_profile in (profile.get("variables") or {default_var: {}}).items():
        variables[var_name] = nc_dir.relative_to(res_dir).as_posix()
        file_glob = scan.get("file_glob")
        if isinstance(var_profile, dict):
            file_glob = var_profile.get("file_glob") or file_glob
        if file_glob:
            file_globs[var_name] = file_glob

    scanned = ScannedDataset(
        name=_profile_dataset_name(profile_name, profile),
        resolution=res_name,
        category=profile.get("category") or _profile_category_from_subdir(root_sub_dir),
        data_type="grid",
        root_dir=_portable_root_dir(res_dir),
        variables=variables,
        file_globs=file_globs,
        file_count=len(files),
        tim_res=profile.get("tim_res") or _detect_tim_res(nc_dir, file_glob=scan.get("file_glob")),
    )
    return scanned, {dataset_root}


def _scan_ignore_profile(ref_root: Path, profile_name: str, scan: dict) -> tuple[None, set[Path]]:
    """Mark a profile root as consumed without producing a catalog entry."""
    root_sub_dir = scan.get("root_sub_dir")
    if not root_sub_dir:
        return None, set()
    ignored_root = _profile_root(ref_root, str(root_sub_dir), profile_name)
    if ignored_root is None:
        return None, set()
    if not ignored_root.exists():
        return None, set()
    return None, {ignored_root}


def _scan_profile_layouts(ref_root: Path, on_skip=None) -> tuple[dict[str, DatasetGroup], set[Path]]:
    """Scan profile-declared layouts before generic directory walking."""
    groups: dict[str, DatasetGroup] = {}
    consumed_dirs: set[Path] = set()

    for profile_name, profile, scan in _profile_scan_specs():
        layout = scan.get("layout")
        scanned = None
        consumed: set[Path] = set()

        if layout in {"station_direct", "station_shared_files"}:
            scanned, consumed = _scan_station_profile_layout(ref_root, profile_name, profile, scan)
        elif layout == "grid_composite_children":
            scanned, consumed = _scan_grid_composite_children_profile(ref_root, profile_name, profile, scan)
        elif layout == "grid_composite_files":
            scanned, consumed = _scan_grid_composite_files_profile(
                ref_root, profile_name, profile, scan, on_skip=on_skip
            )
        elif layout == "grid_nested_root":
            scanned, consumed = _scan_grid_nested_root_profile(ref_root, profile_name, profile, scan, on_skip=on_skip)
        elif layout == "grid_dataset_choice":
            scanned, consumed = _scan_grid_dataset_choice_profile(
                ref_root, profile_name, profile, scan, on_skip=on_skip
            )
        elif layout == "ignore":
            scanned, consumed = _scan_ignore_profile(ref_root, profile_name, scan)
        else:
            logger.warning("Unknown scan layout %r in profile %r", layout, profile_name)

        consumed_dirs.update(consumed)
        if scanned is None:
            continue
        _add_scanned_variant(groups, scanned)

    return groups, consumed_dirs


def scan_reference_directory(ref_root: str | Path, on_progress=None, on_skip=None) -> list[DatasetGroup]:
    """Scan a reference data directory and discover all datasets.

    Args:
        ref_root: Root directory (e.g., /Volumes/work/Reference)
        on_progress: Optional callback(message: str) for progress updates.
        on_skip: Optional callback(ScanSkip) for unsupported folders that
            scanner saw but intentionally did not auto-register.

    Returns:
        List of DatasetGroup, each containing resolution variants.
    """
    ref_root = Path(ref_root)
    if not ref_root.exists():
        logger.warning("Reference directory not found: %s", ref_root)
        return []

    groups, consumed_dirs = _scan_profile_layouts(ref_root, on_skip=on_skip)

    # Scan grid data: Grid/{Res}/<Category>/<Variable>/<Dataset>/*.nc
    # Walk 3 levels of directories. Composite category is scanned too:
    #   - Composite/<Dataset>/{dataset,data}/*.nc → profile-friendly dataset root
    #   - Composite/<Variable>/<Dataset>/*.nc     → normal grid registration
    # If a 3rd-level dir has NC files → standard dataset.
    # If not but its children do → dataset with sub-dirs (depth 4).
    grid_dir = _child_dir_case_insensitive(ref_root, "Grid")
    if grid_dir.exists():
        for res_name in ["LowRes", "MidRes", "HigRes"]:
            res_dir = _child_dir_case_insensitive(grid_dir, res_name)
            if not res_dir.exists():
                continue

            if on_progress:
                on_progress(f"Scanning Grid/{res_name}...")

            for category_dir in _iter_dirs(res_dir):
                cat_name = category_dir.name
                category = _category_label(cat_name)

                for var_dir in _iter_dirs(category_dir):
                    if _is_profile_consumed(var_dir, consumed_dirs):
                        continue

                    var_name = var_dir.name
                    if on_progress:
                        on_progress(f"  {res_name}/{cat_name}/{var_name}")

                    if cat_name.casefold() == "composite":
                        nc_dir, nc_count, status = _find_grid_composite_nc_dir(var_dir)
                        if status == "ambiguous":
                            if _composite_children_match_parent_variable(var_dir, var_name):
                                status = "missing"
                            else:
                                logger.warning(
                                    "Skipped '%s': multiple NC-bearing subdirectories. "
                                    "Composite/multi-variant datasets need a reference profile "
                                    "(reference_profiles.yaml) or manual 'openbench ref register'.",
                                    var_dir.relative_to(ref_root),
                                )
                                _notify_skip(
                                    on_skip,
                                    _scan_skip(
                                        ref_root,
                                        var_dir,
                                        "ambiguous_nc_subdirectories",
                                        "Add a reference profile for this composite layout or register it manually.",
                                    ),
                                )
                                continue
                        if status == "found":
                            dataset_name = var_name
                            tim_res = _detect_tim_res(nc_dir)
                            if dataset_name not in groups:
                                groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                            if res_name not in groups[dataset_name].variants:
                                groups[dataset_name].variants[res_name] = ScannedDataset(
                                    name=dataset_name,
                                    resolution=res_name,
                                    category=category,
                                    data_type="grid",
                                    root_dir=_portable_root_dir(nc_dir),
                                    tim_res=tim_res,
                                )

                            scanned = groups[dataset_name].variants[res_name]
                            if dataset_name not in scanned.variables:
                                scanned.variables[dataset_name] = ""
                                scanned.file_count += nc_count
                            continue

                    for dataset_dir in _iter_dirs(var_dir):
                        if _is_profile_consumed(dataset_dir, consumed_dirs):
                            continue

                        dataset_name = dataset_dir.name

                        # Locate the NC-bearing directory at depth ≤ 3:
                        #   level 0: dataset_dir/*.nc
                        #   level 1: dataset_dir/X/*.nc (single NC-bearing child)
                        #   level 2: dataset_dir/X/Y/*.nc (single chain to NCs)
                        #   level 3+: skip with deeper-glob warning
                        nc_dir, nc_count, status = _find_nc_dir_with_descent(
                            dataset_dir,
                            max_descent=DEFAULT_NC_DESCENT,
                        )

                        if status == "ambiguous":
                            logger.warning(
                                "Skipped '%s': multiple NC-bearing subdirectories. "
                                "Composite/multi-variant datasets need a reference profile "
                                "(reference_profiles.yaml) or manual 'openbench ref register'.",
                                dataset_dir.relative_to(ref_root),
                            )
                            _notify_skip(
                                on_skip,
                                _scan_skip(
                                    ref_root,
                                    dataset_dir,
                                    "ambiguous_nc_subdirectories",
                                    "Add a reference profile for this layout or register it manually.",
                                ),
                            )
                            continue

                        if status == "missing":
                            # Warn if deeper levels still have NC files (beyond supported depth)
                            from openbench.data.coordinates import glob_nc as _deep_glob

                            deep_nc = _deep_glob(dataset_dir, recursive=True)
                            if deep_nc:
                                logger.warning(
                                    "Skipped '%s': %d NC files found in deeper subdirectories "
                                    "(beyond supported 3-level depth). Move files up or register manually.",
                                    dataset_dir.relative_to(ref_root),
                                    len(deep_nc),
                                )
                                _notify_skip(
                                    on_skip,
                                    _scan_skip(
                                        ref_root,
                                        dataset_dir,
                                        "nc_files_too_deep",
                                        "Move NC files into a supported depth, add a profile, or register manually.",
                                    ),
                                )
                            continue

                        # Detect tim_res and record sub_dir from the actual NC location
                        tim_res = _detect_tim_res(nc_dir)
                        sub_dir = nc_dir.relative_to(res_dir).as_posix()

                        if dataset_name not in groups:
                            groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                        if res_name not in groups[dataset_name].variants:
                            groups[dataset_name].variants[res_name] = ScannedDataset(
                                name=dataset_name,
                                resolution=res_name,
                                category=category,
                                data_type="grid",
                                root_dir=_portable_root_dir(res_dir),
                                tim_res=tim_res,
                            )

                        scanned = groups[dataset_name].variants[res_name]
                        if var_name not in scanned.variables:
                            scanned.variables[var_name] = sub_dir
                            scanned.file_count += nc_count

    # Scan station data: Station/<category>/<variable>/<dataset>/
    # Also handles Composite layout: Station/Composite/<dataset>/dataset/*.nc
    stn_dir = _child_dir_case_insensitive(ref_root, "Station")
    if stn_dir.exists():
        if on_progress:
            on_progress("Scanning Station/...")
        for category_dir in _iter_dirs(stn_dir):
            cat_name = category_dir.name
            category = _category_label(cat_name)

            for var_dir in _iter_dirs(category_dir):
                if _is_profile_consumed(var_dir, consumed_dirs):
                    continue

                var_name = var_dir.name

                direct_nc_count = _count_nc(var_dir)
                if direct_nc_count > 0:
                    dataset_name = var_name
                    if dataset_name not in groups:
                        groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                    if "Station" not in groups[dataset_name].variants:
                        groups[dataset_name].variants["Station"] = ScannedDataset(
                            name=dataset_name,
                            resolution="Station",
                            category=category,
                            data_type="stn",
                            root_dir=_portable_root_dir(var_dir),
                            variables={dataset_name: ""},
                            file_count=direct_nc_count,
                            tim_res=_detect_tim_res(var_dir),
                        )
                    continue

                for dataset_dir in _iter_dirs(var_dir):
                    if _is_profile_consumed(dataset_dir, consumed_dirs):
                        continue

                    dataset_name = dataset_dir.name

                    # Use the same depth-3 descent helper as grid path:
                    #   level 0: dataset_dir/*.nc (most common stn layout)
                    #   level 1: dataset_dir/X/*.nc (e.g., dataset_dir/data/*.nc
                    #            or composite Composite/<X>/dataset/*.nc)
                    #   level 2: dataset_dir/X/Y/*.nc
                    # multi-child → ambiguous (previously silently merged
                    # year-subdir layouts with last-alphabetical wins)
                    nc_dir, nc_count, status = _find_nc_dir_with_descent(
                        dataset_dir,
                        max_descent=DEFAULT_NC_DESCENT,
                    )

                    if status == "ambiguous":
                        logger.warning(
                            "Skipped station '%s': multiple NC-bearing subdirectories. "
                            "Year-split or multi-variant station layouts need a reference "
                            "profile (reference_profiles.yaml) or manual "
                            "'openbench ref register'.",
                            dataset_dir.relative_to(ref_root),
                        )
                        _notify_skip(
                            on_skip,
                            _scan_skip(
                                ref_root,
                                dataset_dir,
                                "ambiguous_nc_subdirectories",
                                "Add a station reference profile or register it manually.",
                            ),
                        )
                        continue

                    if status == "missing":
                        from openbench.data.coordinates import glob_nc as _deep_glob

                        deep_nc = _deep_glob(dataset_dir, recursive=True)
                        if deep_nc:
                            logger.warning(
                                "Skipped station '%s': %d NC files found in deeper "
                                "subdirectories (beyond supported 3-level depth). "
                                "Move files up or register manually.",
                                dataset_dir.relative_to(ref_root),
                                len(deep_nc),
                            )
                            _notify_skip(
                                on_skip,
                                _scan_skip(
                                    ref_root,
                                    dataset_dir,
                                    "nc_files_too_deep",
                                    "Move NC files into a supported depth, add a profile, or register manually.",
                                ),
                            )
                        continue

                    # Composite layout: dataset_dir is literally named "dataset"
                    # or "data" — the real dataset name is var_name (one level up).
                    # Detection key is dataset_dir.name (kept for backward compat
                    # with reference roots that already use this convention).
                    is_composite = dataset_name in ("dataset", "data")
                    if is_composite:
                        dataset_name = var_name  # e.g., FLUXNET_PLUMBER2
                        ds_root = _portable_root_dir(nc_dir)  # points directly to NC dir
                    else:
                        ds_root = _portable_root_dir(category_dir.parent)

                    if dataset_name not in groups:
                        groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                    if "Station" not in groups[dataset_name].variants:
                        groups[dataset_name].variants["Station"] = ScannedDataset(
                            name=dataset_name,
                            resolution="Station",
                            category=category,
                            data_type="stn",
                            root_dir=ds_root,
                        )

                    scanned = groups[dataset_name].variants["Station"]
                    if is_composite:
                        # Placeholder: profile will supply real variable mappings
                        if dataset_name not in scanned.variables:
                            scanned.variables[dataset_name] = ""
                    else:
                        # Record the path of the actual NC-bearing dir relative
                        # to ds_root. Previously hardcoded to dataset_dir's path,
                        # which mismatched when NCs lived in a deeper subdir
                        # (e.g., MyStn/data/*.nc) — finalize couldn't locate them.
                        scanned.variables[var_name] = nc_dir.relative_to(category_dir.parent).as_posix()
                    scanned.file_count += nc_count

    return sorted(groups.values(), key=lambda g: g.base_name)


def find_new_datasets(
    ref_root: str | Path,
    existing_names: Optional[set[str]] = None,
    on_progress=None,
    on_skip=None,
) -> list[DatasetGroup]:
    """Scan and return only datasets not already registered.

    Args:
        ref_root: Reference data root directory.
        existing_names: Set of already registered dataset names.
        on_progress: Optional callback(message: str) for progress updates.
        on_skip: Optional callback(ScanSkip) for unsupported folders that
            scanner saw but intentionally did not auto-register.

    Returns:
        List of new DatasetGroup not in existing_names.
    """
    if existing_names is None:
        from openbench.data.registry.manager import RegistryManager

        mgr = RegistryManager()
        existing_names = {r.name for r in mgr.list_references()}

    raw_skips: list[ScanSkip] = []
    all_groups = scan_reference_directory(
        ref_root,
        on_progress=on_progress,
        on_skip=raw_skips.append if on_skip else None,
    )
    if on_skip:
        for skip in raw_skips:
            if _skip_candidate_registry_names(skip).isdisjoint(existing_names):
                on_skip(skip)
    new_groups = []

    # Filter at variant granularity, not group: when only some resolutions of
    # a dataset are new, return a group containing JUST the new variants.
    # Otherwise the CLI/GUI re-registers already-existing variants and silently
    # overwrites top-level descriptor fields (category, fulllist, _provenance,
    # etc.) that the user may have hand-edited.
    for group in all_groups:
        new_variants = {
            res: variant for res, variant in group.variants.items() if variant.registry_name not in existing_names
        }
        if new_variants:
            new_groups.append(
                DatasetGroup(
                    base_name=group.base_name,
                    variants=new_variants,
                )
            )

    return new_groups


def register_scanned_dataset(
    scanned: ScannedDataset,
    catalog_path: Optional[Path] = None,
    existing_descriptor: Optional[dict] = None,
    on_multi_var=None,
) -> Path:
    """Register a scanned dataset into the user catalog.

    Appends to the user's reference_catalog.yaml (single file, not individual files).

    Args:
        scanned: The scanned dataset to register.
        catalog_path: Path to the catalog YAML file.
            Defaults to the writable registry dir.
        existing_descriptor: Optional existing descriptor to merge with
            (preserves hand-edited fields like varname, varunit).
        on_multi_var: Optional callback when NC file has 2+ data variables.
            Called with (var_name, sub_dir, all_vars_list) → selected varname string.
            If None, first variable is used automatically.

    Returns:
        Path to the catalog file.
    """
    if catalog_path is None:
        from openbench.data.registry.manager import get_writable_reference_catalog_path

        catalog_path = get_writable_reference_catalog_path()

    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize across concurrent scan processes; load-modify-write inside lock
    with _catalog_write_lock(catalog_path):
        catalog = _safe_load_catalog(catalog_path)
        builtin_catalog = _load_builtin_catalog_for_user_overlay(catalog_path)
        base_descriptor = builtin_catalog.get(scanned.registry_name)
        existing = existing_descriptor
        if existing is None:
            existing = _merge_descriptor_overlay(
                base_descriptor,
                catalog.get(scanned.registry_name),
            )
        _register_to_dict(
            scanned,
            catalog,
            existing_descriptor=existing,
            on_multi_var=on_multi_var,
            station_list_dir=_station_list_dir_for_catalog(scanned, catalog_path),
        )
        _store_catalog_descriptor(
            catalog,
            scanned.registry_name,
            catalog[scanned.registry_name],
            base_descriptor,
        )
        _backup_then_write(catalog_path, catalog)
        _invalidate_registry_caches()

    return catalog_path


def register_scanned_datasets_batch(
    datasets: list,
    catalog_path: Optional[Path] = None,
    on_multi_var=None,
    on_progress=None,
) -> Path:
    """Register multiple scanned datasets in one pass (avoids O(n²) YAML I/O).

    Args:
        datasets: List of ScannedDataset objects.
        catalog_path: Path to catalog YAML. Defaults to writable registry dir.
        on_multi_var: Optional callback for multi-variable NC selection.
        on_progress: Optional callback(str) for progress updates.

    Returns:
        Path to the catalog file.
    """
    if catalog_path is None:
        from openbench.data.registry.manager import get_writable_reference_catalog_path

        catalog_path = get_writable_reference_catalog_path()

    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize across concurrent scan processes; load-modify-write inside lock
    with _catalog_write_lock(catalog_path):
        catalog = _safe_load_catalog(catalog_path)
        builtin_catalog = _load_builtin_catalog_for_user_overlay(catalog_path)

        # Build all descriptors (this is where NC inspection happens)
        for i, scanned in enumerate(datasets):
            if on_progress:
                on_progress(f"  [{i + 1}/{len(datasets)}] {scanned.registry_name}")

            # Merge with existing descriptor to preserve hand-edited fields
            base_descriptor = builtin_catalog.get(scanned.registry_name)
            existing = _merge_descriptor_overlay(
                base_descriptor,
                catalog.get(scanned.registry_name),
            )
            _register_to_dict(
                scanned,
                catalog,
                existing_descriptor=existing,
                on_multi_var=on_multi_var,
                station_list_dir=_station_list_dir_for_catalog(scanned, catalog_path),
            )
            _store_catalog_descriptor(
                catalog,
                scanned.registry_name,
                catalog[scanned.registry_name],
                base_descriptor,
            )

        # Write once (atomic, with backup of previous state)
        _backup_then_write(catalog_path, catalog)
        _invalidate_registry_caches()

    return catalog_path


def _register_to_dict(
    scanned,
    catalog: dict,
    existing_descriptor: Optional[dict] = None,
    on_multi_var=None,
    station_list_dir: Optional[Path] = None,
) -> None:
    """Build descriptor and add to catalog dict (no file I/O except fulllist).

    Stages:
      1. _build_base_descriptor    — base fields + data_groupby from file counts
      2. _build_variables          — per-variable entries (existing merge or NC inspect)
      3. _apply_nc_corrections     — data_type / tim_res / grid_res from NC findings
      4. _apply_profile_overrides  — reference_profiles.yaml metadata + variables
      5. _finalize_descriptor      — grid_res default, provenance, station fulllist
    """
    prov: dict[str, str] = {}

    # Stage 1: base descriptor
    descriptor = _build_base_descriptor(scanned, prov)

    # Stage 2: build variable entries
    variables = _build_variables(scanned, descriptor, existing_descriptor, on_multi_var)

    # Stage 3: apply NC-detected corrections
    _apply_nc_corrections(scanned, descriptor, variables, prov)

    # Stage 4: apply reference profile overrides
    variables = _apply_profile_overrides(scanned, descriptor, variables, prov, existing_descriptor)

    # Stage 5: finalize (grid_res, provenance, fulllist)
    descriptor["variables"] = variables
    _finalize_descriptor(scanned, descriptor, prov, existing_descriptor, station_list_dir)

    # Stage 6: preserve user-edited descriptor-level fields (free-text /
    # categorization / years that scan can't authoritatively re-derive)
    _preserve_user_edits(descriptor, existing_descriptor)

    catalog[scanned.registry_name] = descriptor


def _preserve_user_edits(descriptor: dict, existing: dict | None) -> None:
    """Preserve user-edited descriptor-level fields when re-registering.

    Variable-level merge already happens in _build_variables. This handles
    descriptor-level fields that:
      - the user often hand-corrects (description, category, years), or
      - scanning can never authoritatively re-derive (free-text fields).

    fulllist is handled in _finalize_descriptor (it requires the file-exists
    check before deciding whether to preserve or regenerate).
    """
    if not existing:
        return

    # Fields where existing user edits ALWAYS win when present (truthy):
    #  - description: free text, never auto-derivable
    #  - category:    scanner picks from directory name, but user may
    #                 re-categorize (e.g., split Water → Water-Energy)
    user_owned = ("description", "category")
    for fld in user_owned:
        if fld in existing and existing.get(fld):
            descriptor[fld] = existing[fld]

    # years: prefer existing if present. Filename-based year extraction is
    # noisy (catches version numbers like "v2010"); the user's authoritative
    # fix should not be undone by a rescan.
    if existing.get("years"):
        if "years" in descriptor:
            descriptor["years"] = existing["years"]
        else:
            rebuilt = {}
            inserted = False
            for key, value in descriptor.items():
                rebuilt[key] = value
                if key == "root_dir":
                    rebuilt["years"] = existing["years"]
                    inserted = True
            if not inserted:
                rebuilt["years"] = existing["years"]
            descriptor.clear()
            descriptor.update(rebuilt)

    # timezone: present-key check (NOT truthy check) — value 0 is the most
    # common legitimate setting (UTC) and would be lost under truthy check.
    # Preserves explicit non-zero settings the user has hand-tuned (e.g.,
    # local-time station data needing offset). Stage 1 always writes 0;
    # without this, every rescan resets the field.
    if "timezone" in existing:
        descriptor["timezone"] = existing["timezone"]

    # station_matching is a Streamflow-only runtime contract. Keep valid
    # user-defined matchers, but drop invalid blocks from other station data.
    if existing.get("station_matching") and "Streamflow" in descriptor.get("variables", {}):
        descriptor["station_matching"] = existing["station_matching"]


# ---------------------------------------------------------------------------
# Stage 1: Build base descriptor
# ---------------------------------------------------------------------------


def _build_base_descriptor(scanned, prov: dict) -> dict:
    """Build the initial descriptor from scanned metadata.

    Detects data_groupby by inspecting filenames in each variable directory:
      - 1 NC file per variable    → Single
      - YYYY-MM-DD or YYYYMMDD    → Day
      - YYYY-MM or YYYYMM         → Month
      - YYYY only                 → Year
      - Mixed/unclear             → Year (most common safe default)
    """
    tim_res = scanned.tim_res or ""

    data_groupby = _detect_data_groupby(scanned)
    prov["data_groupby"] = "scan"

    return {
        "name": scanned.registry_name,
        "description": f"{scanned.name} reference dataset ({scanned.resolution})",
        "category": scanned.category,
        "data_type": scanned.data_type,
        "tim_res": tim_res,
        "data_groupby": data_groupby,
        "timezone": 0,
        "root_dir": scanned.root_dir,
    }


def _detect_data_groupby(scanned) -> str:
    """Detect data_groupby from per-variable file count + filename patterns.

    Returns one of: "Single" | "Day" | "Month" | "Year".
    The previous implementation only ever returned "Year" or "Single",
    breaking processing.py's monthly/daily file iteration when descriptor
    said "Year" but filenames were YYYYMM.
    """
    injected = getattr(scanned, "detected_data_groupby", "")
    if injected:
        # Remote-scanned dataset: the filesystem walk below cannot see the
        # remote files, so trust the value computed on the remote host.
        return injected
    if not scanned.variables:
        return "Year"

    # Collect NC file lists per variable
    var_files: list[list[Path]] = []
    for vn, sub_dir in scanned.variables.items():
        var_path = _expand_path(scanned.root_dir) / sub_dir
        if not var_path.is_dir():
            continue
        file_glob = getattr(scanned, "file_globs", {}).get(vn)
        files = _matching_nc_files(var_path, file_glob) if file_glob else _glob_nc(var_path)
        if not files:
            for child in _iter_dirs(var_path):
                child_files = _matching_nc_files(child, file_glob) if file_glob else _glob_nc(child)
                if child_files:
                    files = child_files
                    break
        if files:
            var_files.append(files)

    if not var_files:
        return "Year"

    # Single file per variable across the board → Single
    if all(len(files) == 1 for files in var_files):
        return "Single"

    # Tally date pattern frequency across all files (multi-variable datasets
    # vote together to handle slight per-variable inconsistencies)
    counts = {"day": 0, "month": 0, "year": 0, "none": 0}
    total = 0
    for files in var_files:
        for f in files:
            counts[_classify_filename_date(f.stem)] += 1
            total += 1

    # Plurality wins among day/month/year. Tiebreak prefers more granular
    # (day > month > year) since over-stating granularity is safer than under.
    if counts["day"] >= counts["month"] and counts["day"] >= counts["year"] and counts["day"] > 0:
        return "Day"
    if counts["month"] >= counts["year"] and counts["month"] > 0:
        return "Month"
    if counts["year"] > 0:
        return "Year"

    # No date markers anywhere — could be a single-file-per-var dataset that
    # missed the all-1 check (e.g., 1 file in some vars, 2 in others without
    # date patterns). Default to Year to match historical behavior.
    return "Year"


# ---------------------------------------------------------------------------
# Stage 2: Build variable entries (merge existing or inspect NC)
# ---------------------------------------------------------------------------


def _build_variables(scanned, descriptor: dict, existing_descriptor, on_multi_var) -> dict:
    """Build per-variable entries.

    For each scanned variable, inspect the NC file for technical metadata
    while preserving existing user-edited variable fields.
    NC-detected fields (_detected_data_type, _nc_tim_res, _nc_grid_res) are
    stored as temporary keys on var entries for Stage 3 to consume.
    """
    variables = {}
    profile = get_reference_profile(scanned.registry_name) or get_reference_profile(scanned.name)
    profile_vars = (
        profile.get("variables", {}) if isinstance(profile, dict) and isinstance(profile.get("variables"), dict) else {}
    )
    profile_replaces_placeholder_variables = bool(profile_vars) and not any(
        vn in profile_vars for vn in scanned.variables
    )
    for var_name, sub_dir in scanned.variables.items():
        var_entry: dict = {"varname": var_name, "varunit": ""}
        if sub_dir:
            var_entry["sub_dir"] = sub_dir
        suppress_existing_filename_pattern = False

        existing_var = None
        if existing_descriptor:
            existing_vars = existing_descriptor.get("variables", {})
            if var_name in existing_vars:
                existing_var = existing_vars[var_name]
        profile_var = profile_vars.get(var_name)
        profile_defines_varname = isinstance(profile_var, dict) and bool(profile_var.get("varname"))
        suppress_multi_var_prompt = profile_defines_varname or profile_replaces_placeholder_variables

        dataset_path = _expand_path(scanned.root_dir) / sub_dir
        # Remote-scanned datasets ship inspection results computed where the
        # data lives; those are authoritative even when the remote root_dir
        # string happens to exist locally too (shared mounts, coincidences).
        nc_info = (getattr(scanned, "nc_inspections", None) or {}).get(var_name)
        if nc_info is None and dataset_path.is_dir():
            file_glob = getattr(scanned, "file_globs", {}).get(var_name)
            nc_info = _inspect_nc_file(dataset_path, file_glob=file_glob)
        if nc_info is not None:
            if nc_info.get("inspection_failed") and existing_var is None and not suppress_multi_var_prompt:
                raise RuntimeError(
                    f"Failed to inspect NetCDF files for {scanned.registry_name}:{var_name} "
                    f"under {dataset_path}: {nc_info['inspection_failed']}"
                )

            all_vars = nc_info.get("all_data_vars", [])
            if len(all_vars) > 1 and on_multi_var and existing_var is None and not suppress_multi_var_prompt:
                chosen = on_multi_var(var_name, sub_dir, all_vars)
                if chosen:
                    nc_info["varname"] = chosen
                    for av in all_vars:
                        if av["name"] == chosen:
                            nc_info["varunit"] = av["unit"]
                            break

            if existing_var is None:
                if nc_info.get("varname"):
                    var_entry["varname"] = nc_info["varname"]
                if nc_info.get("varunit"):
                    var_entry["varunit"] = nc_info["varunit"]
                is_multi_file_station = (
                    scanned.data_type == "stn" or nc_info.get("detected_data_type") == "stn"
                ) and nc_info.get("nc_file_count", 0) > 1
                infer_filename_pattern = not is_multi_file_station
                suppress_existing_filename_pattern = not infer_filename_pattern
                if infer_filename_pattern and nc_info.get("prefix"):
                    var_entry["prefix"] = nc_info["prefix"]
                if infer_filename_pattern and nc_info.get("suffix"):
                    var_entry["suffix"] = nc_info["suffix"]
            else:
                suppress_existing_filename_pattern = (
                    scanned.data_type == "stn" or nc_info.get("detected_data_type") == "stn"
                ) and nc_info.get("nc_file_count", 0) > 1

            if nc_info.get("syear"):
                nc_years = [nc_info["syear"], nc_info.get("eyear", nc_info["syear"])]
                existing_years = descriptor.get("years")
                if isinstance(existing_years, list) and len(existing_years) == 2:
                    descriptor["years"] = [min(existing_years[0], nc_years[0]), max(existing_years[1], nc_years[1])]
                else:
                    descriptor["years"] = nc_years
            if nc_info.get("detected_data_type"):
                var_entry["_detected_data_type"] = nc_info["detected_data_type"]
            if nc_info.get("inconsistent_data_type"):
                var_entry["_inconsistent_data_type"] = nc_info["inconsistent_data_type"]
            if nc_info.get("detected_tim_res"):
                var_entry["_nc_tim_res"] = nc_info["detected_tim_res"]
            if nc_info.get("inconsistent_tim_res"):
                var_entry["_inconsistent_tim_res"] = nc_info["inconsistent_tim_res"]
            if nc_info.get("detected_grid_res"):
                var_entry["_nc_grid_res"] = nc_info["detected_grid_res"]
            if nc_info.get("inconsistent_grid_res"):
                var_entry["_inconsistent_grid_res"] = nc_info["inconsistent_grid_res"]
            if nc_info.get("climatology_candidate"):
                var_entry["_climatology_candidate"] = nc_info["climatology_candidate"]

        if existing_var is not None:
            var_entry["varname"] = existing_var.get("varname", var_name)
            var_entry["varunit"] = existing_var.get("varunit", "")
            for key, value in existing_var.items():
                if key in {"varname", "varunit", "sub_dir"}:
                    continue
                if key in _SCANNER_TEMP_VAR_KEYS or str(key).startswith("_"):
                    continue
                if key in {"prefix", "suffix"} and suppress_existing_filename_pattern:
                    continue
                if value is not None and value != "":
                    var_entry[key] = value

        variables[var_name] = var_entry

    return variables


# ---------------------------------------------------------------------------
# Stage 3: Apply NC-detected corrections (data_type, tim_res, grid_res)
# ---------------------------------------------------------------------------


def _apply_nc_corrections(scanned, descriptor: dict, variables: dict, prov: dict) -> None:
    """Consume temporary NC fields from variable entries and update descriptor.

    Priority for tim_res: nc > scan > default.
    NC-detected data_type overrides directory-inferred data_type.
    NC-detected grid_res is recorded directly.
    """
    # Validate data_type against NC content (first variable with detection result)
    # while removing scratch fields from every variable entry before persisting.
    detected_data_type = None
    data_type_values = []
    inconsistent_data_type = set()
    for _vn, v in variables.items():
        nc_dtype = v.pop("_detected_data_type", None) if isinstance(v, dict) else None
        nc_dtype_inconsistent = v.pop("_inconsistent_data_type", None) if isinstance(v, dict) else None
        if nc_dtype and detected_data_type is None:
            detected_data_type = nc_dtype
        if nc_dtype:
            data_type_values.append(nc_dtype)
        if nc_dtype_inconsistent:
            inconsistent_data_type.update(nc_dtype_inconsistent)

    all_data_types = set(data_type_values) | inconsistent_data_type
    if len(all_data_types) > 1:
        logger.warning(
            "Dataset '%s': variables/files have inconsistent data_type values: %s. Using first detected value '%s'.",
            scanned.registry_name,
            sorted(all_data_types),
            detected_data_type,
        )

    if detected_data_type and detected_data_type != descriptor["data_type"]:
        logger.warning(
            "Dataset '%s': directory says data_type='%s' but NC content looks like '%s'. Using NC-detected type.",
            scanned.registry_name,
            descriptor["data_type"],
            detected_data_type,
        )
        descriptor["data_type"] = detected_data_type
        prov["data_type"] = "nc"

    # Collect NC-detected tim_res and grid_res from first inspected variable
    nc_tim_res = None
    nc_grid_res = None
    climatology_candidate = None
    tim_res_values = []
    grid_res_values = []
    inconsistent_tim_res = set()
    inconsistent_grid_res = set()
    for _vn, v in variables.items():
        if isinstance(v, dict):
            detected_tim_res = v.pop("_nc_tim_res", None)
            detected_grid_res = v.pop("_nc_grid_res", None)
            detected_climatology_candidate = v.pop("_climatology_candidate", None)
            detected_inconsistent_tim = v.pop("_inconsistent_tim_res", None)
            detected_inconsistent_grid = v.pop("_inconsistent_grid_res", None)
            if nc_tim_res is None and detected_tim_res:
                nc_tim_res = detected_tim_res
            if nc_grid_res is None and detected_grid_res:
                nc_grid_res = detected_grid_res
            if climatology_candidate is None and detected_climatology_candidate:
                climatology_candidate = detected_climatology_candidate
            if detected_tim_res:
                tim_res_values.append(detected_tim_res)
            if detected_grid_res:
                grid_res_values.append(detected_grid_res)
            if detected_inconsistent_tim:
                inconsistent_tim_res.update(detected_inconsistent_tim)
            if detected_inconsistent_grid:
                inconsistent_grid_res.update(detected_inconsistent_grid)

    all_tim_res = set(tim_res_values) | inconsistent_tim_res
    if len(all_tim_res) > 1:
        logger.warning(
            "Dataset '%s': variables/files have inconsistent tim_res values: %s. Using first detected value '%s'.",
            scanned.registry_name,
            sorted(all_tim_res),
            nc_tim_res,
        )
    all_grid_res = {float(v) for v in grid_res_values} | {float(v) for v in inconsistent_grid_res}
    if len(all_grid_res) > 1:
        logger.warning(
            "Dataset '%s': variables/files have inconsistent grid_res values: %s. Using first detected value '%s'.",
            scanned.registry_name,
            sorted(all_grid_res),
            nc_grid_res,
        )

    # tim_res priority: nc > scan > default
    scan_tim_res = descriptor["tim_res"]  # from scanned.tim_res via Stage 1
    if nc_tim_res:
        if scan_tim_res and _tim_res_rank(nc_tim_res) != _tim_res_rank(scan_tim_res):
            logger.warning(
                "Dataset '%s': filename/path suggests tim_res='%s' but NC "
                "time coordinate suggests '%s'. Using NC-detected tim_res.",
                scanned.registry_name,
                scan_tim_res,
                nc_tim_res,
            )
        descriptor["tim_res"] = nc_tim_res
        prov["tim_res"] = "nc"
    elif scan_tim_res:
        descriptor["tim_res"] = scan_tim_res
        prov["tim_res"] = "scan"
    else:
        descriptor["tim_res"] = "Month"
        prov["tim_res"] = "default"

    if climatology_candidate:
        logger.warning(
            "Dataset '%s': NC time coordinate is a %s candidate. Keeping "
            "tim_res='%s'; use explicit confirmation before registering it as climatology.",
            scanned.registry_name,
            climatology_candidate,
            descriptor["tim_res"],
        )

    if nc_grid_res:
        res_info = RESOLUTION_MAP.get(scanned.resolution, {})
        bucket_grid_res = res_info.get("typical_grid_res")
        if bucket_grid_res is not None and abs(float(nc_grid_res) - float(bucket_grid_res)) > 1e-6:
            logger.warning(
                "Dataset '%s': resolution bucket '%s' implies grid_res=%s "
                "but NC coordinates suggest %s. Using NC-detected grid_res.",
                scanned.registry_name,
                scanned.resolution,
                bucket_grid_res,
                nc_grid_res,
            )
        descriptor["grid_res"] = nc_grid_res
        prov["grid_res"] = "nc"


# ---------------------------------------------------------------------------
# Stage 4: Apply reference profile overrides
# ---------------------------------------------------------------------------

_PROFILE_VAR_SCAN_KEYS = {"child", "file_glob", "root_sub_dir", "scan", "layout"}


def _merge_profile_var_fields(
    target: dict,
    profile_var: dict,
    fallback_varname: str,
    overwrite_existing: bool = True,
) -> None:
    """Copy descriptor-relevant variable fields from a reference profile."""

    def set_field(key: str, value) -> None:
        if value is None:
            return
        if overwrite_existing or target.get(key) in (None, ""):
            target[key] = value

    set_field("varname", profile_var.get("varname", fallback_varname))
    set_field("varunit", profile_var.get("varunit", ""))

    for key, value in profile_var.items():
        if key in _PROFILE_VAR_SCAN_KEYS:
            continue
        set_field(key, value)


def _apply_profile_overrides(
    scanned,
    descriptor: dict,
    variables: dict,
    prov: dict,
    existing_descriptor: dict | None = None,
) -> dict:
    """Apply reference_profiles.yaml overrides to descriptor and variables.

    Profile override rules:
      - tim_res:      profile overrides scan/default, but NOT nc-detected
      - data_groupby: profile always overrides
      - variables:    standard datasets get varname/varunit merged;
                      composite datasets get variables replaced entirely

    Returns the (possibly replaced) variables dict.
    """
    profile = get_reference_profile(scanned.registry_name) or get_reference_profile(scanned.name)
    if profile:
        profile_vars = profile.get("variables", {})

        # Override metadata from profile (authoritative, except nc-detected tim_res)
        if profile.get("description"):
            descriptor["description"] = profile["description"]
        if profile.get("category"):
            descriptor["category"] = profile["category"]
        if profile.get("tim_res") and prov.get("tim_res") != "nc":
            descriptor["tim_res"] = profile["tim_res"]
            prov["tim_res"] = "profile"
        if profile.get("data_groupby"):
            descriptor["data_groupby"] = profile["data_groupby"]
            prov["data_groupby"] = "profile"
        if profile.get("fulllist"):
            descriptor["fulllist"] = profile["fulllist"]
        if profile.get("station_matching"):
            descriptor["station_matching"] = profile["station_matching"]
        if profile.get("years"):
            descriptor["years"] = profile["years"]

        if profile_vars:
            scanned_matches_profile = any(vn in profile_vars for vn in variables)

            if scanned_matches_profile:
                # Standard dataset: merge descriptor-relevant variable fields
                # from profile while keeping NC-derived fields for variables
                # that the profile does not mention.
                existing_vars = (
                    existing_descriptor.get("variables", {}) if isinstance(existing_descriptor, dict) else {}
                )
                for vn, pv in profile_vars.items():
                    if vn in variables:
                        existing_var = existing_vars.get(vn)
                        if isinstance(existing_var, dict):
                            for key, value in _public_existing_var_fields(existing_var).items():
                                if value is not None and value != "":
                                    variables[vn][key] = value
                        _merge_profile_var_fields(
                            variables[vn],
                            pv,
                            vn,
                            overwrite_existing=vn not in existing_vars,
                        )
                    else:
                        existing_var = existing_vars.get(vn)
                        if isinstance(existing_var, dict):
                            entry = _public_existing_var_fields(existing_var)
                            _merge_profile_var_fields(
                                entry,
                                pv,
                                vn,
                                overwrite_existing=False,
                            )
                            variables[vn] = entry
            else:
                # Composite dataset: scanned "variables" are directory names
                # Replace with profile variable mappings. If a previous catalog
                # already has the profile variable key, preserve its user/locator
                # fields and let profile metadata fill only missing values.
                existing_vars = (
                    existing_descriptor.get("variables", {}) if isinstance(existing_descriptor, dict) else {}
                )
                variables = {}
                for vn, pv in profile_vars.items():
                    existing_var = existing_vars.get(vn)
                    if isinstance(existing_var, dict):
                        entry = _public_existing_var_fields(existing_var)
                        _merge_profile_var_fields(
                            entry,
                            pv,
                            vn,
                            overwrite_existing=False,
                        )
                    else:
                        entry = {"varname": vn, "varunit": ""}
                        _merge_profile_var_fields(entry, pv, vn)
                    variables[vn] = entry
    else:
        # No profile found — warn for datasets with no variable mapping
        has_real_varnames = any(v.get("varname") and v["varname"] != var_name for var_name, v in variables.items())
        if not has_real_varnames and variables:
            logger.info(
                "No reference profile for '%s'. Register with:\n"
                '  openbench ref register-profile %s -v "VarName:ncname:unit"',
                scanned.name,
                scanned.name,
            )

    return variables


# ---------------------------------------------------------------------------
# Stage 5: Finalize descriptor (grid_res default, provenance, fulllist)
# ---------------------------------------------------------------------------


def _finalize_descriptor(
    scanned,
    descriptor: dict,
    prov: dict,
    existing_descriptor: dict | None = None,
    station_list_dir: Path | None = None,
) -> None:
    """Set data_type-dependent fields, record provenance, generate station fulllist.

    For station datasets: if an existing fulllist points to an extant file,
    preserve it (the user may have hand-filtered the station list). Only
    auto-generate when no valid existing fulllist is recorded.
    """
    final_dtype = descriptor["data_type"]

    if final_dtype == "grid" and "grid_res" not in descriptor:
        res_info = RESOLUTION_MAP.get(scanned.resolution, {})
        descriptor["grid_res"] = res_info.get("typical_grid_res", 0.25)
        prov["grid_res"] = "default"
    elif final_dtype == "stn":
        # Remove grid_res if data_type was corrected from grid to stn
        descriptor.pop("grid_res", None)
        prov.pop("grid_res", None)

    descriptor["_provenance"] = prov

    # Auto-generate fulllist for station datasets — UNLESS a profile already
    # supplied one, a profile supplied station_matching, or the user already
    # has a hand-edited fulllist pointing to an existing file. Existing
    # relative paths are interpreted relative to root_dir, matching the
    # register-profile CLI docs.
    if final_dtype == "stn":
        if descriptor.get("fulllist") or descriptor.get("station_matching"):
            return

        if _has_custom_station_filter(scanned.registry_name):
            return

        existing_fulllist = (existing_descriptor or {}).get("fulllist")
        if existing_fulllist and _fulllist_path_exists(
            existing_fulllist,
            descriptor.get("root_dir") or scanned.root_dir,
            (existing_descriptor or {}).get("root_dir"),
        ):
            descriptor["fulllist"] = existing_fulllist
            return

        # Remote-scanned dataset: the CSV was generated on the remote host
        # (where the station files live) and evaluation runs there too, so
        # the remote path is authoritative — even when a same-named local
        # directory with NC files exists (shared mounts).
        if getattr(scanned, "remote_fulllist", ""):
            descriptor["fulllist"] = scanned.remote_fulllist
            return

        nc_dir = resolve_station_nc_dir(scanned.root_dir, scanned.variables)
        if _glob_nc(nc_dir):
            try:
                if station_list_dir is None:
                    lists_dir = _expand_path(scanned.root_dir) / "station_lists"
                else:
                    lists_dir = Path(station_list_dir)
                lists_dir.mkdir(parents=True, exist_ok=True)
                output_csv = lists_dir / f"{scanned.registry_name}.csv"
                generate_station_list(nc_dir, output_csv)
                descriptor["fulllist"] = _portable_path(output_csv)
            except Exception as e:
                logger.warning("Failed to generate station list for %s: %s", scanned.name, e)


def resolve_station_nc_dir(root_dir: str, variables) -> Path:
    """Pick the directory holding a station dataset's NetCDF files.

    Tries the root, then the first variable's sub_dir, then that sub_dir's
    ``dataset`` child. Shared by local registration and the remote scanner
    so both agree on the station layout.
    """
    nc_dir = _expand_path(root_dir)
    if variables:
        first_sub = next(iter(variables.values()), "")
        candidate = _expand_path(root_dir) / first_sub
        if candidate.is_dir():
            if _glob_nc(candidate):
                nc_dir = candidate
            else:
                ds_sub = candidate / "dataset"
                if ds_sub.is_dir() and _glob_nc(ds_sub):
                    nc_dir = ds_sub
    return nc_dir


def _fulllist_path_exists(path_value: str, *roots: str | None) -> bool:
    """Check absolute or root_dir-relative fulllist paths."""
    path = _expand_path(path_value)
    if path.exists():
        return True
    if path.is_absolute():
        return False
    for root in roots:
        if root and (_expand_path(root) / path).exists():
            return True
    return False


# Frequency hierarchy: higher rank = higher frequency
# When a higher-frequency variant exists, lower-frequency variants are disabled.
_TIM_RES_RANK = {
    "climatology-year": 0,
    "climatology_year": 0,
    "year": 0,
    "yearly": 0,
    "y": 0,
    "climatology-month": 1,
    "climatology_month": 1,
    "month": 1,
    "monthly": 1,
    "m": 1,
    "mon": 1,
    "8day": 2,
    "8daily": 2,
    "week": 2,
    "weekly": 2,
    "w": 2,
    "day": 3,
    "daily": 3,
    "d": 3,
    "6hour": 4,
    "6h": 4,
    "6hourly": 4,
    "3hour": 5,
    "3h": 5,
    "3hourly": 5,
    "hour": 6,
    "hourly": 6,
    "h": 6,
    "30min": 7,
    "30mins": 7,
    "30minute": 7,
    "30minutes": 7,
    "halfhour": 7,
    "half-hour": 7,
}


def _tim_res_rank(tim_res: str) -> int:
    """Return the frequency rank for a time resolution string."""
    return _TIM_RES_RANK.get(tim_res.lower().strip(), -1) if tim_res else -1


def get_compatible_resolutions(
    group: DatasetGroup,
    required_tim_res: Optional[str] = None,
) -> list[str]:
    """Get resolutions compatible with a time resolution constraint.

    Rule: only the highest-frequency variant (and equal) are allowed.
    If hourly data exists, daily and monthly are disabled.
    If daily data exists, monthly is disabled. Etc.

    Args:
        group: DatasetGroup with resolution variants.
        required_tim_res: Optional hint (unused currently, kept for API compat).

    Returns:
        List of compatible resolution names.
    """
    if not group.variants:
        return []

    # Find the highest frequency rank among all variants
    max_rank = max(
        (_tim_res_rank(v.tim_res) for v in group.variants.values()),
        default=-1,
    )

    if max_rank <= 0:
        # No frequency info or only yearly — allow all
        return group.available_resolutions

    compatible = []
    for res_name, variant in group.variants.items():
        rank = _tim_res_rank(variant.tim_res)
        if rank >= max_rank:
            # Same or higher frequency → compatible
            compatible.append(res_name)

    return compatible


def _iter_dirs(path: Path):
    """Iterate over source-data subdirectories, skipping generated work dirs."""
    if not path.exists():
        return
    for item in sorted(path.iterdir()):
        if item.is_dir() and not item.name.startswith(".") and not _is_reference_scan_excluded_dir(item):
            yield item


def _is_reference_scan_excluded_dir(path: Path) -> bool:
    name = path.name
    lower_name = name.lower()
    return any(
        fnmatch(name, pattern) or fnmatch(lower_name, pattern.lower()) for pattern in _REFERENCE_SCAN_EXCLUDE_DIRS
    )


def _find_nc_dir_with_descent(start: Path, max_descent: int = 2) -> tuple[Path | None, int, str]:
    """Find a unique NC-bearing directory up to ``max_descent`` levels down.

    Supported NC locations are ``start/*.nc``, ``start/X/*.nc`` and
    ``start/X/Y/*.nc`` by default. The search inspects every branch within
    that depth budget before returning so layouts with one shallow NC child
    and another deeper NC child are reported as ambiguous instead of silently
    registering only the shallow branch.

    Returns ``(nc_dir, count, status)`` where status ∈
    ``{"found", "ambiguous", "missing"}``.
    """

    matches: list[tuple[Path, int]] = []

    def collect(cur: Path, depth: int) -> None:
        cur_count = _count_nc(cur)
        if cur_count > 0:
            matches.append((cur, cur_count))
        if depth == max_descent:
            return
        for sub in _iter_dirs(cur):
            collect(sub, depth + 1)

    collect(start, 0)

    if len(matches) == 1:
        return (*matches[0], "found")
    if len(matches) > 1:
        return (None, 0, "ambiguous")
    return (None, 0, "missing")


def _has_resolution_token(text: str, amount: str, units: tuple[str, ...]) -> bool:
    unit_pattern = "|".join(re.escape(unit) for unit in units)
    return bool(re.search(rf"(?<!\d){re.escape(amount)}\s*(?:{unit_pattern})(?![a-z0-9])", text))


def _detect_tim_res(dataset_dir: Path, file_glob: str | list[str] | tuple[str, ...] | None = None) -> str:
    """Detect time resolution from filename / directory keywords.

    Returns an empty string when no keyword evidence is found, so the caller
    in _apply_nc_corrections can distinguish "scanned and confirmed" from
    "default fallback" and assign provenance accordingly.
    """
    nc_files = _matching_nc_files(dataset_dir, file_glob) if file_glob else _glob_nc(dataset_dir)
    if not nc_files:
        return ""

    name = nc_files[0].stem.lower()
    # Only the dataset folder's own name — NOT the full absolute path. Scanning the
    # whole path lets unrelated parent components (e.g. macOS temp dirs under
    # /var/folders/8d/...) trip the short resolution tokens and mis-detect "8Day".
    dir_str = dataset_dir.name.lower()

    # Check specific sub-daily BEFORE generic "hourly"/"daily" to avoid substring false matches
    # e.g., "3hourly" contains "hourly" — must check "3hour" first
    if _has_resolution_token(name, "3", ("hourly", "hour", "hr", "h")) or _has_resolution_token(
        dir_str, "3", ("hourly", "hour", "hr", "h")
    ):
        return "3Hour"
    if _has_resolution_token(name, "30", ("minutes", "minute", "mins", "min")) or _has_resolution_token(
        dir_str, "30", ("minutes", "minute", "mins", "min")
    ):
        return "30min"
    if _has_resolution_token(name, "6", ("hourly", "hour", "hr", "h")) or _has_resolution_token(
        dir_str, "6", ("hourly", "hour", "hr", "h")
    ):
        return "6Hour"
    if _has_resolution_token(name, "8", ("daily", "day", "d")) or _has_resolution_token(
        dir_str, "8", ("daily", "day", "d")
    ):
        return "8Day"

    # Then generic hourly/daily
    if "hourly" in name or "hourly" in dir_str:
        return "Hour"
    if "daily" in name or "daily" in dir_str:
        return "Day"

    # No keyword evidence — return empty so _apply_nc_corrections falls
    # through to the "default" branch and provenance reflects that honestly.
    # Previously returned "Month" with provenance="scan", lying about
    # whether the value was inferred from data or was a hard-coded fallback.
    return ""


def _detect_data_type_from_nc(nc_file: Path) -> str | None:
    """Detect whether a NC file contains gridded or station data.

    Rules:
      - Station: has a station/site dimension, OR has neither lat nor lon
        with size > 1 (single point or time-only series)
      - Grid: has lat OR lon dimension with size > 1 (covers full 2D grids
        AND 1D profile-style data with single-cell width along one axis)

    Returns "grid", "stn", or None on failure.
    """
    try:
        import netCDF4

        with netCDF4.Dataset(str(nc_file), "r") as nc:
            dims = {k.lower(): nc.dimensions[k].size for k in nc.dimensions}
            variables = {k.lower(): k for k in nc.variables}

            # FMS/LM4-style unstructured land diagnostics carry grid cells on a
            # grid_index axis plus auxiliary geolon_t/geolat_t coordinates,
            # not lat/lon dimensions. Treat that as gridded, not station data.
            grid_dim = next((name for name in ("grid_index", "grid_cell", "cell", "ncol") if name in dims), None)
            if grid_dim and dims[grid_dim] > 1:
                lon_name = variables.get("geolon_t") or variables.get("geolon")
                lat_name = variables.get("geolat_t") or variables.get("geolat")
                if lon_name and lat_name:
                    lon_dims = {dim.lower() for dim in nc.variables[lon_name].dimensions}
                    lat_dims = {dim.lower() for dim in nc.variables[lat_name].dimensions}
                    if grid_dim in lon_dims and grid_dim in lat_dims:
                        return "grid"

        from openbench.data.coordinates import LAT_NAMES, LON_NAMES, STN_DIM_NAMES

        # Check for station-like dimensions
        if STN_DIM_NAMES & set(dims.keys()):
            return "stn"

        # Check lat/lon using shared fallback names
        lat_size = 0
        for name in LAT_NAMES:
            if name.lower() in dims:
                lat_size = dims[name.lower()]
                break
        lon_size = 0
        for name in LON_NAMES:
            if name.lower() in dims:
                lon_size = dims[name.lower()]
                break

        # Any spatial axis > 1 → grid (covers 2D and 1D profile data).
        # Previously the lat>1, lon=1 case (zonal/profile slices along lat)
        # returned None and the caller fell back to "grid" anyway; making
        # this explicit avoids ambiguity downstream.
        if lat_size > 1 or lon_size > 1:
            return "grid"
        # Both ≤1 → single point or time-only series → station
        return "stn"
    except Exception:
        return None


def _has_climatology_hint(dataset_dir: Path, nc_files: list[Path]) -> bool:
    hints = ("clim", "climatology", "climatological")
    parts = [part.lower() for part in dataset_dir.parts]
    parts.extend(file_path.stem.lower() for file_path in nc_files)
    return any(any(hint in part for hint in hints) for part in parts)


def _sample_nc_files(nc_files: list[Path], max_samples: int = 3) -> list[Path]:
    if len(nc_files) <= max_samples:
        return nc_files
    indexes = {0, len(nc_files) // 2, len(nc_files) - 1}
    return [nc_files[i] for i in sorted(indexes)]


def inspect_nc_file(
    dataset_dir: Path,
    file_glob: str | list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Public helper for shared NetCDF inspection used by registry and simulation scanners."""
    return _inspect_nc_file(dataset_dir, file_glob=file_glob)


def _inspect_nc_file(
    dataset_dir: Path,
    file_glob: str | list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Inspect a NetCDF file to extract variable name, unit, prefix, suffix.

    Opens the first .nc file in the directory, finds the primary data
    variable (skips time_bnds, lat, lon, etc.), and extracts metadata.
    Also parses the filename to detect prefix and suffix patterns.

    Returns:
        {"varname": str, "varunit": str, "prefix": str, "suffix": str,
         "detected_data_type": "grid"|"stn"|None}
        or empty dict if inspection fails.
    """
    nc_files = _matching_nc_files(dataset_dir, file_glob) if file_glob else _glob_nc(dataset_dir)
    if not nc_files:
        return {}

    result = {"nc_file_count": len(nc_files)}
    sample_files = _sample_nc_files(nc_files)
    primary_file = sample_files[0]

    data_type_votes = [dt for dt in (_detect_data_type_from_nc(path) for path in sample_files) if dt]
    if data_type_votes:
        result["detected_data_type"] = data_type_votes[0]
        if len(set(data_type_votes)) > 1:
            result["inconsistent_data_type"] = sorted(set(data_type_votes))

    try:
        import netCDF4

        errors = []
        opened_nc = None
        for candidate_file in sample_files:
            try:
                opened_nc = netCDF4.Dataset(str(candidate_file), "r")
                primary_file = candidate_file
                break
            except Exception as e:
                errors.append(f"{candidate_file.name}: {e}")

        if opened_nc is None:
            result["inspection_failed"] = "; ".join(errors) or "no sampled NC files opened"
        else:
            with opened_nc as nc:
                skip_vars = {
                    "time_bnds",
                    "time_bounds",
                    "lat_bnds",
                    "lon_bnds",
                    "lat_bounds",
                    "lon_bounds",
                    "crs",
                    "spatial_ref",
                    "geolat_t",
                    "geolon_t",
                    "geolat",
                    "geolon",
                }
                known_coord_var_names = {
                    "lat",
                    "latitude",
                    "lon",
                    "longitude",
                    "x",
                    "y",
                    "z",
                    "depth",
                    "level",
                    "elev",
                    "elevation",
                    "altitude",
                    "alt",
                    "height",
                    "station",
                    "station_id",
                    "station_name",
                    "site",
                    "site_id",
                    "site_name",
                    "id",
                    "geolat_t",
                    "geolon_t",
                    "geolat",
                    "geolon",
                    "grid_index",
                }
                coord_names = set(nc.dimensions.keys())
                min_dims = 1 if result.get("detected_data_type") == "stn" else 2
                data_vars = [
                    v
                    for v in nc.variables
                    if v not in skip_vars
                    and v not in coord_names
                    and v.lower() not in known_coord_var_names
                    and len(nc.variables[v].dimensions) >= min_dims
                ]
                result["all_data_vars"] = []
                for dv in data_vars:
                    var = nc.variables[dv]
                    unit = getattr(var, "units", getattr(var, "unit", ""))
                    unit = str(unit).replace(".", " ").strip() if unit else ""
                    result["all_data_vars"].append(
                        {
                            "name": dv,
                            "unit": unit,
                            "dims": list(var.dimensions),
                            "long_name": getattr(var, "long_name", ""),
                            "standard_name": getattr(var, "standard_name", ""),
                        }
                    )
                if data_vars:
                    result["varname"] = data_vars[0]
                    result["varunit"] = result["all_data_vars"][0]["unit"]
    except Exception as e:
        result["inspection_failed"] = str(e)
        logger.debug("NC inspection failed for %s: %s", primary_file.name, e)

    try:
        import math
        import netCDF4 as _nc4
        import numpy as _np
        import numpy.ma as _ma

        from openbench.data.coordinates import LAT_NAMES

        detected_tim_values = []
        detected_grid_values = []
        climatology_candidates = []

        for sampled_file in sample_files:
            with _nc4.Dataset(str(sampled_file), "r") as nc:
                time_var = None
                for time_name in ("time", "Time", "TIME", "t", "T"):
                    if time_name in nc.variables and time_name in nc.dimensions:
                        time_var = nc.variables[time_name]
                        break

                if time_var is not None and len(time_var) >= 2:
                    raw_diff = time_var[1] - time_var[0]
                    diff = None
                    if not _ma.is_masked(raw_diff):
                        candidate_diff = float(raw_diff)
                        if math.isfinite(candidate_diff):
                            diff = candidate_diff
                    units = getattr(time_var, "units", "")
                    detected = None
                    if diff is None:
                        pass
                    elif "seconds" in units:
                        if abs(diff - 3600) < 600:
                            detected = "Hour"
                        elif abs(diff - 10800) < 1800:
                            detected = "3Hour"
                        elif abs(diff - 21600) < 3600:
                            detected = "6Hour"
                        elif abs(diff - 86400) < 7200:
                            detected = "Day"
                        elif abs(diff - 691200) < 86400:
                            detected = "8Day"
                        elif abs(diff - 2592000) < 432000:
                            detected = "Month"
                        elif abs(diff - 31536000) < 2592000:
                            detected = "Year"
                    elif "hour" in units:
                        if abs(diff - 1) < 0.2:
                            detected = "Hour"
                        elif abs(diff - 3) < 0.5:
                            detected = "3Hour"
                        elif abs(diff - 6) < 1.0:
                            detected = "6Hour"
                        elif abs(diff - 24) < 2.0:
                            detected = "Day"
                        elif abs(diff - 192) < 24:
                            detected = "8Day"
                        elif 696 <= diff <= 768:
                            detected = "Month"
                        elif abs(diff - 8760) < 720:
                            detected = "Year"
                    elif "day" in units:
                        if abs(diff - 1) < 0.2:
                            detected = "Day"
                        elif abs(diff - 8) < 1.0:
                            detected = "8Day"
                        elif 28 <= diff <= 32:
                            detected = "Month"
                        elif abs(diff - 365) < 30:
                            detected = "Year"
                    if detected:
                        detected_tim_values.append(detected)

                if time_var is not None and len(time_var) == 12 and _has_climatology_hint(dataset_dir, nc_files):
                    units = getattr(time_var, "units", "")
                    if units:
                        raw_values = time_var[:]
                        if _ma.isMaskedArray(raw_values):
                            raw_values = raw_values.compressed() if raw_values.count() == 12 else []
                        dates = _nc4.num2date(
                            raw_values,
                            units=units,
                            calendar=getattr(time_var, "calendar", "standard"),
                        )
                        months = {int(getattr(date, "month")) for date in dates}
                        if months == set(range(1, 13)):
                            climatology_candidates.append("climatology-month")

                for lat_name in LAT_NAMES:
                    if lat_name in nc.variables and lat_name in nc.dimensions and nc.dimensions[lat_name].size > 1:
                        lat_vals = _np.asarray(_ma.filled(nc.variables[lat_name][:], _np.nan), dtype=float)
                        diffs = _np.diff(lat_vals)
                        diffs = diffs[_np.isfinite(diffs)]
                        if diffs.size:
                            grid_res = round(abs(float(_np.median(diffs))), 4)
                            if 0.001 < grid_res < 10:
                                detected_grid_values.append(grid_res)
                        break

        if detected_tim_values:
            result["detected_tim_res"] = detected_tim_values[0]
            if len(set(detected_tim_values)) > 1:
                result["inconsistent_tim_res"] = sorted(set(detected_tim_values))
        if detected_grid_values:
            detected_grid_values = sorted(detected_grid_values)
            result["detected_grid_res"] = detected_grid_values[len(detected_grid_values) // 2]
            if len(set(detected_grid_values)) > 1:
                result["inconsistent_grid_res"] = sorted(set(detected_grid_values))
        if climatology_candidates:
            result["climatology_candidate"] = climatology_candidates[0]
    except Exception as e:
        logger.debug("NC tim_res/grid_res detection failed: %s", e)

    fname = primary_file.stem
    date_match = _filename_split_match(fname)
    if date_match:
        result["prefix"] = fname[: date_match.start("token")]
        result["suffix"] = fname[date_match.end("token") :]
    else:
        result["prefix"] = fname
        result["suffix"] = ""

    years = []
    for f in nc_files:
        for m in _most_specific_date_matches(f.stem):
            if _is_year_range_endpoint(f.stem, m):
                continue
            y = int(m.group("year"))
            if 1900 <= y <= 2100:
                years.append(y)
    if years:
        result["syear"] = min(years)
        result["eyear"] = max(years)

    return result


def generate_station_list(dataset_dir: Path, output_csv: Path | None = None) -> Path:
    """Auto-generate a station list CSV from a directory of station NC files.

    Scans all .nc files in the directory, extracts station ID, lat, lon,
    time range from each file, and writes a CSV in the fulllist format:
        ID, SYEAR, EYEAR, LON, LAT, DIR

    Supports two formats:
    1. One-file-per-station: each NC has lat/lon as variables or scalars
    2. Single merged file: one NC with a station dimension

    Args:
        dataset_dir: Directory containing station NC files
        output_csv: Output CSV path. Defaults to dataset_dir/station_list.csv

    Returns:
        Path to the generated CSV file.
    """
    import pandas as pd

    if output_csv is None:
        output_csv = dataset_dir / "station_list.csv"

    nc_files = _glob_nc(dataset_dir)
    if not nc_files:
        raise FileNotFoundError(f"No NC files found in {dataset_dir}")

    rows = []

    # Try one-file-per-station first
    for nc_file in nc_files:
        row = _parse_single_station_file(nc_file)
        if row:
            rows.append(row)

    # If that found no stations but there are large merged files, try merged parsing
    if not rows:
        merged_rows = []
        for nc_file in nc_files:
            mr = _parse_merged_station_file(nc_file, dataset_dir)
            if mr:
                merged_rows.extend(mr)
        if len(merged_rows) > len(rows):
            rows = merged_rows

    if not rows:
        raise ValueError(f"Could not extract station info from {dataset_dir}")

    df = pd.DataFrame(rows, columns=["ID", "SYEAR", "EYEAR", "LON", "LAT", "DIR"])
    write_file_atomic(output_csv, lambda tmp_path: df.to_csv(tmp_path, index=False), suffix=".tmp.csv")
    logger.info("Generated station list: %s (%d stations)", output_csv, len(df))

    return output_csv


def _numeric_scalar_or_singleton(value) -> float | None:
    """Convert scalar or length-one coordinate values; reject multi-point arrays."""
    import numpy as np

    arr = np.ma.asarray(value)
    if np.ma.is_masked(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr)
    if arr.shape == ():
        return float(arr)
    if arr.size != 1:
        return None
    return float(arr.reshape(-1)[0])


def _station_coordinate_value(ds, coord_var: str | None, station_dim: str, index: int) -> float | None:
    """Return a station coordinate only when it is scalar for this station.

    Merged station files should expose lat/lon either as scalar values or as
    arrays indexed by the station dimension.  Curvilinear/grid coordinates such
    as ``lat(y, x)`` are not station coordinates; treating their first axis as
    station IDs silently creates wrong station lists or raises opaque indexing
    errors.  Reject non-scalar per-station slices instead.
    """
    if not coord_var:
        return None

    import numpy as np

    data_array = ds[coord_var]
    if station_dim in data_array.dims:
        value = data_array.isel({station_dim: index}).values
    elif data_array.size == 1:
        value = data_array.values
    else:
        logger.warning(
            "Skipping merged station coordinate %s: dims %s are not indexed by station dimension %s",
            coord_var,
            data_array.dims,
            station_dim,
        )
        return None

    scalar = _numeric_scalar_or_singleton(value)
    if scalar is not None:
        return scalar

    arr = np.asarray(np.ma.asarray(value).filled(np.nan) if np.ma.is_masked(value) else value, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.allclose(finite, finite[0], equal_nan=False):
        return float(finite[0])
    logger.warning("Skipping merged station coordinate %s: station slice is not scalar", coord_var)
    return None


def _decode_station_id_value(value) -> str | None:
    """Decode scalar/string/char-array station ID values from NetCDF."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, str):
        station_id = value.strip()
        return station_id or None
    if isinstance(value, bytes):
        station_id = value.decode("utf-8", errors="ignore").strip()
        return station_id or None

    arr = np.ma.asarray(value)
    if np.ma.is_masked(arr):
        arr = arr.filled(b"")

    if arr.shape == ():
        return _decode_station_id_value(arr.item())
    if arr.size == 0:
        return None
    if arr.size == 1:
        return _decode_station_id_value(arr.reshape(-1)[0].item())

    # Classic NetCDF stores one string as S1(nchar). Join those characters.
    dtype_kind = getattr(arr.dtype, "kind", "")
    if dtype_kind in {"S", "U"}:
        if arr.ndim >= 2:
            if arr.shape[0] != 1:
                return None
            arr = arr.reshape(arr.shape[-1])
        parts = [_decode_station_id_value(item) or "" for item in arr.reshape(-1)]
        station_id = "".join(parts).strip()
        return station_id or None

    # Multi-element non-char arrays are likely multi-station IDs.
    return None


def _merged_station_dimension(ds, time_dim: str | None) -> str | None:
    from openbench.data.coordinates import STN_DIM_NAMES

    ignored_dims = {
        str(time_dim).lower() if time_dim else "",
        "nv",
        "bnds",
        "bounds",
        "nbnds",
        "nchar",
        "strlen",
        "string_length",
    }
    named_dims = [dim for dim in ds.dims if dim.lower() in STN_DIM_NAMES and dim.lower() not in ignored_dims]
    if named_dims:
        return named_dims[0]

    for dim in ds.dims:
        dim_lower = dim.lower()
        if dim_lower in ignored_dims:
            continue
        for var in ds.data_vars.values():
            var_dims = {d.lower() for d in var.dims}
            if dim in var.dims and (time_dim and time_dim.lower() in var_dims):
                return dim
    return None


def _parse_single_station_file(nc_file: Path) -> list | None:
    """Extract station info from a single-station NC file (fast, metadata only)."""
    try:
        import netCDF4

        with netCDF4.Dataset(str(nc_file), "r") as nc:
            station_like_dims = {
                name.lower(): dim.size
                for name, dim in nc.dimensions.items()
                if name.lower() in {"station", "stations", "site", "sites", "stn", "station_id"}
            }
            if any(size > 1 for size in station_like_dims.values()):
                return None

            # Extract station ID: NC variable/attribute first, then filename
            station_id = None

            # 1. Try NC variables (scalar or single-element)
            for id_var in ("station_id", "site_id", "station_name", "site"):
                if id_var in nc.variables:
                    val = nc.variables[id_var][:]
                    station_id = _decode_station_id_value(val)
                    # Multi-station file — skip (handled by _parse_merged_station_file)
                    if station_id:
                        break

            # 2. Try NC global attributes
            if not station_id:
                for id_attr in ("station_id", "site_id", "station_code"):
                    if id_attr in nc.ncattrs():
                        station_id = str(nc.getncattr(id_attr)).strip()
                        if station_id:
                            break

            # 3. Fallback: extract from filename
            if not station_id:
                stem = nc_file.stem
                year_match = re.search(r"[_-](\d{4})[_-]", stem)
                if year_match:
                    station_id = stem[: year_match.start()].rstrip("_-")
                else:
                    station_id = stem

            # Extract lat/lon using shared fallback names
            from openbench.data.coordinates import LAT_NAMES, LON_NAMES

            lat = lon = None
            for name in LAT_NAMES:
                if name in nc.variables:
                    val = nc.variables[name][:]
                    lat = _numeric_scalar_or_singleton(val)
                    if lat is None:
                        return None
                    break
            for name in LON_NAMES:
                if name in nc.variables:
                    val = nc.variables[name][:]
                    lon = _numeric_scalar_or_singleton(val)
                    if lon is None:
                        return None
                    break

            # Extract time range from filename first (faster than reading time dim)
            stem = nc_file.stem
            syear = eyear = ""
            years = [
                int(m.group("year")) for m in _most_specific_date_matches(stem) if 1900 <= int(m.group("year")) <= 2100
            ]
            if years:
                syear, eyear = min(years), max(years)
            elif "time" in nc.dimensions and nc.dimensions["time"].size > 0:
                # Fallback: read time variable
                try:
                    time_var = nc.variables["time"]
                    times = netCDF4.num2date(
                        time_var[:], time_var.units, time_var.calendar if hasattr(time_var, "calendar") else "standard"
                    )
                    syear = times[0].year
                    eyear = times[-1].year
                except Exception:
                    pass

        if lat is not None and lon is not None:
            return [station_id, syear, eyear, lon, lat, str(nc_file)]
    except Exception as e:
        logger.warning("Failed to parse station file %s: %s", nc_file.name, e)

    return None


def _parse_merged_station_file(nc_file: Path, dataset_dir: Path) -> list:
    """Extract station info from a merged multi-station NC file."""
    rows = []
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        # Use a with-block so the file handle is released even if any of
        # the per-station unpacking below raises (previously the outer
        # except swallowed the error and leaked the open NC file).
        with xr.open_dataset(nc_file) as ds:
            from openbench.data.coordinates import LAT_NAMES, LON_NAMES

            time_dim = next((dim for dim in ds.dims if dim.lower() in ("time", "t")), None)
            stn_dim = _merged_station_dimension(ds, time_dim)

            if not stn_dim:
                return rows

            n_stations = ds.sizes[stn_dim]
            lat_var = _find_var(ds, LAT_NAMES)
            lon_var = _find_var(ds, LON_NAMES)

            # Time range (case-insensitive dim name)
            syear = ""
            eyear = ""
            if time_dim and ds.sizes[time_dim] > 0:
                time_vals = ds[time_dim].values
                try:
                    syear = int(pd.Timestamp(time_vals[0]).year)
                    eyear = int(pd.Timestamp(time_vals[-1]).year)
                except Exception:
                    syear = ""
                    eyear = ""

            for i in range(n_stations):
                station_id = None
                for id_var in ("station_id", "site_id", "station_name", "site"):
                    if id_var in ds:
                        values = ds[id_var].values
                        try:
                            station_id = _decode_station_id_value(values[i])
                        except Exception:
                            station_id = None
                        if station_id:
                            break
                if not station_id:
                    station_id = str(ds[stn_dim].values[i]) if stn_dim in ds.coords else str(i)
                lat = _station_coordinate_value(ds, lat_var, stn_dim, i)
                lon = _station_coordinate_value(ds, lon_var, stn_dim, i)

                if lat is not None and lon is not None and not (np.isnan(lat) or np.isnan(lon)):
                    rows.append([station_id, syear, eyear, lon, lat, str(nc_file)])
    except Exception as e:
        logger.warning("Failed to parse merged station file %s: %s", nc_file.name, e)

    return rows


def _extract_scalar(ds, var_names: list):
    """Extract a scalar value from a dataset, trying multiple variable names."""
    for name in var_names:
        if name in ds:
            val = ds[name].values
            if hasattr(val, "item"):
                return float(val.item())
            elif hasattr(val, "__float__"):
                return float(val)
        if name in ds.attrs:
            return float(ds.attrs[name])
    return None


def _find_var(ds, var_names: list) -> str | None:
    """Find a variable name in a dataset from a list of candidates."""
    for name in var_names:
        if name in ds:
            return name
    return None
