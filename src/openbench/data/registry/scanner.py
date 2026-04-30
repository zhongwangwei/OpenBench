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

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# NetCDF file discovery — supports .nc and .nc4
from openbench.data.coordinates import glob_nc as _glob_nc


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


def _backup_then_write(catalog_path: Path, data: dict) -> None:
    """Backup the previous catalog (if any) before atomic-writing the new one.

    Creates a single-slot backup at ``<catalog>.bak``. The backup is the
    catalog state immediately before this write — useful when a buggy
    rescan overwrites hand-edited fields and the user wants to recover.
    """
    if catalog_path.exists():
        import shutil
        backup_path = Path(str(catalog_path) + ".bak")
        try:
            shutil.copy2(catalog_path, backup_path)
        except OSError as e:
            logger.warning("Could not create catalog backup at %s: %s", backup_path, e)
    _atomic_yaml_write(catalog_path, data)


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

    Falls back gracefully on platforms without flock (Windows native, NFS
    without nlockmgr): logs a debug note and proceeds without serialization.
    Single-user local installs see no behavior change.
    """
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(catalog_path) + ".lock")
    try:
        lock_path.touch(exist_ok=True)
    except OSError as e:
        logger.debug("Could not create lock file %s: %s", lock_path, e)
        yield
        return

    try:
        import fcntl
    except ImportError:
        # Windows native — proceed without locking
        yield
        return

    lock_file = open(lock_path, "r")
    have_lock = False
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            have_lock = True
        except OSError as e:
            logger.debug(
                "flock unavailable for %s (%s); proceeding without lock",
                lock_path, e,
            )
        yield
    finally:
        if have_lock:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
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
        from openbench.data.registry.manager import get_writable_registry_dir

        profiles: dict[str, Any] = {}
        profile_paths = [Path(__file__).parent / "reference_profiles.yaml"]

        writable_profile = get_writable_registry_dir() / "reference_profiles.yaml"
        if writable_profile not in profile_paths:
            profile_paths.append(writable_profile)

        for profile_path in profile_paths:
            if not profile_path.exists():
                continue
            try:
                with open(profile_path) as f:
                    loaded = yaml.safe_load(f) or {}
                profiles.update(loaded)
            except Exception:
                continue

        _REFERENCE_PROFILES = profiles
    return _REFERENCE_PROFILES


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
    # 2. Strip resolution suffix and try base name
    for suffix in ("_LowRes", "_MidRes", "_HigRes"):
        if dataset_name.endswith(suffix):
            base = dataset_name[: -len(suffix)]
            if base in profiles:
                return profiles[base]
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


@dataclass
class ScannedDataset:
    """A discovered dataset with its location and resolution info."""

    name: str  # e.g., "GLEAM_v4.2a"
    resolution: str  # "LowRes", "MidRes", "HigRes", or "Station"
    category: str  # "Water", "Energy", etc.
    data_type: str  # "grid" or "stn"
    root_dir: str  # Full path to the resolution-level root (e.g., .../Grid/LowRes/Water)
    variables: dict[str, str] = field(default_factory=dict)  # var_name -> sub_dir path
    file_count: int = 0
    tim_res: str = ""  # Detected or empty

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


def scan_reference_directory(ref_root: str | Path, on_progress=None) -> list[DatasetGroup]:
    """Scan a reference data directory and discover all datasets.

    Args:
        ref_root: Root directory (e.g., /Volumes/work/Reference)
        on_progress: Optional callback(message: str) for progress updates.

    Returns:
        List of DatasetGroup, each containing resolution variants.
    """
    ref_root = Path(ref_root)
    if not ref_root.exists():
        logger.warning("Reference directory not found: %s", ref_root)
        return []

    groups: dict[str, DatasetGroup] = {}

    # Scan grid data: Grid/{Res}/<Category>/<Variable>/<Dataset>/*.nc
    # Walk 3 levels of directories. Skip "Composite" category (non-standard, register manually).
    # If a 3rd-level dir has NC files → standard dataset.
    # If not but its children do → dataset with sub-dirs (depth 4).
    grid_dir = ref_root / "Grid"
    if grid_dir.exists():
        for res_name in ["LowRes", "MidRes", "HigRes"]:
            res_dir = grid_dir / res_name
            if not res_dir.exists():
                continue

            if on_progress:
                on_progress(f"Scanning Grid/{res_name}...")

            for category_dir in _iter_dirs(res_dir):
                cat_name = category_dir.name
                if cat_name == "Composite":
                    if on_progress:
                        on_progress(f"  Skipping Composite/ (register manually)")
                    continue
                category = CATEGORY_MAP.get(cat_name, cat_name)

                for var_dir in _iter_dirs(category_dir):
                    var_name = var_dir.name
                    if on_progress:
                        on_progress(f"  {res_name}/{cat_name}/{var_name}")

                    for dataset_dir in _iter_dirs(var_dir):
                        dataset_name = dataset_dir.name

                        # Locate the actual NC-bearing directory.
                        # Case A: NCs at dataset_dir/*.nc
                        # Case B: NCs at dataset_dir/<one_child>/*.nc → use the child
                        # Case C: NCs at dataset_dir/<multiple_children>/*.nc → ambiguous,
                        #         skip with warning (composite/multi-variant datasets need
                        #         a reference profile or manual register)
                        nc_count = _count_nc(dataset_dir)
                        nc_dir = dataset_dir

                        if nc_count == 0:
                            children_with_nc = [
                                (sub, _count_nc(sub))
                                for sub in _iter_dirs(dataset_dir)
                                if _count_nc(sub) > 0
                            ]
                            if len(children_with_nc) == 1:
                                nc_dir, nc_count = children_with_nc[0]
                            elif len(children_with_nc) > 1:
                                sub_names = sorted(c[0].name for c in children_with_nc)
                                logger.warning(
                                    "Skipped '%s': %d NC-bearing subdirectories %s. "
                                    "Composite/multi-variant datasets need a reference profile "
                                    "(reference_profiles.yaml) or manual 'openbench data register'.",
                                    dataset_dir.relative_to(ref_root),
                                    len(children_with_nc),
                                    sub_names,
                                )
                                continue

                        if nc_count == 0:
                            # Warn if deeper levels have NC files (beyond supported depth)
                            from openbench.data.coordinates import glob_nc as _deep_glob
                            deep_nc = _deep_glob(dataset_dir, recursive=True)
                            if deep_nc:
                                logger.warning(
                                    "Skipped '%s': %d NC files found in deeper subdirectories "
                                    "(beyond supported 2-level depth). Move files up or register manually.",
                                    dataset_dir.relative_to(ref_root), len(deep_nc),
                                )
                            continue

                        # Detect tim_res and record sub_dir from the actual NC location
                        tim_res = _detect_tim_res(nc_dir)
                        sub_dir = str(nc_dir.relative_to(res_dir))

                        if dataset_name not in groups:
                            groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                        if res_name not in groups[dataset_name].variants:
                            groups[dataset_name].variants[res_name] = ScannedDataset(
                                name=dataset_name,
                                resolution=res_name,
                                category=category,
                                data_type="grid",
                                root_dir=str(res_dir),
                                tim_res=tim_res,
                            )

                        scanned = groups[dataset_name].variants[res_name]
                        if var_name not in scanned.variables:
                            scanned.variables[var_name] = sub_dir
                            scanned.file_count += nc_count

    # Scan station data: Station/<category>/<variable>/<dataset>/
    # Also handles Composite layout: Station/Composite/<dataset>/dataset/*.nc
    stn_dir = ref_root / "Station"
    if stn_dir.exists():
        if on_progress:
            on_progress("Scanning Station/...")
        for category_dir in _iter_dirs(stn_dir):
            cat_name = category_dir.name
            category = CATEGORY_MAP.get(cat_name, cat_name)

            for var_dir in _iter_dirs(category_dir):
                var_name = var_dir.name

                for dataset_dir in _iter_dirs(var_dir):
                    dataset_name = dataset_dir.name
                    nc_dir = dataset_dir  # where NC files live
                    nc_count = _count_nc(dataset_dir)

                    # Check one level deeper if no NCs at this level
                    # Handles Composite layout: Composite/FLUXNET_PLUMBER2/dataset/*.nc
                    if nc_count == 0:
                        for child in _iter_dirs(dataset_dir):
                            child_nc = _count_nc(child)
                            if child_nc > 0:
                                nc_count += child_nc
                                nc_dir = child
                    if nc_count == 0:
                        from openbench.data.coordinates import glob_nc as _deep_glob
                        deep_nc = _deep_glob(dataset_dir, recursive=True)
                        if deep_nc:
                            logger.warning(
                                "Skipped '%s': %d NC files found in deeper subdirectories "
                                "(beyond supported 2-level depth). Move files up or register manually.",
                                dataset_dir.relative_to(ref_root), len(deep_nc),
                            )
                        continue

                    # Composite layout: NC files live in a "dataset/" or "data/" subdir
                    # → the real dataset name is the parent (var_dir.name)
                    is_composite = dataset_name in ("dataset", "data")
                    if is_composite:
                        dataset_name = var_name  # e.g., FLUXNET_PLUMBER2
                        ds_root = str(nc_dir)    # points directly to NC directory
                    else:
                        ds_root = str(category_dir.parent)

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
                        scanned.variables[var_name] = str(dataset_dir.relative_to(category_dir.parent))
                    scanned.file_count += nc_count

    return sorted(groups.values(), key=lambda g: g.base_name)


def find_new_datasets(
    ref_root: str | Path,
    existing_names: Optional[set[str]] = None,
    on_progress=None,
) -> list[DatasetGroup]:
    """Scan and return only datasets not already registered.

    Args:
        ref_root: Reference data root directory.
        existing_names: Set of already registered dataset names.

    Returns:
        List of new DatasetGroup not in existing_names.
    """
    if existing_names is None:
        from openbench.data.registry.manager import RegistryManager

        mgr = RegistryManager()
        existing_names = {r.name for r in mgr.list_references()}

    all_groups = scan_reference_directory(ref_root, on_progress=on_progress)
    new_groups = []

    # Filter at variant granularity, not group: when only some resolutions of
    # a dataset are new, return a group containing JUST the new variants.
    # Otherwise the CLI/GUI re-registers already-existing variants and silently
    # overwrites top-level descriptor fields (category, fulllist, _provenance,
    # etc.) that the user may have hand-edited.
    for group in all_groups:
        new_variants = {
            res: variant
            for res, variant in group.variants.items()
            if variant.registry_name not in existing_names
        }
        if new_variants:
            new_groups.append(DatasetGroup(
                base_name=group.base_name,
                variants=new_variants,
            ))

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
        _register_to_dict(scanned, catalog, existing_descriptor=existing_descriptor, on_multi_var=on_multi_var)
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

        # Build all descriptors (this is where NC inspection happens)
        for i, scanned in enumerate(datasets):
            if on_progress:
                on_progress(f"  [{i + 1}/{len(datasets)}] {scanned.registry_name}")

            # Merge with existing descriptor to preserve hand-edited fields
            existing = catalog.get(scanned.registry_name)
            _register_to_dict(scanned, catalog, existing_descriptor=existing, on_multi_var=on_multi_var)

        # Write once (atomic, with backup of previous state)
        _backup_then_write(catalog_path, catalog)

    _invalidate_registry_caches()

    return catalog_path


def _register_to_dict(
    scanned,
    catalog: dict,
    existing_descriptor: Optional[dict] = None,
    on_multi_var=None,
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
    variables = _apply_profile_overrides(scanned, descriptor, variables, prov)

    # Stage 5: finalize (grid_res, provenance, fulllist)
    descriptor["variables"] = variables
    _finalize_descriptor(scanned, descriptor, prov, existing_descriptor)

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

    # Fields where existing user edits ALWAYS win when present:
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
        descriptor["years"] = existing["years"]


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


# Date patterns at filename position boundaries.
# Use lookarounds (?<![a-zA-Z\d]) / (?![a-zA-Z\d]) to require a non-alphanumeric
# delimiter (or string boundary), so version tokens like "v20240315" don't
# count as YYYYMMDD and "ERA5LAND" doesn't bleed into year detection.
_DATE_PATTERNS = {
    "day":   re.compile(r"(?<![a-zA-Z\d])(?:\d{4}[-_/]\d{2}[-_/]\d{2}|\d{8})(?!\d)"),
    "month": re.compile(r"(?<![a-zA-Z\d])(?:\d{4}[-_/]\d{2}|\d{6})(?!\d)"),
    "year":  re.compile(r"(?<![a-zA-Z\d])\d{4}(?![-_/\d])"),
}


def _classify_filename_date(stem: str) -> str:
    """Return the most-specific date granularity in a filename stem.

    Order: day > month > year > none. Tokens preceded by letters (e.g., the
    "v" in "v2010") are excluded so version markers don't masquerade as years.
    """
    if _DATE_PATTERNS["day"].search(stem):
        return "day"
    if _DATE_PATTERNS["month"].search(stem):
        return "month"
    if _DATE_PATTERNS["year"].search(stem):
        return "year"
    return "none"


def _detect_data_groupby(scanned) -> str:
    """Detect data_groupby from per-variable file count + filename patterns.

    Returns one of: "Single" | "Day" | "Month" | "Year".
    The previous implementation only ever returned "Year" or "Single",
    breaking processing.py's monthly/daily file iteration when descriptor
    said "Year" but filenames were YYYYMM.
    """
    if not scanned.variables:
        return "Year"

    # Collect NC file lists per variable
    var_files: list[list[Path]] = []
    for _vn, sub_dir in scanned.variables.items():
        var_path = Path(scanned.root_dir) / sub_dir
        if not var_path.is_dir():
            continue
        files = _glob_nc(var_path)
        if not files:
            for child in _iter_dirs(var_path):
                child_files = _glob_nc(child)
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

    For each scanned variable, either merge from an existing descriptor
    (preserving hand-edited fields) or inspect the NC file for metadata.
    NC-detected fields (_detected_data_type, _nc_tim_res, _nc_grid_res) are
    stored as temporary keys on var entries for Stage 3 to consume.
    """
    variables = {}
    for var_name, sub_dir in scanned.variables.items():
        var_entry: dict = {"varname": var_name, "varunit": "", "sub_dir": sub_dir}

        merged_from_existing = False
        if existing_descriptor:
            existing_vars = existing_descriptor.get("variables", {})
            if var_name in existing_vars:
                ev = existing_vars[var_name]
                var_entry["varname"] = ev.get("varname", var_name)
                var_entry["varunit"] = ev.get("varunit", "")
                if ev.get("prefix"):
                    var_entry["prefix"] = ev["prefix"]
                if ev.get("suffix"):
                    var_entry["suffix"] = ev["suffix"]
                merged_from_existing = True

        if not merged_from_existing:
            dataset_path = Path(scanned.root_dir) / sub_dir
            if dataset_path.is_dir():
                nc_info = _inspect_nc_file(dataset_path)

                all_vars = nc_info.get("all_data_vars", [])
                if len(all_vars) > 1 and on_multi_var:
                    chosen = on_multi_var(var_name, sub_dir, all_vars)
                    if chosen:
                        nc_info["varname"] = chosen
                        for av in all_vars:
                            if av["name"] == chosen:
                                nc_info["varunit"] = av["unit"]
                                break

                if nc_info.get("varname"):
                    var_entry["varname"] = nc_info["varname"]
                if nc_info.get("varunit"):
                    var_entry["varunit"] = nc_info["varunit"]
                if nc_info.get("prefix"):
                    var_entry["prefix"] = nc_info["prefix"]
                if nc_info.get("suffix"):
                    var_entry["suffix"] = nc_info["suffix"]
                if nc_info.get("syear"):
                    descriptor["years"] = [nc_info["syear"], nc_info.get("eyear", nc_info["syear"])]
                if nc_info.get("detected_data_type"):
                    var_entry["_detected_data_type"] = nc_info["detected_data_type"]
                if nc_info.get("detected_tim_res"):
                    var_entry["_nc_tim_res"] = nc_info["detected_tim_res"]
                if nc_info.get("detected_grid_res"):
                    var_entry["_nc_grid_res"] = nc_info["detected_grid_res"]

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
    for _vn, v in variables.items():
        nc_dtype = v.pop("_detected_data_type", None) if isinstance(v, dict) else None
        if nc_dtype and nc_dtype != descriptor["data_type"]:
            logger.warning(
                "Dataset '%s': directory says data_type='%s' but NC content looks like '%s'. "
                "Using NC-detected type.",
                scanned.registry_name, descriptor["data_type"], nc_dtype,
            )
            descriptor["data_type"] = nc_dtype
            prov["data_type"] = "nc"
            break

    # Collect NC-detected tim_res and grid_res from first inspected variable
    nc_tim_res = None
    nc_grid_res = None
    for _vn, v in variables.items():
        if isinstance(v, dict):
            if v.get("_nc_tim_res"):
                nc_tim_res = v.pop("_nc_tim_res")
            if v.get("_nc_grid_res"):
                nc_grid_res = v.pop("_nc_grid_res")
            if nc_tim_res or nc_grid_res:
                break

    # tim_res priority: nc > scan > default
    scan_tim_res = descriptor["tim_res"]  # from scanned.tim_res via Stage 1
    if nc_tim_res:
        descriptor["tim_res"] = nc_tim_res
        prov["tim_res"] = "nc"
    elif scan_tim_res:
        descriptor["tim_res"] = scan_tim_res
        prov["tim_res"] = "scan"
    else:
        descriptor["tim_res"] = "Month"
        prov["tim_res"] = "default"

    if nc_grid_res:
        descriptor["grid_res"] = nc_grid_res
        prov["grid_res"] = "nc"


# ---------------------------------------------------------------------------
# Stage 4: Apply reference profile overrides
# ---------------------------------------------------------------------------

def _apply_profile_overrides(scanned, descriptor: dict, variables: dict, prov: dict) -> dict:
    """Apply reference_profiles.yaml overrides to descriptor and variables.

    Profile override rules:
      - tim_res:      profile overrides scan/default, but NOT nc-detected
      - data_groupby: profile always overrides
      - variables:    standard datasets get varname/varunit merged;
                      composite datasets get variables replaced entirely

    Returns the (possibly replaced) variables dict.
    """
    profile = get_reference_profile(scanned.name)
    if profile:
        profile_vars = profile.get("variables", {})

        # Override metadata from profile (authoritative, except nc-detected tim_res)
        if profile.get("tim_res") and prov.get("tim_res") != "nc":
            descriptor["tim_res"] = profile["tim_res"]
            prov["tim_res"] = "profile"
        if profile.get("data_groupby"):
            descriptor["data_groupby"] = profile["data_groupby"]
            prov["data_groupby"] = "profile"

        if profile_vars:
            scanned_matches_profile = any(vn in profile_vars for vn in variables)

            if scanned_matches_profile:
                # Standard dataset: merge varname/varunit from profile
                for vn, pv in profile_vars.items():
                    if vn in variables:
                        variables[vn]["varname"] = pv["varname"]
                        variables[vn]["varunit"] = pv.get("varunit", "")
            else:
                # Composite dataset: scanned "variables" are directory names
                # Replace entirely with profile variable mappings
                variables = {}
                for vn, pv in profile_vars.items():
                    variables[vn] = {"varname": pv["varname"], "varunit": pv.get("varunit", "")}
    else:
        # No profile found — warn for datasets with no variable mapping
        has_real_varnames = any(
            v.get("varname") and v["varname"] != var_name
            for var_name, v in variables.items()
        )
        if not has_real_varnames and variables:
            logger.info(
                "No reference profile for '%s'. Register with:\n"
                "  openbench data register-profile %s -v \"VarName:ncname:unit\"",
                scanned.name, scanned.name,
            )

    return variables


# ---------------------------------------------------------------------------
# Stage 5: Finalize descriptor (grid_res default, provenance, fulllist)
# ---------------------------------------------------------------------------

def _finalize_descriptor(scanned, descriptor: dict, prov: dict, existing_descriptor: dict | None = None) -> None:
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

    # Auto-generate fulllist for station datasets — UNLESS the user already
    # has a hand-edited fulllist pointing to an existing file.
    if final_dtype == "stn":
        existing_fulllist = (existing_descriptor or {}).get("fulllist")
        if existing_fulllist and Path(existing_fulllist).exists():
            descriptor["fulllist"] = existing_fulllist
            return

        nc_dir = Path(scanned.root_dir)
        if scanned.variables:
            first_sub = next(iter(scanned.variables.values()), "")
            candidate = Path(scanned.root_dir) / first_sub
            if candidate.is_dir():
                if _glob_nc(candidate):
                    nc_dir = candidate
                else:
                    ds_sub = candidate / "dataset"
                    if ds_sub.is_dir() and _glob_nc(ds_sub):
                        nc_dir = ds_sub
        if _glob_nc(nc_dir):
            try:
                from openbench.data.registry.manager import get_writable_registry_dir
                registry_dir = get_writable_registry_dir()
                lists_dir = registry_dir / "station_lists"
                lists_dir.mkdir(parents=True, exist_ok=True)
                output_csv = lists_dir / f"{scanned.registry_name}.csv"
                generate_station_list(nc_dir, output_csv)
                descriptor["fulllist"] = str(output_csv)
            except Exception as e:
                logger.debug("Failed to generate station list for %s: %s", scanned.name, e)


# Frequency hierarchy: higher rank = higher frequency
# When a higher-frequency variant exists, lower-frequency variants are disabled.
_TIM_RES_RANK = {
    "year": 0, "yearly": 0, "y": 0,
    "month": 1, "monthly": 1, "m": 1, "mon": 1,
    "8day": 2, "8daily": 2, "week": 2, "weekly": 2, "w": 2,
    "day": 3, "daily": 3, "d": 3,
    "6hour": 4, "6h": 4, "6hourly": 4,
    "3hour": 5, "3h": 5, "3hourly": 5,
    "hour": 6, "hourly": 6, "h": 6,
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
    """Iterate over subdirectories, skipping hidden and non-dirs."""
    if not path.exists():
        return
    for item in sorted(path.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            yield item


def _detect_tim_res(dataset_dir: Path) -> str:
    """Detect time resolution from filename / directory keywords.

    Returns an empty string when no keyword evidence is found, so the caller
    in _apply_nc_corrections can distinguish "scanned and confirmed" from
    "default fallback" and assign provenance accordingly.
    """
    nc_files = _glob_nc(dataset_dir)
    if not nc_files:
        return ""

    name = nc_files[0].stem.lower()
    dir_str = str(dataset_dir).lower()

    # Check specific sub-daily BEFORE generic "hourly"/"daily" to avoid substring false matches
    # e.g., "3hourly" contains "hourly" — must check "3hour" first
    if "3hour" in name or "3h" in name or "3hour" in dir_str:
        return "3Hour"
    if "6hour" in name or "6h" in name or "6hour" in dir_str:
        return "6Hour"
    if "8daily" in name or "8day" in name or "8day" in dir_str:
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

        nc = netCDF4.Dataset(str(nc_file), "r")
        dims = {k.lower(): nc.dimensions[k].size for k in nc.dimensions}
        nc.close()

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


def _inspect_nc_file(dataset_dir: Path) -> dict:
    """Inspect a NetCDF file to extract variable name, unit, prefix, suffix.

    Opens the first .nc file in the directory, finds the primary data
    variable (skips time_bnds, lat, lon, etc.), and extracts metadata.
    Also parses the filename to detect prefix and suffix patterns.

    Returns:
        {"varname": str, "varunit": str, "prefix": str, "suffix": str,
         "detected_data_type": "grid"|"stn"|None}
        or empty dict if inspection fails.
    """
    nc_files = _glob_nc(dataset_dir)
    if not nc_files:
        return {}

    result = {}

    # Detect data_type from NC dimensions
    result["detected_data_type"] = _detect_data_type_from_nc(nc_files[0])

    # Extract varname and unit from NC contents (metadata only, no data loading)
    try:
        import netCDF4

        nc = netCDF4.Dataset(str(nc_files[0]), "r")

        # Filter out auxiliary / coordinate / metadata variables.
        # Bounds variables and coordinate-as-variables (lat/lon/elev/station_id)
        # are not data; they must not be picked as the dataset's primary variable.
        skip_vars = {
            "time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
            "lat_bounds", "lon_bounds", "crs", "spatial_ref",
        }
        # Coordinate / station-metadata names commonly stored as variables
        # rather than dimensions (especially in single-station NC files where
        # lat/lon are scalars). Compared case-insensitively below.
        known_coord_var_names = {
            "lat", "latitude", "lon", "longitude",
            "x", "y", "z", "depth", "level",
            "elev", "elevation", "altitude", "alt", "height",
            "station", "station_id", "station_name",
            "site", "site_id", "site_name", "id",
        }
        coord_names = set(nc.dimensions.keys())

        # Minimum dimension count: grid data needs (lat, lon) at minimum (>=2).
        # Station data is often (time,) per file — 1D is the data variable.
        # When detection is uncertain (None), bias toward grid (>=2) to avoid
        # false-positive picking up 1D auxiliaries on grid datasets.
        nc_data_type = result.get("detected_data_type")
        min_dims = 1 if nc_data_type == "stn" else 2

        data_vars = [
            v for v in nc.variables
            if v not in skip_vars
            and v not in coord_names
            and v.lower() not in known_coord_var_names
            and len(nc.variables[v].dimensions) >= min_dims
        ]

        # Store ALL data variables for multi-var detection
        result["all_data_vars"] = []
        for dv in data_vars:
            var = nc.variables[dv]
            unit = getattr(var, "units", getattr(var, "unit", ""))
            unit = str(unit).replace(".", " ").strip() if unit else ""
            long_name = getattr(var, "long_name", "")
            standard_name = getattr(var, "standard_name", "")
            result["all_data_vars"].append({
                "name": dv, "unit": unit, "dims": list(var.dimensions),
                "long_name": long_name, "standard_name": standard_name,
            })

        if data_vars:
            varname = data_vars[0]
            varunit = result["all_data_vars"][0]["unit"]
            result["varname"] = varname
            result["varunit"] = varunit

        nc.close()
    except Exception as e:
        logger.debug("NC inspection failed for %s: %s", nc_files[0].name, e)

    # Detect tim_res from time dimension interval and grid_res from lat interval
    try:
        import netCDF4 as _nc4

        _nc = _nc4.Dataset(str(nc_files[0]), "r")
        _time_names = ("time", "Time", "TIME", "t", "T")
        _time_var = None
        for _tn in _time_names:
            if _tn in _nc.variables and _tn in _nc.dimensions:
                _time_var = _nc.variables[_tn]
                break
        if _time_var is not None and len(_time_var) >= 2:
            _diff = float(_time_var[1] - _time_var[0])
            _units = getattr(_time_var, "units", "")
            _detected = None
            # Tight tolerance buckets: each candidate frequency gets a centered
            # window so half-hourly (1800s) doesn't get bucketed as "Hour" and
            # quarterly (~90 days) doesn't fall through into "Year".
            if "seconds" in _units:
                if abs(_diff - 3600) < 600:           # 1 hour ± 10 min
                    _detected = "Hour"
                elif abs(_diff - 10800) < 1800:       # 3 hour ± 30 min
                    _detected = "3Hour"
                elif abs(_diff - 21600) < 3600:       # 6 hour ± 1 hour
                    _detected = "6Hour"
                elif abs(_diff - 86400) < 7200:       # 1 day ± 2 hours
                    _detected = "Day"
                elif abs(_diff - 691200) < 86400:     # 8 day ± 1 day
                    _detected = "8Day"
                elif abs(_diff - 2592000) < 432000:   # 1 month ± 5 days
                    _detected = "Month"
                elif abs(_diff - 31536000) < 2592000: # 1 year ± 30 days
                    _detected = "Year"
                # else: unrecognized interval (e.g., 1800=30min, 7776000≈90d)
                #       leave _detected = None so caller can fall to default
            elif "hour" in _units:
                if abs(_diff - 1) < 0.2:              # 1 hour ± 12 min
                    _detected = "Hour"
                elif abs(_diff - 3) < 0.5:            # 3 hour
                    _detected = "3Hour"
                elif abs(_diff - 6) < 1.0:            # 6 hour
                    _detected = "6Hour"
                elif abs(_diff - 24) < 2.0:           # 1 day
                    _detected = "Day"
                elif abs(_diff - 192) < 24:           # 8 day
                    _detected = "8Day"
                elif 696 <= _diff <= 768:             # 1 month (29-32 days × 24h)
                    _detected = "Month"
                elif abs(_diff - 8760) < 720:         # 1 year ± 30 days
                    _detected = "Year"
            elif "day" in _units:
                if abs(_diff - 1) < 0.2:
                    _detected = "Day"
                elif abs(_diff - 8) < 1.0:
                    _detected = "8Day"
                elif 28 <= _diff <= 32:               # monthly: 28-32 days
                    _detected = "Month"
                elif abs(_diff - 365) < 30:
                    _detected = "Year"
            if _detected:
                result["detected_tim_res"] = _detected

        # Detect grid_res from lat dimension interval
        from openbench.data.coordinates import LAT_NAMES

        for _lat_name in LAT_NAMES:
            if _lat_name in _nc.variables and _lat_name in _nc.dimensions and _nc.dimensions[_lat_name].size > 1:
                _lat_vals = _nc.variables[_lat_name][:]
                _grid_res = round(abs(float(_lat_vals[1] - _lat_vals[0])), 4)
                if 0.001 < _grid_res < 10:  # sanity check
                    result["detected_grid_res"] = _grid_res
                break

        _nc.close()
    except Exception as _e:
        logger.debug("NC tim_res/grid_res detection failed: %s", _e)

    # Extract prefix and suffix from filename pattern
    # Pattern: <prefix><year><suffix>.nc
    # E.g., "E_2004_GLEAM_v4.2a.nc" → prefix="E_", suffix="_GLEAM_v4.2a"
    #
    # Year token rule: 4 consecutive digits NOT preceded by a letter or digit
    # and NOT followed by a digit. This excludes version markers like "v2010"
    # (preceded by "v") and longer numeric IDs like "200504" (would partially
    # match "2005" but it's followed by digits).
    _YEAR_TOKEN = re.compile(r"(?<![a-zA-Z\d])(\d{4})(?!\d)")

    fname = nc_files[0].stem  # Without .nc
    year_match = _YEAR_TOKEN.search(fname)
    if year_match:
        year_str = year_match.group(1)
        # Use match start (not .index) so we get the same position the regex
        # actually matched, even if the same 4-digit substring appears earlier
        # preceded by a letter (excluded by lookbehind).
        idx = year_match.start(1)
        prefix = fname[:idx]
        suffix = fname[idx + len(year_str):]
        result["prefix"] = prefix
        result["suffix"] = suffix
    else:
        # Single file (no year in filename): use full stem as prefix
        result["prefix"] = fname
        result["suffix"] = ""

    # Detect year range from all filenames using the same token rule
    years = []
    for f in nc_files:
        for m in _YEAR_TOKEN.finditer(f.stem):
            y = int(m.group(1))
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
    df.to_csv(output_csv, index=False)
    logger.info("Generated station list: %s (%d stations)", output_csv, len(df))

    return output_csv


def _parse_single_station_file(nc_file: Path) -> list | None:
    """Extract station info from a single-station NC file (fast, metadata only)."""
    try:
        import netCDF4
        import numpy as np

        nc = netCDF4.Dataset(str(nc_file), "r")

        # Extract station ID: NC variable/attribute first, then filename
        station_id = None

        # 1. Try NC variables (scalar or single-element)
        for id_var in ("station_id", "site_id", "station_name", "site"):
            if id_var in nc.variables:
                val = nc.variables[id_var][:]
                if hasattr(val, "item"):
                    station_id = str(val.item()).strip()
                elif hasattr(val, "__len__") and len(val) == 1:
                    station_id = str(val[0]).strip()
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
                lat = float(np.nanmean(val)) if hasattr(val, '__len__') and len(val) > 0 else float(val)
                break
        for name in LON_NAMES:
            if name in nc.variables:
                val = nc.variables[name][:]
                lon = float(np.nanmean(val)) if hasattr(val, '__len__') and len(val) > 0 else float(val)
                break

        # Extract time range from filename first (faster than reading time dim)
        syear = eyear = ""
        years = [int(m.group(1)) for m in re.finditer(r"(\d{4})", stem) if 1900 <= int(m.group(1)) <= 2100]
        if years:
            syear, eyear = min(years), max(years)
        elif "time" in nc.dimensions and nc.dimensions["time"].size > 0:
            # Fallback: read time variable
            try:
                import cftime
                time_var = nc.variables["time"]
                times = netCDF4.num2date(time_var[:], time_var.units, time_var.calendar if hasattr(time_var, 'calendar') else 'standard')
                syear = times[0].year
                eyear = times[-1].year
            except Exception:
                pass

        nc.close()

        if lat is not None and lon is not None:
            return [station_id, syear, eyear, lon, lat, str(nc_file)]
    except Exception as e:
        logger.debug("Failed to parse station file %s: %s", nc_file.name, e)

    return None


def _parse_merged_station_file(nc_file: Path, dataset_dir: Path) -> list:
    """Extract station info from a merged multi-station NC file."""
    rows = []
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        ds = xr.open_dataset(nc_file)

        # Find the station dimension (non-time dim)
        stn_dim = None
        time_dim = None
        for dim in ds.dims:
            if dim.lower() in ("time", "t"):
                time_dim = dim
            else:
                stn_dim = dim  # First non-time dimension is the station dim

        if not stn_dim:
            ds.close()
            return rows

        from openbench.data.coordinates import LAT_NAMES, LON_NAMES

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
            station_id = str(ds[stn_dim].values[i]) if stn_dim in ds.coords else str(i)
            lat = float(ds[lat_var].values[i]) if lat_var else None
            lon = float(ds[lon_var].values[i]) if lon_var else None

            if lat is not None and lon is not None and not (np.isnan(lat) or np.isnan(lon)):
                rows.append([station_id, syear, eyear, lon, lat, str(nc_file)])

        ds.close()
    except Exception as e:
        logger.debug("Failed to parse merged station file %s: %s", nc_file.name, e)

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
