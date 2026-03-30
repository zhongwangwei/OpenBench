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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

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


def scan_reference_directory(ref_root: str | Path) -> list[DatasetGroup]:
    """Scan a reference data directory and discover all datasets.

    Args:
        ref_root: Root directory (e.g., /Volumes/work/Reference)

    Returns:
        List of DatasetGroup, each containing resolution variants.
    """
    ref_root = Path(ref_root)
    if not ref_root.exists():
        logger.warning("Reference directory not found: %s", ref_root)
        return []

    groups: dict[str, DatasetGroup] = {}

    # Scan grid data: Grid/{LowRes,MidRes,HigRes}/<category>/<variable>/<dataset>/
    grid_dir = ref_root / "Grid"
    if grid_dir.exists():
        for res_name in ["LowRes", "MidRes", "HigRes"]:
            res_dir = grid_dir / res_name
            if not res_dir.exists():
                continue

            for category_dir in _iter_dirs(res_dir):
                cat_name = category_dir.name
                category = CATEGORY_MAP.get(cat_name, cat_name)

                for var_dir in _iter_dirs(category_dir):
                    var_name = var_dir.name

                    for dataset_dir in _iter_dirs(var_dir):
                        dataset_name = dataset_dir.name
                        nc_count = len(list(dataset_dir.glob("*.nc")))
                        if nc_count == 0:
                            continue

                        # Detect time resolution from filenames
                        tim_res = _detect_tim_res(dataset_dir)

                        if dataset_name not in groups:
                            groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                        if res_name not in groups[dataset_name].variants:
                            groups[dataset_name].variants[res_name] = ScannedDataset(
                                name=dataset_name,
                                resolution=res_name,
                                category=category,
                                data_type="grid",
                                root_dir=str(category_dir.parent),  # e.g., .../Grid/LowRes
                                tim_res=tim_res,
                            )

                        scanned = groups[dataset_name].variants[res_name]
                        scanned.variables[var_name] = str(var_dir.relative_to(category_dir.parent))
                        scanned.file_count += nc_count

    # Scan station data: Station/<category>/<variable>/<dataset>/
    stn_dir = ref_root / "Station"
    if stn_dir.exists():
        for category_dir in _iter_dirs(stn_dir):
            cat_name = category_dir.name
            category = CATEGORY_MAP.get(cat_name, cat_name)

            for var_dir in _iter_dirs(category_dir):
                var_name = var_dir.name

                for dataset_dir in _iter_dirs(var_dir):
                    dataset_name = dataset_dir.name
                    nc_count = len(list(dataset_dir.glob("*.nc")))
                    if nc_count == 0:
                        continue

                    if dataset_name not in groups:
                        groups[dataset_name] = DatasetGroup(base_name=dataset_name)

                    if "Station" not in groups[dataset_name].variants:
                        groups[dataset_name].variants["Station"] = ScannedDataset(
                            name=dataset_name,
                            resolution="Station",
                            category=category,
                            data_type="stn",
                            root_dir=str(category_dir.parent),
                        )

                    scanned = groups[dataset_name].variants["Station"]
                    scanned.variables[var_name] = str(var_dir.relative_to(category_dir.parent))
                    scanned.file_count += nc_count

    return sorted(groups.values(), key=lambda g: g.base_name)


def find_new_datasets(
    ref_root: str | Path,
    existing_names: Optional[set[str]] = None,
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

    all_groups = scan_reference_directory(ref_root)
    new_groups = []

    for group in all_groups:
        has_new = False
        for res, variant in group.variants.items():
            if variant.registry_name not in existing_names:
                has_new = True
        if has_new:
            new_groups.append(group)

    return new_groups


def register_scanned_dataset(
    scanned: ScannedDataset,
    catalog_path: Optional[Path] = None,
    existing_descriptor: Optional[dict] = None,
) -> Path:
    """Register a scanned dataset into the user catalog.

    Appends to the user's reference_catalog.yaml (single file, not individual files).

    Args:
        scanned: The scanned dataset to register.
        catalog_path: Path to the catalog YAML file.
            Defaults to ~/.openbench/reference_catalog.yaml.
        existing_descriptor: Optional existing descriptor to merge with
            (preserves hand-edited fields like varname, varunit).

    Returns:
        Path to the catalog file.
    """
    if catalog_path is None:
        try:
            from platformdirs import user_config_dir

            user_dir = Path(user_config_dir("openbench"))
        except ImportError:
            user_dir = Path.home() / ".openbench"
        catalog_path = user_dir / "reference_catalog.yaml"

    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # Build descriptor
    descriptor = {
        "name": scanned.registry_name,
        "description": f"{scanned.name} reference dataset ({scanned.resolution})",
        "category": scanned.category,
        "data_type": scanned.data_type,
        "tim_res": scanned.tim_res or "Month",
        "data_groupby": "Year",
        "timezone": 0,
        "years": [1980, 2023],  # Default, user should verify
        "root_dir": scanned.root_dir,
    }

    if scanned.data_type == "grid":
        res_info = RESOLUTION_MAP.get(scanned.resolution, {})
        descriptor["grid_res"] = res_info.get("typical_grid_res", 0.25)

    # Build variables section — use existing descriptor if available
    variables = {}
    for var_name, sub_dir in scanned.variables.items():
        var_entry = {"varname": var_name, "varunit": "", "sub_dir": sub_dir}

        # Merge from existing descriptor if available
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

        variables[var_name] = var_entry

    descriptor["variables"] = variables

    # Load existing catalog, append, write back
    catalog = {}
    if catalog_path.exists():
        try:
            with open(catalog_path) as f:
                catalog = yaml.safe_load(f) or {}
        except Exception:
            catalog = {}

    catalog[scanned.registry_name] = descriptor

    with open(catalog_path, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return catalog_path


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
    """Detect time resolution from filename patterns."""
    nc_files = list(dataset_dir.glob("*.nc"))
    if not nc_files:
        return ""

    name = nc_files[0].stem.lower()
    if "daily" in name or "_daily" in str(dataset_dir).lower():
        return "Day"
    if "hourly" in name or "_hourly" in str(dataset_dir).lower():
        return "Hour"
    if "3hour" in name or "3h" in name:
        return "3Hour"
    if "8daily" in name or "8day" in name:
        return "8Day"

    # Check if parent directory hints at resolution
    dir_str = str(dataset_dir).lower()
    if "daily" in dir_str:
        return "Day"
    if "hourly" in dir_str:
        return "Hour"

    return "Month"  # Default assumption
