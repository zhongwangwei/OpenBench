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
    output_dir: Optional[Path] = None,
    existing_descriptor: Optional[dict] = None,
) -> Path:
    """Write a registry descriptor YAML for a scanned dataset.

    Args:
        scanned: The scanned dataset to register.
        output_dir: Directory to write the descriptor.
            Defaults to ~/.openbench/references/.
        existing_descriptor: Optional existing descriptor to merge with
            (preserves hand-edited fields like varname, varunit).

    Returns:
        Path to the written YAML file.
    """
    if output_dir is None:
        try:
            from platformdirs import user_config_dir

            output_dir = Path(user_config_dir("openbench")) / "references"
        except ImportError:
            output_dir = Path.home() / ".openbench" / "references"

    output_dir.mkdir(parents=True, exist_ok=True)

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

    out_path = output_dir / f"{scanned.registry_name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(descriptor, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return out_path


def get_compatible_resolutions(
    group: DatasetGroup,
    required_tim_res: Optional[str] = None,
) -> list[str]:
    """Get resolutions compatible with a time resolution constraint.

    Rule: if daily data is available, monthly is not allowed.

    Args:
        group: DatasetGroup with resolution variants.
        required_tim_res: Required time resolution (e.g., "Day", "Month").

    Returns:
        List of compatible resolution names.
    """
    if not required_tim_res:
        return group.available_resolutions

    compatible = []
    has_daily = any(
        v.tim_res in ("Day", "day", "daily", "D")
        for v in group.variants.values()
    )

    for res_name, variant in group.variants.items():
        vtim = variant.tim_res.lower() if variant.tim_res else ""

        if required_tim_res.lower() in ("day", "d", "daily"):
            # Daily requested: only allow daily or higher frequency
            if vtim in ("day", "d", "daily", "hour", "h", "hourly", "3h", "6h"):
                compatible.append(res_name)
        elif required_tim_res.lower() in ("month", "m", "monthly"):
            # Monthly requested: allow monthly, but NOT if daily exists for same dataset
            if has_daily:
                continue  # Skip monthly when daily is available
            compatible.append(res_name)
        else:
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
