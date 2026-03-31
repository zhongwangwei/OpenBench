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
                        scanned.variables[var_name] = str(dataset_dir.relative_to(category_dir.parent))
                        scanned.file_count += nc_count

            # Note: Composite directory is skipped — its structure is non-standard.
            # Use 'openbench data register' to manually register Composite datasets.

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
                    scanned.variables[var_name] = str(dataset_dir.relative_to(category_dir.parent))
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
        from openbench.data.registry.manager import get_writable_registry_dir

        catalog_path = get_writable_registry_dir() / "reference_catalog.yaml"

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

    # Build variables section
    # Priority: existing_descriptor > NC file inspection > directory name fallback
    variables = {}
    for var_name, sub_dir in scanned.variables.items():
        var_entry: dict[str, Any] = {"varname": var_name, "varunit": "", "sub_dir": sub_dir}

        # 1. Try existing descriptor (hand-curated, most reliable)
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

        # 2. If no existing descriptor, inspect NC file for varname/unit/prefix/suffix
        if not merged_from_existing:
            dataset_path = Path(scanned.root_dir) / sub_dir
            if dataset_path.is_dir():
                nc_info = _inspect_nc_file(dataset_path)

                # Multi-variable: if NC has 2+ data vars, ask user to confirm
                all_vars = nc_info.get("all_data_vars", [])
                if len(all_vars) > 1 and on_multi_var:
                    chosen = on_multi_var(var_name, sub_dir, all_vars)
                    if chosen:
                        nc_info["varname"] = chosen
                        # Find unit for chosen var
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
                # Update year range if detected
                if nc_info.get("syear"):
                    descriptor["years"] = [nc_info["syear"], nc_info.get("eyear", nc_info["syear"])]

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


def _inspect_nc_file(dataset_dir: Path) -> dict:
    """Inspect a NetCDF file to extract variable name, unit, prefix, suffix.

    Opens the first .nc file in the directory, finds the primary data
    variable (skips time_bnds, lat, lon, etc.), and extracts metadata.
    Also parses the filename to detect prefix and suffix patterns.

    Returns:
        {"varname": str, "varunit": str, "prefix": str, "suffix": str}
        or empty dict if inspection fails.
    """
    nc_files = sorted(dataset_dir.glob("*.nc"))
    if not nc_files:
        return {}

    result = {}

    # Extract varname and unit from NC contents
    try:
        import xarray as xr

        ds = xr.open_dataset(nc_files[0], engine="netcdf4")

        # Filter out auxiliary variables
        skip_vars = {
            "time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
            "lat_bounds", "lon_bounds", "crs", "spatial_ref",
        }
        data_vars = [v for v in ds.data_vars if v not in skip_vars and len(ds[v].dims) >= 2]

        # Store ALL data variables for multi-var detection
        result["all_data_vars"] = []
        for dv in data_vars:
            da = ds[dv]
            unit = da.attrs.get("units", da.attrs.get("unit", ""))
            unit = str(unit).replace(".", " ").strip() if unit else ""
            long_name = da.attrs.get("long_name", "")
            standard_name = da.attrs.get("standard_name", "")
            result["all_data_vars"].append({
                "name": dv, "unit": unit, "dims": list(da.dims),
                "long_name": long_name, "standard_name": standard_name,
            })

        if data_vars:
            varname = data_vars[0]
            varunit = result["all_data_vars"][0]["unit"]
            result["varname"] = varname
            result["varunit"] = varunit

        ds.close()
    except Exception:
        pass

    # Extract prefix and suffix from filename pattern
    # Pattern: <prefix><year><suffix>.nc
    # E.g., "E_2004_GLEAM_v4.2a.nc" → prefix="E_", suffix="_GLEAM_v4.2a"
    import re

    fname = nc_files[0].stem  # Without .nc
    # Try to find a 4-digit year in the filename
    year_match = re.search(r"(\d{4})", fname)
    if year_match:
        year_str = year_match.group(1)
        idx = fname.index(year_str)
        prefix = fname[:idx]
        suffix = fname[idx + len(year_str):]
        result["prefix"] = prefix
        result["suffix"] = suffix
    else:
        result["prefix"] = ""
        result["suffix"] = ""

    # Detect year range from all filenames
    years = []
    for f in nc_files:
        for m in re.finditer(r"(\d{4})", f.stem):
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

    nc_files = sorted(dataset_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NC files found in {dataset_dir}")

    rows = []

    # Try one-file-per-station first
    for nc_file in nc_files:
        row = _parse_single_station_file(nc_file)
        if row:
            rows.append(row)

    # If that found very few stations but there are large merged files, try merged parsing
    if len(rows) < len(nc_files) // 2:
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
    """Extract station info from a single-station NC file."""
    try:
        import xarray as xr

        ds = xr.open_dataset(nc_file)

        # Extract station ID from filename
        # Pattern: AU-How_2004-2005_OzFlux_Flux.nc → ID=AU-How
        stem = nc_file.stem
        station_id = stem.split("_")[0]

        # Extract lat/lon
        lat = _extract_scalar(ds, ["latitude", "lat", "LAT", "Latitude"])
        lon = _extract_scalar(ds, ["longitude", "lon", "LON", "Longitude"])

        # Extract time range
        if "time" in ds.dims and len(ds.time) > 0:
            syear = int(str(ds.time.values[0])[:4])
            eyear = int(str(ds.time.values[-1])[:4])
        else:
            syear = ""
            eyear = ""

        ds.close()

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

        n_stations = ds.sizes[stn_dim]
        lat_var = _find_var(ds, ["Lat", "lat", "LAT", "latitude", "Latitude"])
        lon_var = _find_var(ds, ["Lon", "lon", "LON", "longitude", "Longitude"])

        # Time range (case-insensitive dim name)
        syear = ""
        eyear = ""
        if time_dim and ds.sizes[time_dim] > 0:
            time_vals = ds[time_dim].values
            syear = int(str(time_vals[0])[:4])
            eyear = int(str(time_vals[-1])[:4])

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
