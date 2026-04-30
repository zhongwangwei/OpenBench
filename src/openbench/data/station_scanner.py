"""Auto-scan station simulation directories to generate station lists.

Supports three directory layouts:

1. Flat (one NC per site):  root_dir/{any_name}.nc  — each file is one station
2. Nested single:           root_dir/{site_id}/{single_file}.nc
3. Nested multi (raw):      root_dir/{site_id}/*.nc or root_dir/{site_id}/history/*.nc
                            — multiple time-step files per site, needs merging

Detection logic:
  - Root has NC files directly → flat
  - Root has subdirectories, each with exactly 1 NC → nested single (no merge)
  - Root has subdirectories, each with >1 NC → nested multi (merge needed)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def scan_station_sim_dir(
    root_dir: str,
    output_dir: str | None = None,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Scan a station simulation directory and return a station list DataFrame.

    Auto-detects the directory layout and generates a station list with
    columns: ID, sim_lon, sim_lat, use_syear, use_eyear, sim_dir.

    Args:
        root_dir: Path to station simulation data directory.
        output_dir: Where to write merged files (only for multi-file sites).
                    If None, merges into root_dir itself.
        num_workers: Parallel workers for merging.

    Returns:
        DataFrame with station metadata.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Station sim directory not found: {root_dir}")

    layout = _detect_layout(root)
    logger.info("Station sim directory layout: %s (%s)", layout, root_dir)

    if layout == "flat":
        return _scan_flat(root)
    elif layout == "nested_single":
        return _scan_nested(root, merge=False)
    elif layout == "nested_multi":
        merge_dir = Path(output_dir) if output_dir else root
        return _scan_nested(root, merge=True, merge_dir=merge_dir, num_workers=num_workers)
    else:
        raise ValueError(
            f"Unrecognized station sim directory layout in {root_dir}. "
            "Expected NC files directly in root, or subdirectories with NC files inside."
        )


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_layout(root: Path) -> str:
    """Detect directory layout by inspecting contents.

    Returns: "flat", "nested_single", "nested_multi", or "unknown".
    """
    # Check root for NC files directly
    root_nc = list(root.glob("*.nc")) + list(root.glob("*.nc4"))
    if root_nc:
        return "flat"

    # Check subdirectories
    multi_file_sites = 0
    single_file_sites = 0
    total_sites = 0

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        # Collect NC files: check history/ first, then the subdir itself
        history = subdir / "history"
        if history.is_dir():
            nc_files = list(history.glob("*.nc")) + list(history.glob("*.nc4"))
        else:
            nc_files = list(subdir.glob("*.nc")) + list(subdir.glob("*.nc4"))

        if not nc_files:
            continue

        total_sites += 1
        if len(nc_files) > 1:
            multi_file_sites += 1
        else:
            single_file_sites += 1

    if total_sites == 0:
        return "unknown"

    # If majority of sites have multiple files → needs merging
    if multi_file_sites > single_file_sites:
        return "nested_multi"
    else:
        return "nested_single"


# ---------------------------------------------------------------------------
# Flat: root_dir/*.nc
# ---------------------------------------------------------------------------

# Common station file patterns:
#   sim_{site_id}_{syear}_{eyear}.nc
#   {prefix}_{site_id}_{syear}_{eyear}.nc
#   {site_id}_{syear}_{eyear}.nc
#   {site_id}.nc  (no year in name — read from file)
_PATTERN_WITH_YEARS = re.compile(
    r"^(?:.+_)?([A-Za-z]{2}[-_][A-Za-z0-9]+)_(\d{4})_(\d{4})\.nc4?$"
)
_PATTERN_ID_ONLY = re.compile(
    r"^(?:sim_)?([A-Za-z]{2}[-_][A-Za-z0-9]+)\.nc4?$"
)


def _scan_flat(root: Path) -> pd.DataFrame:
    """Scan flat directory of station NC files."""
    nc_files = sorted(root.glob("*.nc")) + sorted(root.glob("*.nc4"))
    logger.info("Scanning %d station files in flat directory", len(nc_files))

    records = []
    for nc_path in nc_files:
        record = _parse_station_file(nc_path)
        if record:
            records.append(record)

    if not records:
        raise FileNotFoundError(f"No valid station NC files found in {root}")

    df = pd.DataFrame(records)
    logger.info("Found %d stations", len(df))
    return df


def _parse_station_file(nc_path: Path) -> dict[str, Any] | None:
    """Parse a single station NC file to extract metadata."""
    name = nc_path.name

    # Try pattern with years: *_{site_id}_{syear}_{eyear}.nc
    m = _PATTERN_WITH_YEARS.match(name)
    if m:
        site_id = m.group(1)
        syear = int(m.group(2))
        eyear = int(m.group(3))
    else:
        # Try ID-only pattern or use stem as ID
        m2 = _PATTERN_ID_ONLY.match(name)
        site_id = m2.group(1) if m2 else nc_path.stem
        syear, eyear = None, None

    try:
        lon, lat = _read_coords(nc_path)
    except Exception as e:
        logger.warning("Skipping %s: cannot read coordinates: %s", name, e)
        return None

    # If years not in filename, read from time coordinate
    if syear is None or eyear is None:
        try:
            syear, eyear = _read_year_range(nc_path)
        except Exception as e:
            logger.warning("Skipping %s: cannot determine year range: %s", name, e)
            return None

    return {
        "ID": site_id,
        "sim_lon": lon,
        "sim_lat": lat,
        "use_syear": syear,
        "use_eyear": eyear,
        "sim_dir": str(nc_path),
    }


# ---------------------------------------------------------------------------
# Nested: root_dir/{site_id}/...
# ---------------------------------------------------------------------------

def _scan_nested(
    root: Path,
    merge: bool = False,
    merge_dir: Path | None = None,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Scan nested site directories, optionally merging multi-file sites."""
    sites: list[tuple[str, list[Path]]] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        # Look for NC files: history/ subfolder first, then directly
        history = subdir / "history"
        if history.is_dir():
            nc_files = sorted(history.glob("*.nc")) + sorted(history.glob("*.nc4"))
        else:
            nc_files = sorted(subdir.glob("*.nc")) + sorted(subdir.glob("*.nc4"))

        if nc_files:
            sites.append((subdir.name, nc_files))

    if not sites:
        raise FileNotFoundError(f"No site directories with NC files found in {root}")

    logger.info("Found %d sites (%s)", len(sites), "merging" if merge else "single-file")

    if not merge:
        # Single file per site — just scan
        records = []
        for site_id, nc_files in sites:
            record = _parse_station_file(nc_files[0])
            if record:
                record["ID"] = site_id  # Use directory name as site ID
                records.append(record)
    else:
        # Multiple files per site — merge
        if merge_dir is None:
            merge_dir = root
        merge_dir.mkdir(parents=True, exist_ok=True)

        args_list = [(site_id, nc_files, merge_dir) for site_id, nc_files in sites]
        records = []

        if num_workers > 1 and len(args_list) > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_merge_site, *a): a[0] for a in args_list}
                for future in as_completed(futures):
                    site_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            records.append(result)
                    except Exception as e:
                        logger.warning("Failed to merge site %s: %s", site_id, e)
        else:
            for a in args_list:
                try:
                    result = _merge_site(*a)
                    if result:
                        records.append(result)
                except Exception as e:
                    logger.warning("Failed to merge site %s: %s", a[0], e)

    if not records:
        raise FileNotFoundError(f"No stations successfully processed from {root}")

    df = pd.DataFrame(records)
    logger.info("Scanned %d stations", len(df))
    return df


def _merge_site(
    site_id: str, nc_files: list[Path], output_dir: Path
) -> dict[str, Any] | None:
    """Merge multiple NC files for one site into a single file."""
    # Check if already merged
    merged_pattern = list(output_dir.glob(f"sim_{site_id}_*.nc"))
    if merged_pattern:
        # Already merged — just parse the existing file
        return _parse_station_file(merged_pattern[0])

    try:
        lon, lat = _read_coords(nc_files[0])

        # `with` guarantees close even if to_netcdf raises; otherwise the
        # mfdataset handle leaks and HDF5 lock state lingers, which under
        # joblib parallel scans manifests as random "file already open"
        # / OSError on subsequent passes.
        with xr.open_mfdataset(
            [str(f) for f in nc_files],
            combine="by_coords",
            data_vars="minimal",
            compat="override",
            coords="minimal",
            parallel=False,
        ) as ds:
            times = pd.to_datetime(ds.time.values)
            syear = int(times.min().year)
            eyear = int(times.max().year)

            merged_path = output_dir / f"sim_{site_id}_{syear}_{eyear}.nc"
            ds.to_netcdf(str(merged_path))

        return {
            "ID": site_id,
            "sim_lon": lon,
            "sim_lat": lat,
            "use_syear": syear,
            "use_eyear": eyear,
            "sim_dir": str(merged_path),
        }
    except Exception as e:
        logger.warning("Error merging site %s: %s", site_id, e)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_coords(nc_path: Path) -> tuple[float, float]:
    """Read longitude and latitude from a station NC file."""
    with xr.open_dataset(str(nc_path)) as ds:
        lon = _extract_scalar(ds, ["lon", "longitude", "LON", "LONGITUDE", "x"])
        lat = _extract_scalar(ds, ["lat", "latitude", "LAT", "LATITUDE", "y"])

        if lon is None or lat is None:
            raise ValueError(f"Cannot find lon/lat in {nc_path.name}")
        return float(lon), float(lat)


def _read_year_range(nc_path: Path) -> tuple[int, int]:
    """Read year range from the time coordinate of a NC file."""
    with xr.open_dataset(str(nc_path)) as ds:
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"No time coordinate in {nc_path.name}")
        times = pd.to_datetime(ds.time.values)
        return int(times.min().year), int(times.max().year)


def _extract_scalar(ds: xr.Dataset, names: list[str]) -> float | None:
    """Extract a scalar value by trying multiple variable/coordinate names."""
    for name in names:
        if name in ds.coords:
            val = ds.coords[name].values
            return float(val.item()) if val.ndim == 0 else float(val.flat[0])
        if name in ds.data_vars:
            val = ds[name].values
            return float(val.item()) if val.ndim == 0 else float(val.flat[0])
    return None
