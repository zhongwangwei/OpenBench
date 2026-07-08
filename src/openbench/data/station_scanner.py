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

from openbench.util.dataset_loader import open_mfdataset as _open_mfdataset_chunked
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic

logger = logging.getLogger(__name__)


def scan_station_sim_dir(
    root_dir: str,
    output_dir: str | None = None,
    num_workers: int = 4,
    return_dropped: bool = False,
):
    """Scan a station simulation directory and return a station list DataFrame.

    Auto-detects the directory layout and generates a station list with
    columns: ID, sim_lon, sim_lat, use_syear, use_eyear, sim_dir.

    Args:
        root_dir: Path to station simulation data directory.
        output_dir: Where to write merged files (only for multi-file sites).
                    If None, merges into root_dir itself.
        num_workers: Parallel workers for merging.
        return_dropped: If True, return ``(df, dropped_site_ids)`` so callers
            can decide whether to fail on partial results.

    Returns:
        DataFrame with station metadata, or a tuple of ``(df, dropped)`` when
        ``return_dropped`` is set.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Station sim directory not found: {root_dir}")

    layout = _detect_layout(root)
    metadata = _load_station_metadata(root)
    logger.info("Station sim directory layout: %s (%s)", layout, root_dir)

    if layout == "flat":
        df, dropped = _scan_flat(root, metadata=metadata)
    elif layout == "nested_single":
        df, dropped = _scan_nested(root, merge=False, metadata=metadata)
    elif layout == "nested_multi":
        merge_dir = Path(output_dir) if output_dir else root
        df, dropped = _scan_nested(
            root,
            merge=True,
            merge_dir=merge_dir,
            num_workers=num_workers,
            metadata=metadata,
        )
    else:
        raise ValueError(
            f"Unrecognized station sim directory layout in {root_dir}. "
            "Expected NC files directly in root, or subdirectories with NC files inside."
        )

    if return_dropped:
        return df, dropped
    return df


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

    for subdir in _iter_station_child_dirs(root):
        nc_files = _station_nc_files(subdir)

        if not nc_files:
            continue

        total_sites += 1
        if len(nc_files) > 1:
            multi_file_sites += 1
        else:
            single_file_sites += 1

    if total_sites == 0:
        return "unknown"

    # If any site has multiple files, use the merge path for every site so
    # mixed layouts do not silently drop later files for the multi-file sites.
    if multi_file_sites > 0:
        return "nested_multi"
    return "nested_single"


# ---------------------------------------------------------------------------
# Flat: root_dir/*.nc
# ---------------------------------------------------------------------------

# Common station file patterns:
#   sim_{site_id}_{syear}_{eyear}.nc
#   {prefix}_{site_id}_{syear}_{eyear}.nc
#   {site_id}_{syear}_{eyear}.nc
#   {site_id}.nc  (no year in name — read from file)
_PATTERN_WITH_YEARS = re.compile(r"^(?:.+_)?([A-Za-z]{2}[-_][A-Za-z0-9]+)_(\d{4})(?:[-_]?)(\d{4})\.nc4?$")
_PATTERN_ID_ONLY = re.compile(r"^(?:sim_)?([A-Za-z]{2}[-_][A-Za-z0-9]+)\.nc4?$")


def _scan_flat(root: Path, *, metadata: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Scan flat directory of station NC files."""
    nc_files = sorted(root.glob("*.nc")) + sorted(root.glob("*.nc4"))
    logger.info("Scanning %d station files in flat directory", len(nc_files))

    records = []
    dropped: list[str] = []
    for nc_path in nc_files:
        record = _parse_station_file(nc_path, metadata=metadata)
        if record:
            records.append(record)
        else:
            dropped.append(nc_path.stem)

    if not records:
        raise FileNotFoundError(f"No valid station NC files found in {root}")

    df = pd.DataFrame(records)
    logger.info("Found %d stations (%d dropped)", len(df), len(dropped))
    return df, dropped


def _parse_station_file(
    nc_path: Path,
    *,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
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

    meta_row = _station_metadata_row(metadata, site_id, nc_path)
    try:
        lon, lat = _read_coords(nc_path)
    except Exception as e:
        meta_coords = _metadata_coords(meta_row)
        if meta_coords:
            lon, lat = meta_coords
        else:
            logger.warning("Using blank coordinates for %s: %s", name, e)
            lon = lat = float("nan")

    # If years not in filename, read from time coordinate
    if syear is None or eyear is None:
        meta_years = _metadata_years(meta_row)
        if meta_years:
            syear, eyear = meta_years
        else:
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
    metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Scan nested site directories, optionally merging multi-file sites."""
    sites: list[tuple[str, list[Path]]] = []

    for subdir in _iter_station_child_dirs(root):
        nc_files = _station_nc_files(subdir)

        if nc_files:
            sites.append((subdir.name, nc_files))

    if not sites:
        raise FileNotFoundError(f"No site directories with NC files found in {root}")

    logger.info("Found %d sites (%s)", len(sites), "merging" if merge else "single-file")

    dropped: list[str] = []
    if not merge:
        records = []
        for site_id, nc_files in sites:
            record = _parse_station_file(nc_files[0], metadata=metadata)
            if record:
                record["ID"] = site_id
                records.append(record)
            else:
                dropped.append(site_id)
    else:
        if merge_dir is None:
            merge_dir = root
        merge_dir.mkdir(parents=True, exist_ok=True)

        args_list = [(site_id, nc_files, merge_dir, metadata) for site_id, nc_files in sites]
        records = []

        if num_workers > 1 and len(args_list) > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_merge_site, *a): a[0] for a in args_list}
                for future in as_completed(futures):
                    site_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.warning("Failed to merge site %s: %s", site_id, e)
                        dropped.append(site_id)
                        continue
                    if result:
                        records.append(result)
                    else:
                        dropped.append(site_id)
        else:
            for site_id, nc_files, dir_, meta in args_list:
                try:
                    result = _merge_site(site_id, nc_files, dir_, meta)
                except Exception as e:
                    logger.warning("Failed to merge site %s: %s", site_id, e)
                    dropped.append(site_id)
                    continue
                if result:
                    records.append(result)
                else:
                    dropped.append(site_id)

    if not records:
        raise FileNotFoundError(f"No stations successfully processed from {root}")

    df = pd.DataFrame(records)
    logger.info("Scanned %d stations (%d dropped)", len(df), len(dropped))
    return df, dropped


def _iter_station_child_dirs(root: Path) -> list[Path]:
    """Return unique child directories without following symlink cycles.

    Station scanning is intentionally shallow, but a child symlink pointing
    back to ``root`` (or two child links to the same real directory) can still
    duplicate work and confuse layout detection. Keep the first occurrence of
    each real directory and skip links that resolve to the root itself.
    """
    try:
        root_real = root.resolve(strict=False)
    except OSError:
        root_real = root.absolute()

    seen = {root_real}
    children: list[Path] = []
    try:
        candidates = sorted(root.iterdir())
    except OSError:
        return children

    for child in candidates:
        if not child.is_dir():
            continue
        try:
            child_real = child.resolve(strict=False)
        except OSError:
            child_real = child.absolute()
        if child_real in seen:
            logger.debug("Skipping duplicate or cyclic station directory: %s", child)
            continue
        seen.add(child_real)
        children.append(child)
    return children


def _station_nc_files(site_dir: Path) -> list[Path]:
    """Return station time-series NC files for a child directory.

    CoLM-style case directories often contain siblings such as ``history/``,
    ``landdata/`` and ``restart/``.  Only the history stream is a station time
    series; static auxiliary NetCDF files must not be counted as sites.
    """

    history = site_dir / "history"
    if history.is_dir():
        nc_files = sorted(history.glob("*.nc")) + sorted(history.glob("*.nc4"))
    else:
        nc_files = sorted(site_dir.glob("*.nc")) + sorted(site_dir.glob("*.nc4"))

    if not nc_files:
        return []

    if not _has_valid_time_axis(nc_files[0]):
        logger.debug("Skipping non-time-series station candidate: %s", site_dir)
        return []

    return nc_files


def _has_valid_time_axis(nc_path: Path) -> bool:
    try:
        _read_year_range(nc_path)
        return True
    except Exception:
        return False


def _merge_site(
    site_id: str,
    nc_files: list[Path],
    output_dir: Path,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Merge multiple NC files for one site into a single file."""
    try:
        source_syear, source_eyear = _combined_year_range(nc_files)
        for merged_file in sorted(output_dir.glob(f"sim_{site_id}_*.nc")):
            if _merged_file_is_current(merged_file, nc_files, source_syear, source_eyear):
                return _parse_station_file(merged_file, metadata=metadata)

        meta_row = _station_metadata_row(metadata, site_id, nc_files[0])
        try:
            lon, lat = _read_coords(nc_files[0])
        except Exception as e:
            meta_coords = _metadata_coords(meta_row)
            if meta_coords:
                lon, lat = meta_coords
            else:
                logger.warning("Using blank coordinates for site %s: %s", site_id, e)
                lon = lat = float("nan")

        # Load before writing. ``open_mfdataset`` returns dask-backed arrays;
        # passing that dataset directly to ``to_netcdf`` makes the write path
        # read source NetCDF files while the target HDF5 write is active.  On
        # some CI runners that can hang in netCDF4/HDF5.  Eager loading keeps
        # the source-file read phase and target-file write phase separate.
        with _open_mfdataset_chunked(
            [str(f) for f in nc_files],
            combine="by_coords",
            data_vars="minimal",
            compat="override",
            coords="minimal",
            parallel=False,
        ) as ds:
            loaded = ds.load()
            times = pd.to_datetime(ds.time.values)
            syear = int(times.min().year)
            eyear = int(times.max().year)

        merged_path = output_dir / f"sim_{site_id}_{syear}_{eyear}.nc"
        try:
            _write_netcdf_atomic(loaded, merged_path)
        finally:
            loaded.close()

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
        import numpy as np

        time_var = ds["time"]
        values = time_var.values
        units = str(time_var.attrs.get("units", ""))
        if np.issubdtype(np.asarray(values).dtype, np.number) and not units.strip():
            raise ValueError(f"Numeric time coordinate has no units in {nc_path.name}")
        times = pd.to_datetime(ds.time.values)
        return int(times.min().year), int(times.max().year)


def _combined_year_range(nc_files: list[Path]) -> tuple[int, int]:
    ranges = [_read_year_range(path) for path in nc_files]
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def _merged_file_is_current(merged_file: Path, nc_files: list[Path], source_syear: int, source_eyear: int) -> bool:
    """Return True only when a merged station file matches current inputs."""
    try:
        merged_syear, merged_eyear = _read_year_range(merged_file)
    except Exception as exc:
        logger.debug("Ignoring unreadable merged station file %s: %s", merged_file, exc)
        return False
    if (merged_syear, merged_eyear) != (source_syear, source_eyear):
        logger.info(
            "Ignoring stale merged station file %s: years %s-%s != current %s-%s",
            merged_file,
            merged_syear,
            merged_eyear,
            source_syear,
            source_eyear,
        )
        return False
    try:
        newest_source_mtime = max(path.stat().st_mtime_ns for path in nc_files)
        if merged_file.stat().st_mtime_ns < newest_source_mtime:
            logger.info("Ignoring stale merged station file %s: source files are newer", merged_file)
            return False
    except OSError as exc:
        logger.debug("Could not compare station merge mtimes for %s: %s", merged_file, exc)
        return False
    return True


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


def _load_station_metadata(root: Path) -> pd.DataFrame | None:
    """Load sidecar station metadata near a station simulation directory."""
    names = ("station_case.csv", "stations.csv", "station_list.csv", "station_list.txt")
    candidates = [*(root / name for name in names), *(root.parent / name for name in names)]
    candidates.extend(sorted(root.glob("*_stations.csv")))
    candidates.extend(sorted(root.glob("*_list.csv")))
    candidates.extend(sorted(root.glob("*_list.txt")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Could not read station metadata %s: %s", path, exc)
            continue
        if _metadata_column(data, ["ID", "id", "site_id", "site"]) is not None:
            return data
    return None


def _metadata_column(data: pd.DataFrame, names: list[str]) -> str | None:
    columns = {str(column).lower(): column for column in data.columns}
    for name in names:
        column = columns.get(name.lower())
        if column is not None:
            return column
    return None


def _station_metadata_row(
    data: pd.DataFrame | None,
    site_id: str,
    nc_path: Path,
) -> pd.Series | None:
    if data is None or data.empty:
        return None

    id_col = _metadata_column(data, ["ID", "id", "site_id", "site"])
    if id_col is not None:
        matches = data[data[id_col].astype(str) == site_id]
        if not matches.empty:
            return matches.iloc[0]

    dir_col = _metadata_column(data, ["DIR", "dir", "sim_dir", "path"])
    if dir_col is not None:
        filename = nc_path.name
        matches = data[data[dir_col].astype(str).str.contains(filename, regex=False, na=False)]
        if not matches.empty:
            return matches.iloc[0]
    return None


def _metadata_coords(row: pd.Series | None) -> tuple[float, float] | None:
    if row is None:
        return None
    lon_col = _row_key(row, ["LON", "lon", "sim_lon", "longitude"])
    lat_col = _row_key(row, ["LAT", "lat", "sim_lat", "latitude"])
    if lon_col is None or lat_col is None:
        return None
    if pd.isna(row[lon_col]) or pd.isna(row[lat_col]):
        return None
    return float(row[lon_col]), float(row[lat_col])


def _metadata_years(row: pd.Series | None) -> tuple[int, int] | None:
    if row is None:
        return None
    start_col = _row_key(row, ["SYEAR", "syear", "use_syear", "start_year"])
    end_col = _row_key(row, ["EYEAR", "eyear", "use_eyear", "end_year"])
    if start_col is None or end_col is None:
        return None
    if pd.isna(row[start_col]) or pd.isna(row[end_col]):
        return None
    return int(row[start_col]), int(row[end_col])


def _row_key(row: pd.Series, names: list[str]) -> str | None:
    keys = {str(key).lower(): key for key in row.index}
    for name in names:
        key = keys.get(name.lower())
        if key is not None:
            return key
    return None
