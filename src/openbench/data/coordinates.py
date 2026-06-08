"""Canonical coordinate name definitions — single source of truth.

All modules that need to identify or rename lat/lon/time coordinates
should import from here instead of maintaining their own mapping.
"""

# Maps non-standard coordinate names → canonical names.
# Used by processing, evaluation, visualization, scanner, etc.
COORDINATE_MAP: dict[str, str] = {
    # Longitude variants → "lon"
    "longitude": "lon",
    "Longitude": "lon",
    "LONGITUDE": "lon",
    "LON": "lon",
    "long": "lon",
    "Long": "lon",
    "LONG": "lon",
    "lon_cama": "lon",
    "lon_ucat": "lon",
    "lon0": "lon",
    "x": "lon",
    "X": "lon",
    "XLONG": "lon",
    "nav_lon": "lon",
    # Latitude variants → "lat"
    "latitude": "lat",
    "Latitude": "lat",
    "LATITUDE": "lat",
    "LAT": "lat",
    "lat_cama": "lat",
    "lat_ucat": "lat",
    "lat0": "lat",
    "y": "lat",
    "Y": "lat",
    "XLAT": "lat",
    "nav_lat": "lat",
    # Time variants → "time"
    "Time": "time",
    "TIME": "time",
    "t": "time",
    "T": "time",
    "XTIME": "time",
    # WRF dimension names → standard
    "south_north": "lat",
    "west_east": "lon",
    "bottom_top": "level",
    "soil_layers_stag": "soil",
}

VERTICAL_COORDINATE_MAP: dict[str, str] = {
    "elevation": "elev",
    "Elevation": "elev",
    "ELEV": "elev",
    "height": "elev",
    "HEIGHT": "elev",
    "z": "elev",
    "Z": "elev",
    "h": "elev",
    "H": "elev",
    "alt": "elev",
    "altitude": "elev",
}

COORDINATE_MAP_WITH_VERTICAL: dict[str, str] = {
    **COORDINATE_MAP,
    **VERTICAL_COORDINATE_MAP,
}

# Ordered lists for probing — first match wins.
# Use these when searching for a coordinate in an unknown dataset.
LAT_NAMES: tuple[str, ...] = (
    "lat",
    "latitude",
    "Latitude",
    "LAT",
    "LATITUDE",
    "lat_cama",
    "lat_ucat",
    "lat0",
    "nav_lat",
    "y",
    "Y",
    "XLAT",
)

LON_NAMES: tuple[str, ...] = (
    "lon",
    "longitude",
    "Longitude",
    "LON",
    "LONGITUDE",
    "long",
    "Long",
    "LONG",
    "lon_cama",
    "lon_ucat",
    "lon0",
    "nav_lon",
    "x",
    "X",
    "XLONG",
)

# Station/site dimension names (lowercase for case-insensitive matching)
STN_DIM_NAMES: frozenset[str] = frozenset(
    {
        "station",
        "site",
        "sites",
        "stations",
        "nstations",
        "nstation",
        "location",
        "locations",
        "point",
        "points",
        "stid",
    }
)


def find_lat_name(names) -> str | None:
    """Find the latitude coordinate name in a collection of names."""
    name_set = set(names)
    for candidate in LAT_NAMES:
        if candidate in name_set:
            return candidate
    return None


def find_lon_name(names) -> str | None:
    """Find the longitude coordinate name in a collection of names."""
    name_set = set(names)
    for candidate in LON_NAMES:
        if candidate in name_set:
            return candidate
    return None


# --- NetCDF file extensions ---

NC_EXTENSIONS: tuple[str, ...] = ("*.nc", "*.nc4", "*.NC", "*.NC4")


def glob_nc(directory, recursive: bool = False) -> list:
    """Glob for NetCDF files (both .nc and .nc4) in a directory.

    Args:
        directory: Path or str to search.
        recursive: If True, search subdirectories too.

    Returns:
        Sorted list of Path objects.
    """
    from pathlib import Path

    d = Path(directory)
    results = []
    for ext in NC_EXTENSIONS:
        if recursive:
            results.extend(d.rglob(ext))
        else:
            results.extend(d.glob(ext))
    return sorted(set(results))


# Raw extension strings for use in regex and os.path patterns.
NC_SUFFIXES: tuple[str, ...] = (".nc", ".nc4", ".NC", ".NC4")


def glob_nc_pattern(pattern: str) -> list[str]:
    """Glob with a pattern, trying both .nc and .nc4 extensions.

    The *pattern* must end with ``.nc``; this function automatically
    also tries the same pattern with ``.nc4``.

    Returns:
        Sorted list of matched file path strings.
    """
    import glob as _glob

    results = _glob.glob(pattern)
    if pattern.endswith(".nc"):
        results += _glob.glob(pattern + "4")
        results += _glob.glob(pattern[:-3] + ".NC")
        results += _glob.glob(pattern[:-3] + ".NC4")
    elif pattern.endswith(".NC"):
        results += _glob.glob(pattern + "4")
        results += _glob.glob(pattern[:-3] + ".nc")
        results += _glob.glob(pattern[:-3] + ".nc4")
    return sorted(set(results))


def nc_exists(path: str) -> str | None:
    """Check if a file exists with .nc or .nc4 extension.

    Args:
        path: File path ending in .nc

    Returns:
        The actual path that exists (.nc or .nc4), or None.
    """
    import os
    from pathlib import Path

    if os.path.exists(path):
        return path
    candidate = Path(path)
    if candidate.suffix.lower() in {".nc", ".nc4"} and candidate.parent.exists():
        stem_lower = candidate.stem.lower()
        for sibling in candidate.parent.iterdir():
            if sibling.is_file() and sibling.stem.lower() == stem_lower and sibling.suffix in NC_SUFFIXES:
                return str(sibling)
    for suffix in NC_SUFFIXES:
        if path.endswith(suffix):
            stem = path[: -len(suffix)]
            for alt_suffix in NC_SUFFIXES:
                alt = stem + alt_suffix
                if os.path.exists(alt):
                    return alt
            break
    return None
