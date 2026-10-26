"""Opt-in EGM2008 -> EGM96 correction for station water-surface elevation.

H96 = H08 + N08 - N96. Geoid grids and source/processed station files are read-only.
"""

import os
from pathlib import Path
import re

import numpy as np
import xarray as xr


class VerticalDatumError(ValueError):
    """Invalid water-surface elevation correction (not a station data gap)."""


def normalize_vertical_datum_spec(value, *, context="vertical_datum"):
    if value is None:
        return None
    fixed = {"quantity": "orthometric_height", "source": "EGM2008", "target": "EGM96"}
    allowed = set(fixed) | {"source_geoid", "target_geoid"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise VerticalDatumError(f"{context}: expected a mapping with keys {sorted(allowed)}")
    result = {**fixed, **value}
    for key, expected in fixed.items():
        actual = result[key]
        if key in {"source", "target"} and isinstance(actual, str):
            actual = actual.strip().upper()
        if actual != expected:
            raise VerticalDatumError(
                f"{context}: only EGM2008 -> EGM96 water-surface elevation is supported; {key} must be {expected}"
            )
        result[key] = actual
    for key in ("source_geoid", "target_geoid"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise VerticalDatumError(f"{context}: {key} must be a nonempty path string")
    return result


def _coordinates(lat, lon):
    try:
        if isinstance(lat, (bool, np.bool_)) or isinstance(lon, (bool, np.bool_)):
            raise ValueError
        lat, lon = float(lat), float(lon)
        if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 360:
            raise ValueError
    except (TypeError, ValueError):
        raise VerticalDatumError("Reference WGS84 ref_lat/ref_lon must be finite within [-90,90]/[-180,360]") from None
    return lat, lon % 360


def _read_geoid_header(handle, path, size):
    metadata, tokens = {}, []
    while len(tokens) < 4:
        line = handle.readline(4097)
        if not line or len(line) > 4096 or handle.tell() > 65536 or not line.endswith(b"\n"):
            raise VerticalDatumError(f"{path}: malformed or oversized PGM header")
        try:
            line = line.decode("ascii").strip()
        except UnicodeDecodeError:
            raise VerticalDatumError(f"{path}: non-ASCII PGM header") from None
        if line.startswith("#"):
            parts = line[1:].strip().split(None, 1)
            if len(parts) == 2:
                if parts[0] in metadata and parts[0] in (
                    "Offset",
                    "Scale",
                    "Description",
                    "Origin",
                    "Vertical_Datum",
                ):
                    raise VerticalDatumError(f"{path}: duplicate PGM metadata")
                metadata[parts[0]] = parts[1]
        else:
            tokens.extend(line.split())
    offset_bytes = handle.tell()
    try:
        magic, width, height, maximum = tokens
        width, height, maximum = int(width), int(height), int(maximum)
        offset, scale = float(metadata["Offset"]), float(metadata["Scale"])
        models = set(re.findall(r"\bEGM(?:2008|96)\b", metadata["Description"].upper()))
        if (
            magic != "P5"
            or maximum != 65535
            or height < 2
            or width != 2 * (height - 1)
            or not np.isfinite([offset, scale]).all()
            or scale <= 0
            or len(models) != 1
            or metadata.get("Origin", "90N 0E") != "90N 0E"
            or metadata.get("Vertical_Datum", "WGS84") != "WGS84"
            or size != offset_bytes + width * height * 2
        ):
            raise ValueError
    except (ValueError, KeyError):
        raise VerticalDatumError(
            f"{path}: invalid GeographicLib PGM dimensions, model, Origin, Vertical_Datum, Offset, Scale or byte count"
        ) from None
    return width, height, offset_bytes, offset, scale, models.pop()


def _sample_geoid(path, model, lat, lon):
    lat, lon = _coordinates(lat, lon)
    try:
        path = Path(path).expanduser().resolve()
        # Read only the small header and four pixels; persistent mappings lock updates on Windows.
        with path.open("rb") as handle:
            width, height, offset_bytes, offset, scale, actual_model = _read_geoid_header(
                handle, path, os.fstat(handle.fileno()).st_size
            )
            if actual_model != model:
                raise VerticalDatumError(f"{path}: expected {model}, found {actual_model}; check swapped geoid paths")
            x, y = lon * width / 360, (90 - lat) * (height - 1) / 180
            i, j = int(x) % width, min(int(y), height - 2)
            dx, dy = x - int(x), y - j
            east = (i + 1) % width
            values = []
            for row, col in ((j, i), (j, east), (j + 1, i), (j + 1, east)):
                handle.seek(offset_bytes + 2 * (row * width + col))
                raw = handle.read(2)
                if len(raw) != 2:
                    raise VerticalDatumError(f"{path}: truncated geoid pixel")
                values.append(int.from_bytes(raw, "big"))
    except (OSError, ValueError) as exc:
        raise VerticalDatumError(f"Cannot read geoid {path}: {exc}") from exc
    north = (1 - dx) * values[0] + dx * values[1]
    south = (1 - dx) * values[2] + dx * values[3]
    return offset + scale * ((1 - dy) * north + dy * south)


def _array(data):
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, xr.Dataset) and len(data.data_vars) == 1:
        return data[next(iter(data.data_vars))]
    raise VerticalDatumError("Datum correction requires one selected water-surface elevation variable")


def validate_observation_height(data):
    """Check explicit input metadata without guessing physics from descriptive text."""
    array = _array(data)
    if array.attrs.get("units") != "m":
        raise VerticalDatumError("Water-surface elevation observations must be in metres (units=m)")
    if "openbench_vertical_datum_offset_m" in array.attrs:
        raise VerticalDatumError("Station heights already corrected; refusing double correction")
    for key in ("vertical_datum", "openbench_vertical_datum"):
        if key in array.attrs and str(array.attrs[key]).strip().upper() != "EGM2008":
            raise VerticalDatumError("Observation datum metadata conflicts with EGM2008")
    return array


def harmonize_station_vertical_datum(sim_ds, ref_ds, *, sim_config=None, ref_config, station, item):
    """Correct only WSE observations; the model/evaluation datum is already EGM96."""
    if sim_config is not None:
        raise VerticalDatumError("vertical_datum belongs only to Water_Surface_Elevation observations, not simulation")
    if ref_config is None:
        return sim_ds, ref_ds
    if item != "Water_Surface_Elevation":
        raise VerticalDatumError("vertical_datum supports only Water_Surface_Elevation")
    spec = normalize_vertical_datum_spec(ref_config)
    ref = validate_observation_height(ref_ds)
    if _array(sim_ds).attrs.get("units") != "m":
        raise VerticalDatumError("Processed model water-surface elevation must be in metres (units=m)")
    try:
        lat, lon = _coordinates(station["ref_lat"], station["ref_lon"])
    except (KeyError, TypeError):
        raise VerticalDatumError("Reference WGS84 ref_lat/ref_lon are required") from None
    offset = _sample_geoid(spec["source_geoid"], "EGM2008", lat, lon) - _sample_geoid(
        spec["target_geoid"], "EGM96", lat, lon
    )
    corrected = ref.astype(np.float64) + offset
    corrected.attrs = dict(ref.attrs)
    if "vertical_datum" in corrected.attrs:
        corrected.attrs["vertical_datum"] = "EGM96"
    corrected.attrs.update(
        openbench_vertical_datum="EGM96",
        openbench_vertical_datum_source="EGM2008",
        openbench_vertical_datum_target="EGM96",
        openbench_vertical_datum_offset_m=offset,
        openbench_vertical_datum_interpolation="bilinear",
        openbench_vertical_datum_source_geoid=spec["source_geoid"],
        openbench_vertical_datum_target_geoid=spec["target_geoid"],
    )
    if isinstance(ref_ds, xr.DataArray):
        return sim_ds, corrected
    result = ref_ds.copy(deep=False)
    result[ref.name] = corrected
    return sim_ds, result
