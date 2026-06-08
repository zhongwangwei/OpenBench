"""Simulation output scanner for generating OpenBench config fragments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from hashlib import blake2s
from pathlib import Path
import re
import unicodedata
from typing import Any

from openbench.data.coordinates import glob_nc
from openbench.data.registry.scanner import (
    filename_split_match,
    inspect_nc_file,
    is_year_range_endpoint,
    most_specific_date_matches,
)

DEFAULT_CASE_DEPTH = 5

_DEFAULT_EXCLUDES = {
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

_DATA_DIR_HINTS = {
    "data",
    "hist",
    "history",
    "lnd",
    "nc",
    "output",
    "outputs",
}


@dataclass
class SimulationCase:
    """A discovered simulation case."""

    label: str
    root_dir: Path
    model: str
    depth: int
    source_root: Path | None = None
    data_type: str | None = None
    tim_res: str | None = None
    grid_res: float | None = None
    data_groupby: str | None = None
    temporal_kind: str | None = None
    temporal_kind_candidate: str | None = None
    fulllist: Path | None = None
    station_layout: str | None = None
    station_count: int | None = None
    station_dropped_sites: list[str] = field(default_factory=list)
    merged_dir: Path | None = None
    years: list[int] | None = None
    time_start: str | None = None
    time_end: str | None = None
    time_count: int | None = None
    time_span_days: int | None = None
    prefix: str = ""
    suffix: str = ""
    variables: list[str] = field(default_factory=list)
    variable_metadata: list[dict[str, Any]] = field(default_factory=list)
    variable_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class SimulationScanResult:
    """Result of scanning one or more simulation roots."""

    roots: list[Path]
    cases: list[SimulationCase]
    unresolved: list[SimulationCase] = field(default_factory=list)


def scan_simulation_roots(
    roots: list[str | Path],
    *,
    model_name: str = "auto",
    case_depth: int = DEFAULT_CASE_DEPTH,
    case_pattern: str | None = None,
    exclude: tuple[str, ...] | list[str] = (),
    climatology: str = "auto",
) -> SimulationScanResult:
    """Discover simulation cases below roots.

    A case is a directory containing NetCDF files directly. The traversal is
    bounded by ``case_depth`` from each input root. Directories named like
    ``history`` or ``output`` use their parent directory as the case label.
    """
    root_paths = [Path(root).expanduser() for root in roots]
    exclude_patterns = tuple(_DEFAULT_EXCLUDES | {str(item) for item in exclude})
    cases: list[SimulationCase] = []

    for root in root_paths:
        for case_dir, metadata_dir, depth, station_layout in _iter_case_dirs(root, case_depth, exclude_patterns):
            label = _derive_case_label(root, case_dir)
            if case_pattern and not fnmatch(label, case_pattern):
                continue
            cases.append(
                _build_case(
                    root,
                    case_dir,
                    metadata_dir,
                    label,
                    depth,
                    model_name,
                    climatology,
                    station_layout=station_layout,
                )
            )

    cases.sort(key=lambda case: (case.label, str(case.root_dir)))
    _deduplicate_case_labels(cases)
    unresolved = [case for case in cases if case.unresolved]
    return SimulationScanResult(roots=root_paths, cases=cases, unresolved=unresolved)


def _build_case(
    root: Path,
    case_dir: Path,
    metadata_dir: Path,
    label: str,
    depth: int,
    model_name: str,
    climatology: str,
    *,
    station_layout: str | None = None,
) -> SimulationCase:
    info = inspect_nc_file(metadata_dir)
    multi_undated = _has_multiple_no_date_files(metadata_dir) and not station_layout
    if multi_undated:
        common_prefix, common_suffix = _common_stem_affixes(glob_nc(metadata_dir))
        info["prefix"] = common_prefix
        info["suffix"] = common_suffix
    if station_layout:
        data_groupby, filename_years = "Single", _years_from_station_collection(case_dir)
    else:
        data_groupby, filename_years = _infer_file_grouping(metadata_dir)
    if station_layout:
        time_coverage = _infer_station_time_coverage(case_dir, metadata_dir, station_layout)
    else:
        selected_files = _files_for_case_time_coverage(
            metadata_dir,
            info=info,
        )
        time_coverage = _infer_time_coverage(
            metadata_dir,
            files=selected_files or None,
            data_groupby=data_groupby,
        )
    temporal_kind, temporal_kind_candidate, temporal_source = _infer_temporal_kind(
        metadata_dir,
        data_groupby=data_groupby,
        time_coverage=time_coverage,
        climatology=climatology,
    )
    years = filename_years or time_coverage.get("years")
    variable_metadata = _per_bucket_variable_metadata(metadata_dir, info=info, station_layout=station_layout)
    variables = [item["name"] for item in variable_metadata]
    model, model_source, model_unresolved = _resolve_model(model_name, root, case_dir, label, variables)
    tim_res, tim_res_source = _infer_tim_res(
        metadata_dir,
        info=info,
        temporal_kind=temporal_kind,
        time_coverage=time_coverage,
        data_groupby=data_groupby,
    )
    case_data_type = "stn" if station_layout else info.get("detected_data_type")
    variable_overrides = (
        {}
        if station_layout
        else _infer_variable_file_overrides(
            metadata_dir,
            model=model,
            default_grid_res=info.get("detected_grid_res"),
            default_tim_res=tim_res,
            default_data_type=case_data_type,
            default_data_groupby=data_groupby,
        )
    )

    unresolved = []
    if model_unresolved:
        unresolved.append(model_unresolved)
    if multi_undated:
        unresolved.append("multi_undated_files")
    if any(
        override.get("tim_res") or override.get("data_type") or override.get("data_groupby")
        for override in variable_overrides.values()
    ):
        unresolved.append("variable_stream_inconsistent")

    return SimulationCase(
        label=label,
        root_dir=case_dir,
        source_root=root,
        model=model,
        depth=depth,
        data_type=case_data_type,
        tim_res=tim_res,
        grid_res=info.get("detected_grid_res"),
        data_groupby=data_groupby,
        temporal_kind=temporal_kind,
        temporal_kind_candidate=temporal_kind_candidate,
        station_layout=station_layout,
        years=years,
        time_start=time_coverage.get("time_start"),
        time_end=time_coverage.get("time_end"),
        time_count=time_coverage.get("time_count"),
        time_span_days=time_coverage.get("time_span_days"),
        prefix=info.get("prefix", ""),
        suffix=info.get("suffix", ""),
        variables=variables,
        variable_metadata=variable_metadata,
        variable_overrides=variable_overrides,
        unresolved=unresolved,
        provenance={
            "root_dir": "scan",
            "model": model_source,
            "data_type": "nc" if info.get("detected_data_type") else "unknown",
            "tim_res": tim_res_source,
            "grid_res": "nc" if info.get("detected_grid_res") else "unknown",
            "data_groupby": "filename",
            "temporal_kind": temporal_source,
            "years": "filename" if filename_years else ("nc" if time_coverage.get("years") else "unknown"),
            "time_coverage": "nc" if time_coverage.get("time_start") else "unknown",
            "variable_overrides": "filename" if variable_overrides else "none",
        },
    )


def _infer_time_coverage(
    nc_dir: Path,
    *,
    files: list[Path] | None = None,
    data_groupby: str | None = None,
) -> dict:
    files = files if files is not None else glob_nc(nc_dir)
    if not files:
        return {}

    filename_values = _filename_time_values(files)
    filename_value_count = len(filename_values)
    sample_files = files if not filename_values else _select_time_sample_files(files)
    values = []
    time_sizes = []
    time_units = []
    raw_time_steps = []
    raw_time_units = []
    readable_count = 0
    no_time_count = 0
    for file_path in sample_files:
        info = _time_info_from_file(file_path)
        if not info.get("readable"):
            continue
        readable_count += 1
        if not info.get("has_time"):
            no_time_count += 1
            continue
        if info.get("time_size") is not None:
            time_sizes.append(int(info["time_size"]))
        if info.get("time_units"):
            time_units.append(str(info["time_units"]))
        if info.get("raw_time_step") is not None:
            raw_time_steps.append(float(info["raw_time_step"]))
        if info.get("raw_time_units"):
            raw_time_units.append(str(info["raw_time_units"]))
        values.extend(info.get("values", []))
    read_all_files = len(sample_files) == len(files)
    if not values and filename_values:
        filename_values.sort()
        start = filename_values[0]
        end = filename_values[-1]
        median_size = _median_positive_int(time_sizes) or 1
        end = _expand_filename_time_end(
            end,
            median_size=median_size,
            units=time_units + raw_time_units,
            data_groupby=data_groupby,
        )
        time_count = filename_value_count if median_size == 1 else filename_value_count * median_size
        return {
            "time_start": _format_time(start),
            "time_end": _format_time(end),
            "time_count": time_count,
            "time_span_days": int((end - start).days),
            "time_span_seconds": int((end - start).total_seconds()),
            "years": [start.year, end.year],
            "has_time": True,
            "months": sorted({value.month for value in filename_values}),
            "file_time_counts": time_sizes,
            "time_units": time_units,
            "raw_time_steps": raw_time_steps,
            "raw_time_units": raw_time_units,
        }

    if not values:
        if readable_count and no_time_count == readable_count:
            return {"has_time": False, "time_count": 0}
        if time_sizes:
            if read_all_files:
                time_count = sum(time_sizes)
            else:
                time_count = (_median_positive_int(time_sizes) or 1) * len(files)
            return {
                "has_time": True,
                "time_count": time_count,
                "file_time_counts": time_sizes,
                "time_units": time_units,
                "raw_time_steps": raw_time_steps,
                "raw_time_units": raw_time_units,
            }
        return {}

    values.sort()
    if not read_all_files and filename_values:
        filename_values.sort()
        start = filename_values[0]
        end = filename_values[-1]
    else:
        start = values[0]
        end = values[-1]
    if read_all_files:
        time_count = len(values)
    elif filename_value_count:
        median_size = _median_positive_int(time_sizes) or 1
        end = _expand_filename_time_end(
            end,
            median_size=median_size,
            units=time_units + raw_time_units,
            data_groupby=data_groupby,
        )
        time_count = filename_value_count if median_size == 1 else filename_value_count * median_size
    else:
        time_count = len(values)
    return {
        "time_start": _format_time(start),
        "time_end": _format_time(end),
        "time_count": time_count,
        "time_span_days": int((end - start).days),
        "time_span_seconds": int((end - start).total_seconds()),
        "years": [start.year, end.year],
        "has_time": True,
        "months": sorted({value.month for value in (filename_values or values)}),
        "file_time_counts": time_sizes,
        "time_units": time_units,
        "raw_time_steps": raw_time_steps,
        "raw_time_units": raw_time_units,
    }


def _infer_station_time_coverage(case_dir: Path, metadata_dir: Path, station_layout: str) -> dict:
    if station_layout == "flat":
        coverages = [
            _infer_time_coverage(metadata_dir, files=[file_path], data_groupby="Single")
            for file_path in glob_nc(metadata_dir)
        ]
    else:
        coverages = [
            _infer_time_coverage(nc_dir, files=glob_nc(nc_dir), data_groupby="Single")
            for nc_dir in _station_child_nc_dirs(case_dir)
        ]
    return _combine_station_time_coverages(coverages)


def _select_time_sample_files(files: list[Path], *, limit: int = 12) -> list[Path]:
    if len(files) <= limit:
        return files
    ordered = sorted(files, key=_filename_time_sort_key)
    indexes = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)}
    return [ordered[index] for index in sorted(indexes)]


def _filename_time_sort_key(file_path: Path) -> tuple[datetime, str]:
    values = _filename_time_values([file_path])
    value = values[0] if values else datetime.max
    return value, file_path.name


def _filename_time_values(files: list[Path]) -> list[datetime]:
    values = []
    for file_path in files:
        for match in most_specific_date_matches(file_path.stem):
            year = int(match.group("year"))
            month = int(match.group("month") or 1)
            day = int(match.group("day") or 1)
            values.append(datetime(year, month, day))
    return values


def _expand_filename_time_end(
    end: datetime,
    *,
    median_size: int,
    units: list[str],
    data_groupby: str | None,
) -> datetime:
    if data_groupby != "Year" or not 10 <= median_size <= 13:
        return end
    if not any("month" in str(unit).lower() for unit in units):
        return end
    return _add_months(end, median_size - 1)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month)


def _files_for_filename_pattern(nc_dir: Path, prefix: str, suffix: str) -> list[Path]:
    files = []
    for file_path in glob_nc(nc_dir):
        stem = file_path.stem
        date_match = filename_split_match(stem)
        if date_match:
            file_prefix = stem[: date_match.start("token")]
            file_suffix = stem[date_match.end("token") :]
        else:
            file_prefix = stem
            file_suffix = ""
        if file_prefix == prefix and file_suffix == suffix:
            files.append(file_path)
    return files


def _has_multiple_no_date_files(nc_dir: Path) -> bool:
    files = glob_nc(nc_dir)
    if len(files) <= 1:
        return False
    return not any(most_specific_date_matches(file_path.stem) for file_path in files)


def _common_stem_affixes(files: list[Path]) -> tuple[str, str]:
    """Return the longest common prefix/suffix of NC stems."""
    stems = [file_path.stem for file_path in files]
    if not stems:
        return "", ""
    if len(stems) == 1:
        return stems[0], ""
    import os.path as _os_path

    prefix = _os_path.commonprefix(stems)
    suffix = _os_path.commonprefix([stem[::-1] for stem in stems])[::-1]
    if len(prefix) + len(suffix) >= max(len(stem) for stem in stems):
        suffix = ""
    return prefix, suffix


def _files_for_case_time_coverage(nc_dir: Path, *, info: dict) -> list[Path]:
    """Return files belonging to the inspected stream for time coverage.

    Date-bearing files can be narrowed by prefix/suffix. If multiple files have
    no date token, prefix/suffix cannot describe a time stream, so use all files
    to avoid treating the first arbitrary filename as the whole case.
    """
    files = glob_nc(nc_dir)
    if len(files) <= 1:
        return files

    selected = _files_for_filename_pattern(
        nc_dir,
        info.get("prefix", ""),
        info.get("suffix", ""),
    )
    if len(selected) == 1 and not most_specific_date_matches(selected[0].stem):
        return files
    return selected


def _infer_variable_file_overrides(
    nc_dir: Path,
    *,
    model: str,
    default_grid_res: float | None,
    default_tim_res: str | None = None,
    default_data_type: str | None = None,
    default_data_groupby: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not model or model == "UNRESOLVED":
        return {}
    files = glob_nc(nc_dir)
    if len(files) < 2:
        return {}
    patterns = {_filename_pattern_for_file(file_path) for file_path in files}
    if len(patterns) < 2:
        return {}

    try:
        from openbench.data.registry import RegistryManager

        profile = RegistryManager().get_model(model)
    except Exception:
        profile = None
    if not profile:
        return {}

    overrides: dict[str, dict[str, Any]] = {}
    for variable_name, mapping in profile.variables.items():
        matched = _match_profile_variable_file(files, mapping)
        if matched is None:
            continue
        file_path, candidate = matched
        prefix, suffix = _filename_pattern_for_file(file_path)
        override: dict[str, Any] = {"prefix": prefix}
        if suffix:
            override["suffix"] = suffix

        actual_var = _matching_data_var(file_path, candidate)
        if actual_var and not getattr(mapping, "compute", None):
            primary = _primary_mapping_varname(mapping)
            if not primary or actual_var != primary:
                override["varname"] = actual_var

        grid_res = _grid_res_from_file(file_path)
        if grid_res is not None and (default_grid_res is None or abs(float(grid_res) - float(default_grid_res)) > 1e-6):
            override["grid_res"] = grid_res

        try:
            from openbench.data.registry.scanner import _detect_data_type_from_nc

            file_data_type = _detect_data_type_from_nc(file_path)
        except Exception:
            file_data_type = None
        if file_data_type and default_data_type and file_data_type != default_data_type:
            override["data_type"] = file_data_type

        bucket_files = _files_for_filename_pattern(nc_dir, prefix, suffix) or [file_path]
        bucket_groupby, _ = _infer_file_grouping_for_files(bucket_files)
        if bucket_groupby and default_data_groupby and bucket_groupby != default_data_groupby:
            override["data_groupby"] = bucket_groupby
        sample = _select_time_sample_files(bucket_files)
        time_info = next(
            (
                info
                for info in (_time_info_from_file(path) for path in sample)
                if info.get("readable") and info.get("has_time")
            ),
            {},
        )
        bucket_tim_res = _tim_res_from_time_info(
            time_info,
            data_groupby=bucket_groupby or default_data_groupby,
        )
        if bucket_tim_res and default_tim_res and bucket_tim_res != default_tim_res:
            override["tim_res"] = bucket_tim_res

        overrides[variable_name] = _ordered_variable_override(override)
    return overrides


def _infer_file_grouping_for_files(files: list[Path]) -> tuple[str, list[int] | None]:
    token_lengths = []
    for file_path in files:
        for match in most_specific_date_matches(file_path.stem):
            if is_year_range_endpoint(file_path.stem, match):
                continue
            token = match.group("token")
            token_lengths.append(len("".join(ch for ch in token if ch.isdigit())))
    years = _years_from_files(files)
    if not token_lengths:
        return "Single", years
    if max(token_lengths) >= 8:
        return "Day", years
    if max(token_lengths) >= 6:
        return "Month", years
    return "Year", years


def _tim_res_from_time_info(info: dict, *, data_groupby: str | None) -> str | None:
    """Best-effort tim_res inference from a single file's time metadata."""
    if not info or not info.get("has_time"):
        return None
    units = " ".join(
        [
            info.get("time_units") or "",
            info.get("raw_time_units") or "",
        ]
    ).lower()
    raw_step = info.get("raw_time_step")
    if "hour" in units and raw_step:
        int(round(float(raw_step)))
        if abs(float(raw_step) - 1) < 0.2:
            return "Hour"
        if abs(float(raw_step) - 3) < 0.5:
            return "3Hour"
        if abs(float(raw_step) - 6) < 1.0:
            return "6Hour"
        if abs(float(raw_step) - 24) < 2.0:
            return "Day"
    if "day" in units and raw_step:
        if abs(float(raw_step) - 1) < 0.2:
            return "Day"
        if abs(float(raw_step) - 8) < 1.0:
            return "8Day"
        if 28 <= float(raw_step) <= 32:
            return "Month"
    if "month" in units:
        return "Month"
    coverage = {
        "file_time_counts": [info["time_size"]] if info.get("time_size") is not None else [],
        "time_units": [info.get("time_units")] if info.get("time_units") else [],
        "raw_time_units": [info.get("raw_time_units")] if info.get("raw_time_units") else [],
        "raw_time_steps": [info["raw_time_step"]] if info.get("raw_time_step") is not None else [],
    }
    return _infer_tim_res_from_time_shape(coverage, data_groupby=data_groupby or "Year")


def _filename_pattern_for_file(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem
    date_match = filename_split_match(stem)
    if not date_match:
        return stem, ""
    return stem[: date_match.start("token")], stem[date_match.end("token") :]


def _match_profile_variable_file(files: list[Path], mapping) -> tuple[Path, str] | None:
    candidates = _mapping_candidate_varnames(mapping)
    if not candidates:
        return None
    for candidate in candidates:
        for file_path in files:
            if _stem_contains_var_token(file_path.stem, candidate):
                return file_path, candidate
    return None


def _mapping_candidate_varnames(mapping) -> list[str]:
    candidates: list[str] = []
    varname = getattr(mapping, "varname", None)
    if isinstance(varname, str):
        if varname:
            candidates.append(varname)
    elif varname:
        candidates.extend(str(item) for item in varname if item)

    for fallback in getattr(mapping, "fallbacks", None) or []:
        fallback_name = getattr(fallback, "varname", None)
        if fallback_name:
            candidates.append(str(fallback_name))

    compute = getattr(mapping, "compute", None)
    if compute:
        candidates.extend(re.findall(r"ds\[['\"]([^'\"]+)['\"]\]", str(compute)))

    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _primary_mapping_varname(mapping) -> str | None:
    varname = getattr(mapping, "varname", None)
    if isinstance(varname, str):
        return varname or None
    if varname:
        for item in varname:
            if item:
                return str(item)
    return None


def _stem_contains_var_token(stem: str, candidate: str) -> bool:
    pattern = rf"(^|[^A-Za-z0-9]){re.escape(candidate)}(?=[^A-Za-z0-9]|$)"
    return re.search(pattern, stem, flags=re.IGNORECASE) is not None


def _matching_data_var(file_path: Path, candidate: str) -> str | None:
    data_vars = _data_vars_from_file(file_path)
    for name in data_vars:
        if name == candidate:
            return name
    candidate_lower = candidate.lower()
    for name in data_vars:
        if name.lower() == candidate_lower:
            return name
    return data_vars[0] if len(data_vars) == 1 else None


def _data_vars_from_file(file_path: Path) -> list[str]:
    return [item["name"] for item in _data_var_metadata_from_file(file_path)]


def _data_var_metadata_from_file(file_path: Path) -> list[dict[str, Any]]:
    try:
        import netCDF4

        with netCDF4.Dataset(str(file_path), "r") as nc:
            skip_vars = {
                "time_bnds",
                "time_bounds",
                "lat_bnds",
                "lon_bnds",
                "lat_bounds",
                "lon_bounds",
                "crs",
                "spatial_ref",
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
                "time",
                "t",
            }
            coord_names = set(nc.dimensions.keys())
            metadata: list[dict[str, Any]] = []
            for name in nc.variables:
                if name in skip_vars or name in coord_names or name.lower() in known_coord_var_names:
                    continue
                var = nc.variables[name]
                if len(var.dimensions) < 1:
                    continue
                unit = getattr(var, "units", getattr(var, "unit", ""))
                unit = str(unit).replace(".", " ").strip() if unit else ""
                metadata.append(
                    {
                        "name": name,
                        "unit": unit,
                        "dims": list(var.dimensions),
                        "long_name": getattr(var, "long_name", ""),
                        "standard_name": getattr(var, "standard_name", ""),
                    }
                )
            return metadata
    except Exception:
        return []


def _per_bucket_variable_metadata(
    metadata_dir: Path,
    *,
    info: dict,
    station_layout: str | None,
    max_buckets: int = 8,
) -> list[dict[str, Any]]:
    """Collect variable metadata across distinct prefix/suffix buckets.

    A directory with one file per variable (e.g. CTSM ``*.h0.QFLX.nc``,
    ``*.h0.QSOIL.nc``) cannot be summarised by inspecting the alphabetically
    first file. Group files by their non-date stem, sample one file per group,
    and union the resulting variable lists so model inference and
    register-model see the full set.
    """
    base_metadata = list(info.get("all_data_vars", []))
    if station_layout:
        return base_metadata

    files = glob_nc(metadata_dir)
    if len(files) <= 1:
        return base_metadata

    buckets: dict[tuple[str, str], Path] = {}
    for file_path in files:
        key = _filename_pattern_for_file(file_path)
        buckets.setdefault(key, file_path)
        if len(buckets) >= max_buckets:
            break
    if len(buckets) <= 1:
        return base_metadata

    seen: dict[str, dict[str, Any]] = {item["name"]: dict(item) for item in base_metadata}
    for representative in buckets.values():
        for entry in _data_var_metadata_from_file(representative):
            seen.setdefault(entry["name"], entry)
    return list(seen.values())


def _grid_res_from_file(file_path: Path) -> float | None:
    try:
        import netCDF4
        import numpy as np
        import numpy.ma as ma

        from openbench.data.coordinates import LAT_NAMES

        with netCDF4.Dataset(str(file_path), "r") as nc:
            for lat_name in LAT_NAMES:
                if lat_name in nc.variables and lat_name in nc.dimensions and nc.dimensions[lat_name].size > 1:
                    lat_vals = np.asarray(ma.filled(nc.variables[lat_name][:], np.nan), dtype=float)
                    diffs = np.diff(lat_vals)
                    diffs = diffs[np.isfinite(diffs)]
                    if not diffs.size:
                        return None
                    grid_res = round(abs(float(np.median(diffs))), 4)
                    if 0.001 < grid_res < 10:
                        return grid_res
    except Exception:
        return None
    return None


def _ordered_variable_override(override: dict[str, Any]) -> dict[str, Any]:
    ordered = {}
    for key in ("varname", "prefix", "suffix", "grid_res", "tim_res", "data_groupby"):
        if key in override:
            ordered[key] = override[key]
    for key, value in override.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _median_positive_int(values: list[int]) -> int | None:
    positive = sorted(int(value) for value in values if int(value) > 0)
    if not positive:
        return None
    return positive[len(positive) // 2]


def _median_positive_float(values: list[float]) -> float | None:
    positive = sorted(float(value) for value in values if float(value) > 0)
    if not positive:
        return None
    return positive[len(positive) // 2]


def _time_info_from_file(file_path: Path) -> dict:
    raw = _raw_time_metadata_from_file(file_path)
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with xr.open_dataset(file_path, decode_times=True, decode_timedelta=False) as ds:
            if "time" not in ds.coords and "time" not in ds.variables:
                return {"readable": True, "has_time": False, "values": []}
            time_var = ds["time"]
            raw_values = ds["time"].values
            units = raw.get("raw_time_units") or str(time_var.attrs.get("units", ""))
            if np.issubdtype(np.asarray(raw_values).dtype, np.number) and not units.strip():
                return _raw_time_info_from_file(file_path)
            timestamps = pd.to_datetime(raw_values)
            if getattr(timestamps, "ndim", 1) == 0:
                timestamps = [timestamps]
            values = [ts.to_pydatetime().replace(tzinfo=None) for ts in timestamps if not pd.isna(ts)]
            return {
                "readable": True,
                "has_time": True,
                "values": values,
                "time_size": int(time_var.size),
                "time_units": units,
                "raw_time_step": raw.get("raw_time_step"),
                "raw_time_units": raw.get("raw_time_units"),
            }
    except Exception:
        return _raw_time_info_from_file(file_path)


def _raw_time_metadata_from_file(file_path: Path) -> dict:
    try:
        import numpy as np
        import xarray as xr

        with xr.open_dataset(file_path, decode_times=False, decode_timedelta=False) as ds:
            if "time" not in ds.coords and "time" not in ds.variables:
                return {}
            time_var = ds["time"]
            result = {"raw_time_units": str(time_var.attrs.get("units", ""))}
            values = np.asarray(time_var.values).reshape(-1)
            if values.size >= 2 and np.issubdtype(values.dtype, np.number):
                diffs = np.diff(values.astype(float))
                positive = [float(diff) for diff in diffs if np.isfinite(diff) and diff > 0]
                if positive:
                    result["raw_time_step"] = _median_positive_float(positive)
            return result
    except Exception:
        return {}


def _raw_time_info_from_file(file_path: Path) -> dict:
    try:
        import numpy as np
        import xarray as xr

        with xr.open_dataset(file_path, decode_times=False, decode_timedelta=False) as ds:
            if "time" not in ds.coords and "time" not in ds.variables:
                return {"readable": True, "has_time": False, "values": []}
            time_var = ds["time"]
            values = np.asarray(time_var.values).reshape(-1)
            raw_time_step = None
            if values.size >= 2 and np.issubdtype(values.dtype, np.number):
                diffs = np.diff(values.astype(float))
                raw_time_step = _median_positive_float(
                    [float(diff) for diff in diffs if np.isfinite(diff) and diff > 0]
                )
            raw_time_units = str(time_var.attrs.get("units", ""))
            return {
                "readable": True,
                "has_time": True,
                "values": [],
                "time_size": int(time_var.size),
                "time_units": raw_time_units,
                "raw_time_step": raw_time_step,
                "raw_time_units": raw_time_units,
            }
    except Exception:
        return {"readable": False, "has_time": False, "values": []}


def _format_time(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


def _infer_tim_res(
    nc_dir: Path,
    *,
    info: dict,
    temporal_kind: str | None,
    time_coverage: dict,
    data_groupby: str,
) -> tuple[str | None, str]:
    if temporal_kind:
        return temporal_kind, "climatology"
    if info.get("detected_tim_res"):
        return info["detected_tim_res"], "nc"

    if data_groupby != "Single":
        from_shape = _infer_tim_res_from_time_shape(time_coverage, data_groupby=data_groupby)
        if from_shape:
            return from_shape, "time"

    from_coverage = _infer_tim_res_from_time_coverage(time_coverage)
    if from_coverage:
        return from_coverage, "time"

    if data_groupby == "Single":
        from_shape = _infer_tim_res_from_time_shape(time_coverage, data_groupby=data_groupby)
        if from_shape:
            return from_shape, "time"

    from_filename = _infer_tim_res_from_filename(nc_dir)
    if from_filename:
        return from_filename, "filename"

    return None, "unknown"


def _infer_tim_res_from_time_coverage(time_coverage: dict) -> str | None:
    time_count = time_coverage.get("time_count")
    span_seconds = time_coverage.get("time_span_seconds")
    if not time_count or time_count <= 1 or span_seconds is None:
        return None

    step_seconds = span_seconds / max(time_count - 1, 1)
    if abs(step_seconds - 3600) < 600:
        return "Hour"
    if abs(step_seconds - 3 * 3600) < 1800:
        return "3Hour"
    if abs(step_seconds - 6 * 3600) < 3600:
        return "6Hour"
    if abs(step_seconds - 24 * 3600) < 7200:
        return "Day"
    if abs(step_seconds - 8 * 24 * 3600) < 24 * 3600:
        return "8Day"
    if 28 * 24 * 3600 <= step_seconds <= 32 * 24 * 3600:
        return "Month"
    if abs(step_seconds - 365 * 24 * 3600) < 30 * 24 * 3600:
        return "Year"
    return None


def _infer_tim_res_from_time_shape(time_coverage: dict, *, data_groupby: str) -> str | None:
    counts = [int(count) for count in time_coverage.get("file_time_counts", []) if int(count) > 0]
    count = sorted(counts)[len(counts) // 2] if counts else None
    units = " ".join(
        [
            *time_coverage.get("time_units", []),
            *time_coverage.get("raw_time_units", []),
        ]
    ).lower()
    raw_step = _median_positive_float(
        [float(step) for step in time_coverage.get("raw_time_steps", []) if float(step) > 0]
    )
    if "month" in units:
        if data_groupby == "Year" and count is not None and 10 <= count <= 13:
            return "Month"
        if raw_step is not None:
            rounded = int(round(raw_step))
            if abs(raw_step - 1) < 0.2:
                return "Month"
            if rounded > 1 and abs(raw_step - rounded) < 0.2:
                return f"{rounded}month"
        return "Month"
    if not counts:
        return None

    if data_groupby == "Year":
        if 10 <= count <= 13:
            return "Month"
        if 350 <= count <= 370:
            return "Day"
        if 8000 <= count <= 9000:
            return "Hour"
    if data_groupby == "Month":
        if 27 <= count <= 32:
            return "Day"
        if 650 <= count <= 750:
            return "Hour"
        if count == 1:
            return "Month"
    if data_groupby == "Day":
        if 20 <= count <= 25:
            return "Hour"
        if count == 1:
            return "Day"
    return None


def _infer_tim_res_from_filename(nc_dir: Path) -> str | None:
    """Match unambiguous frequency tokens; single-letter ``m/d/h`` are excluded
    because filename labels (config codes, experiment IDs) trigger them too
    aggressively. ``hr`` is kept because it is widely used as ``3hr/6hr``."""
    stems = [file_path.stem.lower() for file_path in glob_nc(nc_dir)]
    for stem in stems:
        if re.search(r"(^|[^a-z0-9])(hourly|hour|hr)(?=[^a-z0-9]|$)", stem):
            return "Hour"
    for stem in stems:
        if re.search(r"(^|[^a-z0-9])(monthly|month|mon)(?=[^a-z0-9]|$)", stem):
            return "Month"
    for stem in stems:
        if re.search(r"(^|[^a-z0-9])(daily|day)(?=[^a-z0-9]|$)", stem):
            return "Day"
    return None


def _infer_temporal_kind(
    nc_dir: Path,
    *,
    data_groupby: str,
    time_coverage: dict,
    climatology: str,
) -> tuple[str | None, str | None, str]:
    mode = str(climatology or "auto").lower()
    if mode == "off":
        return None, None, "user"
    if mode == "year":
        return "climatology-year", None, "user"
    if mode == "month":
        return "climatology-month", None, "user"

    time_count = time_coverage.get("time_count")
    has_climatology_hint = _has_climatology_hint(nc_dir)

    if data_groupby == "Single":
        if time_count == 0 and time_coverage.get("has_time") is False:
            return "climatology-year", None, "auto"
        if time_count == 1:
            return "climatology-year", None, "auto"
        if time_count == 12 and set(time_coverage.get("months", [])) == set(range(1, 13)) and has_climatology_hint:
            return None, "climatology-month", "candidate"
        return None, None, "auto"

    # Non-Single groupings: only treat single-timestep / 12-month aggregates as
    # climatology candidates when the directory or filenames hint at it.
    if has_climatology_hint:
        if time_count in (0, 1) or time_coverage.get("has_time") is False:
            return "climatology-year", None, "auto"
        if time_count == 12 and set(time_coverage.get("months", [])) == set(range(1, 13)):
            return None, "climatology-month", "candidate"

    return None, None, "auto"


def _has_climatology_hint(nc_dir: Path) -> bool:
    parts = [part.lower() for part in nc_dir.parts]
    parts.extend(file_path.stem.lower() for file_path in glob_nc(nc_dir))
    hints = ("clim", "climatology", "climatological")
    return any(any(hint in part for hint in hints) for part in parts)


def _infer_file_grouping(nc_dir: Path) -> tuple[str, list[int] | None]:
    files = glob_nc(nc_dir)
    token_lengths = []
    for file_path in files:
        matches = most_specific_date_matches(file_path.stem)
        if not matches:
            continue
        for match in matches:
            token = match.group("token")
            token_lengths.append(len("".join(ch for ch in token if ch.isdigit())))

    years = _years_from_files(files)
    if not token_lengths:
        return "Single", years
    if max(token_lengths) >= 8:
        return "Day", years
    if max(token_lengths) >= 6:
        return "Month", years
    return "Year", years


def _deduplicate_case_labels(cases: list[SimulationCase]) -> None:
    """Make case labels unique so YAML generation cannot silently overwrite.

    On collision prefer ``<base>__<root_token>`` (root-name suffix) so multi-root
    scans can trace each entry back to its input root. Fall back to
    ``<base>_<n>`` when the root token is missing or already taken.
    """
    used: set[str] = set()
    counters: dict[str, int] = {}
    for case in cases:
        base = case.label or "case"
        label = base
        if label in used:
            root_token = _safe_path_label(case.source_root.name) if case.source_root else ""
            candidate = f"{base}__{root_token}" if root_token and root_token != base else ""
            if candidate and candidate not in used:
                label = candidate
                case.provenance["label"] = "deduplicated_by_root"
            else:
                next_index = counters.get(base, 1) + 1
                while f"{base}_{next_index}" in used:
                    next_index += 1
                counters[base] = next_index
                label = f"{base}_{next_index}"
                case.provenance["label"] = "deduplicated"
            case.provenance["original_label"] = base
        else:
            counters.setdefault(base, 1)
        case.label = label
        used.add(label)


def _years_from_files(files: list[Path]) -> list[int] | None:
    years = []
    for file_path in files:
        stem = file_path.stem
        for match in most_specific_date_matches(stem):
            if is_year_range_endpoint(stem, match):
                continue
            year = int(match.group("year"))
            if 1900 <= year <= 2100:
                years.append(year)
    if not years:
        return None
    return [min(years), max(years)]


def _years_from_station_collection(case_dir: Path) -> list[int] | None:
    years = []
    direct_years = _years_from_files(glob_nc(case_dir))
    if direct_years:
        years.extend(direct_years)
    for nc_dir in _station_child_nc_dirs(case_dir):
        file_years = _years_from_files(glob_nc(nc_dir))
        if file_years:
            years.extend(file_years)
    if not years:
        return None
    return [min(years), max(years)]


def _resolve_model(
    model_name: str,
    root: Path,
    nc_dir: Path,
    label: str,
    variables: list[str],
) -> tuple[str, str, str | None]:
    if model_name != "auto":
        return model_name, "user", None

    try:
        from openbench.data.registry import RegistryManager
        from openbench.data.registry.manager import canonical_model_key

        models = _deduplicate_equivalent_models(
            RegistryManager().list_models(),
            canonical_key=canonical_model_key,
        )
    except Exception:
        models = []

    path_tokens = set(_name_tokens(" ".join([label, *[part for part in nc_dir.relative_to(root).parts]])))
    path_matches = [model for model in models if _model_name_matches_path_tokens(model.name, path_tokens)]

    variable_set = {variable.lower() for variable in variables}
    scored = []
    score_by_model = {}
    for model in models:
        model_varnames = {name.lower() for name in _model_varnames(model)}
        score = len(variable_set & model_varnames)
        score_by_model[model.name] = score
        if score:
            scored.append((score, model.name))

    if path_matches:
        path_matches.sort(key=lambda model: len(model.name), reverse=True)
        best_path = path_matches[0]
        if scored:
            scored.sort(reverse=True)
            best_score, best_name = scored[0]
            runner_up_score = scored[1][0] if len(scored) > 1 else 0
            path_score = score_by_model.get(best_path.name, 0)
            if best_name != best_path.name and best_score > runner_up_score and best_score >= path_score + 2:
                return best_name, "variables", None
        return best_path.name, "path", None

    if scored:
        scored.sort(reverse=True)
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1], "variables", None

    return "UNRESOLVED", "unresolved", "model"


def _deduplicate_equivalent_models(models, *, canonical_key) -> list:
    by_key = {model.name.lower(): model for model in models}
    deduped = []
    seen = set()
    for model in models:
        key = canonical_key(model.name)
        if key in seen:
            continue
        deduped.append(by_key.get(key, model))
        seen.add(key)
    return deduped


def _model_name_matches_path_tokens(model_name: str, path_tokens: set[str]) -> bool:
    model_tokens = _name_tokens(model_name)
    return bool(model_tokens) and all(token in path_tokens for token in model_tokens)


def _name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", value.lower()) if token]


def _model_varnames(model) -> set[str]:
    names: set[str] = set()
    for mapping in model.variables.values():
        varname = mapping.varname
        if varname is None:
            pass
        elif isinstance(varname, str):
            if varname:
                names.add(varname)
        else:
            names.update(item for item in varname if item)
        if mapping.fallbacks:
            names.update(fallback.varname for fallback in mapping.fallbacks if fallback.varname)
    return names


def materialize_station_cases(
    result: SimulationScanResult,
    output_dir: str | Path,
    *,
    num_workers: int = 4,
    allow_partial: bool = False,
) -> None:
    """Generate station fulllist files and merged per-station NC files.

    When ``allow_partial`` is False (default) any dropped sites mark the case
    as unresolved so the CLI can surface a hard error instead of silently
    publishing a configuration that ran on fewer stations than were on disk.
    """
    output_root = Path(output_dir).expanduser().resolve()
    station_cases = [case for case in result.cases if case.station_layout]
    if not station_cases:
        return

    from openbench.data.station_scanner import scan_station_sim_dir

    for case in station_cases:
        case_label = _safe_path_label(case.label)
        case_output_dir = output_root / case_label
        case_output_dir.mkdir(parents=True, exist_ok=True)

        merged_dir = case_output_dir / "merged"
        df, dropped = scan_station_sim_dir(
            str(case.root_dir),
            output_dir=str(merged_dir),
            num_workers=num_workers,
            return_dropped=True,
        )
        time_coverage = _infer_station_dataframe_time_coverage(df)
        if "sim_dir" in df.columns:
            df["sim_dir"] = df["sim_dir"].apply(lambda value: _portable_sim_dir(value, case.source_root))

        fulllist = case_output_dir / f"{case_label}_stations.csv"
        df.to_csv(fulllist, index=False)

        case.fulllist = fulllist
        case.station_count = int(len(df))
        case.station_dropped_sites = list(dropped)
        case.merged_dir = merged_dir if merged_dir.exists() else None
        if time_coverage:
            case.years = time_coverage.get("years") or case.years
            case.time_start = time_coverage.get("time_start") or case.time_start
            case.time_end = time_coverage.get("time_end") or case.time_end
            case.time_count = time_coverage.get("time_count") or case.time_count
            case.time_span_days = time_coverage.get("time_span_days") or case.time_span_days
        case.provenance["fulllist"] = "station-scan"
        if time_coverage:
            case.provenance["time_coverage"] = "station-materialized"
        if dropped and not allow_partial:
            case.unresolved.append("station_partial")
            case.provenance["station_partial"] = ",".join(sorted(dropped))
    if not allow_partial:
        partial_cases = [case for case in station_cases if case.station_dropped_sites]
        if partial_cases:
            result.unresolved.extend(case for case in partial_cases if case not in result.unresolved)


def _safe_path_label(label: str) -> str:
    text = unicodedata.normalize("NFKC", str(label))
    chars = []
    last_was_sep = False
    for char in text:
        if char.isalnum() or char in "._-":
            chars.append(char)
            last_was_sep = False
            continue
        if not last_was_sep:
            chars.append("_")
            last_was_sep = True

    safe = "".join(chars).strip("._")
    if safe:
        return safe
    digest = blake2s(text.encode("utf-8"), digest_size=4).hexdigest()
    return f"case_{digest}"


def _portable_sim_dir(value, source_root: Path | None) -> str:
    """Replace the ``source_root`` prefix with ``${OPENBENCH_SIM_ROOT}`` when applicable."""
    text = str(value)
    if source_root is None:
        return text
    try:
        target = Path(text).expanduser().resolve()
        base = Path(source_root).expanduser().resolve()
        rel = target.relative_to(base)
    except (OSError, ValueError):
        return text
    return "${OPENBENCH_SIM_ROOT}/" + rel.as_posix()


def _infer_station_dataframe_time_coverage(df) -> dict:
    if df is None or "sim_dir" not in getattr(df, "columns", []):
        return {}
    coverages = []
    for value in df["sim_dir"].dropna().astype(str):
        path = Path(value).expanduser()
        if path.is_file():
            coverages.append(_infer_time_coverage(path.parent, files=[path], data_groupby="Single"))
    return _combine_station_time_coverages(coverages)


def _combine_station_time_coverages(coverages: list[dict]) -> dict:
    coverages = [coverage for coverage in coverages if coverage]
    if not coverages:
        return {}

    counts = [int(coverage["time_count"]) for coverage in coverages if coverage.get("time_count")]
    starts = [datetime.fromisoformat(coverage["time_start"]) for coverage in coverages if coverage.get("time_start")]
    ends = [datetime.fromisoformat(coverage["time_end"]) for coverage in coverages if coverage.get("time_end")]
    months = sorted({month for coverage in coverages for month in (coverage.get("months") or [])})
    years = [year for coverage in coverages for year in (coverage.get("years") or []) if year is not None]
    file_time_counts = [
        int(count) for coverage in coverages for count in (coverage.get("file_time_counts") or []) if int(count) > 0
    ]

    result: dict[str, Any] = {
        "has_time": any(coverage.get("has_time") for coverage in coverages),
    }
    if counts:
        result["time_count"] = _median_positive_int(counts)
    if starts and ends:
        start = min(starts)
        end = max(ends)
        result.update(
            {
                "time_start": _format_time(start),
                "time_end": _format_time(end),
                "time_span_days": int((end - start).days),
                "time_span_seconds": int((end - start).total_seconds()),
            }
        )
    if years:
        result["years"] = [min(years), max(years)]
    if months:
        result["months"] = months
    if file_time_counts:
        result["file_time_counts"] = file_time_counts
    for key in ("time_units", "raw_time_steps", "raw_time_units"):
        values = [value for coverage in coverages for value in (coverage.get(key) or [])]
        if values:
            result[key] = values
    return result


_DATE_DIR_NAME_RE = re.compile(r"^(?:19|20)\d{2}(?:[-_]?(?:0[1-9]|1[0-2])(?:[-_]?(?:0[1-9]|[12]\d|3[01]))?)?$")


def _is_date_dir_name(name: str) -> bool:
    return bool(_DATE_DIR_NAME_RE.fullmatch(name))


def _children_are_date_subdir_layout(children: list[Path]) -> bool:
    """Return True when sibling directories are date-named and each has NC files."""
    if len(children) < 2:
        return False
    found_with_nc = False
    for child in children:
        if not _is_date_dir_name(child.name):
            return False
        if glob_nc(child):
            found_with_nc = True
        else:
            return False
    return found_with_nc


def _iter_case_dirs(
    root: Path,
    max_depth: int,
    exclude_patterns: tuple[str, ...],
):
    def _resolve_dir(path: Path) -> Path:
        try:
            return path.resolve(strict=False)
        except OSError:
            return path.absolute()

    visited: set[Path] = set()
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        current_real = _resolve_dir(current)
        if current_real in visited:
            continue
        visited.add(current_real)

        station_layout, station_metadata_dir = _station_collection_info(current)
        if station_layout and station_metadata_dir is not None:
            yield current, station_metadata_dir, depth, station_layout
            continue

        try:
            children = sorted(child for child in current.iterdir() if child.is_dir())
        except OSError:
            children = []
        children = [
            child
            for child in children
            if not _is_excluded(child, exclude_patterns) and _resolve_dir(child) not in visited
        ]
        direct_nc = bool(glob_nc(current))

        if not direct_nc and _children_are_date_subdir_layout(children):
            yield current, children[0], depth, None
            continue

        if direct_nc:
            sibling_cases = [child for child in children if glob_nc(child) or _station_collection_info(child)[0]]
            if len(sibling_cases) >= 2 and depth < max_depth:
                for child in reversed(children):
                    stack.append((child, depth + 1))
                continue
            yield current, current, depth, None
            continue

        if depth >= max_depth:
            continue
        for child in reversed(children):
            stack.append((child, depth + 1))


def _station_collection_info(path: Path) -> tuple[str | None, Path | None]:
    direct_files = glob_nc(path)
    if direct_files:
        info = inspect_nc_file(path)
        if info.get("detected_data_type") == "stn":
            return "flat", path
        return None, None

    child_dirs = _station_child_nc_dirs(path)
    if not child_dirs:
        return None, None

    inspected = []
    for nc_dir in child_dirs:
        info = inspect_nc_file(nc_dir)
        if info.get("detected_data_type") != "stn":
            return None, None
        inspected.append((nc_dir, len(glob_nc(nc_dir)), info))

    if not inspected:
        return None, None

    site_names = [_station_site_name(path, nc_dir) for nc_dir, _count, _info in inspected]
    has_direct_history_layout = any(
        nc_dir.name.lower() == "history" and len(_station_relative_parts(path, nc_dir)) == 1
        for nc_dir, _count, _info in inspected
    )
    has_named_site = any(_looks_like_station_site_name(name) for name in site_names)
    has_direct_site_identity = (
        len(inspected) >= 2
        and _station_dirs_are_homogeneous(inspected)
        and all(_station_dir_matches_site_identity(path, nc_dir) for nc_dir, _count, _info in inspected)
    )
    if not has_direct_history_layout and not has_named_site and not has_direct_site_identity:
        return None, None

    layout = "nested_multi" if any(count > 1 for _nc_dir, count, _info in inspected) else "nested_single"
    return layout, inspected[0][0]


def _station_child_nc_dirs(path: Path) -> list[Path]:
    try:
        children = sorted(child for child in path.iterdir() if child.is_dir())
    except OSError:
        return []

    nc_dirs = []
    for child in children:
        history = child / "history"
        if history.is_dir() and _station_nc_dir_has_time(history):
            nc_dirs.append(history)
        elif _station_nc_dir_has_time(child):
            nc_dirs.append(child)
    return nc_dirs


def _station_nc_dir_has_time(nc_dir: Path) -> bool:
    files = glob_nc(nc_dir)
    if not files:
        return False
    info = _time_info_from_file(files[0])
    return bool(info.get("readable") and info.get("has_time"))


def _station_site_name(collection_root: Path, nc_dir: Path) -> str:
    """Return the child directory that would represent a station/site."""
    rel_parts = _station_relative_parts(collection_root, nc_dir)
    if not rel_parts:
        return nc_dir.name
    return rel_parts[0]


def _station_relative_parts(collection_root: Path, nc_dir: Path) -> tuple[str, ...]:
    try:
        return nc_dir.relative_to(collection_root).parts
    except ValueError:
        return (nc_dir.name,)


def _looks_like_station_site_name(name: str) -> bool:
    lower = name.lower()
    if re.match(r"^[a-z]{2}[-_][a-z0-9]+$", lower):
        return True
    if lower.startswith(("site", "stn", "station")):
        return True
    return bool(re.fullmatch(r"\d{3,}", lower))


def _station_dirs_are_homogeneous(inspected: list[tuple[Path, int, dict]]) -> bool:
    signatures = {
        tuple((item.get("name"), tuple(item.get("dims", []))) for item in info.get("all_data_vars", []))
        for _nc_dir, _count, info in inspected
    }
    return len(signatures) == 1


def _station_dir_matches_site_identity(collection_root: Path, nc_dir: Path) -> bool:
    site_name = _station_site_name(collection_root, nc_dir)
    normalized_site = _identity_token(site_name)
    if not normalized_site:
        return False

    for file_path in glob_nc(nc_dir):
        station_id = _station_identity_from_nc(file_path)
        if station_id and _identity_token(station_id) == normalized_site:
            return True
    return False


def _station_identity_from_nc(file_path: Path) -> str | None:
    try:
        import xarray as xr

        with xr.open_dataset(file_path, decode_times=False) as ds:
            for key in ("station_id", "site_id", "station", "site"):
                value = ds.attrs.get(key)
                if value:
                    return str(value)
                if key in ds.variables:
                    raw = ds[key].values
                    if getattr(raw, "ndim", 0) == 0:
                        return str(raw.item())
    except Exception:
        return None
    return None


def _identity_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _is_excluded(path: Path, exclude_patterns: tuple[str, ...]) -> bool:
    name = path.name
    lower_name = name.lower()
    return any(fnmatch(name, pattern) or fnmatch(lower_name, pattern.lower()) for pattern in exclude_patterns)


def _derive_case_label(root: Path, nc_dir: Path) -> str:
    try:
        parts = nc_dir.relative_to(root).parts
    except ValueError:
        parts = nc_dir.parts
    if not parts:
        if root.name.lower() in _DATA_DIR_HINTS and root.parent.name:
            return root.parent.name
        return nc_dir.name or root.name
    if parts[-1].lower() in _DATA_DIR_HINTS:
        if len(parts) >= 2:
            return parts[-2]
        return root.name or parts[-1]
    return parts[-1]
