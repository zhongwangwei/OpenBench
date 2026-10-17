"""openbench check command."""

from __future__ import annotations

import functools
import os
import re
from contextlib import contextmanager
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import click

from openbench.cli._names import resolve_variable_filters
from openbench.cli._reference_errors import emit_reference_resolution_error
from openbench.cli._simulation_validation import simulation_root_errors
from openbench.config.provenance import PROVENANCE_FIELDS
from openbench.util.names import get_mapping_key_case_insensitive
from openbench.util.static_datasets import static_dataset_candidates, static_dataset_exists

_UNRESOLVED_ENV_RE = re.compile(r"(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%)")
_VALID_DATA_GROUPBY = {"year", "month", "day", "single"}
_VALID_WEIGHTS = {"area", "mass", "none"}
_BASIC_ANALYSIS_ITEMS = {"Basic", "Mean", "Median", "Max", "Min", "Sum"}


def _reference_root_value(cfg, ref_ds) -> str | None:
    root_dir = getattr(ref_ds, "root_dir", None)
    if getattr(ref_ds, "data_type", None) == "stn":
        return root_dir or cfg.reference.data_root
    return cfg.reference.data_root or root_dir


def _expanded_path(raw: str, what: str) -> tuple[Path | None, str | None]:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if _UNRESOLVED_ENV_RE.search(expanded):
        return None, f"{what} contains unresolved environment variable: {raw}"
    return Path(expanded), None


def _expanded_reference_path(raw: str) -> tuple[Path | None, str | None]:
    return _expanded_path(raw, "Reference root")


def _has_nearby_netcdf_files(path: Path) -> bool:
    from openbench.data.file_lookup import iter_netcdf_paths

    return next(iter_netcdf_paths(str(path)), None) is not None


def _figlib_names(section: str) -> set[str]:
    from importlib.resources import files

    import yaml

    # Keep this as a Traversable so it works when OpenBench is imported
    # directly from a zipped wheel.
    figlib = files("openbench.data.fignml") / "figlib.yaml"
    try:
        data = yaml.safe_load(figlib.read_text(encoding="utf-8")) or {}
    except OSError:
        return set()
    names = set()
    for key in data.get(section) or {}:
        names.add(key[:-7] if key.endswith("_source") else key)
    return names


def _suggestion(name: str, valid: set[str]) -> str:
    matches = get_close_matches(name, sorted(valid), n=1, cutoff=0.55)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def _validate_string_list(raw: Any, label: str) -> tuple[list[str], list[str]]:
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [f"{label} must be a list of strings, got {type(raw).__name__}"]
    errors = []
    values = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            errors.append(f"{label}[{idx}] must be a string, got {type(item).__name__}")
        else:
            values.append(item)
    return values, errors


def _validate_names(kind: str, values: list[str], valid: set[str]) -> list[str]:
    errors = []
    for name in values:
        if name not in valid:
            errors.append(f"Unknown {kind} '{name}'.{_suggestion(name, valid)}")
    return errors


def _timezone_findings(value: Any, label: str) -> tuple[list[str], list[str]]:
    if value is None:
        return [], []
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{label} must be a numeric hour offset, got {type(value).__name__}"], []
    if value < -12 or value > 14:
        return [], [f"{label} timezone {value} is outside the usual [-12, 14] hour range"]
    return [], []


def _data_groupby_error(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.lower() not in _VALID_DATA_GROUPBY:
        return f"{label} data_groupby '{value}' is invalid; expected one of Year, Month, Day, single"
    return None


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path if path.exists() else path.parent
    while current != current.parent:
        if current.exists():
            return current
        current = current.parent
    return current if current.exists() else None


def _config_findings(cfg) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    output_path, output_error = _expanded_path(str(cfg.project.output_dir), "project.output_dir")
    if output_error:
        errors.append(output_error)
    elif output_path is not None:
        if output_path.exists() and not output_path.is_dir():
            errors.append(f"project.output_dir exists but is not a directory: {output_path}")
        else:
            writable_target = output_path if output_path.exists() else _nearest_existing_parent(output_path)
            if writable_target is None:
                warnings.append(f"project.output_dir has no existing parent to check: {output_path}")
            elif output_path.exists() and not os.access(writable_target, os.W_OK):
                errors.append(f"project.output_dir parent is not writable: {writable_target}")
            elif not os.access(writable_target, os.W_OK):
                warnings.append(f"project.output_dir parent may not be writable: {writable_target}")

    weight = cfg.project.weight
    if weight is not None and str(weight).lower() not in _VALID_WEIGHTS:
        errors.append(f"project.weight must be one of {sorted(_VALID_WEIGHTS)}, got '{weight}'")

    tz_errors, tz_warnings = _timezone_findings(cfg.project.timezone, "project")
    errors.extend(tz_errors)
    warnings.extend(tz_warnings)

    metric_values, metric_type_errors = _validate_string_list(cfg.metrics, "metrics")
    score_values, score_type_errors = _validate_string_list(cfg.scores, "scores")
    errors.extend(metric_type_errors)
    errors.extend(score_type_errors)
    if cfg.metrics == [] and cfg.scores == []:
        errors.append("metrics and scores cannot both be empty; select at least one evaluation output")
    if metric_values:
        from openbench.core.registry import IMPLEMENTED_METRICS

        errors.extend(_validate_names("metric", metric_values, IMPLEMENTED_METRICS))
    if score_values:
        from openbench.core.registry import IMPLEMENTED_SCORES

        errors.extend(_validate_names("score", score_values, IMPLEMENTED_SCORES))

    comparison_items, comparison_type_errors = _validate_string_list(
        cfg.comparison.items,
        "comparison.items",
    )
    statistic_items, statistic_type_errors = _validate_string_list(
        cfg.statistics.items,
        "statistics.items",
    )
    errors.extend(comparison_type_errors)
    errors.extend(statistic_type_errors)
    if comparison_items:
        valid = _figlib_names("comparison_nml") | _BASIC_ANALYSIS_ITEMS
        errors.extend(_validate_names("comparison item", comparison_items, valid))
    if statistic_items:
        valid = _figlib_names("statistic_nml") | _BASIC_ANALYSIS_ITEMS | {"False_Discovery_Rate"}
        errors.extend(_validate_names("statistics item", statistic_items, valid))

    for label, entry in cfg.simulation.items():
        err = _data_groupby_error(entry.data_groupby, f"simulation.{label}")
        if err:
            errors.append(err)
        for var_name, inline in (entry.variables or {}).items():
            if not isinstance(inline, dict):
                errors.append(f"simulation.{label}.variables.{var_name} must be a mapping, got {type(inline).__name__}")
                continue
            err = _data_groupby_error(inline.get("data_groupby"), f"simulation.{label}.variables.{var_name}")
            if err:
                errors.append(err)
            tz_errors, tz_warnings = _timezone_findings(
                inline.get("timezone"),
                f"simulation.{label}.variables.{var_name}",
            )
            errors.extend(tz_errors)
            warnings.extend(tz_warnings)

    return errors, warnings


def _groupby_static_dataset_findings(cfg) -> list[str]:
    errors: list[str] = []
    groupby_requirements = {
        "IGBP_groupby": ("IGBP.nc", cfg.project.IGBP_groupby),
        "PFT_groupby": ("PFT.nc", cfg.project.PFT_groupby),
        "climate_zone_groupby": ("Climate_zone.nc", cfg.project.climate_zone_groupby),
    }
    for label, (filename, enabled) in groupby_requirements.items():
        if not enabled:
            continue
        candidates = static_dataset_candidates(filename)
        if not static_dataset_exists(filename):
            checked = ", ".join(Path(p).as_posix() for p in candidates)
            errors.append(f"{label} requires static dataset {filename} (checked: {checked})")
    return errors


# Resolved reference directories for the check() call in progress. Resolving
# one lists its data directory, and the reference, file and simulation checks
# all need it, once per simulation and variable without this.
_EFFECTIVE_REFERENCES: dict[tuple[int, int], tuple] | None = None
_FILE_INVENTORIES: dict[str, list[str]] | None = None
_UNREAD_FOLDERS: dict[str, list[str]] | None = None


@contextmanager
def _reference_resolution_cache():
    global _EFFECTIVE_REFERENCES, _FILE_INVENTORIES, _UNREAD_FOLDERS
    previous = _EFFECTIVE_REFERENCES, _FILE_INVENTORIES, _UNREAD_FOLDERS
    _EFFECTIVE_REFERENCES, _FILE_INVENTORIES, _UNREAD_FOLDERS = {}, {}, {}
    try:
        yield
    finally:
        _EFFECTIVE_REFERENCES, _FILE_INVENTORIES, _UNREAD_FOLDERS = previous


def _file_inventory(directory: str) -> list[str]:
    from openbench.data.file_lookup import netcdf_inventory

    if _FILE_INVENTORIES is None:
        return netcdf_inventory(directory)
    key = os.path.normcase(os.path.abspath(directory))
    if key not in _FILE_INVENTORIES:
        _FILE_INVENTORIES[key] = netcdf_inventory(directory)
    return _FILE_INVENTORIES[key]


def _unread_folders(directory: str) -> list[str]:
    from openbench.data.file_lookup import unread_folders

    if _UNREAD_FOLDERS is None:
        return unread_folders(directory)
    key = os.path.normcase(os.path.abspath(directory))
    if key not in _UNREAD_FOLDERS:
        _UNREAD_FOLDERS[key] = unread_folders(directory)
    return _UNREAD_FOLDERS[key]


def _with_reference_resolution_cache(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with _reference_resolution_cache():
            return function(*args, **kwargs)

    return wrapper


def _effective_reference(cfg, resolved_ref):
    cache = _EFFECTIVE_REFERENCES
    key = (id(cfg), id(resolved_ref))
    if cache is not None and key in cache:
        return cache[key][2]
    result = _resolve_effective_reference(cfg, resolved_ref)
    if cache is not None:
        cache[key] = (cfg, resolved_ref, result)  # holding both keeps their ids from being reused
    return result


def _resolve_effective_reference(cfg, resolved_ref):
    from openbench.config.adapter import _apply_reference_override, _source_override, reference_data_dir

    override = _source_override(cfg, resolved_ref.source_name) or _source_override(cfg, resolved_ref.resolved_name)
    ref_ds, var_map = _apply_reference_override(
        resolved_ref.ref_ds, resolved_ref.var_map, resolved_ref.var_name, override
    )
    root, directory = reference_data_dir(cfg, ref_ds, var_map, override)
    return ref_ds, var_map, root, directory


def reference_data_findings(ref_ds, var_map, root: str, directory: str):
    """Common init/check directory and station-dataset validation."""
    if not root or not str(root).strip():
        return (
            ["Reference root is not configured; set reference.data_root or register the reference with --root-dir."],
            [],
            [],
        )
    root_path, error = _expanded_reference_path(str(root))
    path, path_error = _expanded_reference_path(str(directory))
    if error or path_error or root_path is None or path is None:
        return [error or path_error or "Reference root could not be resolved."], [], []
    info = [f"effective root: {root_path}"]
    matching = getattr(ref_ds, "station_matching", None)
    dataset_file = getattr(matching, "dataset_file", "") if matching else ""
    if dataset_file:
        path = root_path
    if not path.is_dir():
        label = "Reference data path" if getattr(var_map, "sub_dir", None) else "Reference root"
        return [f"{label} does not exist or is not a directory: {path}"], [], info
    warnings = []
    direct_path = root_path / (getattr(var_map, "sub_dir", None) or "")
    if not dataset_file and path != direct_path:
        warnings.append(f"Reference data path uses fallback: {path}")
    if dataset_file:
        from openbench.data.station_matcher import resolve_station_dataset, station_dataset_candidates

        if resolve_station_dataset(root_path, dataset_file) is None:
            tried = " or ".join(str(p) for p in station_dataset_candidates(root_path, dataset_file))
            return [f"Reference dataset file does not exist: {tried}"], warnings, info
    elif not _has_nearby_netcdf_files(path):
        if getattr(ref_ds, "data_type", "grid") != "stn":
            # Gridded preprocessing reads files from this tree; with none it cannot run.
            return [f"Reference data path has no NetCDF files: {path}"], warnings, info
        # Station files may be listed by a station list rather than found here.
        warnings.append(f"Reference root has no NetCDF files found near: {path}")
    return [], warnings, info


def _reference_data_findings(cfg, resolved_ref) -> tuple[list[str], list[str], list[str]]:
    if resolved_ref.status != "ok" or resolved_ref.ref_ds is None:
        return [], [], []
    return reference_data_findings(*_effective_reference(cfg, resolved_ref))


def _probe_years(project_years: list[int], data_years: Any) -> list[int]:
    """Check every required year, clipped when runtime uses an intersection."""
    start, end = project_years[0], project_years[1]
    if isinstance(data_years, list) and len(data_years) >= 2:
        start, end = max(start, data_years[0]), min(end, data_years[1])
    return list(range(start, end + 1)) if start <= end else []


def _station_year_span(fulllist, root, default, *, side):
    """Use the station list's coverage when it narrows catalog/project years."""
    import csv

    from openbench.config.adapter import _resolve_root_relative_path

    if not fulllist:
        return default
    try:
        starts, ends = [], []
        with open(_resolve_root_relative_path(fulllist, root), newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                fields = {key.casefold(): value for key, value in row.items() if key is not None}
                try:
                    start = next(fields[key] for key in (f"{side}_syear", "syear", "use_syear") if key in fields)
                    end = next(fields[key] for key in (f"{side}_eyear", "eyear", "use_eyear") if key in fields)
                    start, end = int(float(start)), int(float(end))
                except (StopIteration, TypeError, ValueError, OverflowError):
                    continue
                if start <= end:
                    starts.append(start)
                    ends.append(end)
        return [min(starts), max(ends)] if starts else default
    except (OSError, csv.Error, UnicodeError):
        return default  # The existing fulllist check reports unreadable paths.


def _simulation_probe_years(cfg, variable, resolved_refs):
    """Grid-only preprocessing reads project years; station pairs read overlaps."""
    required = set()
    references = [ref for ref in resolved_refs if ref.var_name == variable and ref.status == "ok"]
    if not references:
        return _probe_years(cfg.project.years, None)
    for resolved in references:
        ref, mapping, root, _directory = _effective_reference(cfg, resolved)
        if str(getattr(ref, "data_type", "grid")).lower() != "stn":
            return _probe_years(cfg.project.years, None)
        span = _station_year_span(
            getattr(mapping, "fulllist", None) or getattr(ref, "fulllist", None),
            root,
            getattr(ref, "years", None),
            side="ref",
        )
        required.update(_probe_years(cfg.project.years, span))
    return sorted(required)


def _year_ranges(years: list[int]) -> str:
    """``[2001, 2002, 2003, 2008]`` -> ``"2001–2003, 2008"``."""
    ranges: list[str] = []
    ordered = sorted(dict.fromkeys(years))
    start = previous = None
    for year in ordered + [None]:
        if year is not None and previous is not None and year == previous + 1:
            previous = year
            continue
        if start is not None:
            ranges.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = year
    return ", ".join(ranges)


def data_file_findings(
    kind: str,
    data_dir: str,
    *,
    prefix: str,
    suffix: str,
    data_groupby: str,
    years: list[int],
    prefix_fallback: Any = None,
    compute: Any = None,
    candidate_varnames: Any = (),
    standard_varname: str = "",
    fallback_varnames: Any = None,
    inventory: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Check the files runtime selects, including variable-aware compute fallback."""
    from openbench.data.compute import compute_dependency_names
    from openbench.data.file_lookup import mixed_branches, prefix_candidates, select_data_files, year_matches

    single = str(data_groupby).strip().lower() == "single"
    dependencies = compute_dependency_names(compute) if compute else []
    candidates = list(dict.fromkeys([*candidate_varnames, *dependencies]))
    if inventory is None:
        inventory = _file_inventory(data_dir)
    layout_errors, layout_warnings = [], []
    if not single and (unread := _unread_folders(data_dir)):
        layout_warnings.append(
            f"{kind} directory {data_dir} is read through its year folders; {', '.join(unread)} "
            f"also {'holds' if len(unread) == 1 else 'hold'} NetCDF files that are not read "
            "(set sub_dir to read such a folder instead)"
        )

    # Index once per prefix, not once per year. Selection still decides which
    # prefix is usable from its file contents, exactly as preprocessing does.
    indexed = (
        {}
        if single
        else {
            name: year_matches(data_dir, name, suffix, years, inventory=inventory)
            for name in prefix_candidates(prefix or "", prefix_fallback)
        }
    )
    selected, computed = {}, {}
    for year in [None] if single else years:
        files, used_compute = select_data_files(
            data_dir,
            prefix or "",
            suffix or "",
            year,
            prefix_fallback=prefix_fallback,
            candidate_varnames=candidates,
            dependencies=dependencies,
            inventory=inventory,
            named_matches=None if single else {name: matches[year] for name, matches in indexed.items()},
        )
        selected[year] = files
        if used_compute:
            computed[year] = files

    # Month/Day preprocessing computes each file before concatenating time.
    # Only unconditional reads are required: ds.get() and conditional branches
    # may legitimately omit an input, and an independent raw fallback may work.
    if compute and str(data_groupby).strip().lower() not in {"single", "year"}:
        from openbench.data.compute import compute_inputs_known, compute_required_inputs
        from openbench.data.file_lookup import variable_names

        required_names, sum_groups = compute_required_inputs(compute)
        required = {name.lower() for name in required_names}
        sum_groups = [{name.lower() for name in group} for group in sum_groups]
        fallback_order = [*(candidate_varnames if fallback_varnames is None else fallback_varnames), standard_varname]
        inputs = {name.lower() for name in dependencies}
        failures = []
        # Only the first and last selected file, in year order: opening every
        # daily header made check slow. A file in between that lacks an input is
        # still reported by preprocessing.
        ordered = list(dict.fromkeys(path for files in selected.values() for path in files))
        for path in dict.fromkeys(ordered[:1] + ordered[-1:]):
            names = variable_names(path)
            if names is None:
                continue  # The runtime loader reports unreadable files.
            missing = required - names
            target = next((name.lower() for name in fallback_order if name and name.lower() in names), None)
            partial_sum = any(group & names and not group <= names for group in sum_groups)
            if missing and (partial_sum or not (compute_inputs_known(compute) and target and target not in inputs)):
                failures.append(f"{os.path.relpath(path, data_dir)}: missing {', '.join(sorted(missing))}")
        if failures:
            details = "; ".join(failures[:3])
            if len(failures) > 3:
                details += f"; and {len(failures) - 3} more files"
            layout_errors.append(
                f"{kind} {data_groupby} compute runs on each file before combining time; "
                f"inputs in separate files cannot satisfy it ({details})"
            )

    for found, inputs, what in (
        (
            {year: files for year, files in selected.items() if year not in computed},
            candidates if dependencies else None,
            f"{kind} files",
        ),
        (computed, dependencies, f"{kind} files holding the compute inputs ({', '.join(dependencies)})"),
    ):
        groups: dict[tuple, list] = {}
        for year, branch_info in mixed_branches(data_dir, found, dependencies=inputs).items():
            groups.setdefault(branch_info, []).append(year)
        for (branches, repeated), branch_years in groups.items():
            folders = ", ".join(f"{branch}/" for branch in branches)
            span = "" if branch_years == [None] else f" for {_year_ranges(branch_years)}"
            if repeated:
                layout_errors.append(
                    f"{what}{span} exist with the same names in several folders of {data_dir} ({folders}); "
                    "preprocessing would read them together (set sub_dir to choose one folder)"
                )
            else:
                layout_warnings.append(
                    f"{what}{span} are split across folders of {data_dir} ({folders}) and are read together; "
                    "check they belong to one dataset, or set sub_dir to choose one folder"
                )
    missing_years = [year for year, files in selected.items() if not files]
    if not missing_years:
        return layout_errors, layout_warnings
    naming = f"prefix '{prefix or ''}', suffix '{suffix or ''}', data_groupby {data_groupby}"
    if single:
        missing = os.path.join(data_dir, f"{prefix or ''}{suffix or ''}.nc")
        messages = [f"{kind} data file not found: {missing} ({naming})"]
        from openbench.data.coordinates import glob_nc

        nearby = get_close_matches(Path(missing).name, [p.name for p in glob_nc(Path(data_dir))], n=3, cutoff=0.4)
        if nearby:
            messages[0] += f"; nearby files: {', '.join(nearby)}"
    else:
        template = f"{prefix or ''}<year>*{suffix or ''}.nc"
        messages = [
            f"{kind} data files not found in {data_dir}: {template} for {_year_ranges(missing_years)} ({naming})"
        ]
    if compute:
        note = (
            f"; no files holding all compute inputs ({', '.join(dependencies)}) were found either"
            if dependencies
            else "; preprocessing will look for files holding the compute inputs"
        )
        return layout_errors, [*layout_warnings, *(f"{message}{note}" for message in messages)]
    return [*messages, *layout_errors], layout_warnings


def _file_candidate_names(varname, fallbacks=()) -> list[str]:
    names = [varname] if isinstance(varname, str) else list(varname or [])
    names.extend(
        fallback.get("varname", "") if isinstance(fallback, dict) else getattr(fallback, "varname", "")
        for fallback in fallbacks or []
    )
    return [name for name in dict.fromkeys(names) if name]


def _naming_origin_hint(cfg, resolved_ref, var_map) -> str | None:
    """Say where a file-naming field that differs from the bundled catalog was set."""
    from openbench.config.adapter import _source_override
    from openbench.data.registry import overlay_audit as oa
    from openbench.util.names import normalize_name

    match = oa._bundled_lookup(oa._bundled_catalog("references")).get(normalize_name(resolved_ref.resolved_name))
    if match is None:
        return None
    bundled_var = oa._raw_variable(match[1].get("variables") or {}, resolved_ref.var_name) or {}
    changed = [
        field
        for field in ("prefix", "suffix", "sub_dir")
        if (getattr(var_map, field, None) or "") != (bundled_var.get(field) or "")
    ]
    if not changed:
        return None
    override = _source_override(cfg, resolved_ref.source_name) or _source_override(cfg, resolved_ref.resolved_name)
    override_vars = override.get("variables") if isinstance(override.get("variables"), dict) else {}
    var_override = next(
        (value for key, value in override_vars.items() if str(key).lower() == str(resolved_ref.var_name).lower()), {}
    )
    fields = ", ".join(changed)
    if isinstance(var_override, dict) and any(field in var_override for field in changed):
        return f"{fields} set in reference.overrides of this config differ from the bundled catalog"
    wanted = normalize_name(resolved_ref.resolved_name)
    files = [str(path) for name, path in oa.per_file_overrides("references") if normalize_name(name) == wanted]
    if files:
        # Separate files merge after the catalog, and reset does not edit them.
        return f"{fields} differ from the bundled catalog (set in {', '.join(files)}; edit or remove the entry there)"
    return (
        f"{fields} differ from the bundled catalog (set in your ~/.openbench registry overlay; "
        f"`openbench registry reset {resolved_ref.resolved_name}` restores the bundled values)"
    )


def _reference_file_findings(cfg, resolved_ref, registry=None) -> tuple[list[str], list[str]]:
    """Check the reference files preprocessing will read for one resolved reference."""
    ref_ds, var_map, _data_root, ref_dir = _effective_reference(cfg, resolved_ref)
    if getattr(ref_ds, "data_type", None) == "stn" or var_map is None:
        return [], []
    path, error = _expanded_reference_path(str(ref_dir))
    if error or path is None or not path.is_dir() or not _has_nearby_netcdf_files(path):
        return [], []  # the directory checks already report a missing or empty directory
    required_years = _probe_years(
        cfg.project.years,
        None if getattr(cfg.project, "time_alignment", "intersection") == "strict" else getattr(ref_ds, "years", None),
    )
    if registry is None:
        from openbench.data.registry.manager import get_registry

        registry = get_registry()
    station_years = set()
    only_station_pairs = bool(cfg.simulation)
    for entry in cfg.simulation.values():
        profile = registry.get_model(entry.model) if hasattr(registry, "get_model") else None
        values = _effective_sim_values(entry, profile, resolved_ref.var_name)
        if str(values["data_type"]).lower() != "stn":
            only_station_pairs = False
            break
        span = _station_year_span(values["fulllist"], entry.root_dir, cfg.project.years, side="sim")
        station_years.update(_probe_years(cfg.project.years, span))
    if only_station_pairs:
        required_years = sorted(
            station_years.intersection(_probe_years(cfg.project.years, getattr(ref_ds, "years", None)))
        )
    errors, warnings = data_file_findings(
        "Reference",
        str(path),
        prefix=getattr(var_map, "prefix", ""),
        suffix=getattr(var_map, "suffix", ""),
        data_groupby=getattr(ref_ds, "data_groupby", "Year"),
        years=required_years,
        prefix_fallback=getattr(var_map, "prefix_fallback", None),
        compute=getattr(var_map, "compute", None),
        standard_varname=resolved_ref.var_name,
        fallback_varnames=_file_candidate_names(getattr(var_map, "varname", None))[:1]
        + _file_candidate_names([], getattr(var_map, "fallbacks", None)),
        candidate_varnames=_file_candidate_names(
            getattr(var_map, "varname", None), getattr(var_map, "fallbacks", None)
        ),
    )
    if errors or warnings:
        try:
            hint = _naming_origin_hint(cfg, resolved_ref, var_map)
        except Exception:  # a hint must never hide the finding itself
            hint = None
        if hint:
            errors = [f"{message}; {hint}" for message in errors]
            warnings = [f"{message}; {hint}" for message in warnings]
    return errors, warnings


def _tim_res_rank(value: str | None) -> int:
    from openbench.data.registry._tim_res import _tim_res_rank as scanner_tim_res_rank

    return scanner_tim_res_rank(value or "")


def _years_findings(
    name: str,
    data_years: Any,
    project_years: list[int],
    *,
    kind: str = "Reference",
    qualifier: str = "",
) -> tuple[list[str], list[str]]:
    years_label = f"{qualifier} years" if qualifier else "years"
    if not data_years:
        return [], [f"{kind} '{name}' has no registered years; using project years at runtime"]
    if not isinstance(data_years, list) or len(data_years) < 2:
        return [], [f"{kind} '{name}' {years_label} metadata is incomplete: {data_years}"]
    data_start, data_end = data_years[0], data_years[1]
    proj_start, proj_end = project_years[0], project_years[1]
    if data_end < proj_start or data_start > proj_end:
        return [
            f"{kind} '{name}' {years_label} [{data_start}, {data_end}] do not overlap "
            f"project years [{proj_start}, {proj_end}]"
        ], []
    if data_start > proj_start or data_end < proj_end:
        return [], [
            f"{kind} '{name}' {years_label} [{data_start}, {data_end}] only partially cover "
            f"project years [{proj_start}, {proj_end}]"
        ]
    return [], []


def _fulllist_path_findings(raw: str, label: str, root_dir: str | None = None) -> tuple[list[str], list[str]]:
    from openbench.config.adapter import _resolve_root_relative_path

    resolved = _resolve_root_relative_path(raw, root_dir)
    path, error = _expanded_path(resolved, label)
    if error:
        return [error], []
    if path is None or not path.exists() or not path.is_file():
        return [f"Station fulllist does not exist: {path or resolved}"], []
    if not os.access(path, os.R_OK):
        return [f"Station fulllist is not readable: {path}"], []
    return [], []


def _reference_metadata_findings(
    cfg,
    resolved_ref,
    target_tim_res: str | None,
    *,
    file_checks: bool = True,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if resolved_ref.status != "ok" or resolved_ref.ref_ds is None:
        return errors, warnings

    ref_ds, _var_map, _root, _directory = _effective_reference(cfg, resolved_ref)
    ref_name = resolved_ref.resolved_name
    ref_tim_res = getattr(ref_ds, "tim_res", None)
    if ref_tim_res and target_tim_res:
        ref_rank = _tim_res_rank(ref_tim_res)
        target_rank = _tim_res_rank(target_tim_res)
        if ref_rank < 0:
            warnings.append(f"Reference '{ref_name}' time resolution '{ref_tim_res}' is not recognized")
        elif target_rank >= 0 and ref_rank < target_rank:
            errors.append(
                f"Reference '{ref_name}' time resolution {ref_tim_res} is coarser than target {target_tim_res}"
            )

    err = _data_groupby_error(getattr(ref_ds, "data_groupby", None), f"reference.{ref_name}")
    if err:
        errors.append(err)

    tz_errors, tz_warnings = _timezone_findings(getattr(ref_ds, "timezone", None), f"reference.{ref_name}")
    errors.extend(tz_errors)
    warnings.extend(tz_warnings)

    year_errors, year_warnings = _years_findings(ref_name, getattr(ref_ds, "years", None), cfg.project.years)
    errors.extend(year_errors)
    warnings.extend(year_warnings)

    if file_checks and getattr(ref_ds, "data_type", None) == "stn":
        var_map = _var_map
        fulllist = getattr(var_map, "fulllist", None) or getattr(ref_ds, "fulllist", None)
        root_dir = getattr(ref_ds, "root_dir", None) or _reference_root_value(cfg, ref_ds)
        if fulllist:
            list_errors, list_warnings = _fulllist_path_findings(
                str(fulllist),
                f"reference.{ref_name}.fulllist",
                root_dir,
            )
            errors.extend(list_errors)
            warnings.extend(list_warnings)
        elif not getattr(ref_ds, "station_matching", None):
            warnings.append(
                f"Station reference '{ref_name}' has no fulllist; "
                "runtime will rely on station matching or custom filters"
            )

    return errors, warnings


def _effective_sim_values(entry, model_profile, var_name: str) -> dict[str, Any]:
    from openbench.config.adapter import simulation_file_layout

    layout = simulation_file_layout(entry, model_profile, var_name)
    inline_variables = entry.variables or {}
    inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
    inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
    profile_variables = getattr(model_profile, "variables", {}) if model_profile else {}
    profile_key = get_mapping_key_case_insensitive(profile_variables, var_name)
    profile_var = profile_variables.get(profile_key) if profile_key is not None else None
    return {
        "inline": inline,
        "data_type": inline.get("data_type")
        or entry.data_type
        or (getattr(model_profile, "data_type", None) if model_profile else "grid"),
        "tim_res": inline.get("tim_res")
        or entry.tim_res
        or (getattr(model_profile, "tim_res", None) if model_profile else None),
        "grid_res": inline.get("grid_res")
        if inline.get("grid_res") is not None
        else (
            entry.grid_res
            if entry.grid_res is not None
            else (getattr(model_profile, "grid_res", None) if model_profile else None)
        ),
        "fulllist": inline.get("fulllist") if "fulllist" in inline else entry.fulllist,
        "dir": layout["dir"],
        "data_groupby": layout["data_groupby"],
        "sub_dir": inline.get("sub_dir")
        if inline.get("sub_dir") is not None
        else getattr(profile_var, "sub_dir", None),
        "prefix": layout["prefix"],
        "suffix": layout["suffix"],
    }


def _simulation_data_years(entry, values: dict[str, Any], *, max_workers: int | None = None) -> list[int] | None:
    if str(values["data_type"]).lower() == "stn":
        return None

    root, error = _expanded_path(str(values["dir"]), "Simulation root")
    if error or root is None:
        return None
    data_dir = root

    from openbench.data.coordinates import glob_nc
    from openbench.data.sim_scanner import _infer_time_coverage

    prefix = values["prefix"] or ""
    suffix = values["suffix"] or ""
    files = [path for path in glob_nc(data_dir) if path.stem.startswith(prefix) and path.stem.endswith(suffix)]
    if not files:
        return None
    return _infer_time_coverage(
        data_dir,
        files=files,
        data_groupby=values["data_groupby"],
        max_workers=max_workers,
    ).get("years")


def _append_simulation_model_findings(findings: dict[str, dict[str, list[str]]], cfg, registry) -> None:
    for label, entry in cfg.simulation.items():
        can_validate_model = hasattr(registry, "get_model")
        model_profile = registry.get_model(entry.model) if can_validate_model else None
        if can_validate_model and model_profile is None:
            inline_variables = entry.variables or {}
            missing_inline = []
            for var_name in cfg.evaluation.variables:
                inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
                inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
                if not inline.get("varname"):
                    missing_inline.append(var_name)
            if missing_inline:
                findings[label]["errors"].append(
                    f"Model '{entry.model}' is not registered and lacks inline varname for: {', '.join(missing_inline)}"
                )
            else:
                findings[label]["info"].append(
                    f"Model '{entry.model}' has no registry profile; using inline variable mappings"
                )
        elif model_profile is not None:
            profile_name = getattr(model_profile, "name", entry.model)
            if str(profile_name).lower() != str(entry.model).lower():
                findings[label]["info"].append(f"model alias '{entry.model}' resolved to '{profile_name}'")

        for var_name in cfg.evaluation.variables:
            inline_variables = entry.variables or {}
            inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
            inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
            profile_variables = getattr(model_profile, "variables", {}) if model_profile is not None else {}
            profile_key = (
                get_mapping_key_case_insensitive(profile_variables, var_name) if model_profile is not None else None
            )
            if model_profile is not None and profile_key is None and not inline:
                findings[label]["errors"].append(
                    f"Variable '{var_name}' is not defined in model profile "
                    f"'{getattr(model_profile, 'name', entry.model)}'"
                )


def _simulation_model_error_messages(cfg, registry) -> list[str]:
    findings: dict[str, dict[str, list[str]]] = {
        label: {"errors": [], "warnings": [], "info": []} for label in cfg.simulation
    }
    _append_simulation_model_findings(findings, cfg, registry)
    return [message for label in cfg.simulation for message in findings[label]["errors"]]


def _simulation_findings(
    cfg,
    registry,
    *,
    comparison_only: bool,
    only_drawing: bool = False,
    resolved_refs=None,
) -> dict[str, dict[str, list[str]]]:
    findings: dict[str, dict[str, list[str]]] = {
        label: {"errors": [], "warnings": [], "info": []} for label in cfg.simulation
    }

    output_only = comparison_only or only_drawing
    if not output_only:
        for label, message in simulation_root_errors(cfg):
            findings[label]["errors"].append(message)

    _append_simulation_model_findings(findings, cfg, registry)

    if output_only:
        return findings

    from openbench.config.adapter import simulation_file_layout
    from openbench.config.resolver import resolve_all_references

    if resolved_refs is None:
        resolved_refs = (
            resolve_all_references(cfg, registry, strict=False) if hasattr(registry, "get_reference") else []
        )

    probe_years = {
        var_name: tuple(_simulation_probe_years(cfg, var_name, resolved_refs)) for var_name in cfg.evaluation.variables
    }
    year_cache: dict[tuple[str, str, str, str, str], list[int] | None] = {}
    file_cache: dict[tuple, tuple[list[str], list[str]]] = {}
    inventory_cache: dict[str, list[str]] = {}
    root_errors = {label for label, _message in simulation_root_errors(cfg)}
    for label, entry in cfg.simulation.items():
        can_validate_model = hasattr(registry, "get_model")
        model_profile = registry.get_model(entry.model) if can_validate_model else None
        for var_name in cfg.evaluation.variables:
            values = _effective_sim_values(entry, model_profile, var_name)
            cache_key = (
                str(entry.root_dir),
                str(values["sub_dir"] or ""),
                str(values["prefix"] or ""),
                str(values["suffix"] or ""),
                str(values["data_groupby"] or ""),
            )
            if cache_key not in year_cache:
                year_cache[cache_key] = _simulation_data_years(
                    entry,
                    values,
                    max_workers=getattr(cfg.project, "num_cores", None),
                )
            data_years = year_cache[cache_key]
            if data_years:
                year_errors, year_warnings = _years_findings(
                    label,
                    data_years,
                    cfg.project.years,
                    kind="Simulation",
                    qualifier="data",
                )
                for message in year_errors:
                    if message not in findings[label]["errors"]:
                        findings[label]["errors"].append(message)
                for message in year_warnings:
                    if message not in findings[label]["warnings"]:
                        findings[label]["warnings"].append(message)
            layout = simulation_file_layout(entry, model_profile, var_name)
            if label not in root_errors and str(layout["data_type"]).lower() != "stn":
                file_key = (
                    str(layout["dir"]),
                    str(layout["prefix"] or ""),
                    str(layout["suffix"] or ""),
                    str(layout["data_groupby"] or ""),
                    (layout["prefix_fallback"],)
                    if isinstance(layout["prefix_fallback"], str)
                    else tuple(layout["prefix_fallback"] or ()),
                    var_name,
                    str(layout["compute"] or ""),
                    tuple(_simulation_candidate_names(layout)),
                    probe_years[var_name],
                )
                if file_key not in file_cache:
                    file_cache[file_key] = _simulation_file_findings(
                        layout, file_key[-1], inventory_cache=inventory_cache, standard_varname=var_name
                    )
                file_errors, file_warnings = file_cache[file_key]
                for message in file_errors:
                    message = f"{var_name}: {message}"
                    if message not in findings[label]["errors"]:
                        findings[label]["errors"].append(message)
                for message in file_warnings:
                    message = f"{var_name}: {message}"
                    if message not in findings[label]["warnings"]:
                        findings[label]["warnings"].append(message)
            if str(values["data_type"]).lower() == "stn":
                fulllist = values["fulllist"]
                if fulllist:
                    inline_variables = entry.variables or {}
                    inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
                    inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
                    if isinstance(inline, dict) and inline.get("fulllist"):
                        fulllist_label = f"simulation.{label}.variables.{var_name}.fulllist"
                    else:
                        fulllist_label = f"simulation.{label}.fulllist"
                    list_errors, list_warnings = _fulllist_path_findings(
                        str(fulllist),
                        fulllist_label,
                        entry.root_dir,
                    )
                    findings[label]["errors"].extend(list_errors)
                    findings[label]["warnings"].extend(list_warnings)
                else:
                    findings[label]["warnings"].append(
                        f"Station simulation '{label}' variable '{var_name}' has no fulllist; "
                        "runtime will rely on station auto-scan or custom filters"
                    )

    return findings


def _simulation_candidate_names(layout) -> list[str]:
    profile_var = layout.get("profile_var")
    inline = layout.get("inline_vars", {})
    return [
        *_file_candidate_names(getattr(profile_var, "varname", None), getattr(profile_var, "fallbacks", None)),
        *_file_candidate_names(inline.get("varname"), inline.get("fallbacks")),
    ]


def _simulation_file_findings(
    layout: dict[str, Any],
    years: tuple[int, ...],
    *,
    inventory_cache: dict[str, list[str]] | None = None,
    standard_varname: str = "",
) -> tuple[list[str], list[str]]:
    """Check the simulation files preprocessing will read for one variable layout."""
    path, error = _expanded_path(str(layout["dir"]), "Simulation data directory")
    if error or path is None:
        return [error] if error else [], []
    if not path.is_dir():
        return [f"Simulation data directory does not exist: {path}"], []
    if inventory_cache is None:
        inventory_cache = {}
    if str(path) not in inventory_cache:
        inventory_cache[str(path)] = _file_inventory(str(path))
    inventory = inventory_cache[str(path)]
    if not inventory:
        return [f"Simulation data directory has no NetCDF files: {path}"], []
    return data_file_findings(
        "Simulation",
        str(path),
        prefix=layout["prefix"],
        suffix=layout["suffix"],
        data_groupby=layout["data_groupby"],
        years=list(years),
        prefix_fallback=layout["prefix_fallback"],
        compute=layout["compute"],
        candidate_varnames=_simulation_candidate_names(layout),
        standard_varname=standard_varname,
        fallback_varnames=_file_candidate_names(
            layout.get("inline_vars", {}).get("varname", getattr(layout.get("profile_var"), "varname", None))
        )[:1]
        + _file_candidate_names([], layout.get("inline_vars", {}).get("fallbacks"))
        + _file_candidate_names([], getattr(layout.get("profile_var"), "fallbacks", None)),
        inventory=inventory,
    )


@_with_reference_resolution_cache
def data_file_errors(cfg, resolved, registry) -> list[str]:
    """Raw-data errors ``openbench check`` reports, for ``openbench run --dry-run``.

    The reference and simulation lookups are the ones check and preprocessing
    use, so a dry run no longer accepts a configuration whose files are missing.
    """
    errors = []
    for r in resolved or []:
        if getattr(r, "status", "ok") != "ok":
            continue  # the resolver preflight already reports it
        ref_errors, _warnings, _info = _reference_data_findings(cfg, r)
        if not ref_errors:
            ref_errors, _warnings = _reference_file_findings(cfg, r, registry)
        errors.extend(f"{r.var_name} → {r.source_name}: {message}" for message in ref_errors)
    findings = _simulation_findings(cfg, registry, comparison_only=False, resolved_refs=resolved)
    for label, label_findings in findings.items():
        errors.extend(f"{label}: {message}" for message in label_findings["errors"])
    return errors


def _format_optional_list(value: Any) -> str:
    if value is None:
        return "all"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


@click.command()
@click.option(
    "--comparison-only",
    is_flag=True,
    help="Validate for comparison-only runs and skip local simulation root checks.",
)
@click.option(
    "--strict-reference",
    "--strict",
    is_flag=True,
    help="Treat low-confidence reference metadata as errors without editing the YAML.",
)
@click.option(
    "--variable",
    "--variables",
    "variables",
    multiple=True,
    help="Validate only specified evaluation variable (repeatable). --variables retained as alias.",
)
@click.argument("config", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@_with_reference_resolution_cache
def check(config, comparison_only=False, strict_reference=False, variables=()):
    """Validate config and check data files for every required evaluation year."""
    from openbench.cli.run import _expand_config_paths
    from openbench.config import ConfigError, load_config

    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"✗ Config error: {e}", fg="red", bold=True)
        raise SystemExit(1) from e
    _expand_config_paths(cfg)
    if variables:
        cfg.evaluation.variables = resolve_variable_filters(variables, cfg.evaluation.variables)

    click.secho("Config validation:", bold=True)
    click.secho("  ✓ YAML syntax valid", fg="green")
    click.secho("  ✓ Schema validation passed", fg="green")
    click.secho(
        f"  ✓ Year range [{cfg.project.years[0]}, {cfg.project.years[1]}] valid",
        fg="green",
    )

    has_errors = False
    config_errors, config_warnings = _config_findings(cfg)
    for message in config_errors:
        click.secho(f"  ✗ {message}", fg="red")
        has_errors = True
    for message in config_warnings:
        click.secho(f"  ⚠ {message}", fg="yellow")

    _n_ref_total = sum(1 if isinstance(v, str) else len(v) for v in cfg.reference.sources.values())
    _n_ref_vars = len(cfg.reference.sources)
    _ref_summary = (
        f"{_n_ref_total} sources for {_n_ref_vars} variables"
        if _n_ref_total != _n_ref_vars
        else f"{_n_ref_vars} sources"
    )
    click.secho(f"\nReference data ({_ref_summary}):", bold=True)
    from openbench.config.resolver import (
        PROVENANCE_LOW,
        PROVENANCE_MEDIUM,
        derive_target_resolution_context,
        resolve_all_references,
    )
    from openbench.data.registry.manager import get_registry

    mgr = get_registry()
    strict = cfg.project.strict_reference or strict_reference

    try:
        resolved = resolve_all_references(cfg, mgr, strict=strict)
    except Exception as e:
        # Report resolver guidance without Click adding a second error message.
        emit_reference_resolution_error(str(e), prefix="  ✗ ")
        raise SystemExit(1) from e

    for r in resolved:
        if r.status == "ok":
            if r.resolved_name != r.source_name:
                click.secho(
                    f"  • {r.var_name} → {r.source_name} → {r.resolved_name} "
                    f"({r.ref_ds.data_type}, {r.ref_ds.tim_res}, "
                    f"{f'{r.ref_ds.grid_res}°' if r.ref_ds.grid_res is not None else 'N/A'})",
                    fg="cyan",
                )
            else:
                click.secho(
                    f"  • {r.var_name} → {r.source_name} ({r.ref_ds.data_type}, {r.ref_ds.tim_res})",
                    fg="cyan",
                )
            ds_prov = getattr(r.ref_ds, "_provenance", None) or {}
            for fld in PROVENANCE_FIELDS:
                source = ds_prov.get(fld)
                if not source:
                    continue
                value = getattr(r.ref_ds, fld, "?")
                if source in PROVENANCE_LOW:
                    if strict:
                        click.secho(
                            f"    ✗ {fld}: {value} (unconfirmed default)",
                            fg="red",
                        )
                        has_errors = True
                    else:
                        click.secho(
                            f"    ⚠ {fld}: {value} (default - not confirmed from NC or profile)",
                            fg="yellow",
                        )
                elif source in PROVENANCE_MEDIUM:
                    click.secho(
                        f"    ~ {fld}: {value} (inferred from directory structure)",
                        fg="cyan",
                    )

            try:
                target_ctx = derive_target_resolution_context(cfg, mgr, var_name=r.var_name)
            except ConfigError:
                target_ctx = None
            target_tim_res = target_ctx.tim_res if target_ctx is not None else None
            ref_meta_errors, ref_meta_warnings = _reference_metadata_findings(
                cfg,
                r,
                target_tim_res,
                file_checks=not comparison_only,
            )
            if comparison_only or cfg.project.only_drawing:
                # Output-only modes read existing results, not the raw reference files.
                ref_errors, ref_warnings, ref_info = [], [], []
            else:
                ref_errors, ref_warnings, ref_info = _reference_data_findings(cfg, r)
                if not ref_errors and not cfg.project.only_drawing:
                    file_errors, file_warnings = _reference_file_findings(cfg, r, mgr)
                    ref_errors = [*ref_errors, *file_errors]
                    ref_warnings = [*ref_warnings, *file_warnings]
            for message in ref_info:
                click.echo(f"    {message}")
            for message in [*ref_meta_errors, *ref_errors]:
                click.secho(f"    ✗ {message}", fg="red")
                has_errors = True
            for message in [*ref_meta_warnings, *ref_warnings]:
                click.secho(f"    ⚠ {message}", fg="yellow")
        elif r.status == "no_variable":
            click.secho(f"  ✗ {r.var_name} → {r.resolved_name}: {r.message}", fg="red")
            has_errors = True
        elif r.status == "ambiguous":
            click.secho(f"  ✗ {r.var_name} → {r.source_name}", fg="red")
            click.echo(f"    {r.message}")
            has_errors = True
        elif r.status == "not_found":
            if r.source_name:
                click.secho(
                    f"  ✗ {r.var_name} → {r.source_name} "
                    "(not in registry; runtime fallback would use minimal defaults)",
                    fg="red",
                )
                has_errors = True
            else:
                click.secho(f"  ✗ {r.var_name}: no reference configured", fg="red")
                has_errors = True

    n_tasks = len(resolved) * len(cfg.simulation)
    if n_tasks:
        click.echo(f"  Evaluation tasks: {n_tasks} ({len(resolved)} references × {len(cfg.simulation)} simulations)")

    click.secho(f"\nSimulation data ({len(cfg.simulation)} models):", bold=True)
    sim_findings = _simulation_findings(
        cfg,
        mgr,
        comparison_only=comparison_only,
        only_drawing=cfg.project.only_drawing,
        resolved_refs=resolved,
    )
    for label, entry in cfg.simulation.items():
        label_findings = sim_findings[label]
        if label_findings["errors"]:
            click.secho(
                f"  ✗ {label} (model: {entry.model}, root: {entry.root_dir})",
                fg="red",
            )
            has_errors = True
        else:
            click.secho(f"  ✓ {label} (model: {entry.model}, root: {entry.root_dir})", fg="green")
        for message in label_findings["info"]:
            click.secho(f"    ~ {message}", fg="cyan")
        for message in label_findings["errors"]:
            click.echo(f"    {message}")
        for message in label_findings["warnings"]:
            click.secho(f"    ⚠ {message}", fg="yellow")

    if cfg.metrics is not None:
        click.secho(f"\nMetrics: {_format_optional_list(cfg.metrics)}", bold=True)
    if cfg.scores is not None:
        click.secho(f"Scores: {_format_optional_list(cfg.scores)}", bold=True)

    click.secho("\nOptions:", bold=True)
    click.secho(f"  Time alignment: {cfg.project.time_alignment}")
    click.secho(f"  Unified mask: {cfg.project.unified_mask}")
    click.secho(f"  Comparison: {cfg.comparison.enabled}")
    click.secho(f"  Statistics: {cfg.statistics.enabled}")
    if comparison_only and cfg.project.only_drawing:
        click.secho(
            "  ✗ --comparison-only conflicts with project.only_drawing=true; choose one mode",
            fg="red",
        )
        has_errors = True
    if comparison_only:
        click.secho("  Check mode: comparison-only")
        if not cfg.comparison.enabled:
            click.secho("  ✗ comparison-only mode requires comparison.enabled: true", fg="red")
            has_errors = True
        else:
            from openbench.runner.local import comparison_only_preflight_errors

            for error in comparison_only_preflight_errors(cfg):
                click.secho(f"  ✗ {error.get('message', 'comparison-only preflight failed')}", fg="red")
                has_errors = True
    elif cfg.project.only_drawing:
        click.secho("  Check mode: only-drawing")
        from openbench.runner.local import existing_output_preflight_errors

        for error in existing_output_preflight_errors(cfg):
            click.secho(f"  ✗ {error.get('message', 'only-drawing preflight failed')}", fg="red")
            has_errors = True

    for message in _groupby_static_dataset_findings(cfg):
        click.secho(f"  ✗ {message}", fg="red")
        has_errors = True

    if has_errors:
        click.secho("\n✗ Config has errors. Please fix and re-check.", fg="red", bold=True)
        raise SystemExit(1)

    n_refs = len(resolved) if resolved else 0
    n_sims = len(cfg.simulation) if cfg.simulation else 0
    n_vars = len(cfg.evaluation.variables) if cfg.evaluation.variables else 0
    click.secho(
        f"\n✓ Config valid ({n_vars} variables, {n_refs} references, {n_sims} simulations). Ready to run.",
        fg="green",
        bold=True,
    )
