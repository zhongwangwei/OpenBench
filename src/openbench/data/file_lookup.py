"""Data-file naming rules shared by preprocessing and the pre-run checks.

Preprocessing finds grid data files from a source's ``prefix``, ``suffix`` and
``data_groupby``. ``openbench check`` and ``openbench init`` use the same
functions, so a file name that preprocessing would not find is reported before
an evaluation starts instead of failing in the middle of it.
"""

from __future__ import annotations

import fnmatch
import glob
import logging
import os
import re
from collections.abc import Iterable
from functools import lru_cache

from openbench.data.coordinates import NC_SUFFIXES, glob_nc_pattern

_DATED_DIRECTORY = re.compile(r"(\d{4})(?:[-_]\d{2}){0,2}")


def is_dated_directory(name: str) -> bool:
    """``YYYY``, ``YYYY-MM`` or ``YYYY_MM_DD`` with a plausible year (not e.g. ``1000`` or ``5000``)."""
    match = _DATED_DIRECTORY.fullmatch(name)
    return bool(match) and 1800 <= int(match.group(1)) <= 2300


def _has_configured_name(path: str, names: set[str]) -> bool:
    """Whether ``path`` is one of ``names`` under the filesystem's own case rules.

    The names were once probed with ``lexists``, so a case-insensitive
    filesystem (macOS, SMB mounts) matched ``rain.nc`` for ``RAIN`` and a
    case-sensitive one did not. ``normcase`` folds case only on Windows.
    """
    basename = os.path.basename(path)
    if basename in names:
        return True
    folded = basename.casefold()
    directory = os.path.dirname(path)
    for name in names:
        if name.casefold() == folded:
            try:
                return os.path.samefile(path, os.path.join(directory, name))
            except OSError:
                return False
    return False


def prefix_candidates(prefix: str, prefix_fallback: Iterable[str] | None) -> list[str]:
    """Return the primary prefix followed by its ``prefix_fallback`` variants."""
    prefixes = [prefix]
    if isinstance(prefix_fallback, str):  # a single fallback written without a list
        prefix_fallback = [prefix_fallback]
    for fallback in prefix_fallback or []:
        if prefix.endswith("_"):
            prefixes.append(prefix[:-1] + fallback)
        else:
            prefixes.append(prefix + fallback)
    return prefixes


def single_file_paths(dirx: str, prefix: str, suffix: str) -> list[str]:
    """Existing ``{prefix}{suffix}`` files for ``data_groupby: Single``, in extension order."""
    paths = []
    for ext in NC_SUFFIXES:
        path = os.path.join(dirx, f"{prefix}{suffix}{ext}")
        if os.path.exists(path):
            paths.append(path)
    return paths


def iter_netcdf_paths(dirx: str):
    """Walk supported files, following directory links without ancestor cycles.

    Hidden entries are skipped. Each directory is identified by ``(device,
    inode)`` once (one stat), and a link back to a directory on the current
    path is not entered; resolving every ancestor per directory instead cost
    hundreds of thousands of ``lstat`` calls on deep ``YYYY/MM/DD`` trees.
    """
    try:
        top = os.stat(dirx)
    except OSError:
        return
    stack = [(dirx, frozenset({(top.st_dev, top.st_ino)}))]
    while stack:
        current, ancestors = stack.pop()
        try:
            with os.scandir(current) as scanned:
                entries = list(scanned)
        except OSError:
            continue
        subdirectories = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                is_directory = entry.is_dir()
                identity = entry.stat() if is_directory else None
                if identity is not None and not identity.st_ino:
                    # Windows DirEntry.stat() leaves st_ino/st_dev at 0; os.stat() fills them.
                    identity = os.stat(entry.path)
            except OSError:
                continue
            if is_directory:
                key = (identity.st_dev, identity.st_ino)
                if key not in ancestors:
                    subdirectories.append((entry.path, ancestors | {key}))
            elif entry.name.endswith(NC_SUFFIXES):
                yield entry.path
        stack.extend(reversed(subdirectories))


def has_netcdf(directory, *, recursive: bool = False) -> bool:
    """True at the first NetCDF file; no full listing, sort or per-file stat."""
    if recursive:
        return next(iter_netcdf_paths(str(directory)), None) is not None
    try:
        with os.scandir(directory) as entries:
            return any(os.path.splitext(entry.name)[1] in NC_SUFFIXES and entry.is_file() for entry in entries)
    except OSError:
        return False


def netcdf_inventory(dirx: str) -> list[str]:
    """Read a tree once; callers may share this snapshot within one preflight.

    When ``dirx`` holds year folders with NetCDF files, the files directly in
    ``dirx`` and in those folders form the dataset. A sibling folder (such as
    another resolution) is a different branch: reading a year missing from the
    year folders out of it would mix branches, so it is left out.
    """
    paths = sorted(iter_netcdf_paths(dirx))
    prefix = os.path.join(dirx, "")  # walked paths start with it; relpath per file was slow
    tops = []
    for path in paths:
        rest = path[len(prefix) :] if path.startswith(prefix) else os.path.relpath(path, dirx)
        head, sep, _tail = rest.partition(os.sep)
        tops.append(head if sep else None)
    dated = {top: is_dated_directory(top) for top in set(tops) if top is not None}
    if not any(dated.values()):
        return paths
    return [path for path, top in zip(paths, tops) if top is None or dated[top]]


def unread_folders(dirx: str) -> list[str]:
    """Folders of ``dirx`` holding NetCDF files that its year-folder layout leaves unread."""
    try:
        with os.scandir(dirx) as entries:
            children = sorted(entry.name for entry in entries if entry.is_dir() and not entry.name.startswith("."))
    except OSError:
        return []
    dated = [child for child in children if is_dated_directory(child)]
    if not any(has_netcdf(os.path.join(dirx, child), recursive=True) for child in dated):
        return []
    return [child for child in children if child not in dated and has_netcdf(os.path.join(dirx, child), recursive=True)]


def year_file_paths(dirx: str, prefix: str, year: int, suffix: str, *, inventory: list[str] | None = None) -> list[str]:
    """Find yearly files in runtime priority order, then nested date directories.

    Keep the cheap direct-directory lookup for runtime reads. Recursive misses
    use one inventory instead of repeating a tree walk for every extension.
    """
    from fnmatch import fnmatch

    name_pattern = f"{glob.escape(prefix)}{year}*{glob.escape(suffix)}"
    var_files = []
    for directory in (dirx, os.path.join(dirx, str(year))):
        if inventory is None:
            var_files = glob_nc_pattern(os.path.join(directory, name_pattern + ".nc"))
        else:
            var_files = [
                path
                for path in inventory
                if os.path.normcase(os.path.normpath(os.path.dirname(path)))
                == os.path.normcase(os.path.normpath(directory))
                and any(fnmatch(os.path.basename(path), name_pattern + ext) for ext in NC_SUFFIXES)
            ]
        if var_files:
            break
    if not var_files:
        if inventory is None:
            inventory = netcdf_inventory(dirx)
        var_files = [
            path
            for path in inventory
            if any(fnmatch(os.path.basename(path), name_pattern + ext) for ext in NC_SUFFIXES)
        ]
    if var_files:
        pattern = re.compile(rf"^{re.escape(prefix)}{year}[^a-zA-Z]*{re.escape(suffix)}\.nc4?$", re.IGNORECASE)
        var_files = [path for path in var_files if pattern.match(os.path.basename(path))]
    if not var_files and (prefix or suffix):
        if inventory is None:
            inventory = netcdf_inventory(dirx)
        year_dir = re.compile(rf"^{year}(?:[-_](?:0[1-9]|1[0-2]))?(?:[-_](?:0[1-9]|[12]\d|3[01]))?$")
        names = {f"{prefix}{suffix}{ext}" for ext in NC_SUFFIXES}
        var_files = [
            path
            for path in inventory
            if any(year_dir.fullmatch(part) for part in os.path.relpath(path, dirx).split(os.sep)[:-1])
            and _has_configured_name(path, names)
        ]
    return var_files


@lru_cache(maxsize=4096)
def _variable_names(path: str, _mtime_ns: int, _size: int) -> frozenset[str] | None:
    import xarray as xr

    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            return frozenset(str(name).lower() for name in ds.variables)
    except Exception as exc:
        # Cached too: an unreadable file is otherwise reopened for every year and variable.
        logging.debug("Could not read the variables of %s: %s", path, exc)
        return None


def variable_names(path: str) -> frozenset[str] | None:
    """Lower-cased variable names in a NetCDF header, or ``None`` when it cannot be read.

    Cached per file version (path, modification time and size).
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return _variable_names(path, stat.st_mtime_ns, stat.st_size)


def compute_input_files(
    dirx: str, year: int | None, dependencies: Iterable[str], *, inventory: list[str] | None = None
) -> list[str]:
    """Files holding the inputs of a compute expression, as preprocessing selects them.

    Candidates are the NetCDF files of the lookup's branch whose name contains
    ``year`` (any file for Single data). Returns ``[]`` unless every input is
    found in one of them.
    """
    dependencies = list(dict.fromkeys(dependencies))
    if not dependencies:
        return []
    if inventory is None:
        inventory = netcdf_inventory(dirx)
    selected, found = [], set()
    for path in inventory:
        if year is not None and not fnmatch.fnmatchcase(os.path.basename(path), f"*{year}*"):
            continue
        names = variable_names(path)
        if names is None:
            continue
        hits = [name for name in dependencies if name.lower() in names]
        if hits:
            selected.append(path)
            found.update(hits)
    return selected if set(dependencies) <= found else []


def select_data_files(
    dirx: str,
    prefix: str,
    suffix: str,
    year: int | None,
    *,
    prefix_fallback: Iterable[str] | None = None,
    candidate_varnames: Iterable[str] = (),
    dependencies: Iterable[str] = (),
    inventory: list[str] | None = None,
    named_matches: dict[str, list[str]] | None = None,
) -> tuple[list[str], bool]:
    """Select named files or compute inputs using preprocessing's lookup order.

    ``named_matches`` optionally supplies the preflight's already indexed
    matches for this year, keyed by prefix. The boolean marks compute inputs.
    Unreadable headers remain permissive, leaving data errors to preprocessing.
    """
    candidates = {str(name).lower() for name in candidate_varnames if name}
    dependencies = list(dependencies)
    prefixes = prefix_candidates(prefix, prefix_fallback)
    if candidates and len(prefixes) > 1:
        prefixes = [*prefixes[1:], prefixes[0]]

    def groups_of(candidate: str) -> list[list[str]]:
        if year is None:
            return [[path] for path in single_file_paths(dirx, candidate, suffix)]
        if named_matches is not None and candidate in named_matches:
            return [named_matches[candidate]]
        return [year_file_paths(dirx, candidate, year, suffix, inventory=inventory)]

    if not dependencies:
        groups = [files for candidate in prefixes for files in groups_of(candidate) if files]
        if len(groups) <= 1:
            # One candidate group is used whatever its variables (there is no
            # compute input to prefer), so its headers need not be opened.
            return (groups[0] if groups else []), False
        grouped = [groups]
    else:
        grouped = (groups_of(candidate) for candidate in prefixes)
    first_matches = []
    for groups in grouped:
        for files in groups:
            if not files:
                continue
            if not first_matches:
                first_matches = files
            if not candidates:
                return files, False
            inspected = False
            for path in files[:3]:
                names = variable_names(path)
                if names is None:
                    continue
                inspected = True
                if candidates & names:
                    return files, False
            if not inspected:
                return files, False
    compute_files = compute_input_files(dirx, year, dependencies, inventory=inventory)
    return (compute_files, True) if compute_files else (first_matches, False)


def year_matches(
    dirx: str,
    prefix: str,
    suffix: str,
    years: Iterable[int],
    prefix_fallback: Iterable[str] | None = None,
    *,
    inventory: list[str] | None = None,
) -> dict[int, list[str]]:
    """The files preprocessing reads for each year: those of the first prefix that matches, or ``[]``."""
    prefix = prefix or ""
    suffix = suffix or ""
    prefixes = prefix_candidates(prefix, prefix_fallback)
    if inventory is None:
        inventory = netcdf_inventory(dirx)
    requested_years = list(dict.fromkeys(years))
    # Index the relevant filenames once. Rechecking a large inventory for each
    # missing year was expensive even after the filesystem walk was shared.
    year_texts = {str(year): year for year in requested_years}
    dated_directory = re.compile(r"^(\d+)(?:[-_](?:0[1-9]|1[0-2]))?(?:[-_](?:0[1-9]|[12]\d|3[01]))?$")
    inventories = {}
    for candidate in prefixes:
        normalized_prefix = os.path.normcase(candidate)
        normalized_suffix = os.path.normcase(suffix)
        exact_names = {f"{candidate}{suffix}{ext}" for ext in NC_SUFFIXES}
        exact_folded = {name.casefold() for name in exact_names}
        by_year = {year: [] for year in requested_years}
        for path in inventory:
            basename = os.path.basename(path)
            stem = os.path.splitext(os.path.normcase(basename))[0]
            matches = set()
            if stem.startswith(normalized_prefix) and stem.endswith(normalized_suffix):
                tail = stem[len(normalized_prefix) :]
                matches = {year for text, year in year_texts.items() if tail.startswith(text)}
            if (
                (candidate or suffix)
                and basename.casefold() in exact_folded
                and _has_configured_name(path, exact_names)
            ):
                for part in os.path.relpath(path, dirx).split(os.sep)[:-1]:
                    match = dated_directory.fullmatch(part)
                    if match and match.group(1) in year_texts:
                        matches.add(year_texts[match.group(1)])
            for year in matches:
                by_year[year].append(path)
        inventories[candidate] = by_year
    found = {}
    for year in requested_years:
        found[year] = []
        for candidate in prefixes:
            files = year_file_paths(dirx, candidate, year, suffix, inventory=inventories[candidate][year])
            if files:
                found[year] = files
                break
    return found


def missing_data_files(
    dirx: str,
    prefix: str,
    suffix: str,
    data_groupby: str,
    years: Iterable[int],
    prefix_fallback: Iterable[str] | None = None,
    *,
    inventory: list[str] | None = None,
    matches: dict[int, list[str]] | None = None,
) -> list[str]:
    """Describe each file preprocessing would look for and not find.

    ``years`` are the years to probe for groupings other than Single; Single
    needs its one file regardless. Returns an empty list when every probe
    finds a file under the primary prefix or one of its fallbacks.
    ``matches`` reuses an earlier :func:`year_matches` result.
    """
    prefix = prefix or ""
    suffix = suffix or ""
    if str(data_groupby or "").strip().lower() == "single":
        if any(single_file_paths(dirx, candidate, suffix) for candidate in prefix_candidates(prefix, prefix_fallback)):
            return []
        return [os.path.join(dirx, f"{prefix}{suffix}.nc")]
    if matches is None:
        matches = year_matches(dirx, prefix, suffix, years, prefix_fallback, inventory=inventory)
    return [os.path.join(dirx, f"{prefix}{year}*{suffix}.nc") for year, files in matches.items() if not files]


def mixed_branches(
    dirx: str,
    matches: dict[int, list[str]],
    *,
    dependencies: Iterable[str] | None = None,
) -> dict[int, tuple[tuple[str, ...], bool]]:
    """Years whose files come from several folders of ``dirx``, and whether a file name repeats.

    Files directly in ``dirx`` and year folders form one layout; any other
    top-level folder, such as another resolution, is a branch of its own. The
    recursive lookup reads the files of all branches of a year together.
    For compute inputs, only branches providing an overlapping dependency
    conflict; complementary inputs may legitimately use the same filename.
    """
    prefix = os.path.join(dirx, "")
    dated: dict[str, bool] = {}

    def branch_of(path: str) -> str:
        rest = path[len(prefix) :] if path.startswith(prefix) else os.path.relpath(path, dirx)
        head, sep, _tail = rest.partition(os.sep)
        if not sep:
            return "."
        if head not in dated:
            dated[head] = is_dated_directory(head)
        return "." if dated[head] else head

    dependency_names = None if dependencies is None else {str(name).lower() for name in dependencies}
    mixed = {}
    for year, files in matches.items():
        file_branches = [branch_of(path) for path in files]
        if len(set(file_branches)) < 2:
            continue
        branches: dict[str, dict[str, set[str]]] = {}
        for path, branch in zip(files, file_branches):
            inputs = {"*"}
            if dependency_names is not None:
                names = variable_names(path)
                # An unreadable header may hold any input, so it may overlap.
                inputs = dependency_names if names is None else dependency_names & names
            for name in inputs:
                branches.setdefault(branch, {}).setdefault(name, set()).add(os.path.normcase(os.path.basename(path)))
        conflicting = set()
        repeated = False
        items = list(branches.items())
        for index, (left, left_inputs) in enumerate(items):
            for right, right_inputs in items[index + 1 :]:
                overlap = left_inputs.keys() & right_inputs.keys()
                if overlap:
                    conflicting.update((left, right))
                    repeated = repeated or any(left_inputs[name] & right_inputs[name] for name in overlap)
        if conflicting:
            mixed[year] = (tuple(sorted(conflicting)), repeated)
    return mixed
