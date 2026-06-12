"""Name-resolution helpers for case-tolerant user-facing identifiers.

OpenBench has two broad classes of names:

* OpenBench logical names (model profiles, reference datasets, standard
  variables) should be user friendly and case-insensitive.
* External data identifiers (NetCDF variables, CSV columns) should preserve
  their spelling, but runtime lookup can still use an exact-first
  case-insensitive fallback when that fallback is unique.

These helpers implement the shared exact-first/unique-casefold semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar
import unicodedata

T = TypeVar("T")


class NameResolutionError(ValueError):
    """Raised when a name is missing or ambiguous under case-insensitive lookup."""


class AmbiguousNameError(NameResolutionError):
    """Raised when multiple candidates match a requested name ignoring case."""


def normalize_name(value: object) -> str:
    """Return a Unicode-aware normalized key for user-facing name matching."""

    return unicodedata.normalize("NFC", str(value).strip()).casefold()


def find_name_case_insensitive(requested: object, candidates: Iterable[object]) -> object | None:
    """Find *requested* in *candidates* using exact-first, unique casefold fallback.

    Returns the canonical candidate object when found, ``None`` when missing,
    and raises :class:`AmbiguousNameError` when more than one candidate matches
    ignoring case.
    """

    candidate_list = list(candidates)
    requested_text = unicodedata.normalize("NFC", str(requested).strip())
    for candidate in candidate_list:
        if unicodedata.normalize("NFC", str(candidate).strip()) == requested_text:
            return candidate

    requested_key = normalize_name(requested)
    matches = [candidate for candidate in candidate_list if normalize_name(candidate) == requested_key]
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) > 1:
        raise AmbiguousNameError(
            f"name '{requested}' is ambiguous ignoring case: " + ", ".join(str(match) for match in unique_matches)
        )
    return unique_matches[0] if unique_matches else None


def resolve_name_case_insensitive(
    requested: object,
    candidates: Iterable[object],
    *,
    label: str = "name",
) -> object:
    """Resolve *requested* to a canonical candidate or raise a detailed error."""

    match = find_name_case_insensitive(requested, candidates)
    if match is None:
        raise NameResolutionError(f"{label} not found: {requested}")
    return match


def get_mapping_key_case_insensitive(mapping: Mapping[object, T], requested: object) -> object | None:
    """Return the canonical key in *mapping* matching *requested*.

    Exact keys win.  A missing key returns ``None``; an ambiguous
    case-insensitive match raises.
    """

    return find_name_case_insensitive(requested, mapping.keys())


def get_mapping_value_case_insensitive(mapping: Mapping[object, T], requested: object) -> T | None:
    """Return a mapping value using exact-first case-insensitive key lookup."""

    key = get_mapping_key_case_insensitive(mapping, requested)
    if key is None:
        return None
    return mapping[key]


def get_xarray_key_case_insensitive(
    obj: object,
    requested: object,
    *,
    include_coords: bool = True,
) -> str | None:
    """Resolve a variable/coordinate key on an xarray-like object.

    Exact spelling wins.  If exact lookup fails, a unique case-insensitive
    match among data variables and, optionally, coordinates is returned.
    """

    data_var_candidates: list[object] = []
    coord_candidates: list[object] = []
    variable_candidates: list[object] = []
    if hasattr(obj, "data_vars"):
        data_var_candidates.extend(getattr(obj, "data_vars").keys())
    if include_coords and hasattr(obj, "coords"):
        coord_candidates.extend(getattr(obj, "coords").keys())
    if not data_var_candidates and not coord_candidates and hasattr(obj, "variables"):
        variables = getattr(obj, "variables")
        if hasattr(variables, "keys"):
            variable_candidates.extend(variables.keys())

    # Prefer data variables over coordinates for ambiguous case-insensitive
    # matches across namespaces, but still detect ambiguity within each
    # namespace.
    for candidates in (data_var_candidates, coord_candidates, variable_candidates):
        match = find_name_case_insensitive(requested, candidates)
        if match is not None:
            return str(match)
    return None


def select_data_array(ds, *preferred):
    """Pull one data variable out of a (flat/preprocessed) dataset robustly.

    Tries each name in ``preferred`` (case-insensitive). If none match, falls
    back to the single data variable when the dataset has exactly one — which is
    always the case for OpenBench flat ``<item>_<src>_<varname>.nc`` files.

    This decouples the *configured* variable name (e.g. ``f_respc``) from the
    name the data is actually stored under (e.g. the relabelled evaluation item
    ``Net_Ecosystem_Exchange``, produced when a fallback/convert derives the
    variable). Readers must not hard-index ``ds[sim_varname]`` because the saved
    variable is intentionally relabelled to the item for output correctness.

    Raises KeyError if nothing matches and the dataset is not single-variable.
    Each ``preferred`` arg may be a name or a list/tuple of names.
    """
    names: list = []
    for item in preferred:
        if isinstance(item, (list, tuple)):
            names.extend(item)
        elif item:
            names.append(item)
    for name in names:
        if not name:
            continue
        actual = get_xarray_key_case_insensitive(ds, name)
        if actual is not None:
            return ds[actual]
    data_vars = list(getattr(ds, "data_vars", []))
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    raise KeyError(
        f"None of {[n for n in preferred if n]} found in dataset; it has "
        f"{len(data_vars)} data variables {data_vars}, cannot disambiguate."
    )


def resolve_many_case_insensitive(
    requested: Iterable[object],
    candidates: Iterable[object],
    *,
    label: str = "name",
) -> list[object]:
    """Resolve multiple names and reject duplicate requests ignoring case."""

    requested_list = list(requested)
    resolved = [resolve_name_case_insensitive(item, candidates, label=label) for item in requested_list]
    grouped: dict[str, tuple[object, list[object]]] = {}
    for original, canonical in zip(requested_list, resolved, strict=True):
        key = normalize_name(canonical)
        _, originals = grouped.setdefault(key, (canonical, []))
        originals.append(original)
    duplicates = [(canonical, originals) for canonical, originals in grouped.values() if len(originals) > 1]
    if duplicates:
        details = []
        for canonical, originals in duplicates:
            raw_inputs = ", ".join(str(item) for item in originals)
            details.append(f"{raw_inputs} -> {canonical}")
        raise NameResolutionError(f"duplicate {label} values ignoring case: " + "; ".join(details))
    return resolved
