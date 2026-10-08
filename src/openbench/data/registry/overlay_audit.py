"""Audit and re-sparse the user registry overlay against the bundled catalog.

The user overlay (``~/.openbench/references/reference_catalog.yaml`` and
``~/.openbench/models/model_catalog.yaml``) is meant to hold only *sparse
deltas* on top of the bundled catalog. Older OpenBench versions sometimes wrote
the entire merged catalog back as a full snapshot. A full-snapshot overlay
silently shadows the bundled catalog field-by-field, freezing the user on a
stale version so bundled fixes never take effect.

This module classifies each overlay entry, can re-sparse the overlay
(behavior-preserving), and powers a throttled startup notice. It also keeps the
overlay sparse on every write (``sparsify_overlay_catalog``) and, once per
overlay file, removes redundant current defaults (``sync_legacy_overlays``).
Historical values are preserved because their provenance is unknown.

Behavior-preservation note: the manager's deep merge (``_deep_merge_reference`` /
``_deep_merge_model``) overrides per-field — including per field inside a
variable — and never deletes a bundled variable just because the overlay omits
it (only an explicit ``None`` tombstone deletes). So the minimal
behavior-preserving overlay is computed by ``_sparse_delta`` below — which keeps
only fields whose removal would change the merged entry and adds NO tombstones.
(``scanner._descriptor_overlay_diff`` deliberately differs: it tombstones
omitted variables, which is correct when writing a complete descriptor but would
change behavior if applied to an existing snapshot.)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import click
import yaml

from openbench.data.registry import bundled_history
from openbench.util.names import canonical_variable_name, get_mapping_key_case_insensitive, normalize_name

logger = logging.getLogger(__name__)

# Classification kinds
REDUNDANT = "redundant"  # overlay entry byte-equals bundled (pure snapshot leftover)
STALE_FULLCOPY = "stale_fullcopy"  # carries bundled-equal baggage AND differs -> stale snapshot
DELTA = "delta"  # already-minimal override; provenance is unknown
CUSTOM = "custom"  # entry not present in bundled (genuine user addition)
DUPLICATE = "duplicate"  # one of several keys differing only in case; later ones win at load
_KIND_NOUN = {"references": "reference", "models": "model"}

_STATE_FILENAME = ".registry_check"
_SUPPRESS_ENV = "OPENBENCH_NO_REGISTRY_CHECK"


def _raw_variable(mapping: dict, name: str) -> Any:
    """Look up raw variable fields using the registry's canonical aliases."""
    from openbench.data.registry.manager import _is_empty_value

    key = normalize_name(canonical_variable_name(name))
    matches = [value for raw_name, value in mapping.items() if normalize_name(canonical_variable_name(raw_name)) == key]
    return next((value for value in matches if not _is_empty_value(value)), matches[0] if matches else None)


def _duplicate_names(catalog: dict) -> set[str]:
    return {name for name, count in Counter(normalize_name(key) for key in catalog).items() if count > 1}


def _raw_delta(bundled_entry: dict, overlay_entry: dict) -> dict:
    """Drop overlay fields that repeat the bundled entry verbatim (variables per field)."""
    out: dict = {}
    for key, value in (overlay_entry or {}).items():
        if key == "variables" and isinstance(value, dict):
            bundled_vars = bundled_entry.get("variables", {}) or {}
            var_out = {}
            for var_name, var_data in value.items():
                bundled_var = _raw_variable(bundled_vars, var_name)
                if bundled_var == var_data:
                    continue
                if isinstance(var_data, dict) and isinstance(bundled_var, dict):
                    changed = {f: v for f, v in var_data.items() if f not in bundled_var or bundled_var[f] != v}
                    if changed:
                        var_out[var_name] = changed
                else:
                    var_out[var_name] = var_data
            if var_out:
                out["variables"] = var_out
        elif bundled_entry.get(key) != value:
            out[key] = value
    return out


def _bundled_object(kind: str, name: str, bundled_entry: dict):
    """Build the registry object for a raw bundled catalog entry."""
    from openbench.data.registry import manager as registry_manager

    data = dict(bundled_entry)
    data.setdefault("name", name)
    with registry_manager.quiet_unresolved_env():
        if kind == "models":
            return registry_manager._build_model(data)
        return registry_manager._build_reference(data)


def _merged(kind: str, base_obj, overlay: dict):
    from openbench.data.registry import manager as registry_manager

    merge = registry_manager._deep_merge_model if kind == "models" else registry_manager._deep_merge_reference
    with registry_manager.quiet_unresolved_env():
        return merge(base_obj, overlay)


def _same_behavior(kind: str, base_obj, left: dict, right: dict) -> bool:
    """Return True when both overlays merge onto ``base_obj`` identically."""
    try:
        return _merged(kind, base_obj, left) == _merged(kind, base_obj, right)
    except Exception:
        return False


def _bundled_has_variable(base_obj, var_name: str) -> bool:
    try:
        return get_mapping_key_case_insensitive(base_obj.variables, canonical_variable_name(var_name)) is not None
    except Exception:
        return False


# Fields the registry merges. Anything else in an overlay entry (a user's
# ``notes``, keys from other tools) has no effect on the registry, so removing
# it can never "change behavior" — but it is the user's, so keep it.
_MERGED_FIELDS = {
    "references": frozenset(
        {
            "name",
            "description",
            "category",
            "data_type",
            "tim_res",
            "data_groupby",
            "timezone",
            "years",
            "grid_res",
            "fulllist",
            "root_dir",
            "station_matching",
            "_provenance",
            "variables",
            "_deleted",
        }
    ),
    "models": frozenset(
        {
            "name",
            "description",
            "data_type",
            "grid_res",
            "tim_res",
            "variables",
            "time_offset",
            "_delete_variables",
            "_deleted",
        }
    ),
}
_MERGED_VARIABLE_FIELDS = {
    "references": frozenset(
        {
            "varname",
            "varunit",
            "prefix",
            "suffix",
            "sub_dir",
            "fulllist",
            "max_uparea",
            "min_uparea",
            "fallbacks",
            "compute",
            "accumulated",
            "prefix_fallback",
        }
    ),
    "models": frozenset(
        {"varname", "varunit", "prefix", "suffix", "sub_dir", "fallbacks", "compute", "accumulated", "prefix_fallback"}
    ),
}


def _sparse_delta(bundled_entry: dict, overlay_entry: dict, *, kind: str = "references", name: str = "") -> dict:
    """Minimal, behavior-preserving overlay for ``overlay_entry`` over bundled.

    Drops every field — top-level, or a single field inside a bundled
    variable — whose removal leaves the merged entry unchanged, so values that
    merely repeat the bundled catalog never shadow later bundled fixes. Adds NO
    tombstones (matches deep-merge semantics). A name repeating the catalog key
    is dropped; a different descriptor name is preserved.
    """
    bundled_entry = bundled_entry or {}
    original = _strip_name(overlay_entry, name or str(bundled_entry.get("name") or ""))
    try:
        base_obj = _bundled_object(kind, name or str(bundled_entry.get("name") or ""), bundled_entry)
    except Exception:
        # Without a buildable bundled entry, fall back to verbatim comparison.
        return _raw_delta(bundled_entry, original)

    candidate = _raw_delta(bundled_entry, original)
    if not _same_behavior(kind, base_obj, candidate, original):
        candidate = copy.deepcopy(original)

    # Expanded equality cannot prove that a literal path tracks an environment
    # expression. Only identical raw paths may be removed.
    for key in ("root_dir", "fulllist"):
        if key in candidate and key in bundled_entry and candidate[key] == bundled_entry[key]:
            candidate.pop(key)

    for key in list(candidate):
        if key in bundled_history.CONTROL_KEYS or key in {"root_dir", "fulllist"} or key not in _MERGED_FIELDS[kind]:
            continue
        trial = {k: v for k, v in candidate.items() if k != key}
        if _same_behavior(kind, base_obj, trial, candidate):
            candidate = trial

    variables = candidate.get("variables")
    if isinstance(variables, dict):
        for var_name in list(variables):
            var_data = variables[var_name]
            if not isinstance(var_data, dict) or not _bundled_has_variable(base_obj, var_name):
                continue
            for var_field in list(var_data):
                if var_field not in _MERGED_VARIABLE_FIELDS[kind]:
                    continue
                trial = copy.deepcopy(candidate)
                trial_var = trial["variables"][var_name]
                trial_var.pop(var_field)
                if not trial_var:
                    trial["variables"].pop(var_name)
                if not trial["variables"]:
                    trial.pop("variables")
                if _same_behavior(kind, base_obj, trial, candidate):
                    candidate = trial
                    if var_name not in (candidate.get("variables") or {}):
                        break
    offsets = candidate.get("time_offset")
    if isinstance(offsets, dict):
        for resolution in list(offsets):
            group = candidate["time_offset"][resolution]
            fields = list(group) if isinstance(group, dict) else [None]
            for offset_field in fields:
                trial = copy.deepcopy(candidate)
                if offset_field is None:
                    trial["time_offset"].pop(resolution)
                else:
                    trial["time_offset"][resolution].pop(offset_field)
                    if not trial["time_offset"][resolution]:
                        trial["time_offset"].pop(resolution)
                if not trial["time_offset"]:
                    trial.pop("time_offset")
                if _same_behavior(kind, base_obj, trial, candidate):
                    candidate = trial
    return candidate


def _strip_name(entry: dict, name: str = "") -> dict:
    return {k: v for k, v in (entry or {}).items() if k != "name" or (name and v != name)}


def _classify_entry(
    bundled_entry: Optional[dict], overlay_entry: dict, *, kind: str = "references", name: str = ""
) -> tuple[str, dict]:
    """Return (kind, minimal_delta) for one overlay entry."""
    if bundled_entry is None:
        return CUSTOM, copy.deepcopy(overlay_entry)
    delta = _sparse_delta(bundled_entry, overlay_entry, kind=kind, name=name)
    if not delta:
        return REDUNDANT, {}
    if _strip_name(overlay_entry, name) == delta:
        return DELTA, delta
    return STALE_FULLCOPY, delta


@dataclass
class EntryAudit:
    name: str
    kind: str
    minimal: dict = field(default_factory=dict)
    historical_fields: list[str] = field(default_factory=list)
    path: Optional[Path] = None  # a separate ``<kind>/*.yaml`` file instead of the catalog


@dataclass
class CatalogAudit:
    label: str  # "references" | "models"
    overlay_path: Path
    bundled_path: Path
    exists: bool
    entries: list[EntryAudit] = field(default_factory=list)
    error: Optional[str] = None  # the overlay could not be read as a mapping of entries
    # Entries in separate ``<kind>/*.yaml`` files, merged after the catalog.
    # Neither prune nor reset edits those files, so they are reported apart.
    per_file: list[EntryAudit] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[EntryAudit]:
        return [e for e in self.entries if e.kind == kind]

    @property
    def bloat(self) -> list[EntryAudit]:
        """Entries that silently shadow bundled and are freeze/staleness risks."""
        return [e for e in self.entries if e.kind in (REDUNDANT, STALE_FULLCOPY)]

    @property
    def duplicates(self) -> list[EntryAudit]:
        return self.by_kind(DUPLICATE)

    @property
    def per_file_attention(self) -> list[EntryAudit]:
        """Separate-file entries that hold older defaults or copy the bundled entry."""
        return [e for e in self.per_file if e.historical_fields or e.kind in (REDUNDANT, STALE_FULLCOPY)]


@dataclass
class OverlayAudit:
    references: CatalogAudit
    models: CatalogAudit

    @property
    def catalogs(self) -> list[CatalogAudit]:
        return [self.references, self.models]

    @property
    def bloat_count(self) -> int:
        return sum(len(c.bloat) for c in self.catalogs)

    @property
    def stale_count(self) -> int:
        return sum(len(c.by_kind(STALE_FULLCOPY)) for c in self.catalogs)

    @property
    def duplicate_count(self) -> int:
        return sum(len(c.duplicates) for c in self.catalogs)


def _load_yaml(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # pragma: no cover - corrupted file is surfaced elsewhere
        logger.debug("overlay_audit: could not read %s: %s", path, e)
        return {}


def _bundled_lookup(bundled: dict) -> dict[str, tuple[str, dict]]:
    """Map normalized entry name -> (bundled name, bundled entry)."""
    return {normalize_name(k): (k, v) for k, v in bundled.items() if isinstance(v, dict)}


def _historical_fields(label: str, name: str, bundled: dict, overlay: dict) -> list[str]:
    """Flag older shipped values without guessing whether the user chose them."""
    history = bundled_history.load_history().get(label, frozenset())
    fields = []
    for key, value in overlay.items():
        if key in bundled_history.CONTROL_KEYS or bundled.get(key) == value:
            continue
        if bundled_history.field_digest(label, name, key, value) in history:
            fields.append(key)
    for variable, mapping in (overlay.get("variables") or {}).items():
        if not isinstance(mapping, dict):
            continue
        current = _raw_variable(bundled.get("variables") or {}, variable) or {}
        for key, value in mapping.items():
            if current.get(key) != value and any(
                digest in history for digest in bundled_history.var_field_digests(label, name, variable, key, value)
            ):
                fields.append(f"{variable}.{key}")
    return fields


def _read_overlay(path: Path) -> tuple[dict, Optional[str]]:
    """The overlay's entries, or ``({}, reason)`` when it is unreadable or not a mapping."""
    if not path.exists():
        return {}, None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return {}, f"could not be read: {e}"
    if data is None:
        return {}, None
    if not isinstance(data, dict):
        return {}, f"is not a mapping of entries (found {type(data).__name__})"
    return data, None


def _audit_entry(label: str, bundled_ci: dict, name: str, data: dict, path: Optional[Path] = None) -> EntryAudit:
    if data.get("_deleted"):
        # Tombstones are intentional deletions — treat as deliberate.
        return EntryAudit(name=name, kind=DELTA, minimal=data, path=path)
    bundled_name, bundled_entry = bundled_ci.get(normalize_name(name), (name, None))
    kind, minimal = _classify_entry(bundled_entry, data, kind=label, name=bundled_name)
    historical = _historical_fields(label, bundled_name, bundled_entry, minimal) if bundled_entry else []
    return EntryAudit(name=name, kind=kind, minimal=minimal, historical_fields=historical, path=path)


def _audit_catalog(label: str, bundled_path: Path, overlay_path: Path) -> CatalogAudit:
    bundled = _load_yaml(bundled_path)
    overlay, error = _read_overlay(overlay_path)
    bundled_ci = _bundled_lookup(bundled)
    duplicates = _duplicate_names(overlay)
    entries: list[EntryAudit] = []
    for name, data in overlay.items():
        if not isinstance(data, dict):
            continue
        if normalize_name(name) in duplicates and not data.get("_deleted"):
            # Case variants merge in order at load; judge the group, not one key's values.
            entries.append(EntryAudit(name=name, kind=DUPLICATE, minimal=copy.deepcopy(data)))
            continue
        entries.append(_audit_entry(label, bundled_ci, name, data))
    per_file = [
        _audit_entry(label, bundled_ci, name, data, path)
        for name, data, path in _per_file_entries(overlay_path.parent, label)
    ]
    return CatalogAudit(
        label=label,
        overlay_path=overlay_path,
        bundled_path=bundled_path,
        exists=overlay_path.exists(),
        entries=entries,
        error=error,
        per_file=per_file,
    )


def _paths(user_dir: Optional[Path] = None) -> tuple[Path, Path, Path, Path, Path]:
    """Return (user_dir, bundled_ref, overlay_ref, bundled_model, overlay_model)."""
    from openbench.config.user_settings import get_user_config_dir
    from openbench.data.registry.manager import REGISTRY_DIR

    base = Path(get_user_config_dir(user_dir))
    return (
        base,
        REGISTRY_DIR / "reference_catalog.yaml",
        base / "references" / "reference_catalog.yaml",
        REGISTRY_DIR / "model_catalog.yaml",
        base / "models" / "model_catalog.yaml",
    )


_RESERVED_PER_FILE = {
    "references": frozenset({"reference_catalog.yaml", "reference_profiles.yaml"}),
    "models": frozenset({"model_catalog.yaml", "aliases.yaml"}),
}


def _per_file_paths(directory: Path, kind: str) -> list[Path]:
    try:
        if not directory.is_dir():
            return []
        return [path for path in sorted(directory.glob("*.yaml")) if path.name not in _RESERVED_PER_FILE[kind]]
    except OSError:
        return []


def _per_file_entries(directory: Path, kind: str) -> list[tuple[str, dict, Path]]:
    """``(entry name, entry, file)`` for entries in ``<directory>/*.yaml`` outside the catalog.

    The registry merges these after the catalog (same parsing as the manager:
    a mapping with ``name`` is one entry, otherwise a mapping of entries).
    """
    found = []
    for path in _per_file_paths(directory, kind):
        data = _load_yaml(path)
        if not isinstance(data, dict) or not data:
            continue
        entries = {data["name"]: data} if "name" in data else data
        found.extend((str(name), entry, path) for name, entry in entries.items() if isinstance(entry, dict))
    return found


def per_file_overrides(kind: str, user_dir: Optional[Path] = None) -> list[tuple[str, Path]]:
    """``(entry name, file)`` for entries in ``~/.openbench/<kind>/*.yaml`` outside the catalog."""
    return [(name, path) for name, _entry, path in _per_file_entries(Path(_paths(user_dir)[0]) / kind, kind)]


def previously_bundled(kind: str, name: str, entry: dict) -> bool:
    """True when an entry absent from the bundled catalog was shipped by an older one."""
    history = bundled_history.load_history().get(kind, frozenset())
    return any(digest in history for digest in bundled_history.iter_entry_digests(kind, name, entry))


def audit_overlays(user_dir: Optional[Path] = None) -> OverlayAudit:
    """Classify every entry in the user reference + model overlays."""
    _base, bundled_ref, overlay_ref, bundled_model, overlay_model = _paths(user_dir)
    return OverlayAudit(
        references=_audit_catalog("references", bundled_ref, overlay_ref),
        models=_audit_catalog("models", bundled_model, overlay_model),
    )


@dataclass
class PruneResult:
    label: str
    removed: list[str] = field(default_factory=list)  # redundant entries dropped
    minimized: list[str] = field(default_factory=list)  # stale full-copies reduced to deltas
    kept: list[str] = field(default_factory=list)  # delta/custom left as-is
    backup: Optional[Path] = None
    wrote: bool = False


def _prune_catalog(audit: CatalogAudit, *, dry_run: bool) -> PruneResult:
    result = PruneResult(label=audit.label)
    overlay_raw = _load_yaml(audit.overlay_path)
    new_overlay: dict = copy.deepcopy(overlay_raw)
    for entry in audit.entries:
        original = overlay_raw.get(entry.name)
        if entry.kind == REDUNDANT:
            result.removed.append(entry.name)
            new_overlay.pop(entry.name, None)
            continue
        if entry.kind == STALE_FULLCOPY:
            new_overlay[entry.name] = entry.minimal
            result.minimized.append(entry.name)
            continue
        # DELTA / CUSTOM / DUPLICATE: keep exactly as written
        new_overlay[entry.name] = original
        result.kept.append(entry.name)

    changed = result.removed or result.minimized
    if changed and not dry_run:
        from openbench.data.registry.scanner import _backup_then_write, _invalidate_registry_caches

        audit.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        result.backup = _backup_then_write(audit.overlay_path, new_overlay, separate_files="ignore")
        result.wrote = True
        _invalidate_registry_caches()
    return result


def prune_overlays(user_dir: Optional[Path] = None, *, dry_run: bool = False) -> list[PruneResult]:
    """Re-sparse the overlays: drop redundant entries, minimize stale full-copies.

    Behavior-preserving — the merged registry is identical before and after.
    """
    audit = audit_overlays(user_dir)
    return [_prune_catalog(c, dry_run=dry_run) for c in audit.catalogs]


# --------------------------------------------------------------------------- #
# Sparse writes and one-time sync of legacy snapshots
# --------------------------------------------------------------------------- #

_SYNC_MARKER_SUFFIX = ".bundled-sync"
_SYNC_MARKER_FORMAT = 1


class SeparateFileOverride(click.ClickException, ValueError):
    """A catalog write whose change an entry in a separate overlay file would undo."""


_WHOLE_ENTRY = "(the whole entry)"
_UNLOADABLE = object()  # a separate-file entry the registry could not build


def _built(kind: str, name: str, data: dict):
    from openbench.data.registry import manager as registry_manager

    build = registry_manager._build_model if kind == "models" else registry_manager._build_reference
    with registry_manager.quiet_unresolved_env():
        return build({**data, "name": data.get("name") or name})


def _catalog_effective(kind: str, bundled_pair: Optional[tuple[str, dict]], raw_entries: list, name: str):
    """The entry the registry builds from the bundled catalog and these catalog entries, in order."""
    obj = _bundled_object(kind, *bundled_pair) if bundled_pair else None
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        if raw.get("_deleted"):
            obj = None
        elif obj is not None:
            obj = _merged(kind, obj, raw)
        else:
            try:
                obj = _built(kind, name, raw)
            except Exception:
                obj = None  # the registry skips or rejects an incomplete new entry alike
    return obj


def _with_file_entry(kind: str, obj, name: str, data: dict):
    """Apply one separate-file entry the way the registry merges it after the catalog."""
    if obj is _UNLOADABLE:
        return obj
    if kind == "references" and data.get("_deleted"):
        return None
    if obj is not None:
        return _merged(kind, obj, data)
    try:
        return _built(kind, name, data)
    except Exception:
        # A partial reference entry with nothing to merge onto fails the whole load;
        # a model entry is skipped with a warning.
        return _UNLOADABLE if kind == "references" else None


def _changed_fields(left, right) -> set[str]:
    if left is None and right is None:
        return set()
    if left is None or right is None or left is _UNLOADABLE or right is _UNLOADABLE:
        return set() if left is right else {_WHOLE_ENTRY}
    import dataclasses

    changed = {
        f.name
        for f in dataclasses.fields(left)
        if f.name != "variables" and getattr(left, f.name) != getattr(right, f.name)
    }
    left_vars = {str(k).casefold(): v for k, v in (left.variables or {}).items()}
    right_vars = {str(k).casefold(): v for k, v in (right.variables or {}).items()}
    names = {str(k).casefold(): str(k) for k in [*(right.variables or {}), *(left.variables or {})]}
    changed.update(f"variables.{names[key]}" for key in names if left_vars.get(key) != right_vars.get(key))
    return changed


def separate_file_overrides(
    kind: str, catalog_path: Path, new_catalog: dict
) -> list[tuple[str, list[Path], list[str]]]:
    """``(entry, files, fields)`` changed by writing ``new_catalog`` that separate files undo on load.

    The registry merges ``<kind>/*.yaml`` files after the catalog, so a value
    they set wins over the catalog: a saved edit or deletion of that value
    would silently come back after a restart.
    """
    file_entries: dict[str, list[tuple[dict, Path]]] = {}
    for name, data, path in _per_file_entries(Path(catalog_path).parent, kind):
        file_entries.setdefault(normalize_name(name), []).append((data, path))
    if not file_entries:
        return []
    old_catalog, _error = _read_overlay(Path(catalog_path))
    new_catalog = new_catalog if isinstance(new_catalog, dict) else {}
    bundled = _bundled_lookup(_bundled_catalog(kind))
    findings = []
    for key, entries in file_entries.items():
        old_raw = [value for name, value in old_catalog.items() if normalize_name(name) == key]
        new_raw = [value for name, value in new_catalog.items() if normalize_name(name) == key]
        if old_raw == new_raw:
            continue
        name = next(
            (str(name) for name in [*new_catalog, *old_catalog] if normalize_name(name) == key),
            str(entries[0][0].get("name") or key),
        )
        before = _catalog_effective(kind, bundled.get(key), old_raw, name)
        after = _catalog_effective(kind, bundled.get(key), new_raw, name)
        loaded = after
        for data, _path in entries:
            loaded = _with_file_entry(kind, loaded, name, data)
        undone = _changed_fields(before, after) & _changed_fields(after, loaded)
        # A deletion of something only a separate file adds changes nothing in
        # the catalog, so it is judged by what the load still contains.
        undone |= _revived(loaded, _explicit_deletions(new_raw) - _explicit_deletions(old_raw))
        if undone:
            files = list(dict.fromkeys(path for _data, path in entries))
            findings.append((name, files, sorted(undone)))
    return findings


def _explicit_deletions(raw_entries: list) -> set[str]:
    """Casefolded variables, or the whole entry, that catalog entries delete explicitly."""
    deleted = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        if raw.get("_deleted"):
            deleted.add(_WHOLE_ENTRY)
        variables = raw.get("variables")
        if isinstance(variables, dict):
            deleted.update(str(name).casefold() for name, value in variables.items() if value is None)
        deleted.update(str(name).casefold() for name in raw.get("_delete_variables") or [])
    return deleted


def _revived(loaded, deletions: set[str]) -> set[str]:
    """The deletions a load still contains."""
    if not deletions or loaded is None:
        return set()
    if loaded is _UNLOADABLE:
        return {_WHOLE_ENTRY}
    names = {str(name).casefold(): str(name) for name in loaded.variables or {}}
    revived = {f"variables.{names[name]}" for name in deletions if name in names}
    if _WHOLE_ENTRY in deletions:
        revived.add(_WHOLE_ENTRY)
    return revived


def check_saved_entry_against_separate_files(kind: str, catalog_path: Path, name: str, intended) -> None:
    """Refuse saving ``intended`` when separate files would change it on load.

    Comparing the complete object being saved also catches what a catalog
    comparison cannot see, such as deleting a variable only a separate file adds.
    """
    key = normalize_name(name)
    entries = [
        (data, path)
        for entry_name, data, path in _per_file_entries(Path(catalog_path).parent, kind)
        if normalize_name(entry_name) == key
    ]
    if not entries:
        return
    expected = _merged(kind, intended, {})
    loaded = expected
    for data, _path in entries:
        loaded = _with_file_entry(kind, loaded, name, data)
    undone = _changed_fields(expected, loaded)
    if undone:
        files = list(dict.fromkeys(path for _data, path in entries))
        _report_separate_file_overrides([(name, files, sorted(undone))], catalog_path, "raise")


def check_separate_file_overrides(kind: str, catalog_path: Path, new_catalog: dict, *, mode: str = "raise") -> None:
    """Refuse (``raise``) or report (``warn``) a write that separate overlay files would undo."""
    if mode == "ignore":
        return
    findings = separate_file_overrides(kind, catalog_path, new_catalog)
    if findings:
        _report_separate_file_overrides(findings, catalog_path, mode)


def _report_separate_file_overrides(findings: list, catalog_path: Path, mode: str) -> None:
    lines = "\n".join(
        f"  {name} in {', '.join(str(path) for path in files)}: {', '.join(fields)}" for name, files, fields in findings
    )
    message = (
        f"These entries are also defined in separate files, which are merged after {Path(catalog_path).name} "
        f"and would undo the change:\n{lines}\nEdit or remove the entries in those files"
    )
    if mode == "raise":
        raise SeparateFileOverride(f"Not saved. {message}, then save again.")
    logger.warning("%s.", message)


def overlay_kind_for_path(path: Path) -> Optional[str]:
    """Return ``"references"``/``"models"`` when ``path`` is the user overlay catalog."""
    from openbench.data.registry import manager as registry_manager

    if Path(path).name not in {"reference_catalog.yaml", "model_catalog.yaml"}:
        return None
    try:
        target = Path(path).resolve()
        if target == Path(registry_manager.get_writable_reference_catalog_path()).resolve():
            return "references"
        if target == Path(registry_manager.get_writable_model_catalog_path()).resolve():
            return "models"
    except Exception as e:
        logger.debug("overlay_audit: could not classify %s: %s", path, e)
    return None


@lru_cache(maxsize=2)
def _bundled_catalog(kind: str) -> dict:
    """Parsed bundled catalog (read-only: package files do not change at runtime)."""
    from openbench.data.registry.manager import REGISTRY_DIR

    return _load_yaml(REGISTRY_DIR / ("model_catalog.yaml" if kind == "models" else "reference_catalog.yaml"))


def sparsify_overlay_catalog(kind: str, catalog: dict) -> dict:
    """Reduce every overlay of a bundled entry to its minimal, behavior-preserving delta.

    Applied on every write of the user overlay, so values that merely repeat
    the bundled catalog (a GUI save of a whole entry, a rescan that rewrites a
    whole variable) never reach disk and cannot shadow later bundled fixes.
    Custom entries and tombstones are written as given.
    """
    bundled_ci = _bundled_lookup(_bundled_catalog(kind))
    duplicates = _duplicate_names(catalog)
    sparse: dict = {}
    for name, entry in catalog.items():
        match = bundled_ci.get(normalize_name(name))
        if match is None or not isinstance(entry, dict) or entry.get("_deleted") or normalize_name(name) in duplicates:
            sparse[name] = entry
            continue
        bundled_name, bundled_entry = match
        minimal = _sparse_delta(bundled_entry, entry, kind=kind, name=bundled_name)
        if minimal:
            sparse[name] = minimal
    return sparse


@dataclass
class SyncResult:
    label: str
    overlay_path: Path
    compacted: list[str] = field(default_factory=list)
    incomparable: list[str] = field(default_factory=list)
    reset_seeded: bool = False
    backup: Optional[Path] = None


def _sync_marker(overlay_path: Path) -> Path:
    return overlay_path.with_name(f".{overlay_path.name}{_SYNC_MARKER_SUFFIX}")


def _sync_catalog(label: str, bundled_path: Path, overlay_path: Path) -> SyncResult:
    import shutil

    from openbench.data.registry.scanner import _atomic_yaml_write, _safe_load_catalog

    result = SyncResult(label=label, overlay_path=overlay_path)
    overlay = _safe_load_catalog(overlay_path)
    bundled_ci = _bundled_lookup(_load_yaml(bundled_path))
    duplicates = _duplicate_names(overlay)
    # A matching seed hash is direct provenance: the user never edited this
    # copy. Inspect it before compaction changes the file's hash.
    base = overlay_path.parent.parent
    manifest = _load_yaml(base / ".seeded_defaults.yaml")
    seed = manifest.get(overlay_path.relative_to(base).as_posix(), {})
    untouched_seed = (
        isinstance(seed, dict)
        and seed.get("kind") != "empty-overlay"
        and seed.get("sha256") == hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    )
    synced: dict = {}
    for name, entry in () if untouched_seed else overlay.items():
        match = bundled_ci.get(normalize_name(name))
        if match is None or not isinstance(entry, dict) or entry.get("_deleted") or normalize_name(name) in duplicates:
            synced[name] = entry
            continue
        bundled_name, bundled_entry = match
        try:
            base_obj = _bundled_object(label, bundled_name, bundled_entry)
            _merged(label, base_obj, entry)
        except Exception:
            result.incomparable.append(str(name))
            synced[name] = entry
            continue
        # Historical equality does not prove snapshot provenance: even a full
        # catalog may contain intentional overrides. Only remove redundancy
        # against the current bundled catalog without changing behavior.
        minimal = _sparse_delta(bundled_entry, entry, kind=label, name=bundled_name)
        if not _same_behavior(label, base_obj, minimal, entry):
            result.incomparable.append(str(name))
            synced[name] = entry
            continue
        if minimal != entry:
            result.compacted.append(str(name))
        if minimal:
            synced[name] = minimal
    if synced != overlay:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = overlay_path.with_name(f"{overlay_path.name}.before-bundled-sync-{stamp}.bak")
        shutil.copy2(overlay_path, backup)
        result.backup = backup
        _atomic_yaml_write(overlay_path, synced)
        if untouched_seed:
            result.reset_seeded = True
            result.compacted = list(overlay)
            manifest[overlay_path.relative_to(base).as_posix()] = {
                "sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
                "kind": "empty-overlay",
            }
            _atomic_yaml_write(base / ".seeded_defaults.yaml", manifest)
    return result


_sync_suspensions = 0


def _automatically_writable(path: Path) -> bool:
    try:
        return path.resolve() == path.absolute() and os.access(path, os.W_OK)
    except OSError:
        return False


def suspend_sync():
    """Turn the one-time overlay sync off for this process until the returned callable runs.

    Read-only CLI invocations (help, dry runs, inspection) must not rewrite the
    user's catalog, including through ``get_registry()``.
    """
    global _sync_suspensions
    _sync_suspensions += 1
    resumed = False

    def resume() -> None:
        global _sync_suspensions
        nonlocal resumed
        if not resumed:
            resumed = True
            _sync_suspensions = max(0, _sync_suspensions - 1)

    return resume


def sync_legacy_overlays(user_dir: Optional[Path] = None) -> list[SyncResult]:
    """Once per overlay file, remove only behavior-preserving redundancy.

    Historical matches cannot distinguish copied defaults from deliberate
    overrides. Preserve them, along with literal paths that currently happen
    to equal an expanded bundled path. Back up each rewritten catalog.
    """
    if _sync_suspensions or os.environ.get(_SUPPRESS_ENV):
        return []
    _base, bundled_ref, overlay_ref, bundled_model, overlay_model = _paths(user_dir)
    pending = [
        (label, bundled_path, overlay_path)
        for label, bundled_path, overlay_path in (
            ("references", bundled_ref, overlay_ref),
            ("models", bundled_model, overlay_model),
        )
        # A linked overlay (the file or a directory above it, e.g. a team catalog)
        # or a read-only one is never rewritten automatically: the atomic replace
        # would turn a link into a private copy or override the read-only choice.
        if overlay_path.is_file() and _automatically_writable(overlay_path) and not _sync_marker(overlay_path).exists()
    ]
    if not pending:
        # Every run after the first lands here: two stat calls, no heavy imports.
        return []

    from openbench.data.registry.scanner import _catalog_write_lock, _invalidate_registry_caches

    results: list[SyncResult] = []
    for label, bundled_path, overlay_path in pending:
        marker = _sync_marker(overlay_path)
        try:
            with _catalog_write_lock(overlay_path):
                if marker.exists():
                    continue
                result = _sync_catalog(label, bundled_path, overlay_path)
                record = {
                    "format": _SYNC_MARKER_FORMAT,
                    "synced_at": datetime.now().isoformat(timespec="seconds"),
                    "compacted": result.compacted,
                    "incomparable": result.incomparable,
                    "backup": str(result.backup) if result.backup else None,
                }
                marker.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.debug("Could not sync %s with the bundled catalog: %s", overlay_path, e)
            continue
        except Exception as e:
            logger.debug("Could not compact %s: %s", overlay_path, e)
            continue
        if result.incomparable:
            logger.warning(
                "Preserved registry entries that could not be compared in %s: %s",
                overlay_path,
                ", ".join(result.incomparable),
            )
        if result.backup is not None:
            _invalidate_registry_caches()
            results.append(result)
    return results


def format_sync_notice(results: list[SyncResult]) -> Optional[str]:
    """Describe successful compaction without claiming overrides were replaced."""
    lines = []
    for result in results:
        if not result.compacted:
            continue
        count = len(result.compacted)
        shown = ", ".join(result.compacted[:8]) + (f", … (+{count - 8} more)" if count > 8 else "")
        action = (
            "were reset from an unchanged seeded copy"
            if result.reset_seeded
            else "were compacted without changing overrides"
        )
        lines.append(
            f"✓ {count} {_KIND_NOUN[result.label]} entr{'y' if count == 1 else 'ies'} in {result.overlay_path} "
            f"{action}: {shown}"
        )
        lines.append(f"  Previous file backed up to {result.backup}")
    return "\n".join(lines) or None


# --------------------------------------------------------------------------- #
# Throttled startup notice
# --------------------------------------------------------------------------- #


def _fingerprint(user_dir: Path, overlay_ref: Path, overlay_model: Path) -> str:
    from openbench import __version__

    h = hashlib.sha256()
    h.update(__version__.encode())
    for p in (overlay_ref, overlay_model):
        try:
            h.update(p.read_bytes() if p.exists() else b"")
        except OSError:
            pass
    for kind, catalog in (("references", overlay_ref), ("models", overlay_model)):
        for p in _per_file_paths(catalog.parent, kind):
            h.update(p.name.encode())
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
    return h.hexdigest()


def _read_state(state_path: Path) -> dict:
    try:
        return json.loads(state_path.read_text()) if state_path.exists() else {}
    except (OSError, ValueError):
        return {}


def _write_state(state_path: Path, fingerprint: str) -> None:
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"fingerprint": fingerprint}))
    except OSError:
        pass


def overlay_hints(user_dir: Optional[Path] = None) -> list[str]:
    """Overrides worth a look, one line each, for front ends such as the GUI.

    Never raises; an unreadable overlay yields no hints (the registry load
    reports it).
    """
    try:
        audit = audit_overlays(user_dir)
    except Exception as e:
        logger.debug("overlay hints skipped: %s", e)
        return []
    hints = []
    for catalog in audit.catalogs:
        for entry in catalog.entries:
            if entry.historical_fields:
                hints.append(
                    f"{entry.name}: {', '.join(entry.historical_fields)} "
                    f"{'matches' if len(entry.historical_fields) == 1 else 'match'} older bundled defaults and may "
                    f"be outdated (values kept; `openbench registry reset {entry.name}` restores the bundled entry)."
                )
        if catalog.bloat:
            count = len(catalog.bloat)
            hints.append(
                f"{count} {_KIND_NOUN[catalog.label]} overlay entr{'y' if count == 1 else 'ies'} "
                f"cop{'ies' if count == 1 else 'y'} the bundled catalog "
                "and may hide its fixes (`openbench registry prune`)."
            )
        if catalog.duplicates:
            names = ", ".join(entry.name for entry in catalog.duplicates)
            hints.append(f"{_KIND_NOUN[catalog.label]} overlay keys differ only in case ({names}); keep one.")
        if catalog.error:
            hints.append(f"{catalog.overlay_path} {catalog.error}.")
        for entry in catalog.per_file_attention:
            if entry.historical_fields:
                fields = ", ".join(entry.historical_fields)
                verb = "matches" if len(entry.historical_fields) == 1 else "match"
                hints.append(
                    f"{entry.name} in {entry.path}: {fields} {verb} older bundled defaults and may be outdated "
                    "(values kept; edit or remove the entry in that file to follow the bundled catalog)."
                )
            else:
                hints.append(
                    f"{entry.name} in {entry.path} copies the bundled entry and may hide its fixes "
                    "(edit or remove the entry in that file)."
                )
    return hints


def maybe_emit_overlay_notice(user_dir: Optional[Path] = None) -> Optional[str]:
    """Print a one-line stderr notice when the overlay shadows bundled, throttled.

    Returns the message printed (for tests), or None. Silent when the overlay is
    clean, suppressed via ``OPENBENCH_NO_REGISTRY_CHECK``, or unchanged since the
    last check (fingerprint = openbench version + overlay file bytes). Never
    raises — registry hygiene must not break the CLI.
    """
    if os.environ.get(_SUPPRESS_ENV):
        return None
    try:
        base, _bref, overlay_ref, _bmodel, overlay_model = _paths(user_dir)
        state_path = base / _STATE_FILENAME
        fp = _fingerprint(base, overlay_ref, overlay_model)
        if _read_state(state_path).get("fingerprint") == fp:
            return None  # nothing changed since last check — skip all work, stay silent

        audit = audit_overlays(base)
        _write_state(state_path, fp)  # persist regardless so we don't recheck until next change

        bloat = audit.bloat_count
        duplicates = audit.duplicate_count
        historical = [entry for catalog in audit.catalogs for entry in catalog.entries if entry.historical_fields]
        separate = [entry for catalog in audit.catalogs for entry in catalog.per_file_attention]
        if bloat == 0 and not historical and not duplicates and not separate:
            return None
        parts = []
        if bloat:
            stale = audit.stale_count
            stale_note = f" ({stale} may be stale)" if stale else ""
            parts.append(
                f"registry overlay shadows {bloat} bundled entr{'y' if bloat == 1 else 'ies'}{stale_note}; "
                "bundled fixes may be hidden"
            )
        if historical:
            parts.append(
                f"{len(historical)} override(s) match older bundled defaults and may be outdated "
                "(values preserved; openbench registry reset ENTRY restores one)"
            )
        if duplicates:
            parts.append(f"{duplicates} overlay key(s) differ only in case; later ones win")
        if separate:
            parts.append(
                f"{len(separate)} entr{'y' if len(separate) == 1 else 'ies'} in separate overlay files "
                "may hide bundled fixes (edit or remove them)"
            )
        msg = "⚠ " + "; ".join(parts) + ". Run: openbench registry diff"
        import click

        click.echo(msg, err=True)
        return msg
    except Exception as e:  # pragma: no cover - defensive: never break the CLI
        logger.debug("overlay_audit notice skipped: %s", e)
        return None
