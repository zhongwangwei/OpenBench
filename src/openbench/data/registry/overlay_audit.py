"""Audit and re-sparse the user registry overlay against the bundled catalog.

The user overlay (``~/.openbench/references/reference_catalog.yaml`` and
``~/.openbench/models/model_catalog.yaml``) is meant to hold only *sparse
deltas* on top of the bundled catalog. Older OpenBench versions sometimes wrote
the entire merged catalog back as a full snapshot. A full-snapshot overlay
silently shadows the bundled catalog field-by-field, freezing the user on a
stale version so bundled fixes never take effect.

This module classifies each overlay entry, can re-sparse the overlay
(behavior-preserving), and powers a throttled startup notice.

Behavior-preservation note: the manager's deep merge (``_deep_merge_reference`` /
``_deep_merge_model``) overrides per-field/per-variable and never deletes a
bundled variable just because the overlay omits it (only an explicit ``None``
tombstone deletes). So the minimal behavior-preserving overlay is computed by
``_sparse_delta`` below — which keeps only differing fields/variables and adds
NO tombstones. (``scanner._descriptor_overlay_diff`` deliberately differs: it
tombstones omitted variables, which is correct when writing a complete
descriptor but would change behavior if applied to an existing snapshot.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Classification kinds
REDUNDANT = "redundant"  # overlay entry byte-equals bundled (pure snapshot leftover)
STALE_FULLCOPY = "stale_fullcopy"  # carries bundled-equal baggage AND differs -> stale snapshot
DELTA = "delta"  # already-minimal deliberate override
CUSTOM = "custom"  # entry not present in bundled (genuine user addition)

_STATE_FILENAME = ".registry_check"
_SUPPRESS_ENV = "OPENBENCH_NO_REGISTRY_CHECK"


def _ci_get(mapping: dict, key: str) -> Any:
    """Case-insensitive dict lookup (exact first)."""
    if key in mapping:
        return mapping[key]
    lk = str(key).lower()
    for k, v in mapping.items():
        if str(k).lower() == lk:
            return v
    return None


def _sparse_delta(bundled_entry: dict, overlay_entry: dict) -> dict:
    """Minimal, behavior-preserving overlay for ``overlay_entry`` over bundled.

    Keeps only top-level fields and variables whose value differs from bundled.
    Adds NO tombstones (matches deep-merge semantics). ``name`` is structural
    (implied by the catalog key) and is dropped.
    """
    out: dict = {}
    bundled_entry = bundled_entry or {}
    for key, value in (overlay_entry or {}).items():
        if key == "name":
            continue
        if key == "variables" and isinstance(value, dict):
            bundled_vars = bundled_entry.get("variables", {}) or {}
            var_out = {vk: vv for vk, vv in value.items() if _ci_get(bundled_vars, vk) != vv}
            if var_out:
                out["variables"] = var_out
        elif bundled_entry.get(key) != value:
            out[key] = value
    return out


def _strip_name(entry: dict) -> dict:
    return {k: v for k, v in (entry or {}).items() if k != "name"}


def _classify_entry(bundled_entry: Optional[dict], overlay_entry: dict) -> tuple[str, dict]:
    """Return (kind, minimal_delta) for one overlay entry."""
    if bundled_entry is None:
        return CUSTOM, _strip_name(overlay_entry)
    delta = _sparse_delta(bundled_entry, overlay_entry)
    if not delta:
        return REDUNDANT, {}
    if _strip_name(overlay_entry) == delta:
        return DELTA, delta
    return STALE_FULLCOPY, delta


@dataclass
class EntryAudit:
    name: str
    kind: str
    minimal: dict = field(default_factory=dict)


@dataclass
class CatalogAudit:
    label: str  # "references" | "models"
    overlay_path: Path
    bundled_path: Path
    exists: bool
    entries: list[EntryAudit] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[EntryAudit]:
        return [e for e in self.entries if e.kind == kind]

    @property
    def bloat(self) -> list[EntryAudit]:
        """Entries that silently shadow bundled and are freeze/staleness risks."""
        return [e for e in self.entries if e.kind in (REDUNDANT, STALE_FULLCOPY)]


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


def _load_yaml(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # pragma: no cover - corrupted file is surfaced elsewhere
        logger.debug("overlay_audit: could not read %s: %s", path, e)
        return {}


def _audit_catalog(label: str, bundled_path: Path, overlay_path: Path) -> CatalogAudit:
    bundled = _load_yaml(bundled_path)
    overlay = _load_yaml(overlay_path)
    # Case-insensitive bundled lookup keyed by lowercase name.
    bundled_ci = {str(k).lower(): v for k, v in bundled.items()}
    entries: list[EntryAudit] = []
    for name, data in overlay.items():
        if not isinstance(data, dict):
            continue
        if data.get("_deleted"):
            # Tombstones are intentional deletions — treat as deliberate.
            entries.append(EntryAudit(name=name, kind=DELTA, minimal=data))
            continue
        bundled_entry = bundled_ci.get(str(name).lower())
        kind, minimal = _classify_entry(bundled_entry, data)
        entries.append(EntryAudit(name=name, kind=kind, minimal=minimal))
    return CatalogAudit(
        label=label,
        overlay_path=overlay_path,
        bundled_path=bundled_path,
        exists=overlay_path.exists(),
        entries=entries,
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
    new_overlay: dict = {}
    overlay_raw = _load_yaml(audit.overlay_path)
    for entry in audit.entries:
        original = overlay_raw.get(entry.name)
        if entry.kind == REDUNDANT:
            result.removed.append(entry.name)
            continue
        if entry.kind == STALE_FULLCOPY:
            new_overlay[entry.name] = entry.minimal
            result.minimized.append(entry.name)
            continue
        # DELTA / CUSTOM: keep exactly as written
        new_overlay[entry.name] = original
        result.kept.append(entry.name)

    changed = result.removed or result.minimized
    if changed and not dry_run:
        from openbench.data.registry.scanner import _backup_then_write, _invalidate_registry_caches

        audit.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        result.backup = _backup_then_write(audit.overlay_path, new_overlay)
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
        if bloat == 0:
            return None
        stale = audit.stale_count
        stale_note = f" ({stale} may be stale)" if stale else ""
        msg = (
            f"⚠ registry overlay shadows {bloat} bundled "
            f"entr{'y' if bloat == 1 else 'ies'}{stale_note}; "
            f"bundled fixes may be hidden. Run: openbench registry diff"
        )
        import click

        click.echo(msg, err=True)
        return msg
    except Exception as e:  # pragma: no cover - defensive: never break the CLI
        logger.debug("overlay_audit notice skipped: %s", e)
        return None
