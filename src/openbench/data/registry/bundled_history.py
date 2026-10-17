"""Recognize user-overlay values that are copies of an older bundled catalog.

Older OpenBench versions seeded ``~/.openbench`` with a full copy of the bundled
catalogs, and later writes kept that copy around. Every value in such a copy
shadows the bundled catalog, so fixes shipped in a newer bundled catalog never
reach the user. A historical match is only evidence of a possible snapshot;
it cannot distinguish copied defaults from intentional user overrides and
must not be used to replace settings automatically.

``bundled_history.json`` stores one digest per (entry, field, value) that the
bundled reference and model catalogs have ever held. It is generated from git
history by ``scripts/build_bundled_registry_history.py``; a test keeps it in
step with the current catalogs so every value is recorded before it can change.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Iterator

from openbench.util.names import canonical_variable_name, normalize_name

HISTORY_FILENAME = "bundled_history.json"
HISTORY_FORMAT = 1
CATALOG_KINDS = ("references", "models")

# Overlay-only control keys; never compared against bundled history.
CONTROL_KEYS = frozenset({"name", "variables", "_deleted", "_delete_variables"})


def _digest(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _var_keys(var_name: Any) -> tuple[str, ...]:
    raw = normalize_name(var_name)
    canonical = normalize_name(canonical_variable_name(str(var_name)))
    return (raw,) if raw == canonical else (raw, canonical)


def field_digest(kind: str, entry: Any, field: str, value: Any) -> str:
    return _digest(kind, "field", normalize_name(entry), field, value)


def var_field_digests(kind: str, entry: Any, var: Any, field: str, value: Any) -> tuple[str, ...]:
    return tuple(_digest(kind, "var_field", normalize_name(entry), key, field, value) for key in _var_keys(var))


def var_digests(kind: str, entry: Any, var: Any, value: Any) -> tuple[str, ...]:
    return tuple(_digest(kind, "var", normalize_name(entry), key, value) for key in _var_keys(var))


def iter_entry_digests(kind: str, name: Any, entry: Any) -> Iterator[str]:
    """Yield every digest that a bundled catalog entry contributes to history."""
    if not isinstance(entry, dict):
        return
    for field, value in entry.items():
        if field not in CONTROL_KEYS:
            yield field_digest(kind, name, field, value)
    for var, var_data in (entry.get("variables") or {}).items():
        if not isinstance(var_data, dict):
            continue
        yield from var_digests(kind, name, var, var_data)
        for field, value in var_data.items():
            yield from var_field_digests(kind, name, var, field, value)


@lru_cache(maxsize=1)
def load_history() -> dict[str, frozenset[str]]:
    """Return the shipped digests per catalog kind (empty when unavailable)."""
    from openbench.data.registry.manager import REGISTRY_DIR

    resource = REGISTRY_DIR / HISTORY_FILENAME
    try:
        data = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {kind: frozenset() for kind in CATALOG_KINDS}
    if not isinstance(data, dict) or data.get("format") != HISTORY_FORMAT:
        return {kind: frozenset() for kind in CATALOG_KINDS}
    return {kind: frozenset(data.get(kind) or ()) for kind in CATALOG_KINDS}
