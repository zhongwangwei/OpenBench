"""Incremental evaluation cache.

Skips re-computation when config + data haven't changed.
Uses SHA-256 hash of the evaluation parameters to detect changes.
Cache metadata stored in output_dir/.openbench_cache.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvaluationCache:
    """Track which evaluations have been completed with matching configs."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / ".openbench_cache.json"
        self._cache = self._load()

    def _load(self) -> dict[str, str]:
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def is_cached(self, key: str, config_hash: str) -> bool:
        """Check if an evaluation with this config hash is already done."""
        return self._cache.get(key) == config_hash

    def mark_done(self, key: str, config_hash: str) -> None:
        """Mark an evaluation as completed with this config hash."""
        self._cache[key] = config_hash
        self._save()

    def invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        self._cache.pop(key, None)
        self._save()

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._save()

    @staticmethod
    def hash_config(config: dict[str, Any]) -> str:
        """Create a deterministic hash of evaluation config."""
        # Convert to sorted JSON string for deterministic hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def make_cache_key(variable: str, sim_source: str, ref_source: str) -> str:
    """Create a cache key for a variable+sim+ref combination."""
    return f"{variable}__{sim_source}__{ref_source}"
