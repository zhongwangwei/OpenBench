"""Incremental evaluation cache.

Skips re-computation when config + data haven't changed.
Uses SHA-256 hash of the evaluation parameters to detect changes.
Cache metadata stored in output_dir/.openbench_cache.json

Thread/process safe: uses atomic write (write to temp + rename).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load cache file %s: %s", self._cache_file, e)
                return {}
        return {}

    def _save(self) -> None:
        """Save cache atomically (write to temp file, then rename)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Atomic write: temp file + rename prevents race conditions
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".tmp", prefix=".cache_"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._cache, f, indent=2)
                os.replace(tmp_path, str(self._cache_file))  # Atomic on POSIX
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

    def is_cached(self, key: str, config_hash: str) -> bool:
        """Check if an evaluation with this config hash is already done."""
        # Re-load from disk to pick up writes from other processes
        self._cache = self._load()
        return self._cache.get(key) == config_hash

    def mark_done(self, key: str, config_hash: str) -> None:
        """Mark an evaluation as completed with this config hash."""
        # Re-load to merge with other processes' writes
        self._cache = self._load()
        self._cache[key] = config_hash
        self._save()

    def invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        self._cache = self._load()
        self._cache.pop(key, None)
        self._save()

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._save()

    @staticmethod
    def hash_config(config: dict[str, Any]) -> str:
        """Create a deterministic hash of evaluation config."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def make_cache_key(variable: str, sim_source: str, ref_source: str) -> str:
    """Create a cache key for a variable+sim+ref combination."""
    return f"{variable}__{sim_source}__{ref_source}"
