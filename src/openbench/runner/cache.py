"""Incremental evaluation cache.

Skips re-computation when config + data haven't changed.
Uses SHA-256 hash of the evaluation parameters to detect changes.
Cache metadata stored in output_dir/.openbench_cache.json

Thread/process safe: atomic write (temp file + rename) AND a platform file
lock around mark_done/invalidate so concurrent workers writing distinct keys
don't lose each other's updates via a load-modify-save race.

NOTE: POSIX fcntl.flock is advisory and may behave inconsistently on some
NFS implementations. For local filesystems (the common OpenBench case) it
provides full mutual exclusion across processes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import fcntl  # POSIX only (macOS/Linux)

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    _HAS_FCNTL = False

try:
    import msvcrt  # Windows only

    _HAS_MSVCRT = True
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None
    _HAS_MSVCRT = False

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Cross-platform file lock context manager.

    On POSIX uses fcntl.flock(LOCK_EX). On Windows uses msvcrt.locking()
    over the first byte of a dedicated lock file. If the platform lock cannot
    be acquired, fail closed rather than continuing with an unprotected
    load-modify-save sequence that can lose cache entries under concurrency.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    locked = False
    backend = None
    try:
        if _HAS_FCNTL:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                locked = True
                backend = "fcntl"
            except OSError as e:
                logger.warning("fcntl.flock unavailable on %s: %s", lock_path, e)
                raise RuntimeError(f"failed to acquire cache lock {lock_path}") from e
        elif _HAS_MSVCRT:
            try:
                f.seek(0)
                if f.seek(0, os.SEEK_END) == 0:
                    f.write(b"\0")
                    f.flush()
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                locked = True
                backend = "msvcrt"
            except OSError as e:
                logger.warning("msvcrt.locking unavailable on %s: %s", lock_path, e)
                raise RuntimeError(f"failed to acquire cache lock {lock_path}") from e
        else:
            raise RuntimeError("no supported cache file-lock backend is available")
        yield
    finally:
        try:
            if locked and backend == "fcntl":
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            elif locked and backend == "msvcrt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        f.close()


class EvaluationCache:
    """Track which evaluations have been completed with matching configs."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / ".openbench_cache.json"
        self._lock_file = cache_dir / ".openbench_cache.lock"
        self._cache = self._load()

    def _load(self) -> dict[str, str]:
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # Preserve the corrupted file as <cache>.corrupt for diagnostics
                # rather than letting the next save overwrite it. Falling back
                # to empty {} causes a silent re-evaluation of all cached
                # tasks; preserving the broken file lets users inspect what
                # went wrong (e.g., partial write from a crashed process).
                try:
                    import time

                    corrupt_path = self._cache_file.with_suffix(f".corrupt-{int(time.time())}")
                    self._cache_file.rename(corrupt_path)
                    logger.warning(
                        "Failed to load cache file %s: %s. Renamed to %s for "
                        "diagnostics; starting with empty cache (re-evaluation "
                        "expected on next run).",
                        self._cache_file,
                        e,
                        corrupt_path,
                    )
                except OSError as rename_err:
                    logger.warning(
                        "Failed to load cache file %s: %s. Could not preserve "
                        "diagnostic copy (%s); starting with empty cache.",
                        self._cache_file,
                        e,
                        rename_err,
                    )
                return {}
        return {}

    def _save(self) -> None:
        """Save cache atomically (write to temp file, then rename)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Atomic write: temp file + rename prevents race conditions
            fd, tmp_path = tempfile.mkstemp(dir=str(self._cache_dir), suffix=".tmp", prefix=".cache_")
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
        # Re-load from disk to pick up writes from other processes. Hold the lock
        # so a corrupted-file rename inside _load() is serialized with
        # mark_done()/invalidate() instead of racing a concurrent rename.
        with _file_lock(self._lock_file):
            self._cache = self._load()
        return self._cache.get(key) == config_hash

    def mark_done(self, key: str, config_hash: str) -> None:
        """Mark an evaluation as completed with this config hash.

        Uses fcntl.flock to make the reload→modify→save sequence atomic
        across processes. Without the lock, two workers writing distinct
        keys concurrently could lose updates via the load-modify-save race.
        """
        with _file_lock(self._lock_file):
            self._cache = self._load()
            self._cache[key] = config_hash
            self._save()

    def invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        with _file_lock(self._lock_file):
            self._cache = self._load()
            self._cache.pop(key, None)
            self._save()

    def clear(self) -> None:
        """Clear all cache entries."""
        with _file_lock(self._lock_file):
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
