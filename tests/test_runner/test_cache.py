"""Tests for evaluation cache."""

import tempfile
from pathlib import Path

from openbench.runner.cache import EvaluationCache, make_cache_key


def test_cache_miss():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EvaluationCache(Path(tmpdir))
        assert not cache.is_cached("key1", "hash1")


def test_cache_hit():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EvaluationCache(Path(tmpdir))
        cache.mark_done("key1", "hash1")
        assert cache.is_cached("key1", "hash1")


def test_cache_miss_on_hash_change():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EvaluationCache(Path(tmpdir))
        cache.mark_done("key1", "hash1")
        assert not cache.is_cached("key1", "hash2")


def test_cache_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache1 = EvaluationCache(Path(tmpdir))
        cache1.mark_done("key1", "hash1")

        cache2 = EvaluationCache(Path(tmpdir))
        assert cache2.is_cached("key1", "hash1")


def test_cache_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EvaluationCache(Path(tmpdir))
        cache.mark_done("key1", "hash1")
        cache.clear()
        assert not cache.is_cached("key1", "hash1")


def test_cache_key():
    key = make_cache_key("Evapotranspiration", "CoLM2024", "GLEAM_v4.2a")
    assert key == "Evapotranspiration__CoLM2024__GLEAM_v4.2a"


def test_hash_config():
    h1 = EvaluationCache.hash_config({"a": 1, "b": 2})
    h2 = EvaluationCache.hash_config({"b": 2, "a": 1})  # Same content, different order
    assert h1 == h2  # Should be same (sorted keys)

    h3 = EvaluationCache.hash_config({"a": 1, "b": 3})  # Different content
    assert h1 != h3


def test_load_preserves_corrupt_file_for_diagnostics(tmp_path):
    """A corrupted JSON cache file must be renamed to <cache>.corrupt-<ts>
    rather than silently overwritten on the next save. Preserves diagnostic
    evidence (e.g., partial write from a crashed process) for the user.
    """
    cache_file = tmp_path / ".openbench_cache.json"
    cache_file.write_text("{not valid json")
    original = cache_file.read_text()

    cache = EvaluationCache(tmp_path)
    # Empty in-memory cache; re-evaluation expected on next run
    assert cache._cache == {}
    # Corrupted file renamed; original cache_file no longer exists
    assert not cache_file.exists()
    # Find the .corrupt sibling and verify it preserves the broken content
    corrupt_files = list(tmp_path.glob(".openbench_cache.corrupt-*"))
    assert len(corrupt_files) == 1, (
        f"Expected one .corrupt-* file, found: {[f.name for f in corrupt_files]}"
    )
    assert corrupt_files[0].read_text() == original
