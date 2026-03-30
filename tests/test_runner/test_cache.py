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
