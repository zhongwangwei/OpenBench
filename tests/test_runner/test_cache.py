"""Tests for evaluation cache."""

import tempfile
from pathlib import Path

import pytest

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
    assert key.startswith("v2:")
    assert key == make_cache_key("Evapotranspiration", "CoLM2024", "GLEAM_v4.2a")


def test_cache_key_is_unambiguous_when_names_contain_separator():
    assert make_cache_key("A__B", "C", "D") != make_cache_key("A", "B__C", "D")


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
    original = cache_file.read_text(encoding="utf-8")

    cache = EvaluationCache(tmp_path)
    # Empty in-memory cache; re-evaluation expected on next run
    assert cache._cache == {}
    # Corrupted file renamed; original cache_file no longer exists
    assert not cache_file.exists()
    # Find the .corrupt sibling and verify it preserves the broken content
    corrupt_files = list(tmp_path.glob(".openbench_cache.corrupt-*"))
    assert len(corrupt_files) == 1, f"Expected one .corrupt-* file, found: {[f.name for f in corrupt_files]}"
    assert corrupt_files[0].read_text(encoding="utf-8") == original


def test_windows_file_lock_seeds_empty_lock_file_before_locking(monkeypatch, tmp_path):
    import openbench.runner.cache as cache_module

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        def __init__(self):
            self.calls = []

        def locking(self, fileno, mode, nbytes):
            self.calls.append((mode, nbytes))

    fake_msvcrt = FakeMsvcrt()
    monkeypatch.setattr(cache_module, "_HAS_FCNTL", False)
    monkeypatch.setattr(cache_module, "_HAS_MSVCRT", True)
    monkeypatch.setattr(cache_module, "msvcrt", fake_msvcrt)

    lock_path = tmp_path / ".openbench_cache.lock"
    with cache_module._file_lock(lock_path):
        assert lock_path.stat().st_size == 1

    assert fake_msvcrt.calls == [(fake_msvcrt.LK_LOCK, 1), (fake_msvcrt.LK_UNLCK, 1)]


def test_posix_file_lock_failure_does_not_continue_without_lock(monkeypatch, tmp_path, caplog):
    import openbench.runner.cache as cache_module

    class FakeFcntl:
        LOCK_EX = 1
        LOCK_UN = 2

        def flock(self, fileno, mode):
            if mode == self.LOCK_EX:
                raise OSError("lock backend unavailable")

    monkeypatch.setattr(cache_module, "_HAS_FCNTL", True)
    monkeypatch.setattr(cache_module, "_HAS_MSVCRT", False)
    monkeypatch.setattr(cache_module, "fcntl", FakeFcntl())

    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="failed to acquire cache lock"):
        with cache_module._file_lock(tmp_path / ".openbench_cache.lock"):
            raise AssertionError("unlocked body must not run")

    assert "fcntl.flock unavailable" in caplog.text


def test_input_file_signature_includes_ctime_ns(tmp_path):
    from openbench.runner.hashing import input_file_signature

    source_root = tmp_path / "input"
    source_root.mkdir()
    data_file = source_root / "sample.nc"
    data_file.write_bytes(b"abcdef")

    signature = input_file_signature({"Case_dir": str(source_root)}, "Case")

    assert len(signature["files"]) == 1
    assert "ctime_ns" in signature["files"][0]


def test_input_file_signature_refreshes_after_input_changes(tmp_path):
    from openbench.runner.hashing import input_file_signature

    source_root = tmp_path / "input"
    source_root.mkdir()
    data_file = source_root / "sample.nc"
    data_file.write_bytes(b"first")
    section = {"Case_dir": str(source_root)}

    first = input_file_signature(section, "Case")
    data_file.write_bytes(b"second-version")
    refreshed = input_file_signature(section, "Case")

    assert refreshed != first


def test_algorithm_source_fingerprint_tracks_runner_processing_and_comparisons():
    from openbench.runner.hashing import ALGORITHM_SOURCE_MODULES

    modules = set(ALGORITHM_SOURCE_MODULES)

    assert "openbench.core.evaluation" in modules
    assert "openbench.core._comparison_helpers" in modules
    assert "openbench.runner.masking" in modules
    assert "openbench.data._processing_grid_regrid" in modules


def test_source_specific_section_does_not_capture_longer_source_name():
    from openbench.runner.hashing import source_specific_section

    section = {
        "LAI_Yuan2011_varname": "lai",
        "LAI_Yuan2011_8Day_varname": "lai_8day",
    }

    assert source_specific_section(
        section,
        "LAI_Yuan2011",
        all_sources=["LAI_Yuan2011", "LAI_Yuan2011_8Day"],
    ) == {"LAI_Yuan2011_varname": "lai"}


def test_unified_mask_missing_inputs_raise(tmp_path):
    import openbench.runner.masking as masking_module

    info = {
        "casedir": str(tmp_path),
        "ref_varname": "ref",
        "sim_varname": "sim",
        "time_alignment": "intersection",
    }

    with pytest.raises(FileNotFoundError, match="Unified mask input file not found"):
        masking_module.apply_unified_mask(
            info,
            "GPP",
            "RefA",
            "SimA",
            write_netcdf_atomic_fn=lambda *args, **kwargs: None,
        )


def test_unified_mask_keeps_chunked_data_lazy_until_writer(tmp_path, monkeypatch):
    import dask.array as da
    import xarray as xr
    from dask.base import is_dask_collection

    import openbench.runner.masking as masking_module

    ref_path = tmp_path / "data" / "GPP_ref_RefA_ref.nc"
    sim_path = tmp_path / "data" / "GPP_sim_SimA_sim.nc"
    ref_path.parent.mkdir()
    ref_path.touch()
    sim_path.touch()
    coords = {"time": [0, 1], "lat": [0.0], "lon": [0.0]}

    states = {"ref": "closed", "sim": "closed"}

    def dataset(name, values):
        states[name] = "open"
        result = xr.DataArray(
            da.from_array(values, chunks=(1, 1, 1)),
            coords=coords,
            dims=("time", "lat", "lon"),
            name=name,
        ).to_dataset()
        result.set_close(lambda: states.__setitem__(name, "closed"))
        return result

    open_calls = []

    def open_dataset(path, **kwargs):
        open_calls.append(kwargs)
        if str(path) == str(ref_path):
            return dataset("ref", [[[1.0]], [[2.0]]])
        return dataset("sim", [[[1.0]], [[float("nan")]]])

    monkeypatch.setattr(xr, "open_dataset", open_dataset)
    observed = {}

    def writer(data, *_args, **_kwargs):
        observed["lazy"] = is_dask_collection(data.data)
        observed["values"] = data.compute().values
        Path(_args[0]).write_bytes(b"masked")

    real_replace = masking_module.os.replace

    def replace_after_close(source, target):
        assert states == {"ref": "closed", "sim": "closed"}
        real_replace(source, target)

    monkeypatch.setattr(masking_module.os, "replace", replace_after_close)

    masking_module.apply_unified_mask(
        {
            "casedir": str(tmp_path),
            "ref_varname": "ref",
            "sim_varname": "sim",
            "time_alignment": "intersection",
        },
        "GPP",
        "RefA",
        "SimA",
        write_netcdf_atomic_fn=writer,
    )

    assert observed["lazy"] is True
    assert open_calls == [{"chunks": "auto"}, {"chunks": "auto"}]
    assert ref_path.read_bytes() == b"masked"
    assert observed["values"].shape == (1, 1, 1)
    assert observed["values"][0, 0, 0] == 1.0
