import numpy as np
import pytest
import xarray as xr

from openbench.data import cache as cache_module
from openbench.data.cache import CacheManager, DataCache, MemoryCache
from openbench.util.exceptions import CacheError


def test_memory_cache_overwrite_replaces_size_instead_of_accumulating():
    cache = MemoryCache(max_size_mb=1)

    assert cache.set("same", np.arange(4, dtype=np.int64))
    first_size = cache._current_size
    assert cache.set("same", np.arange(4, dtype=np.int64))

    assert cache._current_size == first_size
    assert cache.sizes["same"] == first_size


def test_data_cache_key_includes_dataset_content_not_just_name(tmp_path):
    manager = CacheManager(cache_dir=str(tmp_path), memory_size_mb=1, disk_size_mb=10)
    manager.use_memory = False
    data_cache = DataCache(manager)
    ds_a = xr.Dataset({"v": ("x", np.array([1.0, 2.0]))}, coords={"x": [0, 1]})
    ds_b = xr.Dataset({"v": ("x", np.array([3.0, 4.0]))}, coords={"x": [0, 1]})

    key_a = data_cache.cache_dataset(ds_a, "same-name")
    key_b = data_cache.cache_dataset(ds_b, "same-name")

    assert key_a != key_b
    got_a, _ = data_cache.get_dataset(key_a)
    got_b, _ = data_cache.get_dataset(key_b)
    np.testing.assert_array_equal(got_a["v"].values, ds_a["v"].values)
    np.testing.assert_array_equal(got_b["v"].values, ds_b["v"].values)


def test_data_cache_raises_cache_error_when_xarray_unavailable(monkeypatch):
    monkeypatch.setattr(cache_module, "_HAS_DATA_LIBS", False)

    with pytest.raises(CacheError):
        DataCache().cache_dataset(None, "missing-xarray")
