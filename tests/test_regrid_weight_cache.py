import numpy as np


def test_conservative_regrid_reuses_weight_matrices(monkeypatch):
    from openbench.data.regrid.methods import conservative

    conservative.clear_weight_cache()
    calls = []
    original_overlap = conservative.utils.overlap

    def counting_overlap(source_intervals, target_intervals):
        calls.append((len(source_intervals), len(target_intervals)))
        return original_overlap(source_intervals, target_intervals)

    monkeypatch.setattr(conservative.utils, "overlap", counting_overlap)

    source = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    first = conservative.get_weights(source, target)
    second = conservative.get_weights(source.copy(), target.copy())

    assert first is second
    assert calls == [(3, 5)]
    assert first.flags.writeable is False
    np.testing.assert_allclose(first.sum(axis=0), np.ones(target.size))


def test_conservative_regrid_weight_cache_can_be_disabled(monkeypatch):
    from openbench.data.regrid.methods import conservative

    conservative.clear_weight_cache()
    monkeypatch.setattr(conservative, "_WEIGHTS_CACHE_MAXSIZE", 0)
    calls = []
    original_overlap = conservative.utils.overlap

    def counting_overlap(source_intervals, target_intervals):
        calls.append((len(source_intervals), len(target_intervals)))
        return original_overlap(source_intervals, target_intervals)

    monkeypatch.setattr(conservative.utils, "overlap", counting_overlap)

    source = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 1.0, 2.0])

    conservative.get_weights(source, target)
    conservative.get_weights(source, target)

    assert calls == [(3, 3), (3, 3)]


def test_conservative_regrid_weight_cache_persists_to_disk(tmp_path, monkeypatch):
    from openbench.data.regrid.methods import conservative

    monkeypatch.setattr(conservative, "_WEIGHTS_DISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(conservative, "_WEIGHTS_CACHE_MAXSIZE", 0)
    conservative.clear_weight_cache(clear_disk=True)

    source = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    first = conservative.get_weights(source, target)
    assert list(tmp_path.glob("weights-*.npz"))

    def fail_overlap(*_args):  # pragma: no cover - should load from disk instead
        raise AssertionError("disk cache should avoid recomputing overlap")

    monkeypatch.setattr(conservative.utils, "overlap", fail_overlap)
    conservative.clear_weight_cache()
    second = conservative.get_weights(source.copy(), target.copy())

    np.testing.assert_allclose(second, first)
    assert second.flags.writeable is False


def test_spherical_correction_reuses_latitude_weight_cache(monkeypatch):
    import xarray as xr

    from openbench.data.regrid.methods import conservative

    conservative.clear_weight_cache()
    calls = []
    original_lat_weight = conservative.lat_weight

    def counting_lat_weight(latitude, latitude_res):
        calls.append((tuple(latitude), latitude_res))
        return original_lat_weight(latitude, latitude_res)

    monkeypatch.setattr(conservative, "lat_weight", counting_lat_weight)
    weights = xr.DataArray(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        dims=["lat", "target_lat"],
        coords={"lat": np.array([-0.5, 0.5]), "target_lat": np.array([-0.5, 0.5])},
    )

    first = conservative.apply_spherical_correction(weights, "lat")
    second = conservative.apply_spherical_correction(weights.copy(), "lat")

    assert len(calls) == 1
    np.testing.assert_allclose(second.values, first.values)


def test_conservative_regrid_disk_cache_prunes_by_ttl(tmp_path, monkeypatch):
    from openbench.data.regrid.methods import conservative

    old_file = tmp_path / "weights-old.npz"
    new_file = tmp_path / "weights-new.npz"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")
    import os

    os.utime(old_file, (0.0, 0.0))
    os.utime(new_file, (999.0, 999.0))

    summary = conservative.prune_weight_disk_cache(str(tmp_path), ttl_seconds=10.0, now=1000.0)

    assert not old_file.exists()
    assert new_file.exists()
    assert summary["removed_files"] == 1


def test_conservative_regrid_disk_cache_prunes_by_size(tmp_path):
    from openbench.data.regrid.methods import conservative

    old_file = tmp_path / "weights-old.npz"
    new_file = tmp_path / "weights-new.npz"
    old_file.write_bytes(b"0" * 10)
    new_file.write_bytes(b"1" * 10)
    old_mtime = 100.0
    new_mtime = 200.0
    import os

    os.utime(old_file, (old_mtime, old_mtime))
    os.utime(new_file, (new_mtime, new_mtime))

    summary = conservative.prune_weight_disk_cache(str(tmp_path), max_bytes=10, ttl_seconds=None)

    assert not old_file.exists()
    assert new_file.exists()
    assert summary["files"] == 1
    assert summary["bytes"] == 10
    assert summary["removed_files"] == 1


def test_clear_weight_cache_accepts_explicit_disk_cache_dir(tmp_path):
    from openbench.data.regrid.methods import conservative

    (tmp_path / "weights-a.npz").write_bytes(b"data")
    (tmp_path / "other.npz").write_bytes(b"keep")

    conservative.clear_weight_cache(clear_disk=True, cache_dir=str(tmp_path))

    assert not (tmp_path / "weights-a.npz").exists()
    assert (tmp_path / "other.npz").exists()


def test_regrid_validate_input_respects_custom_time_dim():
    import xarray as xr

    from openbench.data.regrid.regrid import validate_input

    data = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("t", "lat", "lon"),
        coords={"t": [0, 1], "lat": [0, 1], "lon": [10, 11]},
    )
    target = xr.Dataset(coords={"t": [0, 1], "lat": [0, 1], "lon": [10, 11]})

    result = validate_input(data, target, "t")

    assert "t" not in result.coords


def test_conservative_regrid_skipna_is_intensive_not_extensive_total():
    """Missing cells are renormalized for mean fields; totals need caller policy."""
    import xarray as xr

    import openbench.data.regrid  # noqa: F401  register accessor

    source = xr.Dataset({"flux": ("x", [1.0, np.nan])}, coords={"x": [0.5, 1.5]})
    target = xr.Dataset(coords={"x": [1.0]})

    default_result = source.regrid.conservative(target, latitude_coord=None, time_dim=None, nan_threshold=1.0)
    strict_result = source.regrid.conservative(target, latitude_coord=None, time_dim=None, nan_threshold=0.0)

    assert float(default_result["flux"].item()) == 1.0
    assert np.isnan(float(strict_result["flux"].item()))


def test_normalize_overlap_keeps_zero_overlap_columns_zero():
    from openbench.data.regrid import utils

    weights = utils.normalize_overlap(np.array([[0.0, 1.0], [0.0, 1.0]]))

    np.testing.assert_allclose(weights[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(weights[:, 1], [0.5, 0.5])


def test_conservative_regrid_masks_targets_with_no_actual_overlap(monkeypatch):
    import xarray as xr

    from openbench.data.regrid.methods import conservative

    source = xr.Dataset({"v": ("x", [10.0, 20.0])}, coords={"x": [0.0, 1.0]})
    target_coord = xr.DataArray([0.0], dims=["x"], coords={"x": [0.0]})

    def no_overlap_weights(_source_coords, _target_coords):
        return np.zeros((2, 1), dtype=float)

    monkeypatch.setattr(conservative, "get_weights", no_overlap_weights)

    result = conservative.conservative_regrid_dataset(
        source,
        coords={"x": target_coord},
        latitude_coord=None,
        skipna=False,
        nan_threshold=1.0,
        output_chunks=None,
        time_dim=None,
    )

    assert np.isnan(float(result["v"].item()))


def test_format_lon_handles_read_only_padded_coordinate_values(monkeypatch):
    """Global-lon padding must not mutate xarray's coordinate view in-place."""
    import xarray as xr

    from openbench.data.regrid.utils import format_lon

    lon = np.arange(0.5, 360.0, 1.0)
    data = xr.Dataset({"v": ("lon", np.arange(lon.size, dtype=float))}, coords={"lon": lon})
    target = xr.Dataset(coords={"lon": np.arange(-179.5, 180.0, 1.0)})

    original_pad = xr.Dataset.pad

    def readonly_pad(self, *args, **kwargs):
        padded = original_pad(self, *args, **kwargs)
        padded_lon = padded["lon"].values
        padded_lon.flags.writeable = False
        return padded

    monkeypatch.setattr(xr.Dataset, "pad", readonly_pad)
    result = format_lon(data, target, {"lon": "lon"})

    assert result["lon"].values.flags.writeable
    assert result.sizes["lon"] == 362
    assert result["lon"].values[0] == -180.5
    assert result["lon"].values[-1] == 180.5
