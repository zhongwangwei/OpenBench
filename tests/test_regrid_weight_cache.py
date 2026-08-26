import numpy as np
import pytest


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


def test_latitude_correction_uses_nonuniform_cell_bounds_and_keeps_poles_finite():
    from openbench.data.regrid.methods import conservative

    latitude = np.array([0.0, 30.0, 80.0, 90.0])
    weights = conservative.lat_weight(latitude, np.median(np.diff(latitude)))
    bounds = np.array([-15.0, 15.0, 55.0, 85.0, 90.0])
    widths = np.radians(np.diff(bounds))
    expected = np.diff(np.sin(np.radians(bounds))) / widths

    np.testing.assert_allclose(weights, expected)
    assert np.isfinite(weights).all()
    assert weights[-1] > 0


def test_spherical_correction_handles_single_latitude_without_warning():
    import warnings

    import xarray as xr

    from openbench.data.regrid.methods import conservative

    weights = xr.DataArray(
        np.array([[1.0]]),
        dims=["lat", "target_lat"],
        coords={"lat": [90.0], "target_lat": [90.0]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        corrected = conservative.apply_spherical_correction(weights, "lat")

    np.testing.assert_allclose(corrected.values, [[1.0]])


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
    target = xr.Dataset(coords={"x": [1.0, 3.0]})

    default_result = source.regrid.conservative(target, latitude_coord=None, time_dim=None, nan_threshold=1.0)
    strict_result = source.regrid.conservative(target, latitude_coord=None, time_dim=None, nan_threshold=0.0)

    assert float(default_result["flux"].isel(x=0).item()) == 1.0
    assert np.isnan(float(strict_result["flux"].isel(x=0).item()))


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

    padded_lons = []

    def readonly_pad_with_capture(self, *args, **kwargs):
        padded = original_pad(self, *args, **kwargs)
        padded_lon = padded["lon"].values
        padded_lon.flags.writeable = False
        padded_lons.append(padded_lon)
        return padded

    monkeypatch.setattr(xr.Dataset, "pad", readonly_pad_with_capture)
    result = format_lon(data, target, {"lon": "lon"})

    # The portable contract is that format_lon does not mutate xarray's
    # read-only padded coordinate view in place.  Some xarray/pandas versions
    # expose dimension-coordinate index values as read-only after assignment,
    # so do not assert the returned index buffer's private writeability flag.
    assert padded_lons and not padded_lons[0].flags.writeable
    assert not np.shares_memory(result["lon"].values, padded_lons[0])
    assert result.sizes["lon"] == 362
    assert result["lon"].values[0] == -180.5
    assert result["lon"].values[-1] == 180.5


def test_regridding_grid_includes_binary_decimal_endpoint():
    from openbench.data.regrid import Grid

    grid = Grid(north=89.9, south=-89.9, west=0.1, east=359.9, resolution_lat=0.2, resolution_lon=0.2)
    ds = grid.create_regridding_dataset()

    assert ds.sizes["lat"] == 900
    assert ds.sizes["lon"] == 1800
    np.testing.assert_allclose(ds["lat"].values[[0, -1]], [-89.9, 89.9])
    np.testing.assert_allclose(ds["lon"].values[[0, -1]], [0.1, 359.9])


def test_latitude_weights_use_actual_spherical_overlap():
    from openbench.data.regrid.methods import conservative

    source = np.array([-60.0, 0.0, 60.0])
    target = np.array([-30.0, 30.0])
    weights = conservative.get_weights(source, target, spherical=True)

    # Target -30 spans [-60, 0] and source cells are [-90,-30], [-30,30], [30,90].
    south_overlap = np.sin(np.radians(-30.0)) - np.sin(np.radians(-60.0))
    equator_overlap = np.sin(np.radians(0.0)) - np.sin(np.radians(-30.0))
    expected_first = np.array([south_overlap, equator_overlap, 0.0]) / (south_overlap + equator_overlap)
    np.testing.assert_allclose(weights[:, 0], expected_first)
    np.testing.assert_allclose(weights.sum(axis=0), np.ones(target.size))


def test_overlap_matches_dense_reference():
    import pandas as pd

    from openbench.data.regrid import utils

    source = pd.IntervalIndex.from_tuples([(-3.0, -1.0), (-1.0, 2.0), (2.0, 5.0)])
    target = pd.IntervalIndex.from_tuples([(-2.0, 0.0), (0.0, 4.0)])
    expected = np.maximum(
        np.minimum(source.right.to_numpy(), target.right.to_numpy()[:, None])
        - np.maximum(source.left.to_numpy(), target.left.to_numpy()[:, None]),
        0,
    ).T

    np.testing.assert_allclose(utils.overlap(source, target), expected)


def test_conservative_regrid_preserves_single_identical_cell():
    import xarray as xr

    import openbench.data.regrid  # noqa: F401  register accessor

    source = xr.Dataset({"v": (("lat", "lon"), [[7.0]])}, coords={"lat": [0.0], "lon": [0.0]})
    target = xr.Dataset(coords={"lat": [0.0], "lon": [0.0]})

    result = source.regrid.conservative(target, latitude_coord="lat", time_dim=None)

    assert float(result["v"].item()) == 7.0


def test_conservative_regrid_rejects_nonidentical_single_points():
    from openbench.data.regrid.methods import conservative

    with pytest.raises(ValueError, match="cannot infer finite cell bounds"):
        conservative.get_weights(np.array([0.0]), np.array([1.0]))


@pytest.mark.parametrize(
    ("source", "target"),
    [([0.0], [-1.0, 1.0]), ([-1.0, 1.0], [0.0])],
)
def test_conservative_regrid_rejects_one_sided_single_points(source, target):
    from openbench.data.regrid.methods import conservative

    with pytest.raises(ValueError, match="explicit cell bounds"):
        conservative.get_weights(np.asarray(source), np.asarray(target))


def test_weight_disk_cache_rejects_missing_schema_version(tmp_path, monkeypatch):
    from openbench.data.regrid.methods import conservative

    monkeypatch.setattr(conservative, "_WEIGHTS_DISK_CACHE_DIR", str(tmp_path))
    conservative.clear_weight_cache(clear_disk=True)
    source = np.array([0.0, 1.0])
    target = np.array([0.0, 1.0])
    key = (
        conservative.REGRID_WEIGHT_SCHEMA_VERSION,
        "linear",
        conservative._coord_cache_token(source),
        conservative._coord_cache_token(target),
    )
    path = conservative._weights_disk_cache_path(key)
    assert path is not None
    np.savez_compressed(path, weights=np.ones((2, 2)))

    assert conservative._load_weights_from_disk(key) is None


def test_xesmf_weight_cache_reuses_existing_file(tmp_path, monkeypatch):
    import xarray as xr

    from openbench.data.regrid import xesmf_cache

    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "test", "ESMF": "test", "esmpy": None})
    calls = []

    class FakeRegridder:
        def __init__(self, _source, _target, method, *, periodic=False, weights=None):
            calls.append({"method": method, "periodic": periodic, "weights": weights})
            self.weights = weights

        def to_netcdf(self, filename):
            assert self.weights is None
            with open(filename, "wb") as handle:
                handle.write(b"weights")

    xe = type("FakeXe", (), {"Regridder": FakeRegridder})
    source = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})
    target = xr.Dataset(coords={"lat": [0.0, 0.5, 1.0], "lon": [10.0, 10.5, 11.0]})

    xesmf_cache.cached_regridder(xe, source, target, "conservative", cache_dir=tmp_path, periodic=False)
    files = list(tmp_path.glob("xesmf-weights-*.nc"))
    assert len(files) == 1

    xesmf_cache.cached_regridder(xe, source.copy(), target.copy(), "conservative", cache_dir=tmp_path, periodic=False)

    assert calls == [
        {"method": "conservative", "periodic": False, "weights": None},
        {"method": "conservative", "periodic": False, "weights": str(files[0])},
    ]


def test_xesmf_weight_cache_key_includes_mask_bounds_method_and_versions(tmp_path, monkeypatch):
    import xarray as xr

    from openbench.data.regrid import xesmf_cache

    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "1", "ESMF": "1", "esmpy": None})
    source = xr.Dataset(
        {
            "lat_vertices": ("lat_vertices", [-0.5, 0.5, 1.5]),
            "lon_vertices": ("lon_vertices", [9.5, 10.5, 11.5]),
            "mask": (("lat", "lon"), [[1, 1], [1, 0]]),
        },
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )
    source["lat"].attrs["bounds"] = "lat_vertices"
    source["lon"].attrs["bounds"] = "lon_vertices"
    target = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})

    base = xesmf_cache._cache_path(source, target, "conservative", cache_dir=tmp_path, periodic=False)
    changed_mask = source.copy(deep=True)
    changed_mask["mask"].values[0, 0] = 0
    changed_bounds = source.copy(deep=True)
    changed_bounds["lat_vertices"].values[0] = -1.0

    assert xesmf_cache._cache_path(changed_mask, target, "conservative", cache_dir=tmp_path, periodic=False) != base
    assert xesmf_cache._cache_path(changed_bounds, target, "conservative", cache_dir=tmp_path, periodic=False) != base
    assert xesmf_cache._cache_path(source, target, "bilinear", cache_dir=tmp_path, periodic=False) != base
    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "2", "ESMF": "1", "esmpy": None})
    assert xesmf_cache._cache_path(source, target, "conservative", cache_dir=tmp_path, periodic=False) != base


def test_xesmf_weight_cache_key_recognizes_cf_units(tmp_path, monkeypatch):
    import xarray as xr

    from openbench.data.regrid import xesmf_cache

    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "1", "ESMF": "1", "esmpy": None})
    source = xr.Dataset(
        coords={
            "xc": (("y", "x"), [[10.0, 11.0], [10.0, 11.0]], {"units": "degrees_east"}),
            "yc": (("y", "x"), [[0.0, 0.0], [1.0, 1.0]], {"units": "degrees_north"}),
        }
    )
    changed = source.copy(deep=True)
    changed["xc"].values[0, 0] = 9.0

    assert xesmf_cache._cache_path(
        source, source, "conservative", cache_dir=tmp_path, periodic=False
    ) != xesmf_cache._cache_path(changed, source, "conservative", cache_dir=tmp_path, periodic=False)


def test_processing_xesmf_uses_case_weight_cache(tmp_path, monkeypatch):
    import sys
    import types

    import xarray as xr

    from openbench.data._processing_grid_regrid import GridRegridMixin
    from openbench.data.regrid import xesmf_cache

    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "test", "ESMF": "test", "esmpy": None})
    calls = []

    class FakeRegridder:
        def __init__(self, _source, _target, _method, *, periodic=False, weights=None):
            calls.append((periodic, weights))
            self.weights = weights

        def to_netcdf(self, filename):
            with open(filename, "wb") as handle:
                handle.write(b"weights")

        def __call__(self, data):
            return data

    class Processor(GridRegridMixin):
        casedir = str(tmp_path)

    monkeypatch.setitem(sys.modules, "xesmf", types.SimpleNamespace(Regridder=FakeRegridder))
    data = xr.Dataset(
        {"value": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )
    target = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})

    Processor().remap_xesmf(data, target)
    Processor().remap_xesmf(data, target)

    files = list((tmp_path / "scratch" / "xesmf_weights").glob("xesmf-weights-*.nc"))
    assert len(files) == 1
    assert calls == [(False, None), (False, str(files[0]))]


def test_xesmf_weight_cache_ignores_legacy_conservative_env(tmp_path, monkeypatch):
    from openbench.data.regrid.xesmf_cache import default_weight_cache_dir

    monkeypatch.delenv("OPENBENCH_XESMF_WEIGHT_CACHE_DIR", raising=False)
    monkeypatch.setenv("OPENBENCH_REGRID_WEIGHT_CACHE_DIR", str(tmp_path / "legacy"))

    assert default_weight_cache_dir() is None


def test_convert_to_wgs84_xesmf_accepts_explicit_cache_dir(tmp_path, monkeypatch):
    import sys
    import types

    import xarray as xr

    from openbench.data.regrid import xesmf_cache
    from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_xesmf

    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "test", "ESMF": "test", "esmpy": None})
    calls = []

    class FakeRegridder:
        def __init__(self, _source, target_grid, _method, *, periodic=False, weights=None):
            calls.append((periodic, weights))
            self.target_grid = target_grid
            self.weights = weights

        def to_netcdf(self, filename):
            with open(filename, "wb") as handle:
                handle.write(b"weights")

        def __call__(self, _data):
            return xr.DataArray(
                np.zeros((self.target_grid.sizes["lat"], self.target_grid.sizes["lon"])),
                dims=("lat", "lon"),
            )

    monkeypatch.setitem(sys.modules, "xesmf", types.SimpleNamespace(Regridder=FakeRegridder))
    ds = xr.Dataset(
        {"flux": (("y", "x"), np.ones((2, 2)))},
        coords={
            "lat": (("y", "x"), np.array([[0.0, 0.0], [1.0, 1.0]])),
            "lon": (("y", "x"), np.array([[10.0, 11.0], [10.0, 11.0]])),
        },
    )

    convert_to_wgs84_xesmf(ds, resolution=1.0, cache_dir=str(tmp_path))
    convert_to_wgs84_xesmf(ds, resolution=1.0, cache_dir=str(tmp_path))

    files = list(tmp_path.glob("xesmf-weights-*.nc"))
    assert len(files) == 1
    assert calls == [(False, None), (False, str(files[0]))]


def test_core_curvilinear_preprocess_passes_case_local_xesmf_cache(tmp_path, monkeypatch):
    import xarray as xr

    from openbench.data._processing_grid_core import GridProcessingCoreMixin

    seen = []

    def fake_convert(data, resolution, *, cache_dir=None):
        seen.append((resolution, cache_dir))
        return xr.Dataset(
            {"flux": (("lat", "lon"), np.ones((2, 2)))},
            coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
        )

    class Processor(GridProcessingCoreMixin):
        casedir = str(tmp_path)
        compare_grid_res = 1.0

        def check_coordinate(self, data):
            return data

        def _normalize_longitude_axis(self, data):
            return data

    monkeypatch.setattr("openbench.data.regrid.regrid_wgs84.convert_to_wgs84_xesmf", fake_convert)
    data = xr.Dataset(
        {"flux": (("y", "x"), np.ones((2, 2)))},
        coords={
            "lat": (("y", "x"), np.array([[0.0, 0.0], [1.0, 1.0]])),
            "lon": (("y", "x"), np.array([[10.0, 11.0], [10.0, 11.0]])),
        },
    )

    Processor().preprocess_grid_data(data)

    assert seen == [(1.0, str(tmp_path / "scratch" / "xesmf_weights"))]


def test_statistics_xesmf_uses_output_dir_weight_cache(tmp_path, monkeypatch):
    import sys
    import types

    import xarray as xr

    from openbench.core.statistics.Mod_Statistics import BasicProcessing, Convert_Type
    from openbench.data.regrid import xesmf_cache

    monkeypatch.setattr(xesmf_cache, "_versions", lambda: {"xesmf": "test", "ESMF": "test", "esmpy": None})
    calls = []

    class FakeRegridder:
        def __init__(self, _source, _target, _method, *, periodic=False, weights=None):
            calls.append((periodic, weights))
            self.weights = weights

        def to_netcdf(self, filename):
            with open(filename, "wb") as handle:
                handle.write(b"weights")

        def __call__(self, data):
            return data

    monkeypatch.setitem(sys.modules, "xesmf", types.SimpleNamespace(Regridder=FakeRegridder))
    monkeypatch.setattr(Convert_Type, "convert_nc", staticmethod(lambda value: value))
    processor = object.__new__(BasicProcessing)
    processor.output_dir = str(tmp_path / "out")
    data = xr.Dataset(
        {"value": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )
    target = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})

    BasicProcessing.remap_xesmf(processor, data, target)
    BasicProcessing.remap_xesmf(processor, data, target)

    files = list((tmp_path / "out" / "xesmf_weights").glob("xesmf-weights-*.nc"))
    assert len(files) == 1
    assert calls == [(False, None), (False, str(files[0]))]
