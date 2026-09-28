from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _processor_for_target():
    from openbench.data._processing_grid_regrid import GridRegridMixin

    class Processor(GridRegridMixin):
        min_lat = 0.0
        max_lat = 2.0
        min_lon = 10.0
        max_lon = 12.0
        compare_grid_res = 1.0
        regrid_backend = "openbench_conservative"

    return Processor()


def _grid_data(lat=None, lon=None):
    lat = np.array([0.5, 1.5]) if lat is None else np.asarray(lat)
    lon = np.array([10.5, 11.5]) if lon is None else np.asarray(lon)
    return xr.Dataset(
        {"v": (("lat", "lon"), np.array([[1.0, np.nan], [3.0, 4.0]]))},
        coords={"lat": lat, "lon": lon},
        attrs={"source": "kept"},
    )


def test_same_grid_bypasses_backend_and_preserves_values_and_attrs(caplog):
    processor = _processor_for_target()
    processor.remap_interpolate = lambda *_args: (_ for _ in ()).throw(AssertionError("backend called"))

    with caplog.at_level(logging.INFO):
        result = processor.remap_data(_grid_data())

    np.testing.assert_allclose(result["v"], _grid_data()["v"], equal_nan=True)
    assert result.attrs["source"] == "kept"
    assert result.attrs["openbench_regrid_backend"] == "openbench_conservative"
    assert "Skip regridding: source grid already matches target grid" in caplog.text


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ([0.4, 1.4], [10.5, 11.5]),  # same shape, different positions
        ([1.5, 0.5], [10.5, 11.5]),  # reversed latitude
        ([0.5, 1.5], [10.4, 11.4]),  # different longitude
    ],
)
def test_same_grid_does_not_bypass_different_order_or_positions(lat, lon):
    processor = _processor_for_target()
    called = []
    processor.remap_interpolate = lambda data, _grid: called.append(True) or data

    processor.remap_data(_grid_data(lat, lon))

    assert called == [True]


def test_same_grid_allows_small_coordinate_roundoff():
    processor = _processor_for_target()
    processor.remap_interpolate = lambda *_args: (_ for _ in ()).throw(AssertionError("backend called"))
    data = _grid_data(lat=[0.5 + 1e-9, 1.5 - 1e-9], lon=[10.5 - 1e-8, 11.5 + 1e-8])

    result = processor.remap_data(data)

    np.testing.assert_array_equal(result["lat"], [0.5, 1.5])
    np.testing.assert_array_equal(result["lon"], [10.5, 11.5])


def test_same_grid_compares_longitudes_after_normalization():
    processor = _processor_for_target()
    processor.min_lon = -170.0
    processor.max_lon = -168.0
    processor.remap_interpolate = lambda *_args: (_ for _ in ()).throw(AssertionError("backend called"))
    data = _grid_data(lon=[190.5, 191.5])

    result = processor.remap_data(data)

    np.testing.assert_array_equal(result["lon"], [-169.5, -168.5])


@pytest.mark.parametrize(
    ("source_freq", "periods", "target_freq", "expected"),
    [("h", 48, "D", 2), ("D", 59, "ME", 2)],
)
def test_temporal_downsampling_runs_once(source_freq, periods, target_freq, expected, tmp_path):
    from openbench.data._processing_time_core import TimeCoreMixin

    class Processor(TimeCoreMixin):
        item = "Sensible_Heat"
        compare_tim_res = target_freq

    data = xr.DataArray(
        np.arange(periods, dtype=float),
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=periods, freq=source_freq)},
        attrs={"units": "W m-2"},
    )
    first = Processor()._resample_to_compare_resolution(data, "first")
    second = Processor()._resample_to_compare_resolution(first, "second")

    assert first.sizes["time"] == expected
    assert second is first

    scratch = tmp_path / "scratch.nc"
    first.to_dataset(name="value").to_netcdf(scratch)
    with xr.open_dataset(scratch) as reopened:
        assert Processor()._resample_to_compare_resolution(reopened, "scratch") is reopened


def test_same_frequency_input_is_resampled_to_shared_time_labels():
    from openbench.data._processing_time_core import TimeCoreMixin

    class Processor(TimeCoreMixin):
        item = "Sensible_Heat"
        compare_tim_res = "ME"

    # Labels as normalized by time integrity checks: daily at 12:00, monthly on the 15th.
    daily = xr.DataArray(
        np.arange(365, dtype=float),
        dims="time",
        coords={"time": pd.date_range("2001-01-01T12:00", periods=365, freq="D")},
        attrs={"units": "W m-2"},
    )
    monthly = xr.DataArray(
        np.arange(12, dtype=float),
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=12, freq="MS") + pd.Timedelta(days=14)},
        attrs={"units": "W m-2"},
    )

    sim = Processor()._resample_to_compare_resolution(daily, "sim")
    ref = Processor()._resample_to_compare_resolution(monthly, "ref")

    np.testing.assert_array_equal(sim["time"].values, ref["time"].values)
    np.testing.assert_allclose(ref.values, monthly.values)


def test_regrid_worker_budget_is_bounded_and_small_workloads_parallelize():
    from openbench.data._processing_grid_core import _regrid_worker_budget

    workers, _reason = _regrid_worker_budget(
        requested=96,
        year_count=20,
        available_memory_gb=64,
        source_shape=(20, 20),
        target_shape=(10, 10),
        time_length=12,
        data_bytes=20 * 20 * 12 * 4,
        backend="openbench_conservative",
    )

    assert workers == 4


def test_regrid_worker_budget_limits_large_or_memory_heavy_workloads():
    from openbench.data._processing_grid_core import _regrid_worker_budget

    large_workers, _ = _regrid_worker_budget(
        requested=48,
        year_count=30,
        available_memory_gb=128,
        source_shape=(1800, 3600),
        target_shape=(720, 1440),
        time_length=12,
        data_bytes=1800 * 3600 * 12 * 4,
        backend="openbench_conservative",
    )
    memory_workers, _ = _regrid_worker_budget(
        requested=8,
        year_count=8,
        available_memory_gb=1,
        source_shape=(720, 1440),
        target_shape=(360, 720),
        time_length=365,
        data_bytes=720 * 1440 * 365 * 4,
        backend="openbench_conservative",
    )

    assert large_workers <= 2
    assert memory_workers == 1


def test_regrid_worker_budget_respects_requested_cores_and_years():
    from openbench.data._processing_grid_core import _regrid_worker_budget

    common = dict(
        available_memory_gb=64,
        source_shape=(20, 20),
        target_shape=(10, 10),
        time_length=1,
        data_bytes=1600,
        backend="xesmf_conservative",
    )
    one_worker, _ = _regrid_worker_budget(requested=1, year_count=10, **common)
    two_years, _ = _regrid_worker_budget(requested=8, year_count=2, **common)

    assert one_worker == 1
    assert two_years == 2


@pytest.mark.parametrize("nan_threshold", [0.0, 1.0])
def test_sparse_conservative_matches_dense_with_nan_and_multiple_times(nan_threshold):
    from openbench.data.regrid.methods import conservative
    from openbench.data.regrid.utils import create_dot_dataarray

    source_lat = np.array([-1.5, -0.5, 0.5, 1.5])
    source_lon = np.array([10.5, 11.5, 12.5, 13.5])
    target_lat = np.array([-1.0, 1.0])
    target_lon = np.array([11.0, 13.0])
    values = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    values[0, 1, 1] = np.nan
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": source_lat, "lon": source_lon},
    )
    weights = {
        "lat": create_dot_dataarray(
            conservative.get_weights(source_lat, target_lat, spherical=True),
            "lat",
            target_lat,
            source_lat,
        ),
        "lon": create_dot_dataarray(
            conservative.get_weights(source_lon, target_lon),
            "lon",
            target_lon,
            source_lon,
        ),
    }

    dense = conservative.apply_weights(data, weights, True, nan_threshold, use_sparse=False)
    optimized = conservative.apply_weights(data, weights, True, nan_threshold, use_sparse=True)

    xr.testing.assert_allclose(optimized, dense)


def test_sparse_conservative_falls_back_to_dense(caplog, monkeypatch):
    from openbench.data.regrid.methods import conservative

    data = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0.0, 1.0]})
    weight = xr.DataArray(
        np.eye(2),
        dims=("x", "target_x"),
        coords={"x": [0.0, 1.0], "target_x": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        conservative,
        "_apply_weights_sparse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(NotImplementedError("unsupported")),
    )

    with caplog.at_level(logging.DEBUG):
        result = conservative.apply_weights(data, {"x": weight}, True, 1.0, use_sparse=True)

    np.testing.assert_allclose(result, data)
    assert "using dense xr.dot" in caplog.text


def test_dask_conservative_keeps_dense_compatibility_path(monkeypatch):
    pytest.importorskip("dask.array")
    from openbench.data.regrid.methods import conservative

    data = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0.0, 1.0]}).chunk({"x": 1})
    weight = xr.DataArray(
        np.eye(2),
        dims=("x", "target_x"),
        coords={"x": [0.0, 1.0], "target_x": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        conservative,
        "_apply_weights_sparse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sparse path used")),
    )

    result = conservative.apply_weights(data, {"x": weight}, True, 1.0, use_sparse=True).compute()

    np.testing.assert_allclose(result, [1.0, 2.0])


def test_weight_cache_activity_reports_miss_then_memory_hit(monkeypatch):
    from openbench.data.regrid.methods import conservative

    monkeypatch.delenv("OPENBENCH_REGRID_WEIGHT_CACHE_DIR", raising=False)
    monkeypatch.setattr(conservative, "_WEIGHTS_DISK_CACHE_DIR", None)
    conservative.clear_weight_cache()
    conservative.reset_weight_cache_activity()
    conservative.get_weights(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert conservative.consume_weight_cache_activity() == "miss"

    conservative.reset_weight_cache_activity()
    conservative.get_weights(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert conservative.consume_weight_cache_activity() == "memory"
