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


def test_same_grid_extensive_item_bypasses_area_scaling():
    processor = _processor_for_target()
    processor.item = "Inundation_Area"
    processor.remap_interpolate = lambda *_args: (_ for _ in ()).throw(AssertionError("backend called"))
    data = _grid_data()

    result = processor.remap_data(data)

    xr.testing.assert_identical(result["v"], data["v"])


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


@pytest.mark.parametrize("descending_lat", [False, True])
def test_inundation_area_regrids_as_grid_cell_total(descending_lat):
    from openbench.data._processing_grid_regrid import _rectilinear_cell_area

    processor = _processor_for_target()
    processor.item = "Inundation_Area"
    source_grid = xr.Dataset(
        coords={
            "lat": [0.25, 0.75, 1.25, 1.75],
            "lon": [10.25, 10.75, 11.25, 11.75],
        }
    )
    source_area = _rectilinear_cell_area(source_grid)
    data = (source_area * 0.25 / 1e6).rename("flood_area").to_dataset()
    data["flood_area"].attrs = {"units": "km2", "long_name": "inundated area"}
    data.attrs["source"] = "synthetic"
    if descending_lat:
        data = data.sortby("lat", ascending=False)

    result = processor.remap_data(data)

    expected = _rectilinear_cell_area(processor.create_target_grid()) * 0.25 / 1e6
    xr.testing.assert_allclose(result["flood_area"], expected.rename("flood_area"))
    np.testing.assert_allclose(result["flood_area"].sum(), data["flood_area"].sum(), rtol=1e-12)
    assert result["flood_area"].attrs == {"units": "km2", "long_name": "inundated area"}
    assert result.attrs["source"] == "synthetic"
    assert result.attrs["openbench_regrid_quantity"] == "extensive_grid_cell_total"


def test_inundation_fraction_keeps_intensive_regrid_semantics():
    processor = _processor_for_target()
    processor.item = "Inundation_Fraction"
    data = xr.DataArray(
        np.full((4, 4), 0.25),
        dims=("lat", "lon"),
        coords={
            "lat": [0.25, 0.75, 1.25, 1.75],
            "lon": [10.25, 10.75, 11.25, 11.75],
        },
        name="flood_fraction",
    ).to_dataset()

    result = processor.remap_data(data)

    np.testing.assert_allclose(result["flood_fraction"], 0.25)
    assert "openbench_regrid_quantity" not in result.attrs


def test_inundation_area_regrid_preserves_strict_nan_and_zero_cells():
    from openbench.data._processing_grid_regrid import _rectilinear_cell_area

    processor = _processor_for_target()
    processor.item = "Inundation_Area"
    source_grid = xr.Dataset(
        coords={
            "lat": [0.25, 0.75, 1.25, 1.75],
            "lon": [10.25, 10.75, 11.25, 11.75],
        }
    )
    source_totals = _rectilinear_cell_area(source_grid) * 0.25
    source_totals.values[0, 0] = np.nan
    source_totals.values[2:, 2:] = 0.0
    data = source_totals.rename("flood_area").to_dataset()

    result = processor.remap_data(data)

    assert np.isnan(result["flood_area"].sel(lat=0.5, lon=10.5))
    assert result["flood_area"].sel(lat=1.5, lon=11.5) == 0.0
    assert np.isfinite(result["flood_area"].sel(lat=0.5, lon=11.5))
    assert np.isfinite(result["flood_area"].sel(lat=1.5, lon=10.5))


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


@pytest.mark.parametrize(
    ("item", "units", "chunked"),
    [("Sensible_Heat", "W m-2", False), ("Precipitation", "mm", False), ("Sensible_Heat", "W m-2", True)],
)
def test_compare_resolution_resample_bypasses_flox_with_same_values(monkeypatch, item, units, chunked):
    flox_xarray = pytest.importorskip("flox.xarray")
    from openbench.data._processing_time_core import TimeCoreMixin

    class Processor(TimeCoreMixin):
        compare_tim_res = "ME"

    Processor.item = item
    values = np.random.default_rng(0).random((59, 2, 2))
    values[:, 0, 0] = np.nan  # cell with no valid input
    values[3, 1, 1] = np.nan  # partial gap
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2001-01-01", periods=59, freq="D")},
        attrs={"units": units},
    )
    reducer = "sum" if units == "mm" else "mean"
    expected = getattr(data.resample(time="ME"), reducer)()
    if chunked:
        data = data.chunk({"time": 10})
    use_flox = xr.get_options()["use_flox"]

    def fail_flox(*_args, **_kwargs):
        raise AssertionError("flox used for compare-resolution resample")

    monkeypatch.setattr(flox_xarray, "xarray_reduce", fail_flox)

    result = Processor()._resample_to_compare_resolution(data, "test").compute()

    assert xr.get_options()["use_flox"] == use_flox
    np.testing.assert_allclose(result.values, expected.values, rtol=1e-12, equal_nan=True)


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


@pytest.mark.parametrize("time_varying", [False, True])
def test_static_valid_mask_shortcut_matches_per_step_mask(monkeypatch, time_varying):
    from openbench.data.regrid.methods import conservative
    from openbench.data.regrid.utils import create_dot_dataarray

    source = np.array([-1.5, -0.5, 0.5, 1.5])
    target = np.array([-1.0, 1.0])
    values = np.random.default_rng(1).random((3, 4, 4))
    values[:, 0, 1] = np.nan
    if time_varying:
        values[1, 2, 2] = np.nan
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1, 2], "lat": source, "lon": source},
    )
    weights = {
        coord: create_dot_dataarray(
            conservative.get_weights(source, target, spherical=coord == "lat"), coord, target, source
        )
        for coord in ("lat", "lon")
    }

    assert (conservative._shared_valid_mask(data, ["lat", "lon"]) is None) == time_varying
    shortcut = conservative.apply_weights(data, weights, True, 0.0)
    monkeypatch.setattr(conservative, "_shared_valid_mask", lambda *_args: None)
    per_step = conservative.apply_weights(data, weights, True, 0.0)

    xr.testing.assert_identical(shortcut, per_step)


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


@pytest.mark.parametrize("fails", [False, True])
def test_process_grid_data_removes_yearly_preprocessing_scratch(tmp_path, fails):
    from openbench.data._processing_grid_core import GridProcessingCoreMixin

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    kept = scratch / "sim_case_2001.nc"
    for name in ("ref_E_2001_v4.nc", "ref_E_2002_v4.nc", kept.name):
        (scratch / name).write_bytes(b"x")

    class Processor(GridProcessingCoreMixin):
        casedir = str(tmp_path)
        minyear, maxyear = 2001, 2002

        def prepare_grid_data(self, data_params):
            if fails:
                raise RuntimeError("regrid failed")

        def remap_and_combine_data(self, data_params):
            return []

        def extract_station_data_if_needed(self, data_params, yearly_files):
            pass

    params = {"datasource": "ref", "varname": ["E"], "prefix": "E_", "suffix": "_v4"}
    if fails:
        with pytest.raises(RuntimeError, match="regrid failed"):
            Processor().process_grid_data(params)
    else:
        Processor().process_grid_data(params)

    assert sorted(path.name for path in scratch.iterdir()) == [kept.name]


def _single_value_cases():
    import cftime

    rng = np.random.default_rng(5)
    grid = rng.random((6, 2, 3)).astype("float32")
    grid[1, 0, 0], grid[2, 1, 2] = np.nan, np.inf
    noon = pd.date_range("2001-01-01T12:00", periods=6, freq="D")
    grid_da = xr.DataArray(
        grid, dims=("time", "lat", "lon"), coords={"time": noon, "lat": [0.5, 1.5]}, name="et", attrs={"units": "mm"}
    )
    grid_da["time"].attrs["axis"] = "T"
    grid_da.encoding = {"zlib": True, "dtype": "int16", "scale_factor": 0.01}
    hourly = xr.DataArray(
        rng.random(5), dims="time", coords={"time": pd.date_range("2001-01-01T00:30", periods=5, freq="h")}
    )
    monthly = xr.DataArray(
        rng.random(4),
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=4, freq="MS") + pd.Timedelta(days=14)},
    )
    yearly = xr.DataArray(
        rng.random(3), dims="time", coords={"time": pd.date_range("2001-01-01", periods=3, freq="YS")}
    )
    noleap = xr.DataArray(
        rng.random(4), dims="time", coords={"time": [cftime.DatetimeNoLeap(2001, 2, d, 12) for d in range(25, 29)]}
    )
    station = xr.DataArray(np.array([1.0, np.nan, 3.0], dtype="float32"), dims="time", coords={"time": noon[:3]})
    ints = xr.DataArray(np.arange(4), dims="time", coords={"time": noon[:4]})
    dataset = xr.Dataset({"a": grid_da, "b": grid_da * 2}, attrs={"source": "x"})
    return [
        ("grid daily noon", grid_da, "1D"),
        ("hourly", hourly, "1h"),
        ("monthly mid-month", monthly, "1ME"),
        ("yearly", yearly, "1YE"),
        ("noleap daily", noleap, "1D"),
        ("station 1-D", station, "1D"),
        ("integers", ints, "1D"),
        ("dataset", dataset, "1D"),
        ("dask", grid_da.chunk({"time": 2}), "1D"),
    ]


@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("case", range(9))
def test_single_value_period_relabel_matches_resample(case, how):
    from openbench.data._processing_time_core import _relabel_single_value_periods, _resample
    from openbench.util.groupby import without_flox

    _label, data, freq = _single_value_cases()[case]
    with without_flox():
        expected = getattr(data.resample(time=freq), how)().compute()

    assert _relabel_single_value_periods(data, freq, how) is not None
    result = _resample(data, freq, how).compute()

    xr.testing.assert_identical(result, expected)

    def encodings(obj):
        variables = obj.variables if isinstance(obj, xr.Dataset) else {None: obj.variable, **obj.coords.variables}
        return {name: dict(variable.encoding) for name, variable in variables.items()}

    assert encodings(result) == encodings(expected)


def test_single_value_relabel_falls_back_for_gaps_repeats_unsorted_and_static_vars():
    from openbench.data._processing_time_core import _relabel_single_value_periods

    t = pd.date_range("2001-01-01T12:00", periods=4, freq="D")
    da = xr.DataArray(np.arange(4.0), dims="time", coords={"time": t})
    repeated = xr.DataArray(np.arange(2.0), dims="time", coords={"time": [t[0], t[0] + pd.Timedelta(hours=6)]})
    static = xr.Dataset({"v": da, "mask": ("x", [1.0, 0.0])})

    assert _relabel_single_value_periods(da.isel(time=[0, 2, 3]), "1D", "mean") is None  # missing day
    assert _relabel_single_value_periods(repeated, "1D", "mean") is None
    assert _relabel_single_value_periods(da.isel(time=[1, 0, 2, 3]), "1D", "mean") is None
    assert _relabel_single_value_periods(static, "1D", "mean") is None
    assert _relabel_single_value_periods(da, "1ME", "mean") is None  # several days per month


@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("timezone", [None, "Europe/Berlin"])
def test_single_value_resample_preserves_timezone_and_dimension_order(how, dataset, timezone):
    from openbench.data._processing_time_core import _resample
    from openbench.util.groupby import without_flox

    times = pd.date_range("2001-03-24", periods=5, freq="D", tz=timezone)
    data = xr.DataArray(
        np.arange(10.0).reshape(2, 5),
        dims=("lat", "time"),
        coords={"time": times, "lat": [1, 2]},
        name="a",
    )
    if dataset:
        data = xr.Dataset({"a": data, "b": data.transpose("time", "lat")})
    with without_flox():
        expected = getattr(data.resample(time="1D"), how)()
    actual = _resample(data, "1D", how)
    xr.testing.assert_identical(actual, expected)
    assert actual.indexes["time"].equals(expected.indexes["time"])
    xr.testing.assert_identical(actual.sel(time="2001-03-25"), expected.sel(time="2001-03-25"))


@pytest.mark.parametrize("how", ["mean", "sum"])
def test_single_value_resample_preserves_3d_dataset_dimension_order(how):
    from openbench.data._processing_time_core import _relabel_single_value_periods, _resample
    from openbench.util.groupby import without_flox

    data = xr.DataArray(
        np.arange(30.0).reshape(2, 5, 3),
        dims=("lat", "time", "lon"),
        coords={"time": pd.date_range("2001-01-01", periods=5), "lat": [1, 2], "lon": [3, 4, 5]},
    )
    dataset = xr.Dataset({"middle": data, "last": data.transpose("lat", "lon", "time")})
    with without_flox():
        expected = getattr(dataset.resample(time="1D"), how)()

    assert _relabel_single_value_periods(dataset, "1D", how) is not None
    xr.testing.assert_identical(_resample(dataset, "1D", how), expected)
