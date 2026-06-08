from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr


def _chunked_grid(values):
    data = da.from_array(np.asarray(values, dtype=float), chunks=(1, 2, 2))
    return xr.DataArray(
        data,
        coords={
            "time": [0, 1],
            "lat": [10.0, 11.0],
            "lon": [20.0, 21.0],
        },
        dims=("time", "lat", "lon"),
    )


def test_pairwise_valid_mask_preserves_dask_lazy_arrays():
    from openbench.core.evaluation import _apply_pairwise_valid_mask

    sim = _chunked_grid(
        [
            [[1.0, np.nan], [3.0, 4.0]],
            [[5.0, 6.0], [np.nan, 8.0]],
        ]
    )
    ref = _chunked_grid(
        [
            [[1.0, 2.0], [np.nan, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
        ]
    )

    masked_sim, masked_ref = _apply_pairwise_valid_mask(sim, ref)

    assert masked_sim.chunks is not None
    assert masked_ref.chunks is not None
    assert hasattr(masked_sim.data, "compute")
    assert hasattr(masked_ref.data, "compute")

    expected = np.array(
        [
            [[1.0, np.nan], [np.nan, 4.0]],
            [[5.0, np.nan], [np.nan, 8.0]],
        ]
    )
    np.testing.assert_allclose(masked_sim.compute().values, expected)
    np.testing.assert_allclose(masked_ref.compute().values, expected)


def test_tws_change_order_remains_shift_then_lazy_mask():
    from openbench.core.evaluation import _apply_pairwise_valid_mask

    sim = xr.DataArray(
        da.from_array(np.array([[[1.0]], [[3.0]], [[6.0]]]), chunks=(1, 1, 1)),
        coords={"time": [0, 1, 2], "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    ref = xr.DataArray(
        da.from_array(np.array([[[10.0]], [[20.0]], [[30.0]]]), chunks=(1, 1, 1)),
        coords=sim.coords,
        dims=sim.dims,
    )

    # Mirrors evaluation.py's Terrestrial_Water_Storage_Change order:
    # derive the shifted difference first, then apply the shared valid mask.
    shifted_sim = sim - sim.shift(time=1)
    masked_sim, masked_ref = _apply_pairwise_valid_mask(shifted_sim, ref)

    assert masked_sim.chunks is not None
    np.testing.assert_allclose(
        masked_sim.compute().values,
        np.array([[[np.nan]], [[2.0]], [[3.0]]]),
    )
    np.testing.assert_allclose(
        masked_ref.compute().values,
        np.array([[[np.nan]], [[20.0]], [[30.0]]]),
    )


def test_valid_pair_check_detects_all_nan_overlap():
    from openbench.core.evaluation import _has_any_valid_pair

    sim = xr.DataArray([1.0, np.nan], dims=("time",))
    ref = xr.DataArray([np.nan, 2.0], dims=("time",))

    assert _has_any_valid_pair(sim, ref) is False


def test_valid_pair_check_accepts_at_least_one_shared_finite_value():
    from openbench.core.evaluation import _has_any_valid_pair

    sim = xr.DataArray(da.from_array(np.array([np.nan, 3.0]), chunks=(1,)), dims=("time",))
    ref = xr.DataArray(da.from_array(np.array([1.0, 4.0]), chunks=(1,)), dims=("time",))

    assert _has_any_valid_pair(sim, ref) is True


def test_evaluation_no_longer_materializes_masks_with_values_copy():
    source = Path("src/openbench/core/evaluation.py").read_text(encoding="utf-8")

    assert ".values.copy()" not in source
    assert "_apply_pairwise_valid_mask(s, o)" in source


def test_evaluation_side_effect_methods_are_not_cached():
    source = Path("src/openbench/core/evaluation.py").read_text(encoding="utf-8")

    assert "@cached" not in source
    assert 's["time"] = o["time"]' not in source


def test_processing_uses_chunked_open_mfdataset_wrapper():
    processing_source = Path("src/openbench/data/processing.py").read_text(encoding="utf-8")
    selection_source = Path("src/openbench/data/_processing_selection.py").read_text(encoding="utf-8")
    grid_source = Path("src/openbench/data/_processing_grid.py").read_text(encoding="utf-8")
    combined_source = processing_source + selection_source + grid_source

    assert "src_ds = xr.open_mfdataset" not in processing_source
    assert "with xr.open_mfdataset" not in processing_source
    assert 'open_mfdataset_chunked(VarFile, combine="by_coords")' in selection_source
    assert "write_mfdataset_chunked_atomic(" in grid_source
    assert "OPENBENCH_MFDATASET_BATCH_SIZE" not in combined_source


def test_remaining_mfdataset_entrypoints_use_dataset_loader_wrappers():
    station_source = Path("src/openbench/data/station_scanner.py").read_text(encoding="utf-8")
    cli_data_source = Path("src/openbench/cli/data.py").read_text(encoding="utf-8")
    cli_optimize_source = Path("src/openbench/cli/_optimize.py").read_text(encoding="utf-8")

    assert "xr.open_mfdataset(" not in station_source
    assert "_open_mfdataset_chunked(" in station_source
    assert "xr.open_mfdataset(" not in cli_data_source
    assert "write_mfdataset_zarr(" in cli_optimize_source


def test_run_cli_help_mentions_performance_configuration():
    from click.testing import CliRunner

    from openbench.cli.main import cli

    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "project.io" in result.output
    assert "project.dask" in result.output
    assert "OPENBENCH_MFDATASET_BATCH_SIZE" in result.output


def test_metric_worker_count_honors_configured_cores(monkeypatch):
    import openbench.core.evaluation as evaluation

    assert evaluation._metric_worker_count(1, 6) == 1
    assert evaluation._metric_worker_count(2, 6) == 2
    assert evaluation._metric_worker_count(16, 6) == 6
    assert evaluation._metric_worker_count(None, 1) == 1
    monkeypatch.setattr(evaluation.os, "cpu_count", lambda: 8)
    assert evaluation._metric_worker_count(0, 6) == 6


def test_metric_parallelism_no_longer_uses_hard_coded_worker_cap():
    source = Path("src/openbench/core/evaluation.py").read_text(encoding="utf-8")

    assert "len(self.metrics) > 3" not in source
    assert "max_workers=min(4, len(self.metrics))" not in source
    assert "max_workers=metric_workers" in source
    assert '_metric_worker_count(getattr(self, "num_cores", 1), len(self.metrics))' in source


def test_pc_ampli_builds_dask_graph_without_eager_compute():
    from dask.callbacks import Callback

    from openbench.core.metrics import metrics

    metric = metrics()
    sim = _chunked_grid(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [5.0, 8.0]],
        ]
    )
    ref = _chunked_grid(
        [
            [[1.0, 1.0], [3.0, 4.0]],
            [[3.0, 1.0], [7.0, 10.0]],
        ]
    )
    compute_count = [0]

    class CountCompute(Callback):
        def _start(self, dsk):
            compute_count[0] += 1

    with CountCompute():
        result = metric.pc_ampli(sim, ref)

    assert compute_count == [0]
    assert result.chunks is not None
    expected = np.array([[-0.5, np.nan], [-0.5, -1.0 / 3.0]])
    np.testing.assert_allclose(result.compute().values, expected)


def test_kappa_coeff_accepts_multi_chunk_time_dask_arrays():
    from openbench.core.metrics import metrics

    metric = metrics()
    sim_values = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 2], [2, 2]],
            [[2, 2], [1, 1]],
            [[2, 2], [1, 2]],
        ],
        dtype=float,
    )
    obs_values = np.array(
        [
            [[1, 1], [2, 1]],
            [[1, 1], [2, 1]],
            [[2, 2], [1, 2]],
            [[2, 2], [1, 2]],
        ],
        dtype=float,
    )
    sim = xr.DataArray(
        da.from_array(sim_values, chunks=(2, 1, 2)),
        coords={"time": [0, 1, 2, 3], "lat": [10.0, 11.0], "lon": [20.0, 21.0]},
        dims=("time", "lat", "lon"),
    )
    obs = xr.DataArray(da.from_array(obs_values, chunks=(2, 1, 2)), coords=sim.coords, dims=sim.dims)

    result = metric.kappa_coeff(sim, obs)

    assert result.chunks is not None
    expected = metric.kappa_coeff(
        xr.DataArray(sim_values, coords=sim.coords, dims=sim.dims),
        xr.DataArray(obs_values, coords=sim.coords, dims=sim.dims),
    )
    np.testing.assert_allclose(result.compute().values, expected.values, equal_nan=True)


def test_smpi_dask_bootstrap_does_not_compute_during_graph_construction():
    from dask.callbacks import Callback

    from openbench.core.metrics import metrics

    metric = metrics()
    sim = _chunked_grid(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    ref = _chunked_grid(
        [
            [[1.0, 1.0], [3.0, 3.0]],
            [[3.0, 2.0], [5.0, 4.0]],
        ]
    )
    compute_count = [0]

    class CountCompute(Callback):
        def _start(self, dsk):
            compute_count[0] += 1

    with CountCompute():
        value, lower, upper = metric.smpi(sim, ref, n_bootstrap=4, seed=42)

    assert compute_count == [0]
    assert hasattr(value.data, "compute")
    assert hasattr(lower.data, "compute")
    assert hasattr(upper.data, "compute")
    assert np.isfinite(float(value.compute()))
    assert np.isfinite(float(lower.compute())) or np.isnan(float(lower.compute()))
    assert np.isfinite(float(upper.compute())) or np.isnan(float(upper.compute()))


def test_apfb_grid_dask_uses_apply_ufunc_without_eager_compute():
    from dask.callbacks import Callback

    from openbench.core.metrics import metrics

    metric = metrics()
    coords = {
        "time": np.array(["2000-01-01", "2000-02-01", "2001-01-01", "2001-02-01"], dtype="datetime64[ns]"),
        "lat": [10.0, 11.0],
        "lon": [20.0, 21.0],
    }
    sim = xr.DataArray(
        da.from_array(np.full((4, 2, 2), 2.0), chunks=(2, 1, 2)),
        coords=coords,
        dims=("time", "lat", "lon"),
    )
    ref = xr.DataArray(
        da.from_array(np.ones((4, 2, 2)), chunks=(2, 1, 2)),
        coords=coords,
        dims=("time", "lat", "lon"),
    )
    compute_count = [0]

    class CountCompute(Callback):
        def _start(self, dsk):
            compute_count[0] += 1

    with CountCompute():
        result = metric.APFB(sim, ref)

    assert compute_count == [0]
    assert result.chunks is not None
    np.testing.assert_allclose(result.compute().values, np.ones((2, 2)))


def test_write_mfdataset_atomic_batches_large_file_lists(tmp_path, monkeypatch):
    from openbench.util.dataset_loader import write_mfdataset_atomic

    input_files = []
    for index, value in enumerate([3.0, 1.0, 2.0]):
        path = tmp_path / f"part-{index}.nc"
        xr.Dataset(
            {"value": ("time", np.array([value], dtype=np.float32))},
            coords={"time": np.array([index])},
        ).to_netcdf(path)
        input_files.append(str(path))

    output = tmp_path / "combined.nc"
    batch_dir = tmp_path / "batches"
    monkeypatch.setenv("OPENBENCH_MFDATASET_BATCH_SIZE", "2")

    write_mfdataset_atomic(input_files, output, batch_dir=batch_dir, sortby="time")

    with xr.open_dataset(output) as ds:
        np.testing.assert_allclose(ds["value"].values, np.array([3.0, 1.0, 2.0], dtype=np.float32))
    assert not list(batch_dir.glob(".combined.nc.mfbatch-*"))


def test_write_mfdataset_atomic_default_path_uses_single_open(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    calls = []

    class FakeDataset:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_open(paths, **kwargs):
        calls.append((list(paths), kwargs))
        return FakeDataset()

    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.setattr(loader, "open_mfdataset", fake_open)
    monkeypatch.setattr(loader, "write_netcdf_atomic", lambda ds, path: calls.append(("write", str(path))))

    loader.write_mfdataset_atomic(["a.nc", "b.nc", "c.nc"], tmp_path / "out.nc")

    assert calls == [
        (["a.nc", "b.nc", "c.nc"], {"chunks": "auto", "combine": "by_coords"}),
        ("write", str(tmp_path / "out.nc")),
    ]


def test_mfdataset_resource_plan_auto_batches_large_file_lists(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.setattr(loader, "_available_memory_bytes", lambda: None)
    paths = []
    for index in range(5):
        path = tmp_path / f"part-{index}.nc"
        path.write_bytes(b"0")
        paths.append(str(path))

    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", "3")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE", "2")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE", "2")

    plan = loader.build_resource_plan(paths)

    assert plan.file_count == 5
    assert plan.mfdataset_batch_size == 2
    assert plan.reason == "auto-file-count"


def test_mfdataset_resource_plan_uses_memory_cap(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.setattr(loader, "_available_memory_bytes", lambda: 40)
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", "1")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE", "1")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE", "10")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MEMORY_FRACTION", "0.5")

    paths = []
    for index in range(5):
        path = tmp_path / f"large-{index}.nc"
        path.write_bytes(b"0" * 10)
        paths.append(str(path))

    plan = loader.build_resource_plan(paths)

    assert plan.mfdataset_batch_size == 2
    assert plan.reason == "auto-memory"


def test_mfdataset_resource_plan_respects_explicit_batch_size(monkeypatch):
    import openbench.util.dataset_loader as loader

    monkeypatch.setattr(loader, "_available_memory_bytes", lambda: None)

    disabled = loader.build_resource_plan(["a.nc"] * 500, explicit_batch_size=0)
    forced = loader.build_resource_plan(["a.nc"] * 500, explicit_batch_size=7)

    assert disabled.mfdataset_batch_size == 0
    assert disabled.reason == "explicit"
    assert forced.mfdataset_batch_size == 7
    assert forced.reason == "explicit"


def test_write_mfdataset_atomic_auto_batches_when_plan_requires_it(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    calls = []

    class FakeDataset:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_open(paths, **kwargs):
        calls.append(("open", list(paths), kwargs))
        return FakeDataset()

    def fake_write(_ds, path, **kwargs):
        calls.append(("write", str(path), kwargs))

    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", "3")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE", "2")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE", "2")
    monkeypatch.setattr(loader, "_available_memory_bytes", lambda: None)
    monkeypatch.setattr(loader, "open_mfdataset", fake_open)
    monkeypatch.setattr(loader, "write_netcdf_atomic", fake_write)

    input_files = []
    for index in range(5):
        path = tmp_path / f"part-{index}.nc"
        path.write_bytes(b"0")
        input_files.append(str(path))

    loader.write_mfdataset_atomic(input_files, tmp_path / "combined.nc", batch_dir=tmp_path / "batches")

    open_lengths = [len(call[1]) for call in calls if call[0] == "open"]
    assert open_lengths == [2, 2, 1, 3]
    write_calls = [call for call in calls if call[0] == "write"]
    assert [call[2].get("compression") for call in write_calls[:-1]] == [False, False, False]
    assert write_calls[-1] == ("write", str(tmp_path / "combined.nc"), {})


def test_write_mfdataset_atomic_does_not_compress_batch_shards_when_env_enabled(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    calls = []

    class FakeDataset:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_open(paths, **kwargs):
        calls.append(("open", list(paths), kwargs))
        return FakeDataset()

    def fake_write(_ds, path, **kwargs):
        calls.append(("write", str(path), kwargs))

    monkeypatch.setenv("OPENBENCH_NETCDF_COMPRESSION", "1")
    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", "3")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE", "2")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE", "2")
    monkeypatch.setattr(loader, "_available_memory_bytes", lambda: None)
    monkeypatch.setattr(loader, "open_mfdataset", fake_open)
    monkeypatch.setattr(loader, "write_netcdf_atomic", fake_write)

    input_files = []
    for index in range(5):
        path = tmp_path / f"part-{index}.nc"
        path.write_bytes(b"0")
        input_files.append(str(path))

    loader.write_mfdataset_atomic(input_files, tmp_path / "combined.nc", batch_dir=tmp_path / "batches")

    write_calls = [call for call in calls if call[0] == "write"]
    assert [call[2].get("compression") for call in write_calls[:-1]] == [False, False, False]
    assert write_calls[-1] == ("write", str(tmp_path / "combined.nc"), {})


def test_write_mfdataset_atomic_env_zero_disables_auto_batch(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    calls = []

    class FakeDataset:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setenv("OPENBENCH_MFDATASET_BATCH_SIZE", "0")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", "3")
    monkeypatch.setattr(loader, "open_mfdataset", lambda paths, **kwargs: calls.append(list(paths)) or FakeDataset())
    monkeypatch.setattr(loader, "write_netcdf_atomic", lambda _ds, _path: None)

    loader.write_mfdataset_atomic(["a.nc"] * 5, tmp_path / "combined.nc")

    assert calls == [["a.nc"] * 5]


def test_write_mfdataset_zarr_auto_batches_when_plan_requires_it(monkeypatch, tmp_path):
    import openbench.util.dataset_loader as loader

    calls = []

    class FakeDataset:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def to_zarr(self, path, **kwargs):
            calls.append(("zarr", str(path), kwargs))

    def fake_open(paths, **kwargs):
        calls.append(("open", list(paths), kwargs))
        return FakeDataset()

    def fake_write(_ds, path, **kwargs):
        calls.append(("write", str(path), kwargs))

    monkeypatch.delenv("OPENBENCH_MFDATASET_BATCH_SIZE", raising=False)
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES", "3")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE", "2")
    monkeypatch.setenv("OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE", "2")
    monkeypatch.setattr(loader, "_available_memory_bytes", lambda: None)
    monkeypatch.setattr(loader, "open_mfdataset", fake_open)
    monkeypatch.setattr(loader, "write_netcdf_atomic", fake_write)

    input_files = []
    for index in range(5):
        path = tmp_path / f"part-{index}.nc"
        path.write_bytes(b"0")
        input_files.append(str(path))

    loader.write_mfdataset_zarr(input_files, tmp_path / "combined.zarr", batch_dir=tmp_path / "batches")

    open_lengths = [len(call[1]) for call in calls if call[0] == "open"]
    assert open_lengths == [2, 2, 1, 3]
    write_calls = [call for call in calls if call[0] == "write"]
    assert [call[2].get("compression") for call in write_calls] == [False, False, False]
    assert calls[-1] == ("zarr", str(tmp_path / "combined.zarr"), {"mode": "w"})
