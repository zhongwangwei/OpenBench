from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def test_yearly_grid_preprocessing_resamples_before_scratch_write(tmp_path):
    from openbench.data.processing import DatasetProcessing

    src = tmp_path / "src"
    case = tmp_path / "case"
    src.mkdir()
    (case / "scratch").mkdir(parents=True)
    (case / "data").mkdir()

    input_ds = xr.Dataset(
        {"foo": (("time", "lat", "lon"), np.ones((365, 2, 2), dtype="float32"))},
        coords={
            "time": pd.date_range("2001-01-01", periods=365, freq="D"),
            "lat": [0.5, 1.5],
            "lon": [0.5, 1.5],
        },
    )
    input_ds.to_netcdf(src / "foo_2001.nc")

    processor = object.__new__(DatasetProcessing)
    processor.casedir = str(case)
    processor.minyear = 2001
    processor.maxyear = 2001
    processor.compare_tim_res = "ME"
    processor.compare_grid_res = 1.0
    processor.num_cores = 1
    processor.debug_mode = False
    processor.item = "Sensible_Heat"
    processor.sim_source = "Sim"
    processor.ref_source = "Ref"
    processor.sim_data_type = "grid"
    processor.ref_data_type = "grid"
    processor.sim_varname = ["foo"]
    processor.sim_varunit = "W m-2"
    processor.coordinate_map = {}
    processor.timezone = 0
    processor.sim_model = "model"
    processor.sim_dir = str(src)
    processor.sim_prefix = "foo_"
    processor.sim_suffix = ""

    processor.preprocess_yearly_files(
        str(src),
        2001,
        2001,
        "D",
        "W m-2",
        ["foo"],
        str(case),
        "",
        "foo_",
        "sim",
    )

    with xr.open_dataset(case / "scratch" / "sim_foo_2001.nc") as scratch:
        assert scratch.sizes["time"] == 12


def test_lazy_select_var_keeps_fallback_conversion_chunked(tmp_path, monkeypatch):
    import openbench.data._processing_selection as selection
    from openbench.data.processing import DatasetProcessing

    path = tmp_path / "input.nc"
    xr.Dataset(
        {"foo": ("time", np.arange(24, dtype="float32"))},
        coords={"time": pd.date_range("2001-01-01", periods=24, freq="D")},
    ).to_netcdf(path)
    monkeypatch.setattr(
        selection,
        "open_dataset_chunked",
        lambda value, **kwargs: xr.open_dataset(value, chunks={"time": 6}, **kwargs),
    )

    processor = object.__new__(DatasetProcessing)
    processor.item = "Demo"
    processor.sim_source = "Sim"
    processor.sim_varunit = "1"
    processor._fb_convert_sim = "value * 2"
    processor.apply_custom_filter = lambda _source, ds, names: ds[names[0]]

    result, source = processor.select_var(2001, 2001, "D", str(path), ["foo"], "sim", load=False, return_source=True)
    try:
        assert result.chunks is not None
        np.testing.assert_array_equal(result.compute(), np.arange(24, dtype="float32") * 2)
    finally:
        source.close()


def test_single_file_preprocessing_stays_lazy_and_closes_source(tmp_path):
    from openbench.data.processing import DatasetProcessing

    source = type("Source", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    processor = object.__new__(DatasetProcessing)
    processor.minyear = 2001
    processor.maxyear = 2001
    processor.sim_varunit = "W m-2"
    processor._find_single_file = lambda *args, **kwargs: "input.nc"
    processor.select_var = lambda *args, **kwargs: (
        (
            xr.Dataset({"foo": ("time", [1.0])}, coords={"time": [pd.Timestamp("2001-01-01")]}),
            source,
        )
        if kwargs == {"load": False, "return_source": True}
        else (_ for _ in ()).throw(AssertionError(kwargs))
    )
    processor.check_coordinate = lambda ds: ds
    processor.check_dataset_time_integrity = lambda ds, *args: ds
    processor.select_timerange = lambda ds, *args: ds
    processor.process_units = lambda ds, unit: (ds, unit)
    processor.split_year = lambda *args: None

    processor.preprocess_single_file(
        str(tmp_path), 2001, 2001, "ME", "W m-2", ["foo"], str(tmp_path), "", "foo_", "sim"
    )

    assert source.closed


def test_large_dataset_auto_chunking_uses_native_file_chunks(tmp_path, monkeypatch):
    import openbench.util.dataset_loader as loader

    path = tmp_path / "large.nc"
    xr.Dataset({"foo": ("time", np.arange(24, dtype="float32"))}).to_netcdf(
        path, encoding={"foo": {"chunksizes": (6,)}}
    )
    seen = {}
    monkeypatch.setattr(loader.os.path, "getsize", lambda _path: loader.CHUNK_SIZE_THRESHOLD + 1)
    monkeypatch.setattr(loader, "_open_dataset_with_fallback", lambda _path, **kwargs: seen.update(kwargs) or object())

    loader.open_dataset(str(path))

    assert seen["chunks"] == {}
