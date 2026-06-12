"""Regression tests for cached registry usage in data processing hot paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.registry.schema import FallbackVar, ModelProfile, VariableMapping


def _make_processor(processing_module):
    processor = object.__new__(processing_module.BaseDatasetProcessing)
    processor.sim_source = "SimA"
    processor.ref_source = "RefA"
    processor.SimA_model = "ModelA"
    processor.RefA_model = "RefModel"
    processor.item = "Runoff"
    processor.sim_varname = ["runoff"]
    processor.ref_varname = ["runoff"]
    processor.sim_varunit = "mm"
    processor.ref_varunit = "mm"
    return processor


def _bomb_registry_manager(*args, **kwargs):
    raise AssertionError("RegistryManager should not be instantiated in processing hot paths")


class _FakeRegistry:
    def __init__(self, profile):
        self.profile = profile

    def get_model(self, model):
        if model == self.profile.name:
            return self.profile
        return None


def test_apply_model_specific_time_adjustment_uses_cached_registry(monkeypatch):
    import openbench.data.processing as processing
    import openbench.data.registry as registry_pkg
    import openbench.data.registry.manager as registry_manager

    processor = _make_processor(processing)
    ds = xr.Dataset(
        {"runoff": ("time", [1.0, 2.0])},
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="D")},
    )
    profile = ModelProfile(
        name="ModelA",
        description="test model",
        variables={},
        # New nested format: top-level group ("default" or file-pattern) → resolution → offset
        time_offset={"default": {"Day": "1 days"}},
    )

    monkeypatch.setattr(registry_pkg, "RegistryManager", _bomb_registry_manager)
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))

    adjusted = processing.BaseDatasetProcessing.apply_model_specific_time_adjustment(
        processor, ds, "sim", 2000, 2000, "Day"
    )

    assert adjusted.time.values[0] == np.datetime64("2000-01-02T00:30:00.000000000")
    assert adjusted.time.values[1] == np.datetime64("2000-01-03T00:30:00.000000000")


def test_try_compute_from_profile_uses_cached_registry(monkeypatch):
    import openbench.data.processing as processing
    import openbench.data.registry as registry_pkg
    import openbench.data.registry.manager as registry_manager

    processor = _make_processor(processing)
    ds = xr.Dataset(
        {
            "rain": xr.DataArray(np.array([0.5, 1.0, 0.3])),
            "snow": xr.DataArray(np.array([0.1, 0.0, 0.2])),
        }
    )
    profile = ModelProfile(
        name="ModelA",
        description="test model",
        variables={
            "Runoff": VariableMapping(
                varname="runoff",
                varunit="mm",
                compute="ds['rain'] + ds['snow']",
            )
        },
    )

    monkeypatch.setattr(registry_pkg, "RegistryManager", _bomb_registry_manager)
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))

    result = processing.BaseDatasetProcessing._try_compute_from_profile(processor, "ModelA", ds, "sim")

    np.testing.assert_array_equal(result.values, [0.6, 1.0, 0.5])
    assert result.name == "Runoff"
    assert processor.sim_varname == ["Runoff"]
    assert processor.sim_varunit == "mm"


def test_select_var_fallback_uses_cached_registry(monkeypatch):
    import openbench.data.processing as processing
    import openbench.data.registry as registry_pkg
    import openbench.data.registry.manager as registry_manager

    processor = _make_processor(processing)
    ds = xr.Dataset({"fallback_runoff": xr.DataArray(np.array([1.0, 2.0]))})
    profile = ModelProfile(
        name="ModelA",
        description="test model",
        variables={
            "Runoff": VariableMapping(
                varname="runoff",
                varunit="mm",
                fallbacks=[FallbackVar(varname="fallback_runoff", varunit="kg", convert="value * 2")],
            )
        },
    )

    def _raise_filter_error(datasource, opened_ds, varname):
        raise KeyError("force fallback lookup")

    monkeypatch.setattr(registry_pkg, "RegistryManager", _bomb_registry_manager)
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))
    monkeypatch.setattr(processing.xr, "open_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(processing.Convert_Type, "convert_nc", lambda obj: obj)
    processor.apply_custom_filter = _raise_filter_error

    result = processing.BaseDatasetProcessing.select_var(processor, 2000, 2000, "Day", "dummy.nc", ["runoff"], "sim")

    np.testing.assert_array_equal(result.values, [2.0, 4.0])
    assert processor.sim_varname == ["fallback_runoff"]
    # convert is present, so the data is now in the PRIMARY unit (mm), not the
    # fallback's native kg — process_units must not re-convert from kg.
    assert processor.sim_varunit == "mm"


def test_select_var_fallback_conversion_can_reference_peer_variables(monkeypatch):
    import openbench.data.processing as processing
    import openbench.data.registry as registry_pkg
    import openbench.data.registry.manager as registry_manager

    processor = _make_processor(processing)
    processor.item = "Net_Ecosystem_Exchange"
    processor.sim_varname = ["f_nee"]
    processor.sim_varunit = "g m-2 s-1"
    ds = xr.Dataset(
        {
            "f_respc": xr.DataArray(np.array([2.0, 3.0])),
            "f_assim": xr.DataArray(np.array([1.0, 1.5])),
        }
    )
    profile = ModelProfile(
        name="ModelA",
        description="test model",
        variables={
            "Net_Ecosystem_Exchange": VariableMapping(
                varname="f_nee",
                varunit="g m-2 s-1",
                fallbacks=[
                    FallbackVar(
                        varname="f_respc",
                        varunit="mol m-2 s-1",
                        convert="value * 12.011 - f_assim * 12.011",
                    )
                ],
            )
        },
    )

    def _raise_filter_error(datasource, opened_ds, varname):
        raise KeyError("force fallback lookup")

    monkeypatch.setattr(registry_pkg, "RegistryManager", _bomb_registry_manager)
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))
    monkeypatch.setattr(processing.xr, "open_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(processing.Convert_Type, "convert_nc", lambda obj: obj)
    processor.apply_custom_filter = _raise_filter_error

    result = processing.BaseDatasetProcessing.select_var(processor, 2000, 2000, "Day", "dummy.nc", ["f_nee"], "sim")

    np.testing.assert_allclose(result.values, [12.011, 18.0165])
    assert processor.sim_varname == ["f_respc"]
    # The fallback convert already applied the mol→gC molar mass (×12.011), so
    # the unit driving downstream process_units must be the PRIMARY unit
    # (g m-2 s-1), NOT the fallback's native mol m-2 s-1. Using mol m-2 s-1
    # would let process_units re-apply ×12.01, inflating NEE/GPP by ~12×.
    assert processor.sim_varunit == "g m-2 s-1"
    # The derived NEE must not keep the source variable's identity/label.
    assert result.name == "Net_Ecosystem_Exchange"
    assert result.attrs.get("long_name") == "Net Ecosystem Exchange"


def test_select_var_raises_if_materialization_fails(monkeypatch):
    """select_var must not return a lazy object after closing its source dataset."""
    import pytest

    import openbench.data.processing as processing

    processor = _make_processor(processing)
    ds = xr.Dataset({"runoff": xr.DataArray(np.array([1.0, 2.0]))})

    monkeypatch.setattr(processing.xr, "open_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(processing.Convert_Type, "convert_nc", lambda obj: obj)
    monkeypatch.setattr(xr.DataArray, "load", lambda self, **kwargs: (_ for _ in ()).throw(OSError("load failed")))

    with pytest.raises(RuntimeError, match="Failed to materialize selected variable"):
        processing.BaseDatasetProcessing.select_var(processor, 2000, 2000, "Day", "dummy.nc", ["runoff"], "sim")


def test_select_var_closes_source_even_when_returning_loaded_dataset(monkeypatch):
    """If a filter returns the opened Dataset itself, select_var still must close the source."""
    import openbench.data.processing as processing

    processor = _make_processor(processing)
    ds = xr.Dataset({"runoff": xr.DataArray(np.array([1.0, 2.0]))})
    calls = {"close": 0}

    original_close = xr.Dataset.close

    def close_spy(self):
        calls["close"] += 1
        return original_close(self)

    processor.apply_custom_filter = lambda datasource, opened_ds, varname: opened_ds

    monkeypatch.setattr(xr.Dataset, "close", close_spy)
    monkeypatch.setattr(processing.xr, "open_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(processing.Convert_Type, "convert_nc", lambda obj: obj)

    result = processing.BaseDatasetProcessing.select_var(processor, 2000, 2000, "Day", "dummy.nc", ["runoff"], "sim")

    assert result is ds
    assert calls["close"] == 1


def test_processing_atomic_netcdf_write_preserves_existing_file_on_failure(tmp_path):
    """Atomic NetCDF writes should not replace a valid existing file until temp write succeeds."""
    import pytest

    import openbench.data.processing as processing

    target = tmp_path / "existing.nc"
    xr.Dataset({"value": ("time", np.array([1.0]))}).to_netcdf(target)

    class FailingWriter:
        def to_netcdf(self, path, *args, **kwargs):
            Path(path).write_bytes(b"partial invalid netcdf")
            raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated write failure"):
        processing._write_netcdf_atomic(FailingWriter(), target)

    with xr.open_dataset(target) as ds:
        np.testing.assert_allclose(ds["value"].values, [1.0])
    assert not list(tmp_path.glob(".existing.nc.*.tmp.nc"))


def test_data_cache_raises_when_disk_store_fails():
    import pytest

    from openbench.data.cache import CacheError, DataCache

    class FailingManager:
        def set(self, key, value, level=None):
            return False

    ds = xr.Dataset({"value": ("time", np.array([1.0]))})
    data_cache = DataCache(FailingManager())

    with pytest.raises(CacheError, match="Failed to cache dataset"):
        data_cache.cache_dataset(ds, "sample")


def test_find_data_files_escapes_glob_metachars_in_prefix_suffix(tmp_path):
    """Prefix/suffix containing glob metacharacters ([, ?, *) must be matched
    literally, not interpreted as wildcards. The intentional wildcard is only
    between {year} and {suffix}.
    """
    import openbench.data.processing as processing

    # Create real files: only the literal-bracketed name should be picked
    real_file = tmp_path / "case_[v1]_2010_h0.nc"
    real_file.touch()
    decoy = tmp_path / "case_xv1y_2010_h0.nc"  # would match '[v1]' if not escaped
    decoy.touch()

    processor = _make_processor(processing)
    # Disable any prefix-fallback list lookup
    processor.SimA_prefix_fallback = None
    processor.RefA_prefix_fallback = None

    result = processing.BaseDatasetProcessing._find_data_files(
        processor,
        str(tmp_path),
        prefix="case_[v1]_",
        year=2010,
        suffix="_h0",
        datasource="sim",
    )
    # Only the literal-bracketed file should match
    assert [str(real_file)] == result, f"Expected only {real_file.name}, got: {result}"


def test_find_data_files_prefers_prefix_fallback_over_primary_for_routing_variables(
    tmp_path,
    monkeypatch,
):
    import openbench.data.processing as processing
    import openbench.data.registry as registry_pkg
    import openbench.data.registry.manager as registry_manager

    main = tmp_path / "Mediterranean_hist_2000-01.nc"
    unitcat = tmp_path / "Mediterranean_hist_unitcat_2000-01.nc"
    xr.Dataset({"f_discharge": xr.DataArray(np.array([1.0]))}).to_netcdf(main)
    xr.Dataset({"f_discharge": xr.DataArray(np.array([2.0]))}).to_netcdf(unitcat)

    processor = _make_processor(processing)
    processor.item = "Streamflow"
    processor.SimA_prefix_fallback = ["_cama_", "_unitcat_"]
    profile = ModelProfile(
        name="ModelA",
        description="test model",
        variables={
            "Streamflow": VariableMapping(
                varname="outflw",
                varunit="m3 s-1",
                fallbacks=[FallbackVar(varname="f_discharge", varunit="m3 s-1")],
            )
        },
    )

    monkeypatch.setattr(registry_pkg, "RegistryManager", _bomb_registry_manager)
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))

    result = processing.BaseDatasetProcessing._find_data_files(
        processor,
        str(tmp_path),
        prefix="Mediterranean_hist_",
        year=2000,
        suffix="",
        datasource="sim",
        varname=["outflw"],
    )

    assert result == [str(unitcat)]


def test_make_time_integrity_accepts_climatology_frequency():
    import openbench.data.processing as processing

    processor = _make_processor(processing)
    da = xr.DataArray(
        np.arange(12.0),
        dims=["time"],
        coords={"time": pd.date_range("2001-01-15", periods=12, freq="MS")},
        name="runoff",
    )

    result = processing.BaseDatasetProcessing.make_time_integrity(processor, da, 2001, 2001, "climatology-month", "sim")

    assert result.identical(da)


def test_remap_data_uses_configured_backend_without_silent_fallback(monkeypatch):
    import openbench.data.processing as processing

    processor = object.__new__(processing.GridDatasetProcessing)
    processor.regrid_backend = "openbench_conservative"
    processor.create_target_grid = lambda: xr.Dataset({"lat": [0.0], "lon": [0.0]})
    processor.remap_interpolate = lambda data, grid: data.copy()
    processor.remap_cdo = lambda data, grid: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
    processor.remap_xesmf = lambda data, grid: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
    processor.remap_basic_interpolation = lambda data, grid: (_ for _ in ()).throw(
        AssertionError("unexpected fallback")
    )

    data = xr.Dataset({"v": (("lat", "lon"), np.array([[1.0]]))}, coords={"lat": [0.0], "lon": [0.0]})
    result = processing.GridDatasetProcessing.remap_data(processor, data)

    assert result.attrs["openbench_regrid_backend"] == "openbench_conservative"
    assert result.attrs["openbench_regrid_algorithm_version"] == processing.REGRID_ALGORITHM_VERSION


def test_remap_data_fails_configured_backend_instead_of_falling_back():
    import pytest

    import openbench.data.processing as processing

    processor = object.__new__(processing.GridDatasetProcessing)
    processor.regrid_backend = "cdo_remapcon"
    processor.create_target_grid = lambda: xr.Dataset({"lat": [0.0], "lon": [0.0]})
    processor.remap_cdo = lambda data, grid: (_ for _ in ()).throw(RuntimeError("cdo unavailable"))
    processor.remap_interpolate = lambda data, grid: data

    data = xr.Dataset({"v": (("lat", "lon"), np.array([[1.0]]))}, coords={"lat": [0.0], "lon": [0.0]})
    with pytest.raises(RuntimeError, match="Configured regrid backend 'cdo_remapcon' failed"):
        processing.GridDatasetProcessing.remap_data(processor, data)
