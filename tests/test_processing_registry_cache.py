"""Regression tests for cached registry usage in data processing hot paths."""

from __future__ import annotations

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

    result = processing.BaseDatasetProcessing.select_var(
        processor, 2000, 2000, "Day", "dummy.nc", ["runoff"], "sim"
    )

    np.testing.assert_array_equal(result.values, [2.0, 4.0])
    assert processor.sim_varname == ["fallback_runoff"]
    assert processor.sim_varunit == "kg"


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
    assert [str(real_file)] == result, (
        f"Expected only {real_file.name}, got: {result}"
    )
