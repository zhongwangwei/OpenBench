"""Station preprocessing must apply catalog compute the same way the grid path does.

Grid selection runs a catalog ``compute`` before looking up the configured
varname. Station preprocessing used to read a same-named raw variable directly
and skip the compute, so station-mode model output silently lost sign flips,
unit conversions and PFT aggregation (e.g. ``-ds['FIRA']``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.processing import StationDatasetProcessing
from openbench.data.registry.schema import ModelProfile, VariableMapping


class _FakeRegistry:
    def __init__(self, profile):
        self.profile = profile

    def get_model(self, model):
        return self.profile if model == self.profile.name else None

    def get_reference(self, name):
        return None


def _processor(monkeypatch, compute):
    import openbench.data.registry.manager as registry_manager

    profile = ModelProfile(
        name="SiteModel",
        description="station-mode model output",
        variables={
            "Surface_Net_LW_Radiation": VariableMapping(varname="FIRA", varunit="W m-2", compute=compute),
        },
    )
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))

    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.item = "Surface_Net_LW_Radiation"
    proc.ref_source = "Ref"
    proc.sim_source = "SiteModel"
    proc.sim_varname = ["FIRA"]
    proc.sim_varunit = "W m-2"
    proc.compare_tim_res = "D"
    proc._is_climatology_mode = lambda: False
    proc._resample_to_compare_resolution = lambda ds, *args: ds
    proc.check_coordinate = lambda ds: ds
    proc.check_dataset_time_integrity = lambda ds, *args: ds
    proc.process_units = lambda ds, unit: (ds, unit)
    proc.select_timerange = lambda ds, *args: ds
    return proc


def _site_output(**variables):
    time = pd.date_range("2000-01-01", periods=2)
    return xr.Dataset({name: ("time", values) for name, values in variables.items()}, coords={"time": time})


def test_catalog_compute_wins_over_same_named_raw_station_variable(monkeypatch):
    proc = _processor(monkeypatch, compute="-ds['FIRA']")

    out = proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")

    np.testing.assert_allclose(out.values, [-10.0, -20.0])
    assert proc.sim_varname == ["FIRA"]


def test_raw_station_variable_is_used_when_compute_dependency_is_missing(monkeypatch):
    proc = _processor(monkeypatch, compute="ds['FIRE'] - ds['FLDS']")

    out = proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")

    np.testing.assert_allclose(out.values, [10.0, 20.0])
