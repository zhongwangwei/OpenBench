"""Station preprocessing must apply catalog compute the same way the grid path does.

Grid selection runs a catalog ``compute`` before looking up the configured
varname. Station preprocessing used to read a same-named raw variable directly
and skip the compute, so station-mode model output silently lost sign flips,
unit conversions and PFT aggregation (e.g. ``-ds['FIRA']``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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


def _processor(monkeypatch, compute, item="Surface_Net_LW_Radiation", varname="FIRA"):
    import openbench.data.registry.manager as registry_manager

    profile = ModelProfile(
        name="SiteModel",
        description="station-mode model output",
        variables={
            item: VariableMapping(varname=varname, varunit="W m-2", compute=compute),
        },
    )
    monkeypatch.setattr(registry_manager, "get_registry", lambda: _FakeRegistry(profile))

    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.item = item
    proc.ref_source = "Ref"
    proc.sim_source = "SiteModel"
    proc.sim_varname = [varname]
    proc.sim_varunit = "W m-2"
    proc.compare_tim_res = "D"
    proc._is_climatology_mode = lambda: False
    proc._resample_to_compare_resolution = lambda ds, *args: ds
    proc.check_coordinate = lambda ds: ds
    proc.check_dataset_time_integrity = lambda ds, *args: ds
    proc.process_units = lambda ds, unit, datasource=None: (ds, unit)
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


@pytest.mark.parametrize("item", ["Suspended_Sediment_Concentration", "Suspended_Sediment_Load"])
@pytest.mark.parametrize("parts", [(1, 3), (1, 2, 3, 4), (1, 2, 3)])
@pytest.mark.parametrize("raw_present", [True, False])
def test_station_sediment_requires_complete_size_classes(monkeypatch, tmp_path, item, parts, raw_present):
    from openbench.data.compute import ComputeIntegrityError
    from openbench.data.registry.manager import RegistryManager

    mapping = RegistryManager(user_dir=tmp_path).get_model("CoLM2024").variables[item]
    prefix = "f_sedcon_" if item == "Suspended_Sediment_Concentration" else "f_sedout_"
    proc = _processor(monkeypatch, mapping.compute, item=item, varname=mapping.varname if raw_present else item)
    variables = {f"{prefix}{part}": [float(part)] * 2 for part in parts}
    ds = _site_output(**variables)

    if parts == (1, 2, 3):
        out = proc.process_single_station_data(ds, 2000, 2000, "sim")
        np.testing.assert_allclose(out.values, [6.0 * 2650] * 2)
    else:
        with pytest.raises(ComputeIntegrityError, match=prefix):
            proc.process_single_station_data(ds, 2000, 2000, "sim")


@pytest.mark.parametrize("expression", ["ds.sum_prefix('FIRA_', 3)", "ds['FIRA'] + ds['missing']"])
def test_station_missing_derivation_inputs_skip_without_partial_raw_fallback(monkeypatch, expression):
    from openbench.data.station_missing import StationDataUnavailable

    proc = _processor(monkeypatch, expression, varname="FIRA_1" if "sum_prefix" in expression else "FIRA")
    with pytest.raises(StationDataUnavailable):
        proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")


def test_station_all_missing_parts_allows_independent_raw_result(monkeypatch):
    proc = _processor(monkeypatch, "ds.sum_prefix('FIRA_', 3)")
    result = proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")
    np.testing.assert_allclose(result.values, [10.0, 20.0])


@pytest.mark.parametrize("parts", ["0", "True", "2.0"])
def test_station_invalid_part_count_propagates_with_source_context(monkeypatch, parts):
    from openbench.data.compute import ComputeIntegrityError

    proc = _processor(monkeypatch, f"ds.sum_prefix('FIRA_', {parts})")
    dataset = _site_output(FIRA=[10.0, 20.0])
    dataset.encoding["source"] = "station-001.nc"
    with pytest.raises(ComputeIntegrityError, match=r"station-001.nc.*2000.*positive whole number"):
        proc.process_single_station_data(dataset, 2000, 2000, "sim")


@pytest.mark.parametrize("expression", ["ds.FIRA + ds['missing']", "ds.get('FIRA') + ds['missing']"])
def test_station_attribute_and_get_dependencies_cannot_be_raw_fallback(monkeypatch, expression):
    from openbench.data.station_missing import StationDataUnavailable

    proc = _processor(monkeypatch, expression)
    with pytest.raises(StationDataUnavailable):
        proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")


@pytest.mark.parametrize("fallback", ["part1", "independent"])
@pytest.mark.parametrize("complete", [False, True])
def test_station_runtime_fallback_respects_compute_dependencies(monkeypatch, fallback, complete):
    from openbench.data.station_missing import StationDataUnavailable

    proc = _processor(monkeypatch, "ds['part1'] + ds['part2']", varname="total")
    proc.SiteModel_fallbacks = [{"varname": fallback, "convert": "value * 10"}]
    variables = {"part1": [1.0, 2.0], "independent": [4.0, 5.0]}
    if complete:
        variables["part2"] = [2.0, 3.0]
    if fallback == "part1" and not complete:
        with pytest.raises(StationDataUnavailable):
            proc.process_single_station_data(_site_output(**variables), 2000, 2000, "sim")
    else:
        result = proc.process_single_station_data(_site_output(**variables), 2000, 2000, "sim")
        np.testing.assert_allclose(result.values, [3.0, 5.0] if fallback == "part1" else [40.0, 50.0])


@pytest.mark.parametrize("grid", [False, True])
def test_adapter_fallback_conversion_only_applies_to_raw_result(monkeypatch, tmp_path, grid):
    from openbench.config.adapter import _resolve_varname
    from openbench.data.registry.schema import FallbackVar

    expression = "ds['part1'] + ds['part2']"
    mapping = VariableMapping(
        varname="total",
        varunit="W m-2",
        compute=expression,
        fallbacks=[FallbackVar(varname="part1", varunit="W m-2", convert="value * 10")],
    )
    dataset = _site_output(part1=[1.0, 2.0], part2=[2.0, 3.0])
    path = tmp_path / "source.nc"
    dataset.to_netcdf(path)
    varname, unit, conversion = _resolve_varname(mapping, str(tmp_path))
    assert (varname, unit, conversion) == ("part1", "W m-2", "value * 10")
    proc = _processor(monkeypatch, expression, varname=varname)
    proc._fb_convert_sim = conversion

    if grid:
        result = proc.select_var(2000, 2000, "Day", path, [varname], "sim")
    else:
        result = proc.process_single_station_data(dataset, 2000, 2000, "sim")
    np.testing.assert_allclose(result.values, [3.0, 5.0])
    assert proc._fb_convert_sim == conversion

    # The same worker's next file can still use an independent raw fallback.
    proc.SiteModel_compute = "ds['missing1'] + ds['missing2']"
    proc.sim_varname = [varname]
    if grid:
        result = proc.select_var(2000, 2000, "Day", path, [varname], "sim")
    else:
        result = proc.process_single_station_data(dataset, 2000, 2000, "sim")
    np.testing.assert_allclose(result.values, [10.0, 20.0])


@pytest.mark.parametrize(
    "expression",
    [
        " ds['FIRA'] + ds['missing']",
        "t = ds['FIRA'];\n    t + ds['missing']",
        "key = 'FIRA'; ds[key] + ds['missing']",
        "ds[['FIRA']]['FIRA'] + ds['missing']",
    ],
)
def test_station_whitespace_in_compute_cannot_expose_raw_input(monkeypatch, expression):
    from openbench.data.station_missing import StationDataUnavailable

    proc = _processor(monkeypatch, expression)
    with pytest.raises(StationDataUnavailable):
        proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")


def test_station_uses_standard_item_name_when_no_parts_exist(monkeypatch):
    item = "Suspended_Sediment_Concentration"
    proc = _processor(monkeypatch, "ds.sum_prefix('f_sedcon_', 3) * 2650", item=item, varname="f_sedcon_1")

    out = proc.process_single_station_data(_site_output(**{item: [7.0, 7.0]}), 2000, 2000, "sim")

    np.testing.assert_allclose(out.values, [7.0, 7.0])


def test_station_standard_item_name_is_not_used_when_it_is_a_compute_input(monkeypatch):
    from openbench.data.station_missing import StationDataUnavailable

    item = "Surface_Net_LW_Radiation"
    proc = _processor(monkeypatch, f"ds['{item}'] + ds['missing']", item=item, varname="FIRA")
    with pytest.raises(StationDataUnavailable):
        proc.process_single_station_data(_site_output(**{item: [7.0, 7.0]}), 2000, 2000, "sim")


def test_fatal_station_compute_error_names_the_station(monkeypatch, tmp_path):
    from openbench.data.compute import ComputeIntegrityError

    proc = _processor(monkeypatch, "ds.sum_prefix('FIRA_', 0)")
    proc.casedir = str(tmp_path)
    path = tmp_path / "merged.nc"
    _site_output(FIRA=[10.0, 20.0]).to_netcdf(path)
    stations = pd.DataFrame([{"ID": "AU-Tum", "use_syear": 2000, "use_eyear": 2000, "sim_dir": str(path)}])

    with pytest.raises(ComputeIntegrityError, match=r"Station AU-Tum: .*merged.nc"):
        proc._make_stn_parallel(stations, "sim", 0)


def test_station_task_removes_only_its_own_stale_temp_files(monkeypatch, tmp_path):
    proc = _processor(monkeypatch, "-ds['FIRA']")
    proc.casedir = str(tmp_path)
    station_dir = tmp_path / "data" / "stn_Ref_SiteModel"
    station_dir.mkdir(parents=True)
    own = station_dir / ".Surface_Net_LW_Radiation_sim_A_2000_2000.nc.abc123.tmp.nc"
    other_source = station_dir / ".Surface_Net_LW_Radiation_ref_A_2000_2000.nc.def456.tmp.nc"
    output = station_dir / "Surface_Net_LW_Radiation_sim_B_2000_2000.nc"
    for path in (own, other_source, output):
        path.write_bytes(b"")

    proc._remove_stale_station_temp_files("sim")

    assert not own.exists()
    assert other_source.exists() and output.exists()


def test_station_fallback_does_not_skip_a_compute_that_fails_to_parse(monkeypatch):
    from openbench.data.compute import ComputeError

    proc = _processor(monkeypatch, "ds['FIRA'] * 2 +", varname="FIRA_missing")
    proc.SiteModel_fallbacks = [{"varname": "FIRA", "varunit": "W m-2"}]

    with pytest.raises(ComputeError):
        proc.process_single_station_data(_site_output(FIRA=[10.0, 20.0]), 2000, 2000, "sim")


def test_station_runtime_fallback_is_not_trusted_when_compute_inputs_are_unknown(monkeypatch):
    from openbench.data.station_missing import StationDataUnavailable

    proc = _processor(monkeypatch, "key = 'part1'; ds[key] + ds['part2']", varname="total")
    proc.SiteModel_fallbacks = [{"varname": "part1"}]

    with pytest.raises(StationDataUnavailable):
        proc.process_single_station_data(_site_output(part1=[1.0, 2.0]), 2000, 2000, "sim")
