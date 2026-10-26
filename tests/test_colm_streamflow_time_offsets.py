"""History provenance, rather than variable names, determines CoLM time shifts."""

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from openbench.data._processing_time_adjustments import TimeAdjustmentMixin


@pytest.mark.parametrize("model", ["CoLM2024", "CoLM"])
@pytest.mark.parametrize("varname", ["outflw", "fldare", "f_discharge", "f_floodarea", "f_sedcon_1"])
@pytest.mark.parametrize(
    "resolution, endpoint, midpoint",
    [
        ("Hour", "2001-01-01 01:00", "2001-01-01 00:30"),
        ("Day", "2001-01-02 00:00", "2001-01-01 12:00"),
        ("Month", "2001-02-01 00:00", "2001-01-17 00:00"),
    ],
)
def test_colm_offset_uses_actual_history_source(model, varname, resolution, endpoint, midpoint):
    proc = SimpleNamespace(sim_source=model, item="Streamflow")
    # The actual file, not the variable name, selects the CaMa rule.
    for filename, raw, expected in [
        ("case_hist_cama_2001.nc", endpoint, pd.Timestamp(endpoint) - (
            pd.DateOffset(hours=1) if resolution == "Hour" else
            pd.DateOffset(days=1) if resolution == "Day" else pd.DateOffset(months=1)
        )),
        ("case_hist_unitcat_2001-01.nc", midpoint, pd.Timestamp(midpoint)),
    ]:
        data = xr.DataArray([3.0], dims="time", coords={"time": [pd.Timestamp(raw)]}, name=varname)
        actual = TimeAdjustmentMixin.apply_model_specific_time_adjustment(
            proc, data, "sim", 2001, 2001, resolution, source_files=filename
        )
        assert pd.Timestamp(actual.time.values[0]) == expected
        assert actual.values[0] == 3.0


def test_original_year_day_not_shifted_by_coarse_month_comparison():
    proc = SimpleNamespace(sim_source="CoLM2024", item="Inundation_Area")
    data = xr.DataArray(
        [11.0, 22.0],
        dims="time",
        coords={"time": pd.to_datetime(["2001-01-01 12:00", "2001-12-31 12:00"])},
        name="f_floodarea",
    )
    result = TimeAdjustmentMixin.apply_model_specific_time_adjustment(
        proc, data, "sim", 2001, 2001, "Day", source_files="run_hist_unitcat_2001.nc"
    )
    assert list(pd.DatetimeIndex(result.time.values)) == list(pd.DatetimeIndex(data.time.values))
    assert result.values.tolist() == [11.0, 22.0]


def test_mixed_history_sources_rejected():
    proc = SimpleNamespace(sim_source="CoLM2024", item="Streamflow")
    data = xr.DataArray([1.0], dims="time", coords={"time": [pd.Timestamp("2001-01-02")]}, name="outflw")
    with pytest.raises(ValueError, match="Mixed"):
        TimeAdjustmentMixin.apply_model_specific_time_adjustment(
            proc, data, "sim", 2001, 2001, "Day",
            source_files=["run_hist_cama_2001.nc", "run_hist_unitcat_2001.nc"],
        )


def test_yearly_cama_uses_previous_calendar_year():
    proc = SimpleNamespace(sim_source="CoLM2024", item="Streamflow")
    data = xr.DataArray([99.0], dims="time", coords={"time": [pd.Timestamp("2002-01-01")]}, name="outflw")
    result = TimeAdjustmentMixin.apply_model_specific_time_adjustment(
        proc, data, "sim", 2001, 2001, "Year", source_files="run_hist_cama_2001.nc"
    )
    assert pd.Timestamp(result.time.values[0]) == pd.Timestamp("2001-01-01")


def test_user_override_on_native_variable(monkeypatch):
    from openbench.data.registry import manager

    catalog = manager.get_writable_model_catalog_path()
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "CoLM2024:\n  time_offset:\n    Day:\n      f_discharge: '-2 days'\n",
        encoding="utf-8",
    )
    manager.clear_registry_cache()
    proc = SimpleNamespace(sim_source="CoLM2024", item="Streamflow")
    data = xr.DataArray([1.0], dims="time", coords={"time": [pd.Timestamp("2001-01-03 12:00")]}, name="f_discharge")
    result = TimeAdjustmentMixin.apply_model_specific_time_adjustment(
        proc, data, "sim", 2001, 2001, "Day", source_files="run_hist_unitcat_2001.nc"
    )
    assert pd.Timestamp(result.time.values[0]) == pd.Timestamp("2001-01-01 12:00")
