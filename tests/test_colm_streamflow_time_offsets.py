"""CoLM riverlake discharge is already period-labelled; CaMa discharge is not."""

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from openbench.data._processing_time_adjustments import TimeAdjustmentMixin


@pytest.mark.parametrize("model", ["CoLM2024", "CoLM"])
@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("varname", ["outflw", "f_discharge", "v_discharge"])
@pytest.mark.parametrize(
    "resolution, endpoint, midpoint, expected",
    [
        ("Hour", "2001-01-01 01:00", "2001-01-01 00:30", "2001-01-01 00:30"),
        ("Day", "2001-01-02 00:00", "2001-01-01 12:00", "2001-01-01 00:30"),
        ("Month", "2001-02-01 00:00", "2001-01-17 00:00", "2001-01-01 00:30"),
    ],
)
def test_colm_streamflow_offset_depends_on_selected_variable(
    model, as_dataset, varname, resolution, endpoint, midpoint, expected
):
    configured_name = "f_discharge" if varname == "outflw" else "outflw"
    processor = SimpleNamespace(sim_source=model, item="Streamflow", sim_varname=[configured_name])
    raw_time = endpoint if varname == "outflw" else midpoint
    data = xr.DataArray([3.0], dims="time", coords={"time": [pd.Timestamp(raw_time)]}, name=varname)
    if as_dataset:
        data = data.to_dataset()

    adjusted = TimeAdjustmentMixin.apply_model_specific_time_adjustment(processor, data, "sim", 2001, 2001, resolution)

    expected_time = pd.Timestamp(expected) if varname == "outflw" else pd.Timestamp(midpoint).replace(minute=30)
    assert pd.Timestamp(adjusted.time.values[0]) == expected_time
    adjusted_data = adjusted[varname] if as_dataset else adjusted
    assert adjusted_data.values[0] == 3.0


@pytest.mark.parametrize("resolution", ["Hour", "Day", "Month"])
def test_colm_selected_fallback_keeps_its_period(monkeypatch, resolution):
    import openbench.data.processing as processing

    processor = object.__new__(processing.BaseDatasetProcessing)
    processor.sim_source = "CoLM2024"
    processor.item = "Streamflow"
    processor.sim_data_type = "grid"
    processor.sim_varname = ["outflw"]
    processor.sim_varunit = "m3 s-1"
    dataset = xr.Dataset(
        {"F_Discharge": ("time", [3.0])},
        coords={"time": [pd.Timestamp("2001-01-17 12:30")]},
    )
    monkeypatch.setattr(processing.xr, "open_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(processing.Convert_Type, "convert_nc", lambda data: data)

    selected = processor.select_var(2001, 2001, resolution, "dummy.nc", ["outflw"], "sim")
    adjusted = processor.apply_model_specific_time_adjustment(selected, "sim", 2001, 2001, resolution)

    assert selected.name == "F_Discharge"
    assert pd.Timestamp(adjusted.time.values[0]) == pd.Timestamp("2001-01-17 12:30")


def test_native_variable_time_offset_can_be_overridden():
    from openbench.data.registry import manager

    catalog = manager.get_writable_model_catalog_path()
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "CoLM2024:\n  time_offset:\n    Day:\n      f_discharge: '-2 days'\n",
        encoding="utf-8",
    )
    processor = SimpleNamespace(sim_source="CoLM2024", item="Streamflow")
    data = xr.DataArray([3.0], dims="time", coords={"time": [pd.Timestamp("2001-01-03 12:00")]}, name="f_discharge")

    adjusted = TimeAdjustmentMixin.apply_model_specific_time_adjustment(processor, data, "sim", 2001, 2001, "Day")

    assert pd.Timestamp(adjusted.time.values[0]) == pd.Timestamp("2001-01-01 12:30")


@pytest.mark.parametrize("varname", ["outflw", "f_discharge"])
@pytest.mark.parametrize(
    "resolution, endpoint, midpoint, expected",
    [
        ("h", "2001-01-01 01:00", "2001-01-01 00:30", "2001-01-01 00:30"),
        ("D", "2001-01-02 00:00", "2001-01-01 12:00", "2001-01-01 12:00"),
        ("ME", "2001-02-01 00:00", "2001-01-17 00:00", "2001-01-15 00:00"),
    ],
)
def test_colm_time_integrity_preserves_first_period(varname, resolution, endpoint, midpoint, expected):
    from openbench.data.processing import BaseDatasetProcessing

    processor = object.__new__(BaseDatasetProcessing)
    processor.sim_source = "CoLM2024"
    processor.sim_data_type = "grid"
    processor.item = "Streamflow"
    raw_time = endpoint if varname == "outflw" else midpoint
    data = xr.DataArray([3.0], dims="time", coords={"time": [pd.Timestamp(raw_time)]}, name=varname)

    adjusted = processor.check_dataset_time_integrity(data, 2001, 2001, resolution, "sim")

    assert float(adjusted.sel(time=expected)) == 3.0
