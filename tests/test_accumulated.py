"""Running totals declared with `accumulated` are turned into per-step amounts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.config.loader import ConfigError, _validated_variables_mapping
from openbench.data._processing_transforms import ProcessingTransformMixin, deaccumulate


def _year_to_date(monthly: np.ndarray, start: str = "2004-01-31") -> xr.DataArray:
    time = pd.date_range(start, periods=len(monthly), freq="ME")
    totals = np.concatenate([np.cumsum(monthly[time.year == year]) for year in sorted(set(time.year))])
    return xr.DataArray(totals, dims="time", coords={"time": time}, name="f_sum_irrig", attrs={"units": "kg/m2"})


def test_year_mode_restores_monthly_amounts_across_the_january_reset():
    monthly = np.array([5, 0, 10, 20, 0, 0, 30, 0, 0, 0, 0, 0, 7, 0, 3], dtype=float)

    amounts = deaccumulate(_year_to_date(monthly), "year")

    np.testing.assert_allclose(amounts.values, monthly)
    assert amounts.name == "f_sum_irrig"
    assert amounts.attrs == {"units": "kg/m2"}


def test_year_mode_leaves_a_year_that_starts_after_january_missing():
    totals = _year_to_date(np.array([5, 0, 10, 20, 0, 0], dtype=float)).isel(time=slice(3, None))

    amounts = deaccumulate(totals, "year")

    assert np.isnan(amounts.values[0])
    np.testing.assert_allclose(amounts.values[1:], [0.0, 0.0])


def test_run_mode_differences_steps_and_drops_resets():
    time = pd.date_range("2004-06-01", periods=6, freq="h")
    rain = xr.Dataset({"pr": ("time", [10.0, 10.5, 12.0, 0.1, 0.6, 0.6])}, coords={"time": time})

    amounts = deaccumulate(rain, "run")["pr"].values

    np.testing.assert_allclose(amounts, [np.nan, 0.5, 1.5, np.nan, 0.5, 0.0])


def test_unknown_accumulation_mode_is_rejected():
    with pytest.raises(ValueError, match="accumulation mode"):
        deaccumulate(_year_to_date(np.ones(3)), "monthly")
    with pytest.raises(ConfigError, match="accumulated must be 'year' or 'run'"):
        _validated_variables_mapping(
            {"Total_Irrigation_Amount": {"accumulated": "monthly"}}, "simulation.CaseA.variables"
        )


def test_processor_reads_accumulated_by_source_name_before_unit_conversion():
    class Processor(ProcessingTransformMixin):
        item = "Total_Irrigation_Amount"
        sim_source = "CaseA"
        CaseA_accumulated = "year"

    totals = _year_to_date(np.array([5, 0, 10], dtype=float))

    amounts = Processor()._deaccumulate_if_configured(totals, "sim")
    untouched = Processor()._deaccumulate_if_configured(totals, "ref")

    np.testing.assert_allclose(amounts.values, [5.0, 0.0, 10.0])
    assert untouched is totals

