"""File lookup uses the same compute override priority as execution."""

from types import SimpleNamespace

import pytest
import xarray as xr

from openbench.data._processing_selection import SelectionMixin
from openbench.data.registry import manager


@pytest.mark.parametrize("groupby", ["Year", "Single"])
@pytest.mark.parametrize("override", ["datasource", "source", None])
@pytest.mark.parametrize("catalog", ["model", "reference"])
def test_lookup_uses_only_effective_compute(tmp_path, monkeypatch, groupby, override, catalog):
    mapping = SimpleNamespace(varname="precip", fallbacks=[], compute="ds['snow']")
    profile = SimpleNamespace(variables={"Precipitation": mapping})
    registry = SimpleNamespace(
        get_model=lambda name: profile if catalog == "model" else None,
        get_reference=lambda name: profile if catalog == "reference" else None,
    )
    monkeypatch.setattr(manager, "get_registry", lambda: registry)
    processor = SelectionMixin()
    processor.item = "precipitation"
    processor.sim_source = "source"
    if override == "datasource":
        processor.sim_compute = "ds['rain']"
        processor.source_compute = "ds['hail']"
    elif override == "source":
        processor.source_compute = "ds['rain']"
    variable = "rain" if override else "snow"
    path = tmp_path / f"{variable}_2001.nc"
    xr.Dataset({variable: ("time", [1.0])}).to_netcdf(path)

    if groupby == "Year":
        selected = processor._find_data_files(str(tmp_path), "precip_", 2001, "", varname=["precip"])
    else:
        selected = processor._find_single_file(str(tmp_path), "precip_", "", varname=["precip"])

    assert selected == [str(path)]
    assert processor._compute_expressions_for_file_lookup("sim") == [f"ds['{variable}']"]
    assert processor._candidate_varnames_for_file_lookup(["precip"], "sim") == ["precip", variable]
