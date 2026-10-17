"""Grid preprocessing finds its regridded scratch files after a compute renames the variable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _frame(times):
    return xr.Dataset(
        {"raw": (("time", "lat", "lon"), np.ones((len(times), 2, 2), dtype="float32"))},
        coords={"time": times, "lat": [0.5, 1.5], "lon": [0.5, 1.5]},
    )


def _write_inputs(src, groupby):
    if groupby == "single":
        _frame(pd.date_range("2001-01-01", "2002-12-31", freq="D")).to_netcdf(src / "run_all.nc")
        return "run_all"
    for year in (2001, 2002):
        if groupby == "year":
            _frame(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")).to_netcdf(src / f"run_{year}.nc")
        else:
            for month in range(1, 13):
                times = pd.date_range(f"{year}-{month:02d}-01", periods=28, freq="D")
                _frame(times).to_netcdf(src / f"run_{year}-{month:02d}.nc")
    return "run_"


# The compute renames sim_varname to the item in the process that evaluates it:
# the main one for Single files or a single core, a worker otherwise.
@pytest.mark.parametrize("cores", [1, 2])
@pytest.mark.parametrize("groupby", ["single", "year", "month"])
def test_computed_grid_variable_reaches_the_flat_file(tmp_path, groupby, cores):
    from openbench.data.processing import DatasetProcessing

    src, case = tmp_path / "src", tmp_path / "case"
    src.mkdir()
    (case / "scratch").mkdir(parents=True)
    (case / "data").mkdir()
    prefix = _write_inputs(src, groupby)

    processor = object.__new__(DatasetProcessing)
    processor.__dict__.update(
        casedir=str(case),
        minyear=2001,
        maxyear=2002,
        syear=2001,
        eyear=2002,
        compare_tim_res="ME",
        compare_grid_res=1.0,
        num_cores=cores,
        debug_mode=False,
        item="Total_Runoff",
        sim_source="Sim",
        ref_source="Ref",
        sim_data_type="grid",
        ref_data_type="grid",
        sim_varname=["raw"],
        sim_varunit="mm day-1",
        sim_compute="ds['raw'] * 2",
        coordinate_map={},
        timezone=0,
        sim_model="model",
        sim_dir=str(src),
        sim_prefix=prefix,
        sim_suffix="",
        sim_data_groupby=groupby,
        sim_tim_res="D",
        sim_syear=2001,
        sim_eyear=2002,
        min_lat=0,
        max_lat=2,
        min_lon=0,
        max_lon=2,
        regrid_backend="openbench_conservative",
        system_resources={"available_memory_gb": 8.0, "cpu_count": 2},
        time_alignment="intersection",
    )

    processor._preprocess("sim")

    with xr.open_dataset(case / "data" / "Total_Runoff_sim_Sim_raw.nc") as flat:
        assert list(flat.data_vars) == ["Total_Runoff"]
        np.testing.assert_allclose(float(flat["Total_Runoff"].mean()), 2.0)
    assert not list((case / "scratch").glob("sim_*_remap_*.nc"))
    assert processor.sim_varname == ["raw"]
