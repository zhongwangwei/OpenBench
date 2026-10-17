import numpy as np
import pandas as pd
import xarray as xr

from openbench.data._processing_grid_core import GridProcessingCoreMixin


class _Processor(GridProcessingCoreMixin):
    pass


def test_station_grid_year_selection_ignores_stale_scratch_years(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for year in (2001, 2002, 2003, 2004, 2005):
        (scratch / f"sim_case_{year}.nc").touch()

    processor = _Processor()
    processor.casedir = str(tmp_path)
    processor.minyear = 2001
    processor.maxyear = 2002
    processor.ref_data_type = "stn"
    processor.sim_data_type = "grid"
    processor.combine_and_save_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("station workflows must not build a flat NetCDF")
    )

    files = processor.remap_and_combine_data(
        {
            "datasource": "sim",
            "prefix": "case_",
            "suffix": "",
            "varname": ["flow"],
        }
    )

    assert files == [str(scratch / "sim_case_2001.nc"), str(scratch / "sim_case_2002.nc")]


def test_station_grid_extracts_directly_from_year_files_without_flat_output(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source_files = []
    for year in (2001, 2002):
        path = scratch / f"sim_case_{year}.nc"
        xr.Dataset(
            {
                "flow": (
                    ("time", "lat", "lon"),
                    np.arange(9, dtype=np.float32).reshape(1, 3, 3) + year,
                )
            },
            coords={
                "time": pd.date_range(f"{year}-01-01", periods=1),
                "lat": [0.0, 1.0, 2.0],
                "lon": [10.0, 20.0, 30.0],
            },
        ).to_netcdf(path)
        source_files.append(str(path))

    processor = _Processor()
    processor.casedir = str(tmp_path)
    processor.item = "Streamflow"
    processor.ref_source = "Ref"
    processor.sim_source = "Sim"
    processor.num_cores = 1
    processor.ref_data_type = "stn"
    processor.sim_data_type = "grid"
    processor.station_list = pd.DataFrame(
        [
            {
                "ID": "A",
                "use_syear": 2001,
                "use_eyear": 2002,
                "ref_lat": 1.0,
                "ref_lon": 20.0,
            }
        ]
    )
    flat_output = tmp_path / "data" / "flat.nc"
    processor.get_output_filename = lambda _params: str(flat_output)

    seen = []

    def record_dataset(_datasource, dataset, _station_list, _index):
        seen.append(dict(dataset.sizes))

    processor._extract_stn_parallel = record_dataset
    processor.extract_station_data({"datasource": "sim"}, source_files=source_files)

    assert seen == [{"time": 2, "lat": 1, "lon": 1}]
    assert not flat_output.exists()
