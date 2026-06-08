"""Year splitting/combining and file-group preprocessing helpers."""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import List

import xarray as xr
from joblib import Parallel, delayed

from openbench.data._processing_utils import performance_monitor
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def _processing_attr(name: str, fallback=None):
    """Resolve monkeypatch-friendly attributes from openbench.data.processing."""
    processing = sys.modules.get("openbench.data.processing")
    if processing is not None and hasattr(processing, name):
        return getattr(processing, name)
    return fallback


class YearlyPreprocessingMixin:
    @performance_monitor
    def split_year(
        self, ds: xr.Dataset, casedir: str, suffix: str, prefix: str, use_syear: int, use_eyear: int, datasource: str
    ) -> None:
        def save_year(casedir: str, suffix: str, prefix: str, ds: xr.Dataset, year: int) -> None:
            ds_year = None
            try:
                ds_year = ds.sel(time=slice(f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59"))
                ds_year.attrs = {}
                output_file = os.path.join(casedir, "scratch", f"{datasource}_{prefix}{year}{suffix}.nc")
                _write_netcdf_atomic(ds_year, output_file, compression=False)
                logging.debug(f"Saved {output_file}")
            finally:
                # Clean up memory
                if ds_year is not None and hasattr(ds_year, "close"):
                    ds_year.close()
                gc.collect()

        try:
            # Calculate dataset size in GB
            dataset_size_gb = ds.nbytes / (1024**3)

            # Update number of cores based on dataset size
            optimal_cores = min(self.get_optimal_cores(dataset_size_gb), self.num_cores)
            logging.debug(f"Using {optimal_cores} cores for splitting years")

            years = range(use_syear, use_eyear + 1)
            _processing_attr("Parallel", Parallel)(n_jobs=optimal_cores)(
                _processing_attr("delayed", delayed)(save_year)(casedir, suffix, prefix, ds, year) for year in years
            )
        finally:
            # Ensure main dataset is closed
            if hasattr(ds, "close"):
                ds.close()
            gc.collect()

    @performance_monitor
    def combine_year(
        self,
        year: int,
        casedir: str,
        dirx: str,
        suffix: str,
        prefix: str,
        varname: List[str],
        datasource: str,
        tim_res: str,
    ) -> xr.Dataset:
        var_files = self._find_data_files(dirx, prefix, year, suffix, datasource, varname=varname)

        # Verify files were found
        if not var_files:
            raise FileNotFoundError(
                f"No data files found for year {year} in {dirx} (prefix='{prefix}', suffix='{suffix}')"
            )

        datasets = []
        try:
            for file in var_files:
                ds = self.select_var(year, year, tim_res, file, varname, datasource)
                datasets.append(ds)
            data0 = xr.concat(datasets, dim="time").sortby("time")
            return data0
        finally:
            # Clean up memory
            for ds in datasets:
                if hasattr(ds, "close"):
                    ds.close()
            gc.collect()

    def check_file_exist(self, file: str) -> str:
        if not os.path.exists(file):
            logging.error(f"File '{file}' not found.")
            raise FileNotFoundError(f"File '{file}' not found.")
        return file

    def check_all(
        self,
        dirx: str,
        syear: int,
        eyear: int,
        tim_res: str,
        varunit: str,
        varname: List[str],
        groupby: str,
        casedir: str,
        suffix: str,
        prefix: str,
        datasource: str,
    ) -> None:
        if groupby == "single":
            self.preprocess_single_file(
                dirx, syear, eyear, tim_res, varunit, varname, casedir, suffix, prefix, datasource
            )
        elif groupby != "year":
            self.preprocess_non_yearly_files(
                dirx, syear, eyear, tim_res, varunit, varname, casedir, suffix, prefix, datasource
            )
        else:
            self.preprocess_yearly_files(
                dirx, syear, eyear, tim_res, varunit, varname, casedir, suffix, prefix, datasource
            )

    @performance_monitor
    def preprocess_single_file(
        self,
        dirx: str,
        syear: int,
        eyear: int,
        tim_res: str,
        varunit: str,
        varname: List[str],
        casedir: str,
        suffix: str,
        prefix: str,
        datasource: str,
    ) -> None:
        logging.debug("The dataset groupby is Single --> split it to Year")
        varfile = self._find_single_file(dirx, prefix, suffix, datasource, varname=varname)
        ds = self.select_var(syear, eyear, tim_res, varfile, varname, datasource)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        ds = self.select_timerange(ds, self.minyear, self.maxyear)
        # Use updated varunit from filter if available (filter may have modified it)
        current_varunit = getattr(self, f"{datasource}_varunit", varunit)
        ds, varunit = self.process_units(ds, current_varunit)
        self.split_year(ds, casedir, suffix, prefix, self.minyear, self.maxyear, datasource)

    @performance_monitor
    def preprocess_non_yearly_files(
        self,
        dirx: str,
        syear: int,
        eyear: int,
        tim_res: str,
        varunit: str,
        varname: List[str],
        casedir: str,
        suffix: str,
        prefix: str,
        datasource: str,
    ) -> None:
        logging.debug("The dataset groupby is not Year --> combine it to Year")
        ds = self.combine_year(syear, casedir, dirx, suffix, prefix, varname, datasource, tim_res)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        # Use updated varunit from filter if available (filter may have modified it)
        current_varunit = getattr(self, f"{datasource}_varunit", varunit)
        ds, varunit = self.process_units(ds, current_varunit)
        ds = self.select_timerange(ds, syear, eyear)
        _write_netcdf_atomic(
            ds,
            os.path.join(casedir, "scratch", f"{datasource}_{prefix}{syear}{suffix}.nc"),
            compression=False,
        )

    @performance_monitor
    def preprocess_yearly_files(
        self,
        dirx: str,
        syear: int,
        eyear: int,
        tim_res: str,
        varunit: str,
        varname: List[str],
        casedir: str,
        suffix: str,
        prefix: str,
        datasource: str,
    ) -> None:
        # Use fallback-aware file search (supports prefix_fallback for CaMa etc.)
        found_files = self._find_data_files(dirx, prefix, syear, suffix, datasource, varname=varname)
        if not found_files:
            raise FileNotFoundError(f"No data files found for year {syear} with prefix '{prefix}'")
        if len(found_files) > 1:
            logging.info(f"Found {len(found_files)} files for year {syear}, merging with open_mfdataset")
        varfiles = found_files[0] if len(found_files) == 1 else found_files
        ds = self.select_var(syear, eyear, tim_res, varfiles, varname, datasource)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        # Use updated varunit from filter if available (filter may have modified it)
        current_varunit = getattr(self, f"{datasource}_varunit", varunit)
        ds, varunit = self.process_units(ds, current_varunit)
        ds = self.select_timerange(ds, syear, eyear)
        _write_netcdf_atomic(
            ds,
            os.path.join(casedir, "scratch", f"{datasource}_{prefix}{syear}{suffix}.nc"),
            compression=False,
        )
