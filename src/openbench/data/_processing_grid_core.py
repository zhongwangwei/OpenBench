"""Grid-data processing mixin and regridding helpers for OpenBench datasets."""

from __future__ import annotations

import gc
import glob
import logging
import os
import sys
from typing import Any, Dict, List

import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed

from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic

try:
    from openbench.util.dataset_loader import cached_glob, write_mfdataset_atomic as write_mfdataset_chunked_atomic
except ImportError:  # pragma: no cover - mirrors processing.py fallback
    cached_glob = lambda pattern, **kwargs: sorted(glob.glob(pattern))

    def write_mfdataset_chunked_atomic(paths, output_path, *, sortby=None, compression=None, **kwargs):
        with xr.open_mfdataset(paths, **kwargs) as ds:
            if sortby is not None:
                ds = ds.sortby(sortby)
            _write_netcdf_atomic(ds, output_path, compression=compression)


REGRID_ALGORITHM_VERSION = "2026-05-27.regrid-v2"
REGRID_BACKENDS = {
    "openbench_conservative",
    "cdo_remapcon",
    "xesmf_conservative",
    "basic_interpolation",
}


def _processing_attr(name, fallback):
    processing = sys.modules.get("openbench.data.processing")
    return getattr(processing, name, fallback) if processing is not None else fallback


def _parallel():
    return _processing_attr("Parallel", Parallel)


def _delayed():
    return _processing_attr("delayed", delayed)


class GridProcessingCoreMixin:
    """Split grid processing helpers."""

    def process_grid_data(self, data_params: Dict[str, Any]) -> None:
        try:
            self.prepare_grid_data(data_params)
            self.remap_and_combine_data(data_params)
            self.extract_station_data_if_needed(data_params)
        finally:
            gc.collect()

    def prepare_grid_data(self, data_params: Dict[str, Any]) -> None:
        if data_params["data_groupby"] == "single":
            self.process_single_file(data_params)
        elif data_params["data_groupby"] != "year":
            self.process_non_yearly_files(data_params)
        else:
            self.process_yearly_files(data_params)

    def process_single_file(self, data_params: Dict[str, Any]) -> None:
        self.check_all(
            data_params["data_dir"],
            data_params["syear"],
            data_params["eyear"],
            data_params["tim_res"],
            data_params["varunit"],
            data_params["varname"],
            "single",
            self.casedir,
            data_params["suffix"],
            data_params["prefix"],
            data_params["datasource"],
        )
        setattr(self, f"{data_params['datasource']}_data_groupby", "year")

    def process_non_yearly_files(self, data_params: Dict[str, Any]) -> None:
        logging.debug("Combining data to yearly files...")
        years = range(self.minyear, self.maxyear + 1)
        _parallel()(n_jobs=self.num_cores)(
            _delayed()(self.check_all)(
                data_params["data_dir"],
                year,
                year,
                data_params["tim_res"],
                data_params["varunit"],
                data_params["varname"],
                data_params["data_groupby"],
                self.casedir,
                data_params["suffix"],
                data_params["prefix"],
                data_params["datasource"],
            )
            for year in years
        )

    def process_yearly_files(self, data_params: Dict[str, Any]) -> None:
        years = range(self.minyear, self.maxyear + 1)
        _parallel()(n_jobs=self.num_cores)(
            _delayed()(self.check_all)(
                data_params["data_dir"],
                year,
                year,
                data_params["tim_res"],
                data_params["varunit"],
                data_params["varname"],
                data_params["data_groupby"],
                self.casedir,
                data_params["suffix"],
                data_params["prefix"],
                data_params["datasource"],
            )
            for year in years
        )

    def remap_and_combine_data(self, data_params: Dict[str, Any]) -> None:
        data_dir = os.path.join(self.casedir, "scratch")
        years = range(self.minyear, self.maxyear + 1)

        data_source = data_params["datasource"]
        if data_source not in ["ref", "sim"]:
            logging.error(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")
            raise ValueError(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")

        if self.ref_data_type != "stn" and self.sim_data_type != "stn":
            _parallel()(n_jobs=self.num_cores)(
                _delayed()(self._make_grid_parallel)(
                    data_source, data_params["suffix"], data_params["prefix"], data_dir, year
                )
                for year in years
            )
            # Force refresh since files were just created by parallel processing
            var_files = cached_glob(
                os.path.join(self.casedir, "scratch", f"{data_source}_{data_params['varname'][0]}_remap_*.nc"),
                force_refresh=True,
            )
        else:
            var_files = cached_glob(
                os.path.join(data_dir, f"{data_source}_{data_params['prefix']}*{data_params['suffix']}.nc")
            )

        self.combine_and_save_data(var_files, data_params)

    def combine_and_save_data(self, var_files: List[str], data_params: Dict[str, Any]) -> None:
        output_file = self.get_output_filename(data_params)
        batch_dir = os.path.join(self.casedir, "scratch", "mfdataset_batches")
        # Try to use ProgressBar, but fall back to silent mode if it fails (e.g., non-interactive environment)
        try:
            with ProgressBar():
                write_mfdataset_chunked_atomic(
                    var_files,
                    output_file,
                    combine="by_coords",
                    sortby="time",
                    batch_dir=batch_dir,
                    compression=False,
                )
        except (OSError, IOError, BrokenPipeError):
            # ProgressBar failed (likely non-interactive environment), save without progress bar
            write_mfdataset_chunked_atomic(
                var_files,
                output_file,
                combine="by_coords",
                sortby="time",
                batch_dir=batch_dir,
                compression=False,
            )
        gc.collect()  # Add garbage collection after saving combined data

        # Only cleanup temp files if we created them (i.e., when processing grid data)
        if self.ref_data_type != "stn" and self.sim_data_type != "stn":
            self.cleanup_temp_files(data_params)

    def get_output_filename(self, data_params: Dict[str, Any]) -> str:
        if data_params["datasource"] == "ref":
            return os.path.join(
                self.casedir,
                "data",
                f"{self.item}_{data_params['datasource']}_{self.ref_source}_{data_params['varname'][0]}.nc",
            )
        else:
            return os.path.join(
                self.casedir,
                "data",
                f"{self.item}_{data_params['datasource']}_{self.sim_source}_{data_params['varname'][0]}.nc",
            )

    def cleanup_temp_files(self, data_params: Dict[str, Any]) -> None:
        """Clean up temporary files, silently skipping non-existent files."""
        failed_removals = []
        for year in range(self.minyear, self.maxyear + 1):
            temp_file = os.path.join(
                self.casedir, "scratch", f"{data_params['datasource']}_{data_params['varname'][0]}_remap_{year}.nc"
            )
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logging.debug(f"Removed temporary file: {temp_file}")
                except OSError as e:
                    failed_removals.append((temp_file, str(e)))

        # Only warn if we actually failed to remove existing files
        if failed_removals:
            logging.warning(f"Failed to remove {len(failed_removals)} temporary file(s)")
            for file_path, error in failed_removals:
                logging.debug(f"  Failed to remove {file_path}: {error}")

    def extract_station_data_if_needed(self, data_params: Dict[str, Any]) -> None:
        if self.ref_data_type == "stn" or self.sim_data_type == "stn":
            logging.debug(f"Extracting station data for {data_params['datasource']} data")
            self.extract_station_data(data_params)

    def extract_station_data(self, data_params: Dict[str, Any]) -> None:
        output_file = self.get_output_filename(data_params)
        try:
            with xr.open_dataset(output_file) as ds:
                ds = Convert_Type.convert_nc(ds)
                if hasattr(ds, "load"):
                    ds = ds.load()
                _parallel()(n_jobs=self.num_cores)(
                    _delayed()(self._extract_stn_parallel)(data_params["datasource"], ds, self.station_list, i)
                    for i in range(len(self.station_list["ID"]))
                )
                gc.collect()  # Add garbage collection after extracting station data
        finally:
            # Remove the consumed flat NC even if station extraction raised mid-loop.
            # Without this, a partial extraction leaves the flat behind and the next
            # rescan thinks the prep is done. The runner-side backup-restore (in
            # runner.local._catalog_write_lock + _backup_then_write) preserves the
            # flat across this consumption when needed for downstream grid eval.
            try:
                if os.path.exists(output_file):
                    os.remove(output_file)
            except OSError as e:
                logging.debug("Could not remove flat NC %s: %s", output_file, e)

    def _make_grid_parallel(self, data_source: str, suffix: str, prefix: str, dirx: str, year: int) -> None:
        try:
            if data_source not in ["ref", "sim"]:
                logging.error(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")
                raise ValueError(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")

            var_file = os.path.join(dirx, f"{data_source}_{prefix}{year}{suffix}.nc")
            if self.debug_mode:
                logging.debug(f"Processing {var_file} for year {year}")
                logging.debug(f"Processing {data_source} data for year {year}")

            with xr.open_dataset(var_file) as data:
                data = Convert_Type.convert_nc(data)
                data = self.preprocess_grid_data(data)
                # 1. Clip to evaluation region to reduce memory
                from openbench.data.coordinates import find_lat_name, find_lon_name

                lat_name = find_lat_name(data.dims) or find_lat_name(data.coords) or "lat"
                lon_name = find_lon_name(data.dims) or find_lon_name(data.coords) or "lon"
                if lat_name in data and len(data[lat_name]) > 1:
                    lat_vals = data[lat_name].values
                    if lat_vals[0] > lat_vals[-1]:
                        data = data.sel({lat_name: slice(self.max_lat + 1, self.min_lat - 1)})
                    else:
                        data = data.sel({lat_name: slice(self.min_lat - 1, self.max_lat + 1)})
                if lon_name in data:
                    data = data.sel({lon_name: slice(self.min_lon - 1, self.max_lon + 1)})
                # 2. Resample BEFORE remap: e.g. daily→monthly first, then remap
                #    much cheaper than remap daily then resample
                if not self._is_climatology_mode():
                    data = self._resample_to_compare_resolution(data, f"{data_source} grid data")
                # 3. Remap to target grid
                remapped_data = self.remap_data(data)
                self.save_remapped_data(remapped_data, data_source, year)
        finally:
            gc.collect()

    def preprocess_grid_data(self, data: xr.Dataset) -> xr.Dataset:
        # Check if lon and lat are 2D
        data = self.check_coordinate(data)
        if data["lon"].ndim == 2 and data["lat"].ndim == 2:
            try:
                from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_xesmf

                data = convert_to_wgs84_xesmf(data, self.compare_grid_res)
            except (ImportError, ValueError, RuntimeError) as e:
                logging.debug(f"xesmf regridding failed, falling back to scipy: {e}")
                from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_scipy

                data = convert_to_wgs84_scipy(data, self.compare_grid_res)

        return self._normalize_longitude_axis(data)
