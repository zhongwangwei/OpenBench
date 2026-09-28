"""Grid-data processing mixin and regridding helpers for OpenBench datasets."""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed

from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic

try:
    from openbench.util.dataset_loader import (
        open_mfdataset as open_mfdataset_chunked,
        write_mfdataset_atomic as write_mfdataset_chunked_atomic,
    )
except ImportError:  # pragma: no cover - mirrors processing.py fallback
    open_mfdataset_chunked = xr.open_mfdataset

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


def _regrid_worker_budget(
    *,
    requested: int,
    year_count: int,
    available_memory_gb: float,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    time_length: int,
    data_bytes: int,
    backend: str,
) -> tuple[int, str]:
    """Return a conservative, explainable process budget for yearly regridding."""
    requested = max(1, int(requested))
    year_count = max(1, int(year_count))
    source_cells = max(1, int(source_shape[0]) * int(source_shape[1]))
    target_cells = max(1, int(target_shape[0]) * int(target_shape[1]))
    data_bytes = max(int(data_bytes), source_cells * max(1, int(time_length)) * 4)
    target_bytes = int(data_bytes * target_cells / source_cells)

    # The dense fallback needs both source/target work arrays plus the two
    # separable weight matrices. Budget for it even when sparse contraction is
    # available so a runtime fallback cannot overcommit the host.
    dense_weight_bytes = (source_shape[0] * target_shape[0] + source_shape[1] * target_shape[1]) * 8
    estimated_peak = data_bytes * 3 + target_bytes * 4 + dense_weight_bytes * 2
    usable_memory = max(1, int(float(available_memory_gb) * (1024**3) * 0.5))
    memory_cap = max(1, usable_memory // max(1, estimated_peak))

    backend_cap = 4 if backend == "openbench_conservative" else 8
    if source_cells >= 10_000_000 or estimated_peak >= 4 * 1024**3:
        size_cap = 1
    elif source_cells >= 1_000_000 or estimated_peak >= 1024**3:
        size_cap = 2
    else:
        size_cap = 8

    workers = max(1, min(requested, year_count, memory_cap, backend_cap, size_cap))
    reason = (
        f"requested={requested} years={year_count} backend_cap={backend_cap} "
        f"memory_cap={memory_cap} size_cap={size_cap} "
        f"estimated_peak={estimated_peak / 1024**2:.1f}MiB"
    )
    return workers, reason


class GridProcessingCoreMixin:
    """Split grid processing helpers."""

    def process_grid_data(self, data_params: Dict[str, Any]) -> None:
        try:
            self.prepare_grid_data(data_params)
            yearly_files = self.remap_and_combine_data(data_params)
            self.extract_station_data_if_needed(data_params, yearly_files)
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

    def remap_and_combine_data(self, data_params: Dict[str, Any]) -> List[str]:
        data_dir = os.path.join(self.casedir, "scratch")
        years = list(range(self.minyear, self.maxyear + 1))

        data_source = data_params["datasource"]
        if data_source not in ["ref", "sim"]:
            logging.error(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")
            raise ValueError(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")

        if self.ref_data_type != "stn" and self.sim_data_type != "stn":
            regrid_workers, worker_reason = self._get_regrid_worker_count(
                years,
                data_source=data_source,
                prefix=data_params["prefix"],
                suffix=data_params["suffix"],
                data_dir=data_dir,
            )
            logging.info(
                "[REGRID_PERF] yearly workers=%d requested=%d reason=%s",
                regrid_workers,
                self.num_cores,
                worker_reason,
            )
            _parallel()(n_jobs=regrid_workers)(
                _delayed()(self._make_grid_parallel)(
                    data_source,
                    data_params["suffix"],
                    data_params["prefix"],
                    data_dir,
                    year,
                    regrid_workers,
                )
                for year in years
            )
            var_files = [
                os.path.join(data_dir, f"{data_source}_{data_params['varname'][0]}_remap_{year}.nc") for year in years
            ]
        else:
            prefix = data_params.get("prefix") or ""
            suffix = data_params.get("suffix") or ""
            var_files = [os.path.join(data_dir, f"{data_source}_{prefix}{year}{suffix}.nc") for year in years]

        missing_files = [path for path in var_files if not os.path.isfile(path)]
        if missing_files:
            examples = ", ".join(missing_files[:3])
            raise FileNotFoundError(
                f"Missing {len(missing_files)} configured-year scratch file(s) for {data_source}: {examples}"
            )

        if self.ref_data_type == "stn" or self.sim_data_type == "stn":
            logging.info(
                "Station-involved workflow: extracting directly from %d configured-year file(s); "
                "skipping flat NetCDF combine",
                len(var_files),
            )
            return var_files

        self.combine_and_save_data(var_files, data_params)
        return var_files

    def _get_regrid_worker_count(
        self,
        years: List[int],
        *,
        data_source: str,
        prefix: str,
        suffix: str,
        data_dir: str,
    ) -> tuple[int, str]:
        requested = max(1, int(getattr(self, "num_cores", 1) or 1))
        backend = str(getattr(self, "regrid_backend", "openbench_conservative") or "openbench_conservative").lower()
        available_memory_gb = float(getattr(self, "system_resources", {}).get("available_memory_gb", 4.0))
        source_shape = (1, 1)
        time_length = 1
        data_bytes = 4

        if years:
            sample_file = os.path.join(data_dir, f"{data_source}_{prefix}{years[0]}{suffix}.nc")
            try:
                from openbench.data.coordinates import find_lat_name, find_lon_name

                with xr.open_dataset(sample_file, decode_times=False) as sample:
                    lat_name = find_lat_name(sample.dims) or find_lat_name(sample.coords) or "lat"
                    lon_name = find_lon_name(sample.dims) or find_lon_name(sample.coords) or "lon"
                    source_shape = (int(sample.sizes.get(lat_name, 1)), int(sample.sizes.get(lon_name, 1)))
                    time_length = int(sample.sizes.get("time", 1))
                    data_bytes = int(sample.nbytes)
            except (OSError, ValueError, KeyError) as exc:
                logging.debug("Could not inspect yearly regrid workload %s: %s", sample_file, exc)

        target = self.create_target_grid()
        target_shape = (int(target.sizes.get("lat", 1)), int(target.sizes.get("lon", 1)))
        return _regrid_worker_budget(
            requested=requested,
            year_count=len(years),
            available_memory_gb=available_memory_gb,
            source_shape=source_shape,
            target_shape=target_shape,
            time_length=time_length,
            data_bytes=data_bytes,
            backend=backend,
        )

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
                    # None defers to OPENBENCH_NETCDF_COMPRESSION so the flat
                    # sim/ref NetCDF can be compressed (f_discharge is ~47 GB
                    # uncompressed and dominates run time).
                    compression=None,
                )
        except (OSError, IOError, BrokenPipeError):
            write_mfdataset_chunked_atomic(
                var_files,
                output_file,
                combine="by_coords",
                sortby="time",
                batch_dir=batch_dir,
                compression=None,
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

    def extract_station_data_if_needed(
        self,
        data_params: Dict[str, Any],
        yearly_files: List[str] | None = None,
    ) -> None:
        if self.ref_data_type == "stn" or self.sim_data_type == "stn":
            logging.debug(f"Extracting station data for {data_params['datasource']} data")
            self.extract_station_data(data_params, source_files=yearly_files)

    def extract_station_data(
        self,
        data_params: Dict[str, Any],
        source_files: List[str] | None = None,
    ) -> None:
        output_file = self.get_output_filename(data_params)
        try:
            if source_files:
                dataset_context = open_mfdataset_chunked(source_files, combine="by_coords")
            else:
                dataset_context = xr.open_dataset(output_file)

            with dataset_context as ds:
                ds = Convert_Type.convert_nc(ds)
                if source_files:
                    ds = self._subset_grid_for_station_extraction(ds, data_params["datasource"])
                    if "time" in ds.coords:
                        ds = ds.sortby("time")
                if hasattr(ds, "load"):
                    ds = ds.load()
                _parallel()(n_jobs=self.num_cores)(
                    _delayed()(self._extract_stn_parallel)(data_params["datasource"], ds, self.station_list, i)
                    for i in range(len(self.station_list["ID"]))
                )
                gc.collect()  # Add garbage collection after extracting station data
        finally:
            if not source_files:
                # Remove the consumed flat NC even if station extraction raised mid-loop.
                # Direct-from-year extraction never creates this large intermediate.
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except OSError as e:
                    logging.debug("Could not remove flat NC %s: %s", output_file, e)

    def _subset_grid_for_station_extraction(self, dataset: xr.Dataset, datasource: str) -> xr.Dataset:
        """Load only the grid rows/columns needed by the configured stations."""
        from openbench.data.coordinates import find_lat_name, find_lon_name
        from openbench.data._processing_station_extract import _cyclic_lon_delta

        all_names = set(dataset.coords) | set(dataset.dims)
        lat_coord = find_lat_name(all_names) or "lat"
        lon_coord = find_lon_name(all_names) or "lon"
        if lat_coord not in dataset.coords or lon_coord not in dataset.coords:
            return dataset
        if dataset[lat_coord].ndim != 1 or dataset[lon_coord].ndim != 1:
            return dataset

        lat_values = dataset[lat_coord].values
        lon_values = dataset[lon_coord].values
        lat_indices: set[int] = set()
        lon_indices: set[int] = set()
        for _, station in self.station_list.iterrows():
            if datasource == "ref":
                lat_key, lon_key = "sim_lat", "sim_lon"
                fallback_lat_key, fallback_lon_key = "ref_lat", "ref_lon"
            else:
                lat_key, lon_key = "ref_lat", "ref_lon"
                fallback_lat_key, fallback_lon_key = "sim_lat", "sim_lon"

            if lat_key not in station or pd.isna(station.get(lat_key)):
                lat_key = fallback_lat_key
            if lon_key not in station or pd.isna(station.get(lon_key)):
                lon_key = fallback_lon_key

            target_lat = float(station[lat_key])
            target_lon = float(station[lon_key])
            lat_indices.add(int(np.argmin(np.abs(lat_values - target_lat))))
            lon_indices.add(int(np.argmin(_cyclic_lon_delta(lon_values, target_lon))))

        if not lat_indices or not lon_indices:
            return dataset
        return dataset.isel(
            {
                lat_coord: sorted(lat_indices),
                lon_coord: sorted(lon_indices),
            }
        )

    def _make_grid_parallel(
        self,
        data_source: str,
        suffix: str,
        prefix: str,
        dirx: str,
        year: int,
        regrid_workers: int | None = None,
    ) -> None:
        total_start = time.perf_counter()
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
                read_seconds = time.perf_counter() - total_start
                source_shape = (int(data.sizes.get("lat", 0)), int(data.sizes.get("lon", 0)))
                # 1. Clip to evaluation region to reduce memory
                crop_start = time.perf_counter()
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
                crop_seconds = time.perf_counter() - crop_start
                # 2. Resample BEFORE remap: e.g. daily→monthly first, then remap
                #    much cheaper than remap daily then resample
                resample_start = time.perf_counter()
                if not self._is_climatology_mode():
                    data = self._resample_to_compare_resolution(data, f"{data_source} grid data")
                resample_seconds = time.perf_counter() - resample_start

                target_grid = self.create_target_grid()
                target_shape = (int(target_grid.sizes.get("lat", 0)), int(target_grid.sizes.get("lon", 0)))
                backend = str(
                    getattr(self, "regrid_backend", "openbench_conservative") or "openbench_conservative"
                ).lower()
                same_grid = backend == "openbench_conservative" and self._grids_match(data, target_grid)
                conservative = None
                if not same_grid and backend == "openbench_conservative":
                    from openbench.data.regrid.methods import conservative

                    conservative.reset_weight_cache_activity()
                regrid_start = time.perf_counter()
                remapped_data = self.remap_data(data)
                regrid_seconds = time.perf_counter() - regrid_start
                weight_cache = conservative.consume_weight_cache_activity() if conservative is not None else "n/a"
                write_start = time.perf_counter()
                self.save_remapped_data(remapped_data, data_source, year)
                write_seconds = time.perf_counter() - write_start
                logging.info(
                    "[REGRID_PERF] source=%s year=%d backend=%s read=%.3fs crop=%.3fs "
                    "resample=%.3fs regrid=%.3fs write=%.3fs total=%.3fs "
                    "source_grid=%dx%d target_grid=%dx%d time=%d num_cores=%d "
                    "regrid_workers=%d same_grid_bypass=%s weight_cache=%s",
                    data_source,
                    year,
                    backend,
                    read_seconds,
                    crop_seconds,
                    resample_seconds,
                    regrid_seconds,
                    write_seconds,
                    time.perf_counter() - total_start,
                    source_shape[0],
                    source_shape[1],
                    target_shape[0],
                    target_shape[1],
                    int(data.sizes.get("time", 0)),
                    int(getattr(self, "num_cores", 1)),
                    int(regrid_workers or 1),
                    str(same_grid).lower(),
                    weight_cache,
                )
        finally:
            gc.collect()

    def preprocess_grid_data(self, data: xr.Dataset) -> xr.Dataset:
        data = self.check_coordinate(data)
        if data["lon"].ndim == 2 and data["lat"].ndim == 2:
            # This first pass regularizes a curvilinear grid over its own
            # extent. Its cell centres generally differ from create_target_grid;
            # remap_data therefore still performs the evaluation-grid remap.
            # If they happen to match exactly, the shared same-grid check skips it.
            try:
                from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_xesmf
                from openbench.data.regrid.xesmf_cache import default_weight_cache_dir

                data = convert_to_wgs84_xesmf(data, self.compare_grid_res, cache_dir=default_weight_cache_dir(self))
            except (ImportError, ValueError, RuntimeError) as e:
                logging.debug(f"xesmf regridding failed, falling back to scipy: {e}")
                from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_scipy

                data = convert_to_wgs84_scipy(data, self.compare_grid_res)

        return self._normalize_longitude_axis(data)
