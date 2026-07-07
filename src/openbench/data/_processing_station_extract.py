"""Station-data processing mixin for OpenBench datasets."""

from __future__ import annotations

import gc
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def _processing_attr(name, fallback):
    processing = sys.modules.get("openbench.data.processing")
    return getattr(processing, name, fallback) if processing is not None else fallback


def _parallel():
    return _processing_attr("Parallel", Parallel)


def _delayed():
    return _processing_attr("delayed", delayed)


def _cyclic_lon_delta(lon_values, target_lon: float):
    """Shortest absolute distance between longitudes, accounting for wrap."""
    return np.abs((np.asarray(lon_values, dtype=float) - float(target_lon) + 180.0) % 360.0 - 180.0)


class StationExtractionMixin:
    """Split station processing helpers."""

    def _extract_stn_parallel(
        self, datasource: str, dataset: xr.Dataset, station_list: pd.DataFrame, index: int
    ) -> None:
        try:
            station = station_list.iloc[index]
            start_year = int(station["use_syear"])
            end_year = int(station["use_eyear"])

            station_data = self.extract_single_station_data(dataset, station, datasource)
            processed_data = self.process_extracted_data(station_data, start_year, end_year)

            # Only save if processed_data is not None (i.e., had valid time range)
            if processed_data is not None:
                self.save_extracted_data(processed_data, station, datasource)
            else:
                logging.info(f"Skipping station {station['ID']} - no data in time range {start_year}-{end_year}")
        finally:
            gc.collect()

    def extract_single_station_data(self, dataset: xr.Dataset, station: pd.Series, datasource: str) -> xr.Dataset:
        if dataset is None:
            logging.error(f"Dataset is None for station {station['ID']} ({datasource})")
            raise ValueError("Dataset cannot be None when extracting station data")

        if datasource == "ref":
            lat_key, lon_key = "sim_lat", "sim_lon"
        elif datasource == "sim":
            lat_key, lon_key = "ref_lat", "ref_lon"
        else:
            logging.error(f"Invalid datasource: {datasource}")
            raise ValueError(f"Invalid datasource: {datasource}")

        # Fallback to reference coordinates if simulation coordinates are unavailable
        if lat_key not in station or pd.isna(station.get(lat_key)):
            lat_key = "ref_lat"
        if lon_key not in station or pd.isna(station.get(lon_key)):
            lon_key = "ref_lon"

        target_lat = float(station[lat_key])
        target_lon = float(station[lon_key])

        from openbench.data.coordinates import find_lat_name, find_lon_name

        all_names = set(dataset.coords) | set(dataset.dims)
        lat_coord = find_lat_name(all_names) or "lat"
        lon_coord = find_lon_name(all_names) or "lon"

        tolerance = self._station_snap_tolerance()
        try:
            return dataset.sel(
                {lat_coord: [target_lat], lon_coord: [target_lon]}, method="nearest", tolerance=tolerance
            )
        except (KeyError, ValueError, pd.errors.InvalidIndexError) as exc:
            logging.debug(
                "Coordinate selection failed for station %s (%s): %s. Falling back to manual indexing.",
                station["ID"],
                datasource,
                exc,
            )

            lat_values = dataset[lat_coord].values
            lon_values = dataset[lon_coord].values

            lat_idx = int(np.argmin(np.abs(lat_values - target_lat)))
            lon_distances = _cyclic_lon_delta(lon_values, target_lon)
            lon_idx = int(np.argmin(lon_distances))
            lat_delta = abs(float(lat_values[lat_idx]) - target_lat)
            lon_delta = float(lon_distances[lon_idx])
            if lat_delta > tolerance or lon_delta > tolerance:
                raise ValueError(
                    f"Nearest grid cell for station {station['ID']} is outside tolerance "
                    f"({lat_delta:.6g}°, {lon_delta:.6g}° > {tolerance:.6g}°)"
                )

            data = dataset.isel({lat_coord: lat_idx, lon_coord: lon_idx})
            data = data.expand_dims({lat_coord: [lat_values[lat_idx]], lon_coord: [lon_values[lon_idx]]})
            return data

    def _station_snap_tolerance(self) -> float:
        explicit = getattr(self, "station_snap_tolerance", None)
        if explicit is not None:
            return float(explicit)
        try:
            grid_res = float(getattr(self, "compare_grid_res"))
        except (TypeError, ValueError, AttributeError):
            grid_res = 0.5
        return max(grid_res / 2.0, 1e-12)

    def process_extracted_data(self, data: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
        data = data.sel(time=slice(f"{start_year}-01-01T00:00:00", f"{end_year}-12-31T23:59:59"))

        # Check if time dimension is empty after slicing
        if len(data.time) == 0:
            logging.warning(f"No data available in time range {start_year}-{end_year}. Skipping this station.")
            return None

        data = data  # .where((data > -1e20) & (data < 1e20), np.nan)
        # Skip resampling for climatology mode - handled by Mod_Climatology
        if self._is_climatology_mode():
            return data
        return self._resample_to_compare_resolution(data, "extracted station data")

    def save_extracted_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        try:
            output_file = os.path.join(
                self.casedir,
                "data",
                f"stn_{self.ref_source}_{self.sim_source}",
                f"{self.item}_{datasource}_{station['ID']}_{station['use_syear']}_{station['use_eyear']}.nc",
            )

            _write_netcdf_atomic(data, output_file, compression=False)
            logging.debug(f"Saved extracted station data to {output_file}")
        finally:
            if hasattr(data, "close"):
                data.close()
            gc.collect()
