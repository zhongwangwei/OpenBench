"""Station-data processing mixin for OpenBench datasets."""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from openbench.util.converttype import Convert_Type
from openbench.util.names import get_xarray_key_case_insensitive
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic

_MERGED_STATION_DIMS = {
    "station",
    "stations",
    "site",
    "sites",
    "nstations",
    "nstation",
    "location",
    "locations",
    "point",
    "points",
    "stid",
}
_MERGED_STATION_ID_VARS = (
    "station_id",
    "site_id",
    "station_key",
    "station_name",
    "site_name",
    "SITE_NAME",
    "id",
    "ID",
)


def _processing_attr(name, fallback):
    processing = sys.modules.get("openbench.data.processing")
    return getattr(processing, name, fallback) if processing is not None else fallback


def _parallel():
    return _processing_attr("Parallel", Parallel)


def _delayed():
    return _processing_attr("delayed", delayed)


def _station_value_to_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        if scalar is None:
            return ""
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8", errors="ignore").strip()
        try:
            if pd.isna(scalar):
                return ""
        except (TypeError, ValueError):
            pass
        return str(scalar).strip()

    flat = arr.ravel()
    if arr.dtype.kind in {"S", "U"}:
        return "".join(
            _station_value_to_string(item)
            for item in flat
            if _station_value_to_string(item)
        ).strip()
    if arr.dtype.kind == "O" and all(np.asarray(item).ndim == 0 for item in flat):
        parts = [_station_value_to_string(item) for item in flat]
        if len([part for part in parts if part]) > 1 and all(len(part) == 1 for part in parts if part):
            return "".join(parts).strip()
    return _station_value_to_string(flat[0]) if flat.size else ""


class StationProcessingCoreMixin:
    """Split station processing helpers."""

    def process_station_data(self, data_params: Dict[str, Any]) -> None:
        try:
            logging.debug("Processing station data")
            if not hasattr(self, "station_list") or self.station_list is None or self.station_list.empty:
                logging.error("Station list is empty; cannot process station data.")
                return

            indices = range(len(self.station_list["ID"]))
            try:
                _parallel()(n_jobs=self.num_cores)(
                    _delayed()(self._make_stn_parallel)(self.station_list, data_params["datasource"], i)
                    for i in indices
                )
            except (PermissionError, OSError) as exc:
                logging.warning(
                    "Parallel station processing unavailable (%s). Falling back to sequential execution.", exc
                )
                for i in indices:
                    self._make_stn_parallel(self.station_list, data_params["datasource"], i)
        finally:
            gc.collect()

    def process_single_station_data(
        self, stn_data: xr.Dataset, start_year: int, end_year: int, datasource: str
    ) -> xr.Dataset:
        var_attr = self.ref_varname if datasource == "ref" else self.sim_varname
        var_attr_is_list = isinstance(var_attr, list)

        # Work on a copy of the current variable list so that temporary
        # fallbacks do not mutate the original configuration.
        current_var_list = list(var_attr) if var_attr_is_list else [var_attr]
        original_var_list = list(var_attr) if var_attr_is_list else [var_attr]
        original_varname = original_var_list[0] if original_var_list else None
        original_varunit_existed = hasattr(self, f"{datasource}_varunit")
        original_varunit = getattr(self, f"{datasource}_varunit", None)
        original_convert_existed = hasattr(self, f"_fb_convert_{datasource}")
        original_convert = getattr(self, f"_fb_convert_{datasource}", None)

        try:
            # Validate varname list is not empty
            if not current_var_list:
                logging.error("Variable name list is empty")
                raise ValueError("Variable name list cannot be empty for station data")

            # Check if the variable exists in the dataset
            actual_station_var = get_xarray_key_case_insensitive(stn_data, current_var_list[0])
            if actual_station_var is None:
                # Try to apply custom filter for variable fallback
                source_key = self.sim_source if datasource == "sim" else self.ref_source
                try:
                    source_name = getattr(self, f"{source_key}_model")
                except AttributeError:
                    source_name = source_key
                # Same priority: compute → filter → direct
                # Priority 1: compute
                computed = self._try_compute_from_profile(source_name, stn_data, datasource)
                if computed is not None:
                    current_var_list = [getattr(self, "item", current_var_list[0])]
                    ds = computed
                else:
                    # Priority 2: filter (station filters handle CaMA allocation etc.)
                    try:
                        from openbench.data.custom import load_filter

                        stn_module = load_filter(source_name)
                        filter_func = None
                        if stn_module:
                            filter_func = getattr(stn_module, f"filter_{source_name}", None)
                        if filter_func is None:
                            import re as _re

                            base_name = _re.sub(r"[\d.]+$", "", source_name)
                            if base_name and base_name != source_name:
                                stn_module = stn_module or load_filter(base_name)
                                if stn_module:
                                    filter_func = getattr(stn_module, f"filter_{base_name}", None)
                        if stn_module and filter_func:
                            logging.info("Applying station filter for %s", source_name)
                            updated_self, filtered_data = filter_func(self, stn_data)
                            if updated_self is not None and filtered_data is not None:
                                new_var_attr = getattr(self, f"{datasource}_varname")
                                current_var_list = (
                                    list(new_var_attr) if isinstance(new_var_attr, list) else [new_var_attr]
                                )
                                ds = filtered_data
                            else:
                                raise ValueError(f"Station filter returned None for {source_name}")
                        else:
                            raise ImportError(f"No filter function for {source_name}")
                    except (ImportError, AttributeError, ValueError):
                        list(stn_data.data_vars) + list(stn_data.coords)
                        logging.error(f"Variable '{current_var_list[0]}' not found in station data.")
                        raise ValueError(f"Variable '{current_var_list[0]}' not found in station data.")
            else:
                ds = stn_data[actual_station_var]

            # Apply fallback conversion expression if set (e.g., mol→g for GPP)
            # Supports multi-variable expressions: 'value' is current var,
            # other NC variables are accessible by name (e.g., f_assim).
            fb_convert = getattr(self, f"_fb_convert_{datasource}", None)
            if fb_convert:
                try:
                    from openbench.data.compute import _validate_expression

                    value = ds.values
                    ns = {"value": value, "np": np}
                    for vn in stn_data.data_vars:
                        if vn != current_var_list[0] and vn not in ns:
                            ns[vn] = stn_data[vn].values
                    _validate_expression(fb_convert, allowed_names=ns.keys())
                    ds = ds.copy(data=eval(fb_convert, {"__builtins__": {}}, ns))  # noqa: S307
                    logging.info("Applied station fallback conversion: %s", fb_convert)
                except Exception as e:
                    raise RuntimeError(
                        f"Station fallback conversion {fb_convert!r} failed; refusing to continue with unconverted units"
                    ) from e

            # Check the time dimension
            if "time" not in ds.dims:
                logging.error("Time dimension not found in the station data.")
                raise ValueError("Time dimension not found in the station data.")

            # Ensure the time coordinate is datetime
            if not np.issubdtype(ds.time.dtype, np.datetime64):
                ds["time"] = pd.to_datetime(ds.time.values)

            # Select the time range before resampling
            ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

            # Check if there's data in the selected time range
            if len(ds.time) == 0:
                logging.warning(f"No data found for the specified time range {start_year}-{end_year}")
                return None

            # Resample to compare_tim_res (skip for climatology — handled by Mod_Climatology)
            if not self._is_climatology_mode():
                ds = self._resample_to_compare_resolution(ds, f"{datasource} station data")

            # ds = ds.resample(time=self.compare_tim_res).mean()
            ds = self.check_coordinate(ds)
            ds = self.check_dataset_time_integrity(ds, start_year, end_year, self.compare_tim_res, datasource)

            # Apply unit conversion for station data
            current_varunit = getattr(self, f"{datasource}_varunit")
            if current_varunit:
                try:
                    ds, converted_unit = self.process_units(ds, current_varunit)
                    logging.info(
                        f"Applied unit conversion for {datasource} station data: {current_varunit} -> {converted_unit}"
                    )
                except Exception as e:
                    logging.warning(f"Unit conversion failed for {datasource} station data: {e}")

            ds = self.select_timerange(ds, start_year, end_year)

            # Store original variable name as attribute for later renaming if needed
            if original_varname:
                ds.attrs["_original_varname"] = original_varname

            return ds  # .where((ds > -1e20) & (ds < 1e20), np.nan)
        finally:
            # Restore the canonical variable definition and transient fallback
            # state so per-station fallback decisions do not leak into the next
            # station processed by the same worker/process.
            if datasource == "ref":
                self.ref_varname = list(original_var_list) if var_attr_is_list else original_varname
            else:
                self.sim_varname = list(original_var_list) if var_attr_is_list else original_varname
            if original_varunit_existed:
                setattr(self, f"{datasource}_varunit", original_varunit)
            elif hasattr(self, f"{datasource}_varunit"):
                delattr(self, f"{datasource}_varunit")
            if original_convert_existed:
                setattr(self, f"_fb_convert_{datasource}", original_convert)
            elif hasattr(self, f"_fb_convert_{datasource}"):
                delattr(self, f"_fb_convert_{datasource}")

    def _make_stn_parallel(self, station_list: pd.DataFrame, datasource: str, index: int) -> None:
        try:
            station = station_list.iloc[index]
            start_year = int(station["use_syear"])
            end_year = int(station["use_eyear"])
            file_path = station["sim_dir"] if datasource == "sim" else station["ref_dir"]
            with xr.open_dataset(file_path) as stn_data:
                stn_data = Convert_Type.convert_nc(stn_data)
                stn_data = self._select_merged_station_data(stn_data, station, datasource)
                processed_data = self.process_single_station_data(stn_data, start_year, end_year, datasource)
                if processed_data is None:
                    logging.info(f"Skipping station {station['ID']} ({datasource}) - no valid data after processing")
                    return
                self.save_station_data(processed_data, station, datasource)
        except (KeyError, FileNotFoundError, ValueError) as e:
            # Skip individual stations that fail (missing variable, file, or bad data)
            station_id = (
                station_list.iloc[index].get("ID", f"index-{index}") if index < len(station_list) else f"index-{index}"
            )
            logging.warning(f"Skipping station {station_id} ({datasource}): {e}")
        finally:
            gc.collect()

    def _select_merged_station_data(self, stn_data: xr.Dataset, station: pd.Series, datasource: str) -> xr.Dataset:
        """Select one station from a merged station-time dataset when needed."""
        station_id = str(station.get("ID", "")).strip()
        if not station_id:
            return stn_data

        station_dim = self._merged_station_dim_for_configured_variable(stn_data, datasource)
        if station_dim is None:
            return stn_data

        if stn_data.sizes.get(station_dim, 0) <= 1:
            return stn_data.isel({station_dim: 0}, drop=True)

        index = self._merged_station_index(stn_data, station_dim, station_id)
        if index is None:
            raise ValueError(f"Station {station_id} not found in merged station file")
        return stn_data.isel({station_dim: index}, drop=True)

    def _merged_station_dim_for_configured_variable(self, stn_data: xr.Dataset, datasource: str) -> str | None:
        var_attr = self.ref_varname if datasource == "ref" else self.sim_varname
        var_names = list(var_attr) if isinstance(var_attr, list) else [var_attr]
        configured_var = var_names[0] if var_names else None
        actual_var = get_xarray_key_case_insensitive(stn_data, configured_var) if configured_var else None
        candidate_dims = [dim for dim in stn_data.dims if str(dim).lower() in _MERGED_STATION_DIMS]

        if actual_var is not None:
            var_dims = set(stn_data[actual_var].dims)
            matches = [dim for dim in candidate_dims if dim in var_dims]
            return matches[0] if matches else None

        for dim in candidate_dims:
            for var in stn_data.data_vars.values():
                if dim in var.dims and "time" in var.dims:
                    return dim
        return None

    def _merged_station_index(self, stn_data: xr.Dataset, station_dim: str, station_id: str) -> int | None:
        for name in _MERGED_STATION_ID_VARS:
            actual_name = get_xarray_key_case_insensitive(stn_data, name)
            if actual_name is None:
                continue
            values = stn_data[actual_name]
            if station_dim not in values.dims:
                continue
            for index in range(stn_data.sizes[station_dim]):
                station_value = values.isel({station_dim: index}).values
                if _station_value_to_string(station_value) == station_id:
                    return index

        if station_dim in stn_data.coords:
            for index, value in enumerate(stn_data[station_dim].values):
                if _station_value_to_string(value) == station_id:
                    return index
        return None

    def save_station_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        data_to_save = None
        try:
            station = Convert_Type.convert_Frame(station)
            output_file = os.path.join(
                self.casedir,
                "data",
                f"stn_{self.ref_source}_{self.sim_source}",
                f"{self.item}_{datasource}_{station['ID']}_{station['use_syear']}_{station['use_eyear']}.nc",
            )

            # Rename variable back to original name if needed (for variable fallback scenarios)
            if "_original_varname" in data.attrs:
                original_varname = data.attrs["_original_varname"]
                current_varname = data.name if hasattr(data, "name") and data.name else None

                # Always rename if original_varname is different from current name
                if current_varname != original_varname:
                    logging.info(
                        f"Renaming variable '{current_varname}' back to '{original_varname}' before saving (station {station['ID']})"
                    )
                    # Convert DataArray to Dataset with the original variable name
                    if current_varname:
                        data_to_save = data.to_dataset(name=original_varname)
                    else:
                        # If no current name, convert to dataset and rename the variable
                        data_to_save = data.to_dataset()
                        # Get the first (and should be only) data variable name
                        current_var_list = list(data_to_save.data_vars)
                        if current_var_list:
                            data_to_save = data_to_save.rename({current_var_list[0]: original_varname})
                    # Remove the temporary attribute
                    if "_original_varname" in data_to_save.attrs:
                        del data_to_save.attrs["_original_varname"]
                    _write_netcdf_atomic(data_to_save, output_file, compression=False)
                else:
                    # Remove the temporary attribute
                    if "_original_varname" in data.attrs:
                        del data.attrs["_original_varname"]
                    _write_netcdf_atomic(data, output_file, compression=False)
            else:
                _write_netcdf_atomic(data, output_file, compression=False)

            logging.debug(f"Saved station data to {output_file}")
        finally:
            if data_to_save is not None and hasattr(data_to_save, "close"):
                data_to_save.close()
            if hasattr(data, "close"):
                data.close()
            gc.collect()
