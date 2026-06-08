"""Variable transforms, filters, time slicing, and unit conversion."""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data._processing_config import USE_NEW_FREQ_ALIASES
from openbench.data._processing_utils import performance_monitor
from openbench.data.unit import UnitProcessing
from openbench.util.names import get_mapping_key_case_insensitive, get_xarray_key_case_insensitive


class ProcessingTransformMixin:
    @staticmethod
    def _normalize_longitude_axis(ds: xr.Dataset) -> xr.Dataset:
        """Normalize 1-D longitude coordinates and remove duplicate seam cells."""
        if "lon" not in ds.coords or ds["lon"].ndim != 1:
            return ds

        lon = ds["lon"]
        lon_vals = lon.values
        if lon_vals.size == 0:
            return ds

        normalized = ((lon_vals + 180) % 360) - 180 if np.nanmax(lon_vals) > 180 else lon_vals
        attrs = dict(lon.attrs)
        ds = ds.assign_coords(lon=xr.DataArray(normalized, dims=lon.dims, attrs=attrs))
        ds = ds.sortby("lon")

        sorted_lon = np.asarray(ds["lon"].values)
        _, unique_indices = np.unique(sorted_lon, return_index=True)
        if len(unique_indices) != len(sorted_lon):
            logging.warning("Duplicate longitude coordinates after normalization; keeping first seam cell")
            ds = ds.isel(lon=np.sort(unique_indices))

        if "valid_min" in ds["lon"].attrs:
            ds["lon"].attrs["valid_min"] = -180.0
        if "valid_max" in ds["lon"].attrs:
            ds["lon"].attrs["valid_max"] = 180.0
        return ds

    def _try_compute_from_profile(self, source_name: str, ds, datasource: str):
        """Try to compute a derived variable using a compute expression.

        Checks model profiles first, then reference datasets.
        Returns computed DataArray, or None if no compute expression applies.
        """
        try:
            from openbench.data.registry.manager import get_registry

            mgr = get_registry()
            item = getattr(self, "item", "")
            if not item:
                return None

            # Check model profile first, then reference dataset
            var_mapping = None
            profile = mgr.get_model(source_name)
            profile_key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
            if profile and profile_key is not None and profile.variables[profile_key].compute:
                var_mapping = profile.variables[profile_key]

            if var_mapping is None:
                ref = mgr.get_reference(source_name)
                ref_key = get_mapping_key_case_insensitive(ref.variables, item) if ref else None
                if ref and ref_key is not None and ref.variables[ref_key].compute:
                    var_mapping = ref.variables[ref_key]

            if var_mapping is None:
                return None

            logging.info("Computing %s via catalog compute expression", item)
            from openbench.data.compute import execute_compute

            result = execute_compute(ds, var_mapping.compute, item)

            if hasattr(result, "name"):
                result.name = item

            setattr(self, f"{datasource}_varname", [item])
            setattr(self, f"{datasource}_varunit", var_mapping.varunit)

            return result

        except Exception as e:
            logging.debug(f"Compute from profile failed for {source_name}/{getattr(self, 'item', '?')}: {e}")
            return None

    def apply_custom_filter(self, datasource: str, ds: xr.Dataset, varname: List) -> xr.Dataset:
        if datasource == "stat":
            # Validate varname list is not empty
            if not varname or len(varname) == 0:
                raise ValueError("Variable name list cannot be empty for station data")

            # Validate variable exists in dataset
            actual_var = get_xarray_key_case_insensitive(ds, varname[0])
            if actual_var is None:
                available_vars = list(ds.data_vars) + list(ds.coords)
                raise KeyError(f"Variable '{varname[0]}' not found in station dataset. Available: {available_vars}")

            return ds[actual_var]
        else:
            # Resolve source name (model name for sim, dataset name for ref)
            source_key = self.sim_source if datasource == "sim" else self.ref_source
            try:
                source_name = getattr(self, f"{source_key}_model")
            except AttributeError:
                source_name = source_key

            # Priority 1: compute expression from catalog YAML (model or reference)
            computed = self._try_compute_from_profile(source_name, ds, datasource)
            if computed is not None:
                return computed

            # Priority 2: filter module (user ~/.openbench/custom/ → built-in)
            try:
                from openbench.data.custom import load_filter

                filter_module = load_filter(source_name)
                filter_func = None
                if filter_module:
                    filter_func = getattr(filter_module, f"filter_{source_name}", None)
                # Fallback: strip version suffix (CoLM2024 → CoLM)
                if filter_func is None:
                    import re as _re

                    base_name = _re.sub(r"[\d.]+$", "", source_name)
                    if base_name and base_name != source_name:
                        filter_module = filter_module or load_filter(base_name)
                        if filter_module:
                            filter_func = getattr(filter_module, f"filter_{base_name}", None)
                if filter_module and filter_func:
                    logging.info("Applying filter for %s", source_name)
                    result = filter_func(self, ds)
                    ds_or_da = result[1] if isinstance(result, tuple) else result
                    if isinstance(ds_or_da, xr.Dataset):
                        current_varname = getattr(self, f"{datasource}_varname")
                        var_to_extract = current_varname[0] if isinstance(current_varname, list) else current_varname
                        actual_extract = get_xarray_key_case_insensitive(ds_or_da, var_to_extract)
                        if actual_extract is not None:
                            return ds_or_da[actual_extract]
                    elif ds_or_da is not None:
                        return ds_or_da
            except Exception as e:
                logging.debug("Filter failed for %s: %s", source_name, e)

            # Priority 3: direct extraction
            current_varname = getattr(self, f"{datasource}_varname")
            var_to_extract = current_varname[0] if isinstance(current_varname, list) else current_varname
            actual_extract = get_xarray_key_case_insensitive(ds, var_to_extract)
            if actual_extract is not None:
                return ds[actual_extract]

            raise KeyError(f"Variable '{var_to_extract}' not found in dataset")
        return ds

    @performance_monitor
    def select_timerange(self, ds: xr.Dataset, syear: int, eyear: int) -> xr.Dataset:
        if eyear < syear:
            logging.error(f"Error: Invalid time range (syear={syear}, eyear={eyear})")
            raise ValueError(f"Invalid time range: eyear ({eyear}) must be >= syear ({syear})")
        return ds.sel(time=slice(f"{syear}-01-01T00:00:00", f"{eyear}-12-31T23:59:59"))

    @performance_monitor
    def resample_data(self, dfx1: xr.Dataset, tim_res: str, startx: int, endx: int) -> xr.Dataset:
        # Check if climatology mode - skip resampling
        tim_res_lower = str(tim_res).strip().lower()
        if tim_res_lower in ["climatology-year", "climatology-month"]:
            logging.debug(f"resample_data: Climatology mode detected ({tim_res}), returning data unchanged")
            return dfx1

        match = re.match(r"(\d+)\s*([a-zA-Z]+)", tim_res)
        if not match:
            logging.error("Invalid time resolution format. Use '3month', '6hr', etc.")
            raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")

        value, unit = match.groups()
        value = int(value)

        # Get frequency map based on pandas version
        if USE_NEW_FREQ_ALIASES:
            freq_map = {"month": "ME", "day": "D", "hour": "h", "year": "YE", "week": "W"}
        else:
            freq_map = {"month": "M", "day": "D", "hour": "H", "year": "Y", "week": "W"}

        freq = freq_map.get(unit.lower())
        if not freq:
            logging.error(f"Unsupported time unit: {unit}")
            raise ValueError(f"Unsupported time unit: {unit}")

        # Build frequency string
        freq_str = f"{value}{freq}"
        time_index = pd.date_range(start=f"{startx}-01-01T00:00:00", end=f"{endx}-12-31T23:59:59", freq=freq_str)
        ds = xr.Dataset({"data": ("time", np.nan * np.ones(len(time_index)))}, coords={"time": time_index})
        orig_ds_reindexed = dfx1.reindex(time=ds.time)
        return xr.merge([ds, orig_ds_reindexed]).drop_vars("data")

    @performance_monitor
    def process_units(self, ds: xr.Dataset, varunit: str) -> Tuple[xr.Dataset, str]:
        try:
            # Keep xarray objects intact where possible so calendar-aware unit
            # conversions can use coordinates such as ``time``.
            if isinstance(ds, xr.Dataset):
                # 如果是数据集，获取第一个变量的数据
                data_vars_list = list(ds.data_vars)
                if not data_vars_list:
                    logging.error("Dataset has no data variables")
                    raise ValueError("Dataset must contain at least one data variable")
                var_name = data_vars_list[0]
                data_array = ds[var_name]
            elif isinstance(ds, xr.DataArray):
                # 如果是数据数组，保留坐标
                data_array = ds
            else:
                # 如果已经是numpy数组，直接使用
                data_array = ds

            # 进行单位转换
            converted_data, new_unit = UnitProcessing.convert_unit(data_array, varunit.lower())
            # 创建新的数据集或更新现有数据集
            if isinstance(ds, xr.Dataset):
                # Assign through xarray objects rather than mutating .values,
                # which is unreliable for dask-backed or read-only arrays.
                ds = ds.copy()
                if isinstance(converted_data, xr.DataArray):
                    ds[var_name] = converted_data
                else:
                    ds[var_name] = ds[var_name].copy(data=converted_data)
                ds[var_name].attrs["units"] = new_unit
            elif isinstance(ds, xr.DataArray):
                if isinstance(converted_data, xr.DataArray):
                    ds = converted_data.copy()
                else:
                    ds = ds.copy(data=converted_data)

            # 更新单位属性
            ds.attrs["units"] = new_unit
            logging.debug(f"Converted unit from {varunit} to {new_unit}")

            return ds, new_unit

        except ValueError as e:
            logging.warning(f"Warning: {str(e)}. Attempting specific conversion.")
            # 不要直接退出，而是返回原始数据
            return ds, varunit
        except Exception as e:
            logging.error(f"Error in unit conversion: {str(e)}")
            # 返回原始数据
            return ds, varunit
