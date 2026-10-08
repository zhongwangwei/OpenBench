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
from openbench.data.unit import UnitProcessing, warn_if_file_unit_differs
from openbench.util.names import get_mapping_key_case_insensitive, get_xarray_key_case_insensitive

ACCUMULATION_MODES = ("year", "run")


def deaccumulate(data, mode: str):
    """Turn a running total into the amount added during each time step.

    ``mode="year"``: the total restarts on 1 January (e.g. CoLM ``f_sum_irrig``),
    so the first step of a year keeps its value when it falls in January and is
    missing otherwise. ``mode="run"``: the total runs from the start of the
    simulation (e.g. WRF ``RAINNC``), so the first step is missing. A step that
    goes negative, from a restart or a bucket reset, is set to missing.
    """
    if mode not in ACCUMULATION_MODES:
        raise ValueError(f"Unknown accumulation mode {mode!r}; expected one of {ACCUMULATION_MODES}")
    if isinstance(data, xr.Dataset):
        out = data.copy()
        for name in data.data_vars:
            if "time" in data[name].dims:
                out[name] = deaccumulate(data[name], mode)
        return out

    time = data["time"]
    step = data - data.shift(time=1)
    if mode == "year":
        year = time.dt.year
        new_year = year != year.shift(time=1)
        step = xr.where(new_year, data.where(time.dt.month == 1), step)
    step = step.where(step >= 0)
    step.name = data.name
    step.attrs = dict(data.attrs)
    return step.transpose(*data.dims)


class ProcessingTransformMixin:
    def _accumulation_mode(self, datasource: str) -> str:
        """Accumulation mode of the current item for a datasource ('' when the data are not running totals)."""
        # Config sections key per-variable fields by the source name (like fallbacks).
        source_key = getattr(self, f"{datasource}_source", None)
        mode = getattr(self, f"{source_key}_accumulated", "") if source_key else ""
        if not mode:
            from openbench.data.registry.manager import get_registry

            item = getattr(self, "item", "")
            if item and source_key:
                source_name = getattr(self, f"{source_key}_model", source_key)
                mgr = get_registry()
                for profile in (mgr.get_model(source_name), mgr.get_reference(source_name)):
                    key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
                    if key is not None:
                        mode = getattr(profile.variables[key], "accumulated", None) or ""
                        break
        return str(mode).strip().lower()

    def _deaccumulate_if_configured(self, ds, datasource: str):
        """Convert running totals to per-step amounts before resampling and unit conversion."""
        mode = self._accumulation_mode(datasource)
        if not mode:
            return ds
        logging.info("De-accumulating %s %s (accumulated: %s)", datasource, getattr(self, "item", ""), mode)
        return deaccumulate(ds, mode)

    def _reduce_patch_dimension(self, data, source_ds=None):
        """Collapse CABLE-style patch output to grid cells when patch fractions exist."""
        patch_dims = [dim for dim in getattr(data, "dims", ()) if str(dim).lower() == "patch"]
        if not patch_dims:
            return data

        patch_dim = patch_dims[0]
        if getattr(data, "sizes", {}).get(patch_dim) == 1:
            return data.squeeze(patch_dim, drop=True)

        if source_ds is not None:
            patchfrac_name = get_xarray_key_case_insensitive(source_ds, "patchfrac")
            if patchfrac_name is not None and patch_dim in source_ds[patchfrac_name].dims:
                return data.weighted(source_ds[patchfrac_name].fillna(0)).mean(patch_dim)

        logging.warning("Leaving unreduced patch dimension because patchfrac is missing")
        return data

    @staticmethod
    def _normalize_longitude_axis(ds: xr.Dataset) -> xr.Dataset:
        """Normalize 1-D longitude coordinates and remove duplicate seam cells."""
        if "lon" not in ds.coords or ds["lon"].dims != ("lon",):
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

    def _try_compute_from_profile(self, source_name: str, ds, datasource: str, *, fallback_var: str | None = None):
        """Try to compute a derived variable using a compute expression.

        Checks model profiles first, then reference datasets.
        Returns computed DataArray, or None if no compute expression applies.
        """
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        item = getattr(self, "item", "")
        if not item:
            return None

        # Inline namelist compute wins over catalog profiles. Config sections key
        # it by the source name (like fallbacks), not by "ref"/"sim".
        source_key = getattr(self, f"{datasource}_source", "")
        compute_expr = getattr(self, f"{datasource}_compute", "") or (
            getattr(self, f"{source_key}_compute", "") if source_key else ""
        )
        compute_unit = getattr(self, f"{datasource}_varunit", "")

        # Check model profile first, then reference dataset.
        var_mapping = None
        if not compute_expr:
            profile = mgr.get_model(source_name)
            profile_key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
            if profile and profile_key is not None and profile.variables[profile_key].compute:
                var_mapping = profile.variables[profile_key]

        if not compute_expr and var_mapping is None:
            ref = mgr.get_reference(source_name)
            ref_key = get_mapping_key_case_insensitive(ref.variables, item) if ref else None
            if ref and ref_key is not None and ref.variables[ref_key].compute:
                var_mapping = ref.variables[ref_key]

        if var_mapping is not None:
            compute_expr = var_mapping.compute
            compute_unit = var_mapping.varunit
        if not compute_expr:
            return None

        from openbench.data.compute import compute_dependency_names, compute_inputs_known, execute_compute

        # Skip the compute only when its inputs are known: a step that does not
        # parse, or a computed key such as ds[key], may read the fallback.
        if (
            fallback_var is not None
            and compute_inputs_known(compute_expr)
            and fallback_var.casefold() not in {name.casefold() for name in compute_dependency_names(compute_expr)}
        ):
            return None

        logging.info("Computing %s via compute expression", item)

        result = execute_compute(ds, compute_expr, item)

        if hasattr(result, "name"):
            result.name = item
        result = self._reduce_patch_dimension(result, ds)

        setattr(self, f"{datasource}_varname", [item])
        setattr(self, f"{datasource}_varunit", compute_unit)
        # The expression already produced the target quantity and units.
        self.__dict__.pop(f"_fb_convert_{datasource}", None)

        return result

    def apply_custom_filter(self, datasource: str, ds: xr.Dataset, varname: List) -> xr.Dataset:
        if datasource == "stat":
            if not varname or len(varname) == 0:
                raise ValueError("Variable name list cannot be empty for station data")

            actual_var = get_xarray_key_case_insensitive(ds, varname[0])
            if actual_var is None:
                available_vars = list(ds.data_vars) + list(ds.coords)
                raise KeyError(f"Variable '{varname[0]}' not found in station dataset. Available: {available_vars}")

            return ds[actual_var]
        else:
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
                            return self._reduce_patch_dimension(ds_or_da[actual_extract], ds_or_da)
                    elif ds_or_da is not None:
                        return self._reduce_patch_dimension(ds_or_da, ds)
            except Exception as e:
                logging.debug("Filter failed for %s: %s", source_name, e)

            # Priority 3: direct extraction
            current_varname = getattr(self, f"{datasource}_varname")
            var_to_extract = current_varname[0] if isinstance(current_varname, list) else current_varname
            actual_extract = get_xarray_key_case_insensitive(ds, var_to_extract)
            if actual_extract is not None:
                return self._reduce_patch_dimension(ds[actual_extract], ds)

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

        if USE_NEW_FREQ_ALIASES:
            freq_map = {"month": "ME", "day": "D", "hour": "h", "year": "YE", "week": "W"}
        else:
            freq_map = {"month": "M", "day": "D", "hour": "H", "year": "Y", "week": "W"}

        freq = freq_map.get(unit.lower())
        if not freq:
            logging.error(f"Unsupported time unit: {unit}")
            raise ValueError(f"Unsupported time unit: {unit}")

        freq_str = f"{value}{freq}"
        time_index = pd.date_range(start=f"{startx}-01-01T00:00:00", end=f"{endx}-12-31T23:59:59", freq=freq_str)
        ds = xr.Dataset({"data": ("time", np.nan * np.ones(len(time_index)))}, coords={"time": time_index})
        orig_ds_reindexed = dfx1.reindex(time=ds.time)
        return xr.merge([ds, orig_ds_reindexed]).drop_vars("data")

    @performance_monitor
    def process_units(self, ds: xr.Dataset, varunit: str, datasource: str | None = None) -> Tuple[xr.Dataset, str]:
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

            item = getattr(self, "item", None)
            source = getattr(self, f"{datasource}_source", None) or datasource or "data"
            warn_if_file_unit_differs(getattr(data_array, "attrs", {}).get("units"), varunit, item, source)

            # 进行单位转换
            unit_key = UnitProcessing.lookup_key(varunit, item)
            converted_data, new_unit = UnitProcessing.convert_unit(data_array, unit_key)
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
                    name = ds.name
                    ds = converted_data.copy()
                    # Calendar-aware conversions (mm month-1, mm year-1) divide
                    # by a time-derived array, which makes xarray drop the name.
                    ds.name = name
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
