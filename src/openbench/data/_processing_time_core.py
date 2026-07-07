"""Time-coordinate validation helpers for dataset processing."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import xarray as xr


logger = logging.getLogger(__name__)


class TimeCoreMixin:
    """Split temporal processing helpers."""

    @staticmethod
    def _frequency_rank(freq: str) -> int | None:
        """Return temporal coarseness rank; lower means finer resolution."""
        text = str(freq or "").strip().lower()
        if text in {"climatology-year", "climatology-month"}:
            return None
        match = re.match(r"\d*\s*([a-zA-Z]+)", text)
        unit = match.group(1).lower() if match else text
        if unit in {"h", "hr", "hour", "hourly"}:
            return 0
        if unit in {"d", "day", "daily"}:
            return 1
        if unit in {"w", "wk", "week", "weekly"}:
            return 2
        if unit in {"m", "me", "mon", "month", "monthly"}:
            return 3
        if unit in {"y", "ye", "yr", "year", "annual", "yearly"}:
            return 4
        return None

    @staticmethod
    def _infer_time_rank(data: xr.Dataset | xr.DataArray) -> int | None:
        """Infer source temporal coarseness from its time coordinate."""
        if "time" not in getattr(data, "coords", {}):
            return None
        try:
            times = pd.to_datetime(data["time"].values)
        except Exception:
            return None
        if len(times) < 2:
            return None
        deltas = pd.Series(times).sort_values().diff().dropna()
        if deltas.empty:
            return None
        median_days = deltas.median() / pd.Timedelta(days=1)
        if median_days <= 1 / 12:
            return 0
        if median_days <= 1.5:
            return 1
        if median_days <= 10:
            return 2
        if median_days <= 45:
            return 3
        return 4

    def _guard_against_temporal_upsampling(
        self, data: xr.Dataset | xr.DataArray, target_freq: str, context: str
    ) -> None:
        """Reject coarse→fine resampling that would make repeated/empty pseudo-samples."""
        source_rank = self._infer_time_rank(data)
        target_rank = self._frequency_rank(target_freq)
        if source_rank is not None and target_rank is not None and target_rank < source_rank:
            raise ValueError(
                f"{context}: refusing to upsample source time resolution to {target_freq!r}; "
                "choose a comparison time resolution no finer than the input data"
            )

    def _resample_to_compare_resolution(self, data: xr.Dataset | xr.DataArray, context: str):
        self._guard_against_temporal_upsampling(data, self.compare_tim_res, context)
        item = str(getattr(self, "item", "") or "").lower()
        units = str(getattr(data, "attrs", {}).get("units", "") or "").lower().strip()
        if isinstance(data, xr.Dataset) and not units:
            data_units = {
                str(var.attrs.get("units", "")).lower().strip()
                for var in data.data_vars.values()
                if var.attrs.get("units")
            }
            units = next(iter(data_units)) if len(data_units) == 1 else ""

        accumulation_items = {"precipitation", "total_irrigation_amount"}
        accumulation_units = {"mm", "kg m-2", "kg/m2", "kg m**-2"}
        if item in accumulation_items and units in accumulation_units:
            logger.info("Resampling accumulated %s with sum over %s", item, self.compare_tim_res)
            return data.resample(time=self.compare_tim_res).sum()
        return data.resample(time=self.compare_tim_res).mean()

    def check_coordinate(self, ds: xr.Dataset) -> xr.Dataset:
        # Rename both coordinates and dimensions (e.g., WRF south_north → lat).
        # Prefer dimension coordinates when several CABLE-style names point to
        # the same target (x + longitude → lon, y + latitude → lat).
        rename_map = {}
        planned_targets = set(ds.coords) | set(ds.dims)

        for name in ds.dims:
            target = self.coordinate_map.get(name)
            if target and target not in planned_targets:
                rename_map[name] = target
                planned_targets.add(target)

        for name in ds.coords:
            target = self.coordinate_map.get(name)
            if target and target not in planned_targets:
                rename_map[name] = target
                planned_targets.add(target)
        if rename_map:
            ds = ds.rename(rename_map)
        return self._normalize_longitude_axis(ds)

    def check_time(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        # Validate year values
        syear = self.validate_year(syear, default=1990)
        eyear = self.validate_year(eyear, default=2020)
        tim_res_lower = str(tim_res or "").strip().lower()

        if "time" not in ds.coords:
            if tim_res_lower == "climatology-year":
                return ds.expand_dims(time=[pd.Timestamp(f"{syear}-01-01")])
            if tim_res_lower == "climatology-month":
                raise ValueError("Monthly climatology requires a time coordinate with 12 monthly values")
            logging.warning("The dataset does not contain a 'time' coordinate.")
            time_index = pd.date_range(start=f"{syear}-01-01T00:00:00", end=f"{eyear}-12-31T23:59:59", freq=tim_res)
            result = ds.expand_dims(time=time_index)
            try:
                result = result.transpose("time", "lat", "lon")
            except (ValueError, KeyError):
                try:
                    result = result.transpose("time", "lon", "lat")
                except (ValueError, TypeError):
                    result = result.squeeze()
            if isinstance(result, xr.Dataset):
                var_name = ds.name if isinstance(ds, xr.DataArray) else next(iter(result.data_vars), None)
                return result[var_name] if var_name and var_name in result else next(iter(result.data_vars.values()))
            return result

        if not hasattr(ds["time"], "dt"):
            try:
                ds["time"] = pd.to_datetime(ds["time"])
            except (ValueError, TypeError, AttributeError):
                time_index = pd.date_range(start=f"{syear}-01-01T00:00:00", end=f"{eyear}-12-31T23:59:59", freq=tim_res)
                if "time" not in ds.dims:
                    raise ValueError("Cannot repair an unparseable time coordinate without a time dimension")
                if ds.sizes["time"] != len(time_index):
                    raise ValueError(
                        "Cannot repair unparseable time coordinate: "
                        f"data has {ds.sizes['time']} time steps but expected {len(time_index)} for {tim_res}"
                    )
                return ds.assign_coords(time=time_index)

        # Check for duplicate time values
        if ds["time"].to_index().has_duplicates:
            logging.warning("Warning: Duplicate time values found. Removing duplicates...")
            # Remove duplicates by keeping the first occurrence
            _, index = np.unique(ds["time"], return_index=True)
            ds = ds.isel(time=index)

        # Ensure time is sorted
        ds = ds.sortby("time")
        var_name = ds.name if isinstance(ds, xr.DataArray) else next(iter(ds.data_vars), None)
        try:
            result = ds.transpose("time", "lat", "lon")
        except (ValueError, KeyError):
            try:
                result = ds.transpose("time", "lon", "lat")
            except (ValueError, KeyError):
                result = ds.squeeze()
        # Ensure we always return a DataArray
        if isinstance(result, xr.Dataset) and var_name and var_name in result:
            return result[var_name]
        elif isinstance(result, xr.Dataset):
            return next(iter(result.data_vars.values()))
        return result

    def check_dataset_time_integrity(
        self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str
    ) -> xr.Dataset:
        """Checks and fills missing time values in an xarray Dataset with specified comparison scales."""
        # Ensure the dataset has a proper time index
        ds = self.check_time(ds, syear, eyear, tim_res)
        if self._is_climatology_frequency_value(tim_res):
            return ds
        # Apply model-specific time adjustments
        if datasource == "stat":
            ds["time"] = pd.DatetimeIndex(ds["time"].values)
        else:
            if self.sim_data_type != "stn":
                ds = self.apply_model_specific_time_adjustment(ds, datasource, syear, eyear, tim_res)
        ds = self.make_time_integrity(ds, syear, eyear, tim_res, datasource)
        return ds
