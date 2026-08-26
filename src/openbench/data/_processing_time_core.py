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
        item = re.sub(r"[\s-]+", "_", str(getattr(self, "item", "") or "").lower())
        units = str(getattr(data, "attrs", {}).get("units", "") or "").lower().strip()
        if isinstance(data, xr.Dataset) and not units:
            data_units = {
                str(var.attrs.get("units", "")).lower().strip()
                for var in data.data_vars.values()
                if var.attrs.get("units")
            }
            units = next(iter(data_units)) if len(data_units) == 1 else ""

        units = re.sub(r"\s+", " ", units.translate(str.maketrans({"−": "-", "⁻": "-", "²": "2"})))

        accumulation_items = {
            "p",
            "pr",
            "prcp",
            "precip",
            "precipitation",
            "rain",
            "rainfall",
            "runoff",
            "snowfall",
            "streamflow",
            "subsurface_runoff",
            "surface_runoff",
            "total_irrigation_amount",
            "total_precipitation",
            "total_runoff",
            "tot_precip",
        }
        state_items = {
            "dam_storage",
            "dam_water_storage",
            "depth_of_surface_water",
            "lake_water_level",
            "lake_water_volume",
            "river_water_level",
            "root_zone_soil_moisture",
            "snow_depth",
            "snow_water_equivalent",
            "soil_moisture",
            "soil_moisture_lev2",
            "surface_soil_moisture",
            "terrestrial_water_storage_change",
            "total_water_storage",
            "water_storage_in_aquifer",
            "water_table_depth",
        }
        accumulation_units = {"mm", "kg m-2", "kg/m2", "kg m**-2"}
        if item in accumulation_items and units in accumulation_units:
            logger.info("Resampling accumulated %s with sum over %s", item, self.compare_tim_res)
            return data.resample(time=self.compare_tim_res).sum()
        if units in accumulation_units and item not in accumulation_items | state_items:
            raise ValueError(
                f"{context}: units {units!r} are ambiguous for item {item!r}; use a canonical "
                "accumulation/state item name or an explicit rate unit"
            )
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
            raise ValueError("Non-climatology data must include a 'time' coordinate; refusing to broadcast static data")

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
            if getattr(self, "time_alignment", "intersection") == "strict":
                raise ValueError("strict time alignment requires unique time values; duplicate timestamps found")
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
                result = ds.squeeze([dim for dim, size in ds.sizes.items() if dim != "time" and size == 1])
        # Ensure we always return a DataArray
        if isinstance(result, xr.Dataset) and var_name and var_name in result:
            return result[var_name]
        elif isinstance(result, xr.Dataset):
            return next(iter(result.data_vars.values()))
        return result

    def _validate_strict_time_coverage(
        self, ds: xr.Dataset | xr.DataArray, syear: int, eyear: int, tim_res: str
    ) -> None:
        """Strict mode rejects missing/extra timestamps before NaN reindexing can hide them."""
        if getattr(self, "time_alignment", "intersection") != "strict" or "time" not in ds.coords:
            return
        text = str(tim_res or "").strip().lower()
        match = re.match(r"\d*\s*([a-zA-Z]+)", text)
        unit = match.group(1).lower() if match else text
        if unit in {"m", "me", "mon", "month", "monthly"}:
            expected = pd.period_range(f"{syear}-01", f"{eyear}-12", freq="M")
            present = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="M")
            label = "month"
        elif unit in {"d", "day", "daily"}:
            expected = pd.period_range(f"{syear}-01-01", f"{eyear}-12-31", freq="D")
            present = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="D")
            label = "day"
        elif unit in {"h", "hr", "hour", "hourly"}:
            expected = pd.period_range(f"{syear}-01-01 00:00:00", f"{eyear}-12-31 23:00:00", freq="h")
            present = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="h")
            label = "hour"
        elif unit in {"y", "ye", "yr", "year", "annual", "yearly", "a"}:
            expected = pd.period_range(str(syear), str(eyear), freq="Y")
            present = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="Y")
            label = "year"
        else:
            return
        missing = expected.difference(present)
        extra = present.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(
                f"strict time alignment requires complete {label} coverage for {syear}-{eyear}; "
                f"missing={list(map(str, missing[:5]))}, extra={list(map(str, extra[:5]))}"
            )

    def check_dataset_time_integrity(
        self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str
    ) -> xr.Dataset:
        """Checks and fills missing time values in an xarray Dataset with specified comparison scales."""
        # Ensure the dataset has a proper time index
        ds = self.check_time(ds, syear, eyear, tim_res)
        if self._is_climatology_frequency_value(tim_res):
            return ds
        # Apply model-specific time adjustments before strict coverage validation;
        # strict must still run before make_time_integrity fills missing steps.
        if datasource == "stat":
            ds["time"] = pd.DatetimeIndex(ds["time"].values)
        else:
            if getattr(self, f"{datasource}_data_type", "") != "stn":
                ds = self.apply_model_specific_time_adjustment(ds, datasource, syear, eyear, tim_res)
        self._validate_strict_time_coverage(ds, syear, eyear, tim_res)
        ds = self.make_time_integrity(ds, syear, eyear, tim_res, datasource)
        return ds
