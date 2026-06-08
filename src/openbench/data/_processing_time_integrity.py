"""Time-coordinate validation helpers for dataset processing."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.time_utils import normalize_cftime_axis

logger = logging.getLogger(__name__)


class TimeIntegrityWorkflowMixin:
    """Split temporal processing helpers."""

    def make_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str) -> xr.Dataset:
        # Validate year values
        syear = self.validate_year(syear, default=1990)
        eyear = self.validate_year(eyear, default=2020)
        # Convert any cftime axis (noleap/360_day/julian) to datetime64 first
        # so the rest of this method can rely on `ds["time"].dt.*` accessors.
        # Without this, the existing `try/except` ladder would silently drop
        # the cftime values and substitute a Gregorian date_range, producing
        # a 1-day-per-year offset for noleap data.
        ds = normalize_cftime_axis(ds, source_path=datasource)
        if self._is_climatology_frequency_value(tim_res):
            return ds

        match = re.match(r"(\d*)\s*([a-zA-Z]+)", tim_res)
        if match:
            num_value, time_unit = match.groups()
            num_value = int(num_value) if num_value else 1
            time_index = pd.date_range(start=f"{syear}-01-01T00:00:00", end=f"{eyear}-12-31T23:59:59", freq=tim_res)
            if time_unit.lower() in ["m", "month", "mon", "me"]:
                # Normalize to monthly resolution without enforcing a specific day match.
                # Set times to 15th for plotting/consistency, but do NOT reindex/fill missing months.
                # Compare by month presence only.
                mid_month = pd.to_datetime(pd.Series(time_index).dt.strftime("%Y-%m-15T00:00:00"))
                try:
                    ds["time"] = pd.to_datetime(ds["time"].dt.strftime("%Y-%m-15T00:00:00"))
                except (ValueError, AttributeError, TypeError):
                    # If we cannot format, keep existing times but ensure datetime type
                    try:
                        ds["time"] = pd.to_datetime(ds["time"].values)
                        ds["time"] = pd.to_datetime(ds["time"].dt.strftime("%Y-%m-15T00:00:00"))
                    except Exception:
                        ds["time"] = mid_month
                # Use mid-month index for monthly comparison/fill
                time_index = mid_month
                time_var = ds.time
                # Remove potential duplicate timestamps created by monthly normalization
                try:
                    _, index_unique = np.unique(ds["time"], return_index=True)
                    ds = ds.isel(time=np.sort(index_unique))
                    time_var = ds.time
                except Exception as e:
                    logger.warning("Failed to remove duplicate monthly timestamps: %s", e)
                # Safely set calendar attribute using encoding
                try:
                    time_var.encoding["calendar"] = "proleptic_gregorian"
                except Exception:
                    try:
                        new_time = xr.DataArray(
                            time_var.values, dims=["time"], attrs={"calendar": "proleptic_gregorian"}
                        )
                        ds = ds.assign_coords(time=new_time)
                        time_var = ds.time
                    except Exception:
                        logging.debug("Could not set calendar attribute for monthly data; proceeding.")
                        pass
                # For monthly data: only check by Year-Month presence and avoid reindexing/filling.
                expected_months = pd.period_range(start=f"{syear}-01", end=f"{eyear}-12", freq="M")
                # Build a PeriodIndex from the existing timestamps for robust monthly comparison
                try:
                    present_months = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="M")
                except Exception:
                    # Fallback: coerce via Series then to_period
                    present_months = pd.to_datetime(pd.Series(ds["time"].values))
                    present_months = present_months.dt.to_period("M")
                missing_months = expected_months.difference(pd.PeriodIndex(present_months, freq="M"))
                # If months are missing, reindex to monthly midpoints and fill with NaN for missing months
                if len(missing_months) > 0:
                    logging.info(
                        f"Monthly data has {len(missing_months)} missing month(s) between {syear} and {eyear}; filling missing months with NaN."
                    )
                    ds = ds.reindex(time=time_index)
                return ds
            elif time_unit.lower() in ["d", "day", "1d", "1day"]:
                # Normalize to daily resolution (set to 12:00), and fill missing days by reindexing.
                day_noon = pd.to_datetime(pd.Series(time_index).dt.strftime("%Y-%m-%dT12:00:00"))
                try:
                    ds["time"] = pd.to_datetime(ds["time"].dt.floor("D").dt.strftime("%Y-%m-%dT12:00:00"))
                except (ValueError, AttributeError, TypeError):
                    try:
                        ds["time"] = pd.to_datetime(ds["time"].values)
                        ds["time"] = pd.to_datetime(
                            pd.Series(ds["time"]).dt.floor("D").dt.strftime("%Y-%m-%dT12:00:00")
                        )
                    except Exception:
                        ds["time"] = day_noon
                # Use daily noon index for comparison/fill
                time_index = day_noon
                time_var = ds.time
                # Remove duplicates potentially created by normalization
                try:
                    _, index_unique = np.unique(ds["time"], return_index=True)
                    ds = ds.isel(time=np.sort(index_unique))
                    time_var = ds.time
                except Exception as e:
                    logger.warning("Failed to remove duplicate daily timestamps: %s", e)
                # Safely set calendar attribute using encoding
                try:
                    time_var.encoding["calendar"] = "proleptic_gregorian"
                except Exception:
                    try:
                        new_time = xr.DataArray(
                            time_var.values, dims=["time"], attrs={"calendar": "proleptic_gregorian"}
                        )
                        ds = ds.assign_coords(time=new_time)
                        time_var = ds.time
                    except Exception:
                        logging.debug("Could not set calendar attribute for daily data; proceeding.")
                        pass
                # Check by date presence and fill missing by reindexing
                expected_days = pd.period_range(start=f"{syear}-01-01", end=f"{eyear}-12-31", freq="D")
                try:
                    present_days = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="D")
                except Exception:
                    # Fallback: coerce to datetime first, then derive daily periods safely.
                    fallback_days = pd.to_datetime(pd.Series(ds["time"].values))
                    fallback_days = fallback_days.dt.to_period("D")
                    present_days = pd.PeriodIndex(fallback_days, freq="D")
                missing_days = expected_days.difference(present_days)
                if len(missing_days) > 0:
                    logging.info(
                        f"Daily data has {len(missing_days)} missing day(s) between {syear} and {eyear}; filling missing days with NaN."
                    )
                    ds = ds.reindex(time=time_index)
                return ds
            elif time_unit.lower() in ["h", "hour", "1h", "1hour"]:
                # Normalize to hourly resolution (set to HH:30), and fill missing hours by reindexing.
                hour_mid = pd.to_datetime(pd.Series(time_index).dt.floor("H").dt.strftime("%Y-%m-%dT%H:30:00"))
                try:
                    ds["time"] = pd.to_datetime(pd.Series(ds["time"]).dt.floor("H").dt.strftime("%Y-%m-%dT%H:30:00"))
                except (ValueError, AttributeError, TypeError):
                    try:
                        ds["time"] = pd.to_datetime(ds["time"].values)
                        ds["time"] = pd.to_datetime(
                            pd.Series(ds["time"]).dt.floor("H").dt.strftime("%Y-%m-%dT%H:30:00")
                        )
                    except Exception:
                        ds["time"] = hour_mid
                # Use mid-hour index for comparison/fill
                time_index = hour_mid
                time_var = ds.time
                # Remove duplicates potentially created by normalization
                try:
                    _, index_unique = np.unique(ds["time"], return_index=True)
                    ds = ds.isel(time=np.sort(index_unique))
                    time_var = ds.time
                except Exception as e:
                    logger.warning("Failed to remove duplicate hourly timestamps: %s", e)
                # Safely set calendar attribute using encoding
                try:
                    time_var.encoding["calendar"] = "proleptic_gregorian"
                except Exception:
                    try:
                        new_time = xr.DataArray(
                            time_var.values, dims=["time"], attrs={"calendar": "proleptic_gregorian"}
                        )
                        ds = ds.assign_coords(time=new_time)
                        time_var = ds.time
                    except Exception:
                        logging.debug("Could not set calendar attribute for hourly data; proceeding.")
                        pass
                # Check by hour presence and fill missing by reindexing
                expected_hours = pd.period_range(start=f"{syear}-01-01", end=f"{eyear}-12-31 23:00:00", freq="h")
                try:
                    present_hours = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="h")
                except Exception:
                    present_hours = pd.to_datetime(pd.Series(ds["time"].values))
                    present_hours = present_hours.dt.to_period("h")
                missing_hours = expected_hours.difference(pd.PeriodIndex(present_hours, freq="h"))
                if len(missing_hours) > 0:
                    logging.info(
                        f"Hourly data has {len(missing_hours)} missing hour(s) between {syear} and {eyear}; filling missing hours with NaN."
                    )
                    ds = ds.reindex(time=time_index)
                return ds
            elif time_unit.lower() in ["y", "year", "1y", "1year", "ye", "1ye", "a", "1a"]:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime("%Y-01-01T00:00:00"))
                try:
                    ds["time"] = pd.to_datetime(ds["time"].dt.strftime("%Y-01-01T00:00:00"))
                except (ValueError, AttributeError, TypeError):
                    ds["time"] = time_index
            time_var = ds.time
            # Safely set calendar attribute using encoding
            try:
                time_var.encoding["calendar"] = "proleptic_gregorian"
            except Exception:
                try:
                    # If encoding fails, create new time coordinate with calendar attribute
                    new_time = xr.DataArray(time_var.values, dims=["time"], attrs={"calendar": "proleptic_gregorian"})
                    ds = ds.assign_coords(time=new_time)
                    time_var = ds.time
                except Exception:
                    # If all else fails, just continue without setting calendar
                    logging.debug("Could not set calendar attribute, proceeding without it")
                    pass
            time_values = time_var

            # Create a complete time series based on the specified time frequency and range
            # Compare the actual time with the complete time series to find the missing time
            missing_times = time_index[~np.isin(time_index, time_values)]
            if len(missing_times) > 0:
                logging.warning("Time series is not complete. Missing time values found.")
                logging.info("Filling missing time values with np.nan")
                # Cast integer-typed data variables to float before reindex
                # so np.nan can actually be stored. xarray silently coerces
                # NaN → 0 when assigning to int dtype, corrupting fills.
                if hasattr(ds, "data_vars"):
                    for vname, var in list(ds.data_vars.items()):
                        if np.issubdtype(var.dtype, np.integer):
                            ds[vname] = var.astype("float64")
                elif np.issubdtype(getattr(ds, "dtype", np.dtype("O")), np.integer):
                    ds = ds.astype("float64")
                # Fill missing time values with np.nan. reindex already
                # preserves matching timestamps and inserts NaN for missing
                # ones; a second exact isin() mask can erase valid values when
                # timezone/precision representations differ.
                ds = ds.reindex(time=time_index)
        return ds
