# -*- coding: utf-8 -*-
import logging
import re

import numpy as np
import pandas as pd
import xarray as xr


def decode_nonstandard_time(ds: xr.Dataset) -> xr.Dataset:
    """Decode non-standard time units that xarray/cftime cannot handle.

    This function is called automatically when ``xr.open_dataset()`` falls
    back to ``decode_times=False``.  It inspects the raw time variable's
    ``units`` attribute, recognises a wide range of non-CF patterns, and
    converts the numeric offsets into proper ``datetime64[ns]`` coordinates.

    Supported patterns
    ------------------
    CF-like but rejected by xarray
        ``calendar months since 1850-01-01``
        ``calendar years since 1850-01-01``
        ``months since 1850-1-1``
        ``years since 1850-1-1 00:00:00``

    Non-CF calendar units
        ``common_years since 1850-01-01``
        ``seasons since 2000-01-01``

    Numeric-only time values (no units string)
        Integer/float arrays that look like ``YYYYMM`` or ``YYYYMMDD`` or ``YYYY``
        (heuristic, enabled only when no units attribute exists)

    Trailing junk in units string
        ``calendar months since 2002-01-01 00:00:00 ;``
        (semicolons, trailing whitespace, etc. are stripped)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened with ``decode_times=False``.

    Returns
    -------
    xr.Dataset
        Dataset with decoded ``time`` coordinate, or the original dataset
        unchanged if no applicable pattern is found.
    """
    if "time" not in ds.coords:
        return ds
    time_var = ds["time"]

    # Already decoded
    if np.issubdtype(time_var.dtype, np.datetime64):
        return ds

    # ── Gather the units string ──────────────────────────────────────────
    units_str = (
        time_var.attrs.get("units", "")
        or time_var.encoding.get("units", "")
        or ""
    )
    # Strip trailing semicolons, whitespace, and other junk
    units_str = re.sub(r"[;\s]+$", "", units_str).strip()

    offsets = time_var.values.astype(float)

    # ── Strategy 1: "<unit> since <ref_date>" patterns ───────────────────
    if units_str:
        new_time = _decode_unit_since(units_str, offsets)
        if new_time is not None:
            ds = ds.assign_coords(time=new_time)
            logging.info(
                "Decoded non-standard time: '%s' → %d datetime steps",
                units_str, len(new_time),
            )
            return ds

    # ── Strategy 2: numeric-only time values (no units) ──────────────────
    if not units_str:
        new_time = _decode_numeric_time(offsets)
        if new_time is not None:
            ds = ds.assign_coords(time=new_time)
            logging.info(
                "Decoded numeric time values → %d datetime steps", len(new_time),
            )
            return ds

    return ds


# ── Internal helpers ─────────────────────────────────────────────────────

# Regex: optional qualifier + unit + "since" + reference date
_UNIT_SINCE_RE = re.compile(
    r"^(?:calendar\s+|common_?\s*)?(\w+)\s+since\s+(.+)$",
    re.IGNORECASE,
)

# Map unit tokens to a canonical form
_UNIT_MAP = {
    "month": "month", "months": "month", "mon": "month", "mons": "month",
    "year": "year", "years": "year", "yr": "year", "yrs": "year",
    "common_year": "year", "common_years": "year",
    "season": "season", "seasons": "season",
    "day": "day", "days": "day", "d": "day",
    "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour", "h": "hour",
    "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
    "second": "second", "seconds": "second", "sec": "second", "secs": "second", "s": "second",
}


def _parse_ref_date(ref_str: str) -> np.datetime64 | None:
    """Try hard to parse a reference date string."""
    # Strip trailing junk
    ref_str = re.sub(r"[;\s]+$", "", ref_str).strip()
    # Direct numpy parse
    try:
        return np.datetime64(ref_str)
    except ValueError:
        pass
    # Try pandas (very flexible)
    try:
        return np.datetime64(pd.Timestamp(ref_str))
    except (ValueError, TypeError):
        pass
    # Common formats
    from datetime import datetime as _dt
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y-%m", "%Y", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return np.datetime64(_dt.strptime(ref_str, fmt))
        except ValueError:
            continue
    return None


def _decode_unit_since(units_str: str, offsets: np.ndarray) -> np.ndarray | None:
    """Decode ``<unit> since <ref_date>`` patterns."""
    m = _UNIT_SINCE_RE.match(units_str)
    if not m:
        return None

    raw_unit, ref_str = m.group(1).lower(), m.group(2)
    canon = _UNIT_MAP.get(raw_unit)
    if canon is None:
        return None

    ref = _parse_ref_date(ref_str)
    if ref is None:
        logging.warning("Cannot parse reference date '%s' in time units '%s'", ref_str, units_str)
        return None

    ref_ts = pd.Timestamp(ref)

    if canon == "month":
        dates = []
        for off in offsets:
            total = ref_ts.month - 1 + int(off)
            y = ref_ts.year + total // 12
            mo = total % 12 + 1
            # Preserve the day from ref if possible, clamp to month end
            day = min(ref_ts.day, pd.Timestamp(year=y, month=mo, day=1).days_in_month)
            dates.append(pd.Timestamp(year=y, month=mo, day=day))
        return np.array(dates, dtype="datetime64[ns]")

    if canon == "year":
        dates = []
        for off in offsets:
            y = ref_ts.year + int(off)
            # Preserve month/day from ref
            mo = ref_ts.month
            day = min(ref_ts.day, pd.Timestamp(year=y, month=mo, day=1).days_in_month)
            dates.append(pd.Timestamp(year=y, month=mo, day=day))
        return np.array(dates, dtype="datetime64[ns]")

    if canon == "season":
        # 1 season = 3 months
        dates = []
        for off in offsets:
            total_months = ref_ts.month - 1 + int(off) * 3
            y = ref_ts.year + total_months // 12
            mo = total_months % 12 + 1
            dates.append(pd.Timestamp(year=y, month=mo, day=1))
        return np.array(dates, dtype="datetime64[ns]")

    # Standard CF units that xarray should handle — try pd.to_timedelta as
    # fallback (handles fractional days/hours/etc.)
    if canon in ("day", "hour", "minute", "second"):
        try:
            td_unit = {"day": "D", "hour": "h", "minute": "min", "second": "s"}[canon]
            deltas = pd.to_timedelta(offsets, unit=td_unit)
            return np.array(ref_ts + deltas, dtype="datetime64[ns]")
        except Exception:
            return None

    return None


def _decode_numeric_time(offsets: np.ndarray) -> np.ndarray | None:
    """Heuristic decode for numeric-only time values (no units attribute).

    Recognises:
    * ``YYYYMMDD`` (e.g. 20040115)
    * ``YYYYMM``   (e.g. 200401)
    * ``YYYY``     (e.g. 2004)
    * ``YYYY.frac`` (e.g. 2004.5 → mid-year)
    """
    if len(offsets) == 0:
        return None

    vmin, vmax = float(np.nanmin(offsets)), float(np.nanmax(offsets))

    # YYYYMMDD  (e.g. 19800101 .. 20231231)
    if 10000101 <= vmin and vmax <= 99991231:
        try:
            dates = pd.to_datetime(offsets.astype(int).astype(str), format="%Y%m%d")
            return np.array(dates, dtype="datetime64[ns]")
        except Exception:
            pass

    # YYYYMM  (e.g. 198001 .. 202312)
    if 100001 <= vmin and vmax <= 999912:
        try:
            dates = pd.to_datetime(offsets.astype(int).astype(str), format="%Y%m")
            return np.array(dates, dtype="datetime64[ns]")
        except Exception:
            pass

    # YYYY or YYYY.frac  (e.g. 1980 .. 2023  or  1980.0 .. 2023.5)
    if 1800 <= vmin <= 2200 and 1800 <= vmax <= 2200:
        dates = []
        for v in offsets:
            year = int(v)
            frac = v - year
            base = pd.Timestamp(year=year, month=1, day=1)
            if frac > 0:
                end = pd.Timestamp(year=year + 1, month=1, day=1)
                base = base + (end - base) * frac
            dates.append(base)
        return np.array(dates, dtype="datetime64[ns]")

    return None


class timelib:
    def __init__(self):
        self.name = "DatasetPreprocessing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "Mar 2023"
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.freq_map = {
            "month": "ME",
            "mon": "ME",
            "monthly": "ME",
            "day": "D",
            "daily": "D",
            "hour": "h",  # Changed from 'H' to 'h' to avoid FutureWarning
            "hr": "h",  # Changed from 'H' to 'h' to avoid FutureWarning
            "hourly": "h",  # Changed from 'H' to 'h' to avoid FutureWarning
            "year": "Y",
            "yr": "Y",
            "yearly": "Y",
            "week": "W",
            "wk": "W",
            "weekly": "W",
        }

    def check_time(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        print("Checking time coordinate...")
        if "time" not in ds.coords:
            logging.info("The dataset does not contain a 'time' coordinate.")
            # Based on the syear and eyear, create a time index
            time_index = pd.date_range(start=f"{syear}-01-01T00:00:00", end=f"{eyear}-12-31T23:59:59", freq=tim_res)
            ds = ds.expand_dims("time")  # Ensure 'time' dimension exists
            ds = ds.assign_coords(time=time_index)  # Assign the created time index to the dataset
        elif not np.issubdtype(ds.time.dtype, np.datetime64):
            try:
                ds["time"] = pd.to_datetime(ds.time.values)
            except (ValueError, TypeError, AttributeError):
                # Delete the time coordinate
                ds = ds.drop("time")
                # Delete the time dimension
                ds = ds.squeeze("time")
                time_index = pd.date_range(start=f"{syear}-01-01T00:00:00", end=f"{eyear}-12-31T23:59:59", freq=tim_res)
                ds = ds.expand_dims("time")  # Ensure 'time' dimension exists
                ds = ds.assign_coords(time=time_index)  # Assign the created time index to the dataset
        else:
            # Check for duplicate time values and remove them
            if ds["time"].to_index().duplicated().any():
                logging.info("Duplicate time values found. Removing duplicates.")
                _, unique_indices = np.unique(ds["time"], return_index=True)
                ds = ds.isel(time=unique_indices)

            # Replace the existing time index with a new one based on syear, eyear, and tim_res
            time_index = pd.date_range(start=f"{syear}-01-01T00:00:00", end=f"{eyear}-12-31T23:59:59", freq=tim_res)
            ds = ds.reindex(time=time_index)  # Reindex the dataset with the created time index

        return ds
