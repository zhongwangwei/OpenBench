# -*- coding: utf-8 -*-
import logging
import re

import numpy as np
import pandas as pd
import xarray as xr

# CF calendars that produce cftime objects when xarray decodes them.
# `proleptic_gregorian` and `gregorian` decode to numpy.datetime64 directly.
_CFTIME_CALENDARS = {
    "noleap",
    "365_day",
    "all_leap",
    "366_day",
    "360_day",
    "julian",
}


def normalize_cftime_axis(
    ds: xr.Dataset,
    *,
    source_path: str | None = None,
) -> xr.Dataset:
    """Convert a cftime time coordinate to numpy datetime64.

    Climate model output frequently uses non-Gregorian CF calendars
    (``noleap``, ``360_day``, ``julian`` …). xarray decodes these into
    ``cftime`` objects, which most of OpenBench's downstream pipeline
    cannot consume (`pd.to_datetime`, `.dt.strftime`, etc. all raise
    `TypeError` on cftime).

    Previously the pipeline silently fell back to a Gregorian
    ``pd.date_range`` so a noleap year ended up with 366 days,
    introducing a ~1 day-per-year systematic offset for monthly/daily
    climatology comparisons.

    This helper:
      1. Detects the calendar attribute.
      2. Logs which calendar was found.
      3. Uses xarray's native ``xr.coding.times.cftime_to_nptime`` to
         convert to ``datetime64[ns]`` while preserving the original
         calendar string in the time coord's attrs (for downstream
         calendar-aware unit conversion, e.g. days_in_month).

    Returns the dataset unchanged if its time axis is already
    ``datetime64`` (the common case) or has no time coord at all.
    """
    if "time" not in ds.coords:
        return ds
    time_var = ds["time"]
    if np.issubdtype(time_var.dtype, np.datetime64):
        return ds

    # cftime values present — capture the calendar before we convert
    calendar = time_var.encoding.get("calendar") or time_var.attrs.get("calendar") or "unknown"

    # Try to convert via xarray's helper first (handles all cftime types)
    try:
        from xarray.coding.times import cftime_to_nptime

        new_values = cftime_to_nptime(time_var.values)
    except Exception as exc:
        logging.warning(
            "Could not convert cftime axis (%s, calendar=%s): %s. "
            "Downstream time alignment may produce inconsistent results.",
            source_path or "<dataset>",
            calendar,
            exc,
        )
        return ds

    new_time = xr.DataArray(
        new_values,
        dims=time_var.dims,
        attrs={**time_var.attrs, "original_calendar": calendar},
    )
    ds = ds.assign_coords(time=new_time)

    if calendar in _CFTIME_CALENDARS:
        logging.info(
            "Converted CF calendar '%s' to datetime64 for %s "
            "(note: this may introduce small per-step offsets for non-"
            "Gregorian calendars; original calendar preserved in attrs).",
            calendar,
            source_path or "<dataset>",
        )
    return ds


def decode_nonstandard_time(ds: xr.Dataset, source_path: str | None = None) -> xr.Dataset:
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
    units_str = time_var.attrs.get("units", "") or time_var.encoding.get("units", "") or ""
    # Strip trailing semicolons, whitespace, and other junk
    units_str = re.sub(r"[;\s]+$", "", units_str).strip()

    offsets = time_var.values.astype(float)

    # ── Strategy 1: "<unit> since <ref_date>" patterns ───────────────────
    if units_str:
        new_time = _decode_unit_since(units_str, offsets, source_path=source_path)
        if new_time is not None:
            ds = ds.assign_coords(time=new_time)
            logging.info(
                "Decoded non-standard time: '%s' → %d datetime steps",
                units_str,
                len(new_time),
            )
            return ds

    # ── Strategy 2: numeric-only time values (no units) ──────────────────
    if not units_str:
        new_time = _decode_numeric_time(offsets)
        if new_time is not None:
            ds = ds.assign_coords(time=new_time)
            logging.info(
                "Decoded numeric time values → %d datetime steps",
                len(new_time),
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
    "month": "month",
    "months": "month",
    "mon": "month",
    "mons": "month",
    "year": "year",
    "years": "year",
    "yr": "year",
    "yrs": "year",
    "common_year": "year",
    "common_years": "year",
    "season": "season",
    "seasons": "season",
    "day": "day",
    "days": "day",
    "d": "day",
    "hour": "hour",
    "hours": "hour",
    "hr": "hour",
    "hrs": "hour",
    "h": "hour",
    "minute": "minute",
    "minutes": "minute",
    "min": "minute",
    "mins": "minute",
    "second": "second",
    "seconds": "second",
    "sec": "second",
    "secs": "second",
    "s": "second",
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

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m", "%Y", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return np.datetime64(_dt.strptime(ref_str, fmt))
        except ValueError:
            continue
    return None


def _decode_unit_since(
    units_str: str,
    offsets: np.ndarray,
    *,
    source_path: str | None = None,
) -> np.ndarray | None:
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
        if _looks_like_scaled_annual_month_axis(offsets, ref_ts, source_path):
            offsets = np.arange(12, dtype=float)
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


def _looks_like_scaled_annual_month_axis(
    offsets: np.ndarray,
    ref_ts: pd.Timestamp,
    source_path: str | None,
) -> bool:
    """Detect model files that encode one monthly year as 0,3,...,33.

    These files are annual ``MYYYY`` outputs with 12 monthly slices, but the
    non-CF ``calendar months since`` coordinate is scaled by 3. Treating it
    literally expands one annual file into three years and makes consecutive
    year files overlap.
    """
    values = np.asarray(offsets, dtype=float)
    if (
        not source_path
        or values.size != 12
        or ref_ts.month != 1
        or ref_ts.day != 1
        or not re.search(r"(^|[^A-Za-z0-9])M\d{4}([^A-Za-z0-9]|$)", str(source_path))
    ):
        return False
    return bool(np.allclose(values, np.arange(12, dtype=float) * 3))


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
