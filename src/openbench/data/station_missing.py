"""Missing-value handling shared by station input paths."""

from __future__ import annotations

import numpy as np
import xarray as xr

_DEFAULT_SENTINELS = (-999.0, -9999.0)


def missing_sentinels(attrs: dict | None = None, encoding: dict | None = None) -> tuple[float, ...]:
    """Return the default and metadata-advertised numeric sentinels."""
    sentinels = set(_DEFAULT_SENTINELS)
    for metadata in (attrs or {}, encoding or {}):
        for key in ("_FillValue", "missing_value"):
            value = metadata.get(key)
            values = np.ravel(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
            for item in values:
                try:
                    numeric = float(item)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    sentinels.add(numeric)
    return tuple(sentinels)


def valid_station_mask(values, sentinels: tuple[float, ...] = ()) -> np.ndarray:
    """Return finite values that are not station missing-value sentinels."""
    numeric = np.asarray(values, dtype=float)
    mask = np.isfinite(numeric)
    for sentinel in sentinels or _DEFAULT_SENTINELS:
        mask &= numeric != sentinel
    return mask


def mask_station_missing(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Replace station sentinels with NaN before conversion or resampling."""
    if isinstance(data, xr.DataArray):
        if not np.issubdtype(data.dtype, np.number):
            return data
        sentinels = missing_sentinels(data.attrs, data.encoding)
        valid = np.isfinite(data)
        for sentinel in sentinels:
            valid &= data != sentinel
        return data.where(valid)

    masked = data.copy()
    for name, variable in data.data_vars.items():
        if np.issubdtype(variable.dtype, np.number):
            sentinels = missing_sentinels(variable.attrs, variable.encoding)
            valid = np.isfinite(variable)
            for sentinel in sentinels:
                valid &= variable != sentinel
            masked[name] = variable.where(valid)
    return masked
