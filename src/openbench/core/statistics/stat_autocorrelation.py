# -*- coding: utf-8 -*-

import warnings

import numpy as np
import xarray as xr


def _single_dataarray(data):
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError("autocorrelation requires exactly one data variable")
        return next(iter(data.data_vars.values()))
    return data


def stat_autocorrelation(self, data):
    """
    Calculate the autocorrelation of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Autocorrelation of the input data
    """
    data = _single_dataarray(data)
    if "time" not in data.dims:
        raise ValueError("autocorrelation requires a 'time' dimension")
    # Need at least 3 samples for a meaningful lag-1 correlation: with 2
    # samples the shifted arrays have length 1 and xr.corr silently returns
    # NaN, which looked like a real autocorrelation value of NaN rather than
    # an "input too short" signal.
    if data.sizes["time"] < 3:
        warnings.warn(
            f"autocorrelation: time series has only {data.sizes['time']} samples; "
            "need ≥3 for lag-1 correlation, returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return data.isel(time=0, drop=True) * np.nan

    left = data.isel(time=slice(None, -1)).assign_coords(time=np.arange(data.sizes["time"] - 1))
    right = data.isel(time=slice(1, None)).assign_coords(time=np.arange(data.sizes["time"] - 1))
    result = xr.corr(left, right, dim="time")
    result.name = "Autocorrelation"
    return result
