# -*- coding: utf-8 -*-

import xarray as xr


def stat_rolling(self, data, window):
    """
    Rolling window of the input data.

    Args:
        data (xarray.DataArray): Input data (Dataset accepted, unwrapped to
            its single variable to stay consistent with other stat_* modules).
        window (int): Window size

    Returns:
        xarray.DataArray: Rolling window of the input data
    """
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(
                f"stat_rolling expects a single-variable Dataset, got {len(data.data_vars)}: {list(data.data_vars)}"
            )
        data = next(iter(data.data_vars.values()))
    return data.rolling(time=window).mean()
