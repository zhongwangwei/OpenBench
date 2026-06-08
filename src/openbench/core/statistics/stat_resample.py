# -*- coding: utf-8 -*-

import xarray as xr


def stat_resample(self, data, time):
    """
    Resample the input data.

    Args:
        data (xarray.DataArray): Input data (Dataset accepted, unwrapped to
            its single variable to stay consistent with other stat_* modules
            — multi-variable Datasets are explicitly rejected so callers
            don't accidentally get an aggregated multi-var result).
        time (str): Resampling frequency

    Returns:
        xarray.DataArray: Resampled data
    """
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(
                f"stat_resample expects a single-variable Dataset, got {len(data.data_vars)}: {list(data.data_vars)}"
            )
        data = next(iter(data.data_vars.values()))
    return data.resample(time=time).mean()
