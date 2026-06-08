import logging

import numpy as np
import xarray as xr


def _as_single_dataarray(data, label):
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(f"{label} Dataset must contain exactly one data variable")
        return next(iter(data.data_vars.values()))
    if isinstance(data, xr.DataArray):
        return data
    logging.error("Input must be either two xarray Datasets with single variables or two xarray DataArrays")
    raise TypeError("Input must be either two xarray Datasets with single variables or two xarray DataArrays")


def _align_finite_time_pair(data1, data2):
    data1 = _as_single_dataarray(data1, "First")
    data2 = _as_single_dataarray(data2, "Second")
    if "time" not in data1.dims or "time" not in data2.dims:
        raise ValueError("Both inputs must contain a 'time' dimension")
    data1, data2 = xr.align(data1, data2, join="inner")
    mask = np.isfinite(data1) & np.isfinite(data2)
    return data1.where(mask), data2.where(mask)


def stat_correlation(self, data1, data2):
    """
    Calculate the correlation coefficient between two datasets.

    Args:
        data1 (xarray.DataArray or xarray.Dataset): First dataset
        data2 (xarray.DataArray or xarray.Dataset): Second dataset

    Returns:
        xarray.DataArray: Correlation coefficient between the two datasets
    """
    data1, data2 = _align_finite_time_pair(data1, data2)
    return xr.corr(data1, data2, dim="time").to_dataset(name="Correlation")
