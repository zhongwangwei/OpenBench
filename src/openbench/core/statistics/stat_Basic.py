import numpy as np
import xarray as xr


def stat_mean(self, data):
    """
    Calculate the mean of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Mean of the input data
    """
    return data.mean(dim="time", skipna=True)


def stat_median(self, data):
    """
    Calculate the median of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Median of the input data
    """
    return data.median(dim="time", skipna=True)


def stat_max(self, data):
    """
    Calculate the max of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Max of the input data
    """
    return data.max(dim="time", skipna=True)


def stat_min(self, data):
    """
    Calculate the min of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Min of the input data
    """
    return data.min(dim="time", skipna=True)


def stat_sum(self, data):
    """
    Calculate the sum of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Sum of the input data
    """
    return data.sum(dim="time", skipna=True)
