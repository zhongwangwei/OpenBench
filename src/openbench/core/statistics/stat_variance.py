# -*- coding: utf-8 -*-


def stat_variance(self, data):
    """
    Calculate the variance of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Variance of the input data
    """
    return data.var(dim="time")
