# -*- coding: utf-8 -*-

def stat_autocorrelation(self, data):
    """
    Calculate the autocorrelation of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Autocorrelation of the input data
    """
    return data.autocorr(dim="time")
