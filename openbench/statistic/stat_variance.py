# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

def stat_variance(self, data):
    """
    Calculate the variance of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Variance of the input data
    """
    return data.var(dim="time")