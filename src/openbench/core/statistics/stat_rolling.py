# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

def stat_rolling(self, data, window):
    """
    Rolling window of the input data.

    Args:
        data (xarray.DataArray): Input data
        window (int): Window size

    Returns:
        xarray.DataArray: Rolling window of the input data
    """
    return data.rolling(time=window)