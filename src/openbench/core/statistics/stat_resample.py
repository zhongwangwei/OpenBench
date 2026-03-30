# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

def stat_resample(self, data, time):
    """
    Resample the input data.

    Args:
        data (xarray.DataArray): Input data
        time (str): Resampling frequency

    Returns:
        xarray.DataArray: Resampled data
    """
    return data.resample(time)