# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

def stat_covariance(self, data1, data2):
    """
    Calculate the covariance of the input data.

    Args:
        data1 (xarray.DataArray): First dataset
        data2 (xarray.DataArray): Second dataset

    Returns:
        xarray.DataArray: Covariance of the input data
    """
    return xr.cov(data1, data2, dim="time")