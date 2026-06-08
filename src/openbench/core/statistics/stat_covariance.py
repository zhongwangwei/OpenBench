# -*- coding: utf-8 -*-
import xarray as xr

from .stat_correlation import _align_finite_time_pair


def stat_covariance(self, data1, data2):
    """
    Calculate the covariance of the input data.

    Args:
        data1 (xarray.DataArray): First dataset
        data2 (xarray.DataArray): Second dataset

    Returns:
        xarray.DataArray: Covariance of the input data
    """
    data1, data2 = _align_finite_time_pair(data1, data2)
    return xr.cov(data1, data2, dim="time").rename("Covariance")
