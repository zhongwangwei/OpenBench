# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr


def stat_diff(self, data):
    """
    Calculate the difference of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Difference of the input data
    """
    return data.diff(dim="time")
