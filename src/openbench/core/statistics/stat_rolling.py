# -*- coding: utf-8 -*-


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
