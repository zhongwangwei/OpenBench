# -*- coding: utf-8 -*-


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
