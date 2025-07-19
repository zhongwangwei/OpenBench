import xarray as xr

def stat_standard_deviation(self, data):
    """
    Calculate the standard deviation of the input data.

    Args:
        data (xarray.DataArray): Input data

    Returns:
        xarray.DataArray: Standard deviation of the input data
    """
    if isinstance(data, xr.Dataset):
        data = list(data.data_vars.values())[0]
    return data.std(dim="time")
