import xarray as xr
import logging

def stat_correlation(self, data1, data2):
    """
    Calculate the correlation coefficient between two datasets.

    Args:
        data1 (xarray.DataArray or xarray.Dataset): First dataset
        data2 (xarray.DataArray or xarray.Dataset): Second dataset

    Returns:
        xarray.DataArray: Correlation coefficient between the two datasets
    """
    if isinstance(data1, xr.Dataset) and isinstance(data2, xr.Dataset):
        # Assume single-variable datasets and extract the variable
        data1 = list(data1.data_vars.values())[0]
        data2 = list(data2.data_vars.values())[0]

    if isinstance(data1, xr.DataArray) and isinstance(data2, xr.DataArray):
        return xr.corr(data1, data2, dim="time").to_dataset(name=f"Correlation")
    else:
        logging.error("Input must be either two xarray Datasets with single variables or two xarray DataArrays")
        raise TypeError("Input must be either two xarray Datasets with single variables or two xarray DataArrays")

