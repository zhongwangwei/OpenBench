import xarray as xr
import logging
import numpy as np


def stat_hellinger_distance(self, v, u):
    """
    Calculate the Hellinger Distance score for each grid point along the time dimension.

    Args:
        v (xarray.DataArray): First variable
        u (xarray.DataArray): Second variable
        nbins (int): Number of bins for the 2D histogram
        output_file (str): Name of the output NetCDF file

    Returns:
        xarray.DataArray: Hellinger Distance score for each grid point
    """
    nbins = self.stats_nml['Hellinger_Distance']['nbins']

    if isinstance(v, xr.Dataset):
        v = list(v.data_vars.values())[0]
    if isinstance(u, xr.Dataset):
        u = list(u.data_vars.values())[0]

    def calc_hellinger_distance(v_series, u_series):
        # Remove NaN values
        mask = ~np.isnan(v_series) & ~np.isnan(u_series)
        v_valid = v_series[mask]
        u_valid = u_series[mask]

        if len(v_valid) < 2:  # Not enough data points
            return np.nan

        # Calculate 2D histogram
        hist, _, _ = np.histogram2d(v_valid, u_valid, bins=nbins)

        # Normalize the histogram
        hist = hist / hist.sum()

        # Calculate Hellinger distance

        # 公式计算问题
        # hellinger_dist = np.sqrt(1 - np.sum(np.sqrt(hist)))
        hellinger_dist = np.sqrt(np.sum(np.sqrt(hist)))

        # print(hist)
        # print(type(hist))
        # print(hellinger_dist)
        # print(type(hellinger_dist))

        return hellinger_dist

    # Rechunk time dimension to single chunk for apply_ufunc with dask
    if hasattr(v, 'chunks') and v.chunks is not None:
        v = v.chunk({'time': -1})
    if hasattr(u, 'chunks') and u.chunks is not None:
        u = u.chunk({'time': -1})

    # Apply the function to each grid point
    score = xr.apply_ufunc(
        calc_hellinger_distance,
        v, u,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    # Add attributes to the DataArray
    v_name = v.name if v.name is not None else 'unknown'
    u_name = u.name if u.name is not None else 'unknown'
    score.name = 'hellinger_distance_score'
    score.attrs['long_name'] = 'Hellinger Distance Score'
    score.attrs['units'] = '-'
    score.attrs['description'] = 'Hellinger Distance score calculated between variables ' + u_name + ' and ' + v_name

    # Create a dataset with the score
    ds = xr.Dataset({'hellinger_distance_score': score})
    del score
    # Add global attributes
    ds.attrs['title'] = 'Hellinger Distance Score'
    ds.attrs['description'] = 'Hellinger Distance score calculated between variables ' + u_name + ' and ' + v_name
    ds.attrs['created_by'] = 'ILAMB var_hellinger_distance function'

    return ds