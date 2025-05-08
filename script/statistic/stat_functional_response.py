import xarray as xr
import logging
import numpy as np


def stat_functional_response(self, v, u):
    """
    Calculate the functional response score for each grid point along the time dimension.

    Args:
        v (xarray.DataArray): Dependent variable
        u (xarray.DataArray): Independent variable
        nbins (int): Number of bins for the histogram
        output_file (str): Name of the output NetCDF file

    Returns:
        xarray.DataArray: Functional response score for each grid point
    """
    import pandas as pd

    if isinstance(v, xr.Dataset):
        v = list(v.data_vars.values())[0]
    if isinstance(u, xr.Dataset):
        u = list(u.data_vars.values())[0]

    try:
        nbins = self.stats_nml['Functional_Response']['nbins']
    except:
        nbins = self.compare_nml['Functional_Response']['nbins']

    def calc_functional_response(v_series, u_series):
        # Remove NaN values
        mask = ~np.isnan(v_series) & ~np.isnan(u_series)
        v_valid = v_series[mask]
        u_valid = u_series[mask]

        if len(v_valid) < 2:  # Not enough data points
            return np.nan

        # Create bins
        if u_valid.min() == u_valid.max():
            return np.nan

        u_bins = np.linspace(u_valid.min(), u_valid.max(), nbins + 1)

        # Calculate mean v for each bin
        df = pd.DataFrame({'u': u_valid, 'v': v_valid})
        binned_means = df.groupby(pd.cut(df['u'], bins=u_bins))['v'].mean()

        df['bin'] = pd.cut(df['u'], bins=u_bins)
        df['v_binned'] = df['bin'].map(binned_means)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((df['v_binned'].astype(float) - df['v'].astype(float)) ** 2))

        # Calculate relative error
        relative_error = rmse / np.mean(v_valid)

        # Calculate score
        score = np.exp(-relative_error)

        return score

    # Apply the function to each grid point
    score = xr.apply_ufunc(
        calc_functional_response,
        v, u,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    # Add attributes to the DataArray
    v_name = v.name if v.name is not None else 'unknown'
    u_name = u.name if u.name is not None else 'unknown'
    score.name = 'functional_response_score'
    score.attrs['long_name'] = 'Functional Response Score'
    score.attrs['units'] = '1'
    score.attrs['description'] = 'Functional response score calculated between variables ' + u_name + ' and ' + v_name
    # Create a dataset with the score
    ds = xr.Dataset({'functional_response_score': score})
    del score

    # Add global attributes
    ds.attrs['title'] = 'Functional Response Score'
    ds.attrs['description'] = 'Functional response score calculated between variables ' + u_name + ' and ' + v_name
    ds.attrs['created_by'] = 'ILAMB var_functional_response function'

    return ds
