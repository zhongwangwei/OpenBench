# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import gc
from scipy import stats

def stat_mann_kendall_trend_test(self, data):
    """
    Calculates the Mann-Kendall trend test for a time series using scipy's kendalltau.

    Args:
        data (xarray.Dataset or xarray.DataArray): Time series data.

    Returns:
        xarray.Dataset: Dataset containing trend test results for each variable and grid point.
    """
    try:
        significance_level = self.stats_nml['Mann_Kendall_Trend_Test']['significance_level']
    except:
        significance_level = self.compare_nml['Mann_Kendall_Trend_Test']['significance_level']

    def _apply_mann_kendall(da, significance_level):
        """
        Applies Mann-Kendall test to a single DataArray using kendalltau.
        """

        def mk_test(x):
            if len(x) < 4 or np.all(np.isnan(x)):
                return np.array([np.nan, np.nan, np.nan, np.nan])

            # Remove NaN values
            x = x[~np.isnan(x)]

            if len(x) < 4:
                return np.array([np.nan, np.nan, np.nan, np.nan])

            # Calculate Kendall's tau and p-value
            tau, p_value = stats.kendalltau(np.arange(len(x)), x)

            # Determine trend
            trend = np.sign(tau)
            significance = p_value < significance_level

            return np.array([trend, significance, p_value, tau])

        try:
            # Apply the test to each grid point with chunking
            result = xr.apply_ufunc(
                mk_test,
                da,
                input_core_dims=[['time']],
                output_core_dims=[['mk_params']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'mk_params': 4}
            )

            # Create separate variables for each component
            trend = result.isel(mk_params=0)
            significance = result.isel(mk_params=1)
            p_value = result.isel(mk_params=2)
            tau = result.isel(mk_params=3)

            # Create a new Dataset with separate variables
            ds = xr.Dataset({
                'trend': trend,
                'significance': significance,
                'p_value': p_value,
                'tau': tau
            })

            # Add attributes
            ds.trend.attrs['long_name'] = 'Mann-Kendall trend'
            ds.trend.attrs['description'] = 'Trend direction: 1 (increasing), -1 (decreasing), 0 (no trend)'
            ds.significance.attrs['long_name'] = 'Trend significance'
            ds.significance.attrs['description'] = f'True if trend is significant at {significance_level} level, False otherwise'
            ds.p_value.attrs['long_name'] = 'p-value'
            ds.p_value.attrs['description'] = 'p-value of the Mann-Kendall trend test'
            ds.tau.attrs['long_name'] = "Kendall's tau statistic"
            ds.tau.attrs['description'] = "Kendall's tau correlation coefficient"

            ds.attrs['statistical_test'] = 'Mann-Kendall trend test (using Kendall\'s tau)'
            ds.attrs['significance_level'] = significance_level

            # Clean up intermediate result
            del result
            gc.collect()

            return ds
        finally:
            # Ensure cleanup of any remaining objects
            gc.collect()

    try:
        # Process the data with proper memory management
        if isinstance(data, xr.Dataset):
            # If it's a dataset, apply the test to each data variable
            results = []
            for var in data.data_vars:
                result = _apply_mann_kendall(data[var], significance_level)
                result = result.assign_coords(variable=var)
                results.append(result)
            # Save the result
            return xr.concat(results, dim='variable')
        elif isinstance(data, xr.DataArray):
            # If it's a DataArray, apply the test directly
            return _apply_mann_kendall(data, significance_level)
        else:
            logging.error("Input must be an xarray Dataset or DataArray")
            raise TypeError("Input must be an xarray Dataset or DataArray")

    finally:
        # Clean up any remaining objects
        gc.collect()