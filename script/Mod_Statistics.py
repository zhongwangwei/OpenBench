# -*- coding: utf-8 -*-
import glob
import logging
import os
import re
import shutil
import warnings
from typing import List, Dict, Any

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats

from regrid.regrid_wgs84 import convert_to_wgs84_scipy, convert_to_wgs84_xesmf

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class statistics_calculate:
    """
    A class for performing various statistical analyses on xarray datasets.
    """

    def __init__(self, info):
        """
        Initialize the Statistics class.

        Args:
            info (dict): A dictionary containing additional information to be added as attributes.
        """
        self.name = 'statistics'
        self.version = '0.2'
        self.release = '0.2'
        self.date = 'Mar 2024'
        self.author = "Zhongwang Wei"
        self.__dict__.update(info)

    # Basic statistical methods
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
            return xr.corr(data1, data2, dim="time").to_dataset(name=f"correlation")
        else:
            raise TypeError("Input must be either two xarray Datasets with single variables or two xarray DataArrays")

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

    def stat_z_score(self, data):
        """
        Calculate the Z-score of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Z-score of the input data
        """
        if isinstance(data, xr.Dataset):
            data = list(data.data_vars.values())[0]
        return (data - data.mean(dim="time")) / data.std(dim="time")

    def stat_mean(self, data):
        """
        Calculate the mean of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Mean of the input data
        """
        return data.mean(dim="time")

    def stat_median(self, data):
        """
        Calculate the median of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Median of the input data
        """
        return data.median(dim="time")

    def stat_max(self, data):
        """
        Calculate the max of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Max of the input data
        """
        return data.max(dim="time")

    def stat_min(self, data):
        """
        Calculate the min of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Min of the input data
        """
        return data.min(dim="time")

    def stat_sum(self, data):
        """
        Calculate the sum of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Sum of the input data
        """
        return data.sum(dim="time")

    def stat_variance(self, data):
        """
        Calculate the variance of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Variance of the input data
        """
        return data.var(dim="time")

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

    def stat_autocorrelation(self, data):
        """
        Calculate the autocorrelation of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Autocorrelation of the input data
        """
        return data.autocorr(dim="time")

    def stat_diff(self, data):
        """
        Calculate the difference of the input data.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            xarray.DataArray: Difference of the input data
        """
        return data.diff(dim="time")

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

    # Advanced statistical methods
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

        nbins = self.stats_nml['Functional_Response']['nbins']

        def calc_functional_response(v_series, u_series):
            # Remove NaN values
            mask = ~np.isnan(v_series) & ~np.isnan(u_series)
            v_valid = v_series[mask]
            u_valid = u_series[mask]

            if len(v_valid) < 2:  # Not enough data points
                return np.nan

            # Create bins
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
            hellinger_dist = np.sqrt(np.sum(np.sqrt(hist)))

            return hellinger_dist

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

    def stat_three_cornered_hat(self, *variables):
        """
        Calculate uncertainty using the Three-Cornered Hat method.

        Args:
            *variables: Variable number of xarray DataArrays to compare

        Returns:
            xarray.Dataset: Dataset containing uncertainty and relative uncertainty
        """

        def cal_uct(arr):
            def my_fun(r):
                S = np.cov(arr.T)
                f = np.sum(r[:-1] ** 2)
                for j in range(len(S)):
                    for k in range(j + 1, len(S)):
                        f += (S[j, k] - r[-1] + r[j] + r[k]) ** 2
                K = np.linalg.det(S)
                F = f / (K ** (2 * len(S)))
                return F

            N = arr.shape[1]
            S = np.cov(arr.T)
            u = np.ones((1, N - 1))
            R = np.zeros((N, N))
            R[N - 1, N - 1] = 1 / (2 * np.dot(np.dot(u, np.linalg.inv(S)), u.T))

            x0 = R[:, N - 1]
            det_S = np.linalg.det(S)
            inv_S = np.linalg.inv(S)
            Denominator = det_S ** (2 / len(S))

            cons = {'type': 'ineq', 'fun': lambda r: (r[-1] - np.dot(
                np.dot(r[:-1] - r[-1] * u, inv_S),
                (r[:-1] - r[-1] * u).T)) / Denominator}

            x = optimize.minimize(my_fun, x0, method='COBYLA', tol=2e-10, constraints=cons)

            R[:, N - 1] = x.x
            for i in range(N - 1):
                for j in range(i, N - 1):
                    R[i, j] = S[i, j] - R[N - 1, N - 1] + R[i, N - 1] + R[j, N - 1]
            R += R.T - np.diag(R.diagonal())

            uct = np.sqrt(np.diag(R))
            r_uct = uct / np.mean(np.abs(arr), axis=0) * 100

            return uct, r_uct

        # Combine all variables into a single array
        combined_data = xr.concat(variables, dim='variable')
        # Prepare output arrays
        uct = xr.zeros_like(combined_data)
        r_uct = xr.zeros_like(combined_data)

        # Calculate uncertainty for each lat-lon point
        for lat in combined_data.lat:
            for lon in combined_data.lon:
                arr = combined_data.sel(lat=lat, lon=lon).values.T
                if not np.isnan(arr).all():
                    uct_values, r_uct_values = cal_uct(arr)
                    uct.loc[dict(lat=lat, lon=lon)] = uct_values
                    r_uct.loc[dict(lat=lat, lon=lon)] = r_uct_values

        # Create output dataset
        ds = xr.Dataset({
            'uncertainty': uct,
            'relative_uncertainty': r_uct
        })

        # Add metadata
        ds['uncertainty'].attrs['long_name'] = 'Uncertainty from Three-Cornered Hat method'
        ds['uncertainty'].attrs['units'] = 'Same as input variables'
        ds['relative_uncertainty'].attrs['long_name'] = 'Relative uncertainty from Three-Cornered Hat method'
        ds['relative_uncertainty'].attrs['units'] = '%'

        return ds

    def stat_partial_least_squares_regression(self, *variables):
        """
        Calculate the Partial Least Squares Regression (PLSR) analysis with cross-validation and parallel processing.

        Args:
            *variables: Variable number of xarray DataArrays. One should have '_Y' in its name as the dependent variable.
            max_components (int): Maximum number of components to consider
            n_splits (int): Number of splits for time series cross-validation
            n_jobs (int): Number of jobs for parallel processing

        Returns:
            xarray.Dataset: Dataset containing PLSR results
        """
        try:
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        except ImportError:
            raise ImportError("scikit-learn is required for this function")
        from scipy.stats import t

        # Prepare Dependent and Independent data
        # Separate Y and X variables
        # if isinstance(v, xr.Dataset):
        #     v = list(v.data_vars.values())[0]
        # if isinstance(u, xr.Dataset):
        #     u = list(u.data_vars.values())[0]

        max_components = self.stats_nml['Partial_Least_Squares_Regression']['max_components']
        n_splits = self.stats_nml['Partial_Least_Squares_Regression']['n_splits']
        n_jobs = self.stats_nml['Partial_Least_Squares_Regression']['n_jobs']

        Y_vars = variables[0]  # [var for var in variables if '_Y' in var.name]
        X_vars = variables[1:]  # [var for var in variables if '_Y' not in var.name]
        # if not Y_vars:
        #     raise ValueError("No dependent variable (Y) found. Ensure at least one variable has '_Y_' in its name.")

        # Prepare data
        Y_data = Y_vars.values
        X_data = np.array([x.values for x in X_vars])
        X_data = np.moveaxis(X_data, 0, 1)  # Reshape to (time, n_variables, lat, lon)

        # Standardize data
        X_mean = np.mean(X_data, axis=0)
        X_std = np.std(X_data, axis=0)
        X_stand = (X_data - X_mean) / X_std

        Y_mean = np.mean(Y_data, axis=0)
        Y_std = np.std(Y_data, axis=0)
        Y_stand = (Y_data - Y_mean) / Y_std

        # Define helper functions for parallel processing
        def compute_best_components(lat, lon):
            x = X_stand[:, :, lat, lon]
            y = Y_stand[:, lat, lon]
            if np.isnan(x).any() or np.isnan(y).any():
                return lat, lon, np.nan

            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            for n in range(1, max_components + 1):
                pls = PLSRegression(n_components=n, scale=False, max_iter=500)
                score = cross_val_score(pls, x, y, cv=tscv)
                scores.append(score.mean())

            best_n_components = np.argmax(scores) + 1
            return lat, lon, best_n_components

        def compute_plsr(lat, lon, n_components):
            x = X_stand[:, :, lat, lon]
            y = Y_stand[:, lat, lon]
            if np.isnan(x).any() or np.isnan(y).any():
                return lat, lon, np.full(X_data.shape[1], np.nan), np.nan, np.full(X_data.shape[1], np.nan), np.nan

            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(x, y)
            coef = pls.coef_.T
            intercept = pls.intercept_.T
            residuals = y - pls.predict(x).ravel()
            mse = np.mean(residuals ** 2)
            coef_std_err = np.sqrt(mse / len(y))
            df = len(y) - 1
            t_vals = coef.ravel() / coef_std_err
            p_vals = 2 * (1 - t.cdf(np.abs(t_vals), df))
            r_squared = pls.score(x, y)

            return lat, lon, coef.ravel(), intercept.ravel(), p_vals, r_squared
            # Compute best number of components

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_best_components)(lat, lon)
            for lat in range(Y_data.shape[1]) for lon in range(Y_data.shape[2])
        )

        best_n_components = np.zeros((Y_data.shape[1], Y_data.shape[2]), dtype=int)
        for lat, lon, n_components in results:
            if not np.isnan(n_components):
                best_n_components[lat, lon] = n_components

        # Compute PLSR results
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_plsr)(lat, lon, best_n_components[lat, lon])
            for lat in range(Y_data.shape[1]) for lon in range(Y_data.shape[2])
        )

        coef_values = np.zeros((X_data.shape[1], Y_data.shape[1], Y_data.shape[2]))
        intercept_values = np.zeros((X_data.shape[1], Y_data.shape[1], Y_data.shape[2]))
        p_values = np.zeros((X_data.shape[1], Y_data.shape[1], Y_data.shape[2]))
        r_squared_values = np.zeros((Y_data.shape[1], Y_data.shape[2]))

        for lat, lon, coef, intercept, p_vals, r_squared in results:
            coef_values[:, lat, lon] = coef
            intercept_values[:, lat, lon] = intercept
            p_values[:, lat, lon] = p_vals
            r_squared_values[lat, lon] = r_squared

        # Calculate anomaly
        anomaly = coef_values * Y_std[np.newaxis, :, :]
        # Create output dataset
        ds = xr.Dataset(
            data_vars={
                'best_n_components': (['lat', 'lon'], best_n_components),
                'coefficients': (['variable', 'lat', 'lon'], coef_values),
                'intercepts': (['variable', 'lat', 'lon'], intercept_values),
                'p_values': (['variable', 'lat', 'lon'], p_values),
                'r_squared': (['lat', 'lon'], r_squared_values),
                'anomaly': (['variable', 'lat', 'lon'], anomaly)
            },
            coords={
                'lat': Y_vars.lat,
                'lon': Y_vars.lon,
                'variable': [f'x{i + 1}' for i in range(len(X_vars))]
            }
        )

        # Add metadata
        ds['best_n_components'].attrs['long_name'] = 'Best number of components'
        ds['coefficients'].attrs['long_name'] = 'PLSR coefficients'
        ds['intercepts'].attrs['long_name'] = 'PLSR intercepts'
        ds['p_values'].attrs['long_name'] = 'P-values'
        ds['r_squared'].attrs['long_name'] = 'R-squared'
        ds['anomaly'].attrs['long_name'] = 'Anomaly (coefficients * Y standard deviation)'

        return ds

    def stat_mann_kendall_trend_test(self, data):
        """
        Calculates the Mann-Kendall trend test for a time series using scipy's kendalltau.

        Args:
            data (xarray.Dataset or xarray.DataArray): Time series data.

        Returns:
            xarray.Dataset: Dataset containing trend test results for each variable and grid point.
        """
        significance_level = self.stats_nml.get('mann_kendall_trend_test', {}).get('significance_level', 0.05)

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

            # Apply the test to each grid point
            result = xr.apply_ufunc(
                mk_test,
                da,
                input_core_dims=[['time']],
                output_core_dims=[['mk_params']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                output_sizes={'mk_params': 4}
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

            return ds

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
            raise TypeError("Input must be an xarray Dataset or DataArray")

    def stat_False_Discovery_Rate(self, *variables, alpha=0.05):
        """
        Perform optimized False Discovery Rate (FDR) analysis on multiple xarray datasets.

        Args:
            *variables: Variable number of xarray DataArrays to compare
            alpha (float): FDR control level, default is 0.05

        Returns:
            xarray.Dataset: Dataset containing p-values, FDR results, and metadata
        """

        def vectorized_ttest(a, b):
            a_mean = a.mean(dim='time')
            b_mean = b.mean(dim='time')
            a_var = a.var(dim='time')
            b_var = b.var(dim='time')
            a_count = a.count(dim='time')
            b_count = b.count(dim='time')

            t = (a_mean - b_mean) / da.sqrt(a_var / a_count + b_var / b_count)
            df = (a_var / a_count + b_var / b_count) ** 2 / (
                    (a_var / a_count) ** 2 / (a_count - 1) + (b_var / b_count) ** 2 / (b_count - 1)
            )

            # Use dask's map_overlap for efficient computation
            prob = da.map_overlap(
                lambda x, y: stats.t.sf(np.abs(x), y) * 2,
                t.data, df.data,
                depth=(0,) * t.ndim,
                boundary='none'
            )
            return xr.DataArray(prob, coords=t.coords, dims=t.dims)

        def apply_fdr(p_values, alpha):
            p_sorted = da.sort(p_values.data.ravel())
            m = p_sorted.size
            thresholds = da.arange(1, m + 1) / m * alpha
            significant = p_sorted <= thresholds
            if da.any(significant):
                p_threshold = p_sorted[da.argmax(significant[::-1])]
            else:
                p_threshold = 0
            return p_threshold.compute()

        # Compute p-values for all pairs of datasets
        n_datasets = len(variables)
        combinations = [(i, j) for i in range(n_datasets) for j in range(i + 1, n_datasets)]

        # Precompute means and variances
        means = [var.mean(dim='time') for var in variables]
        variances = [var.var(dim='time') for var in variables]
        counts = [var.count(dim='time') for var in variables]

        p_values = []
        for i, j in combinations:
            p_value = vectorized_ttest(variables[i], variables[j])
            p_values.append(p_value)

        p_values = xr.concat(p_values, dim='combination')

        # Apply FDR
        p_threshold = apply_fdr(p_values, alpha)
        significant_mask = p_values <= p_threshold
        proportion_passed = significant_mask.sum('combination') / len(combinations)

        # Create output dataset
        ds = xr.Dataset({
            'p_values': p_values,
            'significant': significant_mask,
            'proportion_passed': proportion_passed
        })

        # Add metadata
        ds['p_values'].attrs['long_name'] = 'P-values from t-test'
        ds['p_values'].attrs['description'] = 'P-values for each combination of datasets'
        ds['significant'].attrs['long_name'] = 'Significant grid points'
        ds['significant'].attrs['description'] = 'Boolean mask of significant grid points'
        ds['proportion_passed'].attrs['long_name'] = 'Proportion of tests passed FDR'
        ds['proportion_passed'].attrs['description'] = 'Proportion of tests that passed the FDR threshold'
        ds.attrs['FDR_threshold'] = p_threshold
        ds.attrs['alpha_FDR'] = alpha

        return ds

    def stat_ANOVA(self, *variables, n_jobs=-1, analysis_type='twoway'):
        """
        Perform statistical analysis (one-way ANOVA or two-way ANOVA) on multiple variables,
        automatically identifying the dependent variable.

        Args:
           *variables: Variable number of xarray DataArrays. The one with '_Y' in its name is treated as the dependent variable.
           n_jobs (int): Number of jobs for parallel processing. Default is -1 (use all cores)
           analysis_type (str): Type of analysis to perform. 'oneway' for one-way ANOVA, 'twoway' for two-way ANOVA

        Returns:
           xarray.Dataset: Dataset containing results of the analysis (F-statistic and p-values for one-way ANOVA,
                          sum of squares and p-values for two-way ANOVA)
        """
        try:
            if analysis_type == 'twoway':
                import statsmodels.formula.api as smf
                import statsmodels.api as sm
                from scipy.stats import t

            elif analysis_type == 'oneway':
                from scipy.stats import f_oneway
            else:
                raise ValueError("Invalid analysis_type. Choose 'oneway' or 'twoway'")
        except ImportError as e:
            raise ImportError(f"{e.name} is required for this function")
        from joblib import Parallel, delayed

        # Separate dependent and independent variables
        Y_vars = [var for var in variables if '_Y' in var.name]
        X_vars = [var for var in variables if '_Y' not in var.name]

        if len(Y_vars) != 1:
            raise ValueError("Exactly one variable with '_Y' in its name should be provided as the dependent variable.")

        Y_data = Y_vars[0]

        # Align and combine datasets
        combined_data = xr.merge([Y_data.rename('Y_data')] + [var.rename(f'var_{i}') for i, var in enumerate(X_vars)])

        # Prepare data for analysis
        data_array = np.stack([combined_data[var].values for var in combined_data.data_vars if var != 'Y_data'], axis=-1)
        Y_data_array = combined_data['Y_data'].values

        if analysis_type == 'twoway':

            def normalize_data(data):
                """Normalize data to [0, 1] range."""
                return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

            def OLS(data_slice, Y_data_slice):
                """Perform OLS analysis on a single lat-lon point."""
                if np.any(np.isnan(data_slice)) or np.any(np.isnan(Y_data_slice)) or \
                        np.any(np.isinf(data_slice)) or np.any(np.isinf(Y_data_slice)) or \
                        np.any(np.all(data_slice < 1e-10, axis=0)) or np.all(Y_data_slice < 1e-10):
                    return np.full(16, np.nan), np.full(16, np.nan)

                # Normalize data
                norm_data = np.apply_along_axis(normalize_data, 0, data_slice)
                norm_Y_data = normalize_data(Y_data_slice)

                # Create DataFrame
                df = pd.DataFrame(norm_data, columns=[f'var_{i}' for i in range(norm_data.shape[1])])
                df['Y_data'] = norm_Y_data

                # Construct formula dynamically
                var_names = df.columns[:-1]
                main_effects = '+'.join(var_names)
                interactions = '+'.join(f'({a}:{b})' for i, a in enumerate(var_names) for b in var_names[i + 1:])
                formula = f'Y_data ~ {main_effects}+{interactions}'

                # Perform OLS
                model = smf.ols(formula, data=df).fit()
                anova_results = sm.stats.anova_lm(model, typ=2)

                return anova_results['sum_sq'].values, anova_results['PR(>F)'].values

            # Parallel processing
            num_cores = n_jobs if n_jobs > 0 else os.cpu_count()

            results = Parallel(n_jobs=num_cores)(
                delayed(OLS)(data_array[..., i, j, :], Y_data_array[..., i, j])
                for i in range(data_array.shape[-3])
                for j in range(data_array.shape[-2])
            )

            # Reshape results
            n_factors = len(results[0][0])  # Number of factors in ANOVA
            sum_sq = np.array([r[0] for r in results]).reshape(data_array.shape[-3], data_array.shape[-2], -1)
            p_values = np.array([r[1] for r in results]).reshape(data_array.shape[-3], data_array.shape[-2], -1)

            # Create output dataset
            output_ds = xr.Dataset(
                {
                    'sum_sq': (['lat', 'lon', 'factors'], sum_sq),
                    'p_value': (['lat', 'lon', 'factors'], p_values)
                },
                coords={
                    'lat': combined_data.lat,
                    'lon': combined_data.lon,
                    'factors': np.arange(n_factors)
                }
            )

            # Add metadata
            output_ds['sum_sq'].attrs['long_name'] = 'Sum of Squares from ANOVA'
            output_ds['sum_sq'].attrs['description'] = 'Sum of squares for each factor in the ANOVA'
            output_ds['p_value'].attrs['long_name'] = 'P-values from ANOVA'
            output_ds['p_value'].attrs['description'] = 'P-values for each factor in the ANOVA'
            output_ds.attrs['analysis_type'] = 'two-way ANOVA'  # Changed to 'two-way ANOVA'
            output_ds.attrs['n_factors'] = n_factors

        elif analysis_type == 'oneway':
            def oneway_anova(data_slice, Y_data_slice):
                """Perform one-way ANOVA on a single lat-lon point."""
                if np.any(np.isnan(data_slice)) or np.any(np.isnan(Y_data_slice)) or \
                        np.any(np.isinf(data_slice)) or np.any(np.isinf(Y_data_slice)) or \
                        np.any(np.all(data_slice < 1e-10, axis=0)) or np.all(Y_data_slice < 1e-10):
                    return np.nan, np.nan

                # Group Y_data values based on unique values in each independent variable
                groups = [Y_data_slice[data_slice[:, i] == val] for i in range(data_slice.shape[1]) for val in
                          np.unique(data_slice[:, i])]

                # Perform one-way ANOVA
                f_statistic, p_value = f_oneway(*groups)

                return f_statistic, p_value

            # Parallel processing
            num_cores = n_jobs if n_jobs > 0 else os.cpu_count()

            results = Parallel(n_jobs=num_cores)(
                delayed(oneway_anova)(data_array[..., i, j, :], Y_data_array)
                for i in range(data_array.shape[-3])
                for j in range(data_array.shape[-2])
            )

            # Reshape results
            f_statistics = np.array([r[0] for r in results]).reshape(data_array.shape[-3], data_array.shape[-2])
            p_values = np.array([r[1] for r in results]).reshape(data_array.shape[-3], data_array.shape[-2])

            # Create output dataset
            output_ds = xr.Dataset(
                {
                    'F_statistic': (['lat', 'lon'], f_statistics),
                    'p_value': (['lat', 'lon'], p_values)
                },
                coords={
                    'lat': combined_data.lat,
                    'lon': combined_data.lon,
                }
            )

            # Add metadata
            output_ds['F_statistic'].attrs['long_name'] = 'F-statistic from one-way ANOVA'
            output_ds['F_statistic'].attrs['description'] = 'F-statistic for the one-way ANOVA'
            output_ds['p_value'].attrs['long_name'] = 'P-values from one-way ANOVA'

        return output_ds

    def save_result(self, method_name: str, result, data_sources: List[str]) -> xr.Dataset:
        # Remove the existing output directory
        filename_parts = [method_name] + data_sources
        filename = "_".join(filename_parts) + "_output.nc"
        output_file = os.path.join(self.output_dir, f"{method_name}", filename)
        print(f"Saving {method_name} output to {output_file}")
        if isinstance(result, xr.DataArray) or isinstance(result, xr.Dataset):
            if isinstance(result, xr.DataArray):
                result = result.to_dataset(name=f"{method_name}")
            result['lat'].attrs['_FillValue'] = float('nan')
            result['lat'].attrs['standard_name'] = 'latitude'
            result['lat'].attrs['long_name'] = 'latitude'
            result['lat'].attrs['units'] = 'degrees_north'
            result['lat'].attrs['axis'] = 'Y'
            result['lon'].attrs['_FillValue'] = float('nan')
            result['lon'].attrs['standard_name'] = 'longitude'
            result['lon'].attrs['long_name'] = 'longitude'
            result['lon'].attrs['units'] = 'degrees_east'
            result['lon'].attrs['axis'] = 'X'
            result.to_netcdf(output_file)
        else:
            # If the result is not xarray object, we might need to handle it differently
            # For now, let's just print it
            print(f"Result of {method_name}: {result}")


class BasicProcessing(statistics_calculate):
    def __init__(self, info):
        """
        Initialize the Statistics class.

        Args:
            info (dict): A dictionary containing additional information to be added as attributes.
        """
        self.name = 'statistics'
        self.version = '0.2'
        self.release = '0.2'
        self.date = 'Mar 2024'
        self.author = "Zhongwang Wei"
        self.__dict__.update(info)

    def check_file_exist(self, file: str) -> str:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File '{file}' not found.")
        return file

    def select_var(self, VarFile: str, varname: List[str]) -> xr.Dataset:
        if isinstance(varname, str): varname = [varname]
        try:
            try:
                ds = xr.open_dataset(VarFile)  # .squeeze()
            except:
                ds = xr.open_dataset(VarFile, decode_times=False)  # .squeeze()
        except Exception as e:
            logging.error(f"Failed to open dataset: {VarFile}")
            logging.error(f"Error: {str(e)}")
            raise
        ds = ds[varname[0]].astype('float32')
        return ds

    def combine_year(self, year: int, dirx: str, suffix: str, prefix: str, varname: List[str]) -> xr.Dataset:
        try:
            var_files = glob.glob(os.path.join(dirx, f'{prefix}{year}*{suffix}.nc'))
        except:
            var_files = glob.glob(os.path.join(dirx, str(year), f'{prefix}{year}*{suffix}.nc'))
        datasets = []
        for file in var_files:
            file = self.check_file_exist(file)
            ds = self.select_var(file, varname)
            datasets.append(ds)
        data0 = xr.concat(datasets, dim="time").sortby('time')
        return data0

    def check_coordinate(self, ds: xr.Dataset):
        for coord in ds.coords:
            if coord in self.coordinate_map:
                ds = ds.rename({coord: self.coordinate_map[coord]})
        if 'lon' in ds.coords:
            ds['lon'] = (ds['lon'] + 180) % 360 - 180
            # Reindex the dataset
            ds = ds.reindex(lon=sorted(ds.lon.values))
        return ds

    def check_time(self, ds: xr.Dataset, syear: int, eyear: int, time_freq: str) -> xr.Dataset:
        if not [time_freq][0].isdigit():
            time_freq = f'1{time_freq}'
        match = re.match(r'(\d+)\s*([a-zA-Z]+)', time_freq)
        if match:
            num_value, unit = match.groups()
            num_value = int(num_value) if num_value else 1
            unit = self.freq_map.get(unit.lower())
            time_freq = f'{num_value}{unit}'
        else:
            raise ValueError(f"Invalid time resolution format: {time_freq}. Use '3month', '6hr', etc.")

        if 'time' not in ds.coords:
            print("The dataset does not contain a 'time' coordinate.")
            # Based on the syear and eyear, create a time index
            lon = ds.lon.values
            lat = ds.lat.values
            data = ds.values
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=time_freq)
            try:
                ds1 = xr.Dataset({f'{ds.name}': (['time', 'lat', 'lon'], data)},
                                 coords={'time': time_index, 'lat': lat, 'lon': lon})
            except:
                try:
                    ds1 = xr.Dataset({f'{ds.name}': (['time', 'lon', 'lat'], data)},
                                     coords={'time': time_index, 'lon': lon, 'lat': lat})
                except:
                    ds1 = xr.Dataset({f'{ds.name}': (['lat', 'lon', 'time'], data)},
                                     coords={'lat': lat, 'lon': lon, 'time': time_index})
            ds1 = ds1.transpose('time', 'lat', 'lon')
            return ds1[f'{ds.name}'], time_freq

        if not hasattr(ds['time'], 'dt'):
            try:
                ds['time'] = pd.to_datetime(ds['time'])
            except:
                lon = ds.lon.values
                lat = ds.lat.values
                data = ds.values
                time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=time_freq)
                try:
                    ds1 = xr.Dataset({f'{ds.name}': (['time', 'lat', 'lon'], data)},
                                     coords={'time': time_index, 'lat': lat, 'lon': lon})
                except:
                    try:
                        ds1 = xr.Dataset({f'{ds.name}': (['time', 'lon', 'lat'], data)},
                                         coords={'time': time_index, 'lon': lon, 'lat': lat})
                    except:
                        ds1 = xr.Dataset({f'{ds.name}': (['lat', 'lon', 'time'], data)},
                                         coords={'lat': lat, 'lon': lon, 'time': time_index})
                    ds1 = ds1.transpose('time', 'lat', 'lon')
                    return ds1[f'{ds.name}'], time_freq

        # Check for duplicate time values
        if ds['time'].to_index().has_duplicates:
            print("Warning: Duplicate time values found. Removing duplicates...")
            # Remove duplicates by keeping the first occurrence
            _, index = np.unique(ds['time'], return_index=True)
            ds = ds.isel(time=index)

        # Ensure time is sorted
        ds = ds.sortby('time')
        try:
            return ds.transpose('time', 'lat', 'lon')[f'{ds.name}'], time_freq
        except:
            try:
                return ds.transpose('time', 'lat', 'lon'), time_freq
            except:
                try:
                    return ds.transpose('time', 'lon', 'lat'), time_freq
                except:
                    return ds.squeeze(), time_freq

    def check_time_bk(self, ds, time_freq):
        if 'time' not in ds.coords:
            raise ValueError("Time dimension not found in the dataset.")
        # if the time is not in datetime64 format, convert it
        if not np.issubdtype(ds.time.dtype, np.datetime64):
            ds['time'] = pd.to_datetime(ds.time.values)
        if not [time_freq][0].isdigit():
            tim_freq = f'1{time_freq}'
            # print(
            #     f"Warning: the time resolution format of dataset is not correct, set the number as 1, the time resolution is {tim_res}")
        match = re.match(r'(\d+)\s*([a-zA-Z]+)', tim_freq)
        if match:
            num_value, unit = match.groups()
            num_value = int(num_value) if num_value else 1
            unit = self.freq_map.get(unit.lower())
            tim_freq = f'{num_value}{unit}E'
            # print(f"Time resolution is {tim_res}, set the time resolution as {tim_freq}")
        else:
            raise ValueError(f"Invalid time resolution format: {tim_res}. Use '3month', '6hr', etc.")

        # if the time is not in the correct frequency, resample it
        return ds, tim_freq

    def check_dataset_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        """Checks and fills missing time values in an xarray Dataset with specified comparison scales."""
        # Ensure the dataset has a proper time index
        ds, tim_res = self.check_time(ds, syear, eyear, tim_res)
        ds = self.make_time_integrity(ds, syear, eyear, tim_res)
        return ds

    def check_dataset_time_integrity_bk(self, ds, syear, eyear, time_freq):
        try:
            ds['time'] = ds['time'].to_index().to_datetimeindex()
        except:
            pass
        time_index = pd.date_range(start=f'{syear}-01-01', end=f'{eyear}-12-31', freq=time_freq)
        ds = ds.reindex(time=time_index)
        return ds

    def make_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
        if match:
            num_value, time_unit = match.groups()
            num_value = int(num_value) if num_value else 1
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
            if time_unit.lower() in ['m', 'month', 'mon']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-15T00:00:00'))
                ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-15T00:00:00'))
            elif time_unit.lower() in ['d', 'day', '1d', '1day']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT12:00:00'))
                ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT12:00:00'))
            elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT%H:30:00'))
                ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))

            time_var = ds.time
            time_values = time_var
            # Create a complete time series based on the specified time frequency and range
            # Compare the actual time with the complete time series to find the missing time
            # print('Checking time series completeness...')
            missing_times = time_index[~np.isin(time_index, time_values)]
            if len(missing_times) > 0:
                print("Time series is not complete. Missing time values found: ")
                print(missing_times)
                print('Filling missing time values with np.nan')
                # Fill missing time values with np.nan
                ds = ds.reindex(time=time_index)
                ds = ds.where(ds.time.isin(time_values), np.nan)
            else:
                # print('Time series is complete.')
                pass
        return ds

    def select_timerange(self, ds: xr.Dataset, syear: int, eyear: int) -> xr.Dataset:
        if (eyear < syear) or (ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59')) is None):
            print(f"Error: Attempting checking the data time range.")
            exit()
        else:
            return ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59'))

    def remap_data(self, data_list):
        """
        Remap all datasets to the resolution specified in main.nml.
        Tries CDO first, then xESMF, and finally xarray's interp method.
        """

        remapping_methods = [
            self.remap_interpolate,
            self.remap_xesmf,
            self.remap_cdo
        ]

        remapped_data = []
        for i, data in enumerate(data_list):
            data = self.preprocess_grid_data(data)
            new_grid = self.create_target_grid()
            for method in remapping_methods:
                try:
                    remapped = method(data, new_grid)
                except Exception as e:
                    logging.warning(f"{method.__name__} failed: {e}")

            remapped = remapped.resample(time=self.compare_tim_res).mean()
            remapped_data.append(remapped)
        return remapped_data

    def preprocess_grid_data(self, data: xr.Dataset) -> xr.Dataset:
        # Check if lon and lat are 2D
        data = self.check_coordinate(data)
        if data['lon'].ndim == 2 and data['lat'].ndim == 2:
            try:
                data = convert_to_wgs84_xesmf(data, self.compare_grid_res)
            except:
                data = convert_to_wgs84_scipy(data, self.compare_grid_res)

        # Convert longitude values
        lon = data['lon'].values
        lon_adjusted = np.where(lon > 180, lon - 360, lon)

        # Create a new DataArray with adjusted longitude values
        new_lon = xr.DataArray(lon_adjusted, dims='lon', attrs=data['lon'].attrs)

        # Assign the new longitude to the dataset
        data = data.assign_coords(lon=new_lon)

        # If needed, sort the dataset by the new longitude values
        data = data.sortby('lon')

        return data

    def create_target_grid(self) -> xr.Dataset:
        min_lon = self.main_nml['general']['min_lon']
        min_lat = self.main_nml['general']['min_lat']
        max_lon = self.main_nml['general']['max_lon']
        max_lat = self.main_nml['general']['max_lat']
        lon_new = np.arange(min_lon + self.compare_grid_res / 2, max_lon, self.compare_grid_res)
        lat_new = np.arange(min_lat + self.compare_grid_res / 2, max_lat, self.compare_grid_res)
        return xr.Dataset({'lon': lon_new, 'lat': lat_new})

    def remap_interpolate(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.DataArray:
        from regrid import Grid
        min_lon = self.main_nml['general']['min_lon']
        min_lat = self.main_nml['general']['min_lat']
        max_lon = self.main_nml['general']['max_lon']
        max_lat = self.main_nml['general']['max_lat']

        grid = Grid(
            north=max_lat - self.compare_grid_res / 2,
            south=min_lat + self.compare_grid_res / 2,
            west=min_lon + self.compare_grid_res / 2,
            east=max_lon - self.compare_grid_res / 2,
            resolution_lat=self.compare_grid_res,
            resolution_lon=self.compare_grid_res,
        )
        target_dataset = grid.create_regridding_dataset(lat_name="lat", lon_name="lon")
        # Convert sparse arrays to dense arrays
        data_regrid = data.regrid.conservative(target_dataset, nan_threshold=0)
        return data_regrid

    def remap_xesmf(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.DataArray:
        import xesmf as xe
        regridder = xe.Regridder(data, new_grid, 'conservative')
        return regridder(data)

    def remap_cdo(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.DataArray:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.nc') as temp_input, \
                tempfile.NamedTemporaryFile(suffix='.nc') as temp_output, \
                tempfile.NamedTemporaryFile(suffix='.txt') as temp_grid:
            data.to_netcdf(temp_input.name)
            self.create_target_grid_file(temp_grid.name, new_grid)

            cmd = f"cdo -s -remapcon,{temp_grid.name} {temp_input.name} {temp_output.name}"
            subprocess.run(cmd, shell=True, check=False)

            return xr.open_dataset(temp_output.name)
            # return list(xr.open_dataset(temp_output.name).data_vars.values())[0]

    def create_target_grid_file(self, filename: str, new_grid: xr.Dataset) -> None:
        min_lon = self.main_nml['general']['min_lon']
        min_lat = self.main_nml['general']['min_lat']
        with open(filename, 'w') as f:
            f.write(f"gridtype = lonlat\n")
            f.write(f"xsize = {len(new_grid.lon)}\n")
            f.write(f"ysize = {len(new_grid.lat)}\n")
            f.write(f"xfirst = {min_lon + self.compare_grid_res / 2}\n")
            f.write(f"xinc = {self.compare_grid_res}\n")
            f.write(f"yfirst = {min_lat + self.compare_grid_res / 2}\n")
            f.write(f"yinc = {self.compare_grid_res}\n")

    def process_all_methods(self):
        for method, data_sources in self.general_config.items():
            if method.endswith('_data_source'):
                method_name = method.replace('_data_source', '')
                # self.process_method(method_name, data_sources)

    def process_method(self, method_name, data_sources):
        print(f"Processing {method_name}...")
        method_config = self.stats_nml[method_name]

        data_sources = [ds.strip() for ds in data_sources.split(',')]
        processed_data = []

        for source in data_sources:
            data = self.process_data_source(source, method_config)
            processed_data.append(data)

        self.generate_output(method_name, processed_data, method_config)

    def process_data_source(self, source: str, config: Dict[str, Any]) -> xr.Dataset:
        print(source)
        source_config = {k: v for k, v in config.items() if k.startswith(source)}
        dirx = source_config[f'{source}_dir']
        syear = int(source_config[f'{source}_syear'])
        eyear = int(source_config[f'{source}_eyear'])
        time_freq = source_config[f'{source}_tim_res']
        varname = source_config[f'{source}_varname']
        groupby = source_config[f'{source}_data_groupby'].lower()
        suffix = source_config[f'{source}_suffix']
        prefix = source_config[f'{source}_prefix']
        print(f"Processing data source '{source}' from '{dirx}'...")

        if groupby == 'single':
            ds = self.process_single_groupby(dirx, suffix, prefix, varname, syear, eyear, time_freq)
        elif groupby == 'year':
            # ds = self.process_yearly_groupby(dirx, suffix, prefix, varname, syear, eyear, time_freq)
            years = range(syear, eyear + 1)
            ds_list = []
            for year in years:
                ds_list.append(self.process_yearly_groupby(dirx, suffix, prefix, varname, year, year, time_freq, ))
            # ds_list = Parallel(n_jobs=self.num_cores)(
            #     delayed(self.process_yearly_groupby)(dirx, suffix, prefix, varname, year, year, time_freq, )
            #     for year in years
            # )
            ds = xr.concat(ds_list, dim='time')
        else:
            logging.info(f"Combining data to one file...")
            years = range(syear, eyear + 1)
            ds_list = Parallel(n_jobs=self.num_cores)(
                delayed(self.process_other_groupby)(dirx, suffix, prefix, varname, year, year, time_freq)
                for year in years
            )
            ds = xr.concat(ds_list, dim='time')
        return ds

    def process_single_groupby(self, dirx: str, suffix: str, prefix: str, varname: List[str], syear: int, eyear: int,
                               time_freq: str) -> xr.Dataset:
        varfile = self.check_file_exist(os.path.join(dirx, f'{prefix}{suffix}.nc'))
        ds = self.select_var(varfile, varname)
        ds = self.load_and_process_dataset(ds, syear, eyear, time_freq)
        return ds

    def process_yearly_groupby(self, dirx: str, suffix: str, prefix, varname: List[str], syear: int, eyear: int,
                               time_freq: str) -> xr.Dataset:
        varfiles = self.check_file_exist(os.path.join(dirx, f'{prefix}{syear}{suffix}.nc'))
        ds = self.select_var(varfiles, varname)
        ds = self.load_and_process_dataset(ds, syear, eyear, time_freq)
        return ds

    def process_other_groupby(self, dirx: str, suffix: str, prefix: str, varname: List[str], syear: int, eyear: int,
                              time_freq: str) -> xr.Dataset:
        ds = self.combine_year(syear, dirx, suffix, prefix, varname)
        ds = self.load_and_process_dataset(ds, syear, eyear, time_freq)
        return ds

    def load_and_process_dataset(self, ds: xr.Dataset, syear: str, eyear: str, time_freq) -> xr.Dataset:
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, time_freq)
        ds = self.select_timerange(ds, syear, eyear)
        return ds

    def generate_output(self, method_name, data_list, config):
        output_file = os.path.join(self.output_dir, f"{method_name}" f"{method_name}_output.nc")

        if len(data_list) > 1:
            ds = xr.merge(data_list)
        else:
            ds = data_list[0]

        print(f"Saving {method_name} output to {output_file}")
        ds.to_netcdf(output_file)

    def run(self):
        self.process_all_methods()
        print("All statistical data processing completed.")

    def save_result(self, method_name: str, result, data_sources: List[str]) -> xr.Dataset:
        # Remove the existing output directory
        filename_parts = [method_name] + data_sources
        filename = "_".join(filename_parts) + "_output.nc"
        output_file = os.path.join(self.output_dir, f"{method_name}", filename)
        print(f"Saving {method_name} output to {output_file}")
        if isinstance(result, xr.DataArray) or isinstance(result, xr.Dataset):
            if isinstance(result, xr.DataArray):
                result = result.to_dataset(name=f"{method_name}")
            result['lat'].attrs['_FillValue'] = float('nan')
            result['lat'].attrs['standard_name'] = 'latitude'
            result['lat'].attrs['long_name'] = 'latitude'
            result['lat'].attrs['units'] = 'degrees_north'
            result['lat'].attrs['axis'] = 'Y'
            result['lon'].attrs['_FillValue'] = float('nan')
            result['lon'].attrs['standard_name'] = 'longitude'
            result['lon'].attrs['long_name'] = 'longitude'
            result['lon'].attrs['units'] = 'degrees_east'
            result['lon'].attrs['axis'] = 'X'
            result.to_netcdf(output_file)
        else:
            # If the result is not xarray object, we might need to handle it differently
            # For now, let's just print it
            print(f"Result of {method_name}: {result}")

    coordinate_map = {
        'longitude': 'lon', 'long': 'lon', 'lon_cama': 'lon', 'lon0': 'lon', 'x': 'lon',
        'latitude': 'lat', 'lat_cama': 'lat', 'lat0': 'lat', 'y': 'lat',
        'Time': 'time', 'TIME': 'time', 't': 'time', 'T': 'time',
        'elevation': 'elev', 'height': 'elev', 'z': 'elev', 'Z': 'elev',
        'h': 'elev', 'H': 'elev', 'ELEV': 'elev', 'HEIGHT': 'elev',
    }
    freq_map = {
        'month': 'ME',
        'mon': 'ME',
        'monthly': 'ME',
        'day': 'D',
        'daily': 'D',
        'hour': 'H',
        'Hour': 'H',
        'hr': 'H',
        'Hr': 'H',
        'h': 'H',
        'hourly': 'H',
        'year': 'Y',
        'yr': 'Y',
        'yearly': 'Y',
        'week': 'W',
        'wk': 'W',
        'weekly': 'W',
    }


class StatisticsProcessing(BasicProcessing):
    def __init__(self, main_nml, stats_nml, output_dir, num_cores=-1):
        super().__init__(main_nml)
        super().__init__(stats_nml)
        self.name = 'StatisticsDataHandler'
        self.version = '0.3'
        self.release = '0.3'
        self.date = 'June 2024'
        self.author = "Zhongwang Wei"

        self.stats_nml = stats_nml
        self.main_nml = main_nml
        self.general_config = self.stats_nml['general']
        self.output_dir = output_dir
        self.num_cores = num_cores

        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml['general']['compare_grid_res']
        self.compare_tim_res = self.main_nml['general'].get('compare_tim_res', '1').lower()

        # this should be done in read_namelist
        # adjust the time frequency
        match = re.match(r'(\d*)\s*([a-zA-Z]+)', self.compare_tim_res)
        if not match:
            raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")
        value, unit = match.groups()
        if not value:
            value = 1
        else:
            value = int(value)  # Convert the numerical value to an integer
        # Get the corresponding pandas frequency
        freq = self.freq_map.get(unit.lower())
        if not freq:
            raise ValueError(f"Unsupported time unit: {unit}")
        self.compare_tim_res = f'{value}{freq}'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def setup_output_directories(self, statistic_method):
        if os.path.exists(os.path.join(self.output_dir, f"{statistic_method}")):
            shutil.rmtree(os.path.join(self.output_dir, f"{statistic_method}"))
        # Create a new output directory
        if not os.path.exists(os.path.join(self.output_dir, f"{statistic_method}")):
            os.makedirs(os.path.join(self.output_dir, f"{statistic_method}"))

    # Basic statistical methods
    def scenarios_Mann_Kendall_Trend_Test_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [source.strip()]
            self.run_analysis(source.strip(), sources, statistic_method)
            make_Mann_Kendall_Trend_Test(self.output_dir, statistic_method, [source], self.main_nml['general'], statistic_nml,
                                         option)

    def scenarios_Correlation_analysis(self, statistic_method, statistic_nml, option):
        # self.setup_output_directories(statistic_method)

        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}1', f'{source}2']
            # self.run_analysis(source.strip(), sources, statistic_method)
            make_Correlation(self.output_dir, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_Standard_Deviation_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [source.strip()]
            self.run_analysis(source.strip(), sources, statistic_method)
            make_Standard_Deviation(self.output_dir, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_Hellinger_Distance_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)

        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}1', f'{source}2']
            self.run_analysis(source.strip(), sources, statistic_method)
            make_Hellinger_Distance(self.output_dir, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_Z_score_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [source.strip()]
            self.run_analysis(source.strip(), sources, statistic_method)
            make_Z_score(self.output_dir, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_Three_Cornered_Hat_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            nX = int(statistic_nml[f'{source}_nX'])
            if nX < 3:
                print('Error: Three Cornered Hat method must be at least 3 dataset.')
                exit(1)
            sources = [f'{source}{i}' for i in range(1, nX + 1)]
            self.run_analysis(source.strip(), sources, statistic_method)
            # make_Correlation(self.output_dir, statistic_method, [source], statistic_nml, option)

    def scenarios_Partial_Least_Squares_Regression_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)

        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            nX = int(statistic_nml[f'{source}_nX'])
            sources = [f'{source}_Y'] + [f'{source}_X{i + 1}' for i in range(nX)]
            self.run_analysis(source.strip(), sources, statistic_method)
            make_Partial_Least_Squares_Regression(self.output_dir, statistic_method, [source], self.main_nml['general'],
                                                  statistic_nml, option)

    # Advanced statistical methods
    def scenarios_Functional_Response_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            print(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}1', f'{source}2']
            self.run_analysis(source.strip(), sources, statistic_method)
            make_Functional_Response(self.output_dir, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_False_Discovery_Rate_analysis(self, statistic_method, statistic_nml, option):
        return

    def scenarios_ANOVA_analysis(self, statistic_method, statistic_nml, option):
        return

    def run_analysis(self, source: str, sources: List[str], statistic_method):
        method_function = getattr(self, f"stat_{statistic_method.lower()}", None)
        if method_function:
            data_list = [self.process_data_source(isource.strip(), self.stats_nml[statistic_method])
                         for isource in sources]

            if statistic_method == 'Partial_Least_Squares_Regression':
                try:
                    Y_vars = self.process_data_source(sources[0].strip(), self.stats_nml[statistic_method])
                except:
                    raise ValueError(
                        "No dependent variable (Y) found. Ensure at least one variable has '_Y_' in its name.")

            if len(data_list) == 0:
                print(f"Warning: No data sources found for '{statistic_method}'.")
                exit()
            # Remap data
            data_list = self.remap_data(data_list)

            # Call the method with the loaded data
            result = method_function(*data_list)
            self.save_result(statistic_method, result, [source])
        else:
            print(f"Warning: Analysis method '{statistic_method}' not implemented.")
