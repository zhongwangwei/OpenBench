# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import gc
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

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
        logging.error("scikit-learn is required for this function")
        raise ImportError("scikit-learn is required for this function")
    from scipy.stats import t

    # Prepare Dependent and Independent data
    max_components = self.stats_nml['Partial_Least_Squares_Regression']['max_components']
    n_splits = self.stats_nml['Partial_Least_Squares_Regression']['n_splits']
    n_jobs = self.stats_nml['Partial_Least_Squares_Regression']['n_jobs']

    Y_vars = variables[0]  # [var for var in variables if '_Y' in var.name]
    X_vars = list(variables[1:])  # [var for var in variables if '_Y' not in var.name]

    def extract_xarray_data(data):
        """统一提取 xarray.Dataset 或 xarray.DataArray 的数据"""
        if isinstance(data, xr.Dataset):
            return data.to_array().squeeze("variable").values  # Dataset → 转多变量DataArray再取值
        elif isinstance(data, xr.DataArray):
            return data.values  # DataArray → 直接取值
        else:
            raise TypeError(f"Unsupported type: {type(data)}. Expected xarray.Dataset or xarray.DataArray")

    # Prepare data
    Y_data = extract_xarray_data(Y_vars)
    X_data = np.concatenate([extract_xarray_data(x)[np.newaxis, ...] for x in X_vars], axis=0)
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