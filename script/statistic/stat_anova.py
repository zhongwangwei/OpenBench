# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import dask.array as da
import logging
import gc
import os

def stat_anova(self, *variables):
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
    # , n_jobs = -1, analysis_type = 'twoway'
    n_jobs = self.stats_nml['ANOVA']['n_jobs']
    analysis_type = self.stats_nml['ANOVA']['analysis_type']

    try:
        if analysis_type == 'twoway':
            import statsmodels.formula.api as smf
            import statsmodels.api as sm
            from scipy.stats import t

        elif analysis_type == 'oneway':
            from scipy.stats import f_oneway
        else:
            logging.error("Invalid analysis_type. Choose 'oneway' or 'twoway'")
            raise ValueError("Invalid analysis_type. Choose 'oneway' or 'twoway'")
    except ImportError as e:
        logging.error(f"{e.name} is required for this function")
        raise ImportError(f"{e.name} is required for this function")
    from joblib import Parallel, delayed
    import gc

    # Separate dependent and independent variables
    Y_vars =variables[0]  # [var for var in variables if '_Y' in var.name]
    X_vars = variables[1:]  # [var for var in variables if '_Y' not in var.name]

    def extract_xarray_data(data):
        """统一提取 xarray.Dataset 或 xarray.DataArray 的数据"""
        if isinstance(data, xr.Dataset):
            varname = next(iter(data.data_vars))
            return data[varname]  # Dataset → 转多变量DataArray再取值
        elif isinstance(data, xr.DataArray):
            return data  # DataArray → 直接取值
        else:
            raise TypeError(f"Unsupported type: {type(data)}. Expected xarray.Dataset or xarray.DataArray")
            # If it's a dataset, apply the test to each data variable

    Y_data = extract_xarray_data(Y_vars)
    # Align and combine datasets
    combined_data = xr.merge([Y_data.rename('Y_data')] + [extract_xarray_data(var).rename(f'var_{i}') for i, var in enumerate(X_vars)])

    # Prepare data for analysis
    data_array = np.stack([combined_data[var].values for var in combined_data.data_vars if var != 'Y_data'], axis=-1)
    Y_data_array = combined_data['Y_data'].values

    # Determine number of cores to use
    num_cores = n_jobs if n_jobs > 0 else os.cpu_count()
    # Limit cores to a reasonable number to avoid memory issues
    num_cores = min(num_cores, os.cpu_count(), 8)

    try:
        if analysis_type == 'twoway':
            def normalize_data(data):
                """Normalize data to [0, 1] range."""
                min_val = np.nanmin(data)
                max_val = np.nanmax(data)
                if max_val == min_val:
                    return np.zeros_like(data)
                return (data - min_val) / (max_val - min_val)

            def OLS(data_slice, Y_data_slice):
                """Perform OLS analysis on a single lat-lon point."""
                # Check for invalid data
                if np.any(np.isnan(data_slice)) or np.any(np.isnan(Y_data_slice)) or \
                        np.any(np.isinf(data_slice)) or np.any(np.isinf(Y_data_slice)) or \
                        np.any(np.all(data_slice < 1e-10, axis=0)) or np.all(Y_data_slice < 1e-10) or \
                        len(Y_data_slice) < data_slice.shape[1] + 2:  # Ensure enough samples for model
                    return np.full(data_slice.shape[1] * 2, np.nan), np.full(data_slice.shape[1] * 2, np.nan)

                try:
                    # Normalize data
                    norm_data = np.apply_along_axis(normalize_data, 0, data_slice)
                    norm_Y_data = normalize_data(Y_data_slice)

                    # Create DataFrame
                    df = pd.DataFrame(norm_data, columns=[f'var_{i}' for i in range(norm_data.shape[1])])
                    df['Y_data'] = norm_Y_data

                    # Construct formula with main effects only
                    var_names = df.columns[:-1]
                    main_effects = '+'.join(var_names)

                    # Add limited interactions - only include first-order interactions
                    # to avoid over-parameterization
                    interactions = ""
                    if len(var_names) > 1:
                        interactions = "+" + '+'.join(f'({a}:{b})'
                                                      for i, a in enumerate(var_names)
                                                      for b in var_names[i + 1:])

                    formula = f'Y_data ~ {main_effects}{interactions}'

                    # Perform OLS
                    model = smf.ols(formula, data=df).fit()
                    anova_results = sm.stats.anova_lm(model, typ=2)

                    return anova_results['sum_sq'].values, anova_results['PR(>F)'].values
                except Exception as e:
                    logging.debug(f"Error in OLS analysis: {e}")
                    n_factors = data_slice.shape[1] * 2
                    return np.full(n_factors, np.nan), np.full(n_factors, np.nan)

            # Parallel processing with chunking to conserve memory
            chunk_size = max(1, data_array.shape[-3] // (num_cores * 2))
            results = []

            for chunk_i in range(0, data_array.shape[-3], chunk_size):
                end_i = min(chunk_i + chunk_size, data_array.shape[-3])
                chunk_results = Parallel(n_jobs=num_cores)(
                    delayed(OLS)(data_array[..., i, j, :], Y_data_array[..., i, j])
                    for i in range(chunk_i, end_i)
                    for j in range(data_array.shape[-2])
                )
                results.extend(chunk_results)
                # Force garbage collection
                gc.collect()

            # Process results
            if not results:
                logging.error("No valid results from ANOVA analysis")
                raise ValueError("No valid results from ANOVA analysis")

            # Determine number of factors from first non-NaN result
            valid_result = next((r for r in results if not np.all(np.isnan(r[0]))), None)
            if valid_result is None:
                logging.error("All ANOVA results are NaN")
                raise ValueError("All ANOVA results are NaN")

            n_factors = len(valid_result[0])

            # Reshape results
            sum_sq = np.array([r[0] if len(r[0]) == n_factors else np.full(n_factors, np.nan)
                               for r in results]).reshape(data_array.shape[-3], data_array.shape[-2], -1)
            p_values = np.array([r[1] if len(r[1]) == n_factors else np.full(n_factors, np.nan)
                                 for r in results]).reshape(data_array.shape[-3], data_array.shape[-2], -1)

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
            output_ds.attrs['analysis_type'] = 'two-way ANOVA'
            output_ds.attrs['n_factors'] = n_factors

        elif analysis_type == 'oneway':
            def oneway_anova(data_slice, Y_data_slice):
                """Perform one-way ANOVA on a single lat-lon point."""
                if np.any(np.isnan(data_slice)) or np.any(np.isnan(Y_data_slice)) or \
                        np.any(np.isinf(data_slice)) or np.any(np.isinf(Y_data_slice)) or \
                        np.any(np.all(data_slice < 1e-10, axis=0)) or np.all(Y_data_slice < 1e-10):
                    return np.nan, np.nan

                try:
                    # More robust grouping approach - discretize continuous variables
                    groups = []
                    for i in range(data_slice.shape[1]):
                        # Use quartiles to discretize the data
                        x = data_slice[:, i]
                        x_valid = x[~np.isnan(x)]
                        if len(x_valid) < 4:  # Not enough data for quartiles
                            continue

                        # Calculate quartiles
                        q1, q2, q3 = np.percentile(x_valid, [25, 50, 75])

                        # Group by quartiles
                        g1 = Y_data_slice[(x <= q1) & ~np.isnan(Y_data_slice)]
                        g2 = Y_data_slice[(x > q1) & (x <= q2) & ~np.isnan(Y_data_slice)]
                        g3 = Y_data_slice[(x > q2) & (x <= q3) & ~np.isnan(Y_data_slice)]
                        g4 = Y_data_slice[(x > q3) & ~np.isnan(Y_data_slice)]

                        # Add non-empty groups
                        for g in [g1, g2, g3, g4]:
                            if len(g) >= 2:  # Need at least 2 samples
                                groups.append(g)

                    if len(groups) < 2:  # Need at least 2 groups for ANOVA
                        return np.nan, np.nan

                    # Perform one-way ANOVA
                    f_statistic, p_value = f_oneway(*groups)
                    return f_statistic, p_value
                except Exception as e:
                    logging.debug(f"Error in one-way ANOVA: {e}")
                    return np.nan, np.nan

            # Parallel processing with chunking to conserve memory
            chunk_size = max(1, data_array.shape[-3] // (num_cores * 2))
            results = []

            for chunk_i in range(0, data_array.shape[-3], chunk_size):
                end_i = min(chunk_i + chunk_size, data_array.shape[-3])
                chunk_results = Parallel(n_jobs=num_cores)(
                    delayed(oneway_anova)(data_array[..., i, j, :], Y_data_array[..., i, j])
                    for i in range(chunk_i, end_i)
                    for j in range(data_array.shape[-2])
                )
                results.extend(chunk_results)
                # Force garbage collection
                gc.collect()

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
            output_ds['p_value'].attrs['description'] = 'P-values for the one-way ANOVA'
            output_ds.attrs['analysis_type'] = 'one-way ANOVA'

        return output_ds

    finally:
        # Clean up memory
        del data_array
        del Y_data_array
        del results
        gc.collect()