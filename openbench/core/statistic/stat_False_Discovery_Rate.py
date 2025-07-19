# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

def stat_False_Discovery_Rate(self, *variables):
    """
    Perform optimized False Discovery Rate (FDR) analysis on multiple xarray datasets.

    Args:
        *variables: Variable number of xarray DataArrays to compare
        alpha (float): FDR control level, default is 0.05

    Returns:
        xarray.Dataset: Dataset containing p-values, FDR results, and metadata
    """
    # , alpha = 0.05
    def vectorized_ttest(a, b):
        a_mean = a.mean(dim='time')
        b_mean = b.mean(dim='time')
        a_var = a.var(dim='time')
        b_var = b.var(dim='time')
        a_count = a.count(dim='time')
        b_count = b.count(dim='time')

        # Avoid division by zero
        a_count_safe = da.maximum(a_count, 1)
        b_count_safe = da.maximum(b_count, 1)
        
        # Set counts to NaN where they are actually zero
        a_count_safe = da.where(a_count > 0, a_count_safe, np.nan)
        b_count_safe = da.where(b_count > 0, b_count_safe, np.nan)
        
        t = (a_mean - b_mean) / da.sqrt(a_var / a_count_safe + b_var / b_count_safe)
        df = (a_var / a_count_safe + b_var / b_count_safe) ** 2 / (
                (a_var / a_count_safe) ** 2 / da.maximum(a_count - 1, 1) + (b_var / b_count_safe) ** 2 / da.maximum(b_count - 1, 1)
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