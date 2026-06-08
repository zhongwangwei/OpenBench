# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from scipy import stats


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
        a, b = xr.align(a, b, join="inner")
        valid = np.isfinite(a) & np.isfinite(b)
        a = a.where(valid)
        b = b.where(valid)

        a_mean = a.mean(dim="time")
        b_mean = b.mean(dim="time")
        # Welch's t-test uses unbiased sample variances.  xarray defaults to
        # ddof=0 (population variance), which makes |t| too large while still
        # feeding the result into the Welch-Satterthwaite sample-df formula.
        a_var = a.var(dim="time", ddof=1)
        b_var = b.var(dim="time", ddof=1)
        a_count = a.count(dim="time")
        b_count = b.count(dim="time")

        enough_samples = (a_count > 1) & (b_count > 1)
        a_count_safe = xr.where(enough_samples, a_count, 1)
        b_count_safe = xr.where(enough_samples, b_count, 1)

        variance_term = a_var / a_count_safe + b_var / b_count_safe
        positive_variance = variance_term > 0
        variance_term_safe = xr.where(positive_variance, variance_term, 1.0)
        t_raw = (a_mean - b_mean) / np.sqrt(variance_term_safe)

        df_denominator = (a_var / a_count_safe) ** 2 / xr.where(a_count > 1, a_count - 1, 1) + (
            b_var / b_count_safe
        ) ** 2 / xr.where(b_count > 1, b_count - 1, 1)
        positive_df_denominator = df_denominator > 0
        df_denominator_safe = xr.where(positive_df_denominator, df_denominator, 1.0)
        # Use the same `_safe` variance term in the df numerator that was used
        # in the t denominator; mixing the raw and safe forms left a 0/safe
        # branch that still emitted numpy RuntimeWarnings under dask.
        df_raw = variance_term_safe**2 / df_denominator_safe

        valid_ttest = enough_samples & positive_variance & positive_df_denominator
        t = xr.where(valid_ttest, t_raw, np.nan)
        df = xr.where(valid_ttest, df_raw, np.nan)

        prob = xr.apply_ufunc(
            lambda t_values, df_values: stats.t.sf(np.abs(t_values), df_values) * 2,
            t,
            df,
            dask="parallelized",
            output_dtypes=[float],
        )
        return prob

    def apply_fdr(p_values, alpha):
        flat = p_values.data.ravel()
        if hasattr(flat, "compute"):
            flat = flat.compute()
        p_sorted = np.sort(np.asarray(flat).ravel())
        p_sorted = p_sorted[np.isfinite(p_sorted)]
        m = p_sorted.size
        if m == 0:
            return 0.0

        thresholds = np.arange(1, m + 1) / m * alpha
        significant_indices = np.nonzero(p_sorted <= thresholds)[0]
        if significant_indices.size == 0:
            return 0.0
        return float(p_sorted[significant_indices[-1]])

    # FDR control level: read from stats namelist or fall back to 0.05.
    # Previously `alpha` was undefined at the call site (NameError), so
    # the analysis would have crashed before the dispatcher's "not yet
    # implemented" guard kicked in.
    try:
        alpha = float(self.stats_nml["False_Discovery_Rate"].get("alpha", 0.05))
    except (AttributeError, KeyError, TypeError, ValueError):
        alpha = 0.05

    # Compute p-values for all pairs of datasets
    n_datasets = len(variables)
    combinations = [(i, j) for i in range(n_datasets) for j in range(i + 1, n_datasets)]

    p_values = []
    for i, j in combinations:
        p_value = vectorized_ttest(variables[i], variables[j])
        p_values.append(p_value)

    p_values = xr.concat(p_values, dim="combination")

    # Apply FDR
    p_threshold = apply_fdr(p_values, alpha)
    significant_mask = p_values <= p_threshold
    proportion_passed = significant_mask.sum("combination") / len(combinations)

    # Create output dataset
    ds = xr.Dataset({"p_values": p_values, "significant": significant_mask, "proportion_passed": proportion_passed})

    # Add metadata
    ds["p_values"].attrs["long_name"] = "P-values from t-test"
    ds["p_values"].attrs["description"] = "P-values for each combination of datasets"
    ds["significant"].attrs["long_name"] = "Significant grid points"
    ds["significant"].attrs["description"] = "Boolean mask of significant grid points"
    ds["proportion_passed"].attrs["long_name"] = "Proportion of tests passed FDR"
    ds["proportion_passed"].attrs["description"] = "Proportion of tests that passed the FDR threshold"
    ds.attrs["FDR_threshold"] = p_threshold
    ds.attrs["alpha_FDR"] = alpha

    return ds
