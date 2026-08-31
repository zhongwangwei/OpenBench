# -*- coding: utf-8 -*-
import gc
import logging

import numpy as np
import xarray as xr
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
        method_config = self.stats_nml["Mann_Kendall_Trend_Test"]
    except (AttributeError, KeyError, TypeError):
        method_config = self.compare_nml["Mann_Kendall_Trend_Test"]
    significance_level = method_config["significance_level"]
    max_sen_pairs = int(method_config.get("max_sen_pairs", 2_000_000))
    if max_sen_pairs < 0:
        raise ValueError("Mann_Kendall_Trend_Test max_sen_pairs must be non-negative")

    def _apply_mann_kendall(da, significance_level):
        """
        Applies Mann-Kendall test to a single DataArray using kendalltau.
        """

        def mk_test(x):
            if len(x) < 4 or not np.isfinite(x).any():
                return np.full(7, np.nan)

            valid = np.isfinite(x)
            positions = np.flatnonzero(valid)
            x = x[valid]
            n = len(x)
            if n < 4:
                return np.full(7, np.nan)

            _, ranks = np.unique(x, return_inverse=True)
            tree = np.zeros(ranks.max() + 2, dtype=np.int64)

            def _rank_count(rank):
                count = 0
                while rank > 0:
                    count += tree[rank]
                    rank -= rank & -rank
                return count

            s_stat = 0
            for seen, zero_based_rank in enumerate(ranks):
                rank = int(zero_based_rank) + 1
                less = _rank_count(rank - 1)
                less_or_equal = _rank_count(rank)
                s_stat += less - (seen - less_or_equal)
                update = rank
                while update < tree.size:
                    tree[update] += 1
                    update += update & -update
            s_stat = float(s_stat)

            _, tie_counts = np.unique(x, return_counts=True)
            tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
            var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
            if var_s > 0:
                if s_stat > 0:
                    z_score = (s_stat - 1) / np.sqrt(var_s)
                elif s_stat < 0:
                    z_score = (s_stat + 1) / np.sqrt(var_s)
                else:
                    z_score = 0.0
                p_value = float(2 * stats.norm.sf(abs(z_score)))
            else:
                z_score = np.nan
                p_value = np.nan

            pair_count = n * (n - 1) // 2
            if pair_count <= max_sen_pairs:
                pair_slopes = np.concatenate(
                    [
                        (x[index + 1 :] - x[index]) / (positions[index + 1 :] - positions[index])
                        for index in range(n - 1)
                    ]
                )
                sen_slope = float(np.median(pair_slopes))
            else:
                sen_slope = np.nan
            tau, _ = stats.kendalltau(positions, x, method="auto")

            trend = np.sign(s_stat)
            significance = np.nan if np.isnan(p_value) else float(p_value < significance_level)

            return np.array([trend, significance, p_value, tau, s_stat, z_score, sen_slope])

        try:
            # Rechunk time dimension to single chunk for apply_ufunc with dask
            if hasattr(da, "chunks") and da.chunks is not None:
                da = da.chunk({"time": -1})

            # Apply the test to each grid point with chunking
            result = xr.apply_ufunc(
                mk_test,
                da,
                input_core_dims=[["time"]],
                output_core_dims=[["mk_params"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"mk_params": 7}},
            )

            # Create separate variables for each component
            trend = result.isel(mk_params=0)
            significance = result.isel(mk_params=1)
            p_value = result.isel(mk_params=2)
            tau = result.isel(mk_params=3)
            s_statistic = result.isel(mk_params=4)
            z_score = result.isel(mk_params=5)
            sen_slope = result.isel(mk_params=6)

            # Create a new Dataset with separate variables
            ds = xr.Dataset(
                {
                    "trend": trend,
                    "significance": significance,
                    "p_value": p_value,
                    "tau": tau,
                    "s_statistic": s_statistic,
                    "z_score": z_score,
                    "sen_slope": sen_slope,
                }
            )

            # Add attributes
            ds.trend.attrs["long_name"] = "Mann-Kendall trend"
            ds.trend.attrs["description"] = "Trend direction: 1 (increasing), -1 (decreasing), 0 (no trend)"
            ds.significance.attrs["long_name"] = "Trend significance"
            ds.significance.attrs["description"] = (
                f"True if trend is significant at {significance_level} level, False otherwise"
            )
            ds.p_value.attrs["long_name"] = "p-value"
            ds.p_value.attrs["description"] = "p-value of the Mann-Kendall trend test"
            ds.tau.attrs["long_name"] = "Kendall's tau statistic"
            ds.tau.attrs["description"] = "Kendall's tau correlation coefficient"
            ds.s_statistic.attrs["long_name"] = "Mann-Kendall S statistic"
            ds.s_statistic.attrs["description"] = "Sum of signs over all forward time pairs"
            ds.z_score.attrs["long_name"] = "Mann-Kendall Z statistic"
            ds.z_score.attrs["description"] = "Tie-corrected normal Z statistic for the Mann-Kendall S statistic"
            ds.sen_slope.attrs["long_name"] = "Sen's slope"
            ds.sen_slope.attrs["description"] = "Median of all pairwise slopes per time step"

            ds.attrs["statistical_test"] = "Mann-Kendall trend test with Sen slope"
            ds.attrs["significance_level"] = significance_level
            ds.attrs["sen_slope_max_exact_pairs"] = max_sen_pairs

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
            return xr.concat(results, dim="variable")
        elif isinstance(data, xr.DataArray):
            # If it's a DataArray, apply the test directly
            return _apply_mann_kendall(data, significance_level)
        else:
            logging.error("Input must be an xarray Dataset or DataArray")
            raise TypeError("Input must be an xarray Dataset or DataArray")

    finally:
        # Clean up any remaining objects
        gc.collect()
