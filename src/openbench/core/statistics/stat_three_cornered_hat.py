# -*- coding: utf-8 -*-
import os

import numpy as np
import xarray as xr

_TCH_NEGATIVE_VARIANCE_RTOL = 1e-10
_TCH_MEAN_ABS_EPS = 1e-10


def _tch_uncertainty_from_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-source uncertainty with the classical TCH equations.

    The classical Three-Cornered Hat method estimates individual error
    variances from variances of pairwise differences.  For exactly three
    sources this is the Gray-Allan closed form; for more than three sources we
    solve the standard overdetermined N-cornered system in least squares form.
    Correlated-error GTCH minimization is intentionally not claimed here.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("TCH samples must be a 2-D array shaped (time, source)")

    _, n_sources = arr.shape
    if n_sources < 3:
        return np.full(n_sources, np.nan), np.full(n_sources, np.nan)

    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 3:
        return np.full(n_sources, np.nan), np.full(n_sources, np.nan)

    pair_rows: list[np.ndarray] = []
    pair_variances: list[float] = []
    pair_lookup: dict[tuple[int, int], float] = {}
    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            variance = float(np.var(arr[:, i] - arr[:, j], ddof=1))
            if not np.isfinite(variance):
                return np.full(n_sources, np.nan), np.full(n_sources, np.nan)
            row = np.zeros(n_sources)
            row[i] = 1.0
            row[j] = 1.0
            pair_rows.append(row)
            pair_variances.append(variance)
            pair_lookup[(i, j)] = variance

    if n_sources == 3:
        variances = np.array(
            [
                0.5 * (pair_lookup[(0, 1)] + pair_lookup[(0, 2)] - pair_lookup[(1, 2)]),
                0.5 * (pair_lookup[(0, 1)] + pair_lookup[(1, 2)] - pair_lookup[(0, 2)]),
                0.5 * (pair_lookup[(0, 2)] + pair_lookup[(1, 2)] - pair_lookup[(0, 1)]),
            ],
            dtype=float,
        )
    else:
        design = np.vstack(pair_rows)
        rhs = np.asarray(pair_variances, dtype=float)
        variances, *_ = np.linalg.lstsq(design, rhs, rcond=None)

    max_pair_variance = max(pair_variances) if pair_variances else 0.0
    tolerance = max(1e-12, max_pair_variance * _TCH_NEGATIVE_VARIANCE_RTOL)
    if np.any(~np.isfinite(variances)) or np.any(variances < -tolerance):
        return np.full(n_sources, np.nan), np.full(n_sources, np.nan)

    variances = np.where(variances < 0, 0.0, variances)
    uncertainty = np.sqrt(variances)

    mean_abs = np.mean(np.abs(arr), axis=0)
    relative_denominator = np.where(mean_abs < _TCH_MEAN_ABS_EPS, np.nan, mean_abs)
    relative_uncertainty = uncertainty / relative_denominator * 100.0
    if np.isnan(relative_denominator).any():
        relative_uncertainty = np.full(n_sources, np.nan)
    return uncertainty, relative_uncertainty


def stat_three_cornered_hat(self, *variables):
    """
    Calculate uncertainty using the Three-Cornered Hat method.

    Args:
        *variables: Variable number of xarray DataArrays to compare.
                   Requires at least 3 variables for the method to work.

    Returns:
        xarray.Dataset: Dataset containing uncertainty and relative uncertainty
    """
    import gc

    # Check if we have enough variables
    if len(variables) < 3:
        raise ValueError("Three-Cornered Hat method requires at least 3 variables")

    def cal_uct(arr):
        """Calculate uncertainty using Three-Cornered Hat method for one grid point."""
        try:
            return _tch_uncertainty_from_samples(arr)
        except Exception:
            source_count = arr.shape[1] if getattr(arr, "ndim", 0) == 2 else len(variables)
            return np.full(source_count, np.nan), np.full(source_count, np.nan)

    try:
        # Extract data from each dataset if it's a Dataset
        data_arrays = []
        for var in variables:
            if isinstance(var, xr.Dataset):
                # Get the first data variable from the dataset
                var_name = list(var.data_vars)[0]
                data_arrays.append(var[var_name])
            else:
                # Already a DataArray
                data_arrays.append(var)

        # Combine all variables into a single array
        combined_data = xr.concat(data_arrays, dim="variable")
        # Get dimensions for processing
        lats = combined_data.lat.values
        lons = combined_data.lon.values
        num_variables = len(data_arrays)

        # Initialize output arrays with NaN values - WITHOUT time dimension
        empty_template = xr.DataArray(
            np.full((num_variables, len(lats), len(lons)), np.nan),
            dims=("variable", "lat", "lon"),
            coords={"variable": range(num_variables), "lat": lats, "lon": lons},
        )
        uct = empty_template.copy()
        r_uct = empty_template.copy()

        # Process in chunks to manage memory better
        # Use joblib to parallelize if data is large enough
        if len(lats) * len(lons) > 100:  # Arbitrary threshold for parallelization
            from joblib import Parallel, delayed

            def process_chunk(lat_chunk):
                """Process a chunk of latitudes in parallel."""
                chunk_results = []
                for lat in lat_chunk:
                    for lon in lons:
                        # Extract numerical values only, ensure it's a proper numpy array
                        arr = combined_data.sel(lat=lat, lon=lon).values
                        if not isinstance(arr, np.ndarray) or arr.size == 0 or np.isnan(arr).all():
                            continue

                        # For Three-Cornered Hat method, we need to transpose the data
                        # so that time is the first dimension and variables are the second
                        arr = arr.T  # shape: (time, variables)
                        uct_values, r_uct_values = cal_uct(arr)
                        chunk_results.append((lat, lon, uct_values, r_uct_values))
                return chunk_results

            # Split lats into chunks for parallel processing
            n_jobs = min(os.cpu_count(), 8)  # Limit to avoid excessive memory use
            chunk_size = max(1, len(lats) // (n_jobs * 2))
            lat_chunks = [lats[i : i + chunk_size] for i in range(0, len(lats), chunk_size)]

            # Process chunks in parallel
            all_results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in lat_chunks)

            # Combine results
            for chunk_results in all_results:
                for lat, lon, uct_values, r_uct_values in chunk_results:
                    # Use .loc indexing to set values
                    lat_idx = int(np.argmin(np.abs(lats - lat)))
                    lon_idx = int(np.argmin(np.abs(lons - lon)))
                    uct.values[:, lat_idx, lon_idx] = uct_values
                    r_uct.values[:, lat_idx, lon_idx] = r_uct_values

            # Clean up
            del all_results
            gc.collect()
        else:
            # For smaller datasets, use simple loop
            for lat_iter_idx, lat in enumerate(lats):
                for lon_iter_idx, lon in enumerate(lons):
                    arr = combined_data.sel(lat=lat, lon=lon).values
                    if isinstance(arr, np.ndarray) and arr.size > 0 and not np.isnan(arr).all():
                        # For Three-Cornered Hat method, transpose the data
                        arr = arr.T  # shape: (time, variables)

                        uct_values, r_uct_values = cal_uct(arr)

                        # Use loop indices directly; the previous
                        # `np.where(lats == lat)[0][0]` pattern relied on
                        # exact float equality and grabbed only the first
                        # match when lats contained repeated values.
                        lat_idx = lat_iter_idx
                        lon_idx = lon_iter_idx
                        uct.values[:, lat_idx, lon_idx] = uct_values
                        r_uct.values[:, lat_idx, lon_idx] = r_uct_values

                # Periodically collect garbage to manage memory. Trigger by
                # iteration index so sub-degree grids (lat values like
                # -89.75, -89.25) actually hit this path — the previous
                # `lat % 10 == 0` test was never true on fractional grids.
                if lat_iter_idx % 32 == 0:
                    gc.collect()

        # Create output dataset
        ds = xr.Dataset({"uncertainty": uct, "relative_uncertainty": r_uct})

        # Add metadata
        ds["uncertainty"].attrs["long_name"] = "Uncertainty from Three-Cornered Hat method"
        ds["uncertainty"].attrs["units"] = "Same as input variables"
        ds["uncertainty"].attrs["description"] = "Absolute uncertainty estimated using the Three-Cornered Hat method"
        ds["relative_uncertainty"].attrs["long_name"] = "Relative uncertainty from Three-Cornered Hat method"
        ds["relative_uncertainty"].attrs["units"] = "%"
        ds["relative_uncertainty"].attrs["description"] = (
            "Relative uncertainty (%) estimated using the Three-Cornered Hat method"
        )
        ds.attrs["method"] = "Three-Cornered Hat"
        ds.attrs["n_datasets"] = len(variables)

        return ds

    finally:
        # Clean up memory
        gc.collect()
