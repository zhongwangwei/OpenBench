# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import dask.array as da


def stat_three_cornered_hat(self, *variables):
    """
    Calculate uncertainty using the Three-Cornered Hat method.

    Args:
        *variables: Variable number of xarray DataArrays to compare.
                   Requires at least 3 variables for the method to work.

    Returns:
        xarray.Dataset: Dataset containing uncertainty and relative uncertainty
    """
    try:
        from scipy import optimize
        import gc
    except ImportError as e:
        logging.error(f"Required package not found: {e}")
        raise ImportError(f"Required package not found: {e}")

    # Check if we have enough variables
    if len(variables) < 3:
        raise ValueError("Three-Cornered Hat method requires at least 3 variables")

    def cal_uct(arr):
        """Calculate uncertainty using Three-Cornered Hat method for one grid point."""
        try:
            # Check if we have enough valid data
            if np.isnan(arr).any() or arr.shape[0] < 3 or arr.shape[1] < 3:
                return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

            def my_fun(r):
                """Objective function for optimization."""
                try:
                    S = np.cov(arr.T)
                    f = np.sum(r[:-1] ** 2)
                    for j in range(len(S)):
                        for k in range(j + 1, len(S)):
                            f += (S[j, k] - r[-1] + r[j] + r[k]) ** 2
                    K = np.linalg.det(S)
                    # Avoid division by zero or very small determinants
                    if abs(K) < 1e-10:
                        return np.inf
                    F = f / (K ** (2 * len(S)))
                    return F
                except Exception as e:
                    logging.debug(f"Error in objective function: {e}")
                    return np.inf

            S = np.cov(arr.T)
            # Check if covariance matrix is valid
            if np.isnan(S).any() or np.isinf(S).any():
                return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

            det_S = np.linalg.det(S)
            # Check if matrix is singular or nearly singular
            if abs(det_S) < 1e-10:
                return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

            N = arr.shape[1]
            u = np.ones((1, N - 1))
            R = np.zeros((N, N))

            try:
                inv_S = np.linalg.inv(S)
                inv_S_sub = inv_S[:N - 1, :N - 1]  # Submatrix for calculations involving u
                # Use inv_S_sub for dot product with u
                R[N - 1, N - 1] = 1 / (2 * np.dot(np.dot(u, inv_S_sub), u.T))
            except np.linalg.LinAlgError:
                print(f"DEBUG: cal_uct returning NaN - LinAlgError during initial R calculation")
                return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

            x0 = R[:, N - 1]
            Denominator = det_S ** (2 / len(S))

            # Set up constraint
            # Use inv_S_sub in the constraint lambda function as well
            cons = {'type': 'ineq', 'fun': lambda r: (r[-1] - np.dot(
                np.dot(r[:-1] - r[-1] * u, inv_S_sub),
                (r[:-1] - r[-1] * u).T)) / Denominator}

            # Perform optimization with error handling
            try:
                x = optimize.minimize(my_fun, x0, method='COBYLA', tol=2e-10, constraints=cons)
                if not x.success:
                    return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

                R[:, N - 1] = x.x
                for i in range(N - 1):
                    for j in range(i, N - 1):
                        R[i, j] = S[i, j] - R[N - 1, N - 1] + R[i, N - 1] + R[j, N - 1]
                R += R.T - np.diag(R.diagonal())

                diag_R = np.diag(R)
                # Check if R has negative values on diagonal (invalid results)
                if np.any(diag_R < 0):
                    return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

                uct = np.sqrt(diag_R)  # Use pre-calculated diagonal

                # Safely calculate relative uncertainty
                mean_abs = np.mean(np.abs(arr), axis=0)
                # Avoid division by zero
                mean_abs_safe = np.where(mean_abs < 1e-10, np.nan, mean_abs)

                if np.isnan(mean_abs_safe).any():
                    print(f"DEBUG: cal_uct returning NaN - mean_abs is NaN (near zero: {mean_abs})")
                    return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

                r_uct = uct / mean_abs_safe * 100

                return uct, r_uct
            except Exception as e:
                # Optionally re-raise or log traceback here for more detail
                import traceback
                traceback.print_exc()
                return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)
        except Exception as e:
            print(f"DEBUG: cal_uct returning NaN due to outer exception: {e}")
            return np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)

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
        combined_data = xr.concat(data_arrays, dim='variable')
        # save the combined_data
        combined_data.to_netcdf('combined_data.nc')
        # Get dimensions for processing
        lats = combined_data.lat.values
        lons = combined_data.lon.values
        num_variables = len(data_arrays)

        # Initialize output arrays with NaN values - WITHOUT time dimension
        empty_template = xr.DataArray(
            np.full((num_variables, len(lats), len(lons)), np.nan),
            dims=('variable', 'lat', 'lon'),
            coords={'variable': range(num_variables), 'lat': lats, 'lon': lons}
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
            lat_chunks = [lats[i:i + chunk_size] for i in range(0, len(lats), chunk_size)]

            # Process chunks in parallel
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(process_chunk)(chunk) for chunk in lat_chunks
            )

            # Combine results
            for chunk_results in all_results:
                for lat, lon, uct_values, r_uct_values in chunk_results:
                    # Use .loc indexing to set values
                    lat_idx = np.where(lats == lat)[0][0]
                    lon_idx = np.where(lons == lon)[0][0]
                    uct.values[:, lat_idx, lon_idx] = uct_values
                    r_uct.values[:, lat_idx, lon_idx] = r_uct_values

            # Clean up
            del all_results
            gc.collect()
        else:
            # For smaller datasets, use simple loop
            for lat in lats:
                for lon in lons:
                    arr = combined_data.sel(lat=lat, lon=lon).values
                    if isinstance(arr, np.ndarray) and arr.size > 0 and not np.isnan(arr).all():
                        # For Three-Cornered Hat method, transpose the data
                        arr = arr.T  # shape: (time, variables)

                        uct_values, r_uct_values = cal_uct(arr)

                        # Use direct indexing
                        lat_idx = np.where(lats == lat)[0][0]
                        lon_idx = np.where(lons == lon)[0][0]
                        uct.values[:, lat_idx, lon_idx] = uct_values
                        r_uct.values[:, lat_idx, lon_idx] = r_uct_values

                # Periodically collect garbage to manage memory
                if lat % 10 == 0:
                    gc.collect()

        # Create output dataset
        ds = xr.Dataset({
            'uncertainty': uct,
            'relative_uncertainty': r_uct
        })

        # Add metadata
        ds['uncertainty'].attrs['long_name'] = 'Uncertainty from Three-Cornered Hat method'
        ds['uncertainty'].attrs['units'] = 'Same as input variables'
        ds['uncertainty'].attrs['description'] = 'Absolute uncertainty estimated using the Three-Cornered Hat method'
        ds['relative_uncertainty'].attrs['long_name'] = 'Relative uncertainty from Three-Cornered Hat method'
        ds['relative_uncertainty'].attrs['units'] = '%'
        ds['relative_uncertainty'].attrs['description'] = 'Relative uncertainty (%) estimated using the Three-Cornered Hat method'
        ds.attrs['method'] = 'Three-Cornered Hat'
        ds.attrs['n_datasets'] = len(variables)

        return ds

    finally:
        # Clean up memory
        gc.collect()