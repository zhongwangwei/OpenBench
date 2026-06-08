import numpy as np
import xarray as xr


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
    # Fall back to a sane default when the config namespace is missing, in
    # line with the other stat_* modules. Without this, any deployment that
    # didn't migrate the Hellinger_Distance section from compare_nml would
    # crash with a bare KeyError here.
    try:
        nbins = int(self.stats_nml["Hellinger_Distance"]["nbins"])
    except (KeyError, AttributeError, TypeError):
        nbins = 20

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

        # Hellinger distance is between two 1-D distributions; bin v and u
        # against a SHARED edge set so the marginal probability vectors are
        # comparable. The previous implementation built a 2-D joint histogram
        # and summed sqrt(joint), which had no defined statistical meaning
        # and was almost always clamped to 0 by the (1 - bc) <= 0 guard.
        combined_min = float(min(v_valid.min(), u_valid.min()))
        combined_max = float(max(v_valid.max(), u_valid.max()))
        if combined_min == combined_max:
            return 0.0  # both constant and identical
        edges = np.linspace(combined_min, combined_max, nbins + 1)

        p, _ = np.histogram(v_valid, bins=edges)
        q, _ = np.histogram(u_valid, bins=edges)
        p_sum, q_sum = p.sum(), q.sum()
        if p_sum == 0 or q_sum == 0:
            return np.nan
        p = p / p_sum
        q = q / q_sum

        # BC(P,Q) = sum sqrt(P_i * Q_i) ∈ [0, 1]; H = sqrt(1 - BC) ∈ [0, 1]
        bc = float(np.sum(np.sqrt(p * q)))
        return float(np.sqrt(max(0.0, 1.0 - bc)))

    # Rechunk time dimension to single chunk for apply_ufunc with dask
    if hasattr(v, "chunks") and v.chunks is not None:
        v = v.chunk({"time": -1})
    if hasattr(u, "chunks") and u.chunks is not None:
        u = u.chunk({"time": -1})

    # Apply the function to each grid point
    score = xr.apply_ufunc(
        calc_hellinger_distance,
        v,
        u,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Add attributes to the DataArray
    v_name = v.name if v.name is not None else "unknown"
    u_name = u.name if u.name is not None else "unknown"
    score.name = "hellinger_distance_score"
    score.attrs["long_name"] = "Hellinger Distance Score"
    score.attrs["units"] = "-"
    score.attrs["description"] = "Hellinger Distance score calculated between variables " + u_name + " and " + v_name

    # Create a dataset with the score
    ds = xr.Dataset({"hellinger_distance_score": score})
    del score
    # Add global attributes
    ds.attrs["title"] = "Hellinger Distance Score"
    ds.attrs["description"] = "Hellinger Distance score calculated between variables " + u_name + " and " + v_name
    ds.attrs["created_by"] = "ILAMB var_hellinger_distance function"

    return ds
