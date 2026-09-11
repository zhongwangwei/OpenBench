import xarray as xr


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

    if "time" not in data.dims:
        raise ValueError("Input data must have a 'time' dimension")

    mean = data.mean(dim="time", skipna=True)
    std = data.std(dim="time", skipna=True)

    # Handle zero or near-zero standard deviation to avoid division by zero
    # Create a mask where std is too small (effectively zero)
    std_mask = std < 1e-10

    z_score = (data - mean) / std

    if std_mask.any():
        z_score = z_score.where(~std_mask)

    if hasattr(data, "name") and data.name is not None:
        z_score.name = f"{data.name}_zscore"
    else:
        z_score.name = "zscore"

    z_score.attrs.update(data.attrs)
    z_score.attrs["long_name"] = "Z-score (standardized anomaly)"
    z_score.attrs["description"] = "Standardized anomaly: (data - mean) / standard deviation"
    z_score.attrs["units"] = "unitless"  # Z-scores are dimensionless

    return z_score
