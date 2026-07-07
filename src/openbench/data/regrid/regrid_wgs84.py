import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def convert_to_wgs84_scipy(ds: xr.Dataset, resolution=0.1) -> xr.Dataset:
    """Regrid curvilinear (2D lat/lon) data to regular WGS84 grid using scipy.

    Handles WRF and other models with 2D coordinate arrays.
    Assumes coordinates have already been renamed to lat/lon/time by check_coordinate().
    """
    from scipy.interpolate import griddata

    min_lon, max_lon = float(ds.lon.min()), float(ds.lon.max())
    min_lat, max_lat = float(ds.lat.min()), float(ds.lat.max())

    new_lon = np.arange(min_lon, max_lon, resolution)
    new_lat = np.arange(min_lat, max_lat, resolution)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)

    orig_lon_flat = ds.lon.values.ravel()
    orig_lat_flat = ds.lat.values.ravel()

    new_data_vars = {}
    for var_name, data in ds.data_vars.items():
        if data.ndim < 2:
            continue
        regridded_data = []
        if "time" in data.dims:
            for time_idx in range(data.time.size):
                orig_values = data.isel(time=time_idx).values
                orig_values_flat = orig_values.ravel()
                mask = np.isfinite(orig_values_flat) & np.isfinite(orig_lon_flat) & np.isfinite(orig_lat_flat)
                if mask.sum() == 0:
                    regridded_data.append(np.full_like(new_lon_2d, np.nan))
                    continue
                regridded = griddata(
                    (orig_lon_flat[mask], orig_lat_flat[mask]),
                    orig_values_flat[mask],
                    (new_lon_2d, new_lat_2d),
                    method="linear",
                    fill_value=np.nan,
                )
                regridded_data.append(regridded)
            new_data_vars[var_name] = (("time", "lat", "lon"), np.array(regridded_data))
        else:
            orig_values_flat = data.values.ravel()
            mask = np.isfinite(orig_values_flat) & np.isfinite(orig_lon_flat) & np.isfinite(orig_lat_flat)
            if mask.sum() > 0:
                regridded = griddata(
                    (orig_lon_flat[mask], orig_lat_flat[mask]),
                    orig_values_flat[mask],
                    (new_lon_2d, new_lat_2d),
                    method="linear",
                    fill_value=np.nan,
                )
                new_data_vars[var_name] = (("lat", "lon"), regridded)

    coords = {"lat": new_lat, "lon": new_lon}
    if "time" in ds.coords:
        coords["time"] = ds.time.values

    new_ds = xr.Dataset(new_data_vars, coords=coords)
    new_ds.lat.attrs.update(
        {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"}
    )
    new_ds.lon.attrs.update(
        {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"}
    )
    return new_ds


def convert_to_wgs84_xesmf(ds: xr.Dataset, resolution=0.1, method: str = "conservative") -> xr.Dataset:
    # Step 2: Create a new regular lon-lat grid (WGS84)
    import xesmf as xe

    min_lon, max_lon = ds.lon.min().item(), ds.lon.max().item()
    min_lat, max_lat = ds.lat.min().item(), ds.lat.max().item()

    new_lon = np.arange(min_lon, max_lon, resolution)
    new_lat = np.arange(min_lat, max_lat, resolution)

    # Create the target grid
    target_grid = xr.Dataset(
        {
            "lat": (["lat"], new_lat),
            "lon": (["lon"], new_lon),
        }
    )

    # Create the regridder
    regridder = xe.Regridder(ds, target_grid, method)

    # Step 3: Perform the regridding
    new_data_vars = {}
    for var_name, data in ds.data_vars.items():
        logger.info("Regridding %s", var_name)
        regridded_data = regridder(data)
        new_data_vars[var_name] = regridded_data

    # Step 4: Create a new dataset with regridded data.
    # Build coords explicitly — putting None as a coord value silently
    # creates a None-valued coordinate that breaks downstream sortby /
    # alignment ops. Only include time when ds actually has it.
    coords = {"lat": new_lat, "lon": new_lon}
    if "time" in ds.coords:
        coords["time"] = ds.time.values
    new_ds = xr.Dataset(new_data_vars, coords=coords)

    # Update attributes for latitude and longitude
    new_ds.lat.attrs.update(
        {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"}
    )

    new_ds.lon.attrs.update(
        {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"}
    )

    return new_ds
