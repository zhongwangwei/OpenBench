import xarray as xr
import numpy as np

def convert_to_wgs84_scipy(ds: xr.Dataset, resolution=0.1) -> xr.Dataset:
    from scipy.interpolate import griddata
    # Step 2: Create a new regular lon-lat grid (WGS84)
    # Define the bounds and resolution of your new grid
    min_lon, max_lon = ds.longitude.min().item(), ds.longitude.max().item()
    min_lat, max_lat = ds.latitude.min().item(), ds.latitude.max().item()

    new_lon = np.arange(min_lon, max_lon, resolution)
    new_lat = np.arange(min_lat, max_lat, resolution)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)

    # Step 3: Perform the regridding
    # Flatten the original 2D lon/lat arrays
    orig_lon_flat = ds.longitude.values.ravel()
    orig_lat_flat = ds.latitude.values.ravel()

    new_data_vars = {}
    for var_name, data in ds.data_vars.items():
        regridded_data = []
        for time_idx in range(data.XTIME.size):
            orig_values = data.isel(XTIME=time_idx).values
            
            # Flatten orig_values and create a mask for valid (non-NaN) values
            orig_values_flat = orig_values.ravel()
            mask = np.isfinite(orig_values_flat)
            
            regridded = griddata(
                (orig_lon_flat[mask], orig_lat_flat[mask]),
                orig_values_flat[mask],
                (new_lon_2d, new_lat_2d),
                method='linear',  # You can change this to 'nearest' or 'cubic'
                fill_value=np.nan
            )
            
            regridded_data.append(regridded)
        
        new_data_vars[var_name] = (('time', 'lat', 'lon'), np.array(regridded_data))

    # Step 4: Create a new dataset with regridded data
    new_ds = xr.Dataset(
        new_data_vars,
        coords={
            'time': ds.XTIME.values,
            'lat': new_lat,
            'lon': new_lon
        }
    )

    # Update attributes for latitude and longitude
    new_ds.lat.attrs.update({
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north',
        'axis': 'Y'
    })

    new_ds.lon.attrs.update({
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east',
        'axis': 'X'
    })

    return new_ds

def convert_to_wgs84_xesmf(ds: xr.Dataset, resolution=0.1) -> xr.Dataset:
    # Step 2: Create a new regular lon-lat grid (WGS84)
    import xesmf as xe
    min_lon, max_lon = ds.lon.min().item(), ds.lon.max().item()
    min_lat, max_lat = ds.lat.min().item(), ds.lat.max().item()

    new_lon = np.arange(min_lon, max_lon, resolution)
    new_lat = np.arange(min_lat, max_lat, resolution)

    # Create the target grid
    target_grid = xr.Dataset({
        'lat': (['lat'], new_lat),
        'lon': (['lon'], new_lon),
    })

    # Create the regridder
    regridder = xe.Regridder(ds, target_grid, 'bilinear')

    # Step 3: Perform the regridding
    new_data_vars = {}
    for var_name, data in ds.data_vars.items():
        print(f"Regridding {var_name}")
        if 'time' in data.dims:
            regridded_data = regridder(data)
            new_data_vars[var_name] = regridded_data
        else:
            # Handle 2D variables without time dimension
            regridded_data = regridder(data)
            new_data_vars[var_name] = regridded_data

    # Step 4: Create a new dataset with regridded data
    new_ds = xr.Dataset(
        new_data_vars,
        coords={
            'time': ds.time.values if 'time' in ds.coords else None,
            'lat': new_lat,
            'lon': new_lon
        }
    )

    # Update attributes for latitude and longitude
    new_ds.lat.attrs.update({
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north',
        'axis': 'Y'
    })

    new_ds.lon.attrs.update({
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east',
        'axis': 'X'
    })

    return new_ds
