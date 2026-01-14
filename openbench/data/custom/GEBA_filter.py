"""Custom filter for the GEBA (Global Energy Balance Archive) reference dataset."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr


def process_site(station_idx, dataset, info, varname):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(dataset['station'].isel(station=station_idx).values)
    lon = float(dataset['lon'].isel(station=station_idx).values)
    lat = float(dataset['lat'].isel(station=station_idx).values)
    
    # Skip stations with missing coordinates
    if pd.isna(lon) or pd.isna(lat):
        return None
    
    # Get time series data for this station and variable
    if varname not in dataset:
        return None
    
    data_var = dataset[varname].isel(station=station_idx)
    
    # Find valid time range (non-missing data)
    valid_mask = ~data_var.isnull()
    if not valid_mask.any():
        return None
    
    valid_times = dataset['time'].where(valid_mask, drop=True)
    start_year = pd.to_datetime(valid_times.values[0]).year
    end_year = pd.to_datetime(valid_times.values[-1]).year

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    # Apply filters: time range, spatial extent
    if ((use_eyear - use_syear) < info.min_year or
            lon < info.min_lon or lon > info.max_lon or
            lat < info.min_lat or lat > info.max_lat):
        return None

    scratch_dir = Path(info.casedir) / "scratch" / f"GEBA_{varname}_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save data
    data_out = data_var.squeeze(drop=True)
    ds_out = xr.Dataset({
        varname: data_out
    })
    ds_out.to_netcdf(file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_GEBA(info, ds=None):
    """Generate required station metadata for GEBA runs or filter dataset.
    
    This function serves two purposes:
    1. When called with only `info`: Generates station metadata for GEBA runs
    2. When called with `info` and `ds`: Acts as a data filter for station processing
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # Get the variable name from info
    varname = getattr(info, 'varname', 'Rn')
    
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        if varname in ds:
            return info, ds[varname]
        else:
            # Return the first data variable
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "GEBA_monthly.nc"

    if not dataset_path.exists():
        logging.error(f"GEBA dataset not found: {dataset_path}")
        return

    with xr.open_dataset(dataset_path) as ds_file:
        station_rows = []
        num_stations = ds_file.dims['station']
        
        for idx in range(num_stations):
            result = process_site(idx, ds_file, info, varname)
            if result:
                station_rows.append(result)

    if not station_rows:
        logging.error(f"No GEBA stations satisfy the selection criteria for {varname}.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_GEBA_{varname}_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info(f'GEBA station list saved: {len(df)} stations for {varname}')
