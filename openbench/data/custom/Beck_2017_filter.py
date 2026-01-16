"""
Beck_2017 Custom Filter for OpenBench

This filter processes the consolidated Beck_2017_daily.nc file
and extracts individual station time series for benchmarking.

Features:
- Parallel processing using joblib for efficient extraction
- Pre-loads data into memory for faster access
- Filters by time range, spatial extent, and drainage area
"""

import os
import logging
import xarray as xr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def process_site(station_idx, station_ids, lons, lats, areas, streamflow_data, times, info, scratch_dir):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(station_ids[station_idx])
    lon = float(lons[station_idx])
    lat = float(lats[station_idx])
    area = float(areas[station_idx])
    
    # Skip if invalid coordinates
    if np.isnan(lon) or np.isnan(lat):
        return None
    
    # Filter by spatial extent
    min_lon = getattr(info, 'min_lon', -180)
    max_lon = getattr(info, 'max_lon', 180)
    min_lat = getattr(info, 'min_lat', -90)
    max_lat = getattr(info, 'max_lat', 90)
    
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        return None
    
    # Filter by drainage area
    min_uparea = getattr(info, 'min_uparea', 0)
    max_uparea = getattr(info, 'max_uparea', 1e12)
    if not np.isnan(area):
        if not (min_uparea <= area <= max_uparea):
            return None
    
    # Get streamflow data for this station
    streamflow = streamflow_data[station_idx, :]

    # Check if station has any valid data
    valid_mask = ~np.isnan(streamflow)
    if not valid_mask.any():
        return None
        
    valid_indices = np.where(valid_mask)[0]
    start_year = pd.to_datetime(times[valid_indices[0]]).year
    end_year = pd.to_datetime(times[valid_indices[-1]]).year

    use_syear = max(start_year, int(getattr(info, 'sim_syear', -9999)), int(getattr(info, 'syear', -9999)))
    use_eyear = min(end_year, int(getattr(info, 'sim_eyear', 9999)), int(getattr(info, 'eyear', 9999)))
    
    # Check if time range is valid and meets minimum length
    if (use_eyear - use_syear) < getattr(info, 'min_year', 1):
        return None
    
    # Create output NetCDF for this station
    output_file = os.path.join(scratch_dir, f"{station_id}.nc")
    
    ds_out = xr.Dataset(
        data_vars={
            'discharge': (['time'], streamflow, {
                'long_name': 'Daily mean discharge',
                'units': 'm3 s-1'
            })
        },
        coords={
            'time': times
        },
        attrs={
            'station_id': station_id,
            'latitude': lat,
            'longitude': lon,
            'drainage_area_km2': area if not np.isnan(area) else -9999.0
        }
    )
    
    ds_out.to_netcdf(output_file)
    
    return [station_id, lon, lat, use_syear, use_eyear, output_file]


def filter_Beck_2017(info, ds=None):
    """
    Filter and extract Beck_2017 station data.
    
    Parameters
    ----------
    info : dict
        Configuration/info object
    ds : xarray.Dataset, optional
        Dataset to filter (for data filtering mode)
    
    Returns
    -------
    tuple or str
        (info, data) in data filtering mode
        Path to station list file in initialization mode
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        varname = 'streamflow'
        if varname in ds:
            return info, ds[varname]
        elif 'discharge' in ds:
            return info, ds['discharge']
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds

    # Setup paths
    data_dir = info.ref_dir
    casedir = info.casedir
    ref_source = info.ref_source
    sim_source = getattr(info, 'sim_source', 'unknown')
    
    # Input file
    nc_file = os.path.join(data_dir, 'Beck_2017_daily.nc')
    
    # Output directories
    scratch_dir = os.path.join(casedir, 'scratch', f'{ref_source}_{sim_source}')
    os.makedirs(scratch_dir, exist_ok=True)
    
    # Station list file
    stn_list_file = os.path.join(casedir, f'stn_{ref_source}_{sim_source}_list.txt')
    
    logger.info(f"Loading Beck_2017 data from {nc_file}")
    
    # Load data
    ds = xr.open_dataset(nc_file)
    
    # Extract arrays for parallel processing
    station_ids = ds['station_id'].values
    lons = ds['lon'].values
    lats = ds['lat'].values
    areas = ds['area'].values
    streamflow_data = ds['discharge'].values
    times = ds['time'].values
    
    n_stations = len(station_ids)
    logger.info(f"Processing {n_stations} stations")
    
    # Close dataset (data is already in memory)
    ds.close()
    
    # Parallel processing
    num_cores = getattr(info, 'num_cores', 4)
    station_rows = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(process_site)(
            i, station_ids, lons, lats, areas, streamflow_data, times, info, scratch_dir
        )
        for i in range(n_stations)
    )
    
    # Filter out None results
    station_rows = [r for r in station_rows if r is not None]
    logger.info(f"Extracted {len(station_rows)} valid stations")
    
    if not station_rows:
        logging.error("No Beck_2017 stations satisfy the selection criteria.")
        return

    # Create DataFrame with standard columns
    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )
    
    # Update info object
    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = stn_list_file
    info.stn_list = df.copy() # Store dataframe in info object
    
    # Write station list to CSV
    df.to_csv(stn_list_file, index=False)
    
    logger.info(f"Station list saved to {stn_list_file}")
    
    # In initialization mode, we don't strictly need to return, 
    # but returning key info or None is standard practice.
    # GRDC_RESG_filter doesn't return anything explicit, so it returns None.
    return
