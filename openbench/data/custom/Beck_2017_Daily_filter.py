"""
Beck_2017 Custom Filter for OpenBench

Updated to read CaMA-Flood allocation data from consolidated NetCDF file.
Supports multi-resolution allocation and area error threshold filtering.

Uses parallel processing for efficient extraction.
"""

import os
import logging
import xarray as xr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def get_resolution_suffix(sim_grid_res):
    """Map simulation grid resolution to CaMA resolution suffix."""
    res_map = {
        0.25: '15min',
        0.1: '06min',
        0.0833: '05min',
        0.05: '03min',
        0.0167: '01min'
    }
    for res, suffix in res_map.items():
        if abs(float(sim_grid_res) - res) < 0.001:
            return suffix
    logger.warning(f"Unknown resolution {sim_grid_res}, defaulting to 03min")
    return '03min'


def process_site(station_idx, station_ids, lons, lats, areas,
                 cama_lons, cama_lats, alloc_errs,
                 streamflow_data, times, info, scratch_dir, area_err_threshold):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(station_ids[station_idx])
    lon = float(lons[station_idx])
    lat = float(lats[station_idx])
    area = float(areas[station_idx]) if not np.isnan(areas[station_idx]) else -9999.0
    
    # Get CaMA allocation data
    cama_lon = float(cama_lons[station_idx])
    cama_lat = float(cama_lats[station_idx])
    alloc_err = float(alloc_errs[station_idx])
    
    # Skip if invalid CaMA allocation
    if np.isnan(cama_lon) or np.isnan(cama_lat) or cama_lon < -180 or cama_lat < -90:
        return None
    
    # Filter by area allocation error threshold
    if not np.isnan(alloc_err) and alloc_err > area_err_threshold:
        return None
    
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
    if not np.isnan(area) and area > 0:
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
        coords={'time': times},
        attrs={
            'station_id': station_id,
            'latitude': lat,
            'longitude': lon,
            'drainage_area_km2': area if not np.isnan(area) else -9999.0
        }
    )
    
    ds_out.to_netcdf(output_file)
    
    return [station_id, cama_lon, cama_lat, use_syear, use_eyear, output_file]


def filter_Beck_2017_Daily(info, ds=None):
    """Filter and extract Beck_2017_Daily station data with CaMA allocation support."""
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
    
    # Get resolution suffix for CaMA variables
    if hasattr(info, 'sim_grid_res'):
        res_suffix = get_resolution_suffix(info.sim_grid_res)
        logger.info(f"Using CaMA resolution: {res_suffix}")
    else:
        res_suffix = '03min'
        logger.warning("sim_grid_res not defined, using default 03min resolution")
    
    # Get area error threshold from config (default 0.2 = 20%)
    area_err_threshold = getattr(info, 'area_err_threshold', 0.2)
    logger.info(f"Area allocation error threshold: {area_err_threshold*100:.1f}%")
    
    # Load data
    ds = xr.open_dataset(nc_file)
    
    # Extract arrays for parallel processing
    station_ids = ds['station_id'].values
    lons = ds['lon'].values
    lats = ds['lat'].values
    areas = ds['area'].values
    # Load streamflow data, handling potential variable name variations
    if 'discharge' in ds:
        streamflow_data = ds['discharge'].values
    elif 'streamflow' in ds:
        streamflow_data = ds['streamflow'].values
    else:
        raise ValueError("Neither 'discharge' nor 'streamflow' found in dataset")
    times = ds['time'].values
    
    # Load CaMA allocation data
    cama_lon_var = f'cama_lon_{res_suffix}'
    cama_lat_var = f'cama_lat_{res_suffix}'
    alloc_err_var = f'cama_alloc_err_{res_suffix}'
    
    if cama_lon_var in ds:
        cama_lons = ds[cama_lon_var].values
        cama_lats = ds[cama_lat_var].values
        alloc_errs = ds[alloc_err_var].values
    else:
        logger.warning(f"CaMA variable {cama_lon_var} not found, using original coordinates")
        cama_lons = lons.copy()
        cama_lats = lats.copy()
        alloc_errs = np.zeros_like(lons)
    
    n_stations = len(station_ids)
    logger.info(f"Processing {n_stations} stations")
    
    # Close dataset (data is already in memory)
    ds.close()
    
    # Parallel processing
    num_cores = getattr(info, 'num_cores', -1)
    station_rows = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(process_site)(
            i, station_ids, lons, lats, areas,
            cama_lons, cama_lats, alloc_errs,
            streamflow_data, times, info, scratch_dir, area_err_threshold
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
    info.stn_list = df.copy()
    
    # Write station list to CSV
    df.to_csv(stn_list_file, index=False)
    logger.info(f"Station list saved to {stn_list_file}")

    return
