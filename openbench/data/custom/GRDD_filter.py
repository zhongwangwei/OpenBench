"""Custom filter for the GRDD (Global River Discharge Database) monthly dataset.

Uses parallel processing for faster station data extraction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


def process_site(station_idx, station_ids, lons, lats, areas, discharge_data, times, info, scratch_dir):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(station_ids[station_idx])
    lon = float(lons[station_idx])
    lat = float(lats[station_idx])
    area = float(areas[station_idx])
    
    # Skip stations with missing coordinates
    if np.isnan(lon) or np.isnan(lat) or lon == 0 or lat == 0:
        return None
    
    # Get time series data for this station
    discharge = discharge_data[station_idx, :].copy()
    
    # Find valid time range (non-missing data)
    valid_mask = ~np.isnan(discharge)
    if not valid_mask.any():
        return None
    
    valid_indices = np.where(valid_mask)[0]
    start_year = pd.to_datetime(times[valid_indices[0]]).year
    end_year = pd.to_datetime(times[valid_indices[-1]]).year

    use_syear = max(start_year, int(getattr(info, 'sim_syear', -9999)), int(getattr(info, 'syear', -9999)))
    use_eyear = min(end_year, int(getattr(info, 'sim_eyear', 9999)), int(getattr(info, 'eyear', 9999)))

    # Apply filters: time range, spatial extent
    if ((use_eyear - use_syear) < getattr(info, 'min_year', 1) or
            lon < getattr(info, 'min_lon', -180) or lon > getattr(info, 'max_lon', 180) or
            lat < getattr(info, 'min_lat', -90) or lat > getattr(info, 'max_lat', 90)):
        return None
    
    # Filter by drainage area if specified
    if hasattr(info, 'min_uparea') and not np.isnan(area) and area < info.min_uparea:
        return None
    if hasattr(info, 'max_uparea') and not np.isnan(area) and area > info.max_uparea:
        return None

    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save discharge data as 1D time series
    ds_out = xr.Dataset({
        'discharge': (['time'], discharge)
    }, coords={'time': times})
    ds_out.to_netcdf(file_path)
    
    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_GRDD(info, ds=None):
    """Generate required station metadata for GRDD runs or filter dataset.
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        varname = 'discharge'
        if varname in ds:
            return info, ds[varname]
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "GRDD_monthly.nc"
    
    if not dataset_path.exists():
        logging.error(f"GRDD dataset not found: {dataset_path}")
        return
    
    logging.info("Loading GRDD station metadata...")
    
    # Create scratch directory
    scratch_dir = Path(info.casedir) / "scratch" / f"GRDD_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    with xr.open_dataset(dataset_path) as ds_file:
        # Pre-load all data into memory for parallel processing
        station_ids = ds_file['station'].values
        lons = ds_file['lon'].values
        lats = ds_file['lat'].values
        areas = ds_file['area'].values
        discharge_data = ds_file['discharge'].values  # (station, time)
        times = ds_file['time'].values
        
        n_stations = len(station_ids)
        logging.info(f"Processing {n_stations} stations in parallel...")
        
        # Parallel processing
        num_cores = getattr(info, 'num_cores', -1)
        station_rows = Parallel(n_jobs=num_cores, verbose=1)(
            delayed(process_site)(
                idx, station_ids, lons, lats, areas, discharge_data, times, info, scratch_dir
            ) for idx in range(n_stations)
        )
        
        # Filter out None results
        station_rows = [row for row in station_rows if row is not None]
        
        if not station_rows:
            logging.error("No GRDD stations satisfy the selection criteria.")
            return
    
    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )
    
    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_GRDD_{info.sim_source}_list.txt"
    info.stn_list = df.copy()
    
    df.to_csv(info.ref_fulllist, index=False)
    logging.info(f'GRDD station list saved: {len(df)} stations')
    return
