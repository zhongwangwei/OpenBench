"""Custom filter for the GSIM (Global Streamflow Indices and Metadata) reference dataset.

Uses parallel processing for faster station data extraction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


def process_site(station_idx, station_ids, lons, lats, areas, streamflow_data, times, info, scratch_dir):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(station_ids[station_idx])
    lon = float(lons[station_idx])
    lat = float(lats[station_idx])
    area = float(areas[station_idx])
    
    # Skip stations with missing area
    if np.isnan(area) or area < 0:
        return None
    
    # Get time series data for this station
    streamflow = streamflow_data[station_idx, :]
    
    # Find valid time range (non-missing data)
    valid_mask = ~np.isnan(streamflow)
    if not valid_mask.any():
        return None
    
    valid_indices = np.where(valid_mask)[0]
    start_year = pd.to_datetime(times[valid_indices[0]]).year
    end_year = pd.to_datetime(times[valid_indices[-1]]).year

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    # Apply filters: time range, spatial extent, and drainage area
    if ((use_eyear - use_syear) < info.min_year or
            lon < info.min_lon or lon > info.max_lon or
            lat < info.min_lat or lat > info.max_lat):
        return None
    
    # Filter by drainage area if specified
    if hasattr(info, 'min_uparea') and area < info.min_uparea:
        return None
    if hasattr(info, 'max_uparea') and area > info.max_uparea:
        return None

    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save streamflow data as 1D time series
    ds_out = xr.Dataset({
        'discharge': (['time'], streamflow)
    }, coords={'time': times})
    ds_out.to_netcdf(file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_GSIM(info, ds=None):
    """Generate required station metadata for GSIM runs or filter dataset.
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        varname = 'MEAN'
        if varname in ds:
            return info, ds[varname]
        elif 'discharge' in ds:
            return info, ds['discharge']
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "GSIM_monthly.nc"

    if not dataset_path.exists():
        logging.error(f"GSIM dataset not found: {dataset_path}")
        return

    logging.info("Loading GSIM station metadata...")
    
    # Create scratch directory
    scratch_dir = Path(info.casedir) / "scratch" / f"GSIM_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    with xr.open_dataset(dataset_path) as ds_file:
        # Pre-load all data into memory for parallel processing
        station_ids = ds_file['station_id'].values
        lons = ds_file['longitude'].values
        lats = ds_file['latitude'].values
        areas = ds_file['area'].values
        streamflow_data = ds_file['MEAN'].values  # (station, time)
        times = ds_file['time'].values
        
        n_stations = len(station_ids)
        logging.info(f"Processing {n_stations} stations in parallel...")
        
        # Process stations in parallel
        station_rows = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_site)(
                idx, station_ids, lons, lats, areas, streamflow_data, times, info, scratch_dir
            ) for idx in range(n_stations)
        )
        
        # Filter out None results
        station_rows = [row for row in station_rows if row is not None]

    if not station_rows:
        logging.error("No GSIM stations satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_GSIM_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info(f'GSIM station list saved: {len(df)} stations')
