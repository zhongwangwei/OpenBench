"""Custom filter for the GRDC-RESG (Remote Sensing-based Extension for GRDC) reference dataset.

Updated to read CaMA-Flood allocation data directly from consolidated NetCDF files.
Supports multi-resolution allocation and area error threshold filtering.

Uses parallel processing for faster station data extraction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


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
    logging.warning(f"Unknown resolution {sim_grid_res}, defaulting to 03min")
    return '03min'


def process_site(station_idx, station_ids, lons, lats,
                 cama_lons, cama_lats, alloc_errs,
                 streamflow_data, times, info, scratch_dir, area_err_threshold):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(int(station_ids[station_idx]))
    lon = float(lons[station_idx])
    lat = float(lats[station_idx])
    
    # Get CaMA allocation data
    cama_lon = float(cama_lons[station_idx])
    cama_lat = float(cama_lats[station_idx])
    alloc_err = float(alloc_errs[station_idx])
    
    # Skip stations with invalid CaMA allocation
    if np.isnan(cama_lon) or np.isnan(cama_lat) or cama_lon < -180 or cama_lat < -90:
        return None
    
    # Filter by area allocation error threshold
    if not np.isnan(alloc_err) and alloc_err > area_err_threshold:
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

    # Apply filters: time range, spatial extent
    if ((use_eyear - use_syear) < info.min_year or
            lon < info.min_lon or lon > info.max_lon or
            lat < info.min_lat or lat > info.max_lat):
        return None

    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save streamflow data as 1D time series
    ds_out = xr.Dataset({
        'discharge': (['time'], streamflow)
    }, coords={'time': times})
    ds_out.to_netcdf(file_path)

    return [station_id, cama_lon, cama_lat, use_syear, use_eyear, str(file_path)]


def filter_GRDC_RESG_Monthly(info, ds=None):
    """Generate required station metadata for GRDC_RESG_Monthly runs or filter dataset."""
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        if 'Disch' in ds:
            return info, ds['Disch']
        elif 'discharge' in ds:
            return info, ds['discharge']
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "RSEG_monthly.nc"

    if not dataset_path.exists():
        logging.error(f"GRDC-RESG dataset not found: {dataset_path}")
        return

    logging.info(f"Loading GRDC-RESG station metadata from {dataset_path}...")
    
    # Get resolution suffix for CaMA variables
    if hasattr(info, 'sim_grid_res'):
        res_suffix = get_resolution_suffix(info.sim_grid_res)
        logging.info(f"Using CaMA resolution: {res_suffix}")
    else:
        res_suffix = '03min'
        logging.warning("sim_grid_res not defined, using default 03min resolution")
    
    # Get area error threshold from config (default 0.2 = 20%)
    area_err_threshold = getattr(info, 'area_err_threshold', 0.2)
    logging.info(f"Area allocation error threshold: {area_err_threshold*100:.1f}%")
    
    # Create scratch directory
    scratch_dir = Path(info.casedir) / "scratch" / f"GRDC_RESG_Monthly_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    with xr.open_dataset(dataset_path) as ds_file:
        # Pre-load all data into memory for parallel processing
        station_ids = ds_file['GRDC_Num'].values
        lons = ds_file['Lon'].values
        lats = ds_file['Lat'].values
        streamflow_data = ds_file['Disch'].values  # (GRDC_Num, Time)
        times = ds_file['Time'].values
        
        # Load CaMA allocation data
        cama_lon_var = f'cama_lon_{res_suffix}'
        cama_lat_var = f'cama_lat_{res_suffix}'
        alloc_err_var = f'cama_alloc_err_{res_suffix}'
        
        if cama_lon_var in ds_file:
            cama_lons = ds_file[cama_lon_var].values
            cama_lats = ds_file[cama_lat_var].values
            alloc_errs = ds_file[alloc_err_var].values
        else:
            logging.warning(f"CaMA variable {cama_lon_var} not found, using original coordinates")
            cama_lons = lons.copy()
            cama_lats = lats.copy()
            alloc_errs = np.zeros_like(lons)
        
        n_stations = len(station_ids)
        logging.info(f"Processing {n_stations} stations in parallel...")
        
        # Process stations in parallel
        station_rows = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_site)(
                idx, station_ids, lons, lats,
                cama_lons, cama_lats, alloc_errs,
                streamflow_data, times, info, scratch_dir, area_err_threshold
            ) for idx in range(n_stations)
        )
        
        # Filter out None results
        station_rows = [row for row in station_rows if row is not None]

    if not station_rows:
        logging.error("No GRDC-RESG stations satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_GRDC_RESG_Monthly_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info(f'GRDC-RESG station list saved: {len(df)} stations')
