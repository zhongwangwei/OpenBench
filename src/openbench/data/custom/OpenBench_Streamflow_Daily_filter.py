"""Custom filter for the OpenBench consolidated daily streamflow dataset.

This is the merged dataset containing all daily streamflow stations.
Uses parallel processing and CaMA-Flood allocation support.
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


def process_site(station_idx, station_ids, lons, lats, areas,
                 cama_lons, cama_lats, alloc_errs,
                 discharge_data, times, info, scratch_dir, area_err_threshold):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(station_ids[station_idx])
    lon = float(lons[station_idx])
    lat = float(lats[station_idx])
    area = float(areas[station_idx]) if not np.isnan(areas[station_idx]) else -9999.0
    
    cama_lon = float(cama_lons[station_idx])
    cama_lat = float(cama_lats[station_idx])
    alloc_err = float(alloc_errs[station_idx])
    
    if np.isnan(cama_lon) or np.isnan(cama_lat) or cama_lon < -180 or cama_lat < -90:
        return None
    
    if not np.isnan(alloc_err) and alloc_err > area_err_threshold:
        return None
    
    if np.isnan(lon) or np.isnan(lat):
        return None
    
    discharge = discharge_data[station_idx, :]
    
    valid_mask = ~np.isnan(discharge)
    if not valid_mask.any():
        return None
    
    valid_indices = np.where(valid_mask)[0]
    start_year = pd.to_datetime(times[valid_indices[0]]).year
    end_year = pd.to_datetime(times[valid_indices[-1]]).year

    use_syear = max(start_year, int(getattr(info, 'sim_syear', -9999)), int(getattr(info, 'syear', -9999)))
    use_eyear = min(end_year, int(getattr(info, 'sim_eyear', 9999)), int(getattr(info, 'eyear', 9999)))

    if ((use_eyear - use_syear) < getattr(info, 'min_year', 1) or
            lon < getattr(info, 'min_lon', -180) or lon > getattr(info, 'max_lon', 180) or
            lat < getattr(info, 'min_lat', -90) or lat > getattr(info, 'max_lat', 90)):
        return None
    
    # Filter by drainage area if specified
    if hasattr(info, 'min_uparea') and area > 0 and area < info.min_uparea:
        return None
    if hasattr(info, 'max_uparea') and area > 0 and area > info.max_uparea:
        return None

    file_path = scratch_dir / f"{station_id}.nc"
    
    ds_out = xr.Dataset({
        'discharge': (['time'], discharge)
    }, coords={'time': times})
    ds_out.to_netcdf(file_path)
    
    return [station_id, cama_lon, cama_lat, use_syear, use_eyear, str(file_path)]


def filter_OpenBench_Streamflow_Daily(info, ds=None):
    """Generate required station metadata for OpenBench Daily runs or filter dataset."""
    if ds is not None:
        if 'discharge' in ds:
            return info, ds['discharge']
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    dataset_path = Path(info.ref_dir) / "OpenBench_Streamflow_Daily.nc"
    
    if not dataset_path.exists():
        logging.error(f"Dataset not found: {dataset_path}")
        return
    
    logging.info(f"Loading OpenBench Daily metadata from {dataset_path}...")
    
    if hasattr(info, 'sim_grid_res'):
        res_suffix = get_resolution_suffix(info.sim_grid_res)
    else:
        res_suffix = '03min'
    
    area_err_threshold = getattr(info, 'area_err_threshold', 0.2)
    logging.info(f"Area allocation error threshold: {area_err_threshold*100:.1f}%")
    
    scratch_dir = Path(info.casedir) / "scratch" / f"OpenBench_Streamflow_Daily_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    with xr.open_dataset(dataset_path) as ds_file:
        station_ids = ds_file['station'].values
        lons = ds_file['lon'].values
        lats = ds_file['lat'].values
        areas = ds_file['area'].values
        discharge_data = ds_file['discharge'].values
        times = ds_file['time'].values
        
        cama_lon_var = f'cama_lon_{res_suffix}'
        if cama_lon_var in ds_file:
            cama_lons = ds_file[cama_lon_var].values
            cama_lats = ds_file[f'cama_lat_{res_suffix}'].values
            alloc_errs = ds_file[f'cama_alloc_err_{res_suffix}'].values
        else:
            cama_lons = lons.copy()
            cama_lats = lats.copy()
            alloc_errs = np.zeros_like(lons)
        
        n_stations = len(station_ids)
        logging.info(f"Processing {n_stations} stations...")
        
        station_rows = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_site)(
                idx, station_ids, lons, lats, areas,
                cama_lons, cama_lats, alloc_errs,
                discharge_data, times, info, scratch_dir, area_err_threshold
            ) for idx in range(n_stations)
        )
        
        station_rows = [row for row in station_rows if row is not None]

    if not station_rows:
        logging.error("No stations satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_OpenBench_Streamflow_Daily_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info(f'Station list saved: {len(df)} stations')
