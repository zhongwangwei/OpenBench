"""Custom filter for the Dai & Trenberth river discharge reference dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def process_site(station_idx, dataset, info):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(int(dataset['id'].isel(station=station_idx).values))
    lon = float(dataset['lon'].isel(station=station_idx).values)
    lat = float(dataset['lat'].isel(station=station_idx).values)
    area = float(dataset['area_stn'].isel(station=station_idx).values)
    
    # Skip stations with missing area
    if pd.isna(area) or area < 0:
        return None
    
    # Get time series data for this station (FLOW is monthly discharge)
    streamflow = dataset['FLOW'].isel(station=station_idx)
    
    # Find valid time range (non-missing data)
    valid_mask = ~np.isnan(streamflow.values) & (streamflow.values != -999)
    if not valid_mask.any():
        return None
    
    # Parse time from YYYYMM format
    time_vals = dataset['time'].values
    valid_time_vals = time_vals[valid_mask]
    start_year = int(str(valid_time_vals[0])[:4])
    end_year = int(str(valid_time_vals[-1])[:4])

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    # Apply filters: time range, spatial extent
    if ((use_eyear - use_syear) < info.min_year or
            lon < info.min_lon or lon > info.max_lon or
            lat < info.min_lat or lat > info.max_lat):
        return None
    
    # Filter by drainage area if specified
    if hasattr(info, 'min_uparea') and area < info.min_uparea:
        return None
    if hasattr(info, 'max_uparea') and area > info.max_uparea:
        return None

    scratch_dir = Path(info.casedir) / "scratch" / f"Dai_Trenberth_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    file_path = scratch_dir / f"{station_id}.nc"
    
    # Convert YYYYMM time to proper datetime and save (handle float values by converting to int first)
    time_dates = pd.to_datetime([str(int(t)) for t in time_vals], format='%Y%m')
    ds_out = xr.Dataset({
        'discharge': xr.DataArray(
            streamflow.values,
            dims=['time'],
            coords={'time': time_dates}
        )
    })
    ds_out.to_netcdf(file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_Dai_Trenberth(info, ds=None):
    """Generate required station metadata for Dai & Trenberth runs or filter dataset.
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        if 'FLOW' in ds:
            return info, ds['FLOW']
        elif 'discharge' in ds:
            return info, ds['discharge']
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "coastal-stns-Vol-monthly.updated-May2019.nc"

    if not dataset_path.exists():
        logging.error(f"Dai & Trenberth dataset not found: {dataset_path}")
        return

    with xr.open_dataset(dataset_path, decode_times=False) as ds_file:
        station_rows = []
        for idx in range(ds_file.dims['station']):
            result = process_site(idx, ds_file, info)
            if result:
                station_rows.append(result)

    if not station_rows:
        logging.error("No Dai & Trenberth stations satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_Dai_Trenberth_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info('Dai & Trenberth station list saved')

