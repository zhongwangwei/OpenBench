"""Custom filter for the GRDC-RESG (Remote Sensing-based Extension for GRDC) reference dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def process_site(station_idx, dataset, info):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(int(dataset['GRDC_Num'].isel(GRDC_Num=station_idx).values))
    lon = float(dataset['Lon'].isel(GRDC_Num=station_idx).values)
    lat = float(dataset['Lat'].isel(GRDC_Num=station_idx).values)
    
    # Get time series data for this station
    streamflow = dataset['Disch'].isel(GRDC_Num=station_idx)
    
    # Find valid time range (non-missing data)
    valid_mask = ~np.isnan(streamflow.values)
    if not valid_mask.any():
        return None
    
    valid_times = dataset['Time'].where(xr.DataArray(valid_mask, dims='Time'), drop=True)
    start_year = pd.to_datetime(valid_times.values[0]).year
    end_year = pd.to_datetime(valid_times.values[-1]).year

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    # Apply filters: time range, spatial extent
    if ((use_eyear - use_syear) < info.min_year or
            lon < info.min_lon or lon > info.max_lon or
            lat < info.min_lat or lat > info.max_lat):
        return None

    scratch_dir = Path(info.casedir) / "scratch" / f"GRDC_RESG_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save streamflow data - isel already removes the GRDC_Num dimension
    # so we just need to squeeze any remaining singleton dimensions and rename Time to time
    streamflow_data = streamflow.squeeze(drop=True)
    # Rename Time to time if needed
    if 'Time' in streamflow_data.dims:
        streamflow_data = streamflow_data.rename({'Time': 'time'})
    ds_out = xr.Dataset({
        'discharge': streamflow_data
    })
    ds_out.to_netcdf(file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_GRDC_RESG(info, ds=None):
    """Generate required station metadata for GRDC-RESG runs or filter dataset.
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
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
    dataset_path = Path(info.ref_dir) / "RSEG_V01.nc"

    if not dataset_path.exists():
        logging.error(f"GRDC-RESG dataset not found: {dataset_path}")
        return

    with xr.open_dataset(dataset_path) as ds_file:
        station_rows = []
        for idx in range(ds_file.dims['GRDC_Num']):
            result = process_site(idx, ds_file, info)
            if result:
                station_rows.append(result)

    if not station_rows:
        logging.error("No GRDC-RESG stations satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_GRDC_RESG_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info('GRDC-RESG station list saved')

