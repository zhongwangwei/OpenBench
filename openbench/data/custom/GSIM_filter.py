"""Custom filter for the GSIM (Global Streamflow Indices and Metadata) reference dataset."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr


def process_site(station_idx, dataset, info):
    """Extract metadata for a single station and persist its series as NetCDF."""
    station_id = str(dataset['station_id'].isel(station=station_idx).values)
    lon = float(dataset['longitude'].isel(station=station_idx).values)
    lat = float(dataset['latitude'].isel(station=station_idx).values)
    area = float(dataset['area'].isel(station=station_idx).values)
    
    # Skip stations with missing area
    if pd.isna(area) or area < 0:
        return None
    
    # Get time series data for this station
    streamflow = dataset['MEAN'].isel(station=station_idx)
    
    # Find valid time range (non-missing data)
    valid_mask = ~streamflow.isnull()
    if not valid_mask.any():
        return None
    
    valid_times = dataset['time'].where(valid_mask, drop=True)
    start_year = pd.to_datetime(valid_times.values[0]).year
    end_year = pd.to_datetime(valid_times.values[-1]).year

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

    scratch_dir = Path(info.casedir) / "scratch" / f"GSIM_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save streamflow data - isel already removes the station dimension
    # so we just need to squeeze any remaining singleton dimensions
    streamflow_data = streamflow.squeeze(drop=True)
    ds_out = xr.Dataset({
        'discharge': streamflow_data
    })
    ds_out.to_netcdf(file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_GSIM(info, ds=None):
    """Generate required station metadata for GSIM runs or filter dataset.
    
    This function serves two purposes:
    1. When called with only `info`: Generates station metadata for GSIM runs
    2. When called with `info` and `ds`: Acts as a data filter for station processing
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        # The variable should already be 'MEAN' as specified in GSIM.yaml
        # Just return the dataset with the correct variable
        varname = 'MEAN'
        if varname in ds:
            return info, ds[varname]
        elif 'discharge' in ds:
            # Fallback to discharge if MEAN not found
            return info, ds['discharge']
        else:
            # Return the first data variable
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "GSIM_monthly.nc"

    if not dataset_path.exists():
        logging.error(f"GSIM dataset not found: {dataset_path}")
        return

    with xr.open_dataset(dataset_path) as ds_file:
        station_rows = []
        for idx in range(ds_file.dims['station']):
            result = process_site(idx, ds_file, info)
            if result:
                station_rows.append(result)

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
    logging.info('GSIM station list saved')

