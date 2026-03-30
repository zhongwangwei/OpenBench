"""Custom filter for the CH4_FluxnetANN (FLUXNET-CH4 Annual Gap-Filled) reference dataset."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr


def process_site(station_idx, dataset, info):
    """Extract metadata for a single station and persist its series as NetCDF."""
    # Get coordinates for this station
    lon = float(dataset['lon'].isel(data=station_idx).values)
    lat = float(dataset['lat'].isel(data=station_idx).values)
    
    # Use index as station ID since no explicit station_id variable
    station_id = f"FCH4_{station_idx:04d}"
    
    # Get time series data for this station
    fch4 = dataset['FCH4'].isel(data=station_idx)
    
    # Find valid time range (non-missing data)
    valid_mask = ~fch4.isnull()
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

    scratch_dir = Path(info.casedir) / "scratch" / f"CH4_FluxnetANN_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    file_path = scratch_dir / f"{station_id}.nc"
    
    # Save FCH4 data - isel already removes the data dimension
    # so we just need to squeeze any remaining singleton dimensions
    fch4_data = fch4.squeeze(drop=True)
    ds_out = xr.Dataset({
        'FCH4': fch4_data
    })
    ds_out.to_netcdf(file_path)

    return [station_id, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_CH4_FluxnetANN(info, ds=None):
    """Generate required station metadata for CH4_FluxnetANN runs or filter dataset.
    
    This function serves two purposes:
    1. When called with only `info`: Generates station metadata for CH4_FluxnetANN runs
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
        # The variable should already be 'FCH4' as specified in CH4_FluxnetANN.yaml
        varname = 'FCH4'
        if varname in ds:
            return info, ds[varname]
        else:
            # Return the first data variable
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    dataset_path = Path(info.ref_dir) / "FCH4_F_ANN_monthly_wetland_tier1.nc"

    if not dataset_path.exists():
        logging.error(f"CH4_FluxnetANN dataset not found: {dataset_path}")
        return

    with xr.open_dataset(dataset_path) as ds_file:
        station_rows = []
        for idx in range(ds_file.dims['data']):
            result = process_site(idx, ds_file, info)
            if result:
                station_rows.append(result)

    if not station_rows:
        logging.error("No CH4_FluxnetANN stations satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_CH4_FluxnetANN_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info('CH4_FluxnetANN station list saved')
