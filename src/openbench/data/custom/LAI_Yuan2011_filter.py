"""Custom filter for the LAI_Yuan2011 reference dataset."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr


def process_site(site, dataset, info):
    """Extract metadata for a single site and persist its series as NetCDF."""
    site_name = str(site.values)
    lon = float(dataset['lon'].sel(site=site).values)
    lat = float(dataset['lat'].sel(site=site).values)
    time = dataset['time']
    start_year = pd.to_datetime(time.values[0]).year
    end_year = pd.to_datetime(time.values[-1]).year

    use_syear = max(start_year, int(info.sim_syear), int(info.syear))
    use_eyear = min(end_year, int(info.sim_eyear), int(info.eyear))

    if ((use_eyear - use_syear) < info.min_year or
            lon < info.min_lon or lon > info.max_lon or
            lat < info.min_lat or lat > info.max_lat):
        return None

    scratch_dir = Path(info.casedir) / "scratch" / f"LAI_Yuan2011_{info.sim_source}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    file_path = scratch_dir / f"{site_name}.nc"
    dataset['lai'].sel(site=site).squeeze().to_netcdf(file_path)

    return [site_name, lon, lat, use_syear, use_eyear, str(file_path)]


def filter_LAI_Yuan2011(info, ds=None):
    """Generate required station metadata for LAI_Yuan2011 runs or filter dataset.
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        if 'lai' in ds:
            return info, ds['lai']
        else:
            data_vars = list(ds.data_vars)
            if data_vars:
                return info, ds[data_vars[0]]
            return info, ds
    
    # Initialization mode: generate station list
    compare_res = str(info.compare_tim_res).lower()
    is_daily = compare_res in {"1d", "day", "daily", "d"}
    filename = f"lai_{'8-day' if is_daily else 'monthly'}_500.nc"
    dataset_path = Path(info.ref_dir) / filename

    if not dataset_path.exists():
        logging.error(f"LAI_Yuan2011 dataset not found: {dataset_path}")
        return

    with xr.open_dataset(dataset_path) as ds_file:
        station_rows = []
        for site in ds_file['site']:
            result = process_site(site, ds_file, info)
            if result:
                station_rows.append(result)

    if not station_rows:
        logging.error("No LAI_Yuan2011 sites satisfy the selection criteria.")
        return

    df = pd.DataFrame(
        station_rows,
        columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir']
    )

    info.use_syear = int(df['use_syear'].min())
    info.use_eyear = int(df['use_eyear'].max())
    info.ref_fulllist = f"{info.casedir}/stn_LAI_Yuan2011_{info.sim_source}_list.txt"
    info.stn_list = df.copy()

    df.to_csv(info.ref_fulllist, index=False)
    logging.info('LAI_Yuan2011 station list saved')

