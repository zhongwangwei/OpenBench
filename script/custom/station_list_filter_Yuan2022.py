import pandas as pd
import numpy as np
import os
import xarray as xr

def process_site(site, ds, info):
    sitename = site.values
    lon = ds['lon'].sel(site=site).values
    lat = ds['lat'].sel(site=site).values
    time = ds['time']
    SYear = pd.to_datetime(time.values[0]).year
    EYear = pd.to_datetime(time.values[-1]).year
    use_syear = max(SYear, info.sim_syear, info.syear)
    use_eyear = min(EYear, info.sim_eyear, info.eyear)
    
    if ((use_eyear - use_syear >= info.min_year) and
        (info.min_lon <= lon <= info.max_lon) and
        (info.min_lat <= lat <= info.max_lat)):
        file_path = f"{info.casedir}/scratch/{sitename}.nc"
        ds['lai'].sel(site=site).squeeze().to_netcdf(file_path)
        return [sitename, float(lon), float(lat), int(use_syear), int(use_eyear), str(file_path)]
    return None

def filter_Yuan2022(info):    
    
    file_path = f'{info.ref_dir}/lai_{"8-day" if info.compare_tim_res.lower() == "1d" else "monthly"}_500.nc'
    
    with xr.open_dataset(file_path) as ds:
        data = [result for site in ds['site'] if (result := process_site(site, ds, info))]
    
    if not data:
        print("No sites match the criteria.")
        return
    
    df = pd.DataFrame(data, columns=['ID', 'ref_lon', 'ref_lat', 'use_syear', 'use_eyear', 'ref_dir'])
    
    info.use_syear = df['use_syear'].min()
    info.use_eyear = df['use_eyear'].max()
    info.ref_fulllist = f"{info.casedir}/stn_list.txt"
    
    df.to_csv(info.ref_fulllist, index=False)
    print('Save list done')

# ... (rest of the code remains unchanged)