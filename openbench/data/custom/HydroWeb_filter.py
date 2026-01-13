import logging

import pandas as pd
import numpy as np
import os, sys
import xarray as xr
from joblib import Parallel, delayed

def process_station(station, info, min_uparea, max_uparea):
    result = {
        'Flag': False,
        'use_syear': -9999,
        'use_eyear': -9999,
        'obs_syear': -9999,
        'obs_eyear': -9999,
        'ref_dir': 'file'
    }
    print(station['ID'])
    if info.compare_tim_res.lower() == '1d':
        file_path = f'{info.ref_dir}/output/river/hydroprd_river_{station["ID"]}.nc'
    else:
        return result
    if os.path.exists(file_path):
        result['ref_dir'] = file_path
        with xr.open_dataset(file_path) as df:
            if info.debug_mode:
               logging.info(f"Processing station {int(station['ID'])}...")
            result['obs_syear'] = int(df["time.year"].values[0])
            result['obs_eyear'] = int(df["time.year"].values[-1])
            result['use_syear'] = max(result['obs_syear'], int(info.sim_syear), int(info.syear))
            result['use_eyear'] = min(result['obs_eyear'], int(info.sim_eyear), int(info.eyear))
            if ((result['use_eyear'] - result['use_syear'] >= info.min_year) and
                (station['lon'] >= info.min_lon) and
                (station['lon'] <= info.max_lon) and
                (station['lat'] >= info.min_lat) and
                (station['lat'] <= info.max_lat)):
               result['Flag'] = True
               if info.debug_mode:
                  logging.info(f"Station {int(station['ID'])} is selected")
    return result

def filter_HydroWeb(info, ds=None):
   """Generate required station metadata for HydroWeb runs or filter dataset.
   
   Args:
       info: Configuration/info object with processing parameters
       ds: Optional xarray Dataset to filter (for data filtering mode)
       
   Returns:
       For data filtering mode: Tuple of (info, filtered_data)
       For initialization mode: None (modifies info in place)
   """
   # If ds is provided, we're in data filtering mode
   if ds is not None:
       data_vars = list(ds.data_vars)
       if data_vars:
           return info, ds[data_vars[0]]
       return info, ds
   
   # Initialization mode
   if info.compare_tim_res.lower() != "d" :
      logging.error('The compare_res should be "Day" ')
      sys.exit(1)
   # Add logging import if not already imported
   import logging
   
   # Confirm the resolution of info.sim_grid_res
   if not hasattr(info, 'sim_grid_res'):
       logging.error("sim_grid_res is not defined in info object")
       sys.exit(1)
   
   if not isinstance(info.sim_grid_res, (int, float)):
       logging.error(f"sim_grid_res must be a number, got {type(info.sim_grid_res)}")
       sys.exit(1)
       
   valid_resolutions = [0.25, 0.0167, 0.0833, 0.1, 0.05]
   #ask user to input the resolution
   #add the reason for the input
   logging.info("The simulation grid resolution of routing model can be different from the LSM grid resolution")
   info.sim_grid_res = float(input("Please input the simulation grid resolution of routing model again: 0.25(15min), 0.0167(1min), 0.0833(5min), 0.1(6min), 0.05(3min)"))
   if info.sim_grid_res not in valid_resolutions:
       logging.error(f"sim_grid_res value {info.sim_grid_res} is not in the list of standard resolutions: {valid_resolutions}")
       sys.exit(1)
   
   if info.debug_mode:
       logging.info(f"Using simulation grid resolution: {info.sim_grid_res} degrees")
   if info.sim_grid_res == 0.25:
      info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_15min.txt"
   elif info.sim_grid_res == 0.0167:
      info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_1min.txt"
   elif info.sim_grid_res == 0.0833:
      info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_5min.txt"
   elif info.sim_grid_res == 0.1:
      info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_6min.txt"
   elif info.sim_grid_res == 0.05:
      info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_3min.txt"
   station_list = pd.read_csv(f"{info.ref_fulllist}", delimiter=r"\s+", header=0)

   results = Parallel(n_jobs=-1)(delayed(process_station)(row, info) for _, row in station_list.iterrows())
   for i, result in enumerate(results):
      for key, value in result.items():
         station_list.at[i, key] = value

   ind = station_list[station_list['Flag']].index
   data_select = station_list.loc[ind]
    
   if info.sim_grid_res == 0.25:
      lat0 = np.arange(89.875, -90, -0.25)
      lon0 = np.arange(-179.875, 180, 0.25)
   elif info.sim_grid_res == 0.0167:  # 01min
      lat0 = np.arange(89.9916666666666600, -90, -0.0166666666666667)
      lon0 = np.arange(-179.9916666666666742, 180, 0.0166666666666667)
   elif info.sim_grid_res == 0.0833:  # 05min
      lat0 = np.arange(89.9583333333333286, -90, -0.0833333333333333)
      lon0 = np.arange(-179.9583333333333428, 180, 0.0833333333333333)
   elif info.sim_grid_res == 0.1:  # 06min
      lat0 = np.arange(89.95, -90, -0.1)
      lon0 = np.arange(-179.95, 180, 0.1)
   elif info.sim_grid_res == 0.05:  # 03min
      lat0 = np.arange(89.975, -90, -0.05)
      lon0 = np.arange(-179.975, 180, 0.05)
   data_select['lon_cama'] = -9999.0
   data_select['lat_cama'] = -9999.0
   for iii in range(len(data_select['ID'])):
      data_select['lon_cama'].values[iii] = float(lon0[int(data_select['ix1'].values[iii]) - 1])
      data_select['lat_cama'].values[iii] = float(lat0[int(data_select['iy1'].values[iii]) - 1])
      if abs(data_select['lat_cama'].values[iii] - data_select['lat'].values[iii]) > 1:
            logging.warning(f"Warning: ID {data_select['ID'][iii]} lat is not match")
      if abs(data_select['lon_cama'].values[iii] - data_select['lon'].values[iii]) > 1:
            logging.warning(f"Warning: ID {data_select['ID'].values[iii]} lon is not match")
   logging.info(f"In total: {len(data_select['ID'])} stations are selected")
   if len(data_select['ID']) == 0:
      logging.error(f"Warning: No stations are selected, please check the station list and the min_year, min_lat, max_lat, min_lon, max_lon")
      sys.exit(1)
   info.use_syear = data_select['use_syear'].min()
   info.use_eyear = data_select['use_eyear'].max()
   data_select['ref_lon'] = data_select['lon_cama']
   data_select['ref_lat'] = data_select['lat_cama']
   #all year here should be int
   data_select['use_syear'] = data_select['use_syear'].astype(int)
   data_select['use_eyear'] = data_select['use_eyear'].astype(int)
   data_select['obs_syear'] = data_select['obs_syear'].astype(int)
   data_select['obs_eyear'] = data_select['obs_eyear'].astype(int)
   data_select['ID'] = data_select['ID'] #.astype(int)
   
   data_select.to_csv(f"{info.casedir}/stn_HydroWeb_{info.sim_source}_list.txt", index=False)

