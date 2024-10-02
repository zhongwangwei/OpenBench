import numpy as np
import pandas as pd
import re 
def adjust_time_CLM5(info, ds,syear,eyear,tim_res):
   match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
   if match:
      # normalize time values
      num_value, time_unit = match.groups()
      num_value = int(num_value) if num_value else 1   
      time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
      if time_unit.lower() in ['m', 'month', 'mon']:
         print('Adjusting time values for monthly CLM5 output...')
         ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-15T00:00:00'))
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(months=1)
         time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-15T00:00:00'))
      elif time_unit.lower() in ['d', 'day', '1d', '1day']:
         print('Adjusting time values for daily CLM5 output...')
         ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT12:00:00'))
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)
         time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT12:00:00'))
         
      elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
         print('Adjusting time values for yearly CLM5 output ...')
         ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
         time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT%H:30:00'))

      time_var = ds.time
      time_values = time_var
      # Create a complete time series based on the specified time frequency and range 
      # Compare the actual time with the complete time series to find the missing time
      #print('Checking time series completeness...')
      missing_times = time_index[~np.isin(time_index, time_values)]
      if len(missing_times) > 0:
         print("Time series is not complete. Missing time values found: ")
         print(missing_times)
         print('Filling missing time values with np.nan')
         # Fill missing time values with np.nan
         ds = ds.reindex(time=time_index)
         ds = ds.where(ds.time.isin(time_values), np.nan)
      else:
         #print('Time series is complete.')
         pass
   else:
      print('tim_res error')
      exit()
   return ds

def filter_CLM5(info, ds):   #update info as well
   if info.item == "Net_Radiation":
      try:
         ds['Net_Radiation'] = ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE']
         info.sim_varname = 'Net_Radiation'
         info.sim_varunit = 'W m-2'
      except Exception as e:
         print(f"Surface Net Radiation calculation processing ERROR: {e}")
         return info, None
      return info, ds['Net_Radiation']
   if info.item == "Surface_Net_LW_Radiation":
      try:
         ds['FIRA']=-ds['FIRA']
         info.sim_varname='FIRA'
         info.sim_varunit='W m-2'
      except:
         print('Surface Net LW Radiation calculation processing ERROR!!!')
      return info, ds['FIRA']
   
   if info.item == "Surface_Soil_Moisture":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1))/0.06/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
   if info.item == "Root_Zone_Soil_Moisture":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1)+
                                       ds['SOILLIQ'].isel(levsoi=2)+
                                       ds['SOILLIQ'].isel(levsoi=3)+
                                       ds['SOILLIQ'].isel(levsoi=4)+
                                       ds['SOILLIQ'].isel(levsoi=5)+
                                       ds['SOILLIQ'].isel(levsoi=6)+
                                       ds['SOILLIQ'].isel(levsoi=7)+
                                       ds['SOILLIQ'].isel(levsoi=8)*0.29
                                       )/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
   if info.item == "Root_Zone_Soil_Temperature":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1)+
                                       ds['SOILLIQ'].isel(levsoi=2)+
                                       ds['SOILLIQ'].isel(levsoi=3)+
                                       ds['SOILLIQ'].isel(levsoi=4)+
                                       ds['SOILLIQ'].isel(levsoi=5)+
                                       ds['SOILLIQ'].isel(levsoi=6)+
                                       ds['SOILLIQ'].isel(levsoi=7)+
                                       ds['SOILLIQ'].isel(levsoi=8)*0.29
                                       )/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
   if info.item == "Ground_Heat":
      try:
            ds['Ground_Heat']=  ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE']-ds['FSH']-ds['EFLX_LH_TOT']

            info.sim_varname = 'Ground_Heat'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['Ground_Heat']

   
   if info.item == "Albedo":
      try:
            ds['Albedo']= ds['FSR'] /  ds['FSDS'] 

            info.sim_varname = 'Albedo'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface Albedo calculation processing ERROR: {e}")
         return info, None
      return info, ds['Albedo']

   return info, ds
