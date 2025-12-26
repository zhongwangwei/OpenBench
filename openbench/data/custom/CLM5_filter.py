import numpy as np
import pandas as pd
import re
import logging
def adjust_time_CLM5(info, ds,syear,eyear,tim_res):
   match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
   if match:
      # normalize time values
      num_value, time_unit = match.groups()
      num_value = int(num_value) if num_value else 1   
      ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))
      if time_unit.lower() in ['m','me', 'month', 'mon']:
         logging.info('Adjusting time values for monthly CLM5 output...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(months=1)
      elif time_unit.lower() in ['d', 'day', '1d', '1day']:
         logging.info('Adjusting time values for daily CLM5 output...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)  
      elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
         logging.info('Adjusting time values for yearly CLM5 output ...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
   else:
      logging.error('tim_res error: invalid time resolution format')
      raise ValueError('Invalid time resolution format')
   return ds

def filter_CLM5(info, ds):   #update info as well
   if info.item == "Net_Radiation":
      try:
         ds['Net_Radiation'] = ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE']
         info.sim_varname = 'Net_Radiation'
         info.sim_varunit = 'W m-2'
      except Exception as e:
         logging.error(f"Surface Net Radiation calculation processing ERROR: {e}")
         return info, None
      return info, ds['Net_Radiation']
   if info.item == "Surface_Net_LW_Radiation":
      try:
         ds['FIRA']=-ds['FIRA']
         info.sim_varname='FIRA'
         info.sim_varunit='W m-2'
      except Exception as e:
         logging.error(f'Surface Net LW Radiation calculation processing ERROR: {e}')
      return info, ds['FIRA']
   
   if info.item == "Surface_Soil_Moisture":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1))/0.06/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         logging.error(f"Surface soil moisture calculation processing ERROR: {e}")
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
         logging.error(f"Surface soil moisture calculation processing ERROR: {e}")
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
         logging.error(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
   if info.item == "Ground_Heat":
      try:
            ds['Ground_Heat']=  ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE']-ds['FSH']-ds['EFLX_LH_TOT']

            info.sim_varname = 'Ground_Heat'
            info.sim_varunit = 'unitless'
      except Exception as e:
         logging.error(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['Ground_Heat']

   
   if info.item == "Albedo":
      try:
            ds['Albedo']= ds['FSR'] /  ds['FSDS'] 

            info.sim_varname = 'Albedo'
            info.sim_varunit = 'unitless'
      except Exception as e:
         logging.error(f"Surface Albedo calculation processing ERROR: {e}")
         return info, None
      return info, ds['Albedo']



