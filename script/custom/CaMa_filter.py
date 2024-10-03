import numpy as np
import pandas as pd
import re 
def adjust_time_CaMa(info, ds,syear,eyear,tim_res):
   match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
   if match:
      # normalize time values
      num_value, time_unit = match.groups()
      num_value = int(num_value) if num_value else 1 
      try: 
         ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S'))
      except:
         print('time format error')
      if time_unit.lower() in ['m', 'month', 'mon']:
         pass
      elif time_unit.lower() in ['d', 'day', '1d', '1day']:
         print('Adjusting time values for daily CaMa output...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)
      elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
         print('Adjusting time values for yearly CaMa output ...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
      else:
         print('tim_res error')
         exit()
   return ds

