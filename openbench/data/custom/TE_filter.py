import re
import pandas as pd
import logging

def adjust_time_TE(info, ds, syear, eyear, tim_res):
    match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
    if match:
        # Extract time resolution information
        num_value, time_unit = match.groups()
        num_value = int(num_value) if num_value else 1

        # Determine the frequency based on time_unit
        if time_unit.lower() in ['m', 'me', 'month', 'mon']:
            freq = f'{num_value}M'
        elif time_unit.lower() in ['d', 'day', '1d', '1day']:
            freq = f'{num_value}D'
        elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
            freq = f'{num_value}H'
        else:
            logging.error(f'Unsupported time unit: {time_unit}')
            raise ValueError(f'Unsupported time unit: {time_unit}')

        # Create a new time range based on syear, eyear, and freq
        new_time_range = pd.date_range(start=f'{syear}-01-01', end=f'{eyear}-12-31', freq=freq)

        # Assign the new time range to ds['time']
        ds['time'] = new_time_range

    else:
        logging.error('tim_res error: invalid time resolution format')
        raise ValueError('Invalid time resolution format')

    return ds


def filter_TE(info, ds):
   """Custom filter for TE model variables."""
   if info.item == "Total_Runoff":
      try:
         ds['Total_Runoff'] = (ds['RUNOFF'][:, 0, :, :]).squeeze()
         info.sim_varname = 'Total_Runoff'
         info.sim_varunit = 'kg m-2 s-1'
      except Exception as e:
         logging.error(f'Total_Runoff calculation processing ERROR: {e}')
         raise
      return info, ds['Total_Runoff']

   elif info.item == "Streamflow":
      try:
         # Check if already processed
         if 'Streamflow' in ds:
            pass  # Already exists, use it directly
         elif 'outflw' in ds:
            ds['Streamflow'] = (ds['outflw']).squeeze()
         else:
            available_vars = list(ds.data_vars)
            raise KeyError(f"Neither 'Streamflow' nor 'outflw' found. Available: {available_vars}")
         info.sim_varname = 'Streamflow'
         info.sim_varunit = 'm3 s-1'
      except Exception as e:
         logging.error(f'Streamflow calculation processing ERROR: {e}')
         raise
      return info, ds['Streamflow']

