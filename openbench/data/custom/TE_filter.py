"""
TE (TerraE) custom filter.

This filter handles variable transformations specific to TE model output.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import re
import pandas as pd
import logging


def adjust_time_TE(info, ds, syear, eyear, tim_res):
    """Adjust time values for TE output based on temporal resolution."""
    match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
    if match:
        num_value, time_unit = match.groups()
        num_value = int(num_value) if num_value else 1

        if time_unit.lower() in ['m', 'me', 'month', 'mon']:
            freq = f'{num_value}M'
        elif time_unit.lower() in ['d', 'day', '1d', '1day']:
            freq = f'{num_value}D'
        elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
            freq = f'{num_value}H'
        else:
            logging.error(f'Unsupported time unit: {time_unit}')
            exit()

        new_time_range = pd.date_range(start=f'{syear}-01-01', end=f'{eyear}-12-31', freq=freq)
        ds['time'] = new_time_range
    else:
        logging.error('tim_res error')
        exit()

    return ds


def filter_TE(info, ds=None):
    """
    Custom filter for TE model variables.

    Returns:
        Tuple of (metadata_dict, data_array) where metadata_dict contains
        varname and varunit updates. Returns None if ds is None.
    """
    if ds is None:
        return None

    # Get current varname/varunit from info (read-only)
    current_varname = info.sim_varname if isinstance(info.sim_varname, list) else [info.sim_varname]
    current_varunit = getattr(info, 'sim_varunit', None)

    if info.item == "Total_Runoff":
        try:
            ds['Total_Runoff'] = (ds['RUNOFF'][:, 0, :, :]).squeeze()
            return {'varname': ['Total_Runoff'], 'varunit': 'kg m-2 s-1'}, ds['Total_Runoff']
        except Exception:
            logging.error('Total_Runoff calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    elif info.item == "Streamflow":
        try:
            ds['Streamflow'] = (ds['outflw']).squeeze()
            return {'varname': ['Streamflow'], 'varunit': 'm3 s-1'}, ds['Streamflow']
        except Exception:
            logging.error('Streamflow calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: return empty metadata and the dataset
    return {}, ds
