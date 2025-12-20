"""
CaMa-Flood custom filter.

This filter handles variable transformations specific to CaMa-Flood model output.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import numpy as np
import pandas as pd
import re
import logging


def adjust_time_CaMa(info, ds, syear, eyear, tim_res):
    """Adjust time values for CaMa-Flood output based on temporal resolution."""
    match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
    if match:
        num_value, time_unit = match.groups()
        num_value = int(num_value) if num_value else 1
        try:
            ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S'))
        except Exception:
            logging.info('time format error')
        if time_unit.lower() in ['m', 'me', 'month', 'mon']:
            pass
        elif time_unit.lower() in ['d', 'day', '1d', '1day']:
            logging.info('Adjusting time values for daily CaMa output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)
        elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
            logging.info('Adjusting time values for hourly CaMa output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
        else:
            logging.error('tim_res error')
            exit()
    return ds


def filter_CaMa(info, ds=None):
    """
    Custom filter for CaMa-Flood model output.

    Returns:
        Tuple of (metadata_dict, data_array) where metadata_dict contains
        varname and varunit updates. Returns None if ds is None.
    """
    if ds is None:
        return None

    # Default: return empty metadata and the dataset
    # CaMa filter currently has no special variable handling
    return {}, ds
