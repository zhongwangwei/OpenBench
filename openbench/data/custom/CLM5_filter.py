"""
CLM5 (Community Land Model 5) custom filter.

This filter handles variable transformations specific to CLM5 output.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import numpy as np
import pandas as pd
import re
import logging


def adjust_time_CLM5(info, ds, syear, eyear, tim_res):
    """Adjust time values for CLM5 output based on temporal resolution."""
    match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
    if match:
        num_value, time_unit = match.groups()
        num_value = int(num_value) if num_value else 1
        ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))
        if time_unit.lower() in ['m', 'me', 'month', 'mon']:
            logging.info('Adjusting time values for monthly CLM5 output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(months=1)
        elif time_unit.lower() in ['d', 'day', '1d', '1day']:
            logging.info('Adjusting time values for daily CLM5 output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)
        elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
            logging.info('Adjusting time values for hourly CLM5 output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
    else:
        logging.error('tim_res error')
        exit()
    return ds


def filter_CLM5(info, ds=None):
    """
    Custom filter for CLM5 model output.

    Returns:
        Tuple of (metadata_dict, data_array) where metadata_dict contains
        varname and varunit updates. Returns None if ds is None.
    """
    if ds is None:
        return None

    # Get current varname/varunit from info (read-only)
    current_varname = info.sim_varname if isinstance(info.sim_varname, list) else [info.sim_varname]
    current_varunit = getattr(info, 'sim_varunit', None)

    if info.item == "Net_Radiation":
        try:
            ds['Net_Radiation'] = ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE']
            return {'varname': ['Net_Radiation'], 'varunit': 'W m-2'}, ds['Net_Radiation']
        except Exception as e:
            logging.error(f"Surface Net Radiation calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Surface_Net_LW_Radiation":
        try:
            ds['FIRA'] = -ds['FIRA']
            return {'varname': ['FIRA'], 'varunit': 'W m-2'}, ds['FIRA']
        except Exception:
            logging.error('Surface Net LW Radiation calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Surface_Soil_Moisture":
        try:
            ds['SOILLIQ'] = (ds['SOILLIQ'].isel(levsoi=0) +
                             ds['SOILLIQ'].isel(levsoi=1)) / 0.06 / 1000.0
            return {'varname': ['SOILLIQ'], 'varunit': 'unitless'}, ds['SOILLIQ']
        except Exception as e:
            logging.error(f"Surface soil moisture calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Root_Zone_Soil_Moisture":
        try:
            ds['SOILLIQ'] = (ds['SOILLIQ'].isel(levsoi=0) +
                             ds['SOILLIQ'].isel(levsoi=1) +
                             ds['SOILLIQ'].isel(levsoi=2) +
                             ds['SOILLIQ'].isel(levsoi=3) +
                             ds['SOILLIQ'].isel(levsoi=4) +
                             ds['SOILLIQ'].isel(levsoi=5) +
                             ds['SOILLIQ'].isel(levsoi=6) +
                             ds['SOILLIQ'].isel(levsoi=7) +
                             ds['SOILLIQ'].isel(levsoi=8) * 0.29) / 1000.0
            return {'varname': ['SOILLIQ'], 'varunit': 'unitless'}, ds['SOILLIQ']
        except Exception as e:
            logging.error(f"Root zone soil moisture calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Root_Zone_Soil_Temperature":
        try:
            ds['SOILLIQ'] = (ds['SOILLIQ'].isel(levsoi=0) +
                             ds['SOILLIQ'].isel(levsoi=1) +
                             ds['SOILLIQ'].isel(levsoi=2) +
                             ds['SOILLIQ'].isel(levsoi=3) +
                             ds['SOILLIQ'].isel(levsoi=4) +
                             ds['SOILLIQ'].isel(levsoi=5) +
                             ds['SOILLIQ'].isel(levsoi=6) +
                             ds['SOILLIQ'].isel(levsoi=7) +
                             ds['SOILLIQ'].isel(levsoi=8) * 0.29) / 1000.0
            return {'varname': ['SOILLIQ'], 'varunit': 'unitless'}, ds['SOILLIQ']
        except Exception as e:
            logging.error(f"Root zone soil temperature calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Ground_Heat":
        try:
            ds['Ground_Heat'] = ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE'] - ds['FSH'] - ds['EFLX_LH_TOT']
            return {'varname': ['Ground_Heat'], 'varunit': 'W m-2'}, ds['Ground_Heat']
        except Exception as e:
            logging.error(f"Ground Heat calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Albedo":
        try:
            ds['Albedo'] = ds['FSR'] / ds['FSDS']
            return {'varname': ['Albedo'], 'varunit': 'unitless'}, ds['Albedo']
        except Exception as e:
            logging.error(f"Surface Albedo calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: return empty metadata and the dataset
    return {}, ds
