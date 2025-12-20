"""
BCC_AVIM custom filter.

This filter handles variable transformations specific to BCC_AVIM model output.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing. Instead of modifying the `info` object directly, it returns
a metadata dictionary with any varname/varunit updates.
"""
import numpy as np
import pandas as pd
import re
import logging


def adjust_time_BCC_AVIM(info, ds, syear, eyear, tim_res):
    """Adjust time values for BCC_AVIM output based on temporal resolution."""
    match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
    if match:
        # normalize time values
        num_value, time_unit = match.groups()
        num_value = int(num_value) if num_value else 1
        ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))
        if time_unit.lower() in ['m', 'me', 'month', 'mon']:
            logging.info('Adjusting time values for monthly BCC_AVIM output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(months=1)
        elif time_unit.lower() in ['d', 'day', '1d', '1day']:
            logging.info('Adjusting time values for daily BCC_AVIM output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)
        elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
            logging.info('Adjusting time values for hourly BCC_AVIM output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
    else:
        logging.error('tim_res error')
        exit()
    return ds


def filter_BCC_AVIM(info, ds=None):
    """
    Custom filter for BCC_AVIM model output.

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
                             ds['SOILLIQ'].isel(levsoi=1)) / 0.0626 / 1000.0
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

    if info.item == "Latent_Heat":
        try:
            ds['Latent_Heat'] = ds['FGEV'] + ds['FCEV'] + ds['FCTR']
            return {'varname': ['Latent_Heat'], 'varunit': 'W m-2'}, ds['Latent_Heat']
        except Exception as e:
            logging.error(f"Latent Heat calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Evapotranspiration":
        try:
            ds['Evapotranspiration'] = ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']
            return {'varname': ['Evapotranspiration'], 'varunit': 'mm s-1'}, ds['Evapotranspiration']
        except Exception as e:
            logging.error(f"Evapotranspiration calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Canopy_Interception":
        try:
            ds['Canopy_Interception'] = ds['H2OCAN']
            return {'varname': ['Canopy_Interception'], 'varunit': 'mm month-1'}, ds['Canopy_Interception']
        except Exception as e:
            logging.error(f"Canopy Interception calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Canopy_Evaporation_Canopy_Transpiration":
        try:
            ds['Canopy_Evaporation_Canopy_Transpiration'] = ds['QVEGE'] + ds['QVEGT']
            return {'varname': ['Canopy_Evaporation_Canopy_Transpiration'], 'varunit': 'mm s-1'}, ds['Canopy_Evaporation_Canopy_Transpiration']
        except Exception as e:
            logging.error(f"Canopy Evaporation and Transpiration calculation processing ERROR: {e}")
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
