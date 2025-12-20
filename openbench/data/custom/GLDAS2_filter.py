"""
GLDAS2 custom filter.

This filter handles variable transformations specific to GLDAS2 data.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import logging


def filter_GLDAS2(info, ds=None):
    """
    Custom filter for GLDAS2 data.

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
            ds['Net_Radiation'] = ds['Swnet_tavg'] + ds['Lwnet_tavg']
            return {'varname': ['Net_Radiation'], 'varunit': 'W m-2'}, ds['Net_Radiation']
        except Exception as e:
            logging.error(f'Net_Radiation calculation processing ERROR: {e}')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Surface_Upward_SW_Radiation":
        try:
            ds['Surface_Upward_SW_Radiation'] = ds['SWdown_f_tavg'] - ds['Swnet_tavg']
            return {'varname': ['Surface_Upward_SW_Radiation'], 'varunit': 'W m-2'}, ds['Surface_Upward_SW_Radiation']
        except Exception:
            logging.error('Surface_Upward_SW_Radiation calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Surface_Upward_LW_Radiation":
        try:
            ds['Surface_Upward_LW_Radiation'] = ds['LWdown_f_tavg'] - ds['Lwnet_tavg']
            return {'varname': ['Surface_Upward_LW_Radiation'], 'varunit': 'W m-2'}, ds['Surface_Upward_LW_Radiation']
        except Exception:
            logging.error('Surface_Upward_LW_Radiation calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Total_Runoff":
        try:
            ds['Total_Runoff'] = ds['Qs_acc'] + ds['Qsb_acc'] + ds['Qsm_acc']
            return {'varname': ['Total_Runoff'], 'varunit': 'mm 3hour-1'}, ds['Total_Runoff']
        except Exception:
            logging.error('Total_Runoff calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: return empty metadata and the dataset
    return {}, ds
