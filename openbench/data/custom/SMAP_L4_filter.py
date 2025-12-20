"""
SMAP_L4 custom filter.

This filter handles variable transformations specific to SMAP L4 data.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import logging


def filter_SMAP_L4(info, ds=None):
    """
    Custom filter for SMAP L4 data.

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
            ds['Net_Radiation'] = ds['net_downward_longwave_flux'] + ds['net_downward_shortwave_flux']
            return {'varname': ['Net_Radiation'], 'varunit': 'W m-2'}, ds['Net_Radiation']
        except Exception as e:
            logging.error(f"Surface Net Radiation calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: return empty metadata and the dataset
    return {}, ds
