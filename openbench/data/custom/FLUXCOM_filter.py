"""
FLUXCOM custom filter.

This filter handles variable transformations specific to FLUXCOM data.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import logging


def filter_FLUXCOM(info, ds=None):
    """
    Custom filter for FLUXCOM data.

    Returns:
        Tuple of (metadata_dict, data_array) where metadata_dict contains
        varname and varunit updates. Returns None if ds is None.
    """
    if ds is None:
        return None

    # Get current varname/varunit from info (read-only)
    current_varname = info.ref_varname if isinstance(info.ref_varname, list) else [info.ref_varname]
    current_varunit = getattr(info, 'ref_varunit', None)

    if info.item == "Net_Radiation":
        try:
            return {'varname': ['Rn'], 'varunit': 'W m-2'}, ds['Rn']
        except Exception:
            logging.error('Net Radiation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Latent_Heat":
        try:
            return {'varname': ['LE'], 'varunit': 'W m-2'}, ds['LE']
        except Exception:
            logging.error('Latent Heat processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    if info.item == "Sensible_Heat":
        try:
            return {'varname': ['H'], 'varunit': 'W m-2'}, ds['H']
        except Exception:
            logging.error('Sensible Heat processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: return empty metadata and the dataset
    return {}, ds
