"""
PLUMBER2S custom filter.

This filter handles variable transformations and fallback logic for PLUMBER2S data.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing.
"""
import logging


def filter_PLUMBER2S(info, ds=None):
    """
    Filter for PLUMBER2S reference data.

    Handles fallback logic for variables that may have alternative names.

    This filter is called from two different contexts:
    1. Configuration context (ds=None): Called from config/processors.py for station filtering
    2. Data processing context (ds provided): Called from Mod_DatasetProcessing.py for variable fallback

    Returns:
        Tuple of (metadata_dict, data_array) where metadata_dict contains
        varname and varunit updates. Returns None if ds is None.
    """
    # If ds is None, this is being called from the configuration context (station filtering).
    if ds is None:
        # Call default filter to handle station time range filtering
        if hasattr(info, '_apply_default_filter'):
            info._apply_default_filter()
        return None

    # Get current varname/varunit from info (read-only)
    current_varname = info.ref_varname if isinstance(info.ref_varname, list) else [info.ref_varname]
    current_varunit = getattr(info, 'ref_varunit', None)

    # Handle Sensible Heat with fallback from Qh_cor to Qh
    if info.item == "Sensible_Heat":
        try:
            # If Qh_cor exists, let default processing handle it
            if 'Qh_cor' in ds.variables:
                return {}, ds
            # Fall back to Qh only if Qh_cor is not available
            elif 'Qh' in ds.variables:
                logging.warning('Qh_cor not found, falling back to Qh for Sensible_Heat')
                return {'varname': ['Qh'], 'varunit': 'W m-2'}, ds['Qh']
            else:
                logging.error('Neither Qh_cor nor Qh found in dataset for Sensible_Heat')
                return {'varname': current_varname, 'varunit': current_varunit}, None
        except Exception as e:
            logging.error(f"Sensible_Heat processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Latent Heat with fallback from Qle_cor to Qle
    if info.item == "Latent_Heat":
        try:
            # If Qle_cor exists, let default processing handle it
            if 'Qle_cor' in ds.variables:
                return {}, ds
            # Fall back to Qle only if Qle_cor is not available
            elif 'Qle' in ds.variables:
                logging.warning('Qle_cor not found, falling back to Qle for Latent_Heat')
                return {'varname': ['Qle'], 'varunit': 'W m-2'}, ds['Qle']
            else:
                logging.error('Neither Qle_cor nor Qle found in dataset for Latent_Heat')
                return {'varname': current_varname, 'varunit': current_varunit}, None
        except Exception as e:
            logging.error(f"Latent_Heat processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: return empty metadata and the dataset (will use default processing)
    return {}, ds
