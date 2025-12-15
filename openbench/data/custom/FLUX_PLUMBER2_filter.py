import logging


def filter_FLUX_PLUMBER2(info, ds=None):
    """
    Filter for FLUX_PLUMBER2 reference data.

    Handles fallback logic for variables that may have alternative names.

    This filter is called from two different contexts:
    1. Configuration context (ds=None): Called from config/processors.py for station filtering
    2. Data processing context (ds provided): Called from Mod_DatasetProcessing.py for variable fallback

    Args:
        info: Information object containing variable metadata (or GeneralInfoReader in config context)
        ds: xarray Dataset containing the data (None in config context, Dataset in data context)

    Returns:
        tuple: (info, data) - Updated info object and processed data
               or (info, None) if ds is None (config context)
    """

    # If ds is None, this is being called from the configuration context (station filtering).
    # For station filtering, we need to call the default filter to properly handle time range filtering.
    if ds is None:
        # Call default filter to handle station time range filtering
        if hasattr(info, '_apply_default_filter'):
            info._apply_default_filter()
        return

    # Handle Sensible Heat with fallback from Qh_cor to Qh
    if info.item == "Sensible_Heat":
        try:
            # If Qh_cor exists, let default processing handle it
            if 'Qh_cor' in ds.variables:
                return None, None
            # Fall back to Qh only if Qh_cor is not available
            elif 'Qh' in ds.variables:
                logging.warning('Qh_cor not found, falling back to Qh for Sensible_Heat')
                info.ref_varname = ['Qh']  # Must be a list, not a string!
                info.ref_varunit = 'w m-2'
                return info, ds['Qh']
            else:
                logging.error('Neither Qh_cor nor Qh found in dataset for Sensible_Heat')
                return info, None
        except Exception as e:
            logging.error(f"Sensible_Heat processing ERROR: {e}")
            return info, None

    # Handle Latent Heat with fallback from Qle_cor to Qle
    if info.item == "Latent_Heat":
        try:
            # If Qle_cor exists, let default processing handle it
            if 'Qle_cor' in ds.variables:
                return None, None
            # Fall back to Qle only if Qle_cor is not available
            elif 'Qle' in ds.variables:
                logging.warning('Qle_cor not found, falling back to Qle for Latent_Heat')
                info.ref_varname = ['Qle']  # Must be a list, not a string!
                info.ref_varunit = 'w m-2'
                return info, ds['Qle']
            else:
                logging.error('Neither Qle_cor nor Qle found in dataset for Latent_Heat')
                return info, None
        except Exception as e:
            logging.error(f"Latent_Heat processing ERROR: {e}")
            return info, None

    # Return None for unhandled items (will use default processing)
    return None, None
