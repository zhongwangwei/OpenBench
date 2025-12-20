"""
CoLM (Common Land Model) custom filter.

This filter handles variable transformations and unit adjustments specific to CoLM output.

IMPORTANT: This filter uses the new metadata-return pattern to avoid race conditions
in parallel processing. Instead of modifying the `info` object directly, it returns
a metadata dictionary with any varname/varunit updates.

Pattern:
    return {'varname': [...], 'varunit': '...'}, data_array
"""
import numpy as np
import pandas as pd
import re
import logging
from openbench.config.readers import NamelistReader


def adjust_time_CoLM(info, ds, syear, eyear, tim_res):
    """Adjust time values for CoLM output based on temporal resolution."""
    match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
    if match:
        # normalize time values
        num_value, time_unit = match.groups()
        num_value = int(num_value) if num_value else 1
        ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))

        river_items = ['outflw', 'rivout', 'rivsto', 'rivout_inst', 'rivsto_inst',
                       'rivdph', 'rivvel', 'fldout', 'fldsto', 'flddph', 'fldfrc',
                       'fldare', 'sfcelv', 'totout', 'totsto', 'storge', 'pthflw',
                       'pthout', 'gdwsto', 'gwsto', 'gwout', 'maxsto', 'maxflw',
                       'maxdph', 'damsto', 'daminf', 'wevap', 'winfilt', 'levsto', 'levdph']

        if time_unit.lower() in ['m', 'me', 'month', 'mon']:
            # Handle river-related variables for monthly data
            if info.item in river_items:
                if getattr(info, 'debug_mode', False):
                    logging.info('Adjusting time values for monthly river data...')
                ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=15)
        elif time_unit.lower() in ['y', 'year', '1y', '1year']:
            pass
        elif time_unit.lower() in ['d', 'day', '1d', '1day']:
            if getattr(info, 'debug_mode', False):
                logging.info('Adjusting time values for daily CoLM output...')
            ds['time'] = pd.DatetimeIndex(ds['time'].values)

            # Handle river-related variables for daily data
            if info.item in river_items:
                if getattr(info, 'debug_mode', False):
                    logging.info('Adjusting time values for daily river data...')
                ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)
        elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
            pass
    else:
        logging.error('tim_res error')
        exit()
    return ds


def filter_CoLM(info, ds=None):
    """
    Custom filter for CoLM model output.

    Args:
        info: Configuration object with item, sim_varname, sim_varunit etc.
        ds: xarray Dataset to filter (None for station configuration context)

    Returns:
        Tuple of (metadata_dict, data_array) where metadata_dict contains
        varname and varunit updates. Returns None if ds is None (station config context).

    IMPORTANT: This function does NOT modify the `info` object to avoid race conditions
    in parallel processing. Instead, it returns metadata that the caller should use locally.
    """
    # If ds is None, this is being called from the configuration context (station filtering).
    if ds is None:
        # Call default filter to handle station time range filtering
        if hasattr(info, '_apply_default_filter'):
            info._apply_default_filter()
        return None

    # Get current varname/varunit from info (read-only)
    current_varname = info.sim_varname if isinstance(info.sim_varname, list) else [info.sim_varname]
    current_varunit = getattr(info, 'sim_varunit', None)

    # Handle Precipitation (first occurrence - simple case)
    if info.item == "Precipitation":
        try:
            ds['Precipitation'] = ds['f_xy_rain'] + ds['f_xy_snow']
            return {'varname': ['Precipitation'], 'varunit': 'mm s-1'}, ds['Precipitation']
        except Exception as e:
            logging.error(f"Precipitation calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Gross Primary Productivity with fallback from f_gpp to f_assim
    if info.item == "Gross_Primary_Productivity":
        metadata = {'varname': current_varname, 'varunit': current_varunit}
        if 'f_gpp' not in ds.variables and 'f_assim' in ds.variables:
            logging.warning('f_gpp not found, falling back to f_assim for Gross_Primary_Productivity')
            # Read f_assim's unit from the model namelist (not from data file)
            try:
                model_namelist_path = getattr(info, 'sim_model_namelist', None) or getattr(info, 'model_namelist', None)
                if model_namelist_path:
                    reader = NamelistReader()
                    model_nml = reader.read_namelist(model_namelist_path)
                    if 'Canopy_Assimilation_Rate' in model_nml and 'varunit' in model_nml['Canopy_Assimilation_Rate']:
                        metadata['varunit'] = model_nml['Canopy_Assimilation_Rate']['varunit']
                        logging.info(f"Updated sim_varunit to {metadata['varunit']} from namelist")
            except Exception as e:
                logging.warning(f"Could not read f_assim unit from namelist: {e}")
            # Rename f_assim to f_gpp in the dataset so downstream code works unchanged
            ds = ds.rename({'f_assim': 'f_gpp'})
        elif 'f_gpp' not in ds.variables and 'f_assim' not in ds.variables:
            logging.error('Neither f_gpp nor f_assim found in dataset for Gross_Primary_Productivity')
        # Continue to default processing at end of function

    # Handle Crop_Yield_Corn
    if info.item == "Crop_Yield_Corn":
        try:
            ds['Crop_Yield_Corn'] = (
                ((ds['f_cropprodc_rainfed_temp_corn'].fillna(0) * ds['area_rainfed_temp_corn'].fillna(0)) +
                 (ds['f_cropprodc_irrigated_temp_corn'].fillna(0) * ds['area_irrigated_temp_corn'].fillna(0)) +
                 (ds['f_cropprodc_rainfed_trop_corn'].fillna(0) * ds['area_rainfed_trop_corn'].fillna(0)) +
                 (ds['f_cropprodc_irrigated_trop_corn'].fillna(0) * ds['area_irrigated_trop_corn'].fillna(0))) *
                (10**6) * 2.5 * (10**(-6)) /
                (ds['area_rainfed_temp_corn'].fillna(0) + ds['area_irrigated_temp_corn'].fillna(0) +
                 ds['area_rainfed_trop_corn'].fillna(0) + ds['area_irrigated_trop_corn'].fillna(0)) *
                (3600. * 24. * 365.)) / 100.
            ds['Crop_Yield_Corn'] = ds['Crop_Yield_Corn'].assign_attrs(varunit='t ha-1')
            return {'varname': ['Crop_Yield_Corn'], 'varunit': 't ha-1'}, ds['Crop_Yield_Corn']
        except Exception:
            logging.error("Crop_Yield_Corn: Missing variables")
            required_vars = [
                'f_cropprodc_rainfed_temp_corn', 'area_rainfed_temp_corn',
                'f_cropprodc_irrigated_temp_corn', 'area_irrigated_temp_corn',
                'f_cropprodc_rainfed_trop_corn', 'area_rainfed_trop_corn',
                'f_cropprodc_irrigated_trop_corn', 'area_irrigated_trop_corn'
            ]
            missing_vars = [var for var in required_vars if var not in ds.variables]
            if missing_vars:
                logging.error(f"Missing variables: {', '.join(missing_vars)}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Crop_Yield_Maize
    if info.item == "Crop_Yield_Maize":
        try:
            total_area = (ds['area_rainfed_temp_corn'].fillna(0) +
                          ds['area_irrigated_temp_corn'].fillna(0) +
                          ds['area_rainfed_trop_corn'].fillna(0) +
                          ds['area_irrigated_trop_corn'].fillna(0))
            total_area = total_area.where(total_area > 0, np.nan)

            ds['Crop_Yield_Maize'] = (((ds['f_cropprodc_rainfed_temp_corn'].fillna(0) * ds['area_rainfed_temp_corn'].fillna(0)) +
                                       (ds['f_cropprodc_irrigated_temp_corn'].fillna(0) * ds['area_irrigated_temp_corn'].fillna(0)) +
                                       (ds['f_cropprodc_rainfed_trop_corn'].fillna(0) * ds['area_rainfed_trop_corn'].fillna(0)) +
                                       (ds['f_cropprodc_irrigated_trop_corn'].fillna(0) * ds['area_irrigated_trop_corn'].fillna(0))) *
                                      (10**6) * 2.5 * (10**(-6)) / total_area * (3600. * 24. * 365.)) / 100.
            ds['Crop_Yield_Maize'] = ds['Crop_Yield_Maize'].assign_attrs(varunit='t ha-1')
            return {'varname': ['Crop_Yield_Maize'], 'varunit': 't ha-1'}, ds['Crop_Yield_Maize']
        except Exception:
            logging.error("Crop_Yield_Maize: Missing variables")
            required_vars = [
                'f_cropprodc_rainfed_temp_corn', 'area_rainfed_temp_corn',
                'f_cropprodc_irrigated_temp_corn', 'area_irrigated_temp_corn',
                'f_cropprodc_rainfed_trop_corn', 'area_rainfed_trop_corn',
                'f_cropprodc_irrigated_trop_corn', 'area_irrigated_trop_corn'
            ]
            missing_vars = [var for var in required_vars if var not in ds.variables]
            if missing_vars:
                logging.error(f"Missing variables: {', '.join(missing_vars)}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Crop_Yield_Soybean
    if info.item == "Crop_Yield_Soybean":
        try:
            total_area = (ds['area_rainfed_temp_soybean'].fillna(0) +
                          ds['area_irrigated_temp_soybean'].fillna(0) +
                          ds['area_rainfed_trop_soybean'].fillna(0) +
                          ds['area_irrigated_trop_soybean'].fillna(0))
            total_area = total_area.where(total_area > 0, np.nan)

            ds['Crop_Yield_Soybean'] = (((ds['f_cropprodc_rainfed_temp_soybean'].fillna(0) * ds['area_rainfed_temp_soybean'].fillna(0)) +
                                         (ds['f_cropprodc_irrigated_temp_soybean'].fillna(0) * ds['area_irrigated_temp_soybean'].fillna(0)) +
                                         (ds['f_cropprodc_rainfed_trop_soybean'].fillna(0) * ds['area_rainfed_trop_soybean'].fillna(0)) +
                                         (ds['f_cropprodc_irrigated_trop_soybean'].fillna(0) * ds['area_irrigated_trop_soybean'].fillna(0))) *
                                        (10**6) * 2.5 * (10**(-6)) / total_area * (3600. * 24. * 365.)) / 100.
            ds['Crop_Yield_Soybean'] = ds['Crop_Yield_Soybean'].assign_attrs(varunit='t ha-1')
            return {'varname': ['Crop_Yield_Soybean'], 'varunit': 't ha-1'}, ds['Crop_Yield_Soybean']
        except Exception:
            logging.error("Crop_Yield_Soybean: Missing variables")
            required_vars = [
                'f_cropprodc_rainfed_temp_soybean', 'area_rainfed_temp_soybean',
                'f_cropprodc_irrigated_temp_soybean', 'area_irrigated_temp_soybean',
                'f_cropprodc_rainfed_trop_soybean', 'area_rainfed_trop_soybean',
                'f_cropprodc_irrigated_trop_soybean', 'area_irrigated_trop_soybean'
            ]
            missing_vars = [var for var in required_vars if var not in ds.variables]
            if missing_vars:
                logging.error(f"Missing variables: {', '.join(missing_vars)}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Crop_Yield_Rice
    if info.item == "Crop_Yield_Rice":
        try:
            total_area = ds['area_rainfed_rice'] + ds['area_irrigated_rice']
            total_area = total_area.where(total_area > 0, np.nan)

            ds['Crop_Yield_Rice'] = (((ds['f_cropprodc_rainfed_rice'] * ds['area_rainfed_rice']) +
                                      (ds['f_cropprodc_irrigated_rice'] * ds['area_irrigated_rice'])) *
                                     (10**6) * 2.5 * (10**(-6)) / total_area * (3600. * 24. * 365.)) / 100.
            ds['Crop_Yield_Rice'] = ds['Crop_Yield_Rice'].assign_attrs(varunit='t ha-1')
            return {'varname': ['Crop_Yield_Rice'], 'varunit': 't ha-1'}, ds['Crop_Yield_Rice']
        except Exception:
            logging.error("Crop_Yield_Rice: Missing variables")
            required_vars = [
                'f_cropprodc_rainfed_rice', 'area_rainfed_rice',
                'f_cropprodc_irrigated_rice', 'area_irrigated_rice'
            ]
            missing_vars = [var for var in required_vars if var not in ds.variables]
            if missing_vars:
                logging.error(f"Missing variables: {', '.join(missing_vars)}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Crop_Yield_Wheat
    if info.item == "Crop_Yield_Wheat":
        try:
            total_area = (ds['area_rainfed_spwheat'].fillna(0) +
                          ds['area_irrigated_spwheat'].fillna(0) +
                          ds['area_rainfed_wtwheat'].fillna(0) +
                          ds['area_irrigated_wtwheat'].fillna(0))
            total_area = total_area.where(total_area > 0, np.nan)

            ds['Crop_Yield_Wheat'] = (((ds['f_cropprodc_rainfed_spwheat'].fillna(0) * ds['area_rainfed_spwheat'].fillna(0)) +
                                       (ds['f_cropprodc_irrigated_spwheat'].fillna(0) * ds['area_irrigated_spwheat'].fillna(0)) +
                                       (ds['f_cropprodc_rainfed_wtwheat'].fillna(0) * ds['area_rainfed_wtwheat'].fillna(0)) +
                                       (ds['f_cropprodc_irrigated_wtwheat'].fillna(0) * ds['area_irrigated_wtwheat'].fillna(0))) *
                                      (10**6) * 2.5 * (10**(-6)) / total_area * (3600. * 24. * 365.)) / 100.
            ds['Crop_Yield_Wheat'] = ds['Crop_Yield_Wheat'].assign_attrs(varunit='t ha-1')
            return {'varname': ['Crop_Yield_Wheat'], 'varunit': 't ha-1'}, ds['Crop_Yield_Wheat']
        except Exception:
            logging.error("Crop_Yield_Wheat: Missing variables")
            required_vars = [
                'f_cropprodc_rainfed_spwheat', 'area_rainfed_spwheat',
                'f_cropprodc_irrigated_spwheat', 'area_irrigated_spwheat',
                'f_cropprodc_rainfed_wtwheat', 'area_rainfed_wtwheat',
                'f_cropprodc_irrigated_wtwheat', 'area_irrigated_wtwheat'
            ]
            missing_vars = [var for var in required_vars if var not in ds.variables]
            if missing_vars:
                logging.error(f"Missing variables: {', '.join(missing_vars)}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Canopy_Interception
    if info.item == "Canopy_Interception":
        try:
            ds['Canopy_Interception'] = ds['f_fevpl'] - ds['f_etr']
            return {'varname': ['Canopy_Interception'], 'varunit': 'mm s-1'}, ds['Canopy_Interception']
        except Exception:
            logging.error('Canopy interception evaporation calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Surface_Net_SW_Radiation
    if info.item == "Surface_Net_SW_Radiation":
        try:
            ds['Surface_Net_SW_Radiation'] = ds['f_xy_solarin'] - ds['f_sr']
            return {'varname': ['Surface_Net_SW_Radiation'], 'varunit': 'W m-2'}, ds['Surface_Net_SW_Radiation']
        except Exception:
            logging.error('Surface Net SW Radiation calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Surface_Net_LW_Radiation
    if info.item == "Surface_Net_LW_Radiation":
        try:
            ds['Surface_Net_LW_Radiation'] = ds['f_xy_frl'] - ds['f_olrg']
            return {'varname': ['Surface_Net_LW_Radiation'], 'varunit': 'W m-2'}, ds['Surface_Net_LW_Radiation']
        except Exception:
            logging.error('Surface Net LW Radiation calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Surface_Soil_Moisture
    if info.item == "Surface_Soil_Moisture":
        try:
            try:
                ds['f_wliq_soisno'] = (ds['f_wliq_soisno'].isel(soilsnow=5) +
                                       ds['f_wliq_soisno'].isel(soilsnow=6)) / 0.0626 / 1000.0
            except Exception:
                ds['f_wliq_soisno'] = (ds['f_wliq_soisno'].isel(soil_snow_lev=5) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=6)) / 0.0626 / 1000.0
            return {'varname': ['f_wliq_soisno'], 'varunit': 'unitless'}, ds['f_wliq_soisno']
        except Exception as e:
            logging.error(f"Surface soil moisture calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Root_Zone_Soil_Moisture
    if info.item == "Root_Zone_Soil_Moisture":
        try:
            try:
                ds['f_wliq_soisno'] = (ds['f_wliq_soisno'].isel(soilsnow=5) +
                                       ds['f_wliq_soisno'].isel(soilsnow=6) +
                                       ds['f_wliq_soisno'].isel(soilsnow=7) +
                                       ds['f_wliq_soisno'].isel(soilsnow=8) +
                                       ds['f_wliq_soisno'].isel(soilsnow=9) +
                                       ds['f_wliq_soisno'].isel(soilsnow=10) +
                                       ds['f_wliq_soisno'].isel(soilsnow=11) +
                                       ds['f_wliq_soisno'].isel(soilsnow=12) * 0.31) / 1000.0
            except Exception:
                ds['f_wliq_soisno'] = (ds['f_wliq_soisno'].isel(soil_snow_lev=5) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=6) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=7) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=8) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=9) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=10) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=11) +
                                       ds['f_wliq_soisno'].isel(soil_snow_lev=12) * 0.31) / 1000.0
            return {'varname': ['f_wliq_soisno'], 'varunit': 'unitless'}, ds['f_wliq_soisno']
        except Exception as e:
            logging.error(f"Root zone soil moisture calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Surface_Albedo
    if info.item == "Surface_Albedo":
        try:
            ds['Surface_Albedo'] = ds['f_sr'] / ds['f_xy_solarin']
            return {'varname': ['Surface_Albedo'], 'varunit': 'unitless'}, ds['Surface_Albedo']
        except Exception as e:
            logging.error(f"Surface Albedo calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Surface_Wind_Speed
    if info.item == "Surface_Wind_Speed":
        try:
            ds['Surface_Wind_Speed'] = (ds['f_us10m']**2 + ds['f_vs10m']**2)**0.5
            return {'varname': ['Surface_Wind_Speed'], 'varunit': 'm s-1'}, ds['Surface_Wind_Speed']
        except Exception as e:
            logging.error(f"Surface Wind Speed calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Urban_Anthropogenic_Heat_Flux
    if info.item == "Urban_Anthropogenic_Heat_Flux":
        try:
            ds['Urban_Anthropogenic_Heat_Flux'] = (ds['f_fhac'] + ds['f_fach'] + ds['f_fhah'] +
                                                   ds['f_fvehc'] + ds['f_fmeta'] + ds['f_fwst'])
            return {'varname': ['Urban_Anthropogenic_Heat_Flux'], 'varunit': 'W m-2'}, ds['Urban_Anthropogenic_Heat_Flux']
        except Exception as e:
            logging.error(f"Urban Anthropogenic Heat Flux calculation processing ERROR: {e}")
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Handle Terrestrial_Water_Storage_Change
    if info.item == "Terrestrial_Water_Storage_Change":
        try:
            ds['f_wat'] = ds['f_wat'].fillna(0)
            ds['f_wa'] = ds['f_wa'].fillna(0)
            ds['f_wdsrf'] = ds['f_wdsrf'].fillna(0)
            ds['f_wetwat'] = ds['f_wetwat'].fillna(0)
            TWS = ds['f_wat'] + ds['f_wa'] + ds['f_wdsrf'] + ds['f_wetwat']
            ds['Terrestrial_Water_Storage_Change'] = TWS.copy()
            return {'varname': ['Terrestrial_Water_Storage_Change'], 'varunit': 'mm'}, ds['Terrestrial_Water_Storage_Change']
        except Exception:
            logging.error('Terrestrial Water Storage Change calculation processing ERROR!')
            return {'varname': current_varname, 'varunit': current_varunit}, None

    # Default: extract the variable from dataset and return
    varname = current_varname
    if varname[0] in ds.variables:
        # Return empty metadata dict - no changes needed
        return {}, ds[varname[0]]
    else:
        # Variable not in dataset, raise exception to trigger default processing
        raise KeyError(f"Variable {varname[0]} not found in dataset")
