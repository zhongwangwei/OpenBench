# -*- coding: utf-8 -*-
"""
Configuration Processors for OpenBench

This module provides complex configuration processing classes that handle
OpenBench-specific data processing and analysis logic.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

import os
import logging
import re
import json
import yaml
import importlib
from typing import Dict, Any, List, Tuple, Union

# Heavy dependencies for data processing
try:
    import numpy as np
    import pandas as pd
    import xarray as xr
    from joblib import Parallel, delayed

    _HAS_DATA_LIBS = True
except ImportError:
    _HAS_DATA_LIBS = False
    logging.warning("Data processing libraries (numpy, pandas, xarray, joblib) not available")

try:
    from .readers import NamelistReader
    from .manager import ConfigurationError
except ImportError:
    from readers import NamelistReader


    class ConfigurationError(Exception):
        pass

# Import caching - CacheSystem is mandatory for data processing modules
try:
    from openbench.data.Mod_CacheSystem import cached, get_cache_manager

    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False


    # Make CacheSystem optional for config processors since they're less compute-intensive
    def cached(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


class GeneralInfoReader(NamelistReader):
    """
    Advanced configuration processor for OpenBench evaluation information.
    
    This class handles complex data processing tasks including time resolution
    normalization, station data filtering, and evaluation parameter processing.
    """

    # Class-level sets to track which warnings have already been shown
    _custom_filter_warnings_shown = set()
    _time_resolution_warning_shown = False

    def __init__(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any], ref_nml: Dict[str, Any],
                 metric_vars: list, score_vars: list, comparison_vars: list, statistic_vars: list,
                 item: str, sim_source: str, ref_source: str):
        """
        Initialize the GeneralInfoReader with configuration data.
        
        Args:
            main_nl: Main namelist containing general settings
            sim_nml: Simulation namelist
            ref_nml: Reference namelist
            metric_vars: List of metric variables
            score_vars: List of score variables
            comparison_vars: List of comparison variables
            statistic_vars: List of statistic variables
            item: The evaluation item being processed
            sim_source: The simulation data source
            ref_source: The reference data source
        """
        if not _HAS_DATA_LIBS:
            raise ConfigurationError("Data processing libraries required for GeneralInfoReader are not available")

        super().__init__()

        self.name = self.__class__.__name__

        # Frequency mapping for time resolution normalization
        self.freq_map = {
            'month': 'M', 'mon': 'M', 'monthly': 'M',
            'day': 'D', 'daily': 'D',
            'hour': 'H', 'Hour': 'H', 'hr': 'H', 'Hr': 'H', 'h': 'H', 'hourly': 'H',
            'year': 'Y', 'yr': 'Y', 'yearly': 'Y',
            'week': 'W', 'wk': 'W', 'weekly': 'W',
        }

        # Coordinate mapping for standardization
        self.coordinate_map = {
            'longitude': 'lon', 'long': 'lon', 'lon_cama': 'lon', 'lon0': 'lon', 'x': 'lon', 'X': 'lon', 'XLONG': 'lon',
            'latitude': 'lat', 'lat_cama': 'lat', 'lat0': 'lat', 'y': 'lat', 'Y': 'lat', 'XLAT': 'lat',
            'Time': 'time', 'TIME': 'time', 'XTIME': 'time', 't': 'time', 'T': 'time',
            'elevation': 'elev', 'height': 'elev', 'z': 'elev', 'Z': 'elev', 'h': 'elev', 'H': 'elev', 'ELEV': 'elev', 'HEIGHT': 'elev',
        }

        self.sim_source = sim_source
        self.ref_source = ref_source

        # Initialize processing
        self._initialize_attributes(main_nl, sim_nml, ref_nml, item, sim_source, ref_source)
        self._set_evaluation_variables(metric_vars, score_vars, comparison_vars, statistic_vars)
        self._process_data_types()
        self._process_time_resolutions()
        self._process_station_data()

        # Handle missing variable names
        if self.sim_varname is None or self.sim_varname == '':
            logging.warning(f"Warning: sim_varname is not specified in namelist. Using item name: {self.item}")
            self.sim_varname = self.item
            self.sim_nml[self.item][f'{self.sim_source}_varname'] = self.item

        if self.ref_varname is None or self.ref_varname == '':
            logging.warning(f"Warning: ref_varname is not specified in namelist. Using item name: {self.item}")
            self.ref_varname = self.item
            self.ref_nml[self.item][f'{self.ref_source}_varname'] = self.item

    def _initialize_attributes(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any],
                               ref_nml: Dict[str, Any], item: str, sim_source: str, ref_source: str):
        """Initialize class attributes from namelists."""
        self.__dict__.update(main_nl['general'])
        self.__dict__.update(ref_nml[item])
        self.__dict__.update(sim_nml[item])
        self.casedir = os.path.join(self.basedir, self.basename)
        self.sim_nml, self.ref_nml, self.item = sim_nml, ref_nml, item
        self._set_source_attributes(sim_nml, ref_nml, item, sim_source, ref_source)
        self.min_year = main_nl['general'].get('min_year', 1)  # Default to 1 if not specified

    def _set_evaluation_variables(self, metric_vars: list, score_vars: list,
                                  comparison_vars: list, statistic_vars: list):
        """Set evaluation-related variables."""
        self.metrics, self.scores = metric_vars, score_vars
        self.comparisons, self.statistics = comparison_vars, statistic_vars

    def _set_source_attributes(self, sim_nml: Dict[str, Any], ref_nml: Dict[str, Any],
                               item: str, sim_source: str, ref_source: str):
        """Set attributes specific to simulation and reference sources."""
        for source_type in ['ref', 'sim']:
            source = ref_source if source_type == 'ref' else sim_source
            nml = ref_nml if source_type == 'ref' else sim_nml
            attributes = ['data_type', 'varname', 'varunit', 'data_groupby', 'dir', 'tim_res',
                          'grid_res', 'syear', 'eyear']
            for attr in attributes:
                value = nml[item].get(f'{source}_{attr}')
                setattr(self, f'{source_type}_{attr}', str(value) if value is not None else '')

            # Handle suffix and prefix separately to ensure they're always strings
            for attr in ['suffix', 'prefix']:
                value = nml[item].get(f'{source}_{attr}')
                setattr(self, f'{source_type}_{attr}', str(value) if value is not None else '')

            # Handle station data
            if nml[item][f'{source}_data_type'] == 'stn':
                try:
                    setattr(self, f'{source_type}_fulllist', str(nml[item][f'{source}_fulllist']))
                except (KeyError, TypeError) as e:
                    try:
                        setattr(self, f'{source_type}_fulllist', str(nml['general'][f'{source}_fulllist']))
                    except (KeyError, TypeError) as e2:
                        logging.error(f'read {source_type}_fulllist namelist error: {e2}')

            # Handle uparea attributes for station data
            try:
                setattr(self, f'{source_type}_max_uparea', str(nml[item][f'{source}_max_uparea']))
                setattr(self, f'{source_type}_min_uparea', str(nml[item][f'{source}_min_uparea']))
            except (KeyError, TypeError, AttributeError):
                pass  # Optional attributes, skip if not present

    def _process_data_types(self):
        """Process and validate data types."""
        valid_types = ['grid', 'stn', 'point']
        for data_type in [self.ref_data_type, self.sim_data_type]:
            if data_type not in valid_types:
                logging.warning(f"Unknown data type: {data_type}. Valid types: {valid_types}")

    def _process_time_resolutions(self):
        """Process and normalize time resolutions."""
        # Handle special case for GRDC data with missing time resolution
        if (self.ref_source == 'GRDC' and
                self.ref_data_type == 'stn' and
                (not self.ref_tim_res or self.ref_tim_res == '')):
            # Set GRDC time resolution to match comparison resolution
            self.ref_tim_res = getattr(self, 'compare_tim_res', 'D')
            logging.info(f"GRDC reference time resolution set to comparison resolution: {self.ref_tim_res}")

        self.ref_tim_res_normalized, self.ref_freq = self._normalize_time_resolution(self.ref_tim_res)
        self.sim_tim_res_normalized, self.sim_freq = self._normalize_time_resolution(self.sim_tim_res)
        self._check_time_resolution_consistency()
        self._set_use_years()

    def _normalize_time_resolution(self, tim_res: str) -> Tuple[str, str]:
        """
        Normalize time resolution string.
        
        Args:
            tim_res: Time resolution string
            
        Returns:
            Tuple of (normalized_resolution, frequency_code)
        """
        if not tim_res:
            return '', ''

        tim_res = tim_res.lower().strip()

        # Handle numeric prefixes (e.g., "3hour" -> "3H")
        match = re.match(r'(\d+)(.+)', tim_res)
        if match:
            number, unit = match.groups()
            freq_code = self.freq_map.get(unit.lower(), unit)
            return f"{number}{freq_code}", f"{number}{freq_code}"

        # Handle standard resolutions
        freq_code = self.freq_map.get(tim_res, tim_res)
        return tim_res, freq_code

    def _check_time_resolution_consistency(self):
        """Check if time resolutions are consistent and valid."""
        if not self._is_valid_resolution(self.ref_freq) or not self._is_valid_resolution(self.sim_freq):
            logging.warning(f"Invalid time resolution detected: ref={self.ref_freq}, sim={self.sim_freq}")

        # Check if resolutions are compatible
        if self.ref_freq != self.sim_freq:
            logging.info(f"Different time resolutions: ref={self.ref_freq}, sim={self.sim_freq}")
            # Try to determine if they can be aligned
            try:
                ref_td = self._resolution_to_timedelta(self.ref_freq)
                sim_td = self._resolution_to_timedelta(self.sim_freq)
                if ref_td != sim_td:
                    # Only show warning once
                    if not GeneralInfoReader._time_resolution_warning_shown:
                        logging.warning(f"Time resolution mismatch may cause alignment issues")
                        GeneralInfoReader._time_resolution_warning_shown = True
            except (ValueError, AttributeError, TypeError) as e:
                logging.warning(f"Could not compare time resolutions: {self.ref_freq} vs {self.sim_freq} ({e})")

    def _is_valid_resolution(self, resolution: str) -> bool:
        """Check if a resolution string is valid."""
        return bool(resolution and (resolution in self.freq_map.values() or
                                    any(resolution.endswith(freq) for freq in self.freq_map.values())))

    def _resolution_to_timedelta(self, resolution: str) -> pd.Timedelta:
        """Convert resolution string to pandas Timedelta."""
        if not resolution:
            raise ValueError("Empty resolution string")

        # Handle frequency codes
        freq_map = {'H': 'hours', 'D': 'days', 'M': 'months', 'Y': 'years', 'W': 'weeks'}

        match = re.match(r'(\d*)([HDMYW])', resolution)
        if match:
            number, unit = match.groups()
            number = int(number) if number else 1
            unit_name = freq_map.get(unit, unit)

            # Handle months and years specially
            if unit == 'M':
                return pd.Timedelta(days=30 * number)  # Approximate
            elif unit == 'Y':
                return pd.Timedelta(days=365 * number)  # Approximate
            else:
                return pd.Timedelta(**{unit_name: number})

        raise ValueError(f"Cannot parse resolution: {resolution}")

    def _process_station_data(self):
        """Process station data if applicable."""

        if self.ref_data_type == 'stn' or self.sim_data_type == 'stn':
            try:
                self._read_and_merge_station_lists()
            except Exception as e:
                logging.error(f"Error processing station data: {e}")
                # Set empty dataframe as fallback
                self.stn_list = pd.DataFrame()
            self._filter_stations()

            # Also initialize stn_info for backward compatibility
            if hasattr(self, 'stn_list'):
                self.stn_info = self.stn_list
            else:
                self.stn_info = pd.DataFrame()

    @cached(key_prefix="station_lists", ttl=3600)
    def _read_and_merge_station_lists(self):
        """Read and merge station lists from reference and simulation sources.

        Note: If fulllist is empty, skip file reading and let custom filters
        populate the station list (e.g., GRDC_filter, CaMa_filter).
        """
        if self.ref_data_type == 'stn' and self.sim_data_type == 'stn':
            # Both ref and sim are station data
            # Only read if fulllist paths are provided
            if self.sim_fulllist and self.ref_fulllist:
                self.sim_stn_list = pd.read_csv(self.sim_fulllist, header=0)
                self.ref_stn_list = pd.read_csv(self.ref_fulllist, header=0)
                self._rename_station_columns()
                self.stn_list = pd.merge(self.sim_stn_list, self.ref_stn_list, how='inner', on='ID')
            else:
                # Empty fulllist - rely on custom filter to populate station list
                logging.debug("fulllist is empty, will rely on custom filter to populate station list")
                self.stn_list = pd.DataFrame()
        elif self.sim_data_type == 'stn':
            # Only sim is station data
            if self.sim_fulllist:
                self.sim_stn_list = pd.read_csv(self.sim_fulllist, header=0)
                self._rename_station_columns(sim_only=True)
                self.stn_list = self.sim_stn_list
            else:
                logging.debug("sim fulllist is empty, will rely on custom filter to populate station list")
                self.stn_list = pd.DataFrame()
        elif self.ref_data_type == 'stn':
            # Only ref is station data
            if self.ref_fulllist:
                self.ref_stn_list = pd.read_csv(self.ref_fulllist, header=0)
                self._rename_station_columns(ref_only=True)
                self.stn_list = self.ref_stn_list
                self.stn_list['use_syear'] = self.stn_list['ref_syear']
                self.stn_list['use_eyear'] = self.stn_list['ref_eyear']
                self.stn_list['Flag'] = False
            else:
                logging.debug("ref fulllist is empty, will rely on custom filter to populate station list")
                self.stn_list = pd.DataFrame()

    def _rename_station_columns(self, sim_only=False, ref_only=False):
        """Rename station columns to standard names."""
        if not ref_only and hasattr(self, 'sim_stn_list'):
            self.sim_stn_list.rename(columns={'SYEAR': 'sim_syear', 'EYEAR': 'sim_eyear',
                                              'DIR': 'sim_dir', 'LON': 'sim_lon', 'LAT': 'sim_lat'},
                                     inplace=True)
        if not sim_only and hasattr(self, 'ref_stn_list'):
            self.ref_stn_list.rename(columns={'SYEAR': 'ref_syear', 'EYEAR': 'ref_eyear',
                                              'DIR': 'ref_dir', 'LON': 'ref_lon', 'LAT': 'ref_lat'},
                                     inplace=True)

        # Also rename in combined station list if it exists
        if hasattr(self, 'stn_list') and not self.stn_list.empty:
            # Standard column mapping for coordinates and other fields
            column_mapping = {
                'id': 'ID', 'Id': 'ID', 'site_id': 'ID', 'site': 'ID',
                'longitude': 'lon', 'Longitude': 'lon', 'LON': 'lon', 'Long': 'lon',
                'latitude': 'lat', 'Latitude': 'lat', 'LAT': 'lat', 'Lat': 'lat',
                'elevation': 'elev', 'Elevation': 'elev', 'ELEV': 'elev', 'alt': 'elev', 'altitude': 'elev'
            }

            # Apply column renaming
            for old_name, new_name in column_mapping.items():
                if old_name in self.stn_list.columns:
                    self.stn_list.rename(columns={old_name: new_name}, inplace=True)

            # Handle DIR column renaming based on data types
            if 'DIR' in self.stn_list.columns:
                if self.ref_data_type == 'stn' and self.sim_data_type == 'stn':
                    # For station-to-station comparison, we need both ref_dir and sim_dir
                    # This case is more complex and handled in merge logic
                    pass
                elif self.ref_data_type == 'stn':
                    # Reference is station data, rename DIR to ref_dir
                    self.stn_list.rename(columns={'DIR': 'ref_dir'}, inplace=True)
                elif self.sim_data_type == 'stn':
                    # Simulation is station data, rename DIR to sim_dir
                    self.stn_list.rename(columns={'DIR': 'sim_dir'}, inplace=True)

    def _filter_stations(self):
        """Filter stations based on criteria."""
        if not hasattr(self, 'stn_list') or self.stn_list is None:
            self.stn_list = pd.DataFrame()

        if self.ref_source.lower() != 'grdc' and self.stn_list.empty:
            logging.warning("No station list available for filtering; attempting to generate one.")

        initial_count = len(self.stn_list)
        # Get custom filter if available
        custom_filter = self._get_custom_filter()
        if custom_filter is not None:
            logging.info(f"Applying custom filter for {self.ref_source}")
            try:
                custom_filter(self)
            except Exception as e:
                logging.error(f"Custom filter failed: {e}")
                self._apply_default_filter()
        else:
            # Apply default filters
            self._apply_default_filter()

        # Save the filtered station list
        if hasattr(self, 'stn_list') and not self.stn_list.empty:
            if len(self.stn_list) == 0:
                logging.error("No stations selected. Check filter criteria.")
                raise ValueError("No stations selected. Check filter criteria.")

            # Use ref_fulllist if set by filter, otherwise use dataset-specific filename
            if hasattr(self, 'ref_fulllist') and self.ref_fulllist:
                stn_list_path = self.ref_fulllist
            else:
                stn_list_path = f"{self.casedir}/stn_{self.ref_source}_{self.sim_source}_list.txt"
                self.ref_fulllist = stn_list_path
            self.stn_list.to_csv(stn_list_path, index=False)
            final_count = len(self.stn_list)
            logging.info(f"Station filtering: {initial_count} -> {final_count} stations")

    def _get_custom_filter(self):
        """Get custom filter function for the reference source."""
        try:
            custom_module = importlib.import_module(f"openbench.data.custom.{self.ref_source}_filter")
            return getattr(custom_module, f"filter_{self.ref_source}")
        except (ImportError, AttributeError) as e:
            # Only show warning once per ref_source
            if self.ref_source not in GeneralInfoReader._custom_filter_warnings_shown:
                logging.warning(f"Custom filter for {self.ref_source} not available. Using default filter.")
                GeneralInfoReader._custom_filter_warnings_shown.add(self.ref_source)
            return None

    def _apply_default_filter(self):
        """Apply default station filtering criteria."""
        if not hasattr(self, 'stn_list') or self.stn_list.empty:
            return

        # Get default year values with fallbacks
        default_syear = self._safe_int(self.syear, 1990)
        default_eyear = self._safe_int(self.eyear, 2020)

        # Apply default time filtering logic
        if self.ref_data_type == 'stn' and self.sim_data_type == 'stn':
            # Both are station data
            sim_years = pd.to_numeric(self.stn_list['sim_syear'], errors='coerce')
            ref_years = pd.to_numeric(self.stn_list['ref_syear'], errors='coerce')
            syear_series = pd.Series([default_syear] * len(self.stn_list))
            self.use_syear = pd.concat([sim_years, ref_years, syear_series], axis=1).max(axis=1)

            sim_eyears = pd.to_numeric(self.stn_list['sim_eyear'], errors='coerce')
            ref_eyears = pd.to_numeric(self.stn_list['ref_eyear'], errors='coerce')
            eyear_series = pd.Series([default_eyear] * len(self.stn_list))
            self.use_eyear = pd.concat([sim_eyears, ref_eyears, eyear_series], axis=1).min(axis=1)

        elif self.sim_data_type == 'stn':
            # Only sim is station data
            sim_years = pd.to_numeric(self.stn_list['sim_syear'], errors='coerce')
            ref_syear_val = self._safe_int(self.ref_syear, default_syear)
            ref_syear = pd.Series([ref_syear_val] * len(self.stn_list))
            syear_series = pd.Series([default_syear] * len(self.stn_list))
            self.use_syear = pd.concat([sim_years, ref_syear, syear_series], axis=1).max(axis=1)

            sim_eyears = pd.to_numeric(self.stn_list['sim_eyear'], errors='coerce')
            ref_eyear_val = self._safe_int(self.ref_eyear, default_eyear)
            ref_eyear = pd.Series([ref_eyear_val] * len(self.stn_list))
            eyear_series = pd.Series([default_eyear] * len(self.stn_list))
            self.use_eyear = pd.concat([sim_eyears, ref_eyear, eyear_series], axis=1).min(axis=1)

        elif self.ref_data_type == 'stn':
            # Only ref is station data
            ref_years = pd.to_numeric(self.stn_list['ref_syear'], errors='coerce')
            sim_syear_val = self._safe_int(self.sim_syear, default_syear)
            sim_syear = pd.Series([sim_syear_val] * len(self.stn_list))
            syear_series = pd.Series([default_syear] * len(self.stn_list))
            self.use_syear = pd.concat([ref_years, sim_syear, syear_series], axis=1).max(axis=1)

            ref_eyears = pd.to_numeric(self.stn_list['ref_eyear'], errors='coerce')
            sim_eyear_val = self._safe_int(self.sim_eyear, default_eyear)
            sim_eyear = pd.Series([sim_eyear_val] * len(self.stn_list))
            eyear_series = pd.Series([default_eyear] * len(self.stn_list))
            self.use_eyear = pd.concat([ref_eyears, sim_eyear, eyear_series], axis=1).min(axis=1)

        # Set the calculated years
        self.stn_list['use_syear'] = self.use_syear
        self.stn_list['use_eyear'] = self.use_eyear

        # Apply basic filtering criteria based on time range validity
        # Only select stations where the time range is valid and meaningful
        valid_time_range = (self.stn_list['use_eyear'] - self.stn_list['use_syear']) >= 0
        self.stn_list['Flag'] = valid_time_range

        # Apply geographical filters if available
        # Check for different possible longitude column names
        lon_col = None
        for col in ['lon', 'LON', 'longitude', 'Longitude']:
            if col in self.stn_list.columns:
                lon_col = col
                break

        lat_col = None
        for col in ['lat', 'LAT', 'latitude', 'Latitude']:
            if col in self.stn_list.columns:
                lat_col = col
                break

        if hasattr(self, 'min_lon') and hasattr(self, 'max_lon') and lon_col:
            lon_filter = (self.stn_list[lon_col] >= float(self.min_lon)) & (self.stn_list[lon_col] <= float(self.max_lon))
            self.stn_list['Flag'] = self.stn_list['Flag'] & lon_filter

        if hasattr(self, 'min_lat') and hasattr(self, 'max_lat') and lat_col:
            lat_filter = (self.stn_list[lat_col] >= float(self.min_lat)) & (self.stn_list[lat_col] <= float(self.max_lat))
            self.stn_list['Flag'] = self.stn_list['Flag'] & lat_filter

        # Apply minimum year criteria if available
        if hasattr(self, 'min_year') and self.min_year:
            try:
                min_year_val = int(self.min_year)
                year_filter = (self.stn_list['use_eyear'] - self.stn_list['use_syear']) >= min_year_val
                self.stn_list['Flag'] = self.stn_list['Flag'] & year_filter
            except (ValueError, AttributeError):
                pass

        # Filter by upstream area if available
        if hasattr(self, 'ref_max_uparea') and hasattr(self, 'ref_min_uparea'):
            try:
                max_uparea = float(self.ref_max_uparea) if self.ref_max_uparea else float('inf')
                min_uparea = float(self.ref_min_uparea) if self.ref_min_uparea else 0

                if 'uparea' in self.stn_list.columns:
                    uparea_filter = (self.stn_list['uparea'] >= min_uparea) & (self.stn_list['uparea'] <= max_uparea)
                    self.stn_list['Flag'] = self.stn_list['Flag'] & uparea_filter
            except (ValueError, AttributeError):
                pass

        # For grid reference data (like GLEAM4.2a) used with station simulation data,
        # apply additional validation to ensure data availability
        if self.ref_data_type == 'grid' and self.sim_data_type == 'stn':
            # Check if reference data covers the station locations and time period
            # This is more conservative than flagging all stations as True
            ref_sy = self._safe_int(self.ref_syear, 1900)
            ref_ey = self._safe_int(self.ref_eyear, 2100)
            ref_time_coverage = (
                    (self.stn_list['use_syear'] >= ref_sy) &
                    (self.stn_list['use_eyear'] <= ref_ey)
            )
            self.stn_list['Flag'] = self.stn_list['Flag'] & ref_time_coverage

        # Keep only flagged stations
        self.stn_list = self.stn_list[self.stn_list['Flag']]
        logging.info(f"Total number of stations selected: {len(self.stn_list)}")

        # Log a warning if no stations are selected
        if len(self.stn_list) == 0:
            logging.warning("No stations selected after filtering. Check filter criteria.")
            logging.warning(f"Reference data type: {self.ref_data_type}, time range: {self.ref_syear}-{self.ref_eyear}")
            logging.warning(f"Simulation data type: {self.sim_data_type}")

    def _station_filter_criteria(self, row):
        """Custom station filtering criteria."""
        # Default implementation - can be overridden
        return True

    @staticmethod
    def _safe_int(value, default=None):
        """Safely convert a value to integer, returning default if conversion fails."""
        if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _set_use_years(self):
        """Set the use years based on the evaluation timeframe."""
        # Use safe conversion with fallbacks
        ref_sy = self._safe_int(self.ref_syear)
        sim_sy = self._safe_int(self.sim_syear)
        gen_sy = self._safe_int(self.syear, 1990)

        ref_ey = self._safe_int(self.ref_eyear)
        sim_ey = self._safe_int(self.sim_eyear)
        gen_ey = self._safe_int(self.eyear, 2020)

        # Filter out None values and calculate
        syear_values = [v for v in [ref_sy, sim_sy, gen_sy] if v is not None]
        eyear_values = [v for v in [ref_ey, sim_ey, gen_ey] if v is not None]

        if syear_values:
            self.use_syear = max(syear_values)
        else:
            self.use_syear = 1990  # Default fallback

        if eyear_values:
            self.use_eyear = min(eyear_values)
        else:
            self.use_eyear = 2020  # Default fallback

    def to_dict(self):
        """Convert the instance attributes to a dictionary."""
        return self.__dict__

    def _check_station_file(self, station_info, required_var, file_path):
        """Helper function to check if required variable exists in station file."""
        try:
            with xr.open_dataset(file_path, engine='netcdf4', chunks={}) as ds:
                has_var = required_var in ds.data_vars
            return {
                'ID': station_info['ID'],
                'valid': has_var,
                'error': None if has_var else f"Required variable '{required_var}' not found"
            }
        except Exception as e:
            return {
                'ID': station_info['ID'],
                'valid': False,
                'error': f"Error reading file: {str(e)}"
            }

    @cached(key_prefix="station_batch", ttl=1800)
    def _process_station_batch(self, stations_df, required_var, dir_column, n_jobs=-1):
        """Process a batch of stations in parallel."""
        if not _HAS_DATA_LIBS:
            logging.warning("Joblib not available, processing stations sequentially")
            results = []
            for _, row in stations_df.iterrows():
                results.append(self._check_station_file(row, required_var, row[dir_column]))
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._check_station_file)(
                    row,
                    required_var,
                    row[dir_column]
                )
                for _, row in stations_df.iterrows()
            )

        # Process results and update flags
        invalid_stations = []
        for result in results:
            if not result['valid']:
                invalid_stations.append(result['ID'])
                if result['error']:
                    logging.warning(f"Warning: Station ID {result['ID']}: {result['error']}")

        return invalid_stations
