import importlib
import os
import re
from typing import Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import logging


class NamelistReader:
    """
    A class for reading and processing namelist files.
    """

    def __init__(self):
        """
        Initialize the NamelistReader with metadata and error settings.
        """
        self.name = 'namelist_read'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        # Ignore all numpy warnings
        np.seterr(all='ignore')

    @staticmethod
    def strtobool(val: str) -> int:
        """
        Convert a string representation of truth to 1 (true) or 0 (false).

        Args:
            val (str): The string to convert.

        Returns:
            int: 1 for true values, 0 for false values.

        Raises:
            ValueError: If the input string is not a valid truth value.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            logging.error(f"Invalid truth value: {val}")
            raise ValueError(f"Invalid truth value: {val}")

    @staticmethod
    def select_variables(namelist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select variables from namelist if the value is truthy.

        Args:
            namelist (Dict[str, Any]): The namelist dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing only the truthy values.
        """
        return {k: v for k, v in namelist.items() if v}

    def read_namelist(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a namelist from a text file.

        Args:
            file_path (str): The path to the namelist file.

        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary representing the namelist structure.
        """
        namelist = {}
        current_dict = None

        def parse_value(key: str, value: str) -> Union[bool, int, float, list, str]:
            """
            Parse a string value into its appropriate type.

            Args:
                key (str): The key of the value being parsed.
                value (str): The string value to parse.

            Returns:
                Union[bool, int, float, list, str]: The parsed value.
            """
            value = value.strip()
            if key in ['suffix', 'prefix']:
                return value  # Return as string for suffix and prefix
            if value.lower() in ['true', 'false']:
                return bool(self.strtobool(value))
            elif value.replace('-', '', 1).isdigit():
                return int(value)
            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                return float(value)
            elif ',' in value:
                return [v.strip() for v in value.split(',')]
            else:
                return value

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('&'):
                    dict_name = line[1:]
                    current_dict = {}
                    namelist[dict_name] = current_dict
                elif line.startswith('/'):
                    current_dict = None
                elif current_dict is not None:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # Remove inline comments
                    current_dict[key] = parse_value(key, value)

        return namelist


class UpdateNamelist(NamelistReader):
    def __init__(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any], ref_nml: Dict[str, Any], evaluation_items: List[str]):
        # Initialize with general settings
        self.__dict__.update(main_nl['general'])
        self.__dict__.update(ref_nml)
        self.__dict__.update(sim_nml)
        # print all the attributes

        # Process each evaluation item
        for evaluation_item in evaluation_items:
            self._process_evaluation_item(evaluation_item, sim_nml, ref_nml)

    def _process_evaluation_item(self, evaluation_item: str, sim_nml: Dict[str, Any], ref_nml: Dict[str, Any]):
        """Process a single evaluation item for both reference and simulation data."""
        sim_sources = self._ensure_list(sim_nml['general'][f'{evaluation_item}_sim_source'])
        ref_sources = self._ensure_list(ref_nml['general'][f'{evaluation_item}_ref_source'])

        # Process reference sources
        for ref_source in ref_sources:
            self._process_ref_source(evaluation_item, ref_source, ref_nml)

        # Process simulation sources
        for sim_source in sim_sources:
            self._process_sim_source(evaluation_item, sim_source, sim_nml)

    @staticmethod
    def _ensure_list(value):
        """Ensure the given value is a list."""
        return [value] if isinstance(value, str) else value

    def _process_ref_source(self, evaluation_item: str, ref_source: str, ref_nml: Dict[str, Any]):
        """Process a single reference source for an evaluation item."""
        # Read the namelist for this reference source
        # print all the attributes
        tmp = self._read_source_namelist(ref_nml, evaluation_item, ref_source, 'ref')
        # Initialize the evaluation item dictionary if it doesn't exist
        ref_nml.setdefault(evaluation_item, {})
        # Process each attribute for the reference source
        attributes = [
            'data_type', 'data_groupby', 'tim_res', 'grid_res', 'syear', 'eyear', 'dir',
            'varname', 'varunit', 'suffix', 'prefix'
        ]
        for attr in attributes:
            self._set_attribute(ref_nml, evaluation_item, ref_source, attr, tmp, 'ref')
        # Special handling for station data
        if ref_nml[evaluation_item][f'{ref_source}_data_type'] == 'stn':
            self._set_attribute(ref_nml, evaluation_item, ref_source, 'fulllist', tmp, 'ref')
            try:
                ref_nml[evaluation_item][f'{ref_source}_max_uparea'] = tmp[evaluation_item]['max_uparea']
                ref_nml[evaluation_item][f'{ref_source}_min_uparea'] = tmp[evaluation_item]['min_uparea']
            except KeyError:
                pass

    def _process_sim_source(self, evaluation_item: str, sim_source: str, sim_nml: Dict[str, Any]):
        """Process a single simulation source for an evaluation item."""
        # Read the namelist for this simulation source
        tmp = self._read_source_namelist(sim_nml, evaluation_item, sim_source, 'sim')

        # Initialize the evaluation item dictionary if it doesn't exist
        sim_nml.setdefault(evaluation_item, {})

        # Process each attribute for the simulation source
        attributes = [
            'data_type', 'data_groupby', 'tim_res', 'grid_res', 'syear', 'eyear',
            'suffix', 'prefix', 'model', 'varname', 'varunit', 'dir'
        ]
        for attr in attributes:
            self._set_attribute(sim_nml, evaluation_item, sim_source, attr, tmp, 'sim')

        # Special handling for station data
        if sim_nml[evaluation_item][f'{sim_source}_data_type'] == 'stn':
            self._set_attribute(sim_nml, evaluation_item, sim_source, 'fulllist', tmp, 'sim')

    def _read_source_namelist(self, nml: Dict[str, Any], evaluation_item: str, source: str, source_type: str) -> Dict[str, Any]:
        """Read the namelist for a given source with file existence check."""
        try:
            file_path = nml[evaluation_item][f"{source}"]
        except:
            try:
                file_path = nml['def_nml'][f"{source}"]
            except KeyError:
                logging.error(f"Could not find namelist path for {source} in {evaluation_item} or def_nml")
                raise KeyError(f"Could not find namelist path for {source} in {evaluation_item} or def_nml")

        if not os.path.exists(file_path):
            logging.error(f"Namelist file not found: {file_path}")
            raise FileNotFoundError(f"Namelist file not found: {file_path}")
        if not os.path.isfile(file_path):
            logging.error(f"Expected file but found directory: {file_path}")
            raise IsADirectoryError(f"Expected file but found directory: {file_path}")
        if not os.access(file_path, os.R_OK):
            logging.error(f"No read permission for file: {file_path}")
            raise PermissionError(f"No read permission for file: {file_path}")

        return self.read_namelist(file_path)

    def _set_attribute(self, nml: Dict[str, Any], evaluation_item: str, source: str, attr: str, tmp: Dict[str, Any],
                       source_type: str):
        """Set an attribute for a source in the namelist."""
        key = f'{source}_{attr}'
        try:
            nml[evaluation_item][key] = tmp[evaluation_item][attr]
        except KeyError:
            try:
                nml[evaluation_item][key] = tmp['general'][attr]
            except KeyError:
                if attr == 'dir':
                    self._set_dir_attribute(nml, evaluation_item, source, tmp, source_type)
                elif attr in ['model', 'varname', 'varunit']:
                    self._set_model_attribute(nml, evaluation_item, source, attr, tmp)
                else:
                    logging.warning(f"Warning: {attr} is missing in namelist for {evaluation_item} - {source}")
                    nml[evaluation_item][key] = None  # Set to None if missing

    def _set_dir_attribute(self, nml: Dict[str, Any], evaluation_item: str, source: str, tmp: Dict[str, Any], source_type: str):
        """Set the directory attribute for a source with directory existence check."""
        try:
            root_dir = tmp['general']['root_dir']
            if not os.path.exists(root_dir):
                logging.error(f"Root directory not found: {root_dir}")
                raise FileNotFoundError(f"Root directory not found: {root_dir}")
            if not os.path.isdir(root_dir):
                logging.error(f"Expected directory but found file: {root_dir}")
                raise NotADirectoryError(f"Expected directory but found file: {root_dir}")

            try:
                sub_dir = tmp[evaluation_item]['sub_dir']
                full_dir = os.path.join(root_dir, sub_dir)
            except KeyError:
                full_dir = root_dir

            if not os.path.exists(full_dir):
                logging.error(f"Data directory not found: {full_dir}")
                raise FileNotFoundError(f"Data directory not found: {full_dir}")
            if not os.path.isdir(full_dir):
                logging.error(f"Expected directory but found file: {full_dir}")
                raise NotADirectoryError(f"Expected directory but found file: {full_dir}")
            if not os.access(full_dir, os.R_OK):
                logging.error(f"No read permission for directory: {full_dir}")
                raise PermissionError(f"No read permission for directory: {full_dir}")

            nml[evaluation_item][f'{source}_dir'] = full_dir
        except KeyError:
            logging.error("dir is missing in namelist")

    def _set_model_attribute(self, nml: Dict[str, Any], evaluation_item: str, source: str, attr: str, tmp: Dict[str, Any]):
        """Set model-related attributes for a simulation source with file existence check."""
        try:
            model_namelist_path = tmp['general']['model_namelist']
            if not os.path.exists(model_namelist_path):
                logging.error(f"Model namelist file not found: {model_namelist_path}")
                raise FileNotFoundError(f"Model namelist file not found: {model_namelist_path}")
            if not os.path.isfile(model_namelist_path):
                logging.error(f"Expected file but found directory: {model_namelist_path}")
                raise IsADirectoryError(f"Expected file but found directory: {model_namelist_path}")
            if not os.access(model_namelist_path, os.R_OK):
                logging.error(f"No read permission for file: {model_namelist_path}")
                raise PermissionError(f"No read permission for file: {model_namelist_path}")

            model_nml = self.read_namelist(model_namelist_path)
            try:
                nml[evaluation_item][f'{source}_{attr}'] = model_nml['general'][attr]
            except KeyError:
                try:
                    nml[evaluation_item][f'{source}_{attr}'] = model_nml[evaluation_item][attr]
                except KeyError:
                    logging.error(f"{attr} is missing in namelist")
        except KeyError:
            logging.error(f"{attr} is missing in namelist")


class UpdateFigNamelist(NamelistReader):
    def __init__(self, main_nl: Dict[str, Any], fig_nml: Dict[str, Any], comparisons: List[str], statistics: List[str]):
        # Initialize with general settings
        self.__dict__.update(fig_nml)

        # print all the attributes
        # Process each validation parameters
        fig_nml.setdefault('Validation', {})
        fig_nml.setdefault('Comparison', {})
        fig_nml.setdefault('Statistic', {})

        self._process_validation_item(fig_nml)

        if main_nl['general']['comparison']:
            self._process_comparison_item(fig_nml, comparisons)
        if main_nl['general']['statistics']:
            self._process_statistic_item(fig_nml, statistics)

    def _process_validation_item(self, fig_nml: Dict[str, Any]):
        # """Process a single evaluation item for both reference and simulation data."""

        # Process reference sources
        for key in fig_nml['validation_nml'].keys():
            self._process_validation_source(fig_nml, key)

    def _process_comparison_item(self, fig_nml: Dict[str, Any], comparisons: List[str]):
        # Process reference sources
        for comparison in comparisons:
            self._process_comparison_source(fig_nml, comparison)

    def _process_statistic_item(self, fig_nml: Dict[str, Any], statistics: List[str]):
        # Process reference sources
        for statistic in statistics:
            self._process_statistic_source(fig_nml, statistic)

    def _process_validation_source(self, fig_nml: Dict[str, Any], key: str):
        """Process a single reference source for an evaluation item."""
        # Read the namelist for this reference source
        tmp = self._read_source_namelist(fig_nml, key, 'Validation')
        # Initialize the evaluation item dictionary if it doesn't exist
        fig_nml['Validation'].setdefault(key[:-7], {})
        fig_nml['Validation'][key[:-7]] = tmp['general']

    def _process_comparison_source(self, fig_nml: Dict[str, Any], comparison: str):
        """Process a single simulation source for an evaluation item."""
        # Read the namelist for this simulation source
        if comparison in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            tmp = self._read_source_namelist(fig_nml, f'Basic_source', 'Comparison')
            tmp['general']['key'] = comparison
        else:
            tmp = self._read_source_namelist(fig_nml, f'{comparison}_source', 'Comparison')
        # Initialize the evaluation item dictionary if it doesn't exist
        fig_nml['Comparison'].setdefault(comparison, {})
        fig_nml['Comparison'][comparison] = tmp['general']

    def _process_statistic_source(self, fig_nml: Dict[str, Any], statistic: str):
        """Process a single simulation source for an evaluation item."""
        # Read the namelist for this simulation source
        if statistic in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            tmp = self._read_source_namelist(fig_nml, f'Basic_source', 'Statistic')
            tmp['general']['key'] = statistic
        else:
            tmp = self._read_source_namelist(fig_nml, f'{statistic}_source', 'Statistic')
        # Initialize the evaluation item dictionary if it doesn't exist
        fig_nml['Statistic'].setdefault(statistic, {})
        fig_nml['Statistic'][statistic] = tmp['general']

    def _read_source_namelist(self, nml: Dict[str, Any], key: str, source_type: str):
        """Read the namelist for a given source."""
        if source_type == 'Validation':
            return self.read_namelist(nml['validation_nml'][key])
        elif source_type == 'Comparison':
            return self.read_namelist(nml['comparison_nml'][key])
        else:
            return self.read_namelist(nml['statistic_nml'][key])


class GeneralInfoReader(NamelistReader):
    """
    A class for reading and processing general information from namelists for simulation and reference data.
    This class handles various data types, time resolutions, and station data processing.
    """

    def __init__(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any], ref_nml: Dict[str, Any],
                 metric_vars: list, score_vars: list, comparison_vars: list, statistic_vars: list,
                 item: str, sim_source: str, ref_source: str):
        """
        Initialize the GeneralInfoReader with configuration data and process the information.

        Args:
            main_nl (Dict[str, Any]): Main namelist containing general settings
            sim_nml (Dict[str, Any]): Simulation namelist
            ref_nml (Dict[str, Any]): Reference namelist
            metric_vars (list): List of metric variables
            score_vars (list): List of score variables
            comparison_vars (list): List of comparison variables
            statistic_vars (list): List of statistic variables
            item (str): The evaluation item being processed
            sim_source (str): The simulation data source
            ref_source (str): The reference data source
        """

        self.name = self.__class__.__name__
        self.freq_map = {
            'month': 'M',
            'mon': 'M',
            'monthly': 'M',
            'day': 'D',
            'daily': 'D',
            'hour': 'H',
            'Hour': 'H',
            'hr': 'H',
            'Hr': 'H',
            'h': 'H',
            'hourly': 'H',
            'year': 'Y',
            'yr': 'Y',
            'yearly': 'Y',
            'week': 'W',
            'wk': 'W',
            'weekly': 'W',
        }

        self.coordinate_map = {
            'longitude': 'lon',
            'long': 'lon',
            'lon_cama': 'lon',
            'lon0': 'lon',
            'x': 'lon',
            'X': 'lon',
            'XLONG': 'lon',
            'latitude': 'lat',
            'lat_cama': 'lat',
            'lat0': 'lat',
            'y': 'lat',
            'Y': 'lat',
            'XLAT': 'lat',
            'Time': 'time',
            'TIME': 'time',
            'XTIME': 'time',
            't': 'time',
            'T': 'time',
            'elevation': 'elev',
            'height': 'elev',
            'z': 'elev',
            'Z': 'elev',
            'h': 'elev',
            'H': 'elev',
            'ELEV': 'elev',
            'HEIGHT': 'elev',
        }
        self.sim_source = sim_source
        self.ref_source = ref_source  # Add this line to set ref_source

        self._initialize_attributes(main_nl, sim_nml, ref_nml, item, sim_source, ref_source)
        self._set_evaluation_variables(metric_vars, score_vars, comparison_vars, statistic_vars)
        self._process_data_types()
        self._process_time_resolutions()
        self._process_station_data()

        # if self.sim_varname is empty, then set it to item
        if self.sim_varname is None or self.sim_varname == '':
            logging.warning(f"Warning: sim_varname is not specified in namelist. Using item name: {self.item}")
            self.sim_varname = self.item
        if self.ref_varname is None or self.ref_varname == '':
            logging.warning(f"Warning: ref_varname is not specified in namelist. Using item name: {self.item}")
            self.ref_varname = self.item

    def _initialize_attributes(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any],
                               ref_nml: Dict[str, Any], item: str, sim_source: str, ref_source: str):
        """Initialize class attributes from namelists."""
        self.__dict__.update(main_nl['general'])
        self.__dict__.update(ref_nml[item])
        self.__dict__.update(sim_nml[item])
        self.casedir = os.path.join(self.basedir, self.basename)
        self.sim_nml, self.ref_nml, self.item = sim_nml, ref_nml, item
        self._set_source_attributes(sim_nml, ref_nml, item, sim_source, ref_source)
        self.min_year = main_nl['general'].get('min_year', 1)  # Default to 5 if not specified

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

            if nml[item][f'{source}_data_type'] == 'stn':
                try:
                    setattr(self, f'{source_type}_fulllist', str(nml[item][f'{source}_fulllist']))
                except:
                    try:
                        setattr(self, f'{source_type}_fulllist', str(nml['general'][f'{source}_fulllist']))
                    except:
                        logging.error(f'read {source_type}_fulllist namelist error')
            try:
                setattr(self, f'{source_type}_max_uparea', str(nml[item][f'{source}_max_uparea']))
                setattr(self, f'{source_type}_min_uparea', str(nml[item][f'{source}_min_uparea']))
            except KeyError:
                pass
        self.sim_model = str(sim_nml[item][f'{sim_source}_model'])
        setattr(self, f'{source_type}_grid_res', float(nml[item][f'{source}_grid_res']))

    def _process_data_types(self):
        """Process and evaluate data types for simulation and reference data."""
        self.ref_data_type = self.ref_data_type.lower()
        self.sim_data_type = self.sim_data_type.lower()
        if self.ref_data_type not in ['stn', 'grid'] or self.sim_data_type not in ['stn', 'grid']:
            logging.error("Invalid data type. Must be 'stn' or 'grid'.")
            raise ValueError("Invalid data type. Must be 'stn' or 'grid'.")

    def _process_time_resolutions(self):
        """Process and normalize time resolutions for comparison, reference, and simulation data."""
        if self.ref_tim_res == '':
            logging.warning("ref_tim_res is empty")
            self.ref_tim_res = self.compare_tim_res
            logging.warning(f"Warning: ref_tim_res was not specified. Using compare_tim_res: {self.compare_tim_res}")
        if self.sim_tim_res == '':
            self.sim_tim_res = self.compare_tim_res
            logging.warning(f"Warning: sim_tim_res was not specified. Using compare_tim_res: {self.compare_tim_res}")
        self.compare_tim_res, self.compare_tim_unit = self._normalize_time_resolution(self.compare_tim_res)
        self.ref_tim_res, self.ref_tim_unit = self._normalize_time_resolution(self.ref_tim_res)
        self.sim_tim_res, self.sim_tim_unit = self._normalize_time_resolution(self.sim_tim_res)
        self._check_time_resolution_consistency()

    def _normalize_time_resolution(self, tim_res: str) -> Tuple[str, str]:
        """
        Normalize the time resolution string to a standard format.

        Args:
            tim_res (str): Time resolution string (e.g., '1month', '6hr')

        Returns:
            Tuple[str, str]: Normalized time resolution and unit
        """
        if not tim_res[0].isdigit():
            tim_res = f'1{tim_res}'
        match = re.match(r'(\d+)\s*([a-zA-Z]+)', tim_res)
        if not match:
            logging.error(f"Invalid time resolution format: {tim_res}. Use '3month', '6hr', etc.")
            raise ValueError(f"Invalid time resolution format: {tim_res}. Use '3month', '6hr', etc.")
        num_value, unit = match.groups()
        num_value = int(num_value)
        unit = self.freq_map.get(unit.lower(), unit.lower())
        return f'{num_value}{unit}', unit

    def _check_time_resolution_consistency(self):
        """Check if time resolutions are consistent and appropriate for comparison."""
        resolutions = [self.compare_tim_res, self.ref_tim_res, self.sim_tim_res]
        for res in resolutions:
            if not self._is_valid_resolution(res):
                logging.error(f"Invalid time resolution: {res}")
                raise ValueError(f"Invalid time resolution: {res}")

        # Convert resolutions to comparable units
        compare_td = self._resolution_to_timedelta(self.compare_tim_res)
        ref_td = self._resolution_to_timedelta(self.ref_tim_res)
        sim_td = self._resolution_to_timedelta(self.sim_tim_res)

        if ref_td > compare_td or sim_td > compare_td:
            logging.error("Reference or simulation time resolution is larger than comparison time resolution")
            raise ValueError("Reference or simulation time resolution is larger than comparison time resolution")

    def _is_valid_resolution(self, resolution: str) -> bool:
        """Check if a given time resolution is valid."""
        match = re.match(r'(\d+)([YMWDHS])', resolution)
        return bool(match)

    def _resolution_to_timedelta(self, resolution: str) -> pd.Timedelta:
        """Convert a resolution string to a Pandas Timedelta."""
        match = re.match(r'(\d+)([YMWDHS])', resolution)
        if not match:
            logging.error(f"Invalid resolution format: {resolution}")
            raise ValueError(f"Invalid resolution format: {resolution}")

        value, unit = match.groups()
        value = int(value)

        if unit == 'Y':
            return pd.Timedelta(days=365 * value)
        elif unit == 'M':
            return pd.Timedelta(days=30 * value)  # Approximation
        else:
            return pd.Timedelta(f"{value}{unit}")

    def _process_station_data(self):
        """Process station data if applicable."""
        if self.ref_data_type == 'stn' or self.sim_data_type == 'stn':
            try:
                self._read_and_merge_station_lists()
            except:
                logging.warning(f"Warning: No station list found for {self.item}. reading station list from custom module")
                # print(f"Warning: No station list found for {self.item}. reading station list from custom module")
            self._filter_stations()
        else:
            self._set_use_years()

    def _read_and_merge_station_lists(self):
        """Read and merge station lists for reference and simulation data."""
        if self.ref_data_type == 'stn' and self.sim_data_type == 'stn':
            self.sim_stn_list = pd.read_csv(self.sim_fulllist, header=0)
            self.ref_stn_list = pd.read_csv(self.ref_fulllist, header=0)
            self._rename_station_columns()
            self.stn_list = pd.merge(self.sim_stn_list, self.ref_stn_list, how='inner', on='ID')
        elif self.sim_data_type == 'stn':
            self.sim_stn_list = pd.read_csv(self.sim_fulllist, header=0)
            self._rename_station_columns(sim_only=True)
            self.stn_list = self.sim_stn_list
        elif self.ref_data_type == 'stn':
            self.ref_stn_list = pd.read_csv(self.ref_fulllist, header=0)
            self._rename_station_columns(ref_only=True)
            self.stn_list = self.ref_stn_list
            self.stn_list['use_syear'] = self.stn_list['ref_syear']
            self.stn_list['use_eyear'] = self.stn_list['ref_eyear']
            self.stn_list['Flag'] = False

    def _rename_station_columns(self, sim_only=False, ref_only=False):
        """Rename columns in station lists for consistency."""
        if not ref_only:
            self.sim_stn_list.rename(columns={'SYEAR': 'sim_syear', 'EYEAR': 'sim_eyear',
                                              'DIR': 'sim_dir', 'LON': 'sim_lon', 'LAT': 'sim_lat'},
                                     inplace=True)
        if not sim_only:
            self.ref_stn_list.rename(columns={'SYEAR': 'ref_syear', 'EYEAR': 'ref_eyear',
                                              'DIR': 'ref_dir', 'LON': 'ref_lon', 'LAT': 'ref_lat'},
                                     inplace=True)

    def _filter_stations(self):
        """Apply filters to select appropriate stations for analysis."""
        # print the self in key value pair
        custom_filter = self._get_custom_filter()
        if custom_filter:
            custom_filter(self)
        else:
            self._apply_default_filter()
            if len(self.stn_list) == 0:
                logging.error("No stations selected. Check filter criteria.")
                raise ValueError("No stations selected. Check filter criteria.")

            self.stn_list.to_csv(f"{self.casedir}/stn_list.txt", index=False)

    def _get_custom_filter(self):
        """Attempt to get a custom filter function for the reference source."""
        try:
            custom_module = importlib.import_module(f"custom.{self.ref_source}_filter")
            return getattr(custom_module, f"filter_{self.ref_source}")
        except (ImportError, AttributeError):
            logging.warning(f"Custom filter for {self.ref_source} not available/or contains errors. Using default filter.")
            return None

    def _apply_default_filter(self):
        """Apply the default station filter based on geographical and temporal criteria."""
        if self.ref_data_type == 'stn' and self.sim_data_type == 'stn':
            # Convert years to integers before comparison
            sim_years = pd.to_numeric(self.stn_list['sim_syear'], errors='coerce')
            ref_years = pd.to_numeric(self.stn_list['ref_syear'], errors='coerce')
            syear_series = pd.Series([int(self.syear)] * len(self.stn_list))
            self.use_syear = pd.concat([sim_years, ref_years, syear_series], axis=1).max(axis=1)

            sim_eyears = pd.to_numeric(self.stn_list['sim_eyear'], errors='coerce')
            ref_eyears = pd.to_numeric(self.stn_list['ref_eyear'], errors='coerce')
            eyear_series = pd.Series([int(self.eyear)] * len(self.stn_list))
            self.use_eyear = pd.concat([sim_eyears, ref_eyears, eyear_series], axis=1).min(axis=1)

        elif self.sim_data_type == 'stn':
            sim_years = pd.to_numeric(self.stn_list['sim_syear'], errors='coerce')
            ref_syear = pd.Series([int(self.ref_syear)] * len(self.stn_list))
            syear_series = pd.Series([int(self.syear)] * len(self.stn_list))
            self.use_syear = pd.concat([sim_years, ref_syear, syear_series], axis=1).max(axis=1)

            sim_eyears = pd.to_numeric(self.stn_list['sim_eyear'], errors='coerce')
            ref_eyear = pd.Series([int(self.ref_eyear)] * len(self.stn_list))
            eyear_series = pd.Series([int(self.eyear)] * len(self.stn_list))
            self.use_eyear = pd.concat([sim_eyears, ref_eyear, eyear_series], axis=1).min(axis=1)

        elif self.ref_data_type == 'stn':
            ref_years = pd.to_numeric(self.stn_list['ref_syear'], errors='coerce')
            sim_syear = pd.Series([int(self.sim_syear)] * len(self.stn_list))
            syear_series = pd.Series([int(self.syear)] * len(self.stn_list))
            self.use_syear = pd.concat([ref_years, sim_syear, syear_series], axis=1).max(axis=1)

            ref_eyears = pd.to_numeric(self.stn_list['ref_eyear'], errors='coerce')
            sim_eyear = pd.Series([int(self.sim_eyear)] * len(self.stn_list))
            eyear_series = pd.Series([int(self.eyear)] * len(self.stn_list))
            self.use_eyear = pd.concat([ref_eyears, sim_eyear, eyear_series], axis=1).min(axis=1)

        self.stn_list['use_syear'] = self.use_syear
        self.stn_list['use_eyear'] = self.use_eyear
        self.stn_list['Flag'] = self.stn_list.apply(self._station_filter_criteria, axis=1)
        if self.sim_data_type == 'stn':
            flagged_stations = self.stn_list[self.stn_list['Flag']].copy()
            invalid_stations = self._process_station_batch(
                flagged_stations,
                self.sim_varname,
                'sim_dir'
            )
            if invalid_stations:
                self.stn_list.loc[self.stn_list['ID'].isin(invalid_stations), 'Flag'] = False
        if self.ref_data_type == 'stn':
            flagged_stations = self.stn_list[self.stn_list['Flag']].copy()
            invalid_stations = self._process_station_batch(
                flagged_stations,
                self.ref_varname,
                'ref_dir'
            )
            if invalid_stations:
                self.stn_list.loc[self.stn_list['ID'].isin(invalid_stations), 'Flag'] = False

        self.stn_list = self.stn_list[self.stn_list['Flag']]
        logging.info(f"Total number of stations selected: {len(self.stn_list)}")
        logging.info(self.stn_list)
        #self.use_syear = self.use_syear[0]
        #self.use_eyear = self.use_eyear[0]

    def _station_filter_criteria(self, row):
        """Define criteria for filtering stations."""
        lon_col = 'sim_lon' if self.sim_data_type == 'stn' else 'ref_lon'
        lat_col = 'sim_lat' if self.sim_data_type == 'stn' else 'ref_lat'
        return (
                (row['use_eyear'] - row['use_syear'] >= (self.min_year - 1.0)) and
                (self.min_lon <= row[lon_col] <= self.max_lon) and
                (self.min_lat <= row[lat_col] <= self.max_lat)
        )

    def _set_use_years(self):
        """Set the years to use for analysis when not using station data."""
        try:
            self.use_syear = max(int(self.syear), int(self.sim_syear), int(self.ref_syear))
            self.use_eyear = min(int(self.eyear), int(self.sim_eyear), int(self.ref_eyear))
        except ValueError as e:
            logging.error(f"Error converting years to integers: {e}")
            logging.error(f"syear: {self.syear}, sim_syear: {self.sim_syear}, ref_syear: {self.ref_syear}")
            logging.error(f"eyear: {self.eyear}, sim_eyear: {self.sim_eyear}, ref_eyear: {self.ref_eyear}")
            raise

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

    def _process_station_batch(self, stations_df, required_var, dir_column, n_jobs=-1):
        """Process a batch of stations in parallel."""
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
