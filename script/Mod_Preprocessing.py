import glob
import importlib
import logging
import os
import re
import shutil
from typing import List, Dict, Any, Tuple, Union
import sys
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from Mod_Namelist import GeneralInfoReader


class FileChecker:
    """
    A class for checking the existence of required files and directories.
    """

    def __init__(self):
        self.name = 'FileChecker'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2024'
        self.author = "Your Name"

    def check_files(self, files: Union[str, List[str]], raise_error: bool = True) -> Tuple[bool, List[str]]:
        """
        Check if specified files exist.
        
        Args:
            files: Single file path or list of file paths to check
            raise_error: If True, raises FileNotFoundError when files are missing
                        If False, returns status and list of missing files
        
        Returns:
            Tuple containing:
            - Boolean indicating if all files exist
            - List of missing files (empty if all exist)
        """
        if isinstance(files, str):
            files = [files]

        missing_files = []
        for file_path in files:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                missing_files.append(file_path)

        if missing_files and raise_error:
            logging.error(f"Required files not found: {', '.join(missing_files)}")
            raise FileNotFoundError(f"Required files not found: {', '.join(missing_files)}")

        return len(missing_files) == 0, missing_files

    def check_directories(self, directories: Union[str, List[str]], create: bool = False) -> Tuple[bool, List[str]]:
        """
        Check if specified directories exist.
        
        Args:
            directories: Single directory path or list of directory paths to check
            create: If True, creates missing directories
                   If False, only checks existence
        
        Returns:
            Tuple containing:
            - Boolean indicating if all directories exist/were created
            - List of missing directories (empty if all exist)
        """
        if isinstance(directories, str):
            directories = [directories]

        missing_dirs = []
        for dir_path in directories:
            if not os.path.exists(dir_path):
                if create:
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                        logging.info(f"Created directory: {dir_path}")
                    except Exception as e:
                        logging.error(f"Failed to create directory {dir_path}: {str(e)}")
                        missing_dirs.append(dir_path)
                else:
                    logging.error(f"Directory not found: {dir_path}")
                    missing_dirs.append(dir_path)

        return len(missing_dirs) == 0, missing_dirs

    def validate_paths(self,
                       required_files: List[str] = None,
                       required_dirs: List[str] = None,
                       create_dirs: bool = False,
                       raise_error: bool = True) -> bool:
        """
        Validate existence of multiple files and directories.
        
        Args:
            required_files: List of required file paths
            required_dirs: List of required directory paths
            create_dirs: If True, attempts to create missing directories
            raise_error: If True, raises exception on missing files
            
        Returns:
            Boolean indicating if all validations passed
        """
        all_valid = True

        if required_dirs:
            dirs_exist, missing_dirs = self.check_directories(required_dirs, create_dirs)
            if not dirs_exist:
                logging.error(f"Missing directories: {', '.join(missing_dirs)}")
                all_valid = False

        if required_files:
            files_exist, missing_files = self.check_files(required_files, raise_error)
            if not files_exist:
                all_valid = False

        return all_valid


def check_required_nml(main_nl, sim_nml=None, ref_nml=None, evaluation_items=None):
    """Check all required files before running the evaluation system."""
    file_checker = FileChecker()
    logging.info("**************************************************")
    logging.info(f"\033[1;32mStart checking required nml files\033[0m")
    # Required namelist files
    required_files = [
        main_nl["general"]["reference_nml"],
        main_nl["general"]["simulation_nml"],
        main_nl["general"]["figure_nml"]
    ]

    # Add statistics namelist if statistics is enabled
    if main_nl['general']['statistics']:
        required_files.append(main_nl["general"]["statistics_nml"])

    # Required directories
    required_dirs = [
        main_nl['general']["basedir"],
        os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'])
    ]

    try:
        # First check and create base directories
        if not file_checker.validate_paths(
                required_dirs=required_dirs,
                create_dirs=True,
                raise_error=True
        ):
            logging.error("\033[1;31mError: Failed to create required directories.\033[0m")
            sys.exit(1)

        # Then check required files
        if not file_checker.validate_paths(
                required_files=required_files,
                raise_error=True
        ):
            logging.error("\033[1;31mError: Missing required files.\033[0m")
            sys.exit(1)

        # Check reference data files if ref_nml and evaluation_items are provided
        if ref_nml and evaluation_items and sim_nml:
            for evaluation_item in evaluation_items:
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources
                # check if ref_sources is empty
                if not ref_sources:
                    logging.error("**************************************************")
                    logging.error(f"Error: {evaluation_item} has no reference sources!")
                    logging.error(f"Please add the {evaluation_item} reference sources in the reference namelist!")
                    logging.error(f"Or check the evaluation items is correct or not: {evaluation_items}")
                    logging.error("**************************************************")
                    sys.exit(1)

                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources
                # check if sim_sources is empty
                if not sim_sources:
                    logging.error("**************************************************")
                    logging.error(f"Error: {evaluation_item} has no simulation sources!")
                    logging.error(f"Please add the {evaluation_item} simulation sources in the simulation namelist!")
                    logging.error(f"Or check the evaluation items is correct or not: {evaluation_items}")
                    logging.error("**************************************************")
                    sys.exit(1)
                for source in ref_sources:
                    # Construct path to variable-specific namelist
                    var_nml_path = ref_nml['def_nml'][source]
                    if not os.path.exists(var_nml_path):
                        logging.error("**************************************************")
                        logging.error(f"Error: Variable namelist not found: {var_nml_path}")
                        logging.error("**************************************************")
                        sys.exit(1)

                for source in sim_sources:
                    # Construct path to variable-specific namelist
                    var_nml_path = sim_nml['def_nml'][source]
                    if not os.path.exists(var_nml_path):
                        logging.error("**************************************************")
                        logging.error(f"Error: Variable namelist not found: {var_nml_path}")
                        logging.error("**************************************************")
                        sys.exit(1)
        logging.info("\033[1;32mDone\033[0m")
        logging.info("**************************************************")

        return True

    except FileNotFoundError as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)


def run_files_check(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars, fig_nml):
    """Run the files check for all variables."""
    for evaluation_item in evaluation_items:
        sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
        ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
        # Ensure sources are lists
        sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources
        ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources
        # Rearrange reference sources to put station data first
        ref_sources = sorted(ref_sources, key=lambda x: 0 if ref_nml[evaluation_item].get(f'{x}_data_type') == 'stn' else 1)
        # Rearrange simulation sources to put station data first
        sim_sources = sorted(sim_sources, key=lambda x: 0 if sim_nml[evaluation_item].get(f'{x}_data_type') == 'stn' else 1)

        for ref_source in ref_sources:
            for sim_source in sim_sources:
                files_check(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item, sim_source, ref_source,
                            fig_nml)


def files_check(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item, sim_source, ref_source, fig_nml):
    """Check the files for the given variables."""
    logging.info("**************************************************")
    logging.info(f"Checking files for {evaluation_item} - ref: {ref_source} - sim: {sim_source}!")
    general_info_object = GeneralInfoReader(main_nl, sim_nml, ref_nml, metric_vars, score_vars,
                                            comparison_vars, statistic_vars, evaluation_item,
                                            sim_source, ref_source)
    general_info = general_info_object.to_dict()
    general_info['ref_source'] = ref_source
    general_info['sim_source'] = sim_source
    if isinstance(general_info['use_syear'], int):
        min_year = general_info['use_syear']
        max_year = general_info['use_eyear']
    else:
        min_year = min(general_info['use_syear'])
        max_year = max(general_info['use_eyear'])

    if general_info['ref_data_type'] == 'grid':
        if general_info['ref_data_groupby'].lower() == 'single':
            file_path = os.path.join(general_info['ref_dir'], f"{general_info['ref_prefix']}{general_info['ref_suffix']}.nc")
            if not os.path.exists(file_path):
                logging.error(f"Error: The reference file {file_path} does not exist!")
                sys.exit(1)
        elif general_info['ref_data_groupby'].lower() == 'year':
            # get min year of general_info['use_syear'] and max year of general_info['use_eyear']
            for year in range(min_year, max_year + 1):
                file_path = os.path.join(general_info['ref_dir'], f'{general_info["ref_prefix"]}{year}{general_info["ref_suffix"]}.nc')
                if not os.path.exists(file_path):
                    logging.error(f"Error: The reference file {file_path} does not exist!")
                    sys.exit(1)
        elif general_info['ref_data_groupby'].lower() == 'month':
            # get min year of general_info['use_syear'] and max year of general_info['use_eyear']
            for year in range(min_year, max_year + 1):
                file_path = os.path.join(general_info['ref_dir'], f'{general_info["ref_prefix"]}{year}*{general_info["ref_suffix"]}.nc')
                file_count = len(glob.glob(file_path))
                # check if the file_count is 12
                if file_count != 12:
                    logging.error(f"Error: The reference file {file_path} does not have 12 months!")
                    sys.exit(1)
        else:
            logging.error(f"The reference data groupby is not checked in current version!")

    elif general_info['ref_data_type'] == 'stn':
        logging.error(f"The reference data type is station!,which is not checked in current version!")
    else:
        logging.error(f"The reference data type is not supported!")
        sys.exit(1)

    if general_info['sim_data_type'] == 'grid':
        if general_info['sim_data_groupby'] == 'single':
            file_path = os.path.join(general_info['sim_dir'], f"{general_info['sim_prefix']}{general_info['sim_suffix']}.nc")
            if not os.path.exists(file_path):
                logging.error(f"Error: The simulation file {file_path} does not exist!")
                sys.exit(1)
        elif general_info['sim_data_groupby'].lower() == 'year':
            # get min year of general_info['use_syear'] and max year of general_info['use_eyear']
            # min_year = min(list(general_info['use_syear']))
            # max_year = max(list(general_info['use_eyear']))
            for year in range(min_year, max_year + 1):
                file_path = os.path.join(general_info['sim_dir'], f'{general_info["sim_prefix"]}{year}{general_info["sim_suffix"]}.nc')
                if not os.path.exists(file_path):
                    logging.error(f"Error: The simulation file {file_path} does not exist!")
                    sys.exit(1)
        elif general_info['sim_data_groupby'].lower() == 'month':
            # get min year of general_info['use_syear'] and max year of general_info['use_eyear']
            # min_year = min(list(general_info['use_syear']))
            # max_year = max(list(general_info['use_eyear']))
            for year in range(min_year, max_year + 1):
                file_path = os.path.join(general_info['sim_dir'], f'{general_info["sim_prefix"]}{year}*{general_info["sim_suffix"]}.nc')
                file_count = len(glob.glob(file_path))
                if file_count != 12:
                    logging.error(f"Error: The simulation file {file_path} does not have 12 months!")
                    sys.exit(1)
        else:
            logging.error(f"The simulation data groupby is not checked in current version!")
    elif general_info['sim_data_type'] == 'stn':
        logging.error(f"The simulation data type is station!,which is not checked in current version!")
    else:
        logging.error(f"The simulation data type is not supported!")
        sys.exit(1)

    logging.info("Done")
    logging.info("**************************************************")
