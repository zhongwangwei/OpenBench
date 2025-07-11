# -*- coding: utf-8 -*-
"""
Configuration Updaters for OpenBench

This module provides configuration updating and processing classes that handle
complex OpenBench-specific configuration logic.

Author: OpenBench Contributors
Version: 2.0
Date: July 2025
"""

import os
import logging
from typing import Dict, Any, List

try:
    from .readers import NamelistReader
    from .manager import ConfigurationError
except ImportError:
    from readers import NamelistReader
    class ConfigurationError(Exception):
        pass


class UpdateNamelist(NamelistReader):
    """
    Configuration updater for evaluation namelists.
    
    This class processes and updates configuration data for evaluation items,
    handling both reference and simulation data sources.
    """
    
    def __init__(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any], ref_nml: Dict[str, Any], evaluation_items: List[str]):
        """
        Initialize UpdateNamelist with configuration data.
        
        Args:
            main_nl: Main configuration dictionary
            sim_nml: Simulation configuration dictionary
            ref_nml: Reference configuration dictionary
            evaluation_items: List of evaluation items to process
        """
        super().__init__()
        
        # Initialize with general settings
        self.__dict__.update(main_nl['general'])
        self.__dict__.update(ref_nml)
        self.__dict__.update(sim_nml)
        
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
    
    def _set_attribute(self, nml: Dict[str, Any], evaluation_item: str, source: str, attr: str, tmp: Dict[str, Any], source_type: str):
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
    """
    Configuration updater for figure/visualization namelists.
    
    This class processes and updates figure configuration data for validation,
    comparison, and statistical analysis.
    """
    
    def __init__(self, main_nl: Dict[str, Any], fig_nml: Dict[str, Any], comparisons: List[str], statistics: List[str]):
        """
        Initialize UpdateFigNamelist with configuration data.
        
        Args:
            main_nl: Main configuration dictionary
            fig_nml: Figure configuration dictionary
            comparisons: List of comparison items to process
            statistics: List of statistical items to process
        """
        super().__init__()
        
        # Initialize with general settings
        self.__dict__.update(fig_nml)
        
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
        """Process validation items."""
        # Process validation sources
        for key in fig_nml['validation_nml'].keys():
            self._process_validation_source(fig_nml, key)
    
    def _process_comparison_item(self, fig_nml: Dict[str, Any], comparisons: List[str]):
        """Process comparison items."""
        # Process comparison sources
        for comparison in comparisons:
            self._process_comparison_source(fig_nml, comparison)
    
    def _process_statistic_item(self, fig_nml: Dict[str, Any], statistics: List[str]):
        """Process statistical items."""
        # Process statistical sources
        for statistic in statistics:
            self._process_statistic_source(fig_nml, statistic)
    
    def _process_validation_source(self, fig_nml: Dict[str, Any], key: str):
        """Process a single validation source."""
        # Read the namelist for this validation source
        tmp = self._read_source_namelist(fig_nml, key, 'Validation')
        # Initialize the evaluation item dictionary if it doesn't exist
        fig_nml['Validation'].setdefault(key[:-7], {})
        fig_nml['Validation'][key[:-7]] = tmp['general']
    
    def _process_comparison_source(self, fig_nml: Dict[str, Any], comparison: str):
        """Process a single comparison source."""
        # Read the namelist for this comparison source
        if comparison in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            tmp = self._read_source_namelist(fig_nml, f'Basic_source', 'Comparison')
            tmp['general']['key'] = comparison
        else:
            tmp = self._read_source_namelist(fig_nml, f'{comparison}_source', 'Comparison')
        # Initialize the evaluation item dictionary if it doesn't exist
        fig_nml['Comparison'].setdefault(comparison, {})
        fig_nml['Comparison'][comparison] = tmp['general']
    
    def _process_statistic_source(self, fig_nml: Dict[str, Any], statistic: str):
        """Process a single statistical source."""
        # Read the namelist for this statistical source
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