# -*- coding: utf-8 -*-
"""
Configuration Readers for OpenBench

This module provides specialized configuration readers for different file formats
and OpenBench-specific configuration processing.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

import os
import re
import json
import yaml
import logging
from typing import Dict, Any, Union

try:
    from .manager import ConfigurationError
except ImportError:
    class ConfigurationError(Exception):
        pass


class NamelistReader:
    """
    Enhanced namelist reader with support for multiple file formats.
    
    This class provides the core file reading functionality that was originally
    in Mod_Namelist.py, but optimized and integrated with the new config system.
    """
    
    def __init__(self):
        """Initialize the NamelistReader."""
        self.name = 'NamelistReader'
        self.version = '2.0'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.extended_by = "OpenBench Contributors"
    
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
    
    def _detect_file_format(self, file_path: str) -> str:
        """
        Detect the format of the configuration file based on its extension.
        
        Args:
            file_path (str): The path to the configuration file.
            
        Returns:
            str: The detected format ('nml', 'yaml', or 'json').
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.nml':
            logging.warning(
                f"\n" + "="*80 + "\n"
                f"⚠️  DEPRECATION WARNING: Fortran NML format (.nml) is deprecated!\n"
                f"    The Fortran NML format is no longer being updated.\n"
                f"    Please switch to YAML format (.yaml) for configuration files.\n"
                f"    File: {file_path}\n" + "="*80
            )
            return 'nml'
        elif ext in ('.yaml', '.yml'):
            return 'yaml'
        elif ext == '.json':
            logging.warning(
                f"\n" + "="*80 + "\n"
                f"⚠️  DEPRECATION WARNING: JSON format (.json) is deprecated!\n"
                f"    The JSON format is no longer being updated.\n"
                f"    Please switch to YAML format (.yaml) for configuration files.\n"
                f"    File: {file_path}\n" + "="*80
            )
            return 'json'
        else:
            # Default to nml for backward compatibility
            logging.warning(f"Unknown file extension: {ext}, defaulting to Fortran namelist format")
            return 'nml'
    
    def _parse_value(self, key: str, value: str) -> Union[bool, int, float, list, str]:
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
    
    def _read_nml(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a Fortran namelist format file.
        
        Args:
            file_path (str): The path to the namelist file.
            
        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary representing the namelist structure.
        """
        namelist = {}
        current_dict = None
        
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
                elif current_dict is not None and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().rstrip(',')
                    # Remove comments (everything after #)
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    current_dict[key] = self._parse_value(key, value)
        
        return namelist
    
    def _read_yaml(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a YAML format file.
        
        Args:
            file_path (str): The path to the YAML file.
            
        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary representing the YAML structure.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read YAML file {file_path}: {e}")
    
    def _read_json(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a JSON format file.
        
        Args:
            file_path (str): The path to the JSON file.
            
        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary representing the JSON structure.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Failed to parse JSON file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read JSON file {file_path}: {e}")
    
    def read_namelist(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a namelist file in any supported format and return its contents.
        
        Args:
            file_path (str): The path to the namelist file.
            
        Returns:
            Dict[str, Dict[str, Any]]: The loaded namelist data.
            
        Raises:
            ConfigurationError: If the file cannot be read or parsed.
        """
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        # Direct reading without using ConfigManager to avoid recursion
        try:
            file_format = self._detect_file_format(file_path)
            
            if file_format == 'nml':
                return self._read_nml(file_path)
            elif file_format == 'yaml':
                return self._read_yaml(file_path)
            elif file_format == 'json':
                return self._read_json(file_path)
            else:
                raise ConfigurationError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file {file_path}: {e}")
    
    def read(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Alias for read_namelist for backward compatibility.
        
        Args:
            file_path (str): The path to the configuration file.
            
        Returns:
            Dict[str, Dict[str, Any]]: The loaded configuration data.
        """
        return self.read_namelist(file_path)


# For backward compatibility, export the FortranNMLReader alias
FortranNMLReader = NamelistReader