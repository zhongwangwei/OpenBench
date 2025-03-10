import numpy as np
import os
import glob
import pandas as pd
import importlib
import re
import sys
from typing import Dict, Any, Tuple, List, Union

import streamlit
import xarray as xr


class NamelistReader():
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

    def Update_namelist(self, nml: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        source = self.read_namelist(nml['general'][f"model_namelist"])
        for key in nml.keys():
            if key != 'general':
                try:
                    for source_key, source_value in source[key].items():
                        nml[key][source_key] = source_value
                except:
                    pass
        return nml

