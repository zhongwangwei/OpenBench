#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversion Tool: Convert Fortran namelist format configuration files to YAML and JSON formats

This script is used to convert all .nml files in the OpenBench project to equivalent .yaml and .json formats,
ensuring that the converted files are completely consistent with the original file contents.

Author: Zhongwang Wei
Date: May 2025
"""

import os
import sys
import json
import yaml
import re
import glob
from typing import Dict, Any, List, Union

def parse_nml_value(key: str, value: str) -> Union[bool, int, float, list, str]:
    """
    Parse a string value into the appropriate type.
    
    Args:
        key (str): The key name of the value to be parsed.
        value (str): The string value to be parsed.
        
    Returns:
        Union[bool, int, float, list, str]: The parsed value.
    """
    value = value.strip()
    if key in ['suffix', 'prefix']:
        return value  # 对于suffix和prefix，保持字符串格式
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    elif value.replace('-', '', 1).isdigit():
        return int(value)
    elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
        return float(value)
    elif ',' in value:
        return [v.strip() for v in value.split(',')]
    else:
        return value

def read_nml(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read a Fortran namelist format file.
    
    Args:
        file_path (str): Path to the namelist file.
        
    Returns:
        Dict[str, Dict[str, Any]]: Nested dictionary representing the namelist structure.
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
            elif current_dict is not None:
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # Remove inline comments
                    current_dict[key] = parse_nml_value(key, value)
                except ValueError:
                    # Skip lines that cannot be parsed
                    print(f"Warning: Unable to parse line '{line}' in file {file_path}")
                    continue
    
    return namelist

def save_as_yaml(data: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Save data in YAML format.
    
    Args:
        data (Dict[str, Dict[str, Any]]): Data to be saved.
        output_path (str): Output file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def save_as_json(data: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Save data in JSON format.
    
    Args:
        data (Dict[str, Dict[str, Any]]): Data to be saved.
        output_path (str): Output file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_file(nml_path: str) -> None:
    """
    Convert a single .nml file to YAML and JSON formats.
    
    Args:
        nml_path (str): Path to the .nml file.
    """
    try:
        # Read .nml file
        data = read_nml(nml_path)
        
        # Generate output file paths
        base_path = os.path.splitext(nml_path)[0]
        yaml_path = f"{base_path}.yaml"
        json_path = f"{base_path}.json"
        
        # Save as YAML and JSON formats
        save_as_yaml(data, yaml_path)
        save_as_json(data, json_path)
        
        print(f"Converted: {nml_path} -> {yaml_path}, {json_path}")
    except Exception as e:
        print(f"Error converting file {nml_path}: {e}")

def convert_all_nml_files(root_dir: str) -> None:
    """
    Convert all .nml files in the specified directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search for .nml files.
    """
    # Find all .nml files
    nml_files = glob.glob(os.path.join(root_dir, "**", "*.nml"), recursive=True)
    
    if not nml_files:
        print(f"No .nml files found in {root_dir}")
        return
    
    print(f"Found {len(nml_files)} .nml files")
    
    # Convert each file
    for nml_file in nml_files:
        convert_file(nml_file)
    
    print(f"Conversion complete. {len(nml_files)} files converted in total.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a command line argument is provided, use it as the root directory
        root_dir = sys.argv[1]
    else:
        # Otherwise use the current directory
        root_dir = os.getcwd()
    
    convert_all_nml_files(root_dir)
