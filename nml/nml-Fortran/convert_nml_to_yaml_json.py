#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换工具：将Fortran namelist格式的配置文件转换为YAML和JSON格式

此脚本用于将OpenBench项目中的所有.nml文件转换为等效的.yaml和.json格式，
确保转换后的文件与原始文件内容完全一致。

作者：OpenBench贡献者
日期：2025年5月
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
    将字符串值解析为适当的类型。
    
    参数:
        key (str): 被解析值的键名。
        value (str): 要解析的字符串值。
        
    返回:
        Union[bool, int, float, list, str]: 解析后的值。
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
    读取Fortran namelist格式文件。
    
    参数:
        file_path (str): namelist文件的路径。
        
    返回:
        Dict[str, Dict[str, Any]]: 表示namelist结构的嵌套字典。
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
                    value = value.split('#')[0].strip()  # 移除行内注释
                    current_dict[key] = parse_nml_value(key, value)
                except ValueError:
                    # 跳过无法解析的行
                    print(f"警告: 无法解析行 '{line}' 在文件 {file_path}")
                    continue
    
    return namelist

def save_as_yaml(data: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    将数据保存为YAML格式。
    
    参数:
        data (Dict[str, Dict[str, Any]]): 要保存的数据。
        output_path (str): 输出文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def save_as_json(data: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    将数据保存为JSON格式。
    
    参数:
        data (Dict[str, Dict[str, Any]]): 要保存的数据。
        output_path (str): 输出文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_file(nml_path: str) -> None:
    """
    将单个.nml文件转换为YAML和JSON格式。
    
    参数:
        nml_path (str): .nml文件的路径。
    """
    try:
        # 读取.nml文件
        data = read_nml(nml_path)
        
        # 生成输出文件路径
        base_path = os.path.splitext(nml_path)[0]
        yaml_path = f"{base_path}.yaml"
        json_path = f"{base_path}.json"
        
        # 保存为YAML和JSON格式
        save_as_yaml(data, yaml_path)
        save_as_json(data, json_path)
        
        print(f"已转换: {nml_path} -> {yaml_path}, {json_path}")
    except Exception as e:
        print(f"转换文件 {nml_path} 时出错: {e}")

def convert_all_nml_files(root_dir: str) -> None:
    """
    转换指定目录及其子目录中的所有.nml文件。
    
    参数:
        root_dir (str): 要搜索.nml文件的根目录。
    """
    # 查找所有.nml文件
    nml_files = glob.glob(os.path.join(root_dir, "**", "*.nml"), recursive=True)
    
    if not nml_files:
        print(f"在 {root_dir} 中未找到.nml文件")
        return
    
    print(f"找到 {len(nml_files)} 个.nml文件")
    
    # 转换每个文件
    for nml_file in nml_files:
        convert_file(nml_file)
    
    print(f"转换完成。共转换了 {len(nml_files)} 个文件。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，使用它作为根目录
        root_dir = sys.argv[1]
    else:
        # 否则使用当前目录
        root_dir = os.getcwd()
    
    convert_all_nml_files(root_dir)
