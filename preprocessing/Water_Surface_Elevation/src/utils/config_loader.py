#!/usr/bin/env python3
"""
Configuration Loader for WSE Pipeline
配置加载器

支持:
- 全局配置 (global.yaml)
- 验证规则配置 (validation_rules.yaml)
- 数据集配置 (dataset_config.yaml)
- 配置合并和覆盖
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ConfigLoader:
    """配置加载器"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            base_dir: 基础目录，默认为脚本所在目录的上两级
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.config_dir = self.base_dir / "config"
        self.templates_dir = self.base_dir / "templates"

        # 缓存已加载的配置
        self._global_paths: Optional[Dict] = None
        self._validation_rules: Optional[Dict] = None

    def load_global_paths(self) -> Dict[str, Any]:
        """加载全局配置"""
        if self._global_paths is None:
            config_file = self.config_dir / "global.yaml"
            self._global_paths = self._load_yaml(config_file)
        return self._global_paths

    def load_validation_rules(self) -> Dict[str, Any]:
        """加载验证规则配置"""
        if self._validation_rules is None:
            config_file = self.config_dir / "validation_rules.yaml"
            self._validation_rules = self._load_yaml(config_file)
        return self._validation_rules

    def load_dataset_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载数据集配置并与全局配置合并

        Args:
            config_path: 数据集配置文件路径

        Returns:
            合并后的配置字典
        """
        # 加载数据集配置
        dataset_config = self._load_yaml(config_path)

        # 加载全局配置
        global_paths = self.load_global_paths()
        validation_rules = self.load_validation_rules()

        # 合并配置 (数据集配置优先级更高)
        merged = {
            'global_paths': global_paths,
            'validation_rules': validation_rules,
            'dataset': dataset_config.get('dataset', {}),
            'processing': dataset_config.get('processing', {}),
            'filters': dataset_config.get('filters', {}),
            'output': dataset_config.get('output', {}),
        }

        # 处理路径覆盖
        if 'paths' in dataset_config:
            merged['global_paths'] = self._deep_merge(
                global_paths, dataset_config['paths']
            )

        # 解析路径
        merged = self._resolve_paths(merged)

        return merged

    def create_default_config(self, source: str) -> Dict[str, Any]:
        """
        为指定数据源创建默认配置

        Args:
            source: 数据源名称 (hydroweb, cgls, icesat, hydrosat)

        Returns:
            默认配置字典
        """
        global_paths = self.load_global_paths()
        validation_rules = self.load_validation_rules()

        # 检查数据源是否有效
        valid_sources = ['hydroweb', 'cgls', 'icesat', 'icesat2', 'hydrosat']
        if source not in valid_sources:
            raise ValueError(f"无效的数据源: {source}。有效选项: {valid_sources}")

        # 获取数据源路径
        source_path = global_paths.get('data_sources', {}).get(source)
        if source_path is None:
            raise ValueError(f"数据源 {source} 的路径未配置")

        return {
            'global_paths': global_paths,
            'validation_rules': validation_rules,
            'dataset': {
                'name': f"{source.capitalize()}_{datetime.now().strftime('%Y%m%d')}",
                'source': source,
                'version': '1.0',
                'description': f'{source.capitalize()} altimetry data',
            },
            'processing': {
                'calculate_egm': True,
                'egm96_model': global_paths.get('geoid_data', {}).get('egm96_model', 'egm96-5'),
                'egm2008_model': global_paths.get('geoid_data', {}).get('egm2008_model', 'egm2008-1'),
                'cama_resolutions': global_paths.get('cama_data', {}).get('resolutions', []),
            },
            'filters': {
                'min_observations': validation_rules.get('quality', {}).get('min_observations', 10),
                'start_date': None,
                'end_date': None,
                'bbox': None,
            },
            'output': {
                'format': 'txt',
                'include_timeseries': False,
                'compress': False,
            },
        }

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """加载 YAML 文件"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        深度合并两个字典，override 优先

        Args:
            base: 基础字典
            override: 覆盖字典

        Returns:
            合并后的字典
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _resolve_paths(self, config: Dict) -> Dict:
        """解析配置中的相对路径为绝对路径"""
        # 解析输出路径
        if 'global_paths' in config and 'output' in config['global_paths']:
            output_config = config['global_paths']['output']
            if 'root' in output_config:
                root = output_config['root']
                if not os.path.isabs(root):
                    output_config['root'] = str(self.base_dir / root)
            if 'logs' in output_config:
                logs = output_config['logs']
                if not os.path.isabs(logs):
                    output_config['logs'] = str(self.base_dir / logs)

        return config

    def get_source_path(self, config: Dict, source: str) -> Optional[str]:
        """获取数据源路径"""
        return config.get('global_paths', {}).get('data_sources', {}).get(source)

    def get_cama_path(self, config: Dict, resolution: str) -> str:
        """获取 CaMa 地图路径"""
        cama_root = config.get('global_paths', {}).get('cama_data', {}).get('root', '')
        return os.path.join(cama_root, resolution)

    def get_geoid_path(self, config: Dict, model: str) -> str:
        """获取 EGM 模型文件路径"""
        geoid_root = config.get('global_paths', {}).get('geoid_data', {}).get('root', '')
        return os.path.join(geoid_root, f"{model}.pgm")


def load_config(config_path: Optional[str] = None,
                source: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：加载配置

    Args:
        config_path: 配置文件路径
        source: 数据源名称 (用于快速创建默认配置)

    Returns:
        配置字典
    """
    loader = ConfigLoader()

    if config_path:
        return loader.load_dataset_config(config_path)
    elif source:
        return loader.create_default_config(source)
    else:
        raise ValueError("必须指定 config_path 或 source")
