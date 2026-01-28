#!/usr/bin/env python3
"""
Step 4: Merge to NetCDF (Interface Only)
合并到 NetCDF 文件 (仅接口)

当前仅保存为文本格式。
NetCDF 合并功能预留给未来实现。
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..readers import StationMetadata
from ..steps.step2_cama import CamaResult, format_allocation_output
from ..core.station import Station, StationList
from ..utils.logger import get_logger


def run_merge(cama_result: CamaResult,
              config: Dict[str, Any],
              logger=None) -> str:
    """
    保存处理结果

    当前实现: 保存为文本文件
    预留接口: NetCDF 合并

    Args:
        cama_result: CaMa 分配结果 (来自 step2/step3)
        config: 配置字典
        logger: 日志记录器

    Returns:
        输出文件路径
    """
    log = lambda level, msg: logger and getattr(logger, level)(msg)

    # 获取配置
    output_config = config['global_paths'].get('output', {})
    processing = config.get('processing', {})
    dataset_config = config.get('dataset', {})

    output_root = output_config.get('root', './output')
    output_format = config.get('output', {}).get('format', 'txt')

    # 确保输出目录存在
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成输出文件名
    source = dataset_config.get('source', 'unknown')
    date_str = datetime.now().strftime('%Y%m%d')
    filename_template = output_config.get('station_list', 'altimetry_{source}_{date}.txt')
    filename = filename_template.format(source=source, date=date_str)
    output_file = output_dir / filename

    # 获取分辨率列表
    resolutions = processing.get('cama_resolutions', [])

    # 格式化输出数据
    output_data = format_allocation_output(
        cama_result.stations,
        cama_result.allocations,
        resolutions
    )

    # 保存文件
    if output_format in ['txt', 'both']:
        save_text_file(output_data, output_file, resolutions, logger)
        log('info', f"已保存文本文件: {output_file}")

    if output_format in ['netcdf', 'both']:
        nc_file = output_file.with_suffix('.nc')
        log('warning', f"NetCDF 输出尚未实现，跳过: {nc_file}")
        # save_netcdf_file(output_data, nc_file, config, logger)

    return str(output_file)


def save_text_file(data: List[Dict[str, Any]],
                   output_file: Path,
                   resolutions: List[str],
                   logger=None):
    """
    保存为文本文件

    Args:
        data: 格式化的输出数据
        output_file: 输出文件路径
        resolutions: 分辨率列表
        logger: 日志记录器
    """
    if not data:
        return

    # 构建列名
    base_columns = ['ID', 'station', 'dataname', 'lon', 'lat', 'satellite', 'elevation']

    res_columns = []
    for res in resolutions:
        suffix = res.replace('glb_', '').replace('min', 'min')
        for col in ['flag', 'kx1', 'ky1', 'kx2', 'ky2', 'dist1', 'dist2',
                    'rivwth', 'ix', 'iy', 'lon_cama', 'lat_cama']:
            res_columns.append(f'{col}_{suffix}')

    egm_columns = ['EGM08', 'EGM96']

    all_columns = base_columns + res_columns + egm_columns

    # 写入文件
    with open(output_file, 'w') as f:
        # 写入标题行
        header = ' '.join(f'{col:>15}' for col in all_columns)
        f.write(header + '\n')

        # 写入数据行
        for row in data:
            values = []
            for col in all_columns:
                val = row.get(col, -9999)
                if isinstance(val, float):
                    values.append(f'{val:15.4f}')
                elif isinstance(val, int):
                    values.append(f'{val:15d}')
                else:
                    values.append(f'{str(val):>15}')
            f.write(' '.join(values) + '\n')


def save_netcdf_file(data: List[Dict[str, Any]],
                     output_file: Path,
                     config: Dict[str, Any],
                     logger=None):
    """
    保存为 NetCDF 文件

    TODO: 未实现

    Args:
        data: 格式化的输出数据
        output_file: 输出文件路径
        config: 配置字典
        logger: 日志记录器
    """
    raise NotImplementedError(
        "NetCDF 输出功能尚未实现。\n"
        "当前请使用文本格式输出。"
    )


def merge_to_existing_netcdf(new_data: List[Dict[str, Any]],
                             existing_file: Path,
                             config: Dict[str, Any],
                             logger=None):
    """
    合并到已有的 NetCDF 文件

    TODO: 未实现

    Args:
        new_data: 新数据
        existing_file: 已有的 NetCDF 文件
        config: 配置字典
        logger: 日志记录器
    """
    raise NotImplementedError(
        "NetCDF 合并功能尚未实现。\n"
        "预留接口供未来实现。"
    )


class Step4Merge:
    """Step 4: Merge and output results."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__)

    def run(self, stations: StationList, merge: bool = False) -> List[str]:
        """
        Run merge/output step.

        Args:
            stations: StationList from Step 3
            merge: Whether to merge with existing files

        Returns:
            List of output file paths
        """
        self.logger.info(f"Step 4: 输出 {len(stations)} 站点")

        output_config = self.config.get('global_paths', {}).get('output', {})
        output_root = Path(output_config.get('root', './output'))
        output_root.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        date_str = datetime.now().strftime('%Y%m%d')
        sources = stations.get_sources()
        source_str = '_'.join(sorted(sources)) if sources else 'unknown'
        filename = f"altimetry_{source_str}_{date_str}.txt"
        output_file = output_root / filename

        # Write output
        with open(output_file, 'w') as f:
            # Header
            f.write("# WSE Pipeline Output\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Stations: {len(stations)}\n")
            f.write("#\n")
            f.write("ID,name,source,lon,lat,elevation,egm08,egm96\n")

            # Data
            for station in stations:
                egm08 = station.egm08 if station.egm08 is not None else -9999
                egm96 = station.egm96 if station.egm96 is not None else -9999
                f.write(f"{station.id},{station.name},{station.source},"
                        f"{station.lon:.6f},{station.lat:.6f},"
                        f"{station.elevation:.2f},{egm08:.4f},{egm96:.4f}\n")

        self.logger.info(f"输出文件: {output_file}")
        return [str(output_file)]
