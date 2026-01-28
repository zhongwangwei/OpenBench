#!/usr/bin/env python3
"""
HydroWeb Data Reader for WSE Pipeline
HydroWeb 数据读取器

文件格式:
- 每个站点一个 txt 文件 (hydroprd_*.txt)
- 头部 33 行以 # 开头的元数据
- 数据部分: 日期 时间 高程 不确定度 ...
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base_reader import BaseReader, create_station_from_reader
from ..core.station import Station


class HydroWebReader(BaseReader):
    """HydroWeb 数据读取器"""

    source_name = "hydroweb"
    file_pattern = "hydroprd_*.txt"

    # 头部元数据行数
    HEADER_LINES = 33

    # 元数据键映射
    METADATA_KEYS = {
        'BASIN': 'basin',
        'RIVER': 'river',
        'ID': 'id',
        'REFERENCE LONGITUDE': 'lon',
        'REFERENCE LATITUDE': 'lat',
        'GEOID MODEL': 'geoid_model',
        'GEOID ONDULATION AT REF POSITION(M.mm)': 'geoid_undulation',
        'MISSION(S)-TRACK(S)': 'missions',
        'STATUS': 'status',
        'MEAN ALTITUDE(M.mm)': 'mean_elevation',
        'NUMBER OF MEASUREMENTS IN DATASET': 'num_observations',
        'FIRST DATE IN DATASET': 'start_date',
        'LAST DATE IN DATASET': 'end_date',
        'COUNTRY': 'country',
        'APPROX. WIDTH OF REACH (m)': 'river_width',
    }

    def scan_directory(self, path: str) -> List[str]:
        """扫描目录获取所有 HydroWeb 文件"""
        pattern = os.path.join(path, "**", self.file_pattern)
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def read_station(self, filepath: str) -> Optional[Station]:
        """读取单个 HydroWeb 站点文件"""
        try:
            metadata = self._parse_header(filepath)
            if not metadata:
                return None

            # 从文件名提取站点名称
            filename = Path(filepath).name
            station_name = filename.replace('hydroprd_', '').replace('_exp.txt', '').replace('.txt', '')

            # 解析卫星信息
            missions = metadata.get('missions', '')
            satellite = missions.split('-')[0] if missions else 'Unknown'

            # 解析日期
            start_date = self._parse_date(metadata.get('start_date'))
            end_date = self._parse_date(metadata.get('end_date'))

            # 创建站点对象
            station = create_station_from_reader(
                id=metadata.get('id', '').strip(),
                station_name=station_name,
                lon=float(metadata.get('lon', 0)),
                lat=float(metadata.get('lat', 0)),
                river=metadata.get('river'),
                basin=metadata.get('basin'),
                country=metadata.get('country'),
                satellite=satellite,
                start_date=start_date,
                end_date=end_date,
                num_observations=int(metadata.get('num_observations', 0)),
                mean_elevation=self._parse_float(metadata.get('mean_elevation')),
                source=self.source_name,
                filepath=filepath,
                extra={
                    'geoid_model': metadata.get('geoid_model'),
                    'geoid_undulation': self._parse_float(metadata.get('geoid_undulation')),
                    'river_width': self._parse_float(metadata.get('river_width')),
                    'status': metadata.get('status'),
                    'missions': missions,
                }
            )

            return station

        except Exception as e:
            self.log('warning', f"解析 HydroWeb 文件失败 {filepath}: {e}")
            return None

    def _parse_header(self, filepath: str) -> Optional[Dict[str, str]]:
        """解析文件头部元数据"""
        metadata = {}

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.startswith('#'):
                        break

                    if '::' in line:
                        # 格式: #KEY:: value
                        key, value = line[1:].split('::', 1)
                        key = key.strip()
                        value = value.strip()

                        # 映射到标准键名
                        if key in self.METADATA_KEYS:
                            metadata[self.METADATA_KEYS[key]] = value
                        else:
                            # 保留原始键名 (小写)
                            metadata[key.lower().replace(' ', '_')] = value

            # 验证必要字段
            if 'lon' not in metadata or 'lat' not in metadata:
                return None

            return metadata

        except Exception as e:
            self.log('warning', f"解析头部失败 {filepath}: {e}")
            return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None

        # 尝试多种日期格式
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        """解析浮点数"""
        if not value or value.strip().upper() in ['NA', 'NAN', '']:
            return None
        try:
            return float(value.strip())
        except ValueError:
            return None

    def read_timeseries(self, filepath: str) -> List[Dict[str, Any]]:
        """
        读取站点时间序列数据

        Args:
            filepath: 文件路径

        Returns:
            时间序列数据列表
        """
        timeseries = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # 跳过头部
            data_lines = [l for l in lines[self.HEADER_LINES:] if not l.startswith('#')]

            for line in data_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                try:
                    # 格式: date time elevation uncertainty ...
                    date_str = parts[0]
                    time_str = parts[1]
                    elevation = float(parts[2])
                    uncertainty = float(parts[3])

                    # 跳过无效值
                    if elevation >= 9999.0:
                        continue

                    dt = self._parse_date(f"{date_str} {time_str}")

                    timeseries.append({
                        'datetime': dt,
                        'elevation': elevation,
                        'uncertainty': uncertainty,
                    })
                except (ValueError, IndexError):
                    continue

        except Exception as e:
            self.log('warning', f"读取时间序列失败 {filepath}: {e}")

        return timeseries
