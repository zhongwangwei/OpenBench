#!/usr/bin/env python3
"""
CGLS (Copernicus Global Land Service) Data Reader for WSE Pipeline
CGLS 数据读取器

文件格式:
- 每个站点一个 JSON 文件 (GeoJSON 格式)
- c_gls_WL_YYYYMMDDHHII_XXXXXXXXXX_ALTI_V*.json
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base_reader import BaseReader, create_station_from_reader
from ..core.station import Station


class CGLSReader(BaseReader):
    """CGLS 数据读取器"""

    source_name = "cgls"
    file_pattern = "c_gls_WL_*.json"

    def scan_directory(self, path: str) -> List[str]:
        """扫描目录获取所有 CGLS 文件"""
        pattern = os.path.join(path, "**", self.file_pattern)
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def read_station(self, filepath: str) -> Optional[Station]:
        """读取单个 CGLS 站点文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 解析 GeoJSON 结构
            geometry = data.get('geometry', {})
            properties = data.get('properties', {})
            observations = data.get('data', [])

            # 获取坐标
            coords = geometry.get('coordinates', [0, 0])
            lon = coords[0]
            lat = coords[1]

            # 解析时间范围
            start_date = self._parse_datetime(properties.get('time_coverage_start'))
            end_date = self._parse_datetime(properties.get('time_coverage_end'))

            # 计算平均高程和标准差
            elevations = [
                obs.get('water_surface_height_above_reference_datum')
                for obs in observations
                if obs.get('water_surface_height_above_reference_datum') is not None
            ]

            mean_elevation = None
            elevation_std = None
            if elevations:
                mean_elevation = sum(elevations) / len(elevations)
                if len(elevations) > 1:
                    variance = sum((e - mean_elevation) ** 2 for e in elevations) / len(elevations)
                    elevation_std = variance ** 0.5

            # 创建站点对象
            station = create_station_from_reader(
                id=properties.get('resource', '').strip(),
                station_name=f"{properties.get('river', 'Unknown')}_{properties.get('resource', '')}",
                lon=lon,
                lat=lat,
                river=properties.get('river'),
                basin=properties.get('basin'),
                country=properties.get('country'),
                satellite=properties.get('platform', 'Unknown'),
                start_date=start_date,
                end_date=end_date,
                num_observations=len(observations),
                mean_elevation=mean_elevation,
                elevation_std=elevation_std,
                source=self.source_name,
                filepath=filepath,
                extra={
                    'geoid_model': properties.get('water_surface_reference_name'),
                    'geoid_undulation': properties.get('water_surface_reference_datum_altitude'),
                    'processing_level': properties.get('processing_level'),
                    'status': properties.get('status'),
                    'institution': properties.get('institution'),
                    'missing_value': properties.get('missing_value'),
                }
            )

            return station

        except Exception as e:
            self.log('warning', f"解析 CGLS 文件失败 {filepath}: {e}")
            return None

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """解析日期时间字符串"""
        if not dt_str:
            return None

        # CGLS 使用多种格式
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y/%m/%d %H:%M',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue

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
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            observations = data.get('data', [])
            missing_value = data.get('properties', {}).get('missing_value', 9999.999)

            for obs in observations:
                elevation = obs.get('water_surface_height_above_reference_datum')
                uncertainty = obs.get('water_surface_height_uncertainty')

                # 跳过无效值
                if elevation is None or elevation == missing_value:
                    continue

                # 解析时间
                dt_str = obs.get('datetime')
                dt = None
                if dt_str:
                    # 格式: "YYYY/MM/DD HH:MM"
                    try:
                        dt = datetime.strptime(dt_str, '%Y/%m/%d %H:%M')
                    except ValueError:
                        pass

                timeseries.append({
                    'datetime': dt,
                    'elevation': elevation,
                    'uncertainty': uncertainty,
                    'identifier': obs.get('identifier'),
                })

        except Exception as e:
            self.log('warning', f"读取时间序列失败 {filepath}: {e}")

        return timeseries
