#!/usr/bin/env python3
"""
Base Reader for WSE Pipeline
数据源读取器基类
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path

from ..core.station import Station
from ..exceptions import ReaderError

# Backward compatibility alias - StationMetadata is now Station
# This allows existing code to continue using StationMetadata while
# the unified Station type is used throughout the codebase
StationMetadata = Station


def create_station_from_reader(
    id: str,
    station_name: str,
    lon: float,
    lat: float,
    source: str,
    river: Optional[str] = None,
    basin: Optional[str] = None,
    country: Optional[str] = None,
    satellite: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    num_observations: int = 0,
    mean_elevation: Optional[float] = None,
    elevation_std: Optional[float] = None,
    egm08: Optional[float] = None,
    egm96: Optional[float] = None,
    filepath: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Station:
    """
    Create a Station object from reader data.

    This helper function maps the old StationMetadata fields to the unified Station type.

    Args:
        id: Station unique identifier
        station_name: Station name
        lon: Longitude (degrees)
        lat: Latitude (degrees)
        source: Data source name
        river: River name
        basin: Basin name
        country: Country
        satellite: Satellite platform
        start_date: Data start date
        end_date: Data end date
        num_observations: Number of observations
        mean_elevation: Mean water surface elevation (m)
        elevation_std: Elevation standard deviation (m)
        egm08: EGM2008 geoid undulation (m)
        egm96: EGM96 geoid undulation (m)
        filepath: Original file path
        extra: Additional metadata

    Returns:
        Station object
    """
    metadata = {
        'river': river,
        'basin': basin,
        'country': country,
        'satellite': satellite,
        'start_date': start_date.isoformat() if start_date else None,
        'end_date': end_date.isoformat() if end_date else None,
        'elevation_std': elevation_std,
        'filepath': filepath,
    }

    # Add extra fields to metadata
    if extra:
        metadata.update(extra)

    # Filter out None values
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return Station(
        id=id,
        name=station_name,
        lon=lon,
        lat=lat,
        source=source,
        elevation=mean_elevation or 0.0,
        num_observations=num_observations,
        egm08=egm08,
        egm96=egm96,
        metadata=metadata,
    )


class BaseReader(ABC):
    """
    数据源读取器抽象基类

    子类必须实现:
    - source_name: 数据源名称
    - scan_directory(): 扫描目录获取文件列表
    - read_station(): 读取单个站点数据
    """

    source_name: str = "unknown"
    file_pattern: str = "*"

    def __init__(self, logger=None):
        """
        初始化读取器

        Args:
            logger: 日志记录器
        """
        self.logger = logger

    def log(self, level: str, message: str):
        """记录日志"""
        if self.logger:
            getattr(self.logger, level.lower())(message)

    @abstractmethod
    def scan_directory(self, path: str) -> List[str]:
        """
        扫描目录，返回所有数据文件路径

        Args:
            path: 数据目录路径

        Returns:
            文件路径列表
        """
        pass

    @abstractmethod
    def read_station(self, filepath: str) -> Optional[Station]:
        """
        读取单个站点数据

        Args:
            filepath: 数据文件路径

        Returns:
            Station 对象，如果读取失败返回 None
        """
        pass

    def read_all_stations(self,
                          path: str,
                          progress_callback: Optional[Callable[[int, int, str], None]] = None,
                          filters: Optional[Dict[str, Any]] = None) -> List[Station]:
        """
        读取所有站点

        Args:
            path: 数据目录路径
            progress_callback: 进度回调函数 (current, total, message)
            filters: 过滤条件

        Returns:
            Station 对象列表
        """
        files = self.scan_directory(path)
        total = len(files)
        self.log('info', f"扫描到 {total} 个文件")

        stations = []
        skipped = 0

        for i, filepath in enumerate(files):
            if progress_callback and (i % 100 == 0 or i == total - 1):
                progress_callback(i + 1, total, f"读取 {Path(filepath).name}")

            try:
                station = self.read_station(filepath)
                if station:
                    # 应用过滤条件
                    if self._apply_filters(station, filters):
                        stations.append(station)
                    else:
                        skipped += 1
            except ReaderError:
                # Re-raise critical reader errors
                raise
            except Exception as e:
                self.log('warning', f"读取文件失败 {filepath}: {e}")

        self.log('info', f"成功读取 {len(stations)} 个站点，跳过 {skipped} 个")
        return stations

    def _apply_filters(self,
                       station: Station,
                       filters: Optional[Dict[str, Any]]) -> bool:
        """
        应用过滤条件

        Args:
            station: Station 对象
            filters: 过滤条件

        Returns:
            是否通过过滤
        """
        if not filters:
            return True

        # 最少观测次数
        min_obs = filters.get('min_observations')
        if min_obs and station.num_observations < min_obs:
            return False

        # 时间范围过滤 - now uses metadata dict
        start_date = filters.get('start_date')
        station_end_date = station.metadata.get('end_date')
        if start_date and station_end_date:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(station_end_date, str):
                station_end_date = datetime.fromisoformat(station_end_date)
            if station_end_date < start_date:
                return False

        end_date = filters.get('end_date')
        station_start_date = station.metadata.get('start_date')
        if end_date and station_start_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            if isinstance(station_start_date, str):
                station_start_date = datetime.fromisoformat(station_start_date)
            if station_start_date > end_date:
                return False

        # 空间范围过滤 [west, south, east, north]
        bbox = filters.get('bbox')
        if bbox:
            west, south, east, north = bbox
            if not (west <= station.lon <= east and south <= station.lat <= north):
                return False

        # 河流名称过滤 - now uses metadata dict
        rivers = filters.get('rivers')
        station_river = station.metadata.get('river')
        if rivers and station_river:
            if station_river.upper() not in [r.upper() for r in rivers]:
                return False

        # 流域名称过滤 - now uses metadata dict
        basins = filters.get('basins')
        station_basin = station.metadata.get('basin')
        if basins and station_basin:
            if station_basin.upper() not in [b.upper() for b in basins]:
                return False

        return True
