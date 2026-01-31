#!/usr/bin/env python3
"""
ICESat Data Reader for WSE Pipeline
ICESat 数据读取器

支持格式:
1. 文本格式 (n00e005.txt 等) - 预处理后的数据
2. GLAH14 HDF5 - ICESat-1 Land Surface Altimetry (2003-2009)
3. ATL13 HDF5 - ICESat-2 Inland Water Surface Height (2018-present)

数据源:
- ICESat-1 GLAH14: https://nsidc.org/data/glah14
- ICESat-2 ATL13: https://nsidc.org/data/atl13
"""

import os
import re
import glob
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from collections import defaultdict

from .base_reader import BaseReader, create_station_from_reader
from ..core.station import Station
from ..exceptions import ReaderError


class ICESatReader(BaseReader):
    """
    ICESat 数据读取器

    支持三种数据格式:
    1. 文本格式 (.txt) - 预处理后的按经纬度分块文件
    2. GLAH14 HDF5 (.H5) - ICESat-1 原始数据
    3. ATL13 HDF5 (.h5) - ICESat-2 原始数据
    """

    source_name = "icesat"
    file_pattern = "*.txt"

    # 文件名正则: n00e005.txt, s10w060.txt 等
    FILENAME_PATTERN = re.compile(r'^([ns])(\d+)([ew])(\d+)\.txt$')

    # GLAH14 文件名正则
    GLAH14_PATTERN = re.compile(r'^GLAH14.*\.H5$', re.IGNORECASE)

    # ATL13 文件名正则
    ATL13_PATTERN = re.compile(r'^ATL13.*\.h5$', re.IGNORECASE)

    # ATL13 时间参考 (ATLAS SDP epoch)
    ATL13_EPOCH = datetime(2018, 1, 1, 0, 0, 0)

    # GLAH14 时间参考 (J2000 epoch)
    GLAH14_EPOCH = datetime(2000, 1, 1, 12, 0, 0)

    # ATL13 Ground tracks
    ATL13_GROUND_TRACKS = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']

    def scan_directory(self, path: str) -> List[str]:
        """扫描目录获取所有 ICESat 文件"""
        if path is None:
            raise ReaderError("数据路径未配置")

        path = Path(path)
        if not path.exists():
            raise ReaderError(f"路径不存在: {path}")

        files = []

        # 扫描文本文件
        for f in glob.glob(str(path / "*.txt")):
            filename = Path(f).name
            if self.FILENAME_PATTERN.match(filename):
                files.append(f)

        # 扫描 GLAH14 HDF5 文件
        for f in glob.glob(str(path / "*.H5")) + glob.glob(str(path / "*.h5")):
            filename = Path(f).name
            if self.GLAH14_PATTERN.match(filename) or self.ATL13_PATTERN.match(filename):
                files.append(f)

        # 递归扫描子目录
        for subdir in ['GLAH14', 'ATL13', 'glah14', 'atl13', 'icesat', 'icesat2']:
            subpath = path / subdir
            if subpath.exists():
                # 扫描 HDF5 文件
                for f in glob.glob(str(subpath / "*.H5")) + glob.glob(str(subpath / "*.h5")):
                    files.append(f)
                # 扫描文本文件
                for f in glob.glob(str(subpath / "*.txt")):
                    filename = Path(f).name
                    if self.FILENAME_PATTERN.match(filename):
                        files.append(f)

        return sorted(set(files))

    def read_station(self, filepath: str) -> Optional[Station]:
        """
        读取单个 ICESat 文件

        根据文件类型自动选择读取方法
        """
        filepath = Path(filepath)
        filename = filepath.name

        try:
            if self.GLAH14_PATTERN.match(filename):
                return self._read_glah14(filepath)
            elif self.ATL13_PATTERN.match(filename):
                return self._read_atl13(filepath)
            elif self.FILENAME_PATTERN.match(filename):
                return self._read_text_file(filepath)
            else:
                self.log('warning', f"未知文件格式: {filename}")
                return None

        except Exception as e:
            self.log('warning', f"读取 ICESat 文件失败 {filepath}: {e}")
            self._move_to_unreadable(filepath)
            return None

    def _move_to_unreadable(self, filepath: Path):
        """将无法读取的文件移动到 unreadable 文件夹"""
        import shutil

        try:
            # 在同级目录创建 unreadable 文件夹
            unreadable_dir = filepath.parent / 'unreadable'
            unreadable_dir.mkdir(exist_ok=True)

            # 移动文件
            dest = unreadable_dir / filepath.name
            shutil.move(str(filepath), str(dest))
            self.log('info', f"已将损坏文件移动到: {dest}")

        except Exception as e:
            self.log('warning', f"移动文件失败 {filepath}: {e}")

    def _read_text_file(self, filepath: Path) -> Optional[Station]:
        """读取文本格式文件"""
        filename = filepath.name
        center_lat, center_lon = self._parse_filename(filename)

        observations = self._read_text_observations(filepath)
        if not observations:
            return None

        lons = [obs['lon'] for obs in observations]
        lats = [obs['lat'] for obs in observations]
        elevations = [obs['elevation'] for obs in observations if obs['elevation'] is not None]
        dates = [obs['date'] for obs in observations if obs['date']]

        mean_lon = np.mean(lons)
        mean_lat = np.mean(lats)
        mean_elevation = np.mean(elevations) if elevations else None
        elevation_std = np.std(elevations) if len(elevations) > 1 else None

        return create_station_from_reader(
            id=f"ICESat_{filename.replace('.txt', '')}",
            station_name=f"ICESat_tile_{filename.replace('.txt', '')}",
            lon=mean_lon,
            lat=mean_lat,
            satellite="ICESat-GLAS",
            start_date=min(dates) if dates else None,
            end_date=max(dates) if dates else None,
            num_observations=len(observations),
            mean_elevation=mean_elevation,
            elevation_std=elevation_std,
            source=self.source_name,
            filepath=str(filepath),
            extra={
                'format': 'text',
                'tile_center_lon': center_lon,
                'tile_center_lat': center_lat,
                'lon_range': (min(lons), max(lons)),
                'lat_range': (min(lats), max(lats)),
            }
        )

    def _read_glah14(self, filepath: Path) -> Optional[Station]:
        """
        读取 GLAH14 HDF5 文件

        GLAH14 数据结构:
        - Data_40HZ/Geolocation/d_lat, d_lon - 坐标
        - Data_40HZ/Elevation_Surfaces/d_elev - 高程
        - Data_40HZ/DS_UTCTime_40 - 时间 (J2000 秒)
        - Data_40HZ/Quality/elev_use_flg - 质量标志
        """
        try:
            import h5py
        except ImportError:
            raise RuntimeError("需要安装 h5py: pip install h5py")

        with h5py.File(filepath, 'r') as f:
            # 读取 40HZ 数据 (高分辨率)
            try:
                lat = f['Data_40HZ/Geolocation/d_lat'][:]
                lon = f['Data_40HZ/Geolocation/d_lon'][:]
                elev = f['Data_40HZ/Elevation_Surfaces/d_elev'][:]
                time_j2000 = f['Data_40HZ/DS_UTCTime_40'][:]

                # 质量标志 (可选)
                try:
                    quality = f['Data_40HZ/Quality/elev_use_flg'][:]
                except KeyError:
                    quality = np.ones(len(lat), dtype=np.int8)

            except KeyError:
                # 尝试 1HZ 数据
                lat = f['Data_1HZ/Geolocation/d_lat'][:]
                lon = f['Data_1HZ/Geolocation/d_lon'][:]
                elev = np.full(len(lat), np.nan)  # 1HZ 没有高程
                time_j2000 = f['Data_1HZ/DS_UTCTime_1'][:]
                quality = np.ones(len(lat), dtype=np.int8)

        # 过滤有效数据
        valid_mask = (
            (quality == 0) &
            (np.isfinite(lat)) &
            (np.isfinite(lon)) &
            (np.isfinite(elev)) &
            (np.abs(lat) <= 90) &
            (np.abs(lon) <= 180) &
            (elev > -1000) & (elev < 10000)  # 合理高程范围
        )

        lat = lat[valid_mask]
        lon = lon[valid_mask]
        elev = elev[valid_mask]
        time_j2000 = time_j2000[valid_mask]

        if len(lat) == 0:
            return None

        # 转换时间
        dates = [self.GLAH14_EPOCH + timedelta(seconds=float(t)) for t in time_j2000]

        return create_station_from_reader(
            id=f"GLAH14_{filepath.stem}",
            station_name=f"GLAH14_{filepath.stem}",
            lon=float(np.mean(lon)),
            lat=float(np.mean(lat)),
            satellite="ICESat-1",
            start_date=min(dates) if dates else None,
            end_date=max(dates) if dates else None,
            num_observations=len(lat),
            mean_elevation=float(np.mean(elev)),
            elevation_std=float(np.std(elev)) if len(elev) > 1 else None,
            source=self.source_name,
            filepath=str(filepath),
            extra={
                'format': 'GLAH14',
                'lon_range': (float(np.min(lon)), float(np.max(lon))),
                'lat_range': (float(np.min(lat)), float(np.max(lat))),
                'elev_range': (float(np.min(elev)), float(np.max(elev))),
            }
        )

    def _read_atl13(self, filepath: Path) -> Optional[Station]:
        """
        读取 ATL13 HDF5 文件

        ATL13 数据结构 (每个 ground track):
        - gt*/segment_lat, segment_lon - 坐标
        - gt*/ht_water_surf - 水面高程
        - gt*/delta_time - 时间 (ATLAS SDP epoch 秒)
        - gt*/segment_geoid - 大地水准面高度
        - gt*/err_ht_water_surf - 高程误差
        - gt*/inland_water_body_id - 水体 ID
        """
        try:
            import h5py
        except ImportError:
            raise RuntimeError("需要安装 h5py: pip install h5py")

        all_lat = []
        all_lon = []
        all_elev = []
        all_time = []
        all_geoid = []
        all_error = []
        all_water_body_id = []

        with h5py.File(filepath, 'r') as f:
            for gt in self.ATL13_GROUND_TRACKS:
                if gt not in f:
                    continue

                try:
                    lat = f[f'{gt}/segment_lat'][:]
                    lon = f[f'{gt}/segment_lon'][:]
                    elev = f[f'{gt}/ht_water_surf'][:]
                    time_delta = f[f'{gt}/delta_time'][:]

                    # 可选字段
                    try:
                        geoid = f[f'{gt}/segment_geoid'][:]
                    except KeyError:
                        geoid = np.zeros(len(lat))

                    try:
                        error = f[f'{gt}/err_ht_water_surf'][:]
                    except KeyError:
                        error = np.full(len(lat), np.nan)

                    try:
                        water_body_id = f[f'{gt}/inland_water_body_id'][:]
                    except KeyError:
                        water_body_id = np.zeros(len(lat), dtype=np.int32)

                    # 过滤有效数据
                    valid_mask = (
                        (np.isfinite(lat)) &
                        (np.isfinite(lon)) &
                        (np.isfinite(elev)) &
                        (np.abs(lat) <= 90) &
                        (np.abs(lon) <= 180) &
                        (elev > -1000) & (elev < 10000)
                    )

                    all_lat.extend(lat[valid_mask])
                    all_lon.extend(lon[valid_mask])
                    all_elev.extend(elev[valid_mask])
                    all_time.extend(time_delta[valid_mask])
                    all_geoid.extend(geoid[valid_mask])
                    all_error.extend(error[valid_mask])
                    all_water_body_id.extend(water_body_id[valid_mask])

                except KeyError as e:
                    self.log('warning', f"ATL13 {gt} 缺少字段: {e}")
                    continue

        if len(all_lat) == 0:
            return None

        # 转换为 numpy 数组
        all_lat = np.array(all_lat)
        all_lon = np.array(all_lon)
        all_elev = np.array(all_elev)
        all_time = np.array(all_time)
        all_geoid = np.array(all_geoid)
        all_error = np.array(all_error)

        # 转换时间
        dates = [self.ATL13_EPOCH + timedelta(seconds=float(t)) for t in all_time]

        return create_station_from_reader(
            id=f"ATL13_{filepath.stem}",
            station_name=f"ATL13_{filepath.stem}",
            lon=float(np.mean(all_lon)),
            lat=float(np.mean(all_lat)),
            satellite="ICESat-2",
            start_date=min(dates) if dates else None,
            end_date=max(dates) if dates else None,
            num_observations=len(all_lat),
            mean_elevation=float(np.mean(all_elev)),
            elevation_std=float(np.std(all_elev)) if len(all_elev) > 1 else None,
            source=self.source_name,
            filepath=str(filepath),
            extra={
                'format': 'ATL13',
                'lon_range': (float(np.min(all_lon)), float(np.max(all_lon))),
                'lat_range': (float(np.min(all_lat)), float(np.max(all_lat))),
                'elev_range': (float(np.min(all_elev)), float(np.max(all_elev))),
                'mean_geoid': float(np.mean(all_geoid)),
                'mean_error': float(np.nanmean(all_error[np.isfinite(all_error)])) if np.any(np.isfinite(all_error)) else None,
            }
        )

    def read_hdf5_observations(self, filepath: str) -> List[Dict[str, Any]]:
        """
        读取 HDF5 文件的所有观测点

        Args:
            filepath: HDF5 文件路径

        Returns:
            观测点列表，每个观测点包含 lon, lat, elevation, date 等字段
        """
        filepath = Path(filepath)
        filename = filepath.name

        if self.GLAH14_PATTERN.match(filename):
            return self._read_glah14_observations(filepath)
        elif self.ATL13_PATTERN.match(filename):
            return self._read_atl13_observations(filepath)
        else:
            raise ValueError(f"不支持的 HDF5 格式: {filename}")

    def _read_glah14_observations(self, filepath: Path) -> List[Dict[str, Any]]:
        """读取 GLAH14 所有观测点"""
        import h5py

        observations = []

        with h5py.File(filepath, 'r') as f:
            try:
                lat = f['Data_40HZ/Geolocation/d_lat'][:]
                lon = f['Data_40HZ/Geolocation/d_lon'][:]
                elev = f['Data_40HZ/Elevation_Surfaces/d_elev'][:]
                time_j2000 = f['Data_40HZ/DS_UTCTime_40'][:]

                try:
                    quality = f['Data_40HZ/Quality/elev_use_flg'][:]
                except KeyError:
                    quality = np.zeros(len(lat), dtype=np.int8)

            except KeyError:
                return []

        for i in range(len(lat)):
            if not (np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(elev[i])):
                continue
            if not (-90 <= lat[i] <= 90 and -180 <= lon[i] <= 180):
                continue
            if not (-1000 < elev[i] < 10000):
                continue

            date = self.GLAH14_EPOCH + timedelta(seconds=float(time_j2000[i]))

            observations.append({
                'lon': float(lon[i]),
                'lat': float(lat[i]),
                'elevation': float(elev[i]),
                'date': date,
                'quality': int(quality[i]),
                'source_file': str(filepath),
                'satellite': 'ICESat-1',
            })

        return observations

    def _read_atl13_observations(self, filepath: Path) -> List[Dict[str, Any]]:
        """读取 ATL13 所有观测点"""
        import h5py

        observations = []

        with h5py.File(filepath, 'r') as f:
            for gt in self.ATL13_GROUND_TRACKS:
                if gt not in f:
                    continue

                try:
                    lat = f[f'{gt}/segment_lat'][:]
                    lon = f[f'{gt}/segment_lon'][:]
                    elev = f[f'{gt}/ht_water_surf'][:]
                    time_delta = f[f'{gt}/delta_time'][:]

                    try:
                        geoid = f[f'{gt}/segment_geoid'][:]
                    except KeyError:
                        geoid = np.zeros(len(lat))

                    try:
                        error = f[f'{gt}/err_ht_water_surf'][:]
                    except KeyError:
                        error = np.full(len(lat), np.nan)

                    try:
                        water_body_id = f[f'{gt}/inland_water_body_id'][:]
                    except KeyError:
                        water_body_id = np.zeros(len(lat), dtype=np.int32)

                except KeyError:
                    continue

                for i in range(len(lat)):
                    if not (np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(elev[i])):
                        continue
                    if not (-90 <= lat[i] <= 90 and -180 <= lon[i] <= 180):
                        continue
                    if not (-1000 < elev[i] < 10000):
                        continue

                    date = self.ATL13_EPOCH + timedelta(seconds=float(time_delta[i]))

                    observations.append({
                        'lon': float(lon[i]),
                        'lat': float(lat[i]),
                        'elevation': float(elev[i]),
                        'date': date,
                        'geoid': float(geoid[i]),
                        'error': float(error[i]) if np.isfinite(error[i]) else None,
                        'water_body_id': int(water_body_id[i]),
                        'ground_track': gt,
                        'source_file': str(filepath),
                        'satellite': 'ICESat-2',
                    })

        return observations

    def _parse_filename(self, filename: str) -> Tuple[float, float]:
        """从文件名解析区域中心坐标"""
        match = self.FILENAME_PATTERN.match(filename)
        if not match:
            raise ValueError(f"无效的文件名格式: {filename}")

        ns, lat_str, ew, lon_str = match.groups()
        lat = float(lat_str)
        lon = float(lon_str)

        if ns == 's':
            lat = -lat
        if ew == 'w':
            lon = -lon

        return lat + 2.5, lon + 2.5

    def read_timeseries(self, filepath: str) -> List[Dict[str, Any]]:
        """
        读取站点时间序列数据

        Args:
            filepath: 文件路径

        Returns:
            时间序列数据列表，每条记录包含:
            - datetime: 观测时间
            - elevation: 高程 (m)
            - uncertainty: 不确定度 (m)，可能为 None
        """
        timeseries = []
        filepath = Path(filepath)
        filename = filepath.name

        try:
            # 根据文件类型选择读取方法
            if self.GLAH14_PATTERN.match(filename):
                observations = self._read_glah14_observations(filepath)
            elif self.ATL13_PATTERN.match(filename):
                observations = self._read_atl13_observations(filepath)
            elif self.FILENAME_PATTERN.match(filename):
                observations = self._read_text_observations(filepath)
            else:
                self.log('warning', f"未知文件格式: {filename}")
                return timeseries

            # 转换为标准时间序列格式
            for obs in observations:
                # 获取不确定度字段 (ATL13 使用 'error', GLAH14/text 可能没有)
                uncertainty = obs.get('error')
                if uncertainty is None:
                    # 对于 GLAH14，使用默认不确定度
                    uncertainty = obs.get('uncertainty')

                timeseries.append({
                    'datetime': obs.get('date'),
                    'elevation': obs.get('elevation'),
                    'uncertainty': uncertainty,
                })

        except Exception as e:
            self.log('warning', f"读取时间序列失败 {filepath}: {e}")

        return timeseries

    def _read_text_observations(self, filepath: Path) -> List[Dict[str, Any]]:
        """读取文本文件中的所有观测"""
        observations = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue

                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        year = int(parts[3])
                        month = int(parts[4])
                        day = int(parts[5])

                        try:
                            date = datetime(year, month, day)
                        except ValueError:
                            date = None

                        elevation = None
                        if len(parts) > 7:
                            try:
                                elevation = float(parts[7])
                            except ValueError:
                                pass

                        observations.append({
                            'lon': lon,
                            'lat': lat,
                            'date': date,
                            'elevation': elevation,
                            'raw': parts,
                        })

                    except (ValueError, IndexError):
                        continue

        except FileNotFoundError:
            self.log('warning', f"File not found: {filepath}")
        except PermissionError:
            self.log('error', f"Permission denied: {filepath}")
            raise ReaderError(f"Permission denied: {filepath}")
        except Exception as e:
            self.log('warning', f"读取观测数据失败 {filepath}: {e}")

        return observations

    def read_all_observations(self, path: str) -> List[Dict[str, Any]]:
        """
        读取所有文件的所有观测点

        Args:
            path: 数据目录

        Returns:
            所有观测点列表
        """
        all_observations = []

        files = self.scan_directory(path)
        for filepath in files:
            filepath = Path(filepath)

            if filepath.suffix.lower() in ['.h5', '.H5']:
                observations = self.read_hdf5_observations(str(filepath))
            else:
                observations = self._read_text_observations(filepath)
                for obs in observations:
                    obs['source_file'] = str(filepath)

            all_observations.extend(observations)

        return all_observations

    def read_all_stations(self,
                          path: str,
                          progress_callback=None,
                          filters=None) -> List[Station]:
        """
        读取所有站点 - 对 ATL13 文件使用聚类

        ATL13 文件包含整个轨道的观测数据，需要按地理位置聚类成多个站点，
        而不是创建一个平均坐标的单一站点。

        Args:
            path: 数据目录路径
            progress_callback: 进度回调函数
            filters: 过滤条件

        Returns:
            Station 对象列表
        """
        files = self.scan_directory(path)
        total = len(files)
        self.log('info', f"扫描到 {total} 个文件")

        stations = []
        skipped = 0
        failed = 0

        # 计算进度显示间隔
        progress_interval = max(1, min(total // 10, 50))

        for i, filepath in enumerate(files):
            filepath = Path(filepath)
            filename = filepath.name

            # 显示进度
            if i % progress_interval == 0 or i == total - 1:
                pct = (i + 1) * 100 // total
                self.log('info', f"  进度: [{i + 1}/{total}] {pct}%")

            if progress_callback and (i % 100 == 0 or i == total - 1):
                progress_callback(i + 1, total, f"读取 {filename}")

            try:
                # ATL13 文件需要聚类处理
                if self.ATL13_PATTERN.match(filename):
                    file_stations = self._read_atl13_clustered(filepath, filters)
                    stations.extend(file_stations)
                else:
                    # 其他格式使用原有逻辑
                    station = self.read_station(str(filepath))
                    if station:
                        if self._apply_filters(station, filters):
                            stations.append(station)
                        else:
                            skipped += 1
            except Exception as e:
                failed += 1
                self.log('warning', f"读取文件失败 {filepath}: {e}")

        self.log('info', f"成功读取 {len(stations)} 个站点，跳过 {skipped} 个，失败 {failed} 个")
        return stations

    def _read_atl13_clustered(self, filepath: Path,
                               filters=None) -> List[Station]:
        """
        读取 ATL13 文件，仅保留大流域河流点（不聚类）

        处理流程:
        1. 读取所有观测点
        2. 使用 MERIT_Hydro 检查每个点的上游面积
        3. 仅保留大流域点 (UPA > 阈值)
        4. 每个观测点作为独立站点

        Args:
            filepath: ATL13 文件路径
            filters: 过滤条件

        Returns:
            Station 列表
        """
        # 初始化 MERIT 读取器（懒加载）
        if not hasattr(self, '_merit_reader'):
            self._init_merit_reader()

        if self._merit_reader is None:
            self.log('warning', "MERIT_Hydro 未加载，无法过滤河流点")
            return []

        # 河流阈值（从配置读取，默认 10 km²）
        river_threshold = 10.0
        if filters:
            river_threshold = filters.get('river_threshold', 10.0)

        stations = []
        file_stem = filepath.stem

        # 直接从 HDF5 读取并过滤（避免加载全部观测点到内存）
        try:
            import h5py
        except ImportError:
            self.log('error', "需要安装 h5py")
            return []

        with h5py.File(filepath, 'r') as f:
            for gt in self.ATL13_GROUND_TRACKS:
                if gt not in f:
                    continue

                try:
                    lat = f[f'{gt}/segment_lat'][:]
                    lon = f[f'{gt}/segment_lon'][:]
                    elev = f[f'{gt}/ht_water_surf'][:]
                    time_delta = f[f'{gt}/delta_time'][:]

                    try:
                        geoid = f[f'{gt}/segment_geoid'][:]
                    except KeyError:
                        geoid = np.zeros(len(lat))

                except KeyError:
                    continue

                # 逐点检查是否为大流域河流
                for i in range(len(lat)):
                    if not (np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(elev[i])):
                        continue
                    if not (-90 <= lat[i] <= 90 and -180 <= lon[i] <= 180):
                        continue

                    # 检查上游面积
                    tile = self._merit_reader.load_tile(lon[i], lat[i])
                    if tile is None:
                        continue

                    ix, iy, _ = self._merit_reader.lonlat_to_pixel(lon[i], lat[i])
                    upa = float(tile['upa'][iy, ix])

                    if upa < river_threshold:
                        continue  # 非大流域点

                    # 创建站点
                    obs_date = self.ATL13_EPOCH + timedelta(seconds=float(time_delta[i]))
                    station_id = f"ATL13_{file_stem}_{gt}_{i:06d}"

                    station = create_station_from_reader(
                        id=station_id,
                        station_name=f"ATL13_{lat[i]:.4f}_{lon[i]:.4f}",
                        lon=float(lon[i]),
                        lat=float(lat[i]),
                        satellite="ICESat-2",
                        start_date=obs_date,
                        end_date=obs_date,
                        num_observations=1,
                        mean_elevation=float(elev[i]),
                        elevation_std=None,
                        source=self.source_name,
                        filepath=str(filepath),
                        extra={
                            'format': 'ATL13',
                            'upa': upa,
                            'geoid': float(geoid[i]),
                            'ground_track': gt,
                            'original_file': filepath.name,
                        }
                    )

                    stations.append(station)

        return stations

    def _init_merit_reader(self):
        """初始化 MERIT_Hydro 读取器"""
        try:
            from ..core.merit_reader import MeritHydroReader

            # 尝试从配置读取路径，否则使用默认值
            merit_root = '/Volumes/Data01/MERIT_Hydro'
            self._merit_reader = MeritHydroReader(merit_root)
            self.log('info', f"已加载 MERIT_Hydro: {merit_root}")
        except Exception as e:
            self.log('warning', f"无法加载 MERIT_Hydro: {e}")
            self._merit_reader = None

    def _apply_filters(self, station: Station, filters) -> bool:
        """应用过滤条件"""
        if not filters:
            return True

        min_obs = filters.get('min_observations')
        if min_obs and station.num_observations < min_obs:
            return False

        return True

    def cluster_observations(self,
                             observations: List[Dict],
                             distance_threshold_km: float = 1.0) -> List[Station]:
        """
        将观测点聚类为虚拟站点

        Args:
            observations: 观测点列表
            distance_threshold_km: 聚类距离阈值 (km)

        Returns:
            聚类后的虚拟站点列表
        """
        grid_size = 0.01  # 约 1 km
        clusters = defaultdict(list)

        for obs in observations:
            grid_lon = int(obs['lon'] / grid_size) * grid_size
            grid_lat = int(obs['lat'] / grid_size) * grid_size
            key = (grid_lon, grid_lat)
            clusters[key].append(obs)

        stations = []
        for (grid_lon, grid_lat), obs_list in clusters.items():
            if len(obs_list) < 3:
                continue

            lons = [o['lon'] for o in obs_list]
            lats = [o['lat'] for o in obs_list]
            elevations = [o['elevation'] for o in obs_list if o.get('elevation')]
            dates = [o['date'] for o in obs_list if o.get('date')]

            # 确定卫星类型
            satellites = set(o.get('satellite', 'ICESat') for o in obs_list)
            satellite = ', '.join(satellites) if len(satellites) > 1 else list(satellites)[0]

            station = create_station_from_reader(
                id=f"ICESat_{grid_lat:.2f}_{grid_lon:.2f}",
                station_name=f"ICESat_cluster_{grid_lat:.2f}_{grid_lon:.2f}",
                lon=np.mean(lons),
                lat=np.mean(lats),
                satellite=satellite,
                start_date=min(dates) if dates else None,
                end_date=max(dates) if dates else None,
                num_observations=len(obs_list),
                mean_elevation=np.mean(elevations) if elevations else None,
                elevation_std=np.std(elevations) if len(elevations) > 1 else None,
                source=self.source_name,
            )
            stations.append(station)

        return stations


# 命令行接口
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ICESat 数据读取器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的数据格式:
  - 文本格式 (.txt): 预处理后的按经纬度分块文件
  - GLAH14 HDF5 (.H5): ICESat-1 原始数据 (2003-2009)
  - ATL13 HDF5 (.h5): ICESat-2 原始数据 (2018-present)

示例:
  # 扫描目录
  python -m src.readers.icesat_reader --scan /path/to/data

  # 读取单个文件信息
  python -m src.readers.icesat_reader --info /path/to/ATL13_xxx.h5

  # 读取所有观测点
  python -m src.readers.icesat_reader --observations /path/to/data --limit 100
        """
    )

    parser.add_argument('--scan', metavar='DIR', help='扫描目录')
    parser.add_argument('--info', metavar='FILE', help='读取文件信息')
    parser.add_argument('--observations', metavar='PATH', help='读取观测点')
    parser.add_argument('--limit', type=int, default=10, help='显示数量限制')

    args = parser.parse_args()

    reader = ICESatReader()

    if args.scan:
        files = reader.scan_directory(args.scan)
        print(f"找到 {len(files)} 个文件:")
        for f in files[:args.limit]:
            print(f"  {f}")
        if len(files) > args.limit:
            print(f"  ... 还有 {len(files) - args.limit} 个文件")

    elif args.info:
        station = reader.read_station(args.info)
        if station:
            print(f"文件: {args.info}")
            print(f"  ID: {station.id}")
            print(f"  卫星: {station.satellite}")
            print(f"  位置: ({station.lat:.4f}, {station.lon:.4f})")
            print(f"  观测数: {station.num_observations}")
            print(f"  平均高程: {station.mean_elevation:.2f} m" if station.mean_elevation else "  平均高程: N/A")
            print(f"  时间范围: {station.start_date} - {station.end_date}")
            if station.extra:
                print(f"  额外信息: {station.extra}")
        else:
            print(f"无法读取文件: {args.info}")

    elif args.observations:
        filepath = Path(args.observations)
        if filepath.is_file():
            observations = reader.read_hdf5_observations(str(filepath))
        else:
            observations = reader.read_all_observations(str(filepath))

        print(f"共 {len(observations)} 个观测点:")
        for obs in observations[:args.limit]:
            print(f"  ({obs['lat']:.4f}, {obs['lon']:.4f}) "
                  f"elev={obs.get('elevation', 'N/A'):.2f}m "
                  f"date={obs.get('date', 'N/A')}")
        if len(observations) > args.limit:
            print(f"  ... 还有 {len(observations) - args.limit} 个观测点")

    else:
        parser.print_help()
