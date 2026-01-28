#!/usr/bin/env python3
"""
HydroSat Data Reader for WSE Pipeline
HydroSat 数据读取器

数据源: https://hydrosat.gis.uni-stuttgart.de/
数据下载: https://hydrosat.gis.uni-stuttgart.de/data/download/WL-HydroSat.zip
"""

import os
import re
import glob
import zipfile
import warnings
import requests
import urllib3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_reader import BaseReader, create_station_from_reader
from ..core.station import Station


class HydroSatReader(BaseReader):
    """
    HydroSat 数据读取器

    数据格式:
    - 文件名: {hydrosat_no}.txt (如 21017600011001.txt)
    - 元数据以 # 开头
    - 数据格式: YYYY,MM,DD,Data,Error

    元数据字段:
    - Object: 河流/湖泊名称
    - Latitude/Longitude: 坐标
    - Altitude: 高程 (m)
    - Mission: 卫星任务
    - Datum: 参考基准 (通常是 EGM 2008)
    """

    source_name = "hydrosat"
    file_pattern = "*.txt"

    # 支持的卫星任务
    SATELLITES = {
        'ENVISAT': 'Envisat',
        'SARAL': 'SARAL',
        'ALTIKA': 'SARAL',
        'JASON': 'Jason',
        'JASON-1': 'Jason-1',
        'JASON-2': 'Jason-2',
        'JASON-3': 'Jason-3',
        'SENTINEL-3A': 'Sentinel-3A',
        'SENTINEL-3B': 'Sentinel-3B',
        'SENTINEL-3': 'Sentinel-3',
        'CRYOSAT': 'CryoSat-2',
        'CRYOSAT-2': 'CryoSat-2',
        'ICESAT': 'ICESat',
        'ICESAT-2': 'ICESat-2',
        'TOPEX': 'TOPEX',
        'SWOT': 'SWOT',
    }

    def scan_directory(self, path: str) -> List[str]:
        """扫描目录获取所有 HydroSat 文件"""
        if path is None:
            raise ValueError(
                "HydroSat 数据路径未配置。\n"
                "请先下载数据:\n"
                "  python -m src.readers.hydrosat_reader --download /path/to/output\n"
                "或手动下载:\n"
                "  https://hydrosat.gis.uni-stuttgart.de/data/download/WL-HydroSat.zip"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")

        # 查找所有 txt 文件
        files = []

        # 直接在目录下
        files.extend(glob.glob(str(path / "*.txt")))

        # WL_hydrosat 子目录 (ZIP 解压后的结构)
        files.extend(glob.glob(str(path / "WL_hydrosat/*.txt")))

        # 其他可能的子目录结构
        files.extend(glob.glob(str(path / "*/*.txt")))

        # 过滤: 只保留数字命名的文件 (HydroSat 格式)
        data_files = []
        for f in files:
            filename = Path(f).stem
            # HydroSat 文件名是纯数字
            if filename.isdigit() or re.match(r'^\d+$', filename):
                data_files.append(f)

        return sorted(set(data_files))

    def read_station(self, filepath: str) -> Optional[Station]:
        """
        读取单个 HydroSat 站点文件

        Args:
            filepath: 文件路径

        Returns:
            Station 或 None
        """
        filepath = Path(filepath)

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            if not lines:
                return None

            # 解析元数据和数据
            metadata = self._parse_metadata(lines)
            data = self._parse_data(lines)

            if not data:
                return None

            # 提取字段
            hydrosat_no = metadata.get('hydrosat_no', filepath.stem)
            object_name = metadata.get('object', 'Unknown')
            lat = metadata.get('latitude', 0.0)
            lon = metadata.get('longitude', 0.0)
            altitude = metadata.get('altitude')
            mission = metadata.get('mission', '')

            # 解析卫星
            satellite = self._parse_satellite(mission)

            # 统计数据 (过滤无效值)
            import math
            water_levels = [
                d['value'] for d in data
                if d.get('value') is not None and math.isfinite(d['value'])
            ]
            dates = [d['date'] for d in data if d.get('date') is not None]

            if not water_levels:
                return None

            return create_station_from_reader(
                id=hydrosat_no,
                station_name=f"{object_name}_{hydrosat_no}",
                lon=lon,
                lat=lat,
                river=object_name,
                country=metadata.get('country'),
                source=self.source_name,
                satellite=satellite,
                start_date=min(dates) if dates else None,
                end_date=max(dates) if dates else None,
                num_observations=len(water_levels),
                mean_elevation=sum(water_levels) / len(water_levels),
                filepath=str(filepath),
                extra={
                    'hydrosat_no': hydrosat_no,
                    'altitude': altitude,
                    'mission': mission,
                    'datum': metadata.get('datum'),
                    'basin_no': metadata.get('basin_no'),
                },
            )

        except Exception as e:
            if self.logger:
                self.logger.warning(f"解析 HydroSat 文件失败 {filepath}: {e}")
            return None

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """解析元数据行"""
        metadata = {}

        for line in lines:
            line = line.strip()
            if not line.startswith('#'):
                break

            # 解析 "# Key: Value" 格式
            if ':' in line:
                # 移除 # 前缀
                content = line.lstrip('#').strip()
                if ':' in content:
                    key, value = content.split(':', 1)
                    key = key.strip().lower().replace('(°)', '').replace('(m)', '').replace(' ', '_').strip('_')
                    value = value.strip()

                    if not value:
                        continue

                    # 特殊处理
                    if key == 'latitude':
                        try:
                            metadata['latitude'] = float(value)
                        except ValueError:
                            pass
                    elif key == 'longitude':
                        try:
                            metadata['longitude'] = float(value)
                        except ValueError:
                            pass
                    elif key == 'altitude':
                        try:
                            metadata['altitude'] = float(value)
                        except ValueError:
                            pass
                    elif key == 'hydrosat_no.':
                        metadata['hydrosat_no'] = value
                    elif key == 'object':
                        metadata['object'] = value
                    elif key == 'mission':
                        metadata['mission'] = value
                    elif key == 'datum':
                        metadata['datum'] = value
                    elif key == 'country':
                        metadata['country'] = value
                    elif key == 'basin_no.':
                        metadata['basin_no'] = value

        return metadata

    def _parse_data(self, lines: List[str]) -> List[Dict[str, Any]]:
        """解析数据行"""
        data = []
        in_data_section = False

        for line in lines:
            line = line.strip()

            # 跳过空行和注释
            if not line:
                continue

            if line.startswith('#'):
                # 检查是否到达数据部分
                if 'DATA' in line.upper():
                    in_data_section = True
                continue

            # 解析数据行: YYYY,MM,DD,Value,Error
            if in_data_section or re.match(r'^\d{4},\d+,\d+,', line):
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        value = float(parts[3])
                        error = float(parts[4]) if len(parts) > 4 else None

                        data.append({
                            'date': datetime(year, month, day),
                            'value': value,
                            'error': error,
                        })
                    except (ValueError, IndexError):
                        continue

        return data

    def _parse_satellite(self, mission: str) -> Optional[str]:
        """从 mission 字段解析卫星名称"""
        if not mission:
            return None

        mission_upper = mission.upper()

        for key, sat_name in self.SATELLITES.items():
            if key in mission_upper:
                return sat_name

        # 返回原始值
        return mission.split(',')[0].strip() if mission else None


class HydroSatDownloader:
    """
    HydroSat 数据下载器

    从 https://hydrosat.gis.uni-stuttgart.de/ 下载 Water Level 数据
    """

    # 直接下载链接
    DOWNLOAD_URL = "https://hydrosat.gis.uni-stuttgart.de/data/download/WL-HydroSat.zip"

    def __init__(self, output_dir: str, logger=None, verify_ssl: bool = True):
        """
        初始化下载器

        Args:
            output_dir: 输出目录
            logger: 日志记录器
            verify_ssl: 是否验证 SSL 证书 (默认 True)
        """
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.verify_ssl = verify_ssl
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not verify_ssl:
            self.log('warning', "SSL verification disabled - connections may be insecure")

    def log(self, level: str, message: str):
        """记录日志"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")

    def download(self, extract: bool = True) -> Path:
        """
        下载 HydroSat Water Level 数据

        Args:
            extract: 是否自动解压

        Returns:
            数据目录路径
        """
        zip_path = self.output_dir / "WL-HydroSat.zip"
        extract_dir = self.output_dir / "WL_hydrosat"

        # 如果数据已存在
        if extract_dir.exists() and any(extract_dir.iterdir()):
            self.log('info', f"数据已存在: {extract_dir}")
            return extract_dir

        # 下载 ZIP 文件
        if not zip_path.exists():
            self.log('info', f"下载 HydroSat 数据...")
            self.log('info', f"URL: {self.DOWNLOAD_URL}")

            try:
                # Use context-scoped warning suppression when SSL verification is disabled
                with warnings.catch_warnings():
                    if not self.verify_ssl:
                        warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)

                    response = requests.get(
                        self.DOWNLOAD_URL,
                        stream=True,
                        timeout=300,
                        verify=self.verify_ssl
                    )
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            # 每 MB 打印一次进度
                            if downloaded % (1024 * 1024) < 8192:
                                self.log('info', f"下载进度: {progress:.1f}% ({downloaded // (1024*1024)} MB)")

                self.log('info', f"下载完成: {zip_path}")
                self.log('info', f"文件大小: {zip_path.stat().st_size / (1024*1024):.1f} MB")

            except requests.RequestException as e:
                self.log('error', f"下载失败: {e}")
                if zip_path.exists():
                    zip_path.unlink()
                raise

        # 解压
        if extract:
            self.log('info', f"解压到: {self.output_dir}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(self.output_dir)

                # 统计文件数量
                txt_files = list(extract_dir.glob("*.txt"))
                self.log('info', f"解压完成: {len(txt_files)} 个站点文件")

            except zipfile.BadZipFile as e:
                self.log('error', f"解压失败: {e}")
                raise

            return extract_dir
        else:
            return zip_path

    def get_station_count(self) -> int:
        """获取已下载的站点数量"""
        extract_dir = self.output_dir / "WL_hydrosat"
        if extract_dir.exists():
            return len(list(extract_dir.glob("*.txt")))
        return 0


def download_hydrosat(output_dir: str, logger=None, verify_ssl: bool = True) -> Path:
    """
    下载 HydroSat 数据的便捷函数

    Args:
        output_dir: 输出目录
        logger: 日志记录器
        verify_ssl: 是否验证 SSL 证书 (默认 True)

    Returns:
        数据目录路径
    """
    downloader = HydroSatDownloader(output_dir, logger, verify_ssl=verify_ssl)
    return downloader.download(extract=True)


# 命令行接口
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='HydroSat 数据下载器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据来源:
  https://hydrosat.gis.uni-stuttgart.de/

示例:
  # 下载数据
  python -m src.readers.hydrosat_reader --download /path/to/output

  # 仅下载不解压
  python -m src.readers.hydrosat_reader --download /path/to/output --no-extract

  # 查看已下载数据的站点数量
  python -m src.readers.hydrosat_reader --info /path/to/data
        """
    )

    parser.add_argument(
        '--download',
        metavar='OUTPUT_DIR',
        help='下载数据到指定目录'
    )

    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='不自动解压 ZIP 文件'
    )

    parser.add_argument(
        '--info',
        metavar='DATA_DIR',
        help='查看已下载数据信息'
    )

    args = parser.parse_args()

    if args.download:
        print(f"下载 HydroSat Water Level 数据到: {args.download}")
        print(f"数据源: https://hydrosat.gis.uni-stuttgart.de/")
        print()

        downloader = HydroSatDownloader(args.download)
        result = downloader.download(extract=not args.no_extract)

        print(f"\n完成! 数据位置: {result}")

        if not args.no_extract:
            count = downloader.get_station_count()
            print(f"站点数量: {count}")

    elif args.info:
        data_dir = Path(args.info)
        if not data_dir.exists():
            print(f"目录不存在: {data_dir}")
        else:
            reader = HydroSatReader()
            files = reader.scan_directory(str(data_dir))
            print(f"数据目录: {data_dir}")
            print(f"站点文件数量: {len(files)}")

            if files:
                print(f"\n示例文件:")
                for f in files[:5]:
                    print(f"  {Path(f).name}")

    else:
        parser.print_help()
