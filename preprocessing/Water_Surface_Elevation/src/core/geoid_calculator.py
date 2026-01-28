#!/usr/bin/env python3
"""
Geoid Calculator Module
计算 EGM2008 和 EGM96 大地水准面起伏值

数据文件下载地址:
    https://sourceforge.net/projects/geographiclib/files/geoids-distrib/
    推荐下载:
    - egm96-5.zip (11 MB) -> 解压得到 egm96-5.pgm
    - egm2008-5.zip (11 MB) -> 解压得到 egm2008-5.pgm

使用方法:
    from geoid_calculator import GeoidCalculator

    calc = GeoidCalculator()
    egm08, egm96 = calc.get_undulation(lat=43.7569, lon=-0.7760)
    print(f"EGM2008: {egm08:.2f} m, EGM96: {egm96:.2f} m")
"""

import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional, List
import warnings


class GeoidCalculator:
    """
    计算 EGM2008 和 EGM96 大地水准面起伏值

    支持两种后端:
    1. PyGeodesy (推荐，纯 Python)
    2. GeographicLib 命令行工具 (备选)
    """

    # 默认数据目录
    DEFAULT_DATA_DIR = Path(__file__).parent / "geoid_data"

    def _validate_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Validate and sanitize coordinate input before subprocess calls.

        Args:
            lat: Latitude value to validate
            lon: Longitude value to validate

        Returns:
            Tuple of validated (lat, lon) as floats

        Raises:
            ValueError: If coordinates are invalid type or out of range
        """
        # Type validation and conversion
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid coordinate type: lat={lat}, lon={lon}") from e

        # Range validation
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude out of range: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude out of range: {lon}")

        return lat, lon

    # 下载 URL
    DOWNLOAD_URLS = {
        'egm96-5': 'https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-5.zip/download',
        'egm2008-5': 'https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-5.zip/download',
        'egm96-15': 'https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-15.zip/download',
        'egm2008-1': 'https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-1.zip/download',
    }

    def __init__(self, data_dir: Optional[str] = None,
                 egm96_model: str = 'egm96-5',
                 egm2008_model: str = 'egm2008-5'):
        """
        初始化 GeoidCalculator

        Args:
            data_dir: 数据文件目录，默认为脚本同目录下的 geoid_data/
            egm96_model: EGM96 模型名称 ('egm96-5' 或 'egm96-15')
            egm2008_model: EGM2008 模型名称 ('egm2008-5', 'egm2008-2_5' 或 'egm2008-1')
        """
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self.egm96_model = egm96_model
        self.egm2008_model = egm2008_model

        self._pygeodesy_available = False
        self._egm96_geoid = None
        self._egm2008_geoid = None

        # 尝试初始化 PyGeodesy
        self._init_pygeodesy()

    def _init_pygeodesy(self):
        """尝试初始化 PyGeodesy 后端"""
        try:
            from pygeodesy.geoids import GeoidPGM
            self._GeoidPGM = GeoidPGM
            self._pygeodesy_available = True

            # 检查数据文件是否存在
            egm96_path = self.data_dir / f"{self.egm96_model}.pgm"
            egm2008_path = self.data_dir / f"{self.egm2008_model}.pgm"

            if egm96_path.exists():
                self._egm96_geoid = GeoidPGM(str(egm96_path))
                print(f"已加载 EGM96 模型: {egm96_path}")
            else:
                warnings.warn(f"EGM96 数据文件不存在: {egm96_path}\n"
                            f"请运行 calc.download_data() 下载")

            if egm2008_path.exists():
                self._egm2008_geoid = GeoidPGM(str(egm2008_path))
                print(f"已加载 EGM2008 模型: {egm2008_path}")
            else:
                warnings.warn(f"EGM2008 数据文件不存在: {egm2008_path}\n"
                            f"请运行 calc.download_data() 下载")

        except ImportError:
            warnings.warn("PyGeodesy 未安装，请运行: pip install pygeodesy\n"
                         "将尝试使用 GeographicLib 命令行工具作为备选")
            self._pygeodesy_available = False

    def download_data(self, models: Optional[List[str]] = None):
        """
        下载大地水准面数据文件

        Args:
            models: 要下载的模型列表，默认为 ['egm96-5', 'egm2008-5']
        """
        if models is None:
            models = [self.egm96_model, self.egm2008_model]

        self.data_dir.mkdir(parents=True, exist_ok=True)

        for model in models:
            if model not in self.DOWNLOAD_URLS:
                print(f"未知模型: {model}，跳过")
                continue

            pgm_path = self.data_dir / f"{model}.pgm"
            if pgm_path.exists():
                print(f"{model}.pgm 已存在，跳过下载")
                continue

            zip_path = self.data_dir / f"{model}.zip"
            url = self.DOWNLOAD_URLS[model]

            print(f"正在下载 {model}...")
            print(f"URL: {url}")
            print("(这可能需要几分钟，取决于网络速度)")

            try:
                # 下载 zip 文件
                urllib.request.urlretrieve(url, zip_path)
                print(f"下载完成: {zip_path}")

                # 解压
                print(f"正在解压...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)

                # 处理嵌套目录结构 (zip 文件解压后可能在 geoids/ 子目录)
                nested_pgm = self.data_dir / "geoids" / f"{model}.pgm"
                if nested_pgm.exists() and not pgm_path.exists():
                    import shutil
                    shutil.move(str(nested_pgm), str(pgm_path))
                    # 清理空的 geoids 目录
                    geoids_dir = self.data_dir / "geoids"
                    if geoids_dir.exists() and not any(geoids_dir.iterdir()):
                        geoids_dir.rmdir()

                print(f"解压完成: {pgm_path}")

                # 删除 zip 文件
                zip_path.unlink()

            except Exception as e:
                print(f"下载失败: {e}")
                print(f"请手动下载: {url}")
                print(f"并解压到: {self.data_dir}")

        # 重新初始化
        self._init_pygeodesy()

    def get_undulation(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        计算指定位置的 EGM2008 和 EGM96 大地水准面起伏值

        Args:
            lat: 纬度 (度)
            lon: 经度 (度)

        Returns:
            (egm2008_undulation, egm96_undulation) 单位: 米

        Raises:
            ValueError: If coordinates are invalid type or out of range
        """
        # Validate coordinates before any processing
        lat, lon = self._validate_coordinates(lat, lon)

        if self._pygeodesy_available:
            return self._get_undulation_pygeodesy(lat, lon)
        else:
            return self._get_undulation_cli(lat, lon)

    def _get_undulation_pygeodesy(self, lat: float, lon: float) -> Tuple[float, float]:
        """使用 PyGeodesy 计算"""
        egm2008 = 0.0
        egm96 = 0.0

        if self._egm2008_geoid is not None:
            egm2008 = self._egm2008_geoid.height(lat, lon)
        else:
            warnings.warn("EGM2008 模型未加载")

        if self._egm96_geoid is not None:
            egm96 = self._egm96_geoid.height(lat, lon)
        else:
            warnings.warn("EGM96 模型未加载")

        return egm2008, egm96

    def _get_undulation_cli(self, lat: float, lon: float) -> Tuple[float, float]:
        """使用 GeographicLib 命令行工具计算 (备选方案)"""
        egm2008 = 0.0
        egm96 = 0.0

        try:
            # EGM2008
            result = subprocess.run(
                ['GeoidEval', '-n', self.egm2008_model, str(lat), str(lon)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                egm2008 = float(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"GeoidEval 计算 EGM2008 失败: {e}")

        try:
            # EGM96
            result = subprocess.run(
                ['GeoidEval', '-n', self.egm96_model, str(lat), str(lon)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                egm96 = float(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"GeoidEval 计算 EGM96 失败: {e}")

        return egm2008, egm96

    def get_undulation_batch(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        批量计算多个位置的大地水准面起伏值

        Args:
            coords: [(lat1, lon1), (lat2, lon2), ...] 坐标列表

        Returns:
            [(egm2008_1, egm96_1), (egm2008_2, egm96_2), ...] 结果列表
        """
        results = []
        total = len(coords)

        for i, (lat, lon) in enumerate(coords):
            if (i + 1) % 100 == 0:
                print(f"处理进度: {i + 1}/{total}")
            results.append(self.get_undulation(lat, lon))

        return results

    def is_ready(self) -> bool:
        """检查计算器是否就绪"""
        return (self._pygeodesy_available and
                self._egm96_geoid is not None and
                self._egm2008_geoid is not None)


def parse_hydroweb_file(filepath: str) -> dict:
    """
    解析 HydroWeb 数据文件，提取站点元数据

    Args:
        filepath: HydroWeb 文件路径

    Returns:
        包含站点信息的字典
    """
    metadata = {}

    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break

            if '::' in line:
                key, value = line[1:].split('::', 1)
                key = key.strip()
                value = value.strip()
                metadata[key] = value

    return metadata


def generate_station_list(hydroweb_dir: str, output_file: str,
                         geoid_calc: Optional[GeoidCalculator] = None):
    """
    从 HydroWeb 数据文件生成包含 EGM08 和 EGM96 的站点列表

    Args:
        hydroweb_dir: HydroWeb 数据文件目录
        output_file: 输出文件路径
        geoid_calc: GeoidCalculator 实例，如果为 None 则自动创建
    """
    import glob

    if geoid_calc is None:
        geoid_calc = GeoidCalculator()
        if not geoid_calc.is_ready():
            print("正在下载大地水准面数据...")
            geoid_calc.download_data()

    # 查找所有 HydroWeb 文件
    files = glob.glob(os.path.join(hydroweb_dir, "hydroprd_*.txt"))
    print(f"找到 {len(files)} 个 HydroWeb 文件")

    stations = []

    for i, filepath in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"处理进度: {i + 1}/{len(files)}")

        try:
            meta = parse_hydroweb_file(filepath)

            # 提取必要信息
            station_id = meta.get('ID', '').strip()
            lon = float(meta.get('REFERENCE LONGITUDE', 0))
            lat = float(meta.get('REFERENCE LATITUDE', 0))

            # 从文件名提取站点名称
            filename = os.path.basename(filepath)
            station_name = filename.replace('hydroprd_', '').replace('_exp.txt', '').replace('.txt', '')

            # 提取其他信息
            river = meta.get('RIVER', 'Unknown')
            basin = meta.get('BASIN', 'Unknown')
            elevation = float(meta.get('MEAN ALTITUDE(M.mm)', 0))

            # HydroWeb 提供的 EGM2008 值
            egm08_hydroweb = float(meta.get('GEOID ONDULATION AT REF POSITION(M.mm)', 0))

            # 计算 EGM96 值 (HydroWeb 不提供)
            egm08_calc, egm96 = geoid_calc.get_undulation(lat, lon)

            # 使用 HydroWeb 提供的 EGM2008 值 (更准确)
            egm08 = egm08_hydroweb if egm08_hydroweb != 0 else egm08_calc

            # 提取卫星信息
            missions = meta.get('MISSION(S)-TRACK(S)', 'Unknown')
            satellite = missions.split('-')[0] if missions else 'Unknown'

            # 提取日期
            start_date = meta.get('FIRST DATE IN DATASET', '')
            end_date = meta.get('LAST DATE IN DATASET', '')
            status = meta.get('STATUS', 'Unknown')

            stations.append({
                'ID': station_id,
                'station': station_name,
                'River': river,
                'Basin': basin,
                'lon': lon,
                'lat': lat,
                'elevation': elevation,
                'EGM08': egm08,
                'EGM96': egm96,
                'satellite': satellite,
                'Start_Date': start_date,
                'End_Date': end_date,
                'Status': status
            })

        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {e}")
            continue

    # 写入输出文件
    print(f"写入 {len(stations)} 个站点到 {output_file}")

    with open(output_file, 'w') as f:
        # 写入标题行
        f.write(f"{'ID':>13} {'station':64} {'River':20} {'Basin':20} "
                f"{'lon':>10} {'lat':>10} {'elevation':>10} {'EGM08':>10} {'EGM96':>10} "
                f"{'satellite':15} {'Start_Date':12} {'End_Date':12} {'Status':12}\n")

        # 写入数据行
        for s in stations:
            f.write(f"{s['ID']:>13} {s['station']:64} {s['River']:20} {s['Basin']:20} "
                    f"{s['lon']:10.4f} {s['lat']:10.4f} {s['elevation']:10.2f} "
                    f"{s['EGM08']:10.2f} {s['EGM96']:10.2f} "
                    f"{s['satellite']:15} {s['Start_Date']:12} {s['End_Date']:12} {s['Status']:12}\n")

    print("完成!")
    return stations


# 命令行接口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='计算 EGM2008 和 EGM96 大地水准面起伏值')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # download 子命令
    download_parser = subparsers.add_parser('download', help='下载大地水准面数据')
    download_parser.add_argument('--data-dir', default=None, help='数据存储目录')

    # calc 子命令
    calc_parser = subparsers.add_parser('calc', help='计算单点的 EGM 值')
    calc_parser.add_argument('lat', type=float, help='纬度')
    calc_parser.add_argument('lon', type=float, help='经度')
    calc_parser.add_argument('--data-dir', default=None, help='数据目录')

    # generate 子命令
    gen_parser = subparsers.add_parser('generate', help='从 HydroWeb 文件生成站点列表')
    gen_parser.add_argument('hydroweb_dir', help='HydroWeb 数据目录')
    gen_parser.add_argument('output_file', help='输出文件路径')
    gen_parser.add_argument('--data-dir', default=None, help='大地水准面数据目录')

    args = parser.parse_args()

    if args.command == 'download':
        calc = GeoidCalculator(data_dir=args.data_dir)
        calc.download_data()

    elif args.command == 'calc':
        calc = GeoidCalculator(data_dir=args.data_dir)
        if not calc.is_ready():
            print("数据文件未就绪，正在下载...")
            calc.download_data()

        egm08, egm96 = calc.get_undulation(args.lat, args.lon)
        print(f"位置: ({args.lat}, {args.lon})")
        print(f"EGM2008: {egm08:.2f} m")
        print(f"EGM96:   {egm96:.2f} m")
        print(f"差值:    {egm08 - egm96:.2f} m")

    elif args.command == 'generate':
        calc = GeoidCalculator(data_dir=args.data_dir)
        generate_station_list(args.hydroweb_dir, args.output_file, calc)

    else:
        parser.print_help()
