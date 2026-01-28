"""
Data Readers for WSE Pipeline
数据源读取器模块

支持的数据源:
- hydroweb: HydroWeb Theia/CNES
- cgls: Copernicus Global Land Service
- icesat: ICESat GLA14
- hydrosat: HydroSat (University of Stuttgart)

使用方法:
    # 方式1: 直接获取读取器 (数据必须已存在)
    reader = get_reader('hydroweb')
    stations = reader.read_all_stations('/path/to/data')

    # 方式2: 确保数据存在后获取读取器 (自动下载)
    reader, data_path = ensure_data_and_get_reader(
        source='hydrosat',
        config=config,
        logger=logger
    )
    stations = reader.read_all_stations(str(data_path))
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .base_reader import BaseReader, StationMetadata, create_station_from_reader
from ..exceptions import ReaderError
from .hydroweb_reader import HydroWebReader
from .cgls_reader import CGLSReader
from .icesat_reader import ICESatReader
from .hydrosat_reader import HydroSatReader, HydroSatDownloader, download_hydrosat
from .downloader import (
    BaseDownloader,
    HydroSatDownloader,
    HydroWebDownloader,
    CGLSDownloader,
    ICESatDownloader,
    DataDownloadManager,
    get_downloader,
)


READERS = {
    'hydroweb': HydroWebReader,
    'cgls': CGLSReader,
    'icesat': ICESatReader,
    'hydrosat': HydroSatReader,
}

DOWNLOADERS = {
    'hydroweb': HydroWebDownloader,
    'cgls': CGLSDownloader,
    'icesat': ICESatDownloader,
    'hydrosat': HydroSatDownloader,
}


def get_reader(source: str, logger=None) -> BaseReader:
    """
    获取指定数据源的读取器

    Args:
        source: 数据源名称 (hydroweb, cgls, icesat, hydrosat)
        logger: 日志记录器

    Returns:
        对应的读取器实例
    """
    if source not in READERS:
        raise ValueError(f"未知的数据源: {source}。支持的数据源: {list(READERS.keys())}")

    return READERS[source](logger=logger)


def ensure_data_and_get_reader(
    source: str,
    config: Dict[str, Any],
    logger=None,
    auto_download: bool = True,
    **download_kwargs
) -> Tuple[BaseReader, Path]:
    """
    确保数据存在并获取读取器

    自动检测数据是否存在，如果不存在则尝试下载。

    Args:
        source: 数据源名称
        config: 配置字典 (包含路径信息)
        logger: 日志记录器
        auto_download: 是否自动下载 (默认 True)
        **download_kwargs: 下载器额外参数 (如认证信息)

    Returns:
        (读取器实例, 数据目录路径)
    """
    if source not in READERS:
        raise ValueError(f"未知的数据源: {source}")

    # 获取数据路径配置
    data_sources = config.get('global_paths', {}).get('data_sources', {})
    source_config = data_sources.get(source, {})

    # 确定数据目录
    data_path = source_config.get('root')
    if not data_path:
        # 使用默认路径
        data_path = f"./data/{source}"

    data_path = Path(data_path)

    # 创建读取器
    reader = get_reader(source, logger)

    # 检查数据是否存在
    try:
        files = reader.scan_directory(str(data_path))
        if files:
            if logger:
                logger.info(f"{source} 数据已就绪: {len(files)} 个文件")
            return reader, data_path
    except (FileNotFoundError, ValueError):
        pass

    # 数据不存在，尝试下载
    if not auto_download:
        raise FileNotFoundError(
            f"{source} 数据不存在: {data_path}\n"
            f"请手动下载或设置 auto_download=True"
        )

    if logger:
        logger.info(f"{source} 数据不存在，尝试下载...")

    # 获取下载器
    downloader_cls = DOWNLOADERS.get(source)
    if not downloader_cls:
        raise ValueError(f"没有可用的下载器: {source}")

    # 准备下载器参数
    downloader_kwargs = {}

    if source == 'hydroweb':
        # HydroWeb 需要检查本地 ZIP 文件
        zip_file = source_config.get('zip_file')
        if zip_file:
            downloader_kwargs['zip_source'] = zip_file

    elif source in ['cgls', 'icesat']:
        # 这些数据源可能需要从已有路径读取
        downloader_kwargs['data_source'] = source_config.get('root')

    # 合并用户提供的参数
    downloader_kwargs.update(download_kwargs)

    # 创建下载器并下载
    downloader = downloader_cls(
        output_dir=str(data_path.parent),
        logger=logger,
        **{k: v for k, v in downloader_kwargs.items() if k in ['zip_source', 'data_source']}
    )

    try:
        downloaded_path = downloader.ensure_data(**download_kwargs)
        if logger:
            logger.info(f"{source} 数据下载完成: {downloaded_path}")
        return reader, downloaded_path

    except NotImplementedError as e:
        # 数据需要手动下载
        if logger:
            logger.warning(str(e))
        raise


def check_data_status(config: Dict[str, Any], logger=None) -> Dict[str, Dict]:
    """
    检查所有数据源状态

    Args:
        config: 配置字典
        logger: 日志记录器

    Returns:
        状态字典 {source: {exists: bool, path: str, count: int}}
    """
    status = {}

    data_sources = config.get('global_paths', {}).get('data_sources', {})

    for source in READERS.keys():
        source_config = data_sources.get(source, {})
        data_path = source_config.get('root', f'./data/{source}')
        data_path = Path(data_path)

        source_status = {
            'exists': False,
            'path': str(data_path),
            'count': 0,
            'requires_auth': DOWNLOADERS[source].requires_auth if source in DOWNLOADERS else True,
        }

        try:
            reader = get_reader(source)
            files = reader.scan_directory(str(data_path))
            source_status['exists'] = len(files) > 0
            source_status['count'] = len(files)
        except (FileNotFoundError, ValueError):
            pass

        status[source] = source_status

    return status


def print_data_status(config: Dict[str, Any]):
    """打印数据源状态"""
    status = check_data_status(config)

    print("\n数据源状态:")
    print("-" * 70)
    print(f"{'数据源':12} {'状态':10} {'文件数':10} {'路径'}")
    print("-" * 70)

    for source, info in status.items():
        status_str = "✓ 就绪" if info['exists'] else "✗ 缺失"
        count_str = str(info['count']) if info['exists'] else "-"
        auth_str = "" if info['exists'] else ("(需认证)" if info['requires_auth'] else "(可下载)")

        print(f"{source:12} {status_str:10} {count_str:10} {info['path']} {auth_str}")

    print("-" * 70)


__all__ = [
    # 基类
    'BaseReader',
    'StationMetadata',  # Alias for Station (backward compatibility)
    'create_station_from_reader',
    # 读取器
    'HydroWebReader',
    'CGLSReader',
    'ICESatReader',
    'HydroSatReader',
    # 下载器
    'BaseDownloader',
    'HydroSatDownloader',
    'HydroWebDownloader',
    'CGLSDownloader',
    'ICESatDownloader',
    'DataDownloadManager',
    # 工厂函数
    'get_reader',
    'get_downloader',
    'ensure_data_and_get_reader',
    # 辅助函数
    'check_data_status',
    'print_data_status',
    'download_hydrosat',
    # 异常
    'ReaderError',
]
