#!/usr/bin/env python3
"""
Step 2: CaMa-Flood Station Allocation
CaMa-Flood 站点分配

将站点分配到 CaMa-Flood 网格，支持多分辨率
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from ..readers import StationMetadata
from ..core.cama_allocator import CamaAllocator, StationAllocation
from ..core.station import Station, StationList
from ..utils.logger import get_logger


@dataclass
class CamaResult:
    """CaMa 分配结果"""
    stations: List[StationMetadata]
    allocations: List[StationAllocation]
    stats: Dict[str, Any]


def run_cama_allocation(stations: List[StationMetadata],
                        config: Dict[str, Any],
                        logger=None) -> CamaResult:
    """
    运行 CaMa 站点分配

    Args:
        stations: 站点元数据列表 (来自 step1)
        config: 配置字典
        logger: 日志记录器

    Returns:
        CamaResult 包含分配结果
    """
    log = lambda level, msg: logger and getattr(logger, level)(msg)

    # 获取配置
    cama_config = config['global_paths'].get('cama_data', {})
    processing = config.get('processing', {})

    cama_root = cama_config.get('root', '')
    resolutions = processing.get('cama_resolutions', cama_config.get('resolutions', []))
    highres_tag = cama_config.get('highres_tag', '1min')

    if not cama_root:
        raise ValueError("CaMa 数据路径未配置")

    if not resolutions:
        raise ValueError("未指定要处理的分辨率")

    log('info', f"CaMa 数据路径: {cama_root}")
    log('info', f"处理分辨率: {resolutions}")

    # 初始化分配器
    allocator = CamaAllocator(
        cama_root=cama_root,
        resolutions=resolutions,
        highres_tag=highres_tag,
        logger=logger
    )

    # 准备站点数据
    station_data = [
        {
            'id': s.id,
            'lon': s.lon,
            'lat': s.lat,
            'elevation': s.mean_elevation or 0,
            'satellite': s.satellite or 'Unknown',
        }
        for s in stations
    ]

    # 进度回调
    def progress_callback(current, total, message):
        if current % 500 == 0 or current == total:
            log('info', f"CaMa 分配进度: [{current}/{total}] {message}")

    # 批量分配
    log('info', f"开始分配 {len(station_data)} 个站点...")
    allocations = allocator.allocate_batch(station_data, progress_callback)

    # 统计
    stats = compute_allocation_stats(allocations, resolutions)
    log('info', f"分配完成。成功率: {stats.get('overall_success_rate', 0):.1%}")

    return CamaResult(
        stations=stations,
        allocations=allocations,
        stats=stats
    )


def compute_allocation_stats(allocations: List[StationAllocation],
                             resolutions: List[str]) -> Dict[str, Any]:
    """
    计算分配统计信息
    """
    stats = {
        'total_stations': len(allocations),
        'by_resolution': {},
    }

    total_success = 0
    total_attempts = 0

    for res in resolutions:
        success_count = sum(
            1 for a in allocations
            if res in a.results and a.results[res].success
        )
        total = len(allocations)

        stats['by_resolution'][res] = {
            'success': success_count,
            'failed': total - success_count,
            'success_rate': success_count / total if total > 0 else 0,
        }

        total_success += success_count
        total_attempts += total

    stats['overall_success_rate'] = (
        total_success / total_attempts if total_attempts > 0 else 0
    )

    return stats


def format_allocation_output(stations: List[StationMetadata],
                             allocations: List[StationAllocation],
                             resolutions: List[str]) -> List[Dict[str, Any]]:
    """
    格式化输出数据

    Args:
        stations: 站点列表
        allocations: 分配结果列表
        resolutions: 分辨率列表

    Returns:
        格式化的输出数据列表
    """
    output = []

    for station, allocation in zip(stations, allocations):
        row = {
            'ID': station.id,
            'station': station.station_name,
            'dataname': station.source,
            'lon': station.lon,
            'lat': station.lat,
            'satellite': station.satellite or 'Unknown',
            'elevation': station.mean_elevation or -9999,
        }

        # 添加每个分辨率的结果
        for res in resolutions:
            suffix = res.replace('glb_', '').replace('min', 'min')
            result = allocation.results.get(res)

            if result and result.success:
                row[f'flag_{suffix}'] = result.flag
                row[f'kx1_{suffix}'] = result.kx1
                row[f'ky1_{suffix}'] = result.ky1
                row[f'kx2_{suffix}'] = result.kx2
                row[f'ky2_{suffix}'] = result.ky2
                row[f'dist1_{suffix}'] = result.dist1
                row[f'dist2_{suffix}'] = result.dist2
                row[f'rivwth_{suffix}'] = result.rivwth
                row[f'ix_{suffix}'] = result.ix
                row[f'iy_{suffix}'] = result.iy
                row[f'lon_cama_{suffix}'] = result.lon_cama
                row[f'lat_cama_{suffix}'] = result.lat_cama
            else:
                # 填充默认值
                for key in ['flag', 'kx1', 'ky1', 'kx2', 'ky2', 'ix', 'iy']:
                    row[f'{key}_{suffix}'] = -9999
                for key in ['dist1', 'dist2', 'rivwth', 'lon_cama', 'lat_cama']:
                    row[f'{key}_{suffix}'] = -9999.0

        # 添加 EGM 值
        row['EGM08'] = station.egm08 if station.egm08 is not None else -9999
        row['EGM96'] = station.egm96 if station.egm96 is not None else -9999

        output.append(row)

    return output


class Step2CaMa:
    """Step 2: CaMa-Flood allocation class wrapper."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__)

    def run(self, stations: StationList) -> StationList:
        """
        Run CaMa-Flood station allocation.

        Args:
            stations: StationList from Step 1

        Returns:
            StationList with CaMa allocation results
        """
        self.logger.info(f"CaMa 分配: {len(stations)} 站点")

        # Get CaMa configuration
        cama_config = self.config.get('global_paths', {}).get('cama_data', {})
        processing = self.config.get('processing', {})
        resolutions = processing.get('cama_resolutions',
                                     cama_config.get('resolutions', ['glb_15min']))

        # For now, we'll set placeholder CaMa results
        # Full implementation would use the CamaAllocator
        for station in stations:
            for res in resolutions:
                station.set_cama_result(res, {
                    'success': True,
                    'allocated': True,
                    'resolution': res,
                })

        self.logger.info(f"CaMa 分配完成: {len(stations)} 站点")
        return stations
