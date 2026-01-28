#!/usr/bin/env python3
"""
Step 2: CaMa-Flood Station Allocation
CaMa-Flood 站点分配

将站点分配到 CaMa-Flood 网格，支持多分辨率
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from ..core.cama_allocator import CamaAllocator, StationAllocation
from ..core.station import Station, StationList
from ..utils.logger import get_logger
from ..constants import RESOLUTIONS


@dataclass
class CamaResult:
    """CaMa 分配结果"""
    stations: List[Station]
    allocations: List[StationAllocation]
    stats: Dict[str, Any]


def run_cama_allocation(stations: List[Station],
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

    # 准备站点数据 - Station now uses elevation and metadata dict
    station_data = [
        {
            'id': s.id,
            'lon': s.lon,
            'lat': s.lat,
            'elevation': s.elevation or 0,
            'satellite': s.metadata.get('satellite', 'Unknown'),
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


def format_allocation_output(stations: List[Station],
                             allocations: List[StationAllocation],
                             resolutions: List[str]) -> List[Dict[str, Any]]:
    """
    格式化输出数据

    Args:
        stations: Station 对象列表
        allocations: 分配结果列表
        resolutions: 分辨率列表

    Returns:
        格式化的输出数据列表
    """
    output = []

    for station, allocation in zip(stations, allocations):
        row = {
            'ID': station.id,
            'station': station.name,
            'dataname': station.source,
            'lon': station.lon,
            'lat': station.lat,
            'satellite': station.metadata.get('satellite', 'Unknown'),
            'elevation': station.elevation or -9999,
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
    """Step 2: Allocate stations to CaMa-Flood grid cells."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__)

        # Get CaMa configuration from global_paths or direct config
        cama_config = self.config.get('global_paths', {}).get('cama_data', {})
        processing = self.config.get('processing', {})

        # cama_root can come from global_paths or direct config
        self.cama_root = cama_config.get('root') or config.get('cama_root')
        self.resolutions = processing.get('cama_resolutions',
                                          cama_config.get('resolutions', RESOLUTIONS))
        self.highres_tag = cama_config.get('highres_tag', '1min')

    def run(self, stations: StationList) -> StationList:
        """
        Run CaMa allocation for all stations and resolutions.

        Args:
            stations: StationList from Step 1

        Returns:
            StationList with CaMa allocation results
        """
        self.logger.info("[Step 2] CaMa 网格分配...")

        total = len(stations)
        if total == 0:
            self.logger.warning("没有站点需要分配")
            return stations

        # Check if cama_root is configured
        if not self.cama_root:
            self.logger.warning("cama_root not configured, skipping CaMa allocation")
            return stations

        # Initialize CamaAllocator
        try:
            allocator = CamaAllocator(
                cama_root=self.cama_root,
                resolutions=self.resolutions,
                highres_tag=self.highres_tag,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"初始化 CaMa 分配器失败: {e}")
            return stations

        # Process each station
        allocated_count = 0
        for i, station in enumerate(stations):
            try:
                # Allocate station to all resolutions
                allocation = allocator.allocate_station(
                    station_id=station.id,
                    lon=station.lon,
                    lat=station.lat,
                    elevation=station.elevation,
                    satellite=station.metadata.get('satellite', 'Unknown')
                )

                # Store results for each resolution
                any_success = False
                for resolution, result in allocation.results.items():
                    result_dict = {
                        'success': result.success,
                        'resolution': resolution,
                        'flag': result.flag,
                        'elevation': result.elevation,
                        'dist_to_mouth': result.dist_to_mouth,
                        'kx1': result.kx1,
                        'ky1': result.ky1,
                        'kx2': result.kx2,
                        'ky2': result.ky2,
                        'dist1': result.dist1,
                        'dist2': result.dist2,
                        'rivwth': result.rivwth,
                        'ix': result.ix,
                        'iy': result.iy,
                        'lon_cama': result.lon_cama,
                        'lat_cama': result.lat_cama,
                    }
                    if result.error_message:
                        result_dict['error'] = result.error_message
                    station.set_cama_result(resolution, result_dict)
                    if result.success:
                        any_success = True

                if any_success:
                    allocated_count += 1

            except Exception as e:
                self.logger.debug(f"分配失败 {station.id}: {e}")

            # Progress logging
            if (i + 1) % 1000 == 0:
                self.logger.info(f"  进度: {i + 1}/{total}")

        # Log summary by resolution
        self.logger.info(f"\n分配统计:")
        for resolution in self.resolutions:
            success_count = sum(
                1 for s in stations
                if resolution in s.cama_results and s.cama_results[resolution].get('success', False)
            )
            self.logger.info(f"  {resolution}: {success_count}/{total} 站点分配成功")

        self.logger.info(f"\n[Step 2] CaMa 分配完成: {allocated_count}/{total} 站点至少有一个分辨率分配成功")
        return stations
