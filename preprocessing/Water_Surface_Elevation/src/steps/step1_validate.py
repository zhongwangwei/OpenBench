#!/usr/bin/env python3
"""
Step 1: Validation and EGM Calculation
验证数据并计算 EGM08/EGM96

流程:
1. 使用对应 Reader 扫描所有站点文件
2. 解析每个站点的元数据
3. 验证数据完整性
4. 计算 EGM08/EGM96 (使用 GeoidCalculator)
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from ..readers import get_reader, StationMetadata
from ..core.geoid_calculator import GeoidCalculator
from ..core.station import Station, StationList
from ..utils.logger import ProgressLogger, get_logger


@dataclass
class ValidationIssue:
    """验证问题"""
    level: str  # 'error', 'warning', 'info'
    code: str
    message: str
    station_id: Optional[str] = None


@dataclass
class ValidationResult:
    """验证结果"""
    stations: List[StationMetadata]
    issues: List[ValidationIssue]
    stats: Dict[str, Any]


def run_validation(config: Dict[str, Any], logger=None) -> ValidationResult:
    """
    运行验证和 EGM 计算

    Args:
        config: 配置字典
        logger: 日志记录器

    Returns:
        ValidationResult 包含站点列表、问题列表和统计信息
    """
    log = lambda level, msg: logger and getattr(logger, level)(msg)

    # 获取配置
    source = config['dataset']['source']
    source_path = config['global_paths']['data_sources'].get(source)
    filters = config.get('filters', {})
    validation_rules = config.get('validation_rules', {})
    processing = config.get('processing', {})

    if not source_path:
        raise ValueError(f"数据源路径未配置: {source}")

    log('info', f"开始处理数据源: {source}")
    log('info', f"数据路径: {source_path}")

    # 1. 读取站点数据
    reader = get_reader(source, logger=logger)

    def progress_callback(current, total, message):
        if current % 500 == 0 or current == total:
            log('info', f"读取进度: [{current}/{total}] {message}")

    stations = reader.read_all_stations(
        source_path,
        progress_callback=progress_callback,
        filters=filters
    )

    log('info', f"读取到 {len(stations)} 个站点")

    # 2. 验证数据
    issues = []
    valid_stations = []

    for station in stations:
        station_issues = validate_station(station, validation_rules)
        issues.extend(station_issues)

        # 只有没有 error 级别问题的站点才保留
        has_error = any(i.level == 'error' for i in station_issues)
        if not has_error:
            valid_stations.append(station)

    log('info', f"验证通过 {len(valid_stations)} 个站点，"
                f"发现 {len(issues)} 个问题")

    # 3. 计算 EGM08/EGM96
    if processing.get('calculate_egm', True):
        log('info', "计算 EGM08/EGM96 值...")
        valid_stations = calculate_egm_values(
            valid_stations, config, logger
        )

    # 4. 检测重复站点
    duplicates = detect_duplicates(
        valid_stations,
        validation_rules.get('duplicates', {})
    )
    for dup in duplicates:
        issues.append(ValidationIssue(
            level='warning',
            code='DUPLICATE_STATION',
            message=f"重复站点: {dup[0]} 和 {dup[1]} 距离 {dup[2]:.0f}m",
        ))

    # 5. 统计信息
    stats = compute_statistics(valid_stations)

    return ValidationResult(
        stations=valid_stations,
        issues=issues,
        stats=stats
    )


def validate_station(station: StationMetadata,
                     rules: Dict[str, Any]) -> List[ValidationIssue]:
    """
    验证单个站点

    Args:
        station: 站点元数据
        rules: 验证规则

    Returns:
        问题列表
    """
    issues = []
    coord_rules = rules.get('coordinates', {})
    quality_rules = rules.get('quality', {})
    elevation_rules = rules.get('elevation', {})

    # 坐标范围检查
    lat_min = coord_rules.get('lat_min', -90)
    lat_max = coord_rules.get('lat_max', 90)
    lon_min = coord_rules.get('lon_min', -180)
    lon_max = coord_rules.get('lon_max', 180)

    if not (lat_min <= station.lat <= lat_max):
        issues.append(ValidationIssue(
            level='error',
            code='INVALID_LATITUDE',
            message=f"纬度超出范围: {station.lat}",
            station_id=station.id
        ))

    if not (lon_min <= station.lon <= lon_max):
        issues.append(ValidationIssue(
            level='error',
            code='INVALID_LONGITUDE',
            message=f"经度超出范围: {station.lon}",
            station_id=station.id
        ))

    # 观测次数检查
    min_obs = quality_rules.get('min_observations', 10)
    if station.num_observations < min_obs:
        issues.append(ValidationIssue(
            level='warning',
            code='LOW_OBSERVATIONS',
            message=f"观测次数不足: {station.num_observations} < {min_obs}",
            station_id=station.id
        ))

    # 高程范围检查
    if station.mean_elevation is not None:
        elev_min = elevation_rules.get('min_value', -500)
        elev_max = elevation_rules.get('max_value', 6000)

        if not (elev_min <= station.mean_elevation <= elev_max):
            issues.append(ValidationIssue(
                level='warning',
                code='UNUSUAL_ELEVATION',
                message=f"高程异常: {station.mean_elevation:.2f}m",
                station_id=station.id
            ))

    return issues


def calculate_egm_values(stations: List[StationMetadata],
                         config: Dict[str, Any],
                         logger=None) -> List[StationMetadata]:
    """
    计算所有站点的 EGM08/EGM96 值

    Args:
        stations: 站点列表
        config: 配置
        logger: 日志记录器

    Returns:
        更新后的站点列表
    """
    log = lambda level, msg: logger and getattr(logger, level)(msg)

    # 获取 EGM 配置
    geoid_config = config['global_paths'].get('geoid_data', {})
    processing = config.get('processing', {})

    geoid_root = geoid_config.get('root', './geoid_data')
    egm96_model = processing.get('egm96_model', geoid_config.get('egm96_model', 'egm96-5'))
    egm2008_model = processing.get('egm2008_model', geoid_config.get('egm2008_model', 'egm2008-1'))

    # 初始化 GeoidCalculator
    try:
        calc = GeoidCalculator(
            data_dir=geoid_root,
            egm96_model=egm96_model,
            egm2008_model=egm2008_model
        )

        if not calc.is_ready():
            log('warning', "EGM 数据文件未就绪，尝试下载...")
            calc.download_data()

    except Exception as e:
        log('error', f"初始化 GeoidCalculator 失败: {e}")
        log('warning', "跳过 EGM 计算")
        return stations

    # 计算每个站点的 EGM 值
    total = len(stations)
    for i, station in enumerate(stations):
        if (i + 1) % 500 == 0 or i == total - 1:
            log('info', f"EGM 计算进度: [{i + 1}/{total}]")

        try:
            egm08, egm96 = calc.get_undulation(station.lat, station.lon)
            station.egm08 = egm08
            station.egm96 = egm96
        except Exception as e:
            log('warning', f"计算 EGM 失败 (站点 {station.id}): {e}")
            station.egm08 = None
            station.egm96 = None

    return stations


def detect_duplicates(stations: List[StationMetadata],
                      rules: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """
    检测重复站点

    Args:
        stations: 站点列表
        rules: 重复检测规则

    Returns:
        重复站点对列表 [(id1, id2, distance_m), ...]
    """
    import math

    distance_threshold = rules.get('distance_threshold_m', 100)
    duplicates = []

    # 简单的 O(n²) 检测
    for i, s1 in enumerate(stations):
        for j, s2 in enumerate(stations[i + 1:], i + 1):
            dist = haversine_distance(s1.lat, s1.lon, s2.lat, s2.lon)
            if dist < distance_threshold:
                duplicates.append((s1.id, s2.id, dist))

    return duplicates


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    计算两点间的 Haversine 距离 (米)
    """
    import math

    R = 6371000  # 地球半径 (米)

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def compute_statistics(stations: List[StationMetadata]) -> Dict[str, Any]:
    """
    计算站点统计信息
    """
    if not stations:
        return {}

    lons = [s.lon for s in stations]
    lats = [s.lat for s in stations]
    obs_counts = [s.num_observations for s in stations]
    elevations = [s.mean_elevation for s in stations if s.mean_elevation is not None]

    stats = {
        'total_stations': len(stations),
        'bbox': {
            'west': min(lons),
            'east': max(lons),
            'south': min(lats),
            'north': max(lats),
        },
        'observations': {
            'total': sum(obs_counts),
            'mean': sum(obs_counts) / len(obs_counts),
            'min': min(obs_counts),
            'max': max(obs_counts),
        },
    }

    if elevations:
        stats['elevation'] = {
            'mean': sum(elevations) / len(elevations),
            'min': min(elevations),
            'max': max(elevations),
        }

    # 按数据源统计
    sources = {}
    for s in stations:
        src = s.source
        if src not in sources:
            sources[src] = 0
        sources[src] += 1
    stats['by_source'] = sources

    return stats


class Step1Validate:
    """Step 1: Validation class wrapper."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__)

    def run(self, sources: List[str]) -> StationList:
        """
        Run validation for specified sources.

        Args:
            sources: List of source names (e.g., ['hydroweb', 'cgls'])

        Returns:
            StationList with validated stations
        """
        station_list = StationList()

        for source in sources:
            # Create source-specific config
            source_config = {**self.config, 'dataset': {'source': source}}

            self.logger.info(f"验证数据源: {source}")

            try:
                result = run_validation(source_config, self.logger)

                # Convert StationMetadata to Station objects
                for sm in result.stations:
                    station = Station(
                        id=sm.id,
                        name=sm.station_name or sm.id,
                        lon=sm.lon,
                        lat=sm.lat,
                        source=sm.source,
                        elevation=sm.mean_elevation or 0.0,
                        num_observations=sm.num_observations,
                        egm08=sm.egm08,
                        egm96=sm.egm96,
                    )
                    station_list.add(station)

                self.logger.info(f"  {source}: {len(result.stations)} 站点验证通过")

            except Exception as e:
                self.logger.error(f"验证 {source} 失败: {e}")

        return station_list
