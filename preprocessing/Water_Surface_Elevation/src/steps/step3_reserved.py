#!/usr/bin/env python3
"""
Step 3: Reserved for Future Extensions
预留扩展接口

未来可能的扩展:
- 人类影响指数 (HII) 计算
- 上游大坝影响分析
- 数据质量评分
- 与实测数据交叉验证
"""

from typing import Dict, Any, List

from ..readers import StationMetadata
from ..steps.step2_cama import CamaResult
from ..core.station import Station, StationList
from ..utils.logger import get_logger


def run_reserved(cama_result: CamaResult,
                 config: Dict[str, Any],
                 logger=None) -> CamaResult:
    """
    预留扩展步骤

    当前直接返回输入，不做任何处理。
    未来可在此添加额外的处理逻辑。

    Args:
        cama_result: CaMa 分配结果 (来自 step2)
        config: 配置字典
        logger: 日志记录器

    Returns:
        处理后的结果 (当前直接返回输入)
    """
    log = lambda level, msg: logger and getattr(logger, level)(msg)

    log('info', "Step 3: 预留扩展步骤 (当前跳过)")

    # 未来扩展示例:
    # if config.get('processing', {}).get('calculate_hii', False):
    #     cama_result = calculate_hii(cama_result, config, logger)
    #
    # if config.get('processing', {}).get('dam_analysis', False):
    #     cama_result = analyze_dam_impact(cama_result, config, logger)

    return cama_result


# 预留的扩展函数占位

def calculate_hii(cama_result: CamaResult,
                  config: Dict[str, Any],
                  logger=None) -> CamaResult:
    """
    计算人类影响指数 (HII)

    TODO: 未实现
    """
    raise NotImplementedError("HII 计算尚未实现")


def analyze_dam_impact(cama_result: CamaResult,
                       config: Dict[str, Any],
                       logger=None) -> CamaResult:
    """
    分析上游大坝影响

    TODO: 未实现
    """
    raise NotImplementedError("大坝影响分析尚未实现")


def calculate_quality_score(cama_result: CamaResult,
                            config: Dict[str, Any],
                            logger=None) -> CamaResult:
    """
    计算数据质量评分

    TODO: 未实现
    """
    raise NotImplementedError("质量评分尚未实现")


class Step3Reserved:
    """Step 3: Reserved for future extensions."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__)

    def run(self, stations: StationList) -> StationList:
        """
        Run reserved processing step.

        Currently a pass-through; reserved for future extensions like:
        - Human Impact Index (HII) calculation
        - Upstream dam analysis
        - Data quality scoring

        Args:
            stations: StationList from Step 2

        Returns:
            StationList (unchanged for now)
        """
        self.logger.info("Step 3: 预留扩展 (当前跳过)")
        return stations
