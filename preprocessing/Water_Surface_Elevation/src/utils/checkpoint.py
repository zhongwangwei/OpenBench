#!/usr/bin/env python3
"""
Checkpoint System for WSE Pipeline
断点续传系统
"""

import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class CheckpointData:
    """检查点数据结构"""
    step: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    timestamp: datetime
    config_hash: str
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class CheckpointManager:
    """检查点管理器"""

    STEPS = ['validate', 'cama', 'reserved', 'merge']

    def __init__(self, checkpoint_dir: str, config: Dict[str, Any]):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点目录
            config: 当前配置 (用于计算哈希)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = self._compute_config_hash(config)
        self._checkpoints: Dict[str, CheckpointData] = {}

        # 加载现有检查点
        self._load_all()

    def _compute_config_hash(self, config: Dict) -> str:
        """计算配置哈希"""
        # 只使用关键配置计算哈希
        key_config = {
            'dataset': config.get('dataset', {}),
            'processing': config.get('processing', {}),
            'filters': config.get('filters', {}),
        }
        config_str = str(sorted(key_config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_checkpoint_file(self, step: str) -> Path:
        """获取检查点文件路径"""
        return self.checkpoint_dir / f"step_{step}.pkl"

    def _load_all(self):
        """加载所有检查点"""
        for step in self.STEPS:
            filepath = self._get_checkpoint_file(step)
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        # 验证配置哈希
                        if data.config_hash == self.config_hash:
                            self._checkpoints[step] = data
                except Exception:
                    pass  # 忽略损坏的检查点

    def save(self, step: str, status: str,
             results: Optional[Dict] = None,
             error_message: Optional[str] = None):
        """
        保存检查点

        Args:
            step: 步骤名称
            status: 状态
            results: 结果数据
            error_message: 错误信息
        """
        data = CheckpointData(
            step=step,
            status=status,
            timestamp=datetime.now(),
            config_hash=self.config_hash,
            results=results or {},
            error_message=error_message
        )
        self._checkpoints[step] = data

        filepath = self._get_checkpoint_file(step)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, step: str) -> Optional[CheckpointData]:
        """加载指定步骤的检查点"""
        return self._checkpoints.get(step)

    def get_resume_step(self) -> Optional[str]:
        """
        获取应该恢复的步骤

        Returns:
            第一个未完成的步骤名称，如果全部完成则返回 None
        """
        for step in self.STEPS:
            checkpoint = self._checkpoints.get(step)
            if checkpoint is None or checkpoint.status != 'completed':
                return step
        return None

    def is_step_completed(self, step: str) -> bool:
        """检查步骤是否已完成"""
        checkpoint = self._checkpoints.get(step)
        return checkpoint is not None and checkpoint.status == 'completed'

    def get_step_results(self, step: str) -> Optional[Dict]:
        """获取步骤结果"""
        checkpoint = self._checkpoints.get(step)
        if checkpoint and checkpoint.status == 'completed':
            return checkpoint.results
        return None

    def clear(self):
        """清除所有检查点"""
        for step in self.STEPS:
            filepath = self._get_checkpoint_file(step)
            if filepath.exists():
                filepath.unlink()
        self._checkpoints.clear()

    def get_status_summary(self) -> Dict[str, str]:
        """获取所有步骤的状态摘要"""
        summary = {}
        for step in self.STEPS:
            checkpoint = self._checkpoints.get(step)
            if checkpoint:
                summary[step] = checkpoint.status
            else:
                summary[step] = 'pending'
        return summary


class Checkpoint:
    """Simple checkpoint for Pipeline controller."""

    def __init__(self, output_dir: str):
        """
        Initialize checkpoint.

        Args:
            output_dir: Directory to store checkpoint files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {}

    def save(self, step: str, results: Dict[str, Any]):
        """Save checkpoint after a step."""
        self._data[step] = results
        checkpoint_file = self.output_dir / 'checkpoint.pkl'
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self._data, f)

    def load_stations(self) -> Optional[Any]:
        """Load stations from checkpoint."""
        checkpoint_file = self.output_dir / 'checkpoint.pkl'
        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'rb') as f:
                self._data = pickle.load(f)
            return self._data.get('stations')
        except Exception:
            return None
