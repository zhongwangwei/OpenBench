#!/usr/bin/env python3
"""
Checkpoint System for WSE Pipeline
断点续传系统

Security: Uses HMAC-signed JSON serialization instead of pickle.
"""

import os
import json
import hmac
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """检查点数据结构"""
    step: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    timestamp: datetime
    config_hash: str
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'step': self.step,
            'status': self.status,
            'timestamp': self.timestamp.isoformat(),
            'config_hash': self.config_hash,
            'results': self.results,
            'error_message': self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            step=data['step'],
            status=data['status'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            config_hash=data['config_hash'],
            results=data.get('results', {}),
            error_message=data.get('error_message'),
        )


class SecureSerializer:
    """HMAC-signed JSON serializer for secure checkpoint storage."""

    SECRET_KEY: Optional[bytes] = None

    @classmethod
    def _get_secret(cls) -> bytes:
        """Get or initialize the secret key for HMAC signing."""
        if cls.SECRET_KEY is None:
            cls.SECRET_KEY = os.environ.get(
                'WSE_CHECKPOINT_KEY', 'default-dev-key'
            ).encode()
        return cls.SECRET_KEY

    @classmethod
    def _sign_data(cls, data: bytes) -> str:
        """Create HMAC-SHA256 signature for data."""
        return hmac.new(cls._get_secret(), data, hashlib.sha256).hexdigest()

    @classmethod
    def _verify_signature(cls, data: bytes, signature: str) -> bool:
        """Verify HMAC signature using constant-time comparison."""
        expected = cls._sign_data(data)
        return hmac.compare_digest(expected, signature)

    @classmethod
    def serialize(cls, data: Any) -> str:
        """Serialize data to signed JSON string."""
        data_json = json.dumps(data, sort_keys=True, default=str)
        signature = cls._sign_data(data_json.encode())
        return json.dumps({
            'data': data_json,
            'signature': signature,
        })

    @classmethod
    def deserialize(cls, content: str) -> Optional[Any]:
        """Deserialize signed JSON string, verifying signature."""
        try:
            parsed = json.loads(content)
            if 'data' not in parsed or 'signature' not in parsed:
                logger.warning("Checkpoint missing required fields")
                return None

            data_str = parsed['data']
            signature = parsed['signature']

            if not cls._verify_signature(data_str.encode(), signature):
                logger.warning("Checkpoint signature verification failed")
                return None

            return json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("Checkpoint contains invalid JSON")
            return None
        except Exception as e:
            logger.warning(f"Failed to deserialize checkpoint: {e}")
            return None


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
        """获取检查点文件路径 (JSON format)"""
        return self.checkpoint_dir / f"step_{step}.json"

    def _get_legacy_checkpoint_file(self, step: str) -> Path:
        """获取旧版检查点文件路径 (pickle format)"""
        return self.checkpoint_dir / f"step_{step}.pkl"

    def _migrate_legacy_checkpoint(self, step: str) -> Optional[CheckpointData]:
        """
        Migrate legacy pickle checkpoint to signed JSON format.

        Returns:
            CheckpointData if migration successful, None otherwise
        """
        legacy_file = self._get_legacy_checkpoint_file(step)
        if not legacy_file.exists():
            return None

        try:
            logger.warning(
                f"Loading legacy pickle checkpoint for step '{step}'. "
                "This format is deprecated and will be converted to signed JSON."
            )
            with open(legacy_file, 'rb') as f:
                data = pickle.load(f)

            # Save in new format
            json_file = self._get_checkpoint_file(step)
            content = SecureSerializer.serialize(data.to_dict())
            with open(json_file, 'w') as f:
                f.write(content)

            # Backup and remove legacy file
            backup_file = legacy_file.with_suffix('.pkl.bak')
            legacy_file.rename(backup_file)
            logger.info(f"Migrated checkpoint to JSON, backed up pickle to {backup_file}")

            return data
        except Exception as e:
            logger.warning(f"Failed to migrate legacy checkpoint: {e}")
            return None

    def _load_all(self):
        """加载所有检查点"""
        for step in self.STEPS:
            filepath = self._get_checkpoint_file(step)
            data = None

            # Try loading JSON format first
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    data_dict = SecureSerializer.deserialize(content)
                    if data_dict:
                        data = CheckpointData.from_dict(data_dict)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint for {step}: {e}")

            # Try migrating legacy pickle if JSON not found
            if data is None:
                data = self._migrate_legacy_checkpoint(step)

            # Verify config hash and store
            if data and data.config_hash == self.config_hash:
                self._checkpoints[step] = data

    def save(self, step: str, status: str,
             results: Optional[Dict] = None,
             error_message: Optional[str] = None):
        """
        保存检查点 (Secure JSON format)

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
        content = SecureSerializer.serialize(data.to_dict())
        with open(filepath, 'w') as f:
            f.write(content)

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
            # Remove JSON checkpoint
            filepath = self._get_checkpoint_file(step)
            if filepath.exists():
                filepath.unlink()
            # Also remove any legacy pickle files
            legacy_file = self._get_legacy_checkpoint_file(step)
            if legacy_file.exists():
                legacy_file.unlink()
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
    """Simple checkpoint for Pipeline controller with secure JSON serialization."""

    SECRET_KEY: Optional[bytes] = None

    def __init__(self, output_dir: str):
        """
        Initialize checkpoint.

        Args:
            output_dir: Directory to store checkpoint files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {}

    @classmethod
    def _get_secret(cls) -> bytes:
        """Get or initialize the secret key for HMAC signing."""
        if cls.SECRET_KEY is None:
            cls.SECRET_KEY = os.environ.get(
                'WSE_CHECKPOINT_KEY', 'default-dev-key'
            ).encode()
        return cls.SECRET_KEY

    def _sign_data(self, data: bytes) -> str:
        """Create HMAC-SHA256 signature for data."""
        return hmac.new(self._get_secret(), data, hashlib.sha256).hexdigest()

    def _verify_signature(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature using constant-time comparison."""
        expected = self._sign_data(data)
        return hmac.compare_digest(expected, signature)

    def save(self, step: str, results: Dict[str, Any]):
        """Save checkpoint after a step using signed JSON."""
        self._data[step] = results
        checkpoint_file = self.output_dir / 'checkpoint.json'

        # Serialize with HMAC signature
        data_json = json.dumps(self._data, sort_keys=True, default=str)
        signature = self._sign_data(data_json.encode())
        content = json.dumps({
            'data': data_json,
            'signature': signature,
        })

        with open(checkpoint_file, 'w') as f:
            f.write(content)

    def _load_legacy_pickle(self) -> Optional[Dict[str, Any]]:
        """
        Load legacy pickle checkpoint and migrate to JSON.

        Returns:
            Loaded data if successful, None otherwise
        """
        pkl_file = self.output_dir / 'checkpoint.pkl'
        if not pkl_file.exists():
            return None

        try:
            logger.warning(
                "Loading legacy pickle checkpoint. "
                "This format is deprecated and will be converted to signed JSON."
            )
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Save in new JSON format
            self._data = data
            json_file = self.output_dir / 'checkpoint.json'
            data_json = json.dumps(data, sort_keys=True, default=str)
            signature = self._sign_data(data_json.encode())
            content = json.dumps({
                'data': data_json,
                'signature': signature,
            })
            with open(json_file, 'w') as f:
                f.write(content)

            # Backup and remove legacy file
            backup_file = pkl_file.with_suffix('.pkl.bak')
            pkl_file.rename(backup_file)
            logger.info(f"Migrated checkpoint to JSON, backed up pickle to {backup_file}")

            return data
        except Exception as e:
            logger.warning(f"Failed to load legacy pickle checkpoint: {e}")
            return None

    def load_stations(self) -> Optional[Any]:
        """Load stations from checkpoint with signature verification."""
        checkpoint_file = self.output_dir / 'checkpoint.json'

        # Try JSON format first
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    content = f.read()

                if not content.strip():
                    return None

                parsed = json.loads(content)

                # Verify required fields
                if 'data' not in parsed or 'signature' not in parsed:
                    logger.warning("Checkpoint missing required fields")
                    return None

                data_str = parsed['data']
                signature = parsed['signature']

                # Verify signature
                if not self._verify_signature(data_str.encode(), signature):
                    logger.warning("Checkpoint signature verification failed")
                    return None

                self._data = json.loads(data_str)
                return self._data.get('stations')

            except json.JSONDecodeError:
                logger.warning("Checkpoint contains invalid JSON")
                return None
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return None

        # Fall back to legacy pickle migration
        data = self._load_legacy_pickle()
        if data:
            return data.get('stations')

        return None
