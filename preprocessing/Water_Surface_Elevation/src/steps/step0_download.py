"""Step 0: Data Download and Completeness Check."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import yaml

from ..readers.downloader import (
    HydroSatDownloader, HydroWebDownloader,
    CGLSDownloader, ICESatDownloader
)
from ..utils.logger import get_logger
from ..constants import COMPLETENESS_THRESHOLDS

logger = get_logger(__name__)

# 文件模式配置 (用于完整性检查)
FILE_PATTERNS = {
    'hydrosat': {'pattern': '*.txt', 'subdir': ''},
    'hydroweb': {'pattern': '*.txt', 'subdir': 'hydroweb_river'},
    'cgls':     {'pattern': '*.geojson', 'subdir': ''},
    'icesat':   {'pattern': '*.txt', 'subdir': ''},
}

DOWNLOADERS = {
    'hydrosat': HydroSatDownloader,
    'hydroweb': HydroWebDownloader,
    'cgls': CGLSDownloader,
    'icesat': ICESatDownloader,
}

@dataclass
class DataStatus:
    """Status of a data source."""
    source: str
    current_files: int
    min_required: int
    is_complete: bool
    path: Path


class Step0Download:
    """Step 0: Check data completeness and download if needed."""

    def __init__(self, config: dict):
        self.config = config
        self.data_root = Path(config.get('data_root', '/Volumes/Data01/Altimetry'))

        # 配置文件目录
        self.config_dir = Path(__file__).parent.parent.parent / 'config'

        # 缓存各数据源配置
        self._source_configs: Dict[str, dict] = {}

    def _load_source_config(self, source: str) -> dict:
        """加载数据源配置文件"""
        if source in self._source_configs:
            return self._source_configs[source]

        config_file = self.config_dir / f"{source}.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                self._source_configs[source] = config
                return config
            except Exception as e:
                logger.warning(f"加载配置文件失败 {config_file}: {e}")

        self._source_configs[source] = {}
        return {}

    def _get_source_data_dir(self, source: str) -> Path:
        """获取数据源的数据目录"""
        source_config = self._load_source_config(source)

        # 优先使用配置文件中的 paths.data_dir
        paths = source_config.get('paths', {})
        data_dir = paths.get('data_dir')

        if data_dir:
            return Path(data_dir)

        # 回退到默认路径
        return self.data_root / source.capitalize()

    def _get_source_credentials(self, source: str) -> dict:
        """获取数据源的凭证"""
        source_config = self._load_source_config(source)
        return source_config.get('credentials', {}) or {}

    def run(self, sources: List[str], skip_download: bool = False,
            interactive: bool = True) -> Dict[str, DataStatus]:
        """Run Step 0: check and optionally download data."""
        logger.info("[Step 0] 检查数据完整性...")

        statuses = {}
        incomplete = []

        for source in sources:
            source_path = self._get_source_data_dir(source)
            status = self._check_source_completeness(source, source_path)
            statuses[source] = status

            symbol = "+" if status.is_complete else "x"
            if status.is_complete:
                logger.info(f"  [{symbol}] {source.capitalize()}: {status.current_files} 文件 (完整)")
                logger.debug(f"      路径: {status.path}")
            else:
                logger.info(f"  [{symbol}] {source.capitalize()}: {status.current_files} / {status.min_required} 文件 (不完整)")
                logger.debug(f"      路径: {status.path}")
                incomplete.append(source)

        if not incomplete or skip_download:
            return statuses

        # Ask user
        if interactive:
            response = self._prompt_download(incomplete)
            if response == 'n':
                logger.info("用户取消，退出")
                sys.exit(0)
            elif response == 'skip':
                logger.info("跳过下载，使用现有数据继续")
                return statuses

        # Download missing data
        self._download_sources(incomplete)

        # Re-check
        for source in incomplete:
            source_path = self._get_source_data_dir(source)
            statuses[source] = self._check_source_completeness(source, source_path)

        return statuses

    def _check_source_completeness(self, source: str, path: Path) -> DataStatus:
        """Check if a data source is complete."""
        patterns = FILE_PATTERNS.get(source, {'pattern': '*', 'subdir': ''})
        min_files = COMPLETENESS_THRESHOLDS.get(source, 0)

        data_path = path / patterns['subdir'] if patterns['subdir'] else path
        if not data_path.exists():
            return DataStatus(source, 0, min_files, False, data_path)

        files = list(data_path.glob(patterns['pattern']))
        current = len(files)
        is_complete = current >= min_files

        return DataStatus(source, current, min_files, is_complete, data_path)

    def _prompt_download(self, incomplete: List[str]) -> str:
        """Prompt user for download decision."""
        print(f"\n是否下载缺失数据 ({', '.join(incomplete)})? [Y/n/skip]")
        print("  Y    - 下载后继续处理")
        print("  n    - 退出")
        print("  skip - 跳过下载，用现有数据继续")

        try:
            response = input("> ").strip().lower()
            if response in ['', 'y', 'yes']:
                return 'y'
            elif response in ['n', 'no']:
                return 'n'
            else:
                return 'skip'
        except (KeyboardInterrupt, EOFError):
            return 'n'

    def _download_sources(self, sources: List[str]):
        """Download specified data sources."""
        num_workers = self.config.get('num_workers', 5)

        for source in sources:
            logger.info(f"\n下载 {source.capitalize()}...")

            downloader_cls = DOWNLOADERS.get(source)
            if not downloader_cls:
                logger.warning(f"未知数据源: {source}")
                continue

            # 使用配置文件中的路径
            output_dir = self._get_source_data_dir(source)
            # 对于需要创建子目录的下载器，使用父目录
            if source == 'hydrosat':
                output_dir = output_dir.parent  # WL_hydrosat 会由下载器创建
            elif source == 'hydroweb':
                output_dir = output_dir.parent if output_dir.name == 'hydroweb_river' else output_dir

            logger.info(f"  输出目录: {output_dir}")

            # 获取凭证
            creds = self._get_source_credentials(source)

            # 获取下载配置
            source_config = self._load_source_config(source)
            download_config = source_config.get('download', {})
            verify_ssl = download_config.get('verify_ssl', True)

            try:
                downloader = downloader_cls(str(output_dir), verify_ssl=verify_ssl)
                downloader.download(
                    username=creds.get('username'),
                    password=creds.get('password'),
                    api_key=creds.get('api_key'),
                    num_workers=num_workers,
                    interactive=False
                )
            except Exception as e:
                logger.error(f"下载 {source} 失败: {e}")
