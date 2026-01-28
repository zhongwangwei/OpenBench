"""Step 0: Data Download and Completeness Check."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

from ..readers.downloader import (
    HydroSatDownloader, HydroWebDownloader,
    CGLSDownloader, ICESatDownloader
)
from ..utils.logger import get_logger
from ..constants import COMPLETENESS_THRESHOLDS

logger = get_logger(__name__)

COMPLETENESS_RULES = {
    'hydrosat': {'min_files': COMPLETENESS_THRESHOLDS['hydrosat'], 'pattern': '*.txt', 'subdir': 'WL_hydrosat'},
    'hydroweb': {'min_files': COMPLETENESS_THRESHOLDS['hydroweb'], 'pattern': '*.txt', 'subdir': 'hydroweb_river'},
    'cgls':     {'min_files': COMPLETENESS_THRESHOLDS['cgls'], 'pattern': '*.geojson', 'subdir': 'cgls_river'},
    'icesat':   {'min_files': COMPLETENESS_THRESHOLDS['icesat'], 'pattern': '*.h5', 'subdir': 'icesat'},
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

    def run(self, sources: List[str], skip_download: bool = False,
            interactive: bool = True) -> Dict[str, DataStatus]:
        """Run Step 0: check and optionally download data."""
        logger.info("[Step 0] 检查数据完整性...")

        statuses = {}
        incomplete = []

        for source in sources:
            source_path = self.data_root / source.capitalize()
            status = self._check_source_completeness(source, source_path)
            statuses[source] = status

            symbol = "+" if status.is_complete else "x"
            if status.is_complete:
                logger.info(f"  {symbol} {source.capitalize()}: {status.current_files} 文件 (完整)")
            else:
                logger.info(f"  {symbol} {source.capitalize()}: {status.current_files} / {status.min_required} 文件 (不完整)")
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
            source_path = self.data_root / source.capitalize()
            statuses[source] = self._check_source_completeness(source, source_path)

        return statuses

    def _check_source_completeness(self, source: str, path: Path) -> DataStatus:
        """Check if a data source is complete."""
        rules = COMPLETENESS_RULES.get(source, {'min_files': 0, 'pattern': '*', 'subdir': ''})

        data_path = path / rules['subdir'] if rules['subdir'] else path
        if not data_path.exists():
            return DataStatus(source, 0, rules['min_files'], False, data_path)

        files = list(data_path.glob(rules['pattern']))
        current = len(files)
        is_complete = current >= rules['min_files']

        return DataStatus(source, current, rules['min_files'], is_complete, data_path)

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
        credentials = self.config.get('credentials', {})
        num_workers = self.config.get('num_workers', 5)

        for source in sources:
            logger.info(f"\n下载 {source.capitalize()}...")

            downloader_cls = DOWNLOADERS.get(source)
            if not downloader_cls:
                logger.warning(f"未知数据源: {source}")
                continue

            output_dir = self.data_root / source.capitalize()
            downloader = downloader_cls(str(output_dir))

            try:
                creds = credentials.get(source, {})
                downloader.download(
                    username=creds.get('username'),
                    password=creds.get('password'),
                    api_key=creds.get('api_key'),
                    num_workers=num_workers,
                    interactive=False
                )
            except Exception as e:
                logger.error(f"下载 {source} 失败: {e}")
