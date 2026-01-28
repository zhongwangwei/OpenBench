# WSE Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor WSE pipeline with Step 0 download integration, clean code structure, and complete functionality.

**Architecture:** 5-step pipeline (Download → Validate → CaMa → Reserved → Merge) with flexible multi-source support and CLI entry point.

**Tech Stack:** Python 3.12, Click (CLI), PyYAML, h5py, requests, numpy

---

## Task 1: Archive Old Files

**Files:**
- Create: `_archive/` directory
- Move: 12 root .py files to `_archive/`

**Step 1: Create archive directory and move files**

```bash
mkdir -p _archive
mv AllocateVS.py availability_data.py bin_to_netcdf.py cama_flood_wse_processor.py \
   convert_cama_base.py convert_merit_flwdir.py download_merit_flwdir.py \
   generate_flwdir.py geoid_calculator.py set_name.py _archive/
```

**Step 2: Move test files to tests directory**

```bash
mkdir -p tests
mv test_allocate_vs.py test_station_allocation.py test_with_real_data.py tests/
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive old files and organize tests"
```

---

## Task 2: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

**Step 1: Create pyproject.toml with wse entry point**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "wse-pipeline"
version = "0.1.0"
description = "Water Surface Elevation Pipeline for OpenBench"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "pyyaml>=6.0",
    "numpy>=1.20",
    "h5py>=3.0",
    "requests>=2.25",
    "tqdm>=4.60",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[project.scripts]
wse = "src.main:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pyproject.toml with wse CLI entry point"
```

---

## Task 3: Create Station Data Structure

**Files:**
- Create: `src/core/station.py`
- Test: `tests/test_station.py`

**Step 1: Write the test**

```python
# tests/test_station.py
import pytest
from src.core.station import Station, StationList

def test_station_creation():
    s = Station(
        id="ST001",
        name="Test Station",
        lon=100.5,
        lat=30.2,
        source="hydroweb"
    )
    assert s.id == "ST001"
    assert s.lon == 100.5
    assert s.source == "hydroweb"

def test_station_validation():
    s = Station(id="ST001", name="Test", lon=200, lat=30, source="test")
    assert not s.is_valid()  # lon out of range

def test_station_list():
    stations = StationList()
    stations.add(Station("S1", "A", 100, 30, "hydroweb"))
    stations.add(Station("S2", "B", 101, 31, "cgls"))
    assert len(stations) == 2
    assert len(stations.filter_by_source("hydroweb")) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_station.py -v
```
Expected: FAIL with "No module named 'src.core.station'"

**Step 3: Write implementation**

```python
# src/core/station.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Station:
    """Virtual station data structure."""
    id: str
    name: str
    lon: float
    lat: float
    source: str
    elevation: float = 0.0
    num_observations: int = 0
    egm08: Optional[float] = None
    egm96: Optional[float] = None
    cama_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if station coordinates are valid."""
        return -180 <= self.lon <= 180 and -90 <= self.lat <= 90

    def set_cama_result(self, resolution: str, result: Dict[str, Any]):
        """Set CaMa allocation result for a resolution."""
        self.cama_results[resolution] = result


class StationList:
    """Collection of stations with filtering and grouping."""

    def __init__(self):
        self._stations: List[Station] = []

    def add(self, station: Station):
        self._stations.append(station)

    def __len__(self):
        return len(self._stations)

    def __iter__(self):
        return iter(self._stations)

    def filter_by_source(self, source: str) -> 'StationList':
        """Filter stations by source."""
        result = StationList()
        for s in self._stations:
            if s.source == source:
                result.add(s)
        return result

    def filter_valid(self) -> 'StationList':
        """Filter only valid stations."""
        result = StationList()
        for s in self._stations:
            if s.is_valid():
                result.add(s)
        return result

    def get_sources(self) -> List[str]:
        """Get unique sources."""
        return list(set(s.source for s in self._stations))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_station.py -v
```
Expected: PASS

**Step 5: Update core __init__.py**

```python
# src/core/__init__.py
from .station import Station, StationList
from .geoid_calculator import GeoidCalculator
from .cama_allocator import CamaAllocator

__all__ = ['Station', 'StationList', 'GeoidCalculator', 'CamaAllocator']
```

**Step 6: Commit**

```bash
git add src/core/station.py src/core/__init__.py tests/test_station.py
git commit -m "feat: add Station and StationList data structures"
```

---

## Task 4: Create Step 0 Download

**Files:**
- Create: `src/steps/step0_download.py`
- Test: `tests/test_step0_download.py`

**Step 1: Write the test**

```python
# tests/test_step0_download.py
import pytest
from unittest.mock import patch, MagicMock
from src.steps.step0_download import Step0Download, DataStatus

def test_check_completeness_complete():
    step = Step0Download(config={})
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = [MagicMock() for _ in range(2500)]
        status = step._check_source_completeness('hydrosat', '/fake/path')
    assert status.is_complete

def test_check_completeness_incomplete():
    step = Step0Download(config={})
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = [MagicMock() for _ in range(100)]
        status = step._check_source_completeness('hydrosat', '/fake/path')
    assert not status.is_complete
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_step0_download.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/steps/step0_download.py
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

logger = get_logger(__name__)

COMPLETENESS_RULES = {
    'hydrosat': {'min_files': 2000, 'pattern': '*.txt', 'subdir': 'WL_hydrosat'},
    'hydroweb': {'min_files': 30000, 'pattern': '*.txt', 'subdir': 'hydroweb_river'},
    'cgls':     {'min_files': 10000, 'pattern': '*.geojson', 'subdir': 'cgls_river'},
    'icesat':   {'min_files': 15000, 'pattern': '*.h5', 'subdir': 'icesat'},
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

            symbol = "✓" if status.is_complete else "✗"
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
```

**Step 4: Update steps __init__.py**

```python
# src/steps/__init__.py
from .step0_download import Step0Download
from .step1_validate import Step1Validate
from .step2_cama import Step2CaMa
from .step3_reserved import Step3Reserved
from .step4_merge import Step4Merge

__all__ = ['Step0Download', 'Step1Validate', 'Step2CaMa', 'Step3Reserved', 'Step4Merge']
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_step0_download.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/steps/step0_download.py src/steps/__init__.py tests/test_step0_download.py
git commit -m "feat: add Step 0 download with completeness check"
```

---

## Task 5: Create Main CLI

**Files:**
- Create: `src/main.py`
- Test: `tests/test_cli.py`

**Step 1: Write the test**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from src.main import main

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'WSE Pipeline' in result.output

def test_cli_source_parsing():
    runner = CliRunner()
    result = runner.invoke(main, ['--source', 'hydroweb', '--dry-run'])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/main.py
"""WSE Pipeline CLI Entry Point."""
import click
from pathlib import Path
from typing import List, Optional

from .pipeline import Pipeline
from .utils.config_loader import load_config
from .utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

@click.command()
@click.option('--source', '-s', default='all',
              help='Data source(s): hydroweb,cgls,icesat,hydrosat or "all"')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Config file path')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory')
@click.option('--merge', is_flag=True,
              help='Merge all sources into single output')
@click.option('--skip-download', is_flag=True,
              help='Skip download check, use existing data')
@click.option('--step', type=click.Choice(['download', 'validate', 'cama', 'reserved', 'merge']),
              help='Run specific step only')
@click.option('--num-workers', '-j', type=int, default=5,
              help='Parallel download workers')
@click.option('--log-level', default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
@click.option('--dry-run', is_flag=True,
              help='Simulation mode, no file writes')
def main(source: str, config: Optional[str], output: Optional[str],
         merge: bool, skip_download: bool, step: Optional[str],
         num_workers: int, log_level: str, dry_run: bool):
    """WSE Pipeline - Water Surface Elevation Processing.

    Process satellite altimetry data from multiple sources and allocate
    to CaMa-Flood grid cells.

    Examples:

        wse --source hydroweb

        wse --source hydroweb,cgls --merge

        wse --source all --skip-download
    """
    setup_logger(log_level)

    # Parse sources
    if source == 'all':
        sources = ['hydrosat', 'hydroweb', 'cgls', 'icesat']
    else:
        sources = [s.strip().lower() for s in source.split(',')]

    # Load config
    if config:
        cfg = load_config(config)
    else:
        cfg = _default_config()

    # Override with CLI options
    if output:
        cfg['output_dir'] = output
    cfg['merge'] = merge
    cfg['skip_download'] = skip_download
    cfg['num_workers'] = num_workers
    cfg['dry_run'] = dry_run

    logger.info(f"WSE Pipeline - 处理数据源: {', '.join(sources)}")

    if dry_run:
        logger.info("[DRY RUN] 模拟模式，不写入文件")
        return

    # Run pipeline
    pipeline = Pipeline(cfg)

    if step:
        pipeline.run_step(step, sources)
    else:
        pipeline.run(sources)


def _default_config() -> dict:
    """Return default configuration."""
    return {
        'data_root': '/Volumes/Data01/Altimetry',
        'output_dir': './output',
        'cama_root': '/Volumes/Data01/2025',
        'geoid_root': '/Volumes/Data01/AltiMaPpy-data/egm-geoids',
        'resolutions': ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min'],
        'validation': {
            'min_observations': 10,
            'check_duplicates': True,
        },
        'credentials': {},
    }


if __name__ == '__main__':
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_cli.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/main.py tests/test_cli.py
git commit -m "feat: add main CLI with click"
```

---

## Task 6: Create Pipeline Controller

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write the test**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import Pipeline

def test_pipeline_initialization():
    config = {'data_root': '/tmp', 'output_dir': '/tmp/out'}
    pipeline = Pipeline(config)
    assert pipeline.config == config

def test_pipeline_step_sequence():
    config = {'data_root': '/tmp', 'output_dir': '/tmp/out', 'skip_download': True}
    pipeline = Pipeline(config)
    assert pipeline.steps == ['download', 'validate', 'cama', 'reserved', 'merge']
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/pipeline.py
"""Pipeline Controller for WSE Processing."""
from pathlib import Path
from typing import List, Optional, Dict, Any

from .steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved, Step4Merge
from .core.station import StationList
from .utils.logger import get_logger
from .utils.checkpoint import Checkpoint

logger = get_logger(__name__)


class Pipeline:
    """WSE Pipeline Controller."""

    steps = ['download', 'validate', 'cama', 'reserved', 'merge']

    def __init__(self, config: dict):
        self.config = config
        self.checkpoint = Checkpoint(config.get('output_dir', './output'))

        # Initialize steps
        self._step_handlers = {
            'download': Step0Download(config),
            'validate': Step1Validate(config),
            'cama': Step2CaMa(config),
            'reserved': Step3Reserved(config),
            'merge': Step4Merge(config),
        }

    def run(self, sources: List[str]) -> Dict[str, Any]:
        """Run full pipeline for specified sources."""
        logger.info("=" * 60)
        logger.info("WSE Pipeline 开始")
        logger.info("=" * 60)

        results = {}
        stations = StationList()

        for step_name in self.steps:
            logger.info(f"\n{'='*20} {step_name.upper()} {'='*20}")

            if step_name == 'download':
                skip = self.config.get('skip_download', False)
                results['download'] = self._step_handlers['download'].run(
                    sources, skip_download=skip
                )
            elif step_name == 'validate':
                stations = self._step_handlers['validate'].run(sources)
                results['validate'] = {'total': len(stations)}
            elif step_name == 'cama':
                stations = self._step_handlers['cama'].run(stations)
                results['cama'] = {'allocated': len(stations)}
            elif step_name == 'reserved':
                stations = self._step_handlers['reserved'].run(stations)
            elif step_name == 'merge':
                output_files = self._step_handlers['merge'].run(
                    stations,
                    merge=self.config.get('merge', False)
                )
                results['merge'] = {'files': output_files}

            self.checkpoint.save(step_name, results)

        logger.info("\n" + "=" * 60)
        logger.info("WSE Pipeline 完成")
        logger.info("=" * 60)

        return results

    def run_step(self, step_name: str, sources: List[str]) -> Any:
        """Run a specific step only."""
        if step_name not in self.steps:
            raise ValueError(f"Unknown step: {step_name}")

        logger.info(f"运行单步: {step_name}")

        handler = self._step_handlers[step_name]

        if step_name == 'download':
            return handler.run(sources)
        elif step_name == 'validate':
            return handler.run(sources)
        else:
            # Load from checkpoint
            stations = self.checkpoint.load_stations()
            if step_name == 'cama':
                return handler.run(stations)
            elif step_name == 'reserved':
                return handler.run(stations)
            elif step_name == 'merge':
                return handler.run(stations, merge=self.config.get('merge', False))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add Pipeline controller with step orchestration"
```

---

## Task 7: Update Step 1 Validate

**Files:**
- Modify: `src/steps/step1_validate.py`

**Step 1: Review and update Step1Validate to use Station dataclass**

Update to integrate with readers and return StationList.

**Step 2: Commit**

```bash
git add src/steps/step1_validate.py
git commit -m "refactor: update Step1Validate to use Station dataclass"
```

---

## Task 8: Update Step 2 CaMa

**Files:**
- Modify: `src/steps/step2_cama.py`

**Step 1: Review and update Step2CaMa to process StationList**

Update to use cama_allocator and process all 5 resolutions.

**Step 2: Commit**

```bash
git add src/steps/step2_cama.py
git commit -m "refactor: update Step2CaMa for multi-resolution processing"
```

---

## Task 9: Update Step 4 Merge

**Files:**
- Modify: `src/steps/step4_merge.py`

**Step 1: Update merge step for output modes**

Support separate files (default) and merged output (--merge flag).

**Step 2: Commit**

```bash
git add src/steps/step4_merge.py
git commit -m "refactor: update Step4Merge with separate/merged output modes"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
from pathlib import Path
from click.testing import CliRunner
from src.main import main

@pytest.mark.integration
def test_full_pipeline_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, [
        '--source', 'hydrosat',
        '--skip-download',
        '--dry-run'
    ])
    assert result.exit_code == 0

@pytest.mark.integration
def test_single_source_pipeline():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, [
            '--source', 'hydrosat',
            '--skip-download',
            '--output', './test_output'
        ])
        # Check output created
        assert Path('./test_output').exists() or result.exit_code == 0
```

**Step 2: Run integration tests**

```bash
pytest tests/test_integration.py -v -m integration
```

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full pipeline"
```

---

## Task 11: Final Cleanup and Documentation

**Step 1: Update README or docs**

**Step 2: Final commit**

```bash
git add -A
git commit -m "docs: update documentation for refactored pipeline"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Archive old files | _archive/, tests/ |
| 2 | Create pyproject.toml | pyproject.toml |
| 3 | Station data structure | src/core/station.py |
| 4 | Step 0 Download | src/steps/step0_download.py |
| 5 | Main CLI | src/main.py |
| 6 | Pipeline controller | src/pipeline.py |
| 7 | Update Step 1 | src/steps/step1_validate.py |
| 8 | Update Step 2 | src/steps/step2_cama.py |
| 9 | Update Step 4 | src/steps/step4_merge.py |
| 10 | Integration tests | tests/test_integration.py |
| 11 | Final cleanup | docs/ |
