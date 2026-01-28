# WSE Pipeline 代码审查修复计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复代码审查中发现的所有关键和重要问题，提升代码安全性、完整性和测试覆盖率

**Architecture:** 分阶段修复，从安全问题开始，逐步完善功能和测试

**Tech Stack:** Python, pytest, pydantic (新增用于配置验证)

---

## Phase 1: 安全问题修复 (Critical)

### Task 1: 修复 SSL 验证问题

**Files:**
- Modify: `src/readers/downloader.py`

**Step 1: 移除全局 SSL 警告禁用**

删除以下代码：
```python
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

**Step 2: 添加配置化的 SSL 验证选项**

```python
class BaseDownloader:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.verify_ssl = self.config.get('verify_ssl', True)

    def _get_session(self) -> requests.Session:
        session = requests.Session()
        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            session.verify = False
            self.log('warning', "SSL verification disabled - use only for trusted sources")
        return session
```

**Step 3: 更新所有 requests.get() 调用使用 session**

**Step 4: 运行测试验证**

```bash
pytest tests/test_step0_download.py -v
```

**Step 5: Commit**

```bash
git commit -m "security: make SSL verification configurable instead of globally disabled"
```

---

### Task 2: 修复 Pickle 安全风险

**Files:**
- Modify: `src/utils/checkpoint.py`

**Step 1: 创建安全的序列化方法**

```python
import json
import hashlib
import hmac
from typing import Optional

class SecureCheckpoint:
    """使用 JSON 替代 pickle，添加完整性验证"""

    SECRET_KEY = None  # 从环境变量或配置加载

    @classmethod
    def _get_secret(cls) -> bytes:
        if cls.SECRET_KEY is None:
            import os
            cls.SECRET_KEY = os.environ.get('WSE_CHECKPOINT_KEY', 'default-dev-key').encode()
        return cls.SECRET_KEY

    @classmethod
    def _sign_data(cls, data: bytes) -> str:
        return hmac.new(cls._get_secret(), data, hashlib.sha256).hexdigest()

    @classmethod
    def _verify_signature(cls, data: bytes, signature: str) -> bool:
        expected = cls._sign_data(data)
        return hmac.compare_digest(expected, signature)
```

**Step 2: 实现 JSON 序列化（保留 pickle 向后兼容）**

```python
def save(self, stations: List[Station], step: str):
    """保存检查点 - 优先使用 JSON"""
    data = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'stations': [self._station_to_dict(s) for s in stations]
    }
    json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
    signature = self._sign_data(json_bytes)

    checkpoint_data = {
        'format': 'json',
        'signature': signature,
        'data': data
    }

    filepath = self.checkpoint_dir / f"checkpoint_{step}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
```

**Step 3: 添加加载时的签名验证**

**Step 4: 添加单元测试**

```python
def test_checkpoint_integrity_verification():
    """测试检查点完整性验证"""
    ...
```

**Step 5: Commit**

```bash
git commit -m "security: replace pickle with signed JSON for checkpoint serialization"
```

---

### Task 3: 修复子进程输入验证

**Files:**
- Modify: `src/core/geoid_calculator.py`

**Step 1: 添加输入验证函数**

```python
def _validate_coordinates(self, lat: float, lon: float) -> tuple:
    """验证并清理坐标输入"""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid coordinate type: lat={lat}, lon={lon}") from e

    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude out of range: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude out of range: {lon}")

    return lat, lon
```

**Step 2: 在所有 subprocess 调用前添加验证**

**Step 3: 添加测试用例**

```python
def test_geoid_calculator_rejects_invalid_input():
    calc = GeoidCalculator(config)
    with pytest.raises(ValueError):
        calc.get_egm08("not_a_number", 10.0)
```

**Step 4: Commit**

```bash
git commit -m "security: add input validation before subprocess calls in geoid calculator"
```

---

## Phase 2: 重要问题修复

### Task 4: 完成 Pipeline 集成

**Files:**
- Modify: `src/main.py`
- Modify: `src/pipeline.py`

**Step 1: 更新 main.py 实际调用 Pipeline**

```python
@cli.command()
@click.pass_context
def run(ctx, source, config, output, merge, skip_download, step, num_workers, log_level, dry_run):
    """运行 WSE 处理 Pipeline"""
    ...

    if dry_run:
        logger.info("[DRY-RUN] Would execute pipeline with sources: %s", sources)
        return

    # 实际执行 Pipeline
    from .pipeline import Pipeline
    pipeline = Pipeline(full_config)

    if step:
        result = pipeline.run_step(step, sources)
    else:
        result = pipeline.run(sources)

    if result.success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed: %s", result.error)
        raise SystemExit(1)
```

**Step 2: 更新 Pipeline.run() 返回结果对象**

```python
@dataclass
class PipelineResult:
    success: bool
    stations_processed: int
    output_files: List[str]
    error: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
```

**Step 3: 添加集成测试**

**Step 4: Commit**

```bash
git commit -m "feat: complete pipeline integration with actual execution"
```

---

### Task 5: 优化重复检测算法 (O(n²) → O(n log n))

**Files:**
- Modify: `src/steps/step1_validate.py`

**Step 1: 添加 scipy 依赖（如果需要）或使用纯 Python 实现**

**Step 2: 实现基于空间索引的重复检测**

```python
from collections import defaultdict

def _detect_duplicates_fast(self, stations: List[Station], threshold_km: float = 0.1) -> List[tuple]:
    """使用网格索引的快速重复检测 O(n)"""
    # 使用 ~10km 的网格单元
    grid_size = 0.1  # 约 10km
    grid = defaultdict(list)

    # 将站点分配到网格
    for i, station in enumerate(stations):
        cell = (int(station.lat / grid_size), int(station.lon / grid_size))
        grid[cell].append((i, station))

    duplicates = []

    # 只检查相邻网格内的站点
    for cell, cell_stations in grid.items():
        # 检查本网格和8个相邻网格
        neighbors = [
            (cell[0] + di, cell[1] + dj)
            for di in [-1, 0, 1] for dj in [-1, 0, 1]
        ]

        candidate_stations = []
        for neighbor in neighbors:
            candidate_stations.extend(grid.get(neighbor, []))

        # 只在候选集内做 O(n²) 检测
        for i, (idx1, s1) in enumerate(cell_stations):
            for idx2, s2 in candidate_stations:
                if idx1 >= idx2:
                    continue
                dist = haversine_distance(s1.lat, s1.lon, s2.lat, s2.lon)
                if dist < threshold_km:
                    duplicates.append((idx1, idx2, dist))

    return duplicates
```

**Step 3: 添加性能测试**

```python
def test_duplicate_detection_performance():
    """确保 50000 站点的重复检测在 10 秒内完成"""
    stations = [Station(id=str(i), lon=random.uniform(-180, 180), lat=random.uniform(-90, 90), ...)
                for i in range(50000)]

    start = time.time()
    result = validator._detect_duplicates_fast(stations)
    elapsed = time.time() - start

    assert elapsed < 10, f"Duplicate detection took {elapsed:.1f}s, expected < 10s"
```

**Step 4: Commit**

```bash
git commit -m "perf: optimize duplicate detection from O(n²) to O(n) using spatial grid"
```

---

### Task 6: 统一数据结构

**Files:**
- Modify: `src/core/station.py`
- Modify: `src/readers/base_reader.py`
- Modify: `src/steps/step1_validate.py`

**Step 1: 将 StationMetadata 作为 Station 的别名**

```python
# src/readers/base_reader.py
from ..core.station import Station

# 向后兼容别名
StationMetadata = Station
```

**Step 2: 更新所有 reader 直接返回 Station**

**Step 3: 移除 step1_validate.py 中的转换代码**

**Step 4: 更新测试**

**Step 5: Commit**

```bash
git commit -m "refactor: unify Station and StationMetadata into single data structure"
```

---

### Task 7: 移除硬编码路径

**Files:**
- Modify: `src/main.py`
- Create: `config/default_paths.yaml`

**Step 1: 创建默认路径配置文件**

```yaml
# config/default_paths.yaml
# 这些是示例路径，请根据您的环境修改

data_root: ${WSE_DATA_ROOT:./data}
output_dir: ${WSE_OUTPUT_DIR:./output}
cama_root: ${WSE_CAMA_ROOT:./cama_data}
geoid_root: ${WSE_GEOID_ROOT:./geoid_data}
```

**Step 2: 更新 main.py 从环境变量和配置文件加载路径**

```python
def _load_default_config() -> dict:
    """从环境变量和配置文件加载默认配置"""
    import os

    return {
        'data_root': os.environ.get('WSE_DATA_ROOT', './data'),
        'output_dir': os.environ.get('WSE_OUTPUT_DIR', './output'),
        'cama_root': os.environ.get('WSE_CAMA_ROOT'),
        'geoid_root': os.environ.get('WSE_GEOID_ROOT'),
        ...
    }
```

**Step 3: 添加配置验证，缺少必要路径时给出清晰错误**

**Step 4: Commit**

```bash
git commit -m "config: remove hardcoded paths, use environment variables and config files"
```

---

### Task 8: 修复 sys.path 操作

**Files:**
- Modify: `src/core/cama_allocator.py`
- Move: `AllocateVS.py` → `src/core/allocate_vs.py`

**Step 1: 将 AllocateVS.py 移入 src/core/**

```bash
cp _archive/AllocateVS.py src/core/allocate_vs.py
```

**Step 2: 更新 cama_allocator.py 的导入**

```python
# 移除 sys.path 操作
try:
    from .allocate_vs import AllocateVS
    HAS_ALLOCATE_VS = True
except ImportError:
    HAS_ALLOCATE_VS = False
```

**Step 3: 更新 __init__.py 导出**

**Step 4: Commit**

```bash
git commit -m "refactor: move AllocateVS into package, remove sys.path manipulation"
```

---

## Phase 3: 代码质量改进

### Task 9: 统一常量定义

**Files:**
- Create: `src/constants.py`
- Modify: `src/steps/step2_cama.py`
- Modify: `src/steps/step4_merge.py`

**Step 1: 创建常量模块**

```python
# src/constants.py
"""WSE Pipeline 常量定义"""

# CaMa 分辨率
RESOLUTIONS = ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min']

# 数据源
VALID_SOURCES = ['hydrosat', 'hydroweb', 'cgls', 'icesat']

# Pipeline 步骤
PIPELINE_STEPS = ['download', 'validate', 'cama', 'reserved', 'merge']

# 数据完整性阈值
COMPLETENESS_THRESHOLDS = {
    'hydrosat': 2000,
    'hydroweb': 30000,
    'cgls': 10000,
    'icesat': 15000,
}
```

**Step 2: 更新所有引用**

**Step 3: Commit**

```bash
git commit -m "refactor: centralize constants in dedicated module"
```

---

### Task 10: 改进错误处理

**Files:**
- Modify: `src/readers/cgls_reader.py`
- Modify: `src/readers/hydroweb_reader.py`
- Modify: `src/readers/icesat_reader.py`
- Modify: `src/readers/hydrosat_reader.py`

**Step 1: 定义自定义异常**

```python
# src/exceptions.py
class WSEError(Exception):
    """WSE Pipeline 基础异常"""
    pass

class ReaderError(WSEError):
    """数据读取错误"""
    pass

class ValidationError(WSEError):
    """数据验证错误"""
    pass

class ConfigurationError(WSEError):
    """配置错误"""
    pass
```

**Step 2: 在 reader 中使用具体异常捕获**

```python
def read_station(self, filepath: str) -> Optional[Station]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        self.log('warning', f"JSON 解析失败 {filepath}: {e}")
        return None
    except FileNotFoundError:
        self.log('warning', f"文件不存在: {filepath}")
        return None
    except PermissionError:
        self.log('error', f"无权限读取: {filepath}")
        raise ReaderError(f"Permission denied: {filepath}")
```

**Step 3: Commit**

```bash
git commit -m "refactor: improve error handling with specific exception types"
```

---

### Task 11: 统一日志模式

**Files:**
- Modify: `src/steps/step1_validate.py`
- Modify: `src/steps/step2_cama.py`
- Modify: `src/steps/step4_merge.py`

**Step 1: 统一使用模块级 logger**

```python
# 每个步骤文件顶部
from ..utils.logger import get_logger

logger = get_logger(__name__)

class StepXXX:
    def __init__(self, config: dict):
        self.config = config
        # 不再创建实例级 logger
```

**Step 2: 更新所有 self.logger 为 logger**

**Step 3: Commit**

```bash
git commit -m "refactor: standardize logging to use module-level loggers"
```

---

## Phase 4: 测试覆盖率提升

### Task 12: 添加 Reader 测试

**Files:**
- Create: `tests/test_readers.py`

**Step 1: 创建测试数据 fixtures**

```python
@pytest.fixture
def sample_hydroweb_file(tmp_path):
    content = """#STATION:: TEST_STATION
#LON:: 10.5
#LAT:: 45.2
...
"""
    filepath = tmp_path / "test_station.txt"
    filepath.write_text(content)
    return str(filepath)
```

**Step 2: 为每个 reader 添加测试**

```python
class TestHydroWebReader:
    def test_read_valid_file(self, sample_hydroweb_file):
        reader = HydroWebReader({})
        station = reader.read_station(sample_hydroweb_file)
        assert station is not None
        assert station.lon == 10.5
        assert station.lat == 45.2

    def test_read_invalid_file(self, tmp_path):
        filepath = tmp_path / "invalid.txt"
        filepath.write_text("garbage data")
        reader = HydroWebReader({})
        station = reader.read_station(str(filepath))
        assert station is None
```

**Step 3: 运行测试确认覆盖率**

```bash
pytest tests/test_readers.py -v --cov=src/readers
```

**Step 4: Commit**

```bash
git commit -m "test: add comprehensive tests for data readers"
```

---

### Task 13: 添加 Step 测试

**Files:**
- Create: `tests/test_steps.py`

**Step 1: 为 Step1Validate 添加测试**

```python
class TestStep1Validate:
    def test_validates_coordinates(self, mock_config, sample_station_list):
        step = Step1Validate(mock_config)
        result = step.run(sample_station_list)
        assert result.valid_count > 0

    def test_rejects_invalid_coordinates(self, mock_config):
        invalid_station = Station(id="1", lon=999, lat=999, ...)
        stations = StationList([invalid_station])
        step = Step1Validate(mock_config)
        result = step.run(stations)
        assert result.invalid_count == 1
```

**Step 2: 为 Step2CaMa 添加测试（需要 mock CaMa 数据）**

**Step 3: 为 Step4Merge 添加测试**

**Step 4: Commit**

```bash
git commit -m "test: add unit tests for pipeline steps"
```

---

### Task 14: 添加 Utils 测试

**Files:**
- Create: `tests/test_utils.py`

**Step 1: 测试 config_loader**

```python
class TestConfigLoader:
    def test_load_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\nnested:\n  inner: 123")
        config = load_config(str(config_file))
        assert config['key'] == 'value'
        assert config['nested']['inner'] == 123
```

**Step 2: 测试 checkpoint**

**Step 3: 测试 logger**

**Step 4: Commit**

```bash
git commit -m "test: add unit tests for utility modules"
```

---

### Task 15: 最终验证

**Step 1: 运行完整测试套件**

```bash
pytest tests/ -v --cov=src --cov-report=html
```

**Step 2: 验证覆盖率 >= 70%**

**Step 3: 运行 dry-run 端到端测试**

```bash
wse --source hydroweb --dry-run
```

**Step 4: 更新文档**

**Step 5: 最终 Commit**

```bash
git commit -m "chore: complete code review fixes with >70% test coverage"
```

---

## 任务总结

| Phase | Tasks | 优先级 |
|-------|-------|--------|
| Phase 1: 安全修复 | Task 1-3 | Critical |
| Phase 2: 重要修复 | Task 4-8 | Important |
| Phase 3: 代码质量 | Task 9-11 | Minor |
| Phase 4: 测试覆盖 | Task 12-15 | Important |

**预计总任务数**: 15 个
**建议执行顺序**: Phase 1 → Phase 2 → Phase 4 → Phase 3
