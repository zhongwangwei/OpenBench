# WSE Pipeline 用户手册

**Water Surface Elevation (WSE) 数据处理管道**

版本: 0.2.0
更新日期: 2026-01-29

---

## 目录

1. [概述](#1-概述)
2. [安装](#2-安装)
3. [快速开始](#3-快速开始)
4. [配置](#4-配置)
5. [命令行使用](#5-命令行使用)
6. [Pipeline 步骤详解](#6-pipeline-步骤详解)
7. [数据源](#7-数据源)
8. [输出格式](#8-输出格式)
9. [API 参考](#9-api-参考)
10. [故障排除](#10-故障排除)

---

## 1. 概述

WSE Pipeline 是一个用于处理卫星测高水面高程数据的 Python 工具。它支持多种数据源（HydroWeb、CGLS、ICESat、ICESat-2、HydroSat），并将虚拟站点分配到 CaMa-Flood 网格。

### 主要功能

- **多数据源支持**: HydroWeb、CGLS、ICESat (GLA14)、ICESat-2 (ATL13)、HydroSat
- **5 步处理流程**: 下载 → 验证 → CaMa分配 → 预留 → 合并
- **5 种 CaMa 分辨率**: glb_01min, glb_03min, glb_05min, glb_06min, glb_15min
- **自动计算 EGM08/EGM96**: 不依赖数据源元数据
- **断点续传**: 支持从检查点恢复处理
- **灵活输出**: 分开存储或合并输出

### 系统要求

- Python 3.9+
- 操作系统: macOS, Linux, Windows
- 磁盘空间: 取决于数据量（建议 50GB+）

---

## 2. 安装

### 2.1 从源码安装

```bash
cd preprocessing/Water_Surface_Elevation
pip install -e .
```

### 2.2 依赖项

核心依赖:
```
click>=8.0
pyyaml>=6.0
numpy>=1.20
requests>=2.25
tqdm>=4.60
```

可选依赖:
```
h5py>=3.0        # ICESat HDF5 文件支持
pygeodesy>=23.0  # EGM 计算（替代 GeoidEval CLI）
```

### 2.3 验证安装

```bash
wse --help
# 或
python -m src.main --help
```

---

## 3. 快速开始

### 3.1 处理单个数据源

```bash
# 处理 HydroWeb 数据
wse --source hydroweb

# Dry-run 模式（不实际执行）
wse --source hydroweb --dry-run
```

### 3.2 处理多个数据源

```bash
# 处理指定数据源
wse --source hydroweb,cgls,icesat

# 处理全部数据源
wse --source all
```

### 3.3 合并输出

```bash
# 处理全部并合并为单个文件
wse --source all --merge
```

---

## 4. 配置

### 4.1 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `WSE_DATA_ROOT` | 数据根目录 | `./data` |
| `WSE_OUTPUT_DIR` | 输出目录 | `./output` |
| `WSE_CAMA_ROOT` | CaMa 地图数据目录 | (必需) |
| `WSE_GEOID_ROOT` | EGM Geoid 数据目录 | (必需) |
| `WSE_CHECKPOINT_KEY` | 检查点签名密钥 | `default-dev-key` |

设置示例:
```bash
export WSE_DATA_ROOT=/Volumes/Data01/Altimetry
export WSE_CAMA_ROOT=/Volumes/Data01/CaMa
export WSE_GEOID_ROOT=/Volumes/Data01/egm-geoids
export WSE_OUTPUT_DIR=./output
```

### 4.2 配置文件

创建 `config/my_dataset.yaml`:

```yaml
dataset:
  name: "MyDataset_2024"
  source: "hydroweb"
  version: "1.0"

processing:
  calculate_egm: true
  egm96_model: egm96-5
  egm2008_model: egm2008-1
  cama_resolutions:
    - glb_01min
    - glb_03min
    - glb_05min
    - glb_06min
    - glb_15min

filters:
  min_observations: 10
  start_date: null
  end_date: null
  bbox: null  # [min_lon, min_lat, max_lon, max_lat]

output:
  format: txt
  include_timeseries: false
```

使用配置文件:
```bash
wse --config config/my_dataset.yaml
```

### 4.3 凭证配置

凭证已整合到各数据源的配置文件中（`config/{source}.yaml`）:

**HydroWeb** (`config/hydroweb.yaml`):
```yaml
credentials:
  api_key: "your-api-key"
```

**CGLS** (`config/cgls.yaml`):
```yaml
credentials:
  username: "your-email@example.com"
  password: "your-password"
```

**ICESat / ICESat-2** (`config/icesat.yaml`, `config/icesat2.yaml`):
```yaml
credentials:
  username: "earthdata-username"
  password: "earthdata-password"
```

也可以通过环境变量设置:
```bash
export HYDROWEB_API_KEY="your-api-key"
export CDSE_USERNAME="your-email"
export CDSE_PASSWORD="your-password"
export EARTHDATA_USERNAME="your-username"
export EARTHDATA_PASSWORD="your-password"
```

---

## 5. 命令行使用

### 5.1 基本语法

```bash
wse [OPTIONS]
```

### 5.2 选项说明

| 选项 | 说明 | 示例 |
|------|------|------|
| `--source` | 数据源（单个/多个/all） | `--source hydroweb,cgls` |
| `--config` | 配置文件路径 | `--config config/my.yaml` |
| `--output` | 输出目录 | `--output ./results` |
| `--merge` | 合并多源输出为单文件 | `--merge` |
| `--skip-download` | 跳过数据下载检查 | `--skip-download` |
| `--step` | 只运行指定步骤 | `--step validate` |
| `--num-workers` | 并行下载线程数 | `--num-workers 4` |
| `--log-level` | 日志级别 | `--log-level DEBUG` |
| `--dry-run` | 模拟运行，不写入文件 | `--dry-run` |

### 5.3 使用示例

```bash
# 完整 Pipeline
wse --source hydroweb

# 只运行验证步骤
wse --source hydroweb --step validate

# 指定输出目录和日志级别
wse --source all --output ./results --log-level DEBUG

# 跳过下载，使用现有数据
wse --source cgls --skip-download

# 并行下载（4线程）
wse --source icesat --num-workers 4
```

---

## 6. Pipeline 步骤详解

### 6.1 Step 0: Download（数据下载）

检查数据完整性，提示下载缺失数据。

**数据完整性阈值:**

| 数据源 | 最小文件数 | 说明 |
|--------|-----------|------|
| HydroSat | 2,000 | Stuttgart 大学 |
| HydroWeb | 30,000 | Theia/CNES |
| CGLS | 10,000 | Copernicus |
| ICESat | 15,000 | ICESat-1 GLA14 (2003-2009) |
| ICESat-2 | 10,000 | ICESat-2 ATL13 (2018-至今) |

**交互流程:**
```
[Step 0] 检查数据完整性...

  ✓ HydroSat: 2,036 文件 (完整)
  ✓ HydroWeb: 35,880 文件 (完整)
  ✗ CGLS: 1,910 / 10,000 文件 (不完整)

是否下载缺失数据？ [Y/n/skip]
```

### 6.2 Step 1: Validate（验证 + EGM 计算）

验证站点数据，计算 EGM08/EGM96 大地水准面差距。

**验证规则:**
- 经度范围: -180 ~ 180
- 纬度范围: -90 ~ 90
- 最小观测数: 10（可配置）
- 重复站点检测

**输出统计:**
```
[Step 1] 验证完成
  总站点: 50,000
  有效站点: 48,532
  无效站点: 1,468 (坐标异常: 23, 观测不足: 1,445)
```

### 6.3 Step 2: CaMa Allocate（CaMa 分配）

将站点分配到 CaMa-Flood 网格。

**分配算法:**
1. 确定所属 10°×10° 瓦片
2. 加载 CaMa 地图数据 (uparea, nextxy, elevtn, rivwth)
3. 搜索最近河道中心线
4. 计算分配参数: kx1, ky1, kx2, ky2, dist1, dist2, rivwth, ix, iy

**5 种分辨率:**
- `glb_01min` (1 弧分 ≈ 1.85 km)
- `glb_03min` (3 弧分 ≈ 5.55 km)
- `glb_05min` (5 弧分 ≈ 9.25 km)
- `glb_06min` (6 弧分 ≈ 11.1 km)
- `glb_15min` (15 弧分 ≈ 27.75 km)

### 6.4 Step 3: Reserved（预留）

预留扩展，当前直接透传数据。

**未来可扩展:**
- 人类影响指数 (HII) 计算
- 上游大坝影响分析
- 数据质量评分

### 6.5 Step 4: Merge（合并输出）

生成最终输出文件。

**输出模式:**

| 模式 | 命令 | 输出 |
|------|------|------|
| 分开存储 | `wse --source all` | `hydroweb_stations.txt`, `cgls_stations.txt`, ... |
| 合并存储 | `wse --source all --merge` | `all_stations.txt` (含 source 列) |

---

## 7. 数据源

### 7.1 HydroWeb

**格式:** 文本文件，`#KEY:: value` 头部

**示例文件:**
```
#STATION:: AMAZON_MANAUS
#LON:: -60.025
#LAT:: -3.142
#SATELLITE:: Jason-2
...
DATE        HEIGHT    ERROR
2008-07-04  16.234    0.05
2008-07-14  16.189    0.04
```

**路径:** `$WSE_DATA_ROOT/hydroweb_river/`

### 7.2 CGLS (Copernicus Global Land Service)

**格式:** GeoJSON

**示例文件:**
```json
{
  "type": "Feature",
  "properties": {
    "station_id": "VS_001",
    "longitude": 10.5,
    "latitude": 45.2
  },
  "data": [
    {"date": "2020-01-01", "wse": 123.45, "uncertainty": 0.1}
  ]
}
```

**路径:** `$WSE_DATA_ROOT/CGLS/river/`

### 7.3 ICESat-1 GLA14

**卫星:** ICESat-1 (2003-2009, 已退役)
**产品:** GLAH14 Land Surface Altimetry
**格式:** 空格分隔文本，无头部
**文件命名:** `lat_lon_xxx.txt` (如 `45.5_10.2_001.txt`)
**路径:** `/Volumes/Data01/Altimetry/ICESat_GLA14/txt_water/`
**配置文件:** `config/icesat.yaml`

```bash
wse --source icesat
```

### 7.4 ICESat-2 ATL13

**卫星:** ICESat-2 (2018-至今)
**产品:** ATL13 Inland Water Surface Height
**格式:** HDF5 (.h5)
**路径:** `/Volumes/Data01/Altimetry/ICESat2_ATL13/`
**配置文件:** `config/icesat2.yaml`

```bash
wse --source icesat2
```

**同时下载两个 ICESat 数据源:**
```bash
wse --source icesat,icesat2
```

### 7.5 HydroSat

**格式:** 文本文件，头部 + 数据

**路径:** `$WSE_DATA_ROOT/WL_hydrosat/`

---

## 8. 输出格式

### 8.1 站点列表文件

输出为制表符分隔的文本文件。

**列名:**
```
ID  station  dataname  lon  lat  satellite  flag  elevation  dist_to_mouth
kx1_glb_01min  ky1_glb_01min  kx2_glb_01min  ky2_glb_01min  dist1_glb_01min  dist2_glb_01min  rivwth_glb_01min  ix_glb_01min  iy_glb_01min  lon_cama_glb_01min  lat_cama_glb_01min
kx1_glb_03min  ky1_glb_03min  ...
kx1_glb_05min  ky1_glb_05min  ...
kx1_glb_06min  ky1_glb_06min  ...
kx1_glb_15min  ky1_glb_15min  ...
EGM08  EGM96
```

**合并模式额外列:** `source` (数据源标识)

### 8.2 列说明

| 列名 | 说明 | 单位 |
|------|------|------|
| `ID` | 站点唯一标识 | - |
| `station` | 站点名称 | - |
| `lon`, `lat` | 经纬度 | 度 |
| `elevation` | 平均高程 | 米 |
| `kx1`, `ky1` | 上游网格坐标 | - |
| `kx2`, `ky2` | 下游网格坐标 | - |
| `dist1`, `dist2` | 到上下游距离 | 米 |
| `rivwth` | 河宽 | 米 |
| `ix`, `iy` | 最终分配网格 | - |
| `EGM08`, `EGM96` | 大地水准面差距 | 米 |

---

## 9. API 参考

### 9.1 Pipeline 类

```python
from src.pipeline import Pipeline, PipelineResult

# 初始化
pipeline = Pipeline(config)

# 运行完整 Pipeline
result: PipelineResult = pipeline.run(['hydroweb', 'cgls'])

# 运行单个步骤
result: PipelineResult = pipeline.run_step('validate', ['hydroweb'])

# 检查结果
if result.success:
    print(f"处理站点: {result.stations_processed}")
    print(f"输出文件: {result.output_files}")
else:
    print(f"错误: {result.error}")
```

### 9.2 Station 类

```python
from src.core.station import Station, StationList

# 创建站点
station = Station(
    id="VS001",
    name="Amazon_Manaus",
    lon=-60.025,
    lat=-3.142,
    source="hydroweb",
    elevation=16.5,
    num_observations=100
)

# 验证站点
if station.is_valid():
    print("站点有效")

# 设置 CaMa 结果
station.set_cama_result('glb_01min', {
    'kx1': 100, 'ky1': 200,
    'ix': 150, 'iy': 250
})

# 站点列表
stations = StationList([station1, station2, ...])
valid_stations = stations.filter_valid()
hydroweb_stations = stations.filter_by_source('hydroweb')
```

### 9.3 Reader 类

```python
from src.readers import get_reader, HydroWebReader

# 使用工厂函数
reader = get_reader('hydroweb', config)

# 直接实例化
reader = HydroWebReader(config)

# 扫描目录
filepaths = reader.scan_directory('/path/to/data')

# 读取单个站点
station = reader.read_station('/path/to/station.txt')

# 读取所有站点
stations = reader.read_all_stations('/path/to/data')
```

### 9.4 Step 类

```python
from src.steps import Step1Validate, Step2CaMa, Step4Merge

# Step 1: 验证
step1 = Step1Validate(config)
result = step1.run(stations)

# Step 2: CaMa 分配
step2 = Step2CaMa(config)
result = step2.run(stations)

# Step 4: 合并输出
step4 = Step4Merge(config)
result = step4.run(stations, merge=True)
```

### 9.5 异常类

```python
from src.exceptions import (
    WSEError,           # 基础异常
    ReaderError,        # 数据读取错误
    ValidationError,    # 数据验证错误
    ConfigurationError, # 配置错误
    DownloadError       # 下载错误
)

try:
    reader.read_station(filepath)
except ReaderError as e:
    print(f"读取失败: {e}")
```

---

## 10. 故障排除

### 10.1 常见问题

**Q: 提示 "GeoidEval not found"**

A: 安装 GeographicLib 或使用 pygeodesy:
```bash
# macOS
brew install geographiclib

# 或安装 Python 替代
pip install pygeodesy
```

**Q: 提示 "cama_root not configured"**

A: 设置环境变量:
```bash
export WSE_CAMA_ROOT=/path/to/cama/data
```

**Q: 下载失败 "SSL verification error"**

A: 在配置中禁用 SSL 验证（仅限内网）:
```yaml
verify_ssl: false
```

**Q: 内存不足**

A: 减少并行线程数:
```bash
wse --source all --num-workers 1
```

### 10.2 日志查看

日志文件位于 `logs/` 目录:
```bash
tail -f logs/wse_2026-01-28.log
```

调试模式:
```bash
wse --source hydroweb --log-level DEBUG
```

### 10.3 检查点恢复

Pipeline 自动保存检查点。如果中断，重新运行会提示是否从检查点恢复:
```
发现检查点: step_validate
是否从检查点恢复？ [Y/n]
```

### 10.4 获取帮助

- GitHub Issues: https://github.com/anthropics/claude-code/issues
- 查看帮助: `wse --help`

---

## 附录

### A. 目录结构

```
preprocessing/Water_Surface_Elevation/
├── src/                    # 源代码
│   ├── main.py            # CLI 入口
│   ├── pipeline.py        # Pipeline 控制器
│   ├── constants.py       # 常量定义
│   ├── exceptions.py      # 自定义异常
│   ├── core/              # 核心算法
│   ├── steps/             # Pipeline 步骤
│   ├── readers/           # 数据读取器
│   └── utils/             # 工具函数
├── config/                 # 配置文件
├── tests/                  # 测试文件
├── docs/                   # 文档
├── output/                 # 输出目录
└── pyproject.toml         # 包配置
```

### B. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.2.0 | 2026-01-29 | 添加 ICESat-2 ATL13 支持，整合凭证配置 |
| 0.1.0 | 2026-01-28 | 初始版本，完成代码审查修复 |

### C. 许可证

MIT License
