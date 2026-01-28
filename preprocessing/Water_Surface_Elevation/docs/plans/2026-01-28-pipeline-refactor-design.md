# WSE Pipeline 重构设计

**Date**: 2026-01-28
**Status**: Approved
**Author**: Claude + User

## 概述

重构 WSE Pipeline，将数据下载整合为 Step 0，清理无用代码，完善 Pipeline 功能。

## 设计决策

| 问题 | 决策 |
|------|------|
| 数据下载整合方式 | Step 0 前置步骤 |
| 旧文件处理 | 归档到 `_archive/` |
| Step 0 行为 | 提示并询问用户 |
| 多数据源处理 | 灵活选择（单个/多个/全部） |
| 输出组织 | 默认分开，`--merge` 可合并 |
| CLI 入口 | 模块方式 + `wse` 快捷命令 |

## Pipeline 架构

```
Step 0: Download (新增)
    ↓ 检测数据完整性，询问是否下载缺失数据
Step 1: Validate + EGM
    ↓ 验证数据，计算 EGM08/EGM96
Step 2: CaMa Allocate
    ↓ 分配到 CaMa 网格（5 种分辨率）
Step 3: Reserved
    ↓ 预留扩展
Step 4: Merge (可选)
    ↓ 合并多数据源或与现有 NC 合并
```

## CLI 使用

```bash
# 处理单个数据源
wse --source hydroweb

# 处理多个数据源
wse --source hydroweb,cgls,icesat

# 处理全部并合并
wse --source all --merge

# 跳过下载检查
wse --source hydroweb --skip-download

# 模块方式运行
python -m src.main --source hydroweb
```

## 项目结构

```
preprocessing/Water_Surface_Elevation/
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI 入口
│   ├── pipeline.py                # Pipeline 控制器
│   │
│   ├── steps/
│   │   ├── step0_download.py      # 新增：数据下载
│   │   ├── step1_validate.py      # 验证 + EGM 计算
│   │   ├── step2_cama.py          # CaMa 分配
│   │   ├── step3_reserved.py      # 预留
│   │   └── step4_merge.py         # 合并输出
│   │
│   ├── readers/                   # 数据读取器
│   │   ├── base_reader.py
│   │   ├── hydroweb_reader.py
│   │   ├── cgls_reader.py
│   │   ├── icesat_reader.py
│   │   ├── hydrosat_reader.py
│   │   └── downloader.py          # 下载器
│   │
│   ├── core/                      # 核心算法
│   │   ├── geoid_calculator.py
│   │   ├── cama_allocator.py
│   │   └── station.py             # 站点数据结构
│   │
│   └── utils/                     # 工具
│       ├── config_loader.py
│       ├── checkpoint.py
│       └── logger.py
│
├── config/                        # 配置文件
├── tests/                         # 测试
├── _archive/                      # 归档旧代码
├── pyproject.toml                 # 包配置 + wse 入口点
└── docs/plans/
```

## Step 0: Download

### 数据完整性检查

```python
COMPLETENESS_RULES = {
    'hydrosat': {'min_files': 2000, 'pattern': '*.txt'},
    'hydroweb': {'min_files': 30000, 'pattern': '*.txt'},
    'cgls':     {'min_files': 10000, 'pattern': '*.geojson'},
    'icesat':   {'min_files': 15000, 'pattern': '*.h5'},
}
```

### 交互流程

```
[Step 0] 检查数据完整性...

  ✓ HydroSat: 2,036 文件 (完整)
  ✓ HydroWeb: 35,880 文件 (完整)
  ✗ CGLS: 1,910 / 11,759 文件 (不完整)
  ✗ ICESat: 753 / 17,968 文件 (不完整)

是否下载缺失数据？ [Y/n/skip]
  Y - 下载后继续处理
  n - 退出
  skip - 跳过下载，用现有数据继续
```

### 凭证管理

- 从 `config/credentials.yaml` 读取账号密码
- 支持 `--num-workers` 设置并行数
- 下载进度保存到 checkpoint

## Step 1: Validate + EGM

### 验证规则

```python
VALIDATION_RULES = {
    'lon_range': (-180, 180),
    'lat_range': (-90, 90),
    'min_observations': 10,
    'max_elevation_error': 100,
    'check_duplicates': True,
}
```

### 处理流程

1. 使用对应 Reader 读取数据源
2. 验证每个站点（坐标、观测数量、重复检测）
3. 计算 EGM（忽略数据源自带值）
4. 输出 StationList 供 Step 2 使用

### 统计报告

```
[Step 1] 验证完成
  总站点: 50,000
  有效站点: 48,532
  无效站点: 1,468 (坐标异常: 23, 观测不足: 1,445)
```

## Step 2: CaMa Allocate

### 分配算法

```
对每个站点：
  1. 确定所属 10°×10° 瓦片
  2. 加载 CaMa 地图数据 (uparea, nextxy, elevtn, rivwth)
  3. 搜索最近河道中心线
  4. 计算分配参数：
     - kx1, ky1: 上游网格坐标
     - kx2, ky2: 下游网格坐标
     - dist1, dist2: 到上下游距离
     - rivwth: 河宽
     - ix, iy: 最终分配网格
```

### 5 种分辨率

```python
RESOLUTIONS = ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min']
```

输出列名带分辨率后缀：`kx1_glb_01min`, `ky1_glb_01min`, ...

## Step 3: Reserved

预留扩展，当前直接透传数据。

未来可扩展：
- 人类影响指数 (HII) 计算
- 上游大坝影响分析
- 数据质量评分
- 与实测数据交叉验证

## Step 4: Merge

### 输出模式

**模式 1: 默认分开存储**
```
output/
├── hydroweb_stations.txt
├── cgls_stations.txt
└── icesat_stations.txt
```

**模式 2: --merge 合并存储**
```
output/
└── all_stations.txt  (增加 source 列)
```

**模式 3: --merge-nc existing.nc**
- 检测重复站点（经纬度匹配）
- 合并时间序列
- 更新元数据

## 实现计划

### Phase 1: 代码清理

1. 创建 `_archive/` 目录
2. 移动根目录 12 个 .py 文件到 `_archive/`
3. 创建 `tests/` 目录，移动测试文件
4. 创建 `pyproject.toml`（wse 入口点）

### Phase 2: Step 0 Download 整合

1. 创建 `src/steps/step0_download.py`
2. 实现数据完整性检查
3. 整合现有 `downloader.py`
4. 添加交互式提示

### Phase 3: Step 1-2 完善

1. 完善 4 个 Reader（确保都能正常读取）
2. 整合 geoid_calculator 到 Step 1
3. 整合 cama_allocator 到 Step 2
4. 实现 5 种分辨率并行处理

### Phase 4: 集成测试

1. 端到端测试单数据源
2. 测试多数据源 + 合并
3. 测试断点续传

## 数据源凭证

| 数据源 | 认证方式 |
|--------|----------|
| HydroSat | 无需认证 |
| HydroWeb | API Key |
| CGLS | CDSE OAuth2 |
| ICESat | NASA Earthdata Basic Auth |

## 当前数据状态

| 数据源 | 已下载 | 总数 | 状态 |
|--------|--------|------|------|
| HydroSat | 2,036 | 2,036 | ✅ 完成 |
| HydroWeb | 35,880 | 35,880 | ✅ 完成 |
| CGLS | 1,910 | 11,759 | ⏸ 待继续 |
| ICESat | 753 | 17,968 | ⏸ 待继续 |
