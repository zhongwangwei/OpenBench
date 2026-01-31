# NetCDF 输出功能设计

**日期:** 2026-01-31
**状态:** 待实现

---

## 概述

为 WSE Pipeline 添加 NetCDF 输出功能，生成与 `OpenBench_Streamflow_Daily.nc` 格式兼容的站点时间序列数据文件。

### 目标

- 输出包含完整时间序列的 NetCDF 文件
- 符合 CF-1.8 规范
- 统一时间轴，缺失值填 NaN
- 过滤上游面积 > 100 km² 的站点

---

## 输出文件结构

```
dimensions:
    station = <N>
    time = <T>

variables:
    # 时间序列数据
    float wse(station, time)           # 水面高程 (m)
    byte data_source(station, time)    # 数据源代码

    # 站点元数据
    double lat(station)
    double lon(station)
    string station_id(station)
    string station_name(station)
    float elevation(station)           # 平均高程
    int num_observations(station)
    float EGM08(station)
    float EGM96(station)
    string source(station)             # hydroweb/cgls/icesat/hydrosat

    # CaMa 分配结果 (每个分辨率)
    float cama_lat_01min(station)
    float cama_lon_01min(station)
    byte cama_flag_01min(station)
    float cama_lat_03min(station)
    float cama_lon_03min(station)
    byte cama_flag_03min(station)
    ... (05min, 06min, 15min)

    # 时间坐标
    int64 time(time)                   # days since 1800-01-01

global attributes:
    title = "OpenBench Water Surface Elevation Dataset"
    Conventions = "CF-1.8"
    institution = "OpenBench"
    source = "HydroWeb, CGLS, ICESat, ICESat-2, HydroSat"
    references = "https://github.com/zhongwangwei/OpenBench"
```

### data_source 编码

| 代码 | 数据源 |
|------|--------|
| 1 | HydroWeb |
| 2 | CGLS |
| 3 | ICESat (GLA14) |
| 4 | ICESat-2 (ATL13) |
| 5 | HydroSat |

---

## 实现架构

### 文件结构

```
src/
├── steps/
│   └── step4_merge.py      # 修改：添加 NetCDF 输出模式
├── writers/                 # 新增目录
│   ├── __init__.py
│   └── netcdf_writer.py    # 新增：NetCDF 写入器
└── core/
    └── station.py          # 无需修改
```

### 处理流程

```
Step 1-3 (现有)          Step 4 (修改)
    │                        │
    ▼                        ▼
StationList ──────────► NetCDFWriter
(元数据+CaMa结果)              │
                              │ 回读源文件获取时间序列
                              ▼
                        ┌─────────────┐
                        │ Readers     │
                        │ (hydroweb,  │
                        │  cgls, ...) │
                        └─────────────┘
                              │
                              ▼
                        OpenBench_WSE.nc
```

### 关键设计决策

1. **时间序列不存储在 Station 对象中** - 避免内存爆炸
2. **NetCDF 写入时按需读取源文件** - 使用现有 Reader 的 `read_timeseries()` 方法
3. **分批写入** - 每次处理一批站点，避免一次性加载所有数据

---

## 站点过滤规则

```python
def should_include_station(station) -> bool:
    """
    进入 NetCDF 的站点必须满足：
    - 至少一个分辨率有有效的 CaMa 分配 (flag > 0)
    - 该分辨率的上游面积 > 100 km²
    """
    for res in ['01min', '03min', '05min', '06min', '15min']:
        cama = station.cama_results.get(f'glb_{res}', {})
        if cama.get('flag', 0) > 0:  # 有效分配
            uparea = cama.get('uparea', 0)
            if uparea > 100:  # km²
                return True
    return False
```

---

## 时间轴

### 各数据源时间范围

| 数据源 | 起始 | 结束 | 频率 |
|--------|------|------|------|
| HydroWeb | 1995-01-01 | 2024-12-31 | 日 |
| CGLS | 2016-01-01 | 2024-12-31 | 日 |
| ICESat (GLA14) | 2003-01-01 | 2009-12-31 | 轨道 |
| ICESat-2 (ATL13) | 2018-10-01 | 2024-12-31 | 轨道 |
| HydroSat | 2002-01-01 | 2024-12-31 | 日 |

### 统一时间轴

- 范围: 1995-01-01 至 2024-12-31
- 频率: 日
- 时间点数: ~11000
- 参考: days since 1800-01-01

---

## NetCDFWriter 类设计

```python
# src/writers/netcdf_writer.py

class NetCDFWriter:
    """将站点数据写入 CF-1.8 兼容的 NetCDF 文件"""

    SOURCE_CODES = {
        'hydroweb': 1,
        'cgls': 2,
        'icesat': 3,
        'icesat2': 4,
        'hydrosat': 5,
    }

    RESOLUTIONS = ['01min', '03min', '05min', '06min', '15min']

    def __init__(self, config: dict):
        self.output_path = Path(config.get('netcdf_file', 'OpenBench_WSE.nc'))
        self.time_ref = config.get('time_reference', '1800-01-01')
        self.chunk_size = config.get('chunk_size', 1000)
        self.min_uparea = config.get('min_uparea', 100.0)

        # Reader 实例 (用于回读时间序列)
        self.readers = {}

    def write(self, stations: StationList, data_paths: dict) -> Path:
        """
        写入 NetCDF 文件

        Args:
            stations: 处理完成的站点列表
            data_paths: 各数据源路径 {'hydroweb': '/path/...', ...}

        Returns:
            输出文件路径
        """
        # 1. 过滤站点
        filtered = self._filter_stations(stations)

        # 2. 确定时间轴
        time_axis = self._build_time_axis(filtered, data_paths)

        # 3. 创建 NetCDF 结构
        self._create_netcdf(filtered, time_axis)

        # 4. 分批写入数据
        self._write_data_batched(filtered, data_paths, time_axis)

        return self.output_path

    def _filter_stations(self, stations: StationList) -> List[Station]:
        """过滤：上游面积 > 100km² 且有有效 CaMa 分配"""
        ...

    def _build_time_axis(self, stations, data_paths) -> np.ndarray:
        """扫描所有时间序列，确定统一时间轴"""
        ...

    def _create_netcdf(self, stations, time_axis):
        """创建 NetCDF 文件结构"""
        ...

    def _write_data_batched(self, stations, data_paths, time_axis):
        """分批读取时间序列并写入 NetCDF"""
        for batch in chunks(stations, self.chunk_size):
            for station in batch:
                ts = self._read_timeseries(station, data_paths)
                self._write_station_timeseries(station, ts, time_axis)

    def _read_timeseries(self, station, data_paths) -> List[dict]:
        """使用对应 Reader 读取站点时间序列"""
        ...

    def _write_station_timeseries(self, station, timeseries, time_axis):
        """将单个站点的时间序列写入 NetCDF"""
        ...
```

---

## 配置选项

```yaml
# config/global.yaml

output:
  format: netcdf              # txt (默认) 或 netcdf
  netcdf_file: OpenBench_WSE.nc
  time_reference: "1800-01-01"
  chunk_size: 1000            # 每批处理站点数
  min_uparea: 100.0           # km², 最小上游面积过滤
```

---

## 实现计划

### 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `src/writers/__init__.py` | 模块初始化 |
| 新增 | `src/writers/netcdf_writer.py` | NetCDF 写入器核心类 |
| 修改 | `src/steps/step4_merge.py` | 集成 NetCDF 输出选项 |
| 修改 | `config/global.yaml` | 添加 output.format 配置 |
| 新增 | `tests/test_netcdf_writer.py` | 单元测试 |

### 实现步骤

1. **创建 NetCDFWriter 基础结构**
   - 站点过滤逻辑 (uparea > 100km²)
   - 时间轴构建
   - NetCDF 文件创建（维度、变量定义）

2. **实现时间序列读取**
   - 复用现有 Reader 的 `read_timeseries()` 方法
   - 时间格式统一转换为 days since 1800-01-01

3. **实现分批写入**
   - 站点元数据写入
   - 时间序列数据分批写入
   - 进度显示

4. **集成到 Step4**
   - 添加 format 配置选项
   - 修改 run() 方法

5. **测试**
   - 单元测试
   - 与参考文件格式对比验证

---

## 依赖

```
netCDF4>=1.6.0
numpy>=1.20.0
```

---

## 预估

### 输出文件大小

- 假设 5000 站点，11000 时间点
- wse: 5000 × 11000 × 4 bytes ≈ 210 MB
- 加上元数据和压缩后 ≈ 100-150 MB

### 处理时间

- 主要瓶颈在于回读源文件获取时间序列
- 分批处理可控制内存使用

---

## 参考

- 参考文件: `/Users/zhongwangwei/Desktop/Github/OpenBench-wei/dataset/Reference/Station/Water/StreamFlow/Daily/OpenBench_Streamflow_Daily.nc`
- CF Conventions: http://cfconventions.org/
