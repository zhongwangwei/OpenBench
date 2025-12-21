# 气候态评估指南 (Climatology Evaluation Guide)

## 概述

气候态评估功能允许您将长时间序列的模拟数据与代表典型气候状态的参考数据进行比较。

## 气候态类型

### 1. 年平均气候态 (Annual Climatology)
- **参考数据**: 单个时间点或无时间维度
- **模拟数据**: 多年时间序列
- **处理方式**: 计算模拟数据的多年平均值,与参考数据比较

### 2. 月平均气候态 (Monthly Climatology)
- **参考数据**: 12个月的气候平均值 (1月-12月)
- **模拟数据**: 多年月度时间序列
- **处理方式**: 按月份分组计算模拟数据的多年月平均值,与参考数据的对应月份比较

## 配置方法

### 方法一: 使用 `data_groupby` 参数 (推荐)

在参考数据或模拟数据的配置文件中设置 `data_groupby: "climatology"`:

```json
{
  "general": {
    "root_dir": "./dataset/reference/climatology",
    "data_type": "grid",
    "data_groupby": "climatology",
    "syear": 2000,
    "eyear": 2020,
    "tim_res": "Month",
    "grid_res": 0.5
  },
  "Evapotranspiration": {
    "varname": "ET",
    "varunit": "mm month-1",
    "prefix": "ET_climatology",
    "suffix": ""
  }
}
```

**优点**:
- 类型一致 (`syear` 和 `eyear` 保持为整数)
- 语义清晰 (`data_groupby` 描述数据组织方式)
- 向后兼容

### 方法二: 自动检测

如果不设置 `data_groupby: "climatology"`,系统会根据参考数据的时间维度自动检测:

- **单个时间点**: 自动识别为年平均气候态
- **12个时间点**: 自动识别为月平均气候态
- **其他时间点数量**: 作为普通时间序列处理

## 配置示例

### 示例 1: 月平均气候态评估

**参考数据** (气候态,12个月):
```json
{
  "general": {
    "data_groupby": "climatology",
    "syear": 1980,
    "eyear": 2020
  }
}
```

**模拟数据** (时间序列):
```json
{
  "general": {
    "data_groupby": "year",
    "syear": 2000,
    "eyear": 2020
  }
}
```

**处理流程**:
1. 参考数据保持为12个月的气候平均值
2. 模拟数据 (2000-2020) 按月份分组,计算每个月的多年平均值
3. 对比12个月的气候态值

### 示例 2: 年平均气候态评估

**参考数据** (单个时间点):
```json
{
  "general": {
    "data_groupby": "climatology"
  }
}
```

**模拟数据** (时间序列):
```json
{
  "general": {
    "data_groupby": "year",
    "syear": 2000,
    "eyear": 2020
  }
}
```

**处理流程**:
1. 参考数据保持为单个年平均值
2. 模拟数据 (2000-2020) 计算所有年份的平均值
3. 对比年平均气候态值

## 支持的评估指标

气候态评估**支持**以下指标:
- bias (偏差)
- rmse (均方根误差)
- corr (相关系数)
- mae (平均绝对误差)
- std_ratio (标准差比率)
- 等空间统计指标

气候态评估**不支持**以下指标 (需要完整时间序列):
- nPhaseScore (相位得分)
- nIavScore (年际变率得分)

## 数据要求

### 参考数据要求
- **年平均气候态**:
  - 时间维度为1个时间点,或
  - 无时间维度
- **月平均气候态**:
  - 时间维度为12个时间点

### 模拟数据要求
- 包含完整的时间序列
- 时间范围应覆盖足够长的时期以计算稳定的气候平均值

## 数据文件结构

### 月平均气候态数据结构示例:

```python
<xarray.Dataset>
Dimensions:  (time: 12, lat: 360, lon: 720)
Coordinates:
  * time     (time) datetime64[ns] 2000-01-15 2000-02-15 ... 2000-12-15
  * lat      (lat) float64 -89.75 -89.25 -88.75 ... 89.25 89.75
  * lon      (lon) float64 -179.8 -179.2 -178.8 ... 179.2 179.8
Data variables:
    ET       (time, lat, lon) float32 ...
```

### 年平均气候态数据结构示例:

```python
<xarray.Dataset>
Dimensions:  (time: 1, lat: 360, lon: 720)
Coordinates:
  * time     (time) datetime64[ns] 2000-06-15
  * lat      (lat) float64 -89.75 -89.25 -88.75 ... 89.25 89.75
  * lon      (lon) float64 -179.8 -179.2 -178.8 ... 179.2 179.8
Data variables:
    ET       (time, lat, lon) float32 ...
```

## 时间坐标标准化

系统会自动标准化时间坐标:
- **年平均气候态**: 时间设为 `2000-06-15`
- **月平均气候态**: 时间设为 `2000-01-15` 到 `2000-12-15`

## 日志输出

启用气候态评估时,系统会输出以下信息:

```
INFO - 气候态评估模式已激活 (data_groupby='climatology')
INFO - 12 time points found - treating as monthly climatology
INFO - Reference set to monthly climatology with times: 2000-01-15 to 2000-12-15
INFO - Calculated monthly climatology from simulation data
INFO - ================================================================================
INFO - CLIMATOLOGY EVALUATION MODE DETECTED
INFO - ================================================================================
```

## 常见问题

### Q1: 如何确定使用年平均还是月平均气候态?
A: 取决于您的参考数据:
- 如果参考数据只有一个时间点,使用年平均
- 如果参考数据有12个时间点,使用月平均

### Q2: 模拟数据需要多长的时间序列?
A: 建议至少10年以上的数据,以获得稳定的气候平均值。更长的时间序列(如20-30年)会得到更稳定的气候态。

### Q3: 可以对站点数据使用气候态评估吗?
A: 可以,气候态评估同时支持网格数据和站点数据。

### Q4: 为什么某些指标不能用于气候态评估?
A: 某些指标 (如相位得分、年际变率) 需要完整的时间变化信息,而气候态数据只包含平均状态,因此不适用。

## 技术细节

### 处理流程

1. **模式检测**:
   - 检查 `data_groupby` 是否为 "climatology"
   - 或自动检测参考数据的时间维度

2. **参考数据处理**:
   - 标准化时间坐标
   - 验证时间点数量 (1 或 12)

3. **模拟数据处理**:
   - 年平均: `ds.mean(dim='time')`
   - 月平均: `ds.groupby('time.month').mean(dim='time')`

4. **兼容性验证**:
   - 检查两个数据集的时间维度是否匹配
   - 验证空间维度的兼容性

5. **指标筛选**:
   - 过滤不支持的指标
   - 继续评估支持的指标

### 内部实现

核心类: `ClimatologyProcessor`
- 位置: `openbench/data/Mod_Climatology.py`
- 主要方法:
  - `is_climatology_mode()`: 检测气候态模式
  - `detect_climatology_type()`: 检测气候态类型
  - `prepare_reference_climatology()`: 准备参考气候态
  - `prepare_simulation_climatology()`: 准备模拟气候态
  - `validate_climatology_compatibility()`: 验证兼容性

## 相关文档

- [Configuration Guide](./configuration_guide.md) - 配置文件详细说明
- [Metrics Guide](./metrics_guide.md) - 评估指标说明
