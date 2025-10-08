# Climatology Evaluation

OpenBench现在支持气候态(climatology)评估,可以将模拟结果与气候态参考数据进行对比。

## 功能概述

气候态评估允许您将长时间序列的模拟数据与代表典型气候状态的参考数据进行比较:

1. **年平均气候态**: 参考数据为单个时间点或无时间维度
2. **月平均气候态**: 参考数据为12个月的气候平均值

## 工作原理

### 自动检测

系统会自动检测参考数据的时间维度:

- **年气候态**: 如果参考数据只有1个时间点或无时间维度
- **月气候态**: 如果参考数据有12个时间点

### 数据处理

#### 年平均气候态 (Annual Climatology)

**参考数据**:
- 时间维度: 1个时间点或无时间维度
- 处理后: 时间设置为 `2000-06-15`

**模拟数据**:
- 原始数据: 多年时间序列 (例如 2010-2020)
- 处理后: 计算所有年份的平均值,时间设置为 `2000-06-15`

**示例**:
```python
# 参考数据 (年平均蒸散发气候态)
ref_ET = 50.0  # mm/month (单值或单时间点)

# 模拟数据 (2010-2020年逐月数据)
sim_ET = [45, 48, 51, 49, 52, ...]  # 132个月的数据

# 处理后
ref_ET_clim = 50.0  @ time='2000-06-15'
sim_ET_clim = 49.5  @ time='2000-06-15'  # 132个月的平均值
```

#### 月平均气候态 (Monthly Climatology)

**参考数据**:
- 时间维度: 12个时间点 (代表12个月)
- 处理后: 时间设置为 `2000-01-15, 2000-02-15, ..., 2000-12-15`

**模拟数据**:
- 原始数据: 多年时间序列
- 处理后: 计算多年的月平均值,得到12个值,时间设置为 `2000-01-15, ..., 2000-12-15`

**示例**:
```python
# 参考数据 (月平均蒸散发气候态)
ref_ET = [20, 25, 35, 45, 55, 60, 65, 60, 50, 40, 30, 22]  # 12个月的气候值

# 模拟数据 (2010-2020年逐月数据)
# 2010年: [18, 23, 33, ...]
# 2011年: [19, 24, 34, ...]
# ...
# 2020年: [21, 26, 36, ...]

# 处理后 (对每个月计算多年平均)
sim_ET_clim = [19.5, 24.5, 34.5, ...]  @ times=['2000-01-15', '2000-02-15', ...]
```

### 指标兼容性

#### 支持的指标

气候态评估支持以下指标和评分:

- ✅ `bias` - 偏差
- ✅ `percent_bias` - 百分比偏差
- ✅ `absolute_percent_bias` - 绝对百分比偏差
- ✅ `RMSE` - 均方根误差
- ✅ `ubRMSE` - 无偏均方根误差
- ✅ `MAE` - 平均绝对误差
- ✅ `corr` - 相关系数
- ✅ `R2` - 决定系数
- ✅ `NSE` - Nash-Sutcliffe效率系数
- ✅ `KGE` - Kling-Gupta效率系数
- ✅ `nBiasScore` - 归一化偏差评分
- ✅ `nRMSEScore` - 归一化RMSE评分
- ✅ `nSpatialScore` - 空间分布评分

#### 不支持的指标

以下指标需要时间变化信息,不适用于气候态评估:

- ❌ `nPhaseScore` - 相位评分(需要季节变化)
- ❌ `nIavScore` - 年际变率评分(需要多年数据)

系统会自动跳过不支持的指标,并在日志中记录。

## 使用方法

### 1. 准备参考数据

#### 年气候态格式

NetCDF文件应包含:
- 一个时间点,或
- 无时间维度(仅空间维度)

```python
# 创建年气候态参考数据示例
import xarray as xr
import numpy as np

# 方式1: 单时间点
ds = xr.Dataset({
    'ET': (['time', 'lat', 'lon'], data)
}, coords={
    'time': ['2000-01-01'],  # 任意时间,会被重置
    'lat': lats,
    'lon': lons
})

# 方式2: 无时间维度
ds = xr.Dataset({
    'ET': (['lat', 'lon'], data)
}, coords={
    'lat': lats,
    'lon': lons
})
```

#### 月气候态格式

NetCDF文件应包含12个时间点:

```python
# 创建月气候态参考数据示例
import pandas as pd

# 12个月的气候平均值
times = pd.date_range('2000-01-01', periods=12, freq='MS')

ds = xr.Dataset({
    'ET': (['time', 'lat', 'lon'], monthly_data)  # shape: (12, nlat, nlon)
}, coords={
    'time': times,
    'lat': lats,
    'lon': lons
})
```

### 2. 配置评估

在配置文件中正常设置评估项:

```json
{
  "evaluation_items": ["Evapotranspiration"],
  "metrics": {
    "Evapotranspiration": ["bias", "rmse", "corr", "nPhaseScore"]
  },
  "scores": {
    "Evapotranspiration": "Collier2018"
  }
}
```

系统会自动:
1. 检测参考数据的气候态类型
2. 计算模拟数据的气候平均值
3. 过滤不支持的指标 (如 `nPhaseScore`)
4. 进行评估

### 3. 运行评估

```bash
python openbench/openbench.py nml/your-config.json
```

### 4. 查看结果

#### 控制台输出

系统会显示气候态检测信息:

```
================================================================================
CLIMATOLOGY EVALUATION MODE DETECTED
================================================================================
Detected climatology type: annual
Calculated annual climatology from simulation data
Skipped metrics for climatology: {'nPhaseScore', 'nIavScore'}
================================================================================
```

或

```
================================================================================
CLIMATOLOGY EVALUATION MODE DETECTED
================================================================================
Detected climatology type: monthly
Calculated monthly climatology from simulation data
Skipped metrics for climatology: {'nPhaseScore', 'nIavScore'}
================================================================================
```

#### 输出文件

评估结果保存在正常的输出目录:

```
output/
├── metrics/
│   ├── Evapotranspiration_ref_SOURCE_sim_MODEL_bias.nc
│   ├── Evapotranspiration_ref_SOURCE_sim_MODEL_rmse.nc
│   └── ...
└── scores/
    ├── Evapotranspiration_ref_SOURCE_sim_MODEL_nBiasScore.nc
    └── ...
```

## 技术细节

### 时间戳标准化

为确保一致性,系统使用标准时间戳:

- **年气候态**: `2000-06-15` (年中)
- **月气候态**: `2000-01-15, 2000-02-15, ..., 2000-12-15` (每月中旬)

### 数据对齐

1. 参考数据的时间被设置为标准气候态时间
2. 模拟数据通过聚合转换为相同的气候态格式
3. 两者的时间坐标对齐后进行评估

### 内存优化

- 模拟数据聚合使用 `skipna=True`,自动处理缺失值
- 气候态数据通常比完整时间序列小得多,内存效率高

## 示例用例

### 用例1: 评估年平均蒸散发

**参考数据**: MODIS ET气候平均值 (2000-2020年平均)
**模拟数据**: 模型输出 (2010-2020年逐月)

评估指标: bias, RMSE, correlation, spatial score

### 用例2: 评估月平均温度

**参考数据**: CRU TS月气候值 (12个月)
**模拟数据**: 模型输出 (1980-2020年逐月)

评估指标: bias, RMSE, correlation, monthly pattern

### 用例3: 与卫星数据对比

**参考数据**: 卫星观测的多年月平均LAI
**模拟数据**: 陆面模式模拟的LAI时间序列

评估指标: 空间偏差, 月际变化的相关性

## 注意事项

### 数据要求

1. **参考数据时间点数**: 必须为1(年气候态)或12(月气候态)
2. **模拟数据**: 需要足够的时间长度来计算有代表性的气候平均值
3. **空间范围**: 参考和模拟数据的空间范围应一致或可重叠

### 指标限制

- 依赖时间变化的指标(相位、年际变率)会自动跳过
- 如果所有指标都不支持,评估将不会执行

### 最佳实践

1. **参考数据**: 使用至少10-30年的观测数据计算的气候态
2. **模拟数据**: 使用至少10年的模拟结果计算气候平均值
3. **变量选择**: 优先使用适合气候态分析的变量(如ET, 温度, 降水等)
4. **时间对齐**: 确保模拟期间包含足够的完整年份

## 实现模块

气候态评估功能由以下模块实现:

- `openbench/data/Mod_Climatology.py` - 核心气候态处理器
- `openbench/core/evaluation/Mod_Evaluation.py` - 评估引擎集成

所有气候态处理都是自动的,无需额外配置。
