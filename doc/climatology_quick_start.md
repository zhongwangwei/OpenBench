# 气候态评估快速入门

## 一句话总结

在配置文件中设置 `data_groupby: "climatology"` 即可启用气候态评估。

## 最简配置示例

### 参考数据配置 (月平均气候态)

```json
{
  "general": {
    "root_dir": "./dataset/reference/climatology",
    "data_type": "grid",
    "data_groupby": "climatology",
    "syear": 1980,
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

### 模拟数据配置 (时间序列)

```json
{
  "general": {
    "root_dir": "./dataset/simulation/model_output",
    "data_type": "grid",
    "data_groupby": "year",
    "syear": 2000,
    "eyear": 2020,
    "tim_res": "Month",
    "grid_res": 0.5
  },
  "Evapotranspiration": {
    "varname": "ET",
    "varunit": "mm month-1",
    "prefix": "ET_",
    "suffix": ""
  }
}
```

## 关键要点

1. **参数位置**: `data_groupby: "climatology"` 放在 `general` 部分
2. **大小写**: 不区分大小写 (`"climatology"`, `"Climatology"`, `"CLIMATOLOGY"` 都可以)
3. **年份**: `syear` 和 `eyear` 保持为整数,表示数据覆盖的时间范围
4. **数据格式**:
   - 月平均气候态: 12个时间点
   - 年平均气候态: 1个时间点

## 运行评估

```bash
python openbench/openbench.py nml/nml-json/main-climatology.json
```

## 查看结果

评估完成后,系统会在日志中显示:

```
INFO - 气候态评估模式已激活 (data_groupby='climatology')
INFO - CLIMATOLOGY EVALUATION MODE DETECTED
```

结果保存在 `output/` 目录下,与普通评估相同的结构。

## 常见用例

### 用例 1: 评估模式对气候态的模拟能力

- **参考**: 观测数据的多年月平均值 (12个月)
- **模拟**: 模式输出的时间序列 (多年月度数据)
- **目的**: 评估模式是否能正确模拟季节循环

### 用例 2: 评估模式对长期平均状态的模拟

- **参考**: 观测数据的多年年平均值 (1个值)
- **模拟**: 模式输出的时间序列 (多年数据)
- **目的**: 评估模式对长期平均状态的偏差

## 下一步

详细文档请参考: [climatology_evaluation_guide.md](./climatology_evaluation_guide.md)
