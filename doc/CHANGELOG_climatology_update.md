# Climatology Evaluation Enhancement

## 变更日期: 2025-01-XX

## 变更类型: Enhancement (功能增强)

## 概述

改进了气候态评估的配置方式,现在使用 `data_groupby: "climatology"` 来明确指定气候态评估模式,替代之前依赖数据维度的隐式检测方式。

## 主要变更

### 1. 新增配置参数

**推荐方式**: 在配置文件的 `general` 部分设置:

```json
{
  "general": {
    "data_groupby": "climatology",
    "syear": 2000,
    "eyear": 2020
  }
}
```

**优点**:
- ✅ 明确的意图表达
- ✅ 类型一致性 (`syear` 和 `eyear` 保持为整数)
- ✅ 语义清晰 (`data_groupby` 描述数据组织方式)
- ✅ 向后兼容 (自动检测机制仍然保留)

### 2. 代码变更

#### 2.1 Mod_Climatology.py

**新增方法**:
```python
def is_climatology_mode(self, data_groupby: str) -> bool:
    """检查配置是否指定了气候态模式"""
    return str(data_groupby).strip().lower() == 'climatology'
```

**更新函数签名**:
```python
def process_climatology_evaluation(
    ref_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    metrics: List[str],
    ref_data_groupby: str = None,
    sim_data_groupby: str = None
) -> Tuple[Optional[xr.Dataset], Optional[xr.Dataset], List[str]]:
```

#### 2.2 Mod_Evaluation.py

**更新评估调用**:
```python
# 获取 data_groupby 信息
ref_data_groupby = getattr(self, 'ref_data_groupby', None)
sim_data_groupby = getattr(self, 'sim_data_groupby', None)

# 调用气候态处理
o_clim, s_clim, supported_evaluations = process_climatology_evaluation(
    ref_ds, sim_ds, all_evaluations,
    ref_data_groupby=ref_data_groupby,
    sim_data_groupby=sim_data_groupby
)
```

### 3. 新增文档

#### 3.1 详细指南
- `docs/climatology_evaluation_guide.md`: 完整的气候态评估使用指南
- 包含配置方法、数据要求、支持的指标、常见问题等

#### 3.2 快速入门
- `docs/climatology_quick_start.md`: 一页式快速入门指南
- 最简配置示例和常见用例

#### 3.3 示例配置
- `nml/nml-json/Ref_variables_definition/climatology_example.json`:
  气候态评估配置示例

### 4. 新增测试

`tests/test_climatology_mode_detection.py`:
- 测试 `data_groupby='climatology'` 的正确检测
- 测试月平均气候态评估
- 测试年平均气候态评估
- 测试自动检测回退机制
- 测试非气候态数据的透传

**测试结果**: ✅ 所有测试通过

### 5. 文档更新

更新 `CLAUDE.md`:
- 添加 "Climatology Evaluation" 章节
- 更新 "Data Processing Flow" 以包含气候态检测步骤
- 提供配置示例和使用指南链接

## 向后兼容性

✅ **完全向后兼容**

现有的隐式检测机制仍然保留:
- 参考数据有1个时间点 → 自动识别为年平均气候态
- 参考数据有12个时间点 → 自动识别为月平均气候态
- 其他 → 作为普通时间序列处理

使用 `data_groupby: "climatology"` 可以**显式**启用气候态模式,优先级高于自动检测。

## 日志输出

启用气候态评估时,新增日志信息:

```
INFO - 气候态评估模式已激活 (data_groupby='climatology')
INFO - 12 time points found - treating as monthly climatology
INFO - Reference set to monthly climatology with times: 2000-01-15 to 2000-12-15
INFO - Calculated monthly climatology from simulation data
```

## 使用示例

### 之前的方式 (隐式检测,仍然支持)

```json
{
  "general": {
    "data_groupby": "single",
    "syear": 2000,
    "eyear": 2020
  }
}
```

系统根据数据的时间维度自动检测气候态类型。

### 新的推荐方式 (显式配置)

```json
{
  "general": {
    "data_groupby": "climatology",
    "syear": 2000,
    "eyear": 2020
  }
}
```

明确告诉系统这是气候态评估。

## 受益场景

1. **混合数据集**: 当参考数据可能有不同的时间维度时,显式配置避免歧义
2. **文档清晰**: 配置文件更加自说明,无需查看数据就知道评估类型
3. **错误检测**: 如果配置为气候态但数据格式不对,系统会明确报错
4. **未来扩展**: 为将来添加更多气候态类型预留了清晰的接口

## 测试覆盖

- ✅ 单元测试: `test_climatology_mode_detection.py`
- ✅ 配置示例验证
- ✅ 向后兼容性测试
- ✅ 自动检测回退测试

## 文件清单

### 修改的文件
1. `openbench/data/Mod_Climatology.py` - 添加 `is_climatology_mode()` 方法
2. `openbench/core/evaluation/Mod_Evaluation.py` - 更新调用方式
3. `CLAUDE.md` - 添加气候态评估说明

### 新增的文件
1. `docs/climatology_evaluation_guide.md` - 详细使用指南
2. `docs/climatology_quick_start.md` - 快速入门
3. `nml/nml-json/Ref_variables_definition/climatology_example.json` - 配置示例
4. `tests/test_climatology_mode_detection.py` - 单元测试
5. `CHANGELOG_climatology_update.md` - 本变更日志

## 迁移指南

**对于现有用户**: 无需任何修改,现有配置继续工作

**对于新用户**: 推荐使用 `data_groupby: "climatology"` 来明确指定气候态评估

**对于更新配置**: 可选择将现有隐式气候态评估改为显式配置,以提高可读性

## 下一步建议

1. 考虑添加季节气候态评估 (4个季节)
2. 添加自定义气候态时间窗口
3. 支持不同气候基准期 (如 1961-1990, 1981-2010)
4. 添加气候态异常值评估
