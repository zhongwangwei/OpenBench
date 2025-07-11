# OpenBench Unified API

OpenBench 2.0现在提供了简洁统一的API接口，让您能够轻松配置和运行完整的陆面模式评估。

## 🚀 快速开始

### 基本用法

```python
from openbench import OpenBench

# 创建OpenBench实例
ob = OpenBench.from_config('config.yaml')

# 运行完整评估
results = ob.run()

# 获取结果
print(results)
```

### 完整示例

```python
from openbench import OpenBench
import xarray as xr

# 方法1: 从配置文件创建
ob = OpenBench.from_config('nml/main-Debug.json')
print(f"配置已加载: {len(ob.get_config())} 个部分")

# 方法2: 从字典创建  
config_dict = {
    'engines': {'modular': {'type': 'modular'}},
    'metrics': ['bias', 'RMSE', 'correlation']
}
ob = OpenBench.from_dict(config_dict)

# 方法3: 基础构造器
ob = OpenBench()

# 加载数据并运行评估
simulation = xr.open_dataset('simulation.nc')
reference = xr.open_dataset('reference.nc')

results = ob.run(
    simulation_data=simulation,
    reference_data=reference,
    metrics=['bias', 'RMSE', 'correlation'],
    engine_type='modular'
)

# 保存结果
ob.save_results('results.json', format_type='json')
```

## 📖 API参考

### 创建实例

```python
# 从配置文件
ob = OpenBench.from_config('config.yaml')  # 支持 JSON, YAML, NML

# 从配置字典
ob = OpenBench.from_dict(config_dict)

# 基础实例
ob = OpenBench()
```

### 运行评估

```python
# 完整评估
results = ob.run()

# 指定数据和参数
results = ob.run(
    simulation_data='path/to/sim.nc',
    reference_data='path/to/ref.nc', 
    metrics=['bias', 'RMSE', 'correlation'],
    engine_type='modular'
)

# 使用xarray数据集
results = ob.run(
    simulation_data=simulation_dataset,
    reference_data=reference_dataset,
    metrics=['bias', 'RMSE'],
    engine_type='grid'
)
```

### 配置管理

```python
# 获取配置
config = ob.get_config()

# 更新配置
ob.update_config({'new_setting': 'value'})

# 验证配置
validation = ob.validate_config()
print(f"配置有效: {validation['valid']}")
```

### 系统信息

```python
# 获取可用组件
engines = ob.get_available_engines()     # ['modular', 'grid', 'station']
metrics = ob.get_available_metrics()     # ['bias', 'RMSE', 'correlation', 'NSE']

# 系统状态
info = ob.get_system_info()
print(f"版本: {info['version']}")
print(f"模块可用: {info['modules_available']}")
```

### API服务

```python
# 创建API服务
api_service = ob.create_api_service(
    host='127.0.0.1',
    port=8080,
    max_concurrent_tasks=10
)

# 启动API服务
ob.start_api_service(host='0.0.0.0', port=8000)
```

### 上下文管理器

```python
# 自动资源清理
with OpenBench.from_config('config.yaml') as ob:
    results = ob.run()
    # 资源自动清理
```

## 🔧 便利函数

```python
from openbench import run_evaluation, create_openbench

# 快速评估
results = run_evaluation('config.yaml')

# 灵活创建实例
ob = create_openbench()                    # 空实例
ob = create_openbench('config.yaml')       # 从文件
ob = create_openbench(config_dict)         # 从字典
```

## 📊 结果处理

```python
# 获取结果
results = ob.get_results()

# 保存不同格式
ob.save_results('results.json', format_type='json')
ob.save_results('results.csv', format_type='csv')
ob.save_results('results.nc', format_type='netcdf')

# 结果结构
{
    'evaluation_type': 'modular',
    'engine_type': 'modular', 
    'results': {
        'metrics': {
            'bias': {'value': 0.1234, 'info': {...}},
            'RMSE': {'value': 5.6789, 'info': {...}},
            'correlation': {'value': 0.8901, 'info': {...}}
        }
    },
    'metadata': {
        'metrics': ['bias', 'RMSE', 'correlation'],
        'config': {...},
        'engine': 'modular'
    }
}
```

## 🌐 REST API端点

当启动API服务后，可用以下REST端点：

```
GET  /                     - 服务根路径
GET  /status               - 系统状态
POST /evaluate             - 创建评估任务
GET  /evaluate/{task_id}   - 查询任务状态  
GET  /evaluate/{task_id}/download - 下载结果
GET  /metrics              - 可用指标列表
GET  /engines              - 评估引擎类型
POST /config/validate      - 配置验证
```

## 🔄 集成示例

### 与现有代码集成

```python
# 替换现有的OpenBench调用
# 旧方式:
# python openbench.py config.json

# 新方式:
from openbench import OpenBench
ob = OpenBench.from_config('config.json')
results = ob.run()
```

### 批量处理

```python
import glob
from openbench import OpenBench

# 批量处理多个配置
for config_file in glob.glob('configs/*.json'):
    with OpenBench.from_config(config_file) as ob:
        results = ob.run()
        ob.save_results(f'results/{config_file}.json')
```

### 自定义评估

```python
# 自定义评估引擎和指标
ob = OpenBench()
ob.update_config({
    'engines': {
        'custom': {
            'type': 'modular',
            'spatial_aggregation': 'mean',
            'temporal_aggregation': 'monthly'
        }
    }
})

results = ob.run(engine_type='custom')
```

## 🔍 故障排除

### 常见问题

1. **模块导入错误**
   ```python
   # 确保在正确目录
   import sys
   sys.path.insert(0, 'script')
   from openbench import OpenBench
   ```

2. **配置文件未找到**
   ```python
   # 使用绝对路径
   ob = OpenBench.from_config('/full/path/to/config.json')
   ```

3. **依赖缺失**
   ```bash
   # 安装可选依赖
   pip install fastapi uvicorn xarray pandas numpy
   ```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

ob = OpenBench.from_config('config.yaml')
results = ob.run()  # 将显示详细日志
```

## 🎯 性能优化

```python
# 启用缓存
ob = OpenBench()
ob.config.update({'use_cache': True})

# 并行处理
ob.config.update({'parallel_processing': True, 'n_workers': 8})

# 清理缓存
ob.clear_cache()
```

## 🔒 安全配置

```python
# API服务安全设置
ob.create_api_service(
    enable_auth=True,
    api_keys=['your-secure-api-key'],
    cors_origins=['https://your-domain.com'],
    rate_limit=60
)
```

---

**OpenBench 2.0 - 统一API让陆面模式评估更简单！**

完整文档: [CLAUDE.md](CLAUDE.md)  
技术细节: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)