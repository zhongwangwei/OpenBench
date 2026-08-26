# OpenBench 深度代码审查与修复验收报告

> 审查日期：2026-08-26
> 阶段：**阶段二修复与性能优化（已完成）**
> 原始审查冻结快照：分支 `fix/doc2-real-gui-report-audit-main`，提交 `3671375203e59457bda5bef8d263e7c72b5fb285`
> 最终闭环验收快照：分支 `fix/time-mask-singleton-contracts`，提交 `1a3bd64f2e15ef66fcdc0151e4ebab2742a7e976`
> 性能优化最终快照：提交 `011a26e`（xESMF 权重复用、MFM 共享计算、unified mask 单次写入）
> 审查环境：macOS 26.5.2 / arm64，Python 3.12.12，16 逻辑核，64 GiB 内存

## 0. 审查口径与快照说明

- 原始发现和行号以 `3671375` 为准；最终 `[DONE]` 验收结论以已提交快照 `1a3bd64` 为准。
- 审查开始时仓库位于另一分支/提交（`e16e0ec`），且已有大量未提交改动；审查期间另一个本地 Codex 进程继续修改代码、切换分支并生成提交 `3671375`。这些源码修改和提交**不是本次阶段一审查所作**。
- 修复验收重新运行全量测试、Ruff、全部变更测试和独立最小反例；随后按确认顺序逐项实施性能优化，每项独立测试和提交。
- 本报告中的“证据”分为：直接代码证据、可运行最小复现、解析/高精度对照和合成微基准。未在真实 HPC、Windows、NFS 或生产 SSH 环境验证的结论均明确标注。

---

## 1. 执行摘要

### 1.1 总体健康度

**总体状态：PASS（原报告 22 条发现均已通过代码、反例和自动化验收）。**

当前测试规模和静态检查表现良好，基础 RMSE、ubRMSE、NSE、KGE、相关系数的抽样数值对照也通过。最后两条残余已闭合：S-004 在关闭 unified mask 时保留精确 timestamp intersection；S-011 对无法推导有限 bounds 的单侧 singleton conservative 输入明确 fail fast。未计入 22 条发现的三个性能候选——xESMF 权重持久化复用、MFM 共享计算和 unified mask 单次写入——也已全部完成并通过回归、全量测试和 Ruff。

### 1.2 发现统计

| 严重级别 | 数量 | 含义 |
|---|---:|---|
| P0 阻塞 | 0 | 未发现普遍阻止运行或确定性破坏所有结果的问题 |
| P1 高 | 9 | 可静默改变科学结果、时间/空间支持或复用陈旧结果 |
| P2 中 | 13 | 边界正确性、缓存契约、跨平台或可量化性能问题 |
| P3 低 | 0 | 风格问题未计入缺陷；最终 Ruff 基线已通过 |
| **合计** | **22** | 科学正确性 11、缓存/配置正确性 7、性能/并发 4 |

### 1.3 原始最高风险结论

1. **CF 非公历时间会丢样本**：`360_day` 的 2 月 28/29/30 日被压成同一 Gregorian 日期，后续去重静默删除数据。
2. **默认 OpenBench conservative backend 的网格与球面权重均存在可复现偏差**：常用 `0.1°` 全球目标网格少一行/列；部分纬向重叠使用整源格面积修正而非实际重叠面积。
3. **时间对齐模式与配置语义不一致**：`intersection` 没有形成跨所有模型的共同时间支持；`strict` 在缺测时间被填成 NaN 后不再报错。
4. **缓存可能复用陈旧科学结果**：算法源指纹漏掉重网格、单位、时间工具等关键模块；station fulllist 指向的外部数据文件没有进入输入签名。

### 1.4 环境与测试基线

| 时点 | 结果 |
|---|---|
| 开始审查时（原工作树） | `1920 passed, 5 skipped, 114 warnings in 135.13s`；114 条均为全 NaN slice 警告 |
| 开始审查时 Ruff | `ruff check src tests` 有 2 个 `I001`；`ruff format --check` 有 15 个文件待格式化 |
| 冻结快照 `3671375` | `1925 passed, 5 skipped in 131.40s` |
| 最终 Ruff | `ruff check src tests`：通过；`ruff format --check src tests`：`376 files already formatted` |
| 子域验证 | 科学核心相关 19 个测试通过；运行时/缓存相关 378 个测试通过 |
| 中间修复验收 `2e0c4df` | 定向测试 `332 passed in 17.08s`；全量 `1946 passed, 5 skipped in 136.96s` |
| 中间 uncertainty 验收 `c72046f` | 定向测试 `12 passed in 1.66s`；全量 `1986 passed, 5 skipped in 132.99s` |
| 最终主线验收 `3de10ef` | 全量 `1952 passed, 5 skipped in 226.52s`；测试前后 tree `b78aa2c` 未变化 |
| 最终闭环验收 `1a3bd64` | S-004/S-011 定向测试 `8 passed in 10.94s`；全量 `1954 passed, 5 skipped in 240.91s`；测试前后内容 fingerprint 一致 |
| xESMF 权重复用 `c25d5d9` | xESMF/cache 相关定向测试通过；全量 `1962 passed, 5 skipped in 233.10s`；缓存 key 覆盖 source/target 坐标、CF bounds、mask、method、periodic 与 xESMF/ESMF 版本 |
| MFM 共享计算 `413e180` | MFM/evaluation 定向测试 `57 passed`；全量 `1969 passed, 5 skipped in 139.74s`；四项联合 synthetic median `0.1546 s → 0.0794 s`（`-48.6%`） |
| unified mask 单次写入 `011a26e` | 定向 `240 passed in 13.66s`；全量 `1975 passed, 5 skipped in 200.98s`；7 次中位写入 `4 → 1`，spatial mask 路径 `0.0801 s → 0.0380 s`（`-52.5%`） |
| 验收 Ruff | `ruff check src tests`：通过；`ruff format --check src tests`：`378 files already formatted` |

### 1.5 数据流认知

从 README 和 runner 调用链确认的主流程为：

1. `config/loader.py` 读取 YAML、include/default/env 等配置；`resolver.py`/`adapter.py` 解析参考数据、模拟数据和 legacy namelist。
2. `runner/task_planning.py` 建立 `variable × simulation × reference` 任务并计算缓存哈希。
3. `runner/orchestration.py` 依变量串行调度预处理；grid/station 内部再使用 joblib、xarray/Dask。
4. `data/_processing_*` 完成坐标规范化、时间完整性、单位转换、重网格、站点抽取与输出物化。
5. `runner/masking.py` 累积统一掩码；`core/evaluation.py` 对齐 sim/ref 后计算 metrics/scores。
6. `core/comparison*`、`core/statistics/*`、`visualization/` 生成比较、统计、图表及 HTML/PDF 报告。
7. `.openbench_cache.json`、任务哈希和重网格权重缓存共同支持增量执行与断点恢复。

### 1.6 已通过的数值抽检

使用同一 pairwise finite mask，以 NumPy `float64` 参考实现对 5 点序列（含 sim/ref 错位 NaN）核验。OpenBench 与参考值的最大绝对误差为 `2.22e-16`：

| 指标 | OpenBench | NumPy 参考 | 绝对误差 |
|---|---:|---:|---:|
| RMSE | 0.707106781187 | 0.707106781187 | 0 |
| ubRMSE | 0.707106781187 | 0.707106781187 | 0 |
| NSE | 0.865671641791 | 0.865671641791 | 0 |
| correlation | 0.998373737819 | 0.998373737819 | 1.11e-16 |
| KGE | 0.639574974051 | 0.639574974051 | 2.22e-16 |

单样本相关、常量观测 NSE/KGE 均返回 NaN，退化行为合理。公式对照依据包括 [Nash–Sutcliffe 原始论文](https://doi.org/10.1016/0022-1694(70)90255-6)、[Gupta 等 2009 KGE 论文](https://doi.org/10.1016/j.jhydrol.2009.08.003) 和 [OpenBench 方法论文](https://doi.org/10.5194/gmd-18-6517-2025)。这只是对核心公式的抽检，不代表 `metrics.py` 中所有水文指标都已用真实数据验证。

### 1.7 与旧报告反馈的口径对齐

- 用户反馈中的 `S-004` / `B-004` / `B-003` 指向旧报告；本报告的同编号分别是全局时间交集、内置 conservative 权重缓存版本化、缓存键可逆性，不是 xESMF periodic、ref/sim 坐标取样或 HDF5 文件锁。
- 冻结快照中已有对 station 哨兵值、累积量重采样、cgroup/BLAS 资源预算、日期变更线、kappa 标签、单时刻维度以及全 NaN 分组警告等的修复和回归测试；本报告不重复将这些已修复症状计为发现。
- 相关但不同的剩余问题需分开：S-003 是“部分纬向交叠子区间”的球面积分偏差；S-006 是不完整样本被重标为 12 月气候态；S-009 是小类别分位数裁剪；S-011 是单**空间**格点，均不是已修复项的重复报告。
- xESMF 官方 API 明确说明 conservative regridding 会强制 `periodic=False`，因此不接受旧报告中“conservative 应启用 `periodic=True`”的建议，见 [xESMF User API](https://xesmf.readthedocs.io/en/stable/user_api.html)。
- 本报告未引用 `4044c69`，也未把 float32 权重单独定性为缺陷。

### 1.8 修复验收摘要（`1a3bd64`）

**验收结论：PASS。22 条发现全部 `[DONE]`。** 全量测试、Ruff、独立最小反例和代码审查均通过。

| 最后闭合项 | 验收证据 | 结论 |
|---|---|---|
| S-004 | `runner/masking.py:105-124` 仅在 `apply_spatial_mask=True` 时按 finite support 裁时次。独立反例中 `apply_spatial_mask=False` 的 2 个相同 timestamp 均被保留；开启 spatial mask 时仍正确删除无有限支持时次。 | `[DONE]`，timestamp intersection 与 cumulative NaN mask 语义已分离。 |
| S-011 | `conservative.py:267-276` 对 identical 1×1 返回单位权重，对非同点或单侧 singleton 明确报错；`regrid/utils.py:130-131` 不再构造 `[-inf, inf]`。双向 1×N 反例均得到预期 `ValueError`。 | `[DONE]`，不再制造无有限 bounds 的权重。 |

P-001 按原报告的“保留最终 dense API、先去掉 broadcast 临时矩阵”口径标记为 `[DONE]`：3600→3600 构造附加峰值从 `395.6 MiB` 降至 `198.92 MiB`，权重本体为 `98.88 MiB`。最终 dense O(N×M) 仍是可继续优化的性能债，但不再视为本条验收阻塞。

---

## 2. 发现清单

### 2.1 P1：高优先级

#### S-001 [DONE] — P1｜科学正确性 / 时间 / CF calendar

- **文件:行号**：`src/openbench/data/time_utils.py:21-40,81-112`；`src/openbench/data/_processing_time_integrity.py:55-58,108-111`
- **问题与影响**：原生 `cftime_to_nptime` 无法表示非 Gregorian 日期时，fallback 把非法日夹到 Gregorian 月末。`360_day` 日尺度数据中的 2 月 28/29/30 日会全部变成 2 月 28 日；后续完整性处理保留第一个重复时间，静默删除另外两个样本。多年日尺度模型会形成系统性时间错位，进一步污染月平均、相位和模型间对齐。
- **证据**：

  ```text
  输入：cftime.Datetime360Day(2001, 2, 1..30)
  输出步数=30，唯一时间=28，2001-02-28 出现 3 次
  ```

  CF 明确定义 `360_day` 为 12 个 30 日月，不能把其日期标签当作 Gregorian 日期无损表示，见 [CF Conventions Calendar](https://cfconventions.org/cf-conventions/cf-conventions.html#calendar)。
- **建议修复**：保留 `CFTimeIndex` 到时间聚合/对齐边界，或使用显式、单调且一一映射的 calendar conversion policy。不能无损转换时应 fail fast，禁止 clamp 后去重。增加 `360_day` 闰月、跨年、日/月重采样和缓存复跑测试。
- **置信度**：高。

#### S-002 [DONE] — P1｜科学正确性 / Regrid / 空间边界

- **文件:行号**：`src/openbench/data/regrid/utils.py:57-76`；调用入口 `src/openbench/data/_processing_grid_regrid.py:74-97`
- **问题与影响**：`create_lat_lon_coords()` 用浮点 `np.remainder(...) > 0` 判断端点是否整除。`0.1`、`0.2`、`0.05` 等常见分辨率的二进制误差会把整除误判为有余数，OpenBench conservative backend 的目标网格少北侧一行、东侧一列。相同配置在 xESMF/CDO 与内置 backend 可得到不同空间形状。
- **证据**：全球 cell-center 网格最小/最大边界复现：

  | 分辨率 | 实际 lat / 预期 | 实际 lon / 预期 | 实际末坐标 |
  |---:|---:|---:|---|
  | 0.2° | 899 / 900 | 1799 / 1800 | 89.7 / 179.7 |
  | 0.1° | 1799 / 1800 | 3599 / 3600 | 89.85 / 179.85 |
  | 0.05° | 3599 / 3600 | 7199 / 7200 | 89.925 / 179.925 |

- **建议修复**：用整数格点数构造坐标（`n = round(span / resolution) + 1` 后 `start + arange(n)*resolution`），并用 `isclose` 验证最后一点；与 `_processing_grid_regrid.create_target_grid()` 统一为一个既有实现。增加全球/区域及升降序网格测试。
- **置信度**：高。

#### S-003 [DONE] — P1｜科学正确性 / Conservative regrid / 球面面积

- **文件:行号**：`src/openbench/data/regrid/methods/conservative.py:154-170,515-570`；`src/openbench/data/regrid/utils.py:130-160`
- **问题与影响**：先按纬度角的线性重叠长度计算权重，再乘“整个源纬带”的平均球面修正。目标格只覆盖源格一部分时，正确权重应使用该**重叠子区间**的球面面积积分，而不是整源格平均 `cos(lat)`。高纬、粗网格或不对齐网格会产生系统性保守重网格偏差。
- **解析证据**：

  ```text
  源 lat 中心 [-60, 0, 60]，值 [0, 1, 2]
  目标 lat 中心 [-30, 30]
  OpenBench 公开 regrid accessor: [0.61681394, 1.38318606]
  精确球面面积均值: [0.57735027, 1.42264973]
  每格绝对误差: 0.03946367
  ```

  精确面积与 `sin(phi_upper)-sin(phi_lower)` 成正比；保守映射权重应基于 source/destination 的 fractional area overlap，见 [Jones 1999](https://doi.org/10.1175/1520-0493(1999)127%3C2204:FASOCR%3E2.0.CO;2)。
- **建议修复**：在构造纬向 overlap 时直接对每个实际交叠上下界积分 `sin(phi_hi)-sin(phi_lo)`，不要事后用整源格面积缩放。用常量场、解析分段场、极区和广度不等间距网格验证守恒与均值精度。
- **缓存影响**：修复会改变权重，必须同时版本化磁盘权重缓存（见 B-004）；旧权重应明确失效。
- **置信度**：高。

#### S-004 [DONE] — P1｜科学正确性 / 时间对齐 / Unified mask

- **文件:行号**：`src/openbench/config/runtime_info.py:885-927`；`src/openbench/runner/masking.py:80-111`；`src/openbench/core/evaluation.py:220-238`
- **问题与影响**：配置注释把 `intersection` 定义为 ref、所有 sim 和配置范围的共同交集，但实现按 pair 计算，并明确复用第一对任务的年份。掩码遇到不等时间轴时只在重叠区累积 invalid mask，然后把非重叠 ref 时间填成 `False`；最终每个 sim 又各自 `xr.align(join="inner")`。因此不同模型会在不同时间样本上得到指标，破坏跨模型可比性。
- **证据**：

  ```text
  ref: Jan, Feb；Sim A: Jan；Sim B: Jan, Feb；mode=intersection
  shared_ref_steps=2
  pair_A_steps=1
  pair_B_steps=2
  ```

- **建议修复**：任务规划后先计算该 variable/ref 下所有 sim 的**精确 timestamp 全局交集**，再一次性裁剪 ref 和所有 sim；`per_pair` 才保留逐对交集。统一掩码必须在同一时间支持上累积。增加三模型、同长度不同时间值、空交集和任务顺序置换测试。
- **置信度**：高。

#### S-005 [DONE] — P1｜科学正确性 / `strict` 时间对齐

- **文件:行号**：`src/openbench/config/runtime_info.py:904-915`；`src/openbench/data/_processing_time_integrity.py:75-90`；`src/openbench/core/evaluation.py:220-232`
- **问题与影响**：`strict` 对配置年份覆盖不足只记录 warning；时间完整性处理先把缺失月份/日期 reindex 成 NaN。进入 evaluation 时 sim/ref 时间坐标已经相同，`_align_grid_times()` 不再抛错，metrics 的 pairwise mask 又跳过缺测值。结果是 strict 模式对实际缺时次并不严格。
- **证据**：12 个月参考数据、模拟缺 6 月：

  ```text
  raw_steps sim/ref = 11/12
  after_integrity_steps = 12/12
  timestamps_equal = True
  sim_missing_count = 1
  strict_raised = False
  ```

- **建议修复**：在补齐缺时次之前保存/验证原始时间覆盖；strict 应对缺失或额外 timestamp、重复时间及覆盖不足直接报错。非 strict 模式仍可 reindex+NaN。
- **置信度**：高。

#### S-006 [DONE] — P1｜科学正确性 / Climatology

- **文件:行号**：`src/openbench/data/climatology.py:210-235`
- **问题与影响**：reference 只要 `time_size == 12` 就被当作 Jan–Dec 月气候态并直接重标时间，没有验证原始月份集合、频率或顺序。12 个日样本、12 个不连续样本或重复月份会被伪装成完整年循环，污染 annual/monthly climatology、nPhaseScore 和 nSeasonalityScore。
- **证据**：

  ```text
  输入：2001-01-01..2001-01-12，原始月份集合 [1]
  输出月份：[1,2,...,12]
  输出值仍为 [1,2,...,12]
  ```

- **建议修复**：只有时间月份恰好覆盖 `{1..12}` 且每月一个代表值，或配置显式声明 monthly climatology 时才走 12 点快路径；否则按实际时间 groupby，并对缺月报错。
- **置信度**：高。

#### S-007 [DONE] — P1｜科学正确性 / 缺失时间坐标

- **文件:行号**：`src/openbench/data/_processing_time_core.py:159-183`；当前行为由 `tests/test_processing_time_missing.py:7-21` 固化
- **问题与影响**：普通非 climatology 数据没有 `time` 坐标时，代码把整个静态 2-D 场广播到配置期内的每个时间步。一个损坏或选错变量的文件可被转换为“全年恒定但完整”的时间序列，产生非常可信的偏差、相关和评分结果，而不是暴露输入错误。
- **证据**：2×2 无时间场、2000 年日尺度：

  ```text
  input_shape=(2,2)
  output_shape=(366,2,2)
  first_slice == last_slice == input
  ```

- **建议修复**：除显式 static/climatology 数据类型外，缺少 time 坐标应报可读错误。若兼容旧静态数据，必须用明确配置字段选择广播，而非自动猜测。
- **置信度**：高。

#### B-001 [DONE] — P1｜缓存正确性 / 算法指纹不完整

- **文件:行号**：`src/openbench/runner/hashing.py:28-93,160-173,427-430`
- **问题与影响**：`algorithm_source_fingerprint()` 只哈希固定模块列表；`openbench.data.regrid` 只会哈希包的 `__init__.py`，不会递归哈希实现。实际影响预处理/评估缓存输出的 `time_utils`、`unit`、`regrid.utils`、`regrid.methods.conservative`、station matcher/scanner 等均未列入。同一包版本下修正这些算法时，task hash 可保持不变并复用旧结果。
- **证据**：上述关键模块对 `module in ALGORITHM_SOURCE_MODULES` 均为 `False`。`openbench_version` 不能覆盖 editable checkout、同版本补丁或未升版本的部署。landcover/climatezone groupby 位于 post-evaluation 阶段，不纳入本条 task-cache 主证据。
- **建议修复**：最小方案是补齐所有直接影响预处理/评估输出的模块并加入显式 cache schema version；更稳妥但仍无需新依赖的方案是对受控 package 文件清单计算内容哈希。新增测试应证明修改每类算法 salt 会改变 task hash。
- **缓存影响**：修复后相关旧 task cache 应一次性失效，这是必要而非兼容性回归。
- **置信度**：高。

#### B-002 [DONE] — P1｜缓存正确性 / Station 输入

- **文件:行号**：`src/openbench/runner/hashing.py:214-276`；真实路径解析在 `src/openbench/config/runtime_info.py:387-428`
- **问题与影响**：`input_file_signature()` 哈希 fulllist 文件本身以及 `root_dir` 下匹配的 NetCDF，却不解析 station CSV 中 `sim_dir`/`ref_dir` 指向的数据文件。站点数据位于 fulllist 外部目录时，数据内容改变但 task hash 不变，会静默复用陈旧指标和报告。
- **证据**：临时 catalog 的 `stations.csv` 指向 sibling `external/station.nc`；把 station 文件从 `first` 改成 `second-content` 后：

  ```text
  signature_equal=True
  signed_files=['stations.csv']
  listed_station_signed=False
  ```

- **建议修复**：复用 runtime 的相对路径/env 展开规则解析 fulllist，收集实际 `sim_dir`/`ref_dir` 文件并加入签名；对不存在、重复、大小写扩展名和 Windows 路径加测试。
- **缓存影响**：需要提升 input-signature schema；旧 station task cache 应失效，grid cache 不应无关失效。
- **置信度**：高。

### 2.2 P2：中优先级

#### S-008 [DONE] — P2｜科学正确性 / 年循环评分

- **文件:行号**：`src/openbench/core/scores.py:111-136,238-257`
- **问题与影响**：`nPhaseScore` 和 `nSeasonalityScore` 只要求“任意月份有值”，不要求完整 12 个月；Jan–Feb 两个月完全一致即可得到 1.0，容易把不完整窗口误报为完美季节性。OpenBench 方法论文把 `nstep` 定义为完整年循环（monthly 为 12），见 [OpenBench scoring definition](https://gmd.copernicus.org/articles/18/6517/2025/)。
- **证据**：Jan/Feb 值 `[1,2]` 的相同 sim/ref：`nPhaseScore=1.0`，`nSeasonalityScore=1.0`。
- **建议修复**：要求每个空间单元具备 12 个共同有效月份；不足时返回 NaN/明确跳过。日尺度相位应相应验证完整年循环或明确可接受覆盖率。
- **置信度**：高。

#### S-009 [DONE] — P2｜科学正确性 / 分组统计

- **文件:行号**：`src/openbench/core/landcover_groupby.py:79-95`；`src/openbench/core/climatezone_groupby.py:79-95`
- **问题与影响**：每个类别先按 5%/95% quantile 删除两端，再算统计。类别只有两个有效格点时，0 和 100 的 quantile 为 5 和 95，两个真实值都被删成 NaN；小面积 landcover/climate zone 会错误显示 N/A。
- **证据**：`[[0,100]]` 的中位数原为 50；clip 后 `[NaN,NaN]`，finite count=0。
- **建议修复**：中位数汇总无需预裁剪；若保留 outlier 策略，应设置最小样本数并同时输出 `n_valid`。两个模块应复用同一既有 helper，避免规则漂移。
- **置信度**：高。

#### S-010 [DONE] — P2｜时间解析 / 小数偏移

- **文件:行号**：`src/openbench/data/time_utils.py:272-326`
- **问题与影响**：自定义 `<month|year|season> since ...` 解码对 offset 直接 `int(off)`，小数部分被静默截断。`0.5 years/months/seasons since 2000-01-01` 全部变为起始日期，引入隐蔽时间错位。
- **证据**：三个单位的 `0.5` 偏移实际输出均为 `2000-01-01T00:00:00`。
- **建议修复**：这些非标准单位若只支持整数偏移，应验证 `off == round(off)` 并报错；不要静默截断。若要支持小数，必须定义明确 calendar-aware 语义。
- **置信度**：高。

#### S-011 [DONE] — P2｜Regrid / 单格点边界

- **文件:行号**：`src/openbench/data/regrid/utils.py:102-126,150-160,300-301`
- **问题与影响**：公开 regrid accessor 对 1×1 网格会先在 `format_lat()` 对空 `diff()` 求最大值时崩溃。若绕过 accessor 进入底层权重路径，单坐标又会被转换为 `[-inf, inf]` 区间；source 和 target 的 overlap 为 `inf`，归一化执行 `inf/inf` 而得到 NaN。1×1 区域切片或异常最小网格既不能稳定返回常量，也没有一致错误契约。
- **证据**：单格值 7 从 `(lat=0, lon=0)` 重网格到相同 1×1 目标，公开 accessor 抛出 `ValueError: numpy.nanmax raises on a.size==0 and axis=None`；隔离调用底层 overlap/normalize 路径则产生 `inf/inf -> NaN`。
- **建议修复**：单点必须要求/推导有限 bounds；source/target 相同单点可直接单位权重。无法推导物理 cell bounds 时应报错而非用无限区间。
- **置信度**：高。

#### B-003 [DONE] — P2｜缓存键 / 可逆性

- **文件:行号**：`src/openbench/runner/cache.py:201-203`
- **问题与影响**：三元组用固定 `__` 拼接，不可逆：`('A__B','C','D')` 和 `('A','B__C','D')` 都得到 `A__B__C__D`。task hash 本身仍包含独立字段，通常会防止错误 cache hit，但同一映射槽会互相覆盖，造成持续 false miss、错误 invalidate 和断点状态丢失。
- **证据**：上述两个合法 path-safe 名称组合的 key 完全相同。
- **建议修复**：用稳定 JSON tuple 的 digest 或长度前缀编码。若需要读取旧 `.openbench_cache.json`，可在一个兼容窗口同时查询旧 key，新写只用新 key；随后提升 cache schema。
- **置信度**：高。

#### B-004 [DONE] — P2｜Regrid 权重磁盘缓存 / 版本化

- **文件:行号**：`src/openbench/data/regrid/methods/conservative.py:257-297,440-489`
- **问题与影响**：内存/磁盘权重 key 只含 source/target 坐标 dtype、shape 和内容 digest；磁盘文件名不含权重算法或 schema 版本。升级 overlap/normalize 算法后，即使 task hash 触发重算，进程仍可能从 `OPENBENCH_REGRID_WEIGHT_CACHE_DIR` 读出旧权重。
- **证据**：`_weights_disk_cache_path()` 只对 coordinate key 做 SHA256；`.npz` 也只保存 `weights`，没有版本元数据。
- **建议修复**：把 `REGRID_WEIGHT_SCHEMA_VERSION` 加入 key 和 `.npz` metadata，加载时验证版本、shape、finite/column sum。首次修复 S-003 时必须让旧权重失效。
- **缓存影响**：旧权重应明确失效；坐标相同且算法版本相同的缓存命中保持兼容。
- **置信度**：高。

#### B-005 [DONE] — P2｜缓存 / NetCDF 输出契约

- **文件:行号**：`src/openbench/config/schema.py:50-61`；`src/openbench/runner/hashing.py:419-478`
- **问题与影响**：`project.io.netcdf_compression` 和 compression level 不进入 task hash。用户从未压缩切到压缩后，已有任务可被跳过，磁盘文件仍保持旧物理编码；数值不变，但配置声明的输出契约没有实现。
- **证据**：只切换 compression 配置的两个 payload 得到相同 16 位 task hash。
- **建议修复**：把影响物化文件格式的 IO 字段加入对应输出阶段 hash；不要把纯调优、不会改变文件的 batch planner 字段混入科学结果 hash。
- **缓存影响**：只应使相关物化输出失效，不应无谓重算更早阶段。
- **置信度**：高。

#### B-006 [DONE] — P2｜缓存 / 数值栈版本

- **文件:行号**：`src/openbench/runner/hashing.py:176-202,427-451`
- **问题与影响**：backend signature 只在选择 xESMF/CDO/basic interpolation 时记录少量后端信息；内置路径没有 NumPy、pandas、xarray、cftime 等版本摘要。依赖升级改变时间解析、reduction、dtype promotion 或 backend 行为时，同 OpenBench 版本和源码仍可能命中旧结果。
- **证据**：task payload 的 `openbench` 只有版本/算法/source fingerprint；`openbench_conservative` 环境值只是布尔 `True`。
- **建议修复**：只加入会影响数值语义的核心依赖和实际选中 backend 的版本，避免把所有依赖都加入造成过度失效。为版本 salt 增加单元测试。
- **置信度**：中高；需要先定义项目接受的跨依赖版本缓存兼容政策。

#### B-007 [DONE] — P2｜跨平台 / 配置路径展开

- **文件:行号**：`src/openbench/config/loader.py:1003-1030`；`src/openbench/cli/run.py:243-260`
- **问题与影响**：`reference.overrides.<source>.root_dir/fulllist` 只校验字符串，不展开 `$ENV`/`~`；CLI 的集中展开也遗漏 reference overrides。HPC、macOS、Linux 和 Windows 上使用可移植环境变量配置时，resolver/adapter 可能访问字面路径，并连带使输入签名错误。
- **证据**：`$OB_REVIEW_ROOT/data` 经 override validator 后仍为原字符串；CLI `_expand_config_paths()` 只处理 `reference.data_root` 和 simulation。
- **建议修复**：在一个统一 path-normalization 边界展开所有路径字段，包括 reference override 和变量级 fulllist；CLI、GUI、Python API 共用，避免只在 CLI 补丁。
- **置信度**：高。

#### P-001 [DONE] — P2｜性能 / Regrid 权重内存峰值

- **文件:行号**：`src/openbench/data/regrid/utils.py:130-160`；`src/openbench/data/regrid/methods/conservative.py:282-288`
- **问题与影响**：`overlap()` 通过 broadcast 同时构造 `source × target` 的 `mins`、`maxs` 和 overlap dense 矩阵。经纬规则网格的非零重叠实际是窄带稀疏结构，首次构造却按 O(N×M) 内存。0.1° 全球 longitude 3600→3600 的单轴临时峰值接近 396 MiB；多进程/Dask 会放大。
- **基准证据**：见第 3 节；3600→3600 权重本体 98.9 MiB，构造附加峰值 395.6 MiB。
- **建议修复**：用双指针扫描有序区间，只生成实际相交 band；先保持最终 dense API 以缩小改动，验证后再考虑已有 optional `sparse` 路径。优化前后必须逐元素对照权重和守恒结果。
- **置信度**：高。

#### P-002 [DONE] — P2｜性能 / Metrics 重复掩码与线程峰值

- **文件:行号**：`src/openbench/core/evaluation.py:116-119,396-414`；`src/openbench/core/metrics.py:43-64`；`src/openbench/core/scores.py:27-42,200-215`
- **问题与影响**：evaluation 先对 sim/ref 做一次 pairwise finite mask，每个 metric/score 又各自 align 并重建同样的 mask；`Overall_Score` 内部 5 个分量再次各做验证。metric threading 并行这些大中间数组，速度略升但峰值内存大幅增加。
- **基准证据**：365×90×180 float64 合成数据，4 个 metrics：串行 0.3171 s / 412.1 MiB 附加峰值；4 线程 0.2081 s / 767.1 MiB，速度约 1.52×，内存约 1.86×。KGE 单项附加峰值 411.9 MiB。
- **建议修复**：保持公开 metric API 自验证，但 evaluation 内部使用一次已验证输入的私有计算路径，或删除外层重复 mask；对大数组按内存预算限制 metric workers。不要通过降 float 精度优化。
- **置信度**：高。

#### P-003 [DONE] — P2｜性能 / 缓存过度失效

- **文件:行号**：`src/openbench/runner/hashing.py:423-426`
- **问题与影响**：evaluation task hash 同时包含 `comparisons` 和 `statistics`。只改画图/比较/统计选项也会使预处理、重网格和指标任务 hash 改变，大数据运行发生无谓重算。
- **证据**：仅改变 comparison 配置，task hash 从 `e2e395296b4764ea` 变为 `09f56e497258b23a`。
- **建议修复**：按已有流程边界拆分阶段 hash：preprocess、evaluation、comparison/statistics、report 各验证自身输入；不要先引入通用 DAG/cache 抽象。
- **置信度**：高。

#### P-004 [DONE] — P2｜并发 / Dask × joblib 超额订阅

- **文件:行号**：`src/openbench/runner/local.py:360-383`；`src/openbench/runner/orchestration.py:332-373`；`src/openbench/data/_processing_grid_core.py:88-142,227-237`
- **问题与影响**：Dask client 在整个 runner 之前启动；evaluation task-level pool 会感知 Dask 并避免再开进程，但 preprocessing 没收到该预算，仍多处执行 joblib `Parallel(n_jobs=self.num_cores)`。Dask worker 与 loky worker 因而可同时存在并竞争内存/句柄；若 xarray/Dask compute 与 joblib preprocessing 重叠，还会竞争 CPU。
- **证据**：调用链静态可证；尚未在真实 distributed cluster 复现死锁。
- **建议修复**：最小方案是在 Dask active 时把 preprocessing 的 joblib worker 限制为 1，或延迟 Dask client 到真正需要 Dask distributed 的阶段。增加资源预算单元测试和一个本地 cluster smoke test。
- **置信度**：中；真实影响依数据规模、Dask 配置和平台而异。

---

## 3. 性能基准表（修复前与优化对照）

### 3.1 方法

- 原始缺陷基准来自冻结快照 `3671375`；三个性能候选的前后对照分别在对应优化提交前后运行。环境均为单机 macOS/arm64、Python 3.12.12。
- 计时使用 `time.perf_counter()`；原始表使用 `tracemalloc`，unified mask 对照使用 `psutil` 采样进程 RSS 增量。二者都不是生产数据的总内存需求。
- 数据为合成数组；未包含真实 NetCDF 网络存储、解压、Dask scheduler 或报告绘图时间。

### 3.2 Conservative weight 冷构造

关闭内存/磁盘权重缓存：

| source→target | 耗时 | 附加峰值 | 最终权重大小 |
|---:|---:|---:|---:|
| 180→360 | 0.00089 s | 2.00 MiB | 0.49 MiB |
| 720→1440 | 0.00462 s | 31.68 MiB | 7.91 MiB |
| 1800→1800 | 0.01371 s | 98.9 MiB | 24.7 MiB |
| 1800→3600 | 0.02633 s | 197.84 MiB | 49.44 MiB |
| 3600→3600 | 0.04994 s | 395.6 MiB | 98.9 MiB |

结论：耗时不高，但 O(N×M) 临时矩阵是明显内存热点；在多 worker 下比单线程耗时更危险。

### 3.3 Metrics / Scores

输入：`time=365, lat=90, lon=180`，约 591 万个 float64/数组，sim/ref 各含约 1% 独立 NaN。

| 计算 | 耗时 | 附加峰值 | 输出形状 |
|---|---:|---:|---:|
| RMSE | 0.0322 s | 191.8 MiB | 90×180 |
| ubRMSE | 0.0543 s | 225.6 MiB | 90×180 |
| KGE | 0.1712 s | 411.9 MiB | 90×180 |
| nRMSEScore | 0.0741 s | 226.0 MiB | 90×180 |
| Overall_Score | 1.0978 s | 247.9 MiB | 90×180 |

并行对照（RMSE、ubRMSE、KGE、NSE）：

| 模式 | 耗时 | 附加峰值 |
|---|---:|---:|
| 串行 | 0.3171 s | 412.1 MiB |
| 4 线程 | 0.2081 s | 767.1 MiB |

### 3.4 输入签名扫描

临时目录内 1 KiB `.nc` 文件，包含 stat 和 head/middle/tail sample hash：

| 文件数 | 耗时 | 峰值 |
|---:|---:|---:|
| 100 | 0.0240 s | 0.5 MiB |
| 1000 | 0.2226 s | 1.0 MiB |
| 5000 | 1.5168 s | 5.7 MiB |

结论：当前 signature 成本近似线性，单次尚可；本轮更高风险是签名**漏文件**而不是签名本身太慢，不建议为了速度进一步弱化内容指纹。

### 3.5 保留的性能候选项（未计入 22 条发现）

| 候选项 | 当前证据 | 本轮处理 |
|---|---|---|
| MFM 逐格 histogram/FFT | 原实现同时请求 `MFM/MFM_omega/MFM_varphi/MFM_eta` 时，组合 MFM 会再次执行已经单独请求的三个组件。固定随机种子合成 `time=120, lat=20, lon=30` float64 数据（含 pairwise NaN），7 次中位耗时 `0.1546 s`，`np.histogram=7200`、`np.fft.rfft=2400`。 | `[DONE]` 提交 `413e180` 不改 histogram/FFT 数学实现，只在联合请求时一次计算并复用组件；中位耗时 `0.0794 s`（`-48.6%`），调用数减半为 `3600/1200`，tracemalloc 峰值 `2.392 → 2.380 MiB`。边界、Dask/eager 与公共 API 严格数值等价测试通过。单独请求 MFM 的逐格算法保持不变。 |
| xESMF 权重持久化复用 | 提交 `c25d5d9` 新增内容寻址权重缓存，所有三条 xESMF 调用路径统一复用；key 覆盖 source/target 坐标、CF bounds、mask、method、periodic、schema 与 xESMF/ESMF/esmpy 版本，并复用现有跨平台文件锁和原子写。 | `[DONE]` 同网格第二次构造从磁盘加载权重；不同 bounds/mask/method/version 均产生 cache miss。全量 `1962 passed, 5 skipped`，Ruff 通过。环境未安装真实 xESMF/ESMF，故未给出真实权重生成耗时，仅以 mock 调用计数和官方“权重生成昂贵、可复用”的接口契约验收。 |
| unified mask 合并后单次写入 | 优化前每个 sim 都通过 staged NetCDF + `os.replace` 重写共享 ref。固定随机种子 `1 ref + 4 sims`、`24×60×120` float64，7 次交替次序中位：spatial mask 开启 `0.0801 s`、4 次写入、累计 `2.674 MiB`、65 个 writer 图任务；关闭时 `0.0322 s`、4 次写入、9 个任务。 | `[DONE]` 提交 `011a26e` 在 `runner/masking.py:47-211` 懒合并 sibling finite mask，并在 `preprocessing.py:414-441` 恢复 mixed grid/station 文件后只原子替换一次。开启 spatial mask 时 `0.0380 s`（`-52.5%`）、1 次/`0.669 MiB`/32 个任务；关闭时 `0.0161 s`（`-49.9%`）、1 次/3 个任务。RSS 增量分别 `0.547 → 0.891 MiB` 与 `1.828 → 1.828 MiB`，即用很小的 graph/handle 常驻换取 I/O 减少。76 组随机时空错位/NaN 数值等价对照及定向/全量测试通过；关闭 spatial mask 时现在只收缩 time，不再静默收缩 ref 的 lat/lon。 |

---

## 4. 风险与已知限制

1. **真实数据覆盖不足**：未获得多年度 360_day、极区高分辨率、真实 station fulllist、0.1° 全球多变量案例；当前证据以解析构造和合成数据为主。
2. **Dask/HPC 未实测**：未运行远程 Dask cluster、NFS/Lustre、scheduler crash/restart 或 Dask worker 内嵌 joblib 的压力测试；P-004 为静态调用链加资源模型推断。
3. **Windows 未实测**：CI 目标包含 Windows，但本次机器为 macOS；路径、文件替换、spawn、文件锁和 NetCDF/HDF5 句柄未做真实 Windows 运行。
4. **GUI/SSH 未交互实测**：只做代码走查和现有自动化测试；没有启动 GUI、建立真实 SSH、测试断线重连或凭据生命周期。
5. **统计假设需要领域数据**：Mann–Kendall、ANOVA、PLSR、FDR、three-cornered hat 的实现和现有测试已走查，未发现可直接证明的 P1/P2 公式错误；但 three-cornered hat 的误差独立性、ANOVA 分布/方差假设不能由代码自动保证，真实报告应显式呈现适用前提。
6. **指标覆盖不是穷尽式**：本轮用解析/NumPy 对照验证了核心 RMSE/ubRMSE/NSE/KGE/correlation 及退化输入；未逐一对照 `metrics.py` 中所有复合水文指标和 bootstrap 指标。
7. **报告图形未做像素验证**：`visualization/Fig_*.py` 和 HTML/PDF 仅代码审查；轴、单位、标签的视觉一致性未在所有组合下实测，按任务约束不列为高优先级。
8. **并发工作区事件**：审查中仓库被另一个本地进程切换分支并多次提交。原始发现已基于 `3671375` 冻结，最终缺陷闭环基于 `1a3bd64` 验收；性能优化从该提交继续，首项为 `c25d5d9`。
9. **xESMF 真实后端未实测**：当前环境没有 xESMF/ESMF，权重文件生成与命中通过接口一致的 mock 测试验证；真实 ESMF 运行时耗时、NFS 行为和跨版本文件兼容仍需在可选后端环境补测。
10. **unified mask 大模型数压力未实测**：单次写入路径为保持 Dask lazy graph 会在一次写盘前保持每个 sibling sim 的数据集句柄；当前 4-sim 基准稳定，但数百模型、低文件句柄上限、Windows/NFS 和 distributed Dask 仍需真实压力测试。

---

## 5. 剩余修复顺序

原报告 22 条发现和 §3.5 的三个性能候选均已闭合，无剩余修复顺序。

### 阶段二验证要求补充

- **时间类**：Gregorian/noleap/all_leap/360_day、闰年、日/月/年、同长度错 timestamp、空交集、缺月。
- **重网格类**：常量守恒、解析纬带积分、极区、regional edge、升/降序坐标、1×1、backend 形状一致性。
- **缓存类**：记录旧/新 hash 与 schema；明确哪些旧缓存应失效，验证同版本新缓存仍命中；不得用降低输入指纹强度换性能。
- **指标类**：使用 float64 高精度参考；NaN 必须 pairwise 一致；退化输入明确返回 NaN 或抛错。
- **并发类**：Linux/macOS/Windows 至少覆盖 spawn、异常传播、取消和 worker 预算；真实 Dask cluster 无法覆盖时明确保留验证缺口。

---

## 6. 阶段状态

**阶段二修复与性能优化验收通过。** 22 条发现与三个性能候选全部 `[DONE]`；xESMF 权重持久化复用（`c25d5d9`）、MFM 共享计算（`413e180`）和 unified mask 单次写入（`011a26e`）均已独立提交，并通过各自定向测试、全量 pytest 与 Ruff。
