# OpenBench 开发卷内容实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 撰写开发卷（卷 II）11 章 + 5 真实附录的中文 LaTeX 内容；替换 Plan 1 留下的 hello-world 占位（`00-hello.tex` / `A-stub.tex`）；写作期间深度审查各子系统并在 `docs/superpowers/reviews/2026-04-30-manual-bugs.md` 记录发现的 bug。

**Architecture:** 卷内章节按子系统组织（不按工作流），每章对应一个或一组源码包。每章先读对应代码模块、确认行为、再写文档。3 个真实附录由 Plan 2 的生成器驱动（包结构图 / public API / registry schema），2 个由 Plan 4 自己写（CONVENTIONS / 内部 interfaces 文字串接 + 已生成片段）。

**Tech Stack:** 已就绪。本 plan 只产出 `.tex` 与 `.md`。

**关联 spec:** `docs/superpowers/specs/2026-04-30-openbench-manual-design.md`（§4.2 开发卷大纲）

**前置依赖:**
- Plan 1 (基础设施) 完成
- Plan 2 (生成器) 完成
- Plan 3 (用户卷) 完成 —— 用户卷的章节会被开发卷引用作为"上下文"

**不在本计划范围:**
- 运维卷（Plan 5）
- 真正实现还没写完的 Public API（开发卷描述当前 API；缺失的能力不补，仅指出）
- 解决"appendix anchor 冲突"（待 Plan 5 完成后统一处理三卷前缀方案）

---

## 文件结构

新建：

| 路径 | 估计行数 | 责任 |
|---|---|---|
| `docs/manual/developer/chapters/01-architecture.tex` | 280 | 8 子包 + 端到端数据流图 |
| `docs/manual/developer/chapters/02-dev-environment.tex` | 200 | clone / uv / pre-commit / superpowers 流程 |
| `docs/manual/developer/chapters/03-config-subsystem.tex` | 380 | schema/loader/adapter/migration/resolver |
| `docs/manual/developer/chapters/04-data-subsystem.tex` | 350 | pipeline/cache/climatology/regrid/station |
| `docs/manual/developer/chapters/05-registry.tex` | 450 | 含两个 walkthrough（新 reference / 新 model） |
| `docs/manual/developer/chapters/06-core-engine.tex` | 450 | metrics/scores/eval/comparison/groupby/statistics |
| `docs/manual/developer/chapters/07-visualization.tex` | 280 | Fig 模块约定 + 添加新图 walkthrough |
| `docs/manual/developer/chapters/08-runner-cli.tex` | 300 | runner/local + cli click 注册 + 添加新命令 |
| `docs/manual/developer/chapters/09-gui-extension.tex` | 350 | 14 page + 添加新 page walkthrough |
| `docs/manual/developer/chapters/10-testing.tex` | 250 | tests/ 布局 + 各类测试 + CI |
| `docs/manual/developer/chapters/11-contributing.tex` | 220 | 分支 / PR / superpowers / commit conventions |
| `docs/manual/developer/appendices/A-package-graph.tex` | 80 | 含一个 TikZ 包依赖图（手画） |
| `docs/manual/developer/appendices/B-public-api.tex` | 150 | 主要 public API 函数签名（手写） |
| `docs/manual/developer/appendices/C-registry-schema.tex` | 50 | `\input` registry_schema |
| `docs/manual/developer/appendices/D-internal-interfaces.tex` | 50 | `\input` internal_interfaces + 文字串接 |
| `docs/manual/developer/appendices/E-conventions.tex` | 200 | 命名 / 类型 / 错误 / 日志 / 测试约定 |

修改：

| 路径 | 改动 |
|---|---|
| `docs/manual/developer/main_developer.tex` | 把 `\include{chapters/00-hello}` 与 `\include{appendices/A-stub}` 替换为 11 + 5 真实文件 |

删除：

| 路径 | 原因 |
|---|---|
| `docs/manual/developer/chapters/00-hello.tex` | 被章节替代 |
| `docs/manual/developer/appendices/A-stub.tex` | 被附录替代 |

---

## 写作约定（与 Plan 3 一致）

- 每章先读对应代码模块（task 列出范围），写作前确认行为
- 发现代码 bug → 立即停笔报告 → 等用户决议 → 续写
- 每章节奏：导言 + 核心节 + 示例 (`exampleBox`) + 陷阱 (`warnBox`) + 末尾"下一步"
- 代码引用：`\modname{openbench.core.metrics}` 而非裸 `openbench.core.metrics`
- 文件路径：`\file{src/openbench/...}` 而非裸路径
- 不用 ✓✗ 等 unicode 字符（minted 字体问题）

---

## Phase 0: 健康检查 + 删除占位

### Task 0: 验证前置 + 不删 stub

- [ ] **Step 1:** Plan 1+2+3 健康检查

```bash
cd /Volumes/Data01/Openbench/docs/manual && make clean && make all 2>&1 | tail -5
# 预期：4 PDF 全部产出
cd /Volumes/Data01/Openbench && python -m pytest tests/manual/ tests/test_cli_integration.py::test_init_output_is_loadable -q 2>&1 | tail -3
# 预期：38 passed
```

- [ ] **Step 2:** Stub 暂留，等对应章节写完再删（Task 16 一并清理）

---

## Phase 1: 入门 2 章（1-2）

### Task 1: Chapter 1 — 项目架构总览

**Files:** Create `docs/manual/developer/chapters/01-architecture.tex`

**代码审查范围:**
- `src/openbench/__init__.py`
- 8 个子包的 `__init__.py`：cli / config / core / data / gui / remote / runner / util / visualization
- `pyproject.toml` 的依赖与 entry points
- spec section 3 包结构图

**章节大纲:**
- §1.1 高层架构（前端 / 配置层 / 数据层 / 计算层 / 可视化层 / 运行层 / 远程层）
- §1.2 8 子包关系（含 TikZ 依赖图）
- §1.3 端到端数据流：YAML → schema dataclass → adapter → namelist → runner → engine → output
- §1.4 公开 vs 内部 API 边界（哪些 from openbench import 可用，哪些不该跨包用）
- §1.5 设计原则（无 Fortran 依赖、registry-driven、incremental cache、optional extras）
- §1.6 与 v2.0 的架构差异

**审查要点:**
- 子包之间的 import 依赖是否真的单向（用 grep 自检）
- 公开 API 的实际导出与 README 声称是否一致

- [ ] Step 1-4 (read / write / compile / commit)

---

### Task 2: Chapter 2 — 开发环境

**Files:** Create `docs/manual/developer/chapters/02-dev-environment.tex`

**代码审查范围:**
- `pyproject.toml`（dev / all extras）
- `.github/workflows/`（如有 CI 配置）
- `.ruff_cache` 与 ruff 配置
- `docs/superpowers/` 流程

**章节大纲:**
- §2.1 fork + clone
- §2.2 用 uv 安装可编辑模式：`uv pip install -e ".[all,dev]"`
- §2.3 ruff lint + format（pyproject.toml `[tool.ruff]` 解释）
- §2.4 pytest 跑测试
- §2.5 pre-commit hooks（如有）
- §2.6 superpowers 工作流：spec → plan → execute → review，对应 `docs/superpowers/{specs,plans,reviews}/`
- §2.7 build & 本地 PDF 文档：`docs/manual/`

- [ ] Step 1-4

---

## Phase 2: 子系统 5 章（3-7）

### Task 3: Chapter 3 — config 子系统

**Files:** Create `docs/manual/developer/chapters/03-config-subsystem.tex`

**代码审查范围:**
- `src/openbench/config/schema.py`（120 行）
- `src/openbench/config/loader.py`（361 行）
- `src/openbench/config/adapter.py`（996 行 — 最大；只读关键 build_runner_bindings）
- `src/openbench/config/migration.py`（429 行）
- `src/openbench/config/legacy_processors.py`（909 行 — 只读结构）
- `src/openbench/config/resolver.py`（314 行）
- `src/openbench/CONVENTIONS.md`（years / syear/eyear 双轨）

**章节大纲:**
- §3.1 schema dataclass 设计原则
- §3.2 loader 加载流程（与用户卷第 3 章对接，但视角转向"实现")
  - YAML → !include 展开 → 顶层 dispatch → 各子构造器
  - backward compat: options → project / sources / data_root 迁移
- §3.3 adapter：把新 schema 翻译成 evaluator 期待的 legacy 字典
  - 双轨命名：years（list）vs syear/eyear（runtime）
  - bindings 数据结构
- §3.4 migration：JSON / NML → YAML 转换
  - 旧 namelist 解析路径
  - field rename 表
- §3.5 resolver：reference 名解析
  - LowRes/MidRes 后缀逻辑
  - provenance 三档（HIGH/MEDIUM/LOW）
- §3.6 添加新配置字段的步骤（schema → loader 校验 → adapter 翻译 → docs）

**审查要点:**
- adapter.py 的内部数据结构是否还在演进
- legacy_processors.py 是否有死代码（最近重构）
- resolver 的 PROVENANCE_LOW 列表是否完整

- [ ] Step 1-4

---

### Task 4: Chapter 4 — data 子系统

**Files:** Create `docs/manual/developer/chapters/04-data-subsystem.tex`

**代码审查范围:**
- `src/openbench/data/pipeline.py`（462 行）
- `src/openbench/data/cache.py`
- `src/openbench/data/climatology.py`
- `src/openbench/data/compute.py`
- `src/openbench/data/coordinates.py`
- `src/openbench/data/file_processing.py`
- `src/openbench/data/processing.py`
- `src/openbench/data/regrid/regrid_cdo.py`、`regrid_wgs84.py`、`utils.py`
- `src/openbench/data/station_matcher.py`、`station_scanner.py`
- `src/openbench/data/time_utils.py`
- `src/openbench/data/unit.py`

**章节大纲:**
- §4.1 data 子系统总览（pipeline → cache → regrid → unit → climatology）
- §4.2 DataPipeline 阶段
- §4.3 三层缓存（memory + disk + zarr）
- §4.4 重网格化两条路径（CDO / WGS84）
- §4.5 单位转换框架（unit.py + compute 表达式）
- §4.6 站点处理：matcher + scanner
- §4.7 climatology 计算
- §4.8 file_processing 与 coordinates 的辅助职能

**审查要点:**
- pipeline.py 的阶段衔接是否清晰
- station_matcher 的近邻 vs 双线性策略选择
- regrid 模块对负向坐标系的处理

- [ ] Step 1-4

---

### Task 5: Chapter 5 — registry 子系统（含 2 个 walkthrough）

**Files:** Create `docs/manual/developer/chapters/05-registry.tex`

**代码审查范围:**
- `src/openbench/data/registry/manager.py`
- `src/openbench/data/registry/scanner.py`
- `src/openbench/data/registry/converter.py`
- `src/openbench/data/registry/schema.py`
- `src/openbench/data/registry/reference_catalog.yaml`（结构）
- `src/openbench/data/registry/model_catalog.yaml`（结构）
- `src/openbench/data/registry/reference_profiles.yaml`
- `src/openbench/data/registry/station_lists/`

**章节大纲:**
- §5.1 registry 总览（catalog YAML → dataclass → manager API）
- §5.2 RegistryManager 公开方法
- §5.3 reference_catalog.yaml schema（字段全列）
- §5.4 model_catalog.yaml schema
- §5.5 scanner：从 NC 自动提取元信息
- §5.6 converter：catalog 升级路径
- §5.7 user-level vs system-level catalog（可写位置）
- §5.8 **Walkthrough A**：从零添加一个新 reference 数据集（10 步）
- §5.9 **Walkthrough B**：从零添加一个新 model profile（8 步）

**审查要点:**
- 用户级 catalog 的合并优先级
- 注册时是否会校验数据集真的能被打开

- [ ] Step 1-4

---

### Task 6: Chapter 6 — core 评估引擎（含 metric/score/statistic walkthrough）

**Files:** Create `docs/manual/developer/chapters/06-core-engine.tex`

**代码审查范围:**
- `src/openbench/core/__init__.py`
- `src/openbench/core/metrics.py`（25+ 方法）
- `src/openbench/core/scores.py`（8 方法）
- `src/openbench/core/evaluation.py`、`evaluation_engine.py`
- `src/openbench/core/comparison.py`
- `src/openbench/core/landcover_groupby.py`、`climatezone_groupby.py`
- `src/openbench/core/statistics/Mod_Statistics.py`
- `src/openbench/core/statistics/stat_*.py`（19 个）

**章节大纲:**
- §6.1 core 总览
- §6.2 metrics 类的方法约定（s, o 参数 + numpy 向量化）
- §6.2.1 **Walkthrough**：添加一个新 metric（4 步：定义函数、加 docstring、注册、加测试）
- §6.3 scores 的归一化设计
- §6.3.1 **Walkthrough**：添加一个新 score
- §6.4 evaluation_engine 主循环
- §6.5 comparison 模块的多模型聚合
- §6.6 groupby（IGBP / PFT / Climate zone）
- §6.7 statistics 19 个模块（命名约定、I/O 约定）
- §6.7.1 **Walkthrough**：添加一个新 statistic 模块

**审查要点:**
- metrics 的 NaN / shape mismatch 处理是否一致
- Mod_Statistics.py 的 dispatch 表完整性
- evaluation_engine 与 evaluator 的命名是否还在过渡

- [ ] Step 1-4

---

### Task 7: Chapter 7 — visualization

**Files:** Create `docs/manual/developer/chapters/07-visualization.tex`

**代码审查范围:**
- `src/openbench/visualization/__init__.py`
- 17 个 `Fig_*.py` 模块（命名约定）
- `src/openbench/visualization/cmaps/`
- `src/openbench/visualization/Fig_geo_plot_index.py`（重用基础）
- `src/openbench/visualization/Mod_Only_Drawing.py`

**章节大纲:**
- §7.1 visualization 总览：一图一函数 + 单一公开入口
- §7.2 命名约定（Fig\_<KIND>.py，函数名等于文件名）
- §7.3 cmap 管理：复用 `cmaps/` 下的预定义 colormap
- §7.4 Fig\_geo\_plot\_index 作为通用 geo 渲染基础
- §7.5 配色与字体约定
- §7.6 输出路径约定（output/<run>/figures/<var>/<kind>/...）
- §7.7 **Walkthrough**：添加一个新 Fig 模块（5 步）
- §7.8 only_drawing 模式与 viz 模块的契约

**审查要点:**
- 17 个 Fig 模块是否风格一致
- cmaps 是否与 matplotlib 默认有冲突

- [ ] Step 1-4

---

## Phase 3: 集成 2 章（8-9）

### Task 8: Chapter 8 — runner & CLI

**Files:** Create `docs/manual/developer/chapters/08-runner-cli.tex`

**代码审查范围:**
- `src/openbench/runner/local.py`（906 行）
- `src/openbench/runner/cache.py`（146 行）
- `src/openbench/runner/remote.py`（651 行）
- `src/openbench/cli/main.py`（lazy group）
- `src/openbench/cli/_parsing.py`、`_reference_errors.py`
- 7 个命令模块（已读过）

**章节大纲:**
- §8.1 runner.local 主流程编排
- §8.2 EvaluationCache 增量缓存（task key、hash 算法）
- §8.3 unified mask、HDF5 lock 处理
- §8.4 runner.remote 现状（SSH 框架就绪、CLI 入口未实现）
- §8.5 CLI 设计：LazyGroup 模式
- §8.6 click 命令注册约定
- §8.7 错误传播：单 task 失败如何处理
- §8.8 **Walkthrough**：添加新 CLI 命令（5 步）

**审查要点:**
- runner.local 的并行实际作用范围（spec 提到变量级未启用）
- runner.remote 是否完全孤立
- CLI 命令组织是否还有未注册的死代码

- [ ] Step 1-4

---

### Task 9: Chapter 9 — GUI 扩展

**Files:** Create `docs/manual/developer/chapters/09-gui-extension.tex`

**代码审查范围:**
- `src/openbench/gui/__init__.py`
- `src/openbench/gui/app.py`、`main_window.py`、`controller.py`
- `src/openbench/gui/pages/base_page.py`（基类）
- `src/openbench/gui/pages/page_*.py`（14 个）选 1-2 个仔细读
- `src/openbench/gui/dialogs/`、`widgets/`
- `src/openbench/gui/config_manager.py`
- `src/openbench/gui/path_utils.py`、`progress_parser.py`
- `src/openbench/gui/runner.py`、`remote_runner.py`
- `src/openbench/gui/data_validator.py`、`validation.py`

**章节大纲:**
- §9.1 GUI 架构总览（PySide6 + signal/slot）
- §9.2 controller / page / widget / dialog 四层
- §9.3 base_page 抽象与 14 page 的共同接口
- §9.4 config_manager 的双向同步（schema ↔ GUI 状态）
- §9.5 progress_parser：CLI 输出 → GUI 进度
- §9.6 GUI 自身的 runner 与 remote_runner
- §9.7 validation 与 data_validator 的关系
- §9.8 **Walkthrough**：添加一个新 wizard page（7 步）
- §9.9 GUI 测试现状（headless 限制）

**审查要点:**
- 14 page 是否都继承同一基类
- config_manager 的同步逻辑死代码
- progress_parser 的事件解析覆盖

- [ ] Step 1-4

---

## Phase 4: 流程 2 章（10-11）

### Task 10: Chapter 10 — 测试

**Files:** Create `docs/manual/developer/chapters/10-testing.tex`

**代码审查范围:**
- `tests/` 整体布局
- `tests/test_smoke.py`
- `tests/test_cli_integration.py`、`tests/test_cli_stubs.py`
- `tests/test_config/`（fixture + 5 个 test）
- `tests/test_registry/`
- `tests/test_runner/`
- `tests/test_dead_code_cleanup.py`、`test_processing_registry_cache.py`
- `tests/manual/`（生成器测试）
- `pyproject.toml [tool.pytest.ini_options]`
- `TEST_PLAN.md`、`TEST_PLAN_CLI.md`、`TEST_PLAN_GUI.md`

**章节大纲:**
- §10.1 测试目录布局
- §10.2 测试金字塔：unit / integration / smoke
- §10.3 fixtures：tests/test_config/fixtures/
- §10.4 各模块测试覆盖：config / registry / runner / cli / manual generators
- §10.5 集成测试：用 CliRunner 跑 CLI 命令
- §10.6 跑测试：`pytest`、`pytest -k pattern`、`pytest --cov`
- §10.7 写新测试的约定
- §10.8 GUI 测试现状（headless 难度）
- §10.9 CI 期望（如有 GitHub Actions）

**审查要点:**
- TEST_PLAN*.md 是否仍代表当前状态（可能过期）
- tests/test_dead_code_cleanup.py 的覆盖范围

- [ ] Step 1-4

---

### Task 11: Chapter 11 — 提交贡献

**Files:** Create `docs/manual/developer/chapters/11-contributing.tex`

**代码审查范围:**
- `CHANGELOG.md`
- `.github/`（如有 issue/PR templates）
- 最近 50 个 commit 的风格（`git log --oneline -50`）
- `docs/superpowers/` 各模板

**章节大纲:**
- §11.1 贡献流程总览（fork → branch → PR）
- §11.2 分支模型（main / feature / topic）
- §11.3 commit message 约定（`docs(user):` / `feat(...):` / `fix(...):`）
- §11.4 PR 模板与 review checklist
- §11.5 superpowers 流程：spec → plan → execute → review
  - `docs/superpowers/specs/`、`plans/`、`reviews/` 各自模板
- §11.6 代码风格：CONVENTIONS.md 摘要 + ruff 配置
- §11.7 测试要求（新功能必须带测试）
- §11.8 文档更新（新功能改 manual 哪几章）
- §11.9 发布流程（version bump / changelog / tag）

**审查要点:**
- 现有 commit 风格是否一致（部分老 commit 可能不规范）
- CHANGELOG 是否在维护

- [ ] Step 1-4

---

## Phase 5: 5 个真实附录

### Task 12: Appendix A — 包结构与依赖图

**Files:** Create `docs/manual/developer/appendices/A-package-graph.tex`

**结构:**
- §A.1 一张 TikZ 包依赖图（手画，与 Chapter 1 相同但更详细）
- §A.2 公开 vs 内部模块分界

包依赖图节点（基于 chapter 1 信息）：
- 8 子包 + 它们的双向依赖
- 颜色区分公开/内部

- [ ] Step 1-3 (write / compile / commit)

---

### Task 13: Appendix B — Public Python API

**Files:** Create `docs/manual/developer/appendices/B-public-api.tex`

**代码审查范围:**
- `src/openbench/__init__.py`
- `src/openbench/config/__init__.py`
- `src/openbench/runner/__init__.py`
- `src/openbench/data/__init__.py`、`data/registry/__init__.py`
- `src/openbench/core/__init__.py`

**结构:**
- §B.1 顶层 import：`openbench.__version__`
- §B.2 config 公开 API：`load_config`、`OpenBenchConfig`、`ConfigError`、其他
- §B.3 runner：`run_evaluation`、其他
- §B.4 data: 主要 entry points
- §B.5 registry: `RegistryManager`、`get_registry`、其他
- §B.6 core: 主要类与函数
- §B.7 不公开的内部（绝对不要 import）

每个 API 给签名 + 1-2 行说明 + 简短示例。

- [ ] Step 1-3

---

### Task 14: Appendix C — Registry YAML schema

**Files:** Create `docs/manual/developer/appendices/C-registry-schema.tex`

**结构:** 类似用户卷附录 A，引用 generator：

```latex
\chapter{Registry YAML schema}

下表自动从 \file{src/openbench/data/registry/schema.py} 生成 ...

\input{../_generated/registry_schema}
```

加少量过渡文字。

- [ ] Step 1-3

---

### Task 15: Appendix D — 内部 interfaces

**Files:** Create `docs/manual/developer/appendices/D-internal-interfaces.tex`

**结构:** 引用 generator + 文字串接：

```latex
\chapter{内部 interfaces}

\openbench{} 中的 ABC / Protocol 类列表，由 \cli{make generated} 从源码自动扫描产出。
完整签名见对应模块；本附录提供索引。

\input{../_generated/internal_interfaces}
```

- [ ] Step 1-3

---

### Task 16: Appendix E — CONVENTIONS

**Files:** Create `docs/manual/developer/appendices/E-conventions.tex`

**代码审查范围:**
- `src/openbench/CONVENTIONS.md`（短文档，已读过）

**结构:**
扩充并整理 CONVENTIONS：
- §E.1 命名（年份双轨、变量名、模块名、配置 key、分辨率后缀）
- §E.2 类型注解约定
- §E.3 错误处理（ConfigError、ValueError 何时用何种）
- §E.4 日志级别约定（INFO/DEBUG/WARN/ERROR）
- §E.5 测试粒度（unit/integration/smoke）
- §E.6 docstring 与 type hint
- §E.7 commit message 约定（指向第 11 章）

- [ ] Step 1-3

---

## Phase 6: 集成 + 验收

### Task 17: 更新 main_developer.tex 与端到端编译

**Files:** Modify `docs/manual/developer/main_developer.tex`

替换 stub 引用为 11 章 + 5 附录。

- [ ] **Step 1:** 改 main_developer.tex
- [ ] **Step 2:** 删 stub

```bash
git rm docs/manual/developer/chapters/00-hello.tex docs/manual/developer/appendices/A-stub.tex
```

- [ ] **Step 3:** `make dev` 验收

```bash
cd /Volumes/Data01/Openbench/docs/manual && make dev 2>&1 | tail -5
# 预期：developer/main_developer.pdf ~120-150 页
```

- [ ] **Step 4:** `make all` 端到端验收

```bash
make clean && make all 2>&1 | tail -5
ls -la *.pdf user/*.pdf developer/*.pdf operations/*.pdf
```

预期：
- user: 134 页
- developer: 120-150 页（new）
- operations: 11 页（still stub）
- manual: 250-280 页

- [ ] **Step 5:** Bug log 复审

```bash
cat docs/superpowers/reviews/2026-04-30-manual-bugs.md
```

- [ ] **Step 6:** 整批 commit

---

## 自审清单

- [ ] **Spec 覆盖**：spec §4.2 列出的 11 章 + 5 附录全部产出
- [ ] **占位词扫描**：无 \TODO / TBD 留在文中
- [ ] **类型一致性**：`main_developer.tex` `\include{}` 路径与文件名一致；附录 C/D 引用 `_generated/`
- [ ] **代码审查覆盖**：每章 task 列出的代码模块已实际阅读
- [ ] **不破坏 Plan 1+2+3**：Task 17 验证 4 个 PDF（含 user 卷）仍 0 错误

---

## 完成后状态

- 开发卷 PDF 约 120-150 页
- 5 真实附录中 2 个由生成器驱动（C、D），3 个手写（A、B、E）
- 9 个 walkthrough（添加 reference / model / metric / score / statistic / Fig / CLI 命令 / wizard page）覆盖最常见扩展场景
- bug log 含开发卷写作期收集的代码缺陷
- 三卷中第二卷"完工"，可作为贡献者文档先行发布

下一份 plan：**`2026-04-30-manual-volume-operations.md`** —— 运维卷 8 章 + 5 真实附录。
