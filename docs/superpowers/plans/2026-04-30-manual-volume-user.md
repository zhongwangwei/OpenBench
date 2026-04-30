# OpenBench 用户卷内容实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 撰写用户卷（卷 I）全部 10 章 + 5 真实附录的中文 LaTeX 内容；替换 Plan 1 留下的 hello-world 占位（`00-hello.tex` / `A-stub.tex`）；写作期间深度审查涉及代码并即时报告 bug 到 `docs/superpowers/reviews/2026-04-30-manual-bugs.md`。

**Architecture:**
- 每章一个 `docs/manual/user/chapters/NN-name.tex` 文件，被 `main_user.tex` `\include{}`；
- 每附录一个 `docs/manual/user/appendices/X-name.tex` 文件；
- 真实内容附录会 `\input{../_generated/...}`（Plan 2 已就绪），章节文本环绕生成内容讲述背景；
- 写作伴随的代码审查覆盖：CLI 9 命令、config schema 与 loader、registry manager、runner.local 主流程、visualization 主入口、report 模块。

**Tech Stack:** 已就绪（Plan 1+2）。本 plan 只产出 `.tex` 与 `.md`（bug log）。

**关联 spec:** `docs/superpowers/specs/2026-04-30-openbench-manual-design.md`（§4.1 用户卷大纲）

**前置依赖:**
- Plan 1（基础设施）已完成
- Plan 2（生成器）已完成；`make generated` 可用

**不在本计划范围:**
- 开发卷（Plan 4）/ 运维卷（Plan 5）的内容
- 把生成器扩展到错误消息半自动化（暂用人工编纂 + 引用 logging 调用点的方式）
- 解决基础设施 plan #4 锚点冲突（本卷只引入卷 I 的 5 个附录 A-E，独立编译时无冲突；合订模式仍是已知限制，待 Plan 5 完成后统一引入卷前缀方案修一次）

---

## 文件结构

新建：

| 路径 | 估计行数 | 责任 |
|---|---|---|
| `docs/manual/user/chapters/01-overview.tex` | 200 | 概述、特性、安装、版本验证 |
| `docs/manual/user/chapters/02-quickstart.tex` | 180 | 10 分钟教程：init → check → run |
| `docs/manual/user/chapters/03-config-structure.tex` | 350 | 4 大块详解、!include、_defaults |
| `docs/manual/user/chapters/04-variables-references.tex` | 400 | 56 变量分类 + 数据集映射 |
| `docs/manual/user/chapters/05-simulation-profiles.tex` | 300 | model profile + 自定义 + time alignment |
| `docs/manual/user/chapters/06-running.tex` | 350 | run/check 详解 + 缓存 + only_drawing |
| `docs/manual/user/chapters/07-results.tex` | 350 | 输出目录 + metrics/scores/comparison/report |
| `docs/manual/user/chapters/08-gui.tex` | 300 | 14 wizard page 流程 |
| `docs/manual/user/chapters/09-migration.tex` | 200 | migrate + JSON/NML 兼容 |
| `docs/manual/user/chapters/10-troubleshooting.tex` | 300 | FAQ + 错误诊断流程 |
| `docs/manual/user/appendices/A-config-reference.tex` | 80 | `\input` config_schema + 文字串联 |
| `docs/manual/user/appendices/B-cli-reference.tex` | 250 | 9 CLI 命令逐一描述 |
| `docs/manual/user/appendices/C-reference-datasets.tex` | 80 | `\input` reference_table + 引用说明 |
| `docs/manual/user/appendices/D-model-profiles.tex` | 60 | `\input` model_table + 用法说明 |
| `docs/manual/user/appendices/E-error-messages.tex` | 200 | 关键错误消息分类与处置 |

修改：

| 路径 | 改动 |
|---|---|
| `docs/manual/user/main_user.tex` | 把 `\include{chapters/00-hello}` 与 `\include{appendices/A-stub}` 替换为 10 + 5 真实文件 |

删除：

| 路径 | 原因 |
|---|---|
| `docs/manual/user/chapters/00-hello.tex` | 被 01-overview.tex 取代 |
| `docs/manual/user/appendices/A-stub.tex` | 被 A-config-reference.tex 取代 |

---

## 写作约定

每章遵循统一节奏：

1. **章导言段**（1-2 段）：本章要解决的问题，读者读完能做什么
2. **核心节**：按读者执行流程或概念分层组织
3. **示例**：用 `exampleBox`+`minted` 展示 YAML / shell / Python
4. **陷阱标注**：用 `warnBox` / `tipBox` 提示常见错误与最佳实践
5. **章尾"下一步"**：下一章预告，1-2 句

代码审查纪律：

- 每章撰写前**先读对应代码模块**（路径在每个 task 列出）
- 发现代码 bug 立即停笔，按 `docs/superpowers/reviews/2026-04-30-manual-bugs.md` 流程报告
- 不在错误代码上撰写"正确"文档

---

## Phase 0: 验证前置 + 删除占位

### Task 0: 健康检查 + 删除 stub

- [ ] **Step 1:** `make all` 当前能跑通

```bash
cd /Volumes/Data01/Openbench/docs/manual && make clean && make all 2>&1 | tail -3
```

预期：4 PDF 全部产出。

- [ ] **Step 2:** Plan 2 测试全 pass

```bash
cd /Volumes/Data01/Openbench && python -m pytest tests/manual/ -v 2>&1 | tail -3
```

预期：37 passed。

- [ ] **Step 3:** 删除 user 卷 stub（被本 plan 替换的）

不在本 task 删除——保留到对应章节写完再删（避免 main_user.tex include 不存在的文件导致编译失败）。

---

## Phase 1: 入门 3 章（1-3）

### Task 1: Chapter 1 — 概述与安装

**Files:** Create `docs/manual/user/chapters/01-overview.tex`

**代码审查范围:**
- `pyproject.toml`（依赖、extras、Python 版本）
- `README.md`（特性介绍）
- `src/openbench/__init__.py`（public API）
- `src/openbench/cli/main.py`（CLI 入口、版本命令）
- `src/openbench/cli/__init__.py`

**章节大纲:**
- §1.1 OpenBench 是什么（定位 + 用例 + 与同类工具区别）
- §1.2 核心特性（统一配置、66 数据集、3 model profile、CLI、GUI、远程、迁移）
- §1.3 系统要求（Python 3.10+、依赖列表、磁盘/内存建议）
- §1.4 安装方式（pip、uv、conda；core / [gui] / [remote] / [report] / [all]）
- §1.5 验证安装（`openbench version`、`openbench --help`）
- §1.6 下一步预告

**关键示例:**
- `pip install "openbench[all]"` 完整命令
- `openbench --help` 输出截图（textual）
- 常见安装错误的处置（PySide6 编译失败、conda 通道、HPC module）

**审查要点（可能 bug 点）:**
- 检查 `pyproject.toml` 中声明的版本与 README 是否一致
- 检查 `cli/main.py` 是否所有命令都注册了
- 检查 `__init__.py` 导出的公共 API 是否完整

- [ ] **Step 1:** 读代码

```bash
head -30 README.md
cat pyproject.toml | head -60
cat src/openbench/__init__.py
cat src/openbench/cli/main.py
```

- [ ] **Step 2:** 撰写 LaTeX，使用 `\chapter{概述与安装}` 起始

- [ ] **Step 3:** 临时编译验证（在 main_user.tex 加 `\include{chapters/01-overview}` 替换 `00-hello`）

```bash
cd docs/manual && make user
```

- [ ] **Step 4:** Commit

```bash
git add docs/manual/user/chapters/01-overview.tex
git commit -m "docs(user): write chapter 1 — overview and installation"
```

---

### Task 2: Chapter 2 — Quick Start

**Files:** Create `docs/manual/user/chapters/02-quickstart.tex`

**代码审查范围:**
- `src/openbench/cli/init_cmd.py`（init 命令逻辑）
- `src/openbench/cli/check.py`（check 命令）
- `src/openbench/cli/run.py`（run 命令）
- `openbench_full_options.yaml`（最小配置示例蓝本）

**章节大纲:**
- §2.1 工作流总览（一图：init → check → run → 看输出）
- §2.2 步骤 1：生成最小配置（`openbench init`）
  - 交互式选项；常见输入
- §2.3 步骤 2：检查配置（`openbench check config.yaml`）
  - 检查项：YAML 语法、必填字段、数据可用性
- §2.4 步骤 3：运行评估（`openbench run config.yaml`）
  - 阶段：preprocess → eval → comparison → vis → report
- §2.5 看输出目录（output/<run_name>/...）
- §2.6 下一步：去哪里学更多

**审查要点:**
- `init_cmd.py` 默认值是否合理；error path 是否输出有用信息
- `check.py` 是否覆盖关键失败场景
- `run.py` 是否对常见错误（缺数据、内存）有清晰诊断

- [ ] **Step 1-4** 同 Task 1 模板

---

### Task 3: Chapter 3 — 配置文件结构

**Files:** Create `docs/manual/user/chapters/03-config-structure.tex`

**代码审查范围:**
- `src/openbench/config/schema.py`（120 行，最关键）
- `src/openbench/config/loader.py`（361 行 — YAML 解析、_defaults、!include）
- `src/openbench/config/__init__.py`
- `openbench_full_options.yaml`（完整示例）

**章节大纲:**
- §3.1 顶层四块：project / evaluation / reference / simulation
- §3.2 必填 vs 可选（与 schema.py 字段对照）
- §3.3 spatial-temporal bounds 详解（years、lat/lon range、tim_res、grid_res）
- §3.4 reference 块（data_root、sources 映射）
- §3.5 simulation 块（多模型字典、profile vs 自定义）
- §3.6 高级：`_defaults` 块合并机制
- §3.7 高级：`!include` 路径拆分大配置
- §3.8 完整最小示例与逐行注释

**审查要点:**
- `loader.py` 对 `!include` 循环引用的防护
- `_defaults` 合并语义是否文档化（深合并 vs 浅合并）
- `strict_reference` 行为是否与文档一致

- [ ] **Step 1-4** 同 Task 1 模板

---

## Phase 2: 内容 4 章（4-7）

### Task 4: Chapter 4 — 选择变量与参考数据集

**Files:** Create `docs/manual/user/chapters/04-variables-references.tex`

**代码审查范围:**
- `src/openbench/data/registry/reference_catalog.yaml`（66 数据集源）
- `src/openbench/data/registry/manager.py`（registry 入口）
- `src/openbench/data/registry/scanner.py`（自动 scan 时序/空间）
- `src/openbench/data/registry/converter.py`
- `src/openbench/cli/data.py`（data list / download / status / path / optimize 子命令）

**章节大纲:**
- §4.1 评估变量分类（5 大类：Bio/Water/Energy/Meteorology/Urban）
- §4.2 grid vs station 数据形态差异
- §4.3 分辨率后缀 `_LowRes` / `_MidRes` 自动解析机制
- §4.4 浏览数据集：`openbench data list`
- §4.5 下载数据集：`openbench data download`
- §4.6 检查可用性：`openbench data status`
- §4.7 全 56 变量速查表（哪些变量有哪些 reference 选择）
- §4.8 引用 `\dataset{}` 命名约定

**审查要点:**
- registry/manager.py 解析 LowRes/MidRes 的逻辑是否健壮
- data.py 的 download 子命令是否处理网络错误
- 是否有未注册但 catalog 中存在的孤立条目

- [ ] **Step 1-4** 同 Task 1 模板

---

### Task 5: Chapter 5 — 配置 simulation 与模型 profile

**Files:** Create `docs/manual/user/chapters/05-simulation-profiles.tex`

**代码审查范围:**
- `src/openbench/data/registry/model_catalog.yaml`（21 个 profile）
- `src/openbench/data/registry/schema.py`（ModelProfile / VariableMapping）
- `src/openbench/cli/model.py`（model list / show / create）
- `src/openbench/config/adapter.py`（simulation entry → evaluator）
- `src/openbench/config/resolver.py`（reference 解析）

**章节大纲:**
- §5.1 内置 model profile 速览（CoLM2024 / CLM5 / ERA5-Land / ...）
- §5.2 用 profile 的最小配置
- §5.3 字段覆盖：在 simulation entry 中覆盖 profile 默认
- §5.4 自定义模型（不使用 profile）
- §5.5 多 simulation 对比配置
- §5.6 time alignment 三种策略：intersection / per_pair / strict
- §5.7 单位换算与变量映射机制

**审查要点:**
- adapter.py 是否正确处理 profile 缺失字段
- model.py create 子命令是否能产出有效 profile
- time alignment 三种策略在 evaluator 中的实现是否完整

- [ ] **Step 1-4** 同 Task 1 模板

---

### Task 6: Chapter 6 — 运行评估

**Files:** Create `docs/manual/user/chapters/06-running.tex`

**代码审查范围:**
- `src/openbench/cli/run.py`（127 行）
- `src/openbench/cli/check.py`（123 行）
- `src/openbench/runner/local.py`（906 行 — 主流程编排）
- `src/openbench/runner/cache.py`（146 行）
- `src/openbench/data/pipeline.py`（462 行）
- `src/openbench/util/logging_system.py`（581 行 — 日志层级）

**章节大纲:**
- §6.1 `openbench check` 检查项目录
- §6.2 `openbench run` 阶段流程
  - preprocess（regrid / climatology / cache）
  - evaluation（grid + station 双引擎）
  - comparison（多模型对比）
  - statistics（统计模块）
  - visualization（出图）
  - report（HTML/PDF 总结）
- §6.3 增量缓存机制（zarr）
- §6.4 `--force` 全量重算
- §6.5 `only_drawing` 模式（跳过计算只重出图）
- §6.6 `debug_mode` 与日志级别
- §6.7 中断与续跑

**审查要点（这是 906 行的大模块，重点审）:**
- runner/local.py 的阶段衔接是否清晰
- 缓存 key 的命中率与失效条件
- 中断后能否从合理 checkpoint 续跑
- 错误传播：单 sim 失败是否拖累其他 sim

- [ ] **Step 1-4** 同 Task 1 模板

---

### Task 7: Chapter 7 — 结果解读

**Files:** Create `docs/manual/user/chapters/07-results.tex`

**代码审查范围:**
- `src/openbench/util/output.py`（输出目录布局）
- `src/openbench/core/metrics.py`（25+ metrics）
- `src/openbench/core/scores.py`（normalized scores）
- `src/openbench/core/comparison.py`
- `src/openbench/core/statistics/Mod_Statistics.py`
- `src/openbench/util/report.py`（HTML/PDF 报告）
- `src/openbench/visualization/__init__.py` 与若干 `Fig_*.py`

**章节大纲:**
- §7.1 输出目录树（output/<run>/<phase>/<sim>/...）
- §7.2 metrics CSV 解读
- §7.3 scores 与 weighted score
- §7.4 comparison 输出（多模型并列、ranking）
- §7.5 statistics 模块输出（ANOVA、相关、KDE 等）
- §7.6 可视化产物（Fig_geo / Fig_portrait / Fig_radar 等）
- §7.7 HTML/PDF 总结报告（report.py）
- §7.8 二次分析：用 Python 直接读结果

**审查要点:**
- 输出目录结构在不同 phase / time_alignment 下是否稳定
- metrics 与 scores 的归一化方式是否文档化
- report.py 在 [report] extra 缺失时的回退行为

- [ ] **Step 1-4** 同 Task 1 模板

---

## Phase 3: 进阶 3 章（8-10）

### Task 8: Chapter 8 — GUI Wizard

**Files:** Create `docs/manual/user/chapters/08-gui.tex`

**代码审查范围:**
- `src/openbench/cli/gui.py`（启动入口）
- `src/openbench/gui/app.py`、`main_window.py`、`controller.py`
- `src/openbench/gui/pages/*.py`（14 个 page）
- `src/openbench/gui/dialogs/`、`widgets/`、`config_manager.py`

**章节大纲:**
- §8.1 启动 GUI（`openbench gui`、依赖检查）
- §8.2 14 个 wizard page 流程（按用户操作顺序，附截图占位）
  - General / Variables / Ref Data / Sim Data / Evaluation / Metrics / Scores / Comparison / Statistics / Runtime / Registry / Preview / Run Monitor / Options
- §8.3 配置加载、保存、模板
- §8.4 远程运行入口
- §8.5 进度监控
- §8.6 GUI 与 CLI 输出一致性

**说明:** 本章需大量截图。Plan 不要求生成截图（`docs/manual/common/figures/` 用占位 `\TODO{截图: ...}`），由后续 plan 或人工补图。

**审查要点:**
- 14 page 是否都对应有效 schema 字段
- config_manager.py 的双向同步（schema ↔ GUI 状态）是否完整
- progress_parser.py 的事件解析是否覆盖所有阶段

- [ ] **Step 1-4** 同 Task 1 模板

---

### Task 9: Chapter 9 — 从旧版迁移

**Files:** Create `docs/manual/user/chapters/09-migration.tex`

**代码审查范围:**
- `src/openbench/cli/migrate.py`
- `src/openbench/config/migration.py`（429 行）
- `src/openbench/config/legacy_processors.py`（909 行 — JSON/NML 旧格式）

**章节大纲:**
- §9.1 v2.0 vs v3.0 配置差异概览
- §9.2 `openbench migrate` 用法
- §9.3 JSON 配置迁移（v2.0 主要）
- §9.4 Fortran NML 配置迁移（v1.x 历史）
- §9.5 迁移后核对清单
- §9.6 数值一致性保证

**审查要点:**
- migration.py 是否覆盖所有 v2.0 字段
- 迁移生成的 YAML 是否能被 check 通过
- legacy_processors.py 的 deprecation 路径

- [ ] **Step 1-4** 同 Task 1 模板

---

### Task 10: Chapter 10 — 常见问题与排错

**Files:** Create `docs/manual/user/chapters/10-troubleshooting.tex`

**代码审查范围:**
- `src/openbench/util/exceptions.py`
- `src/openbench/util/logging_system.py`（错误消息源）
- `docs/superpowers/reviews/2026-04-30-manual-bugs.md`（写作期收集的）

**章节大纲（FAQ 形式）:**
- §10.1 安装类问题
- §10.2 配置类问题（YAML 语法、缺字段）
- §10.3 数据类问题（找不到数据、坐标系、单位）
- §10.4 运行类问题（OOM、超时、HDF5 lock）
- §10.5 时间对齐失败
- §10.6 站点匹配为空
- §10.7 远程 SSH 失败
- §10.8 GUI 不启动
- §10.9 一般诊断流程（debug_mode → log → reproduce）

**审查要点:**
- 错误消息是否清楚指引下一步
- 已知 bug 是否记录到 manual-bugs.md

- [ ] **Step 1-4** 同 Task 1 模板

---

## Phase 4: 5 个真实附录

### Task 11: Appendix A — 完整配置项参考

**Files:** Create `docs/manual/user/appendices/A-config-reference.tex`

**结构:**
```latex
\chapter{完整配置项参考}

本附录列出 openbench.yaml 全部字段。表格内容由 \cli{make generated} 从 \file{src/openbench/config/schema.py} 自动产出，确保与代码同步。

\input{../_generated/config_schema}
```

加少量过渡文字 + 必读注解。

- [ ] **Step 1:** 写文件
- [ ] **Step 2:** Compile + 验证 PDF 中表格正确显示
- [ ] **Step 3:** Commit

---

### Task 12: Appendix B — CLI 命令完整参考

**Files:** Create `docs/manual/user/appendices/B-cli-reference.tex`

**代码审查范围:**
- `src/openbench/cli/*.py` 全部 9 命令模块

**结构:**
- 9 个命令逐一描述：name / synopsis / 参数 / 选项 / 退出码 / 示例
- 每个用 `subsection*` + `tcolorbox` example

- [ ] **Step 1:** 读 9 个 cli 模块
- [ ] **Step 2:** 写 LaTeX
- [ ] **Step 3:** Compile + commit

---

### Task 13: Appendix C — Reference 数据集清单

**Files:** Create `docs/manual/user/appendices/C-reference-datasets.tex`

**结构:**
```latex
\chapter{Reference 数据集清单}

按物理类别分组列出 OpenBench 注册的全部 reference 数据集，由 \cli{make generated} 从 \file{reference_catalog.yaml} 自动产出。

\input{../_generated/reference_table}
```

- [ ] Step 1-3 同 Task 11

---

### Task 14: Appendix D — Model Profile 清单

**Files:** Create `docs/manual/user/appendices/D-model-profiles.tex`

**结构:** 类似 Task 13，引用 `model_table`。

---

### Task 15: Appendix E — 错误信息与日志参考

**Files:** Create `docs/manual/user/appendices/E-error-messages.tex`

**代码审查范围:**
- `grep -nE 'logger\.(error|warning|critical)' src/openbench/`

**结构:**
- 按 logger 名分组（cli / config / data / runner / ...）
- 每条：错误关键字、含义、可能原因、建议处置

**说明:** 这是半自动内容；本 plan 由人工编写，未来可能加 generator。

- [ ] Step 1-3

---

## Phase 5: 集成 + 验收

### Task 16: 更新 main_user.tex 引用列表

**Files:** Modify `docs/manual/user/main_user.tex`

替换：

```latex
\volpart{教程}
\include{chapters/01-overview}
\include{chapters/02-quickstart}
\include{chapters/03-config-structure}
\include{chapters/04-variables-references}
\include{chapters/05-simulation-profiles}
\include{chapters/06-running}
\include{chapters/07-results}
\include{chapters/08-gui}
\include{chapters/09-migration}
\include{chapters/10-troubleshooting}

\appendix
\volpart{参考}
\include{appendices/A-config-reference}
\include{appendices/B-cli-reference}
\include{appendices/C-reference-datasets}
\include{appendices/D-model-profiles}
\include{appendices/E-error-messages}
```

- [ ] **Step 1:** 改 main_user.tex
- [ ] **Step 2:** 删除 stub 文件

```bash
git rm docs/manual/user/chapters/00-hello.tex docs/manual/user/appendices/A-stub.tex
```

- [ ] **Step 3:** Commit

---

### Task 17: 端到端编译 + bug log review

- [ ] **Step 1:** `make clean && make all` 端到端

```bash
cd /Volumes/Data01/Openbench/docs/manual && make clean && make all 2>&1 | tail -10
```

预期：4 PDF 全部产出。用户卷预计 ~80-130 页（vs. 目前 14）。

- [ ] **Step 2:** 检查所有 .log 干净

```bash
for f in user/main_user.log developer/main_developer.log operations/main_operations.log manual.log; do
  echo "=== $f ==="
  grep -c -E "^\!|Undefined reference|Citation undefined" "$f"
done
```

预期：全部 0。

- [ ] **Step 3:** 复审 bug log

```bash
cat docs/superpowers/reviews/2026-04-30-manual-bugs.md
```

把写作期间发现的 bug 是否都已分类记录确认。

- [ ] **Step 4:** Commit（如有 bug log 更新）

---

## 自审清单

- [ ] **Spec 覆盖**：spec §4.1 用户卷 10 章 + 5 附录全部产出
- [ ] **占位词扫描**：`\TODO{}` 仅出现在第 8 章 GUI 截图位置；其他章节无 TODO/FIXME
- [ ] **类型一致性**：`main_user.tex` `\include{}` 路径与文件名一致；`appendices/A-config-reference.tex` 引用 `_generated/config_schema`
- [ ] **代码审查覆盖**：每章 task 列出的"代码审查范围"已实际阅读，发现 bug 已报告
- [ ] **不破坏 Plan 1+2**：Task 17 验证 4 PDF 仍 0 错误

---

## 完成后状态

- 用户卷 PDF 约 80-130 页，含真实可读内容
- 5 真实附录中 3 个由生成器驱动（A、C、D），与代码同步
- bug log 含写作期收集的代码缺陷及决议
- 三卷中第一卷"完工"，可作为发布候选先行交付
