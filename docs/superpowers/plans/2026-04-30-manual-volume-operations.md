# OpenBench 运维卷内容实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 撰写运维卷（卷 III）8 章 + 5 真实附录的中文 LaTeX 内容；替换 Plan 1 留下的 hello-world 占位（`00-hello.tex` / `A-stub.tex`）；写作期间深度审查 SSH/parallel/cache/logging/report 模块并在 `docs/superpowers/reviews/2026-04-30-manual-bugs.md` 记录发现的 bug。

**Architecture:** 卷内章节按部署/运维场景组织（不按子系统），从"我应该把 OpenBench 装在哪里"到"已经在 HPC 上跑出问题怎么排查"。3 个真实附录由代码扫描或手写，2 个是文字串接 + 引用。

**Tech Stack:** 已就绪。本 plan 只产出 `.tex` 与 `.md`。

**关联 spec:** `docs/superpowers/specs/2026-04-30-openbench-manual-design.md`（§4.3 运维卷大纲）

**前置依赖:**
- Plan 1 (基础设施) ✅
- Plan 2 (生成器) ✅
- Plan 3 (用户卷) ✅
- Plan 4 (开发卷) ✅

**不在本计划范围:**
- 真实 HPC 性能基准实测（spec §2 决定的"建议性指导"路线，不跑实测）
- 修复 `--remote` CLI 入口（在 manual-bugs.md 已记录，留给单独 bug fix）
- 解决三卷合订 appendix anchor 冲突（Plan 5 完成后做最后一次集成清理）

---

## 文件结构

新建：

| 路径 | 估计行数 | 责任 |
|---|---|---|
| `docs/manual/operations/chapters/01-deployment.tex` | 200 | 4 种部署形态（笔记本/工作站/HPC/云）+ 决策表 |
| `docs/manual/operations/chapters/02-hpc-install.tex` | 250 | module load / conda / uv / 离线 / 共享只读 |
| `docs/manual/operations/chapters/03-remote-ssh.tex` | 380 | credentials/connections/sync/storage 全流程 |
| `docs/manual/operations/chapters/04-performance.tex` | 350 | num_cores / parallel / dask / IO/CPU/Memory 瓶颈 |
| `docs/manual/operations/chapters/05-cache.tex` | 250 | zarr 布局 / 增量 / cleanup / 多用户共享 |
| `docs/manual/operations/chapters/06-stations-at-scale.tex` | 280 | station_scanner / matcher / 千站点预算 / HDF5 lock |
| `docs/manual/operations/chapters/07-monitoring.tex` | 250 | logging_system / progress / report / Prometheus 思路 |
| `docs/manual/operations/chapters/08-recovery.tex` | 220 | Ctrl-C / lock 残留 / 断连 / 磁盘满 / nested deadlock |
| `docs/manual/operations/appendices/A-runtime-options.tex` | 180 | 全部 runtime/runner/parallel knob 速查 |
| `docs/manual/operations/appendices/B-remote-config.tex` | 150 | profile YAML schema + 密钥 / 端口 / 权限 |
| `docs/manual/operations/appendices/C-cache-layout.tex` | 100 | scratch/data/_minted/output 完整目录树 |
| `docs/manual/operations/appendices/D-performance-guidance.tex` | 150 | 建议性指导（不实测）+ 资源画像 |
| `docs/manual/operations/appendices/E-log-index.tex` | 200 | 半自动错误消息分类索引 |

修改：

| 路径 | 改动 |
|---|---|
| `docs/manual/operations/main_operations.tex` | 替换 hello-world stub 为 8 + 5 真实文件 |

删除：

| 路径 | 原因 |
|---|---|
| `docs/manual/operations/chapters/00-hello.tex` | 被 01-deployment.tex 替代 |
| `docs/manual/operations/appendices/A-stub.tex` | 被 A-runtime-options.tex 替代 |

---

## 写作约定（与 Plan 3+4 一致）

- 每章先读对应代码模块（task 列出范围），写作前确认行为
- 发现代码 bug → 立即停笔报告 → 用户决议 → 续写
- 节奏：导言 + 核心节 + 示例 (`exampleBox`) + 陷阱 (`warnBox`) + 末尾"下一步"
- 代码引用：`\modname{}` / `\file{}` / `\cli{}` / `\yamlkey{}`
- 不用 ✓✗ unicode；不用 ↔；中英混排避开 fandol 缺字
- 修写作 typo 立即（`\end{minted>}` 等）

---

## Phase 0: 健康检查

### Task 0: 验证前置

- [ ] **Step 1:** Plan 1+2+3+4 健康

```bash
cd /Volumes/Data01/Openbench/docs/manual && make clean && make all 2>&1 | tail -5
# 预期：4 PDF（user/dev/ops/manual）全部产出；ops 仍是 hello-world
```

- [ ] **Step 2:** 测试通过

```bash
cd /Volumes/Data01/Openbench && python -m pytest tests/manual/ tests/test_cli_integration.py::test_init_output_is_loadable -q 2>&1 | tail -3
# 预期：38 passed
```

---

## Phase 1: 部署 2 章（1-2）

### Task 1: Chapter 1 — 部署形态选择

**Files:** Create `docs/manual/operations/chapters/01-deployment.tex`

**代码审查范围:**
- `pyproject.toml` 依赖（最低硬件需求推断）
- `src/openbench/util/memory.py`（内存检测）
- `src/openbench/util/parallel.py:1-50`（并行能力概览）

**章节大纲:**
- §1.1 四种部署形态：笔记本 / 工作站 / HPC 节点 / 云 VM
- §1.2 各形态的资源画像表（CPU、内存、磁盘、网络、IO）
- §1.3 推荐配置（每形态：num_cores、cache_strategy、是否启用 zarr）
- §1.4 IO bound vs CPU bound 决策
- §1.5 数据是否本地（本地盘 vs NAS vs object storage）
- §1.6 选不对会怎么样（典型失败模式：HPC 上跑笔记本配置）
- §1.7 下一步

**审查要点:**
- memory.py 中是否对 HPC 共享内存有特殊处理
- pyproject 最低 Python 与依赖版本是否仍合理

- [ ] Step 1-4

---

### Task 2: Chapter 2 — HPC 安装与依赖管理

**Files:** Create `docs/manual/operations/chapters/02-hpc-install.tex`

**代码审查范围:**
- `pyproject.toml`（extras 与 dev）
- 用户卷第 1 章（避免重复）

**章节大纲:**
- §2.1 HPC 环境变体：module 系统 / 共享 conda / 容器 / 直接 pip
- §2.2 module load 路径：python module 选哪个
- §2.3 conda 路径：env 隔离 + 镜像源 + 大小控制
- §2.4 uv 路径：HPC 受限网络下用 uv 离线
- §2.5 离线安装：wheelhouse 流程
- §2.6 共享只读环境 + 用户私有 cache 模式（典型 HPC 部署）
- §2.7 Login 节点 vs 计算节点 Python 不一致问题
- §2.8 不要装 [gui]（无图形系统）
- §2.9 测试安装：openbench version / openbench data list
- §2.10 升级路径：pip install -U

**审查要点:**
- 共享只读环境的具体路径建议是否明确
- platformdirs 在 HPC 不同账号下的表现

- [ ] Step 1-4

---

## Phase 2: 远程执行 1 章（3）

### Task 3: Chapter 3 — 远程 SSH 执行

**Files:** Create `docs/manual/operations/chapters/03-remote-ssh.tex`

**代码审查范围:**
- `src/openbench/runner/remote.py`（651 行）
- `src/openbench/remote/ssh.py`
- `src/openbench/remote/credentials.py`
- `src/openbench/remote/connections.py`
- `src/openbench/remote/sync.py`
- `src/openbench/remote/storage.py`
- `src/openbench/cli/run.py:63-66`（CLI --remote stub 现状）
- `src/openbench/gui/remote_runner.py`

**章节大纲:**
- §3.1 远程执行现状（v3.0a1）
  - GUI 路径：可用
  - CLI 路径：未实现（已记录 bug）
  - 替代：手动 SSH + 远程跑 CLI
- §3.2 [remote] extra 安装：paramiko + cryptography
- §3.3 Connection profile（host/port/user/key/sudo 配置）
- §3.4 Credentials：fernet 加密本地存储
  - 凭据文件位置（platformdirs）
  - 密钥与口令二选一
- §3.5 Sync 引擎：哪些文件传，哪些不传
  - rsync-like 比较
  - .gitignore-style 排除
- §3.6 Storage 抽象：local vs remote
- §3.7 端到端：本地 GUI 提交 → 远程跑 → 拉回结果
- §3.8 中途断网：sync 是否能续？
- §3.9 长任务：远程 nohup / screen / tmux
- §3.10 替代方案 A：手动 ssh 然后远程跑 CLI（推荐稳妥）
- §3.11 替代方案 B：通过 GUI 触发但任务在远程节点
- §3.12 安全：密钥权限、known_hosts、堡垒机
- §3.13 已知限制（CLI --remote / 结果同步偶发失败 / 多账号）

**审查要点:**
- credentials.py 是否真用 fernet（spec 说的）
- ssh.py 的密码 vs 密钥流程
- sync.py 是否处理大文件 / 软链接
- storage.py 抽象的实际使用范围

- [ ] Step 1-4

---

## Phase 3: 性能 + 缓存 2 章（4-5）

### Task 4: Chapter 4 — 性能调优

**Files:** Create `docs/manual/operations/chapters/04-performance.tex`

**代码审查范围:**
- `src/openbench/util/parallel.py`（731 行）
- `src/openbench/util/memory.py`（326 行）
- `src/openbench/data/processing.py`（DatasetProcessing 内部并行）
- `src/openbench/runner/local.py:399-410`（num_cores 解析）
- 用户卷第 6 章（不要重复）

**章节大纲:**
- §4.1 性能瓶颈三大类：CPU / 内存 / I/O
- §4.2 num_cores 选择
  - cfg.project.num_cores = None → auto-detect
  - 显式指定的好处（资源调度可预测）
  - 与 SLURM/PBS 的 \texttt{\$SLURM\_CPUS\_PER\_TASK} 对齐
- §4.3 当前并行边界（spec 已说：站点级、按年级 OK；变量级 reserved）
- §4.4 内存监控
  - util/memory.py 的 max_rss / 实时采样
  - psutil 路径
- §4.5 内存爆炸的三种典型情形
  - 全球 hourly + 多年 + 多 sim
  - dask chunk 太大
  - 站点 CSV + grid sim 笛卡尔积
- §4.6 I/O 优化
  - HDF5 文件锁（已修复但要懂）
  - NFS / Lustre 上的小文件抖动
  - zarr cache 启用
- §4.7 Dask scheduler 选择（threaded / process / distributed）
- §4.8 oversubscription 风险（嵌套并行：joblib × MKL）
- §4.9 实测建议（怎么测）：先小规模 / 看 \texttt{/usr/bin/time -v}
- §4.10 调优 checklist

**审查要点:**
- parallel.py 的 backend 选择（joblib threading vs process）
- memory.py 是否在 task 之间释放
- DatasetProcessing 内部并行的 \texttt{n\_jobs} 参数

- [ ] Step 1-4

---

### Task 5: Chapter 5 — 缓存策略

**Files:** Create `docs/manual/operations/chapters/05-cache.tex`

**代码审查范围:**
- `src/openbench/data/cache.py`
- `src/openbench/runner/cache.py`（146 行）
- `src/openbench/util/cache_cleanup.py`（216 行）
- 用户卷第 6 章（不要重复）

**章节大纲:**
- §5.1 缓存层次：内存 / 磁盘 / zarr（用户卷 6 章已介绍）
- §5.2 输出目录结构与各 cache 文件位置
- §5.3 cache key 算法（Sha256 hash of canonical config dict）
- §5.4 cache 命中触发条件 / 失效规则
- §5.5 多用户共享 cache（同一 reference 多人评估）
  - 路径权限
  - 并发写冲突（atomic write 已实现）
- §5.6 cache_cleanup.py：自动清旧 cache
  - LRU 策略
  - 大小限额
  - 手动触发
- §5.7 何时该清缓存（升级 / catalog 改 / 异常断电）
- §5.8 zarr 格式细节
  - chunking
  - 压缩
  - 跨版本兼容性
- §5.9 cache 损坏检测与修复

**审查要点:**
- cache_cleanup.py 是否真有 LRU / size limit
- zarr write 是否原子
- 多进程同时写同一 cache 文件的行为

- [ ] Step 1-4

---

## Phase 4: 站点 + 监控 2 章（6-7）

### Task 6: Chapter 6 — 大规模站点评估实战

**Files:** Create `docs/manual/operations/chapters/06-stations-at-scale.tex`

**代码审查范围:**
- `src/openbench/data/station_matcher.py`
- `src/openbench/data/station_scanner.py`
- `tests/test_runner/test_local.py`（如有 station 集成测试）
- bug log 中关于 HDF5 locking + station 评估的修复（2026-04-03 fixes）

**章节大纲:**
- §6.1 站点评估的特殊性：每站独立 + IO 密集
- §6.2 站点列表 CSV 格式（fulllist）
- §6.3 station_scanner：批量扫描 NC 提取站点
- §6.4 station_matcher：grid sim → 站点采样
  - 最近邻 vs 双线性
  - 落空（站点位置无 grid 数据）的处理
- §6.5 千站点的资源预算
  - 内存：每站独立 → 与 num_cores 线性
  - 时间：与 num_cores 反比（理论上）
- §6.6 失败站点的批量诊断
  - 单站 NaN 的常见原因
  - 站点位置出 grid 范围
- §6.7 HDF5 locking 跨进程坑（已修复）
  - HDF5_USE_FILE_LOCKING=FALSE 自动设
  - Notebook 手动调时要自己设
  - 历史背景（2026-04-03 修复）
- §6.8 站点 NC 与 grid NC 的混合输入
- §6.9 实战配置示例（FLUXNET PLUMBER2 多站点）

**审查要点:**
- station_matcher 处理坐标系反向
- 落空站点是否会让整个评估失败
- ref symlink fix（memory 中提到的 2026-04-03 修）

- [ ] Step 1-4

---

### Task 7: Chapter 7 — 监控与日志

**Files:** Create `docs/manual/operations/chapters/07-monitoring.tex`

**代码审查范围:**
- `src/openbench/util/logging_system.py`（581 行）
- `src/openbench/util/progress.py`
- `src/openbench/util/report.py`
- `src/openbench/util/output.py`

**章节大纲:**
- §7.1 logging_system 总览
  - 默认 logger 配置
  - 级别与 handler
- §7.2 在 yaml 中调日志：debug_mode + 各模块单独
- §7.3 文件输出：output/<run>/run.log
- §7.4 实时监控：tail -f run.log
- §7.5 progress.py：CLI 与 GUI 共用的进度事件
  - 事件类型
  - 解析格式
- §7.6 report.py：HTML/PDF 总结报告（用户卷已介绍，这里讲细节）
  - 模板位置
  - 自定义模板
  - PDF 生成可选依赖
- §7.7 集成外部监控
  - Prometheus exporter 思路
  - 日志聚合（ELK / Loki）
- §7.8 长任务监控建议（dashboard、邮件提醒）
- §7.9 日志归档与清理

**审查要点:**
- logging_system 的循环与文件大小限制
- progress 事件解析是否健壮（GUI 用）
- report 模板的自定义入口

- [ ] Step 1-4

---

## Phase 5: 故障恢复 1 章（8）

### Task 8: Chapter 8 — 故障恢复与排错

**Files:** Create `docs/manual/operations/chapters/08-recovery.tex`

**代码审查范围:**
- `src/openbench/util/cache_cleanup.py`
- `src/openbench/util/exceptions.py`
- `src/openbench/runner/local.py:_validate_comparison_only_inputs`
- 用户卷第 10 章 troubleshooting（不重复，专注运维角度）

**章节大纲:**
- §8.1 中断后续跑（基础）
- §8.2 残留 lock 文件的清理
  - HDF5 lock
  - zarr lock
- §8.3 SSH 断连重试
  - 自动重连机制（如有）
  - 手动 ssh 续
- §8.4 磁盘满 / 配额超限
  - 输出目录清理
  - cache 清理
  - 估算磁盘需求
- §8.5 nested parallelism 死锁诊断
  - py-spy 抓堆栈
  - 看具体哪层 joblib 卡住
- §8.6 partial 失败的清理与重跑
- §8.7 从 partial 重跑的 cache 行为
- §8.8 损坏的 zarr cache 重建
- §8.9 升级 OpenBench 后旧输出怎么处理
- §8.10 不可恢复的失败：重启评估的最快路径

**审查要点:**
- cache_cleanup 的强制清理
- partial 状态的 _validate_comparison_only_inputs
- 异常 traceback 的可读性

- [ ] Step 1-4

---

## Phase 6: 5 个真实附录

### Task 9: Appendix A — Runtime / Runner / Parallel 选项

**Files:** Create `docs/manual/operations/appendices/A-runtime-options.tex`

**代码审查范围:**
- `src/openbench/config/schema.py`（runtime 字段）
- `src/openbench/util/parallel.py`（parallel knob）
- `src/openbench/runner/local.py`、`runner/remote.py`

**结构:** 一张大表格，行是字段，列是：默认值 / 含义 / 何时该改 / 关联章节

字段范围：
- project.num_cores / force / debug_mode / only_drawing / time_alignment / unified_mask / generate_report
- runner 内部 knob（如果有暴露）
- parallel 选项（threading vs process）

- [ ] Step 1-3 (write / compile / commit)

---

### Task 10: Appendix B — SSH / Remote 配置参考

**Files:** Create `docs/manual/operations/appendices/B-remote-config.tex`

**代码审查范围:**
- 同 Chapter 3

**结构:**
- §B.1 Connection profile YAML schema 字段
- §B.2 凭据存储路径与权限（macOS / Linux / Windows）
- §B.3 端口转发场景（堡垒机、跳板机）
- §B.4 known_hosts 处理
- §B.5 已知问题与解决

- [ ] Step 1-3

---

### Task 11: Appendix C — 缓存目录布局

**Files:** Create `docs/manual/operations/appendices/C-cache-layout.tex`

**结构:**
- 完整目录树（每层文件用途）
- 与版本演进的关系（zarr 格式可能变）
- 升级时哪些目录可以保留 / 必须删

- [ ] Step 1-3

---

### Task 12: Appendix D — 性能建议（建议性指导）

**Files:** Create `docs/manual/operations/appendices/D-performance-guidance.tex`

**结构:**
- §D.1 资源画像（按变量数 × 年数 × 分辨率估算）
  - Global × Monthly × 5yr × 4 vars × 1 sim
  - Global × Daily × 5yr × ...
  - Station × Hourly × ...
- §D.2 单 sim vs 多 sim 的资源差异
- §D.3 调参第一性原理：先 profiling 再调
- §D.4 不要做的事（典型错误：HPC 上 num_cores=1）

不跑实测；纯文字建议（spec §2 决定的方向）。

- [ ] Step 1-3

---

### Task 13: Appendix E — 日志消息分类索引

**Files:** Create `docs/manual/operations/appendices/E-log-index.tex`

**代码审查范围:**
- `grep -nE 'logger\.(error|warning|critical|info)' src/openbench/`（半自动）

**结构:**
- 按 logger 名分组（cli / config / data / runner / core / gui / util）
- 每条消息：模板、级别、典型场景、建议处置
- 与用户卷附录 E 区别：用户卷面向"我看到这条错误怎么修"；运维卷面向"我作为管理员要建监控告警，哪些消息需要告警 / 抑制 / 转发"

- [ ] Step 1-3

---

## Phase 7: 集成 + 验收

### Task 14: 更新 main_operations.tex 与端到端编译

**Files:** Modify `docs/manual/operations/main_operations.tex`

替换 stub 引用为 8 章 + 5 附录。

- [ ] **Step 1:** 改 main_operations.tex
- [ ] **Step 2:** 删 stub

```bash
git rm docs/manual/operations/chapters/00-hello.tex docs/manual/operations/appendices/A-stub.tex
```

- [ ] **Step 3:** `make ops`

```bash
cd /Volumes/Data01/Openbench/docs/manual && make ops 2>&1 | tail -5
# 预期：operations/main_operations.pdf ~80-100 页
```

- [ ] **Step 4:** `make all` 端到端

```bash
make clean && make all 2>&1 | tail -5
ls -la *.pdf user/*.pdf developer/*.pdf operations/*.pdf
```

预期：
- user: 134 页
- developer: 147 页
- operations: 80-100 页（new！）
- manual: 380+ 页

- [ ] **Step 5:** Bug log 复审 + 决议汇总

```bash
cat docs/superpowers/reviews/2026-04-30-manual-bugs.md
```

- [ ] **Step 6:** 整批 commit

---

### Task 15: 三卷集成清理（合订模式 appendix 锚点冲突）

**Files:** Modify `docs/manual/common/macros.tex` + 各 main_*.tex

合订模式（manual.pdf）下三卷各有 5 个 Appendix A-E，锚点冲突。修法（已记录在 manual-bugs.md，留给本 task）：

\subsection{方案}

在每卷 main_*.tex 的 \\appendix 之后立刻加：

```latex
% 用户卷
\renewcommand{\thechapter}{U\Alph{chapter}}
% 开发卷
\renewcommand{\thechapter}{D\Alph{chapter}}
% 运维卷
\renewcommand{\thechapter}{O\Alph{chapter}}
```

合订时 appendix 编号变为 UA/UB/.../DA/DB/.../OA/OB/...，PDF 锚点不冲突。独立编译时仍走 \\Alph{chapter} 的 A/B/C/D/E。

- [ ] **Step 1:** 改 main_user.tex / main_developer.tex / main_operations.tex
- [ ] **Step 2:** make clean && make all 验证
- [ ] **Step 3:** 检查 manual.log 没有 "Object @appendix.A already defined"
- [ ] **Step 4:** 更新 bug log 标记为已解决
- [ ] **Step 5:** Commit

---

## 自审清单

- [ ] **Spec 覆盖**：spec §4.3 列出的 8 章 + 5 附录全部产出
- [ ] **占位词扫描**：无 \\TODO / TBD 留在文中
- [ ] **类型一致性**：`main_operations.tex` `\\include{}` 路径与文件名一致
- [ ] **代码审查覆盖**：每章 task 列出的代码模块已实际阅读
- [ ] **不破坏 Plan 1-4**：Task 14/15 验证 4 个 PDF 仍 0 错误

---

## 完成后状态

- 运维卷 PDF 约 80-100 页
- 4 真实附录手写 + 1 半自动（log index）
- 三卷 appendix 锚点冲突已解决
- bug log 含全部写作期收集的代码缺陷及决议
- **三卷全部完工，OpenBench v3.0 中文 LaTeX 手册第一版可发布**

最终预期：
- user: ~134 页
- developer: ~147 页
- operations: ~90 页
- manual (合订): ~380 页

总计约 750+ 页 PDF，覆盖陆面模型评估的用户、贡献者、运维三类读者。

---

## 后续（不在本 plan）

手册第一版完工后的可能后续项目：

1. **英文版翻译**（spec §10 已声明不在范围）
2. **Sphinx / MkDocs site**（spec §10 已声明不在范围）
3. **真实性能基准实测**（如要把附录 D 从"建议性"升级为"实测"）
4. **GUI 截图补全**（用户卷第 8 章 \\TODO 占位）
5. **半自动错误消息生成器**（把附录 E 从手写转代码驱动）
6. **CI 集成**（手册编译纳入 GitHub Actions）

每项可独立成 plan / spec。
