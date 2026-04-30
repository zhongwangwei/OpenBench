# OpenBench 手册编写期 Bug 记录

**起始日期**：2026-04-30
**关联 spec**：`docs/superpowers/specs/2026-04-30-openbench-manual-design.md`
**关联 plan**：`docs/superpowers/plans/2026-04-30-manual-infrastructure.md`（首份）

## 流程

撰写章节时若发现代码 bug：

1. 立即停笔
2. 在主对话中报告：模块、文件:行、症状、推测根因、修复建议
3. 等待用户决议（修 / 跳过 / 留待后处理）
4. 把决议追加到本文件
5. 续写当前章节

不在本文件堆积细节；只记录"何时发现、何处、决议"。

## 记录

| 发现日期 | 模块 / 文件:行 | 症状摘要 | 决议 | 关联 commit |
|---|---|---|---|---|
| 2026-04-30 | docs/manual/manual.tex:4 | TL2026 ctex 在 macOS 自动选 `macold` fontset 但目标不可用，编译终止 | 改用 `fontset=fandol`（跨平台保底，已在 README 文档化） | `5f1de05` |
| 2026-04-30 | docs/manual/Makefile:user/dev/ops 目标 | `cd <subdir> && latexmk` 不读取上一级 `latexmkrc`，回退 DVI 模式 → 不出 PDF | 显式传 `-r ../latexmkrc`；将 `SUBRC` 变量化 | `5f1de05` |
| 2026-04-30 | docs/manual/common/macros.tex:41 | `\volumedivider` 中"─" (U+2500) 在 lmroman10-bold 没 glyph，警告 missing character | 替换为 `\textemdash{}` | (本批次 commit) |
| 2026-04-30 | docs/manual/manual.pdf | xdvipdfmx 警告 `Object @appendix.A already defined` —— 三卷各有 Appendix A-E，合订时锚点冲突 | 在 user/dev/ops main_*.tex 的 \appendix 后加 `\renewcommand{\thechapter}{<X>\Alph{chapter}}` + `\renewcommand{\theHchapter}{<X>\Alph{chapter}}`（U/D/O 卷前缀）。**hyperref 内部 anchor 用 \theHchapter，必须双修** | Plan 5 Task 15 ✅ |
| 2026-04-30 | docs/manual/scripts/generate_config_schema.py + 同款 registry_schema | 生成的 longtable col_spec 是 `l l l p{5cm}`，"类型"列遇到 `Optional[list[str]]` / `dict[str, SimulationEntry]` 等长类型 → 大量 overfull \\hbox 警告（每页 6+ 个） | 改 col_spec 为 `p{2.5cm} p{3.5cm} p{2cm} p{4cm}` 或类似定宽，让 LaTeX 可换行；下一次 generator 维护时一并修 | （留待下一份 plan / 单独 follow-up） |
| 2026-04-30 | src/openbench/cli/run.py:15,63-66 | `--remote` 标志 help text "Remote host or saved profile name" 与运行时行为脱节：传入后立刻 SystemExit(1) 且提示"Remote execution not yet implemented. Install openbench[remote] and use openbench gui"。脚本化使用会困惑 | UX bug，建议任一：(a) 在 help text 加 "[NOT IMPLEMENTED]" 前缀；(b) 隐藏标志直到 CLI 远程实现；(c) 真正实现 CLI 远程（可能在 runner/remote.py 已有底层）。Plan 5 (Operations) 时再统一审 | 未修 |
| 2026-04-30 | src/openbench/cli/init_cmd.py:148 | **真 bug**：`openbench init` 写出 `reference: {sources: {<var>: <src>}}`（嵌套），但 `loader._build_reference` 期望 `reference: {<var>: <src>}`（flat）。结果：`init` 生成的 YAML 立刻被 `check` 拒绝，报 "reference.sources must be a string (source name), got dict"。 | 改为 flat：`config["reference"] = reference`（修法 A）；加端到端回归测试 `test_init_output_is_loadable` 防止再发生 | `67acd71` |
| 2026-04-30 | src/openbench/cli/data.py:42, 54 | `openbench data download` 与 `openbench data status` 子命令是 stub：download 打印"not yet implemented"；status 报告 registry 数量但 cache 部分写"not yet implemented"。help text 不写明 → 用户预期与实际不符 | 不算崩溃 bug，但 help text 应加 `[NOT IMPLEMENTED]` 前缀；或在 spec/Roadmap 中标注。Manual 第 4 章如实说明 | 未修（文档侧已说明） |
| 2026-04-30 | docs/manual/user/chapters/04-variables-references.tex | 中文 body 字体（fandol）没 U+2194 (↔) glyph；日志大量 "Missing character" 警告 | 把 "变量↔数据集" 替换为 "变量到数据集" | 已修（写作期内自修）|
| 2026-04-30 | src/openbench/config/{schema,loader,resolver,adapter}.py | **真 regression**：v3.0a1 把 reference schema 从 v2.x 的 `dict[str, str|list[str]]` 简化为 `dict[str, str]`，丢失多源能力。Urban 类变量典型 `["MODIST", "TRIMS_LST"]` 双源在 migration 后会被 loader 拒绝。引擎层（v2.x openbench.py）原本支持 list；v3.0 adapter 也只取单值 | 全栈恢复多源：schema 改 `str|list[str]`；loader 接受 list / 拆 comma；resolver 每变量产多个 ResolvedReference；adapter 写 list 并 (sim × ref) 笛卡尔积；8 个新测试 | `c9fcbdc`（配置层完成；下两条补 runtime 漏修）|
| 2026-04-30 | src/openbench/runner/local.py:504 | **`c9fcbdc` 漏修**：runner 用 `ref_done: bool` 仅追踪"该变量已处理任意 ref"，多 ref 中只第一个被 `prepare_source("ref")`；后续 ref 永不被预处理。所有 task 标 `ref_preprocessed=True`，per-task fallback 也跳过。结果：check 通过、4 个 task 入 list、但运行时第 2/N 个 ref 缺预处理 NC | `ref_done: bool` → `refs_done: set[str]`；`ref_data_dir` / `ref_file_path` 改 `dict[ref_source]`；stn×stn symlink 与 per_pair mask copy 都按 ref 分别追踪。加 `test_preprocess_runs_for_each_ref_source` 端到端 mock 验证 | 后续 commit ✅ |
| 2026-04-30 | src/openbench/config/adapter.py:138-145 | **`c9fcbdc` 漏修**：`has_grid_evaluation()` 循环 ref 但 sim 只看 `sim_sources[0]`。例：`ref=stn` + `sim=[SimStn, SimGrid]` 返回 `False`，但 SimGrid 实际需要 grid 评估 → 下游 phase 错跳过 | 改完整 (ref × sim) 笛卡尔积；加 mixed sim types + pure stn×stn 两个回归测试 | 后续 commit ✅ |
| 2026-04-30 | src/openbench/runner/local.py（混合形态边缘） | **`48bee6f` 仍漏修**：`refs_done: set[str]` 按 ref_source 单键去重，但同一 ref 在不同 sim 数据类型下输出位置不同：grid×grid 写 flat NC 可共享；stn-涉及任务运行 `extract_station_data_if_needed` 后 \emph{删除 flat NC}。如果 `simulation: {SimGrid, SimStn}` 共用同一 grid ref，序列 (RefA,SimGrid) → (RefA,SimStn,删 flat) → (RefA,SimGrid2) 让第三任务找不到 flat | 改双键去重 `(ref_source, "_grid" or sim_source)`：grid×grid 跨 sim 共享；stn-涉及任务每对独立。加 `test_preprocess_mixed_grid_and_stn_sims_with_same_grid_ref` 验证同 ref 在 mixed sim 下走 2 次 prep（grid + stn-pair） | `bf561bc` ✅ |
| 2026-04-30 | src/openbench/runner/local.py:107 | **`c9fcbdc` 系列 \emph{再次}漏修**：cache 命中后的输出文件校验 glob `f"{var_name}*{sim_source}*"` 不含 ref_source。multi-ref 配置下 (Var,SimA,RefA) 的输出文件 `Var_ref_RefA_sim_SimA_*.nc` 会让 (Var,SimA,RefB) 的 cache 校验误判通过 → `_evaluate_single` 返回 `skipped=True`，但 RefB 实际从未跑过 | 复用同文件第 321 行已有的 `_find_existing_outputs`（其 pattern 含 ref_source）；加 `test_cache_validation_pattern_includes_ref_source` | `27877e5` ✅ |
| 2026-04-30 | src/openbench/runner/local.py（grid → stn → grid 序列） | **`bf561bc` 又漏修**：双键去重解决了 \emph{何时跑预处理}，但没解决 \emph{预处理副作用}。`extract_station_data_if_needed` 在 stn-涉及预处理末尾删除共享 flat NC。runner 先跑全部预处理再统一评估。序列 [SimGrid1, SimStn, SimGrid2]：grid prep 写 flat → stn prep 删 flat → grid2 命中 dedupe 跳过 → 两个 grid 评估都读不到 flat。原 mixed-type 测试只数次数没模拟删除 | 加 \"flat NC restoration\" 阶段：在 `_preprocess_variable` 末尾，对 `refs_with_stn_prep ∩ refs_with_grid_tasks` 的每个 ref 重跑一次 grid prep 恢复 flat。加 `test_preprocess_grid_stn_grid_sequence_restores_deleted_flat` 用 FakeProcessor 模拟 stn 删 flat，断言函数返回后 flat 仍在磁盘 | 本次 commit ✅ |

## 写作期总结（用户卷完成时）

- 真 bug 1 个：`init_cmd.py` 写出的 reference 嵌套结构与 loader 不兼容 → 已修 + 加端到端回归测试 (`67acd71`)
- UX/Roadmap 问题 2 个（CLI `--remote` 未实现、`data download/status` stub）：未修，文档侧如实说明
- LaTeX/字体/工具链问题 5 个：`fandol` fontset、`-r ../latexmkrc`、`underscore` 包、`Menlo` monofont、`↔` 缺字；全部已修
- 写作错字 4 个：`\end{itemize>` / `\end{enumerate>` / `\end{minted>`；写完即修
