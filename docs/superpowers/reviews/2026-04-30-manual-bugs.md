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
| 2026-04-30 | docs/manual/manual.pdf | xdvipdfmx 警告 `Object @appendix.A already defined` —— 三卷各有 Appendix A，合订时锚点冲突 | **已知限制**：占位阶段每卷只有 1 个附录；正式撰写时每卷 5 个不同附录 A-E 仍会冲突，需在 generator/真实附录 plan 中通过 `\renewcommand{\theappendix}` 加卷前缀解决 | （留待下一份 plan 处理） |
| 2026-04-30 | docs/manual/scripts/generate_config_schema.py + 同款 registry_schema | 生成的 longtable col_spec 是 `l l l p{5cm}`，"类型"列遇到 `Optional[list[str]]` / `dict[str, SimulationEntry]` 等长类型 → 大量 overfull \\hbox 警告（每页 6+ 个） | 改 col_spec 为 `p{2.5cm} p{3.5cm} p{2cm} p{4cm}` 或类似定宽，让 LaTeX 可换行；下一次 generator 维护时一并修 | （留待下一份 plan / 单独 follow-up） |
| 2026-04-30 | src/openbench/cli/run.py:15,63-66 | `--remote` 标志 help text "Remote host or saved profile name" 与运行时行为脱节：传入后立刻 SystemExit(1) 且提示"Remote execution not yet implemented. Install openbench[remote] and use openbench gui"。脚本化使用会困惑 | UX bug，建议任一：(a) 在 help text 加 "[NOT IMPLEMENTED]" 前缀；(b) 隐藏标志直到 CLI 远程实现；(c) 真正实现 CLI 远程（可能在 runner/remote.py 已有底层）。Plan 5 (Operations) 时再统一审 | 未修 |
| 2026-04-30 | src/openbench/cli/init_cmd.py:148 | **真 bug**：`openbench init` 写出 `reference: {sources: {<var>: <src>}}`（嵌套），但 `loader._build_reference` 期望 `reference: {<var>: <src>}`（flat）。结果：`init` 生成的 YAML 立刻被 `check` 拒绝，报 "reference.sources must be a string (source name), got dict"。 | 改为 flat：`config["reference"] = reference`（修法 A）；加端到端回归测试 `test_init_output_is_loadable` 防止再发生 | `67acd71` |
| 2026-04-30 | src/openbench/cli/data.py:42, 54 | `openbench data download` 与 `openbench data status` 子命令是 stub：download 打印"not yet implemented"；status 报告 registry 数量但 cache 部分写"not yet implemented"。help text 不写明 → 用户预期与实际不符 | 不算崩溃 bug，但 help text 应加 `[NOT IMPLEMENTED]` 前缀；或在 spec/Roadmap 中标注。Manual 第 4 章如实说明 | 未修（文档侧已说明） |
| 2026-04-30 | docs/manual/user/chapters/04-variables-references.tex | 中文 body 字体（fandol）没 U+2194 (↔) glyph；日志大量 "Missing character" 警告 | 把 "变量↔数据集" 替换为 "变量到数据集" | 已修（写作期内自修）|

## 写作期总结（用户卷完成时）

- 真 bug 1 个：`init_cmd.py` 写出的 reference 嵌套结构与 loader 不兼容 → 已修 + 加端到端回归测试 (`67acd71`)
- UX/Roadmap 问题 2 个（CLI `--remote` 未实现、`data download/status` stub）：未修，文档侧如实说明
- LaTeX/字体/工具链问题 5 个：`fandol` fontset、`-r ../latexmkrc`、`underscore` 包、`Menlo` monofont、`↔` 缺字；全部已修
- 写作错字 4 个：`\end{itemize>` / `\end{enumerate>` / `\end{minted>`；写完即修
