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
