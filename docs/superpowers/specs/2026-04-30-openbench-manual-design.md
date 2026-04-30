# OpenBench 用户手册设计规范

**日期**：2026-04-30
**状态**：草案
**作者**：zhongwangwei + Claude

## 1. 概述

为 OpenBench v3.0 撰写一套面向不同读者的中文 LaTeX 手册，配合代码深度审查；审查中发现的 bug 在出现时立即报告，确认后再继续撰写。

本规范只定义**手册结构、技术选型、编写流程**；具体章节内容由后续 implementation plan 与 LaTeX 文件本身承载。

## 2. 设计决定汇总

| 项 | 选择 | 理由 |
|---|---|---|
| 受众分层 | 三卷：User / Developer / Operations | v3.0 同时面向科研用户、贡献者、HPC 运维三类读者；统一一卷会顾此失彼 |
| 内容深度 | 双层结构：教程 (Part I) + 参考 (Part II) | 同一卷既能用作上手教程，也能用作长期速查 |
| 输出格式 | LaTeX → PDF | 用户指定；适合长篇技术手册、印刷友好 |
| 文档类 | `book` | 通用、宏包兼容性最好；`memoir`/`scrbook` 增加学习成本 |
| 多文件结构 | 每卷独立 `main_*.tex` + 总卷 `manual.tex` 用 `subfiles` 合并 | 既能分卷交付，也能合订 |
| 中文支持 | `xelatex` + `ctex` | 最成熟的中文 LaTeX 路径；自动选择系统字体 |
| 代码块 | `minted` + `pygments`（需要 `-shell-escape`） | YAML / Python / bash 高亮质量最好 |
| 标注框 | `tcolorbox`（Note / Warning / Tip / Example 四种） | 视觉一致 |
| 图表 | `tikz`（架构图）+ `booktabs`（表格） | 专业表格 + 矢量图 |
| 链接与书签 | `hyperref` | PDF 内部跳转 + 大纲 |
| 章节组织风格 | 每卷选最自然结构（User=工作流 / Dev=子系统 / Ops=场景），统一用 `\part{教程}` / `\part{参考}` 划层 | 与"双层结构"目标对齐，又不强加同一骨架 |
| 语言 | 全中文 | 用户指定；代码与 CLI 字符串保留英文原貌 |
| Bug 报告时机 | 发现立即停笔报告，等用户确认后再继续 | 用户指定；保证不在错误信息基础上继续撰写 |
| 性能基准（卷 III 附录 D） | **建议性指导**（不跑实测） | 默认；避免基准数字误导跨环境读者；用户可在 spec review 时改为实测 |

## 3. 仓库目录布局

```
docs/manual/
├── manual.tex                  # 总卷主文件（subfiles 合并三卷）
├── Makefile                    # make user / make dev / make ops / make all / make clean
├── latexmkrc                   # latexmk 默认 xelatex + -shell-escape
├── README.md                   # 编译说明：依赖、命令、字体兜底
│
├── common/                     # 三卷共用资源
│   ├── preamble.tex            # 宏包导入：ctex / minted / tcolorbox / hyperref / booktabs / tikz / fancyhdr
│   ├── macros.tex              # 自定义命令：\openbench, \cli, \yamlkey, \var, \modname
│   ├── styles.tex              # tcolorbox 样式：noteBox / warnBox / tipBox / exampleBox
│   ├── glossary.tex            # 术语表（grid / station / reference / simulation / variable / profile / climatology / regrid …）
│   ├── bibliography.bib        # 文献：OpenBench 论文、关键数据集、相关方法
│   └── figures/                # 共享图：架构图、数据流图（.tikz / .pdf / .png）
│
├── user/
│   ├── main_user.tex           # \documentclass{book} + \input{../common/preamble}
│   ├── chapters/
│   │   ├── 01-overview.tex
│   │   ├── 02-quickstart.tex
│   │   ├── 03-config-structure.tex
│   │   ├── 04-variables-references.tex
│   │   ├── 05-simulation-profiles.tex
│   │   ├── 06-running.tex
│   │   ├── 07-results.tex
│   │   ├── 08-gui.tex
│   │   ├── 09-migration.tex
│   │   └── 10-troubleshooting.tex
│   └── appendices/
│       ├── A-config-reference.tex
│       ├── B-cli-reference.tex
│       ├── C-reference-datasets.tex
│       ├── D-model-profiles.tex
│       └── E-error-messages.tex
│
├── developer/
│   ├── main_developer.tex
│   ├── chapters/
│   │   ├── 01-architecture.tex
│   │   ├── 02-dev-environment.tex
│   │   ├── 03-config-subsystem.tex
│   │   ├── 04-data-subsystem.tex
│   │   ├── 05-registry.tex
│   │   ├── 06-core-engine.tex
│   │   ├── 07-visualization.tex
│   │   ├── 08-runner-cli.tex
│   │   ├── 09-gui-extension.tex
│   │   ├── 10-testing.tex
│   │   └── 11-contributing.tex
│   └── appendices/
│       ├── A-package-graph.tex
│       ├── B-public-api.tex
│       ├── C-registry-schema.tex
│       ├── D-internal-interfaces.tex
│       └── E-conventions.tex
│
├── operations/
│   ├── main_operations.tex
│   ├── chapters/
│   │   ├── 01-deployment.tex
│   │   ├── 02-hpc-install.tex
│   │   ├── 03-remote-ssh.tex
│   │   ├── 04-performance.tex
│   │   ├── 05-cache.tex
│   │   ├── 06-stations-at-scale.tex
│   │   ├── 07-monitoring.tex
│   │   └── 08-recovery.tex
│   └── appendices/
│       ├── A-runtime-options.tex
│       ├── B-remote-config.tex
│       ├── C-cache-layout.tex
│       ├── D-performance-guidance.tex
│       └── E-log-index.tex
│
└── _generated/                 # 由脚本生成的 LaTeX 片段（registry / config 同步）
    ├── reference_table.tex     # 自动从 reference_catalog.yaml 生成
    ├── model_table.tex         # 自动从 model_catalog.yaml 生成
    └── config_schema.tex       # 自动从 config/schema.py 生成
```

## 4. 三卷章节大纲

### 4.1 卷 I — User Guide（约 120 页）

**Part I 教程（工作流驱动）**

1. 概述与安装（6–8 页）
2. Quick Start（5–7 页）
3. 配置文件结构（10–12 页）
4. 选择变量与参考数据集（12–15 页）
5. 配置 simulation 与 model profile（10–12 页）
6. 运行评估（10–12 页）
7. 结果解读（12–15 页）
8. GUI Wizard（10–12 页；按真实代码 14 个 page）
9. 从旧版迁移（6–8 页）
10. 常见问题与排错（8–10 页）

**Part II 参考**

- 附录 A 完整配置项参考
- 附录 B CLI 命令完整参考
- 附录 C Reference 数据集清单（66 个）
- 附录 D Model Profile 清单
- 附录 E 错误信息与日志参考

### 4.2 卷 II — Developer Guide（约 150 页）

**Part I 教程（子系统驱动）**

1. 项目架构总览（8–10 页）
2. 开发环境（6–8 页）
3. config 子系统（14–18 页）
4. data 子系统（12–15 页）
5. registry 子系统（18–22 页，含两个 walkthrough）
6. core 评估引擎（18–22 页，含 metric/score/statistic walkthrough）
7. visualization（10–12 页，含 Fig 模块 walkthrough）
8. runner & CLI（10–12 页，含新 CLI 命令 walkthrough）
9. GUI 扩展（12–15 页，含新 wizard page walkthrough）
10. 测试（8–10 页）
11. 提交贡献（8–10 页）

**Part II 参考**

- 附录 A 包结构与依赖图
- 附录 B Public Python API
- 附录 C Registry YAML schema 完整字段
- 附录 D 内部 interfaces
- 附录 E CONVENTIONS 完整版

### 4.3 卷 III — Operations Guide（约 90 页）

**Part I 教程（场景驱动）**

1. 部署形态选择（6–8 页）
2. HPC 安装与依赖管理（8–10 页）
3. 远程 SSH 执行（14–18 页）
4. 性能调优（12–15 页）
5. 缓存策略（8–10 页）
6. 大规模站点评估实战（10–12 页）
7. 监控与日志（8–10 页）
8. 故障恢复与排错（8–10 页）

**Part II 参考**

- 附录 A Runtime / Runner / Parallel 选项
- 附录 B SSH / Remote 配置参考
- 附录 C 缓存目录布局
- 附录 D 性能建议（建议性指导）
- 附录 E 日志消息分类索引

## 5. 与代码同步策略

文档中所有"穷举式"参考内容必须**单一事实源在代码**：

| 参考章节 | 数据来源 | 生成方式 |
|---|---|---|
| 用户卷 附录 A 配置项 | `src/openbench/config/schema.py`（dataclass + 默认值） | 脚本扫描 dataclass，生成 `_generated/config_schema.tex` |
| 用户卷 附录 C 数据集 | `src/openbench/data/registry/reference_catalog.yaml` | 脚本生成 `_generated/reference_table.tex` |
| 用户卷 附录 D 模型 | `src/openbench/data/registry/model_catalog.yaml` | 脚本生成 `_generated/model_table.tex` |
| 开发卷 附录 C registry schema | `src/openbench/data/registry/schema.py` | 脚本生成 |
| 开发卷 附录 D internal interfaces | dataclass / Protocol / ABC | 脚本生成 |
| 用户卷 附录 E 错误消息 | `grep` `logging_system.py` 调用点 | 半自动：脚本提取候选，人工校对 |
| 运维卷 附录 E 日志索引 | 同上 | 半自动 |

**生成器位置**：`docs/manual/scripts/`（独立 Python 脚本，pytest 可调用）。
生成器在 `make all` 时自动运行；CI 中应把"是否需要重新生成"作为一个检查（避免代码改了但生成产物没刷新）。

## 6. Bug 报告流程

- 文件位置：`docs/superpowers/reviews/2026-04-30-manual-bugs.md`
- 触发：撰写任一章节时、若审查代码发现 bug
- 流程：
  1. **立即停笔**
  2. 在主对话里报告：模块、文件:行、症状、推测根因、修复建议
  3. 等用户回应（修 / 跳过 / 留待后处理）
  4. 把这次决议写入上述 reviews 文件
  5. 续写当前章节
- 不在 reviews 文件中堆积细节 —— 只记录"何时发现、何处、决议"

## 7. 编译与发布

```bash
# 第一次安装
brew install --cask mactex                  # 或者 tinytex
pip install Pygments                        # minted 依赖

cd docs/manual
make user        # 编译用户卷
make dev         # 编译开发者卷
make ops         # 编译运维卷
make all         # 编译三卷 + 总卷
make clean       # 清中间产物
```

`Makefile` 强制使用 `xelatex -shell-escape` 与 `latexmk` 自动多轮编译。`README.md` 提供 Linux / macOS / Docker 三种编译路径。

## 8. 撰写顺序（implementation plan 将细化）

按风险与依赖性，建议顺序：

1. **基础设施先行**：`common/preamble.tex` / `macros.tex` / `styles.tex` + `Makefile` + `latexmkrc` + 一个 hello world 的 `main_user.tex` 跑通编译
2. **生成器先写**：`docs/manual/scripts/*.py` —— 在写任何参考章节前，确保参考表能从代码自动产出
3. **卷 I 先全写**：用户卷面向读者最广，先打磨稳定再开后两卷
4. **卷 II 与卷 III 可并行**，但建议串行（卷 II 先），避免上下文切换成本
5. **每卷写完一轮**：spec self-review → 用户审阅 → 修订 → 移交下一卷

每章撰写时同步深度审查对应代码，发现 bug 触发第 6 节流程。

## 9. 风险与权衡

| 风险 | 影响 | 缓解 |
|---|---|---|
| 三卷工作量大（~360 页），周期长 | 写到中段时部分代码已经变化 | 启用第 5 节"自动生成"，把易变内容（registry、schema）从手写转为生成 |
| `minted -shell-escape` 在受限环境不可用 | 用户卷读者编译不出来 | `Makefile` 提供 `make user-listings` 后备，用纯 LaTeX 的 `listings` 替代 |
| 中文字体跨系统差异 | macOS 上看着好，Linux 编不过 | `ctex` 默认 fallback 到 fandol；`README.md` 列出三系统的字体推荐 |
| 全中文，国际读者不友好 | 海外用户访问受限 | 英文版作为后续工作（不在本 spec 范围内） |
| Walkthrough 章节示例与代码漂移 | 读者跑不通 | 每个 walkthrough 在 `tests/manual/` 下放 smoke test，CI 跑 |

## 10. 不在本 spec 范围

- 英文翻译（后续工作）
- 在线托管（readthedocs / mkdocs-material）（后续工作；当前优先 PDF）
- 视频教程
- 国际化（i18n）框架

## 11. 验收标准

1. 三卷分别可独立编译为 PDF：无 LaTeX 报错；未解析引用 (`undefined reference`)、未解析引用 (`citation undefined`) 为 0；overfull/underfull box 警告允许存在但需 < 5 处/卷
2. 总卷 `manual.tex` 可合并编译为单一 PDF
3. 所有"参考章节"由 `_generated/*.tex` 提供，并可由 `make generated` 重新生成（不依赖手抄）；附录章节通过 `\input{../_generated/...}` 引用
4. 每个 walkthrough 在仓库里有可运行的对应代码：开发卷 / 运维卷的 walkthrough 在 `tests/manual/` 下有 smoke test 由 CI 调用；用户卷不要求 smoke test，但所有示例 YAML 必须能被 `openbench check` 接受
5. 在审查中发现的 bug 全部记录在 `docs/superpowers/reviews/2026-04-30-manual-bugs.md` 并各有决议（修复 / 跳过 / 留待后处理）
