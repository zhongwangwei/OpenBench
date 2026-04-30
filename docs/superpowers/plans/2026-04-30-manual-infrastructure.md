# OpenBench 手册基础设施实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `docs/manual/` 下建立完整的 LaTeX 构建环境，使 `make all` 能编译出三卷独立 PDF（User / Developer / Operations）以及一个合并的总卷 PDF；所有 PDF 当前含占位内容（hello-world + 验证宏命令的 smoke 章节）。

**Architecture:** 每卷一个 `main_*.tex` 用 `subfiles` 类，可独立编译；总卷 `manual.tex` 用 `\subfile{}` 合并三卷。共享 `common/preamble.tex` + `styles.tex` + `macros.tex` 由顶层 `manual.tex` 在 \\documentclass 后 \\input；subfiles 自动让独立编译时也继承顶层 preamble。`Makefile` + `latexmkrc` 强制 `xelatex -shell-escape`。

**Tech Stack:** TeX Live 2026 (xelatex + latexmk)、ctex (ctexbook)、subfiles、minted (依赖 pygmentize)、tcolorbox、hyperref、booktabs、longtable、tikz、fancyhdr、xcolor。

**关联 spec:** `docs/superpowers/specs/2026-04-30-openbench-manual-design.md`

**不在本计划范围:** 章节真实内容（每卷各有独立 plan）；自动生成器脚本（独立 plan）；CI 集成。

---

## 文件结构

本计划新建以下文件（不修改任何已有代码）：

| 路径 | 行数估计 | 责任 |
|---|---|---|
| `docs/manual/.gitignore` | ~20 | 忽略 LaTeX 构建产物 |
| `docs/manual/Makefile` | ~50 | `user` / `dev` / `ops` / `master` / `all` / `clean` / `probe` 目标 |
| `docs/manual/latexmkrc` | ~15 | latexmk 配置：xelatex + shell-escape |
| `docs/manual/README.md` | ~80 | 编译说明：依赖、命令、字体兜底 |
| `docs/manual/manual.tex` | ~40 | 总卷主文件，用 subfiles 合并三卷 |
| `docs/manual/common/preamble.tex` | ~60 | 宏包导入与全局配置 |
| `docs/manual/common/styles.tex` | ~50 | 4 种 tcolorbox 标注框样式 |
| `docs/manual/common/macros.tex` | ~30 | `\openbench` / `\cli` / `\yamlkey` / `\var` / `\modname` 等 |
| `docs/manual/common/glossary.tex` | ~30 | 术语表骨架（5–8 条） |
| `docs/manual/common/bibliography.bib` | ~20 | BibTeX 骨架（OpenBench 论文条目） |
| `docs/manual/common/figures/.gitkeep` | 0 | 占位 |
| `docs/manual/user/main_user.tex` | ~40 | 用户卷主文件（subfiles 类） |
| `docs/manual/user/chapters/00-hello.tex` | ~30 | 用户卷 smoke 章节 |
| `docs/manual/user/appendices/A-stub.tex` | ~25 | 用户卷 smoke 附录 |
| `docs/manual/developer/main_developer.tex` | ~40 | 开发卷主文件 |
| `docs/manual/developer/chapters/00-hello.tex` | ~25 | 开发卷 smoke 章节 |
| `docs/manual/developer/appendices/A-stub.tex` | ~20 | 开发卷 smoke 附录 |
| `docs/manual/operations/main_operations.tex` | ~40 | 运维卷主文件 |
| `docs/manual/operations/chapters/00-hello.tex` | ~25 | 运维卷 smoke 章节 |
| `docs/manual/operations/appendices/A-stub.tex` | ~20 | 运维卷 smoke 附录 |
| `docs/manual/_generated/.gitkeep` | 0 | 占位（生成器输出目录） |
| `docs/superpowers/reviews/2026-04-30-manual-bugs.md` | ~30 | Bug 记录初始化 |

---

## Phase 0: 准备与环境检查

### Task 1: 验证 LaTeX 环境

**Files:** 仅查询，不创建。

- [ ] **Step 1:** 确认 `xelatex`、`latexmk`、`pygmentize` 可执行

```bash
which xelatex && xelatex --version | head -1
which latexmk && latexmk --version | head -1
which pygmentize && pygmentize -V | head -1
```

预期：三条都打印路径与版本号。

- [ ] **Step 2:** 确认 ctex 与 minted 已安装

```bash
kpsewhich ctexbook.cls && kpsewhich minted.sty && kpsewhich tcolorbox.sty && kpsewhich subfiles.cls
```

预期：四条都打印 `.cls` 或 `.sty` 文件的绝对路径；任何一条空白则需安装对应宏包。

如果 `pygmentize` 不存在：`pip install Pygments`。
如果某宏包缺失：在 macOS `tlmgr install <pkgname>`，Linux 见发行版包管理器。

---

### Task 2: 创建目录骨架与 Bug 日志

**Files:**
- Create: `docs/manual/common/figures/.gitkeep`
- Create: `docs/manual/user/chapters/.gitkeep`
- Create: `docs/manual/user/appendices/.gitkeep`
- Create: `docs/manual/developer/chapters/.gitkeep`
- Create: `docs/manual/developer/appendices/.gitkeep`
- Create: `docs/manual/operations/chapters/.gitkeep`
- Create: `docs/manual/operations/appendices/.gitkeep`
- Create: `docs/manual/_generated/.gitkeep`
- Create: `docs/superpowers/reviews/2026-04-30-manual-bugs.md`

- [ ] **Step 1:** 创建目录树

```bash
mkdir -p docs/manual/{common/figures,user/{chapters,appendices},developer/{chapters,appendices},operations/{chapters,appendices},_generated}
touch docs/manual/common/figures/.gitkeep \
      docs/manual/user/chapters/.gitkeep \
      docs/manual/user/appendices/.gitkeep \
      docs/manual/developer/chapters/.gitkeep \
      docs/manual/developer/appendices/.gitkeep \
      docs/manual/operations/chapters/.gitkeep \
      docs/manual/operations/appendices/.gitkeep \
      docs/manual/_generated/.gitkeep
```

- [ ] **Step 2:** 创建 bug 日志初始文件

写 `docs/superpowers/reviews/2026-04-30-manual-bugs.md`：

```markdown
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
| _（暂无）_ | | | | |
```

- [ ] **Step 3:** Commit

```bash
git add docs/manual/ docs/superpowers/reviews/2026-04-30-manual-bugs.md
git commit -m "docs: scaffold manual directory tree and bug log"
```

---

### Task 3: 添加 .gitignore

**Files:**
- Create: `docs/manual/.gitignore`

- [ ] **Step 1:** 写 `.gitignore`

```
# LaTeX build artifacts
*.aux
*.log
*.out
*.toc
*.lof
*.lot
*.fls
*.fdb_latexmk
*.synctex.gz
*.bbl
*.blg
*.run.xml
*.bcf
*.idx
*.ilg
*.ind
*.nav
*.snm
*.vrb

# minted cache
_minted-*/
*.pyg

# subfiles intermediate
*-blx.bib

# PDF outputs (do NOT commit; use make to rebuild)
*.pdf
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/.gitignore
git commit -m "docs: ignore LaTeX build artifacts"
```

---

## Phase 1: 构建配置

### Task 4: 写 latexmkrc

**Files:**
- Create: `docs/manual/latexmkrc`

- [ ] **Step 1:** 写 `latexmkrc`

```perl
# docs/manual/latexmkrc — latexmk 配置
# - 使用 xelatex（中文支持）
# - 启用 -shell-escape（minted 需要）
# - synctex=1（PDF 反查源码定位）

$pdf_mode = 5;  # 5 = xelatex
$xelatex = 'xelatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
$pdflatex = $xelatex;

# 额外清理的扩展名
push @generated_exts, 'synctex.gz', 'thm', 'pyg';
$clean_ext .= ' %R-blx.bib %R.run.xml %R.bcf';
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/latexmkrc
git commit -m "docs: configure latexmk for xelatex with shell-escape"
```

---

### Task 5: 写 Makefile

**Files:**
- Create: `docs/manual/Makefile`

- [ ] **Step 1:** 写 `Makefile`

```makefile
# OpenBench 手册编译 Makefile
# 用法：
#   make probe   — 检查 LaTeX 工具链
#   make user    — 编译用户卷（user/main_user.pdf）
#   make dev     — 编译开发卷（developer/main_developer.pdf）
#   make ops     — 编译运维卷（operations/main_operations.pdf）
#   make master  — 编译总卷（manual.pdf）
#   make all     — 三卷 + 总卷全部编译
#   make clean   — 清理所有中间产物与 PDF

LATEXMK = latexmk
RM = rm -rf

.PHONY: all user dev ops master clean probe

all: user dev ops master

user:
	cd user && $(LATEXMK) main_user.tex

dev:
	cd developer && $(LATEXMK) main_developer.tex

ops:
	cd operations && $(LATEXMK) main_operations.tex

master:
	$(LATEXMK) manual.tex

clean:
	-cd user && $(LATEXMK) -C main_user.tex
	-cd developer && $(LATEXMK) -C main_developer.tex
	-cd operations && $(LATEXMK) -C main_operations.tex
	-$(LATEXMK) -C manual.tex
	$(RM) user/_minted-* developer/_minted-* operations/_minted-* _minted-*

probe:
	@echo "=== xelatex ==="
	@which xelatex && xelatex --version | head -1 || (echo "MISSING" && exit 1)
	@echo "=== latexmk ==="
	@which latexmk && latexmk --version | head -1 || (echo "MISSING" && exit 1)
	@echo "=== pygmentize ==="
	@which pygmentize && pygmentize -V | head -1 || (echo "MISSING" && exit 1)
	@echo "=== ctex / minted / tcolorbox / subfiles ==="
	@kpsewhich ctexbook.cls && kpsewhich minted.sty && kpsewhich tcolorbox.sty && kpsewhich subfiles.cls
```

- [ ] **Step 2:** 测试 probe 目标

```bash
cd docs/manual && make probe
```

预期：三个工具的版本号 + 四个 `.cls/.sty` 路径。

- [ ] **Step 3:** Commit

```bash
git add docs/manual/Makefile
git commit -m "docs: add Makefile with per-volume and master targets"
```

---

### Task 6: 写 README

**Files:**
- Create: `docs/manual/README.md`

- [ ] **Step 1:** 写 `README.md`

```markdown
# OpenBench 用户手册（LaTeX 源）

本目录是 OpenBench v3.0 三卷中文用户手册的 LaTeX 源文件。设计规范见 [`docs/superpowers/specs/2026-04-30-openbench-manual-design.md`](../superpowers/specs/2026-04-30-openbench-manual-design.md)。

## 卷划分

| 卷 | 主文件 | 输出 PDF | 受众 |
|---|---|---|---|
| 用户指南 | `user/main_user.tex` | `user/main_user.pdf` | 陆面模型研究者 |
| 开发者指南 | `developer/main_developer.tex` | `developer/main_developer.pdf` | 贡献者 |
| 运维指南 | `operations/main_operations.tex` | `operations/main_operations.pdf` | HPC 管理员 |
| 合订总卷 | `manual.tex` | `manual.pdf` | 全部读者 |

## 依赖

- TeX Live ≥ 2023（含 `xelatex`、`latexmk`、`ctex`、`minted`、`tcolorbox`、`subfiles`、`hyperref`、`booktabs`、`longtable`、`tikz`、`fancyhdr`、`xcolor`）
- Pygments（minted 的代码高亮后端）：`pip install Pygments`

### macOS 安装

```bash
brew install --cask mactex          # 完整 TeX Live
pip install Pygments
```

### Linux (Debian/Ubuntu) 安装

```bash
sudo apt install texlive-full
pip install Pygments
```

### 运行环境检查

```bash
cd docs/manual
make probe
```

应当打印 xelatex / latexmk / pygmentize 版本号以及 4 个核心宏包的路径。

## 编译

```bash
cd docs/manual

# 单卷
make user
make dev
make ops

# 总卷（合并三卷）
make master

# 全部（4 个 PDF）
make all

# 清理中间产物
make clean
```

## 中文字体

使用 `ctexbook` 文档类，字体由 `ctex` 自动检测：

- macOS：默认 STSong / STKaiti（系统自带）
- Linux：默认尝试 Noto CJK，失败则回退到 `fandol`（ctex 自带的免费中文字体；编译可成功但视觉略简陋）
- Windows：默认 SimSun / KaiTi / SimHei / FangSong

如要强制使用特定字体，在 `common/preamble.tex` 中调整 `\setCJKmainfont`。

## 单章节调试

每卷的 `main_*.tex` 使用 `subfiles` 类，单文件可独立编译；这意味着可以只编译某一章节用于快速预览。

```bash
cd docs/manual/user
xelatex -shell-escape chapters/00-hello.tex
```

## Bug 跟踪

撰写期间发现 OpenBench 代码 bug 记录在 [`docs/superpowers/reviews/2026-04-30-manual-bugs.md`](../superpowers/reviews/2026-04-30-manual-bugs.md)。
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/README.md
git commit -m "docs: add manual build README"
```

---

## Phase 2: 共享 LaTeX 资源

### Task 7: 写 common/preamble.tex

**Files:**
- Create: `docs/manual/common/preamble.tex`

- [ ] **Step 1:** 写 `preamble.tex`

```latex
% docs/manual/common/preamble.tex
% 由顶层 main_*.tex 在 \documentclass 之后 \input
% 与 styles.tex / macros.tex 平级，但只负责宏包导入与全局配置

% --- 链接与 PDF 书签 ---
\usepackage[
  bookmarks=true,
  bookmarksnumbered=true,
  bookmarksopen=true,
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black,
  pdfencoding=auto
]{hyperref}

% --- 表格 ---
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}

% --- 图形 ---
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric,backgrounds,fit,calc}

% --- 颜色 ---
\usepackage{xcolor}
\definecolor{obblue}{HTML}{0B5394}
\definecolor{obamber}{HTML}{C18B00}
\definecolor{obred}{HTML}{B23A3A}
\definecolor{obgreen}{HTML}{2E7D32}
\definecolor{obgray}{HTML}{4A4A4A}

% --- 代码高亮 ---
\usepackage{minted}
\setminted{
  fontsize=\small,
  breaklines=true,
  breakanywhere=true,
  frame=lines,
  framesep=2mm,
  bgcolor=gray!5,
  numbersep=5pt,
  tabsize=2,
  autogobble=true,
}

% --- 标注框 ---
\usepackage[most]{tcolorbox}

% --- 页眉页脚 ---
\usepackage{fancyhdr}
\setlength{\headheight}{14pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\nouppercase\rightmark}
\fancyhead[LO]{\nouppercase\leftmark}
\renewcommand{\headrulewidth}{0.4pt}

% --- subfiles（顶层 manual.tex 用，subfile 内不会重复加载） ---
\usepackage{subfiles}

% --- 元数据（PDF 属性，被各卷覆盖） ---
\hypersetup{
  pdftitle={OpenBench 手册},
  pdfauthor={OpenBench 项目组},
  pdfsubject={陆面模型评估框架},
  pdfkeywords={OpenBench, land surface model, benchmarking, 中文}
}
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/common/preamble.tex
git commit -m "docs: add shared LaTeX preamble"
```

---

### Task 8: 写 common/styles.tex

**Files:**
- Create: `docs/manual/common/styles.tex`

- [ ] **Step 1:** 写 `styles.tex`

```latex
% docs/manual/common/styles.tex
% 4 种 tcolorbox 标注框样式：Note / Warning / Tip / Example
% 由顶层 main_*.tex 在 preamble 之后 \input

% Note：信息性提示
\newtcolorbox{noteBox}{
  enhanced,
  colback=obblue!5,
  colframe=obblue,
  fonttitle=\bfseries,
  title=注意,
  attach boxed title to top left={xshift=4mm,yshift*=-\tcboxedtitleheight/2},
  boxed title style={size=small,colback=obblue,colframe=obblue},
  top=3mm, bottom=2mm, left=3mm, right=3mm,
  arc=2mm,
  before skip=2mm, after skip=2mm,
}

% Warning：陷阱与潜在错误
\newtcolorbox{warnBox}{
  enhanced,
  colback=obred!5,
  colframe=obred,
  fonttitle=\bfseries,
  title=警告,
  attach boxed title to top left={xshift=4mm,yshift*=-\tcboxedtitleheight/2},
  boxed title style={size=small,colback=obred,colframe=obred},
  top=3mm, bottom=2mm, left=3mm, right=3mm,
  arc=2mm,
  before skip=2mm, after skip=2mm,
}

% Tip：最佳实践与省力技巧
\newtcolorbox{tipBox}{
  enhanced,
  colback=obgreen!5,
  colframe=obgreen,
  fonttitle=\bfseries,
  title=提示,
  attach boxed title to top left={xshift=4mm,yshift*=-\tcboxedtitleheight/2},
  boxed title style={size=small,colback=obgreen,colframe=obgreen},
  top=3mm, bottom=2mm, left=3mm, right=3mm,
  arc=2mm,
  before skip=2mm, after skip=2mm,
}

% Example：完整可运行示例
\newtcolorbox{exampleBox}[1][示例]{
  enhanced,
  colback=obamber!5,
  colframe=obamber,
  fonttitle=\bfseries,
  title=#1,
  attach boxed title to top left={xshift=4mm,yshift*=-\tcboxedtitleheight/2},
  boxed title style={size=small,colback=obamber,colframe=obamber},
  top=3mm, bottom=2mm, left=3mm, right=3mm,
  arc=2mm,
  before skip=2mm, after skip=2mm,
}
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/common/styles.tex
git commit -m "docs: add tcolorbox annotation styles"
```

---

### Task 9: 写 common/macros.tex

**Files:**
- Create: `docs/manual/common/macros.tex`

- [ ] **Step 1:** 写 `macros.tex`

```latex
% docs/manual/common/macros.tex
% 自定义语义命令；调整全局样式时只改本文件

% --- 项目名 ---
\newcommand{\openbench}{\textsf{OpenBench}}

% --- 代码语义命令 ---
\newcommand{\cli}[1]{\texttt{#1}}                         % CLI 命令
\newcommand{\yamlkey}[1]{\texttt{#1}}                      % YAML 键
\newcommand{\var}[1]{\textsf{#1}}                          % 评估变量名
\newcommand{\modname}[1]{\texttt{#1}}                      % Python 模块/类名
\newcommand{\file}[1]{\texttt{#1}}                         % 文件名
\newcommand{\dataset}[1]{\textsf{#1}}                      % 数据集名

% --- 标注命令简写（一行内可用） ---
\newcommand{\noteinline}[1]{\textcolor{obblue}{\textbf{注：}#1}}
\newcommand{\warninline}[1]{\textcolor{obred}{\textbf{警告：}#1}}

% --- 章节内的 TODO 占位（生产前必须清空） ---
\newcommand{\TODO}[1]{\textcolor{obred}{\textbf{[TODO: #1]}}}

% --- 卷感知的 part：独立编译用 \part（编号），合订用 \part*（不编号）---
% 解决总卷 TOC 中 \part 跨卷连续编号导致卷边界混乱的问题
% \ifSubfilesClassLoaded 由 subfiles 包提供（subfiles 1.3+，TeX Live 2018+）
\providecommand{\volpart}[1]{%
  \ifSubfilesClassLoaded{\part{#1}}{\part*{#1}\addcontentsline{toc}{part}{#1}}%
}

% --- 卷分隔（仅合订时使用，独立编译时由 \maketitle 替代）---
\providecommand{\volumedivider}[2]{%
  % #1 = 卷序号（如"第一卷"），#2 = 卷名（如"用户指南"）
  \cleardoublepage
  \thispagestyle{empty}
  \vspace*{4cm}
  \begin{center}
    {\Huge\bfseries #1}\\[1.5ex]
    {\Huge\bfseries #2}
  \end{center}
  \cleardoublepage
  \markboth{#2}{#2}
  \addcontentsline{toc}{chapter}{#1 ─ #2}
}
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/common/macros.tex
git commit -m "docs: add semantic LaTeX macros"
```

---

### Task 10: 写 common/glossary.tex 与 bibliography.bib 骨架

**Files:**
- Create: `docs/manual/common/glossary.tex`
- Create: `docs/manual/common/bibliography.bib`

- [ ] **Step 1:** 写 `glossary.tex`（手册附录可以 \input 它）

```latex
% docs/manual/common/glossary.tex
% 共享术语表骨架；各卷如需可在自己的章节里 \input{../common/glossary}
% 后续每个 plan 撰写时按需新增条目

\section*{术语表}
\addcontentsline{toc}{section}{术语表}

\begin{description}
  \item[Reference dataset]
    用作评估"真值"的观测或再分析数据集，例如 \dataset{GLEAM\_v4.2a}、\dataset{ERA5LAND}。
  \item[Simulation]
    待评估的模型输出。配置中通过 \yamlkey{simulation.<name>} 注册。
  \item[Variable]
    评估的物理量，如 \var{Evapotranspiration}、\var{Latent\_Heat}；标准命名见用户卷第 4 章。
  \item[Model profile]
    模型变量映射模板，定义文件名模式与单位换算；内置 CoLM2024 / CLM5 / ERA5-Land。
  \item[Grid / Station]
    数据形态：grid 为规则经纬度网格；station 为离散观测点（FLUXNET、GRDC 等）。
  \item[Climatology]
    多年平均的季节循环；OpenBench 的可选预处理阶段，详见用户卷第 6 章。
  \item[Regrid]
    将不同分辨率的源数据重网格化到目标分辨率（\yamlkey{project.grid\_res}）。
  \item[Time alignment]
    多个 simulation 与 reference 的时间窗口对齐策略：
    \texttt{intersection} / \texttt{per\_pair} / \texttt{strict}。
\end{description}
```

- [ ] **Step 2:** 写 `bibliography.bib` 骨架

```bibtex
% docs/manual/common/bibliography.bib
% 文献库骨架；后续每章引用时新增条目

@article{openbench2024,
  author = {Wei, Zhongwang and others},
  title = {OpenBench: An Open-source Land Surface Model Benchmarking System},
  journal = {{TBD}},
  year = {2024},
  note = {手册撰写期占位条目；正式条目待论文发表后补充}
}

@misc{colm2024,
  author = {{CoLM Development Team}},
  title = {CoLM2024: Common Land Model 2024},
  year = {2024},
  note = {占位条目}
}
```

- [ ] **Step 3:** Commit

```bash
git add docs/manual/common/glossary.tex docs/manual/common/bibliography.bib
git commit -m "docs: add glossary and bibliography skeletons"
```

---

## Phase 3: 三卷骨架

### Task 11: 写用户卷 main_user.tex

**Files:**
- Create: `docs/manual/user/main_user.tex`

- [ ] **Step 1:** 写 `main_user.tex`

```latex
% docs/manual/user/main_user.tex
% 用户卷主文件；既能独立编译也能被 manual.tex \subfile{} 包含

\documentclass[../manual]{subfiles}

\begin{document}

% --- 独立编译 vs 合订模式分支 ---
\ifSubfilesClassLoaded{%
  % 独立编译：完整标题页 + 目录
  \frontmatter
  \title{\openbench{} 用户指南\\[1ex]\large User Guide}
  \author{OpenBench 项目组}
  \date{\today}
  \maketitle
  \tableofcontents
  \mainmatter
}{%
  % 合订时：插入卷分隔页（替代 \maketitle）
  \volumedivider{第一卷}{用户指南}
}

% --- 教程部分 ---
\volpart{教程}
\include{chapters/00-hello}

% --- 参考部分（附录）---
\appendix
\volpart{参考}
\include{appendices/A-stub}

\end{document}
```

- [ ] **Step 2:** Commit（先不编译，等章节文件就绪）

```bash
git add docs/manual/user/main_user.tex
git commit -m "docs: add user volume main file"
```

---

### Task 12: 写用户卷 smoke 章节与附录

**Files:**
- Create: `docs/manual/user/chapters/00-hello.tex`
- Create: `docs/manual/user/appendices/A-stub.tex`

- [ ] **Step 1:** 写 `chapters/00-hello.tex`（验证全部宏命令）

```latex
% docs/manual/user/chapters/00-hello.tex
% 占位 smoke 章节；正式撰写时由 01-overview.tex 等替换

\chapter{Hello, \openbench{}}

\section{这是测试章节}

本章存在的唯一目的是验证 LaTeX 工具链：
xelatex 中文渲染、minted 代码高亮、tcolorbox 标注框、自定义宏命令。

\section{命令测试}

调用 \cli{openbench run config.yaml} 启动评估，
其中 \yamlkey{project.years} 指定评估时段，
\var{Evapotranspiration} 是常见评估变量；
内部由 \modname{openbench.core.evaluation} 模块负责，
读取的参考数据集是 \dataset{GLEAM\_v4.2a}。

\section{标注框测试}

\begin{noteBox}
这是一个 Note 标注框：用于普通信息提示。
\end{noteBox}

\begin{warnBox}
这是一个 Warning 标注框：用于陷阱与已知问题。
\end{warnBox}

\begin{tipBox}
这是一个 Tip 标注框：用于最佳实践与省力技巧。
\end{tipBox}

\begin{exampleBox}[YAML 配置示例]
最小可运行配置：
\begin{minted}{yaml}
project:
  name: hello_world
  output_dir: ./output
  years: [2010, 2014]

evaluation:
  variables:
    - Evapotranspiration

reference:
  Evapotranspiration: GLEAM_v4.2a
\end{minted}
\end{exampleBox}

\section{表格测试}

\begin{table}[h]
  \centering
  \caption{占位表格}
  \begin{tabular}{l l l}
    \toprule
    字段 & 类型 & 默认值 \\
    \midrule
    \yamlkey{project.name} & string & --- \\
    \yamlkey{project.years} & list[int] & --- \\
    \yamlkey{project.tim\_res} & enum & Month \\
    \bottomrule
  \end{tabular}
\end{table}
```

- [ ] **Step 2:** 写 `appendices/A-stub.tex`（占位附录）

```latex
% docs/manual/user/appendices/A-stub.tex
% 占位附录；正式撰写时由 A-config-reference.tex 等替换

\chapter{占位附录}

本附录是基础设施 plan 的占位文件，正式手册撰写时由 \file{A-config-reference.tex} 等真实参考章节替换。

\noteinline{当前用于验证 \cli{\textbackslash{}appendix} 与 \cli{\textbackslash{}part\{参考\}} 协作的 LaTeX 编号是否正确。}

\begin{longtable}{l l p{6cm}}
  \toprule
  字段 & 类型 & 说明 \\
  \midrule
  \endhead
  \yamlkey{project.name} & string & 项目名称（占位） \\
  \yamlkey{project.years} & list[int] & 评估时段（占位） \\
  \yamlkey{evaluation.variables} & list[str] & 评估变量列表（占位） \\
  \bottomrule
\end{longtable}
```

- [ ] **Step 3:** 编译用户卷

```bash
cd docs/manual && make user
```

预期：
- 退出码 0
- 生成 `user/main_user.pdf`
- `user/main_user.log` 中无 `! ` 开头的 fatal 错误
- 控制台无 `Undefined reference` / `Citation undefined`

如果失败，根据错误信息排查（最常见：minted 未启用 shell-escape、ctex 字体问题、命令拼写）。

- [ ] **Step 4:** 人工核对 PDF

```bash
open docs/manual/user/main_user.pdf  # macOS
```

核对清单：
- [ ] 中文正常显示
- [ ] 代码块带语法高亮
- [ ] 三种标注框颜色正确（蓝/红/绿）+ Example 框（橙）
- [ ] 表格用 booktabs 三横线
- [ ] PDF 书签可展开（用户指南 → 教程 / 参考）

- [ ] **Step 5:** Commit

```bash
git add docs/manual/user/chapters/00-hello.tex docs/manual/user/appendices/A-stub.tex
git commit -m "docs: add user volume smoke chapter and appendix"
```

---

### Task 13: 写开发卷 main_developer.tex 与 smoke 文件

**Files:**
- Create: `docs/manual/developer/main_developer.tex`
- Create: `docs/manual/developer/chapters/00-hello.tex`
- Create: `docs/manual/developer/appendices/A-stub.tex`

- [ ] **Step 1:** 写 `main_developer.tex`

```latex
% docs/manual/developer/main_developer.tex
\documentclass[../manual]{subfiles}

\begin{document}

\ifSubfilesClassLoaded{%
  \frontmatter
  \title{\openbench{} 开发者指南\\[1ex]\large Developer Guide}
  \author{OpenBench 项目组}
  \date{\today}
  \maketitle
  \tableofcontents
  \mainmatter
}{%
  \volumedivider{第二卷}{开发者指南}
}

\volpart{教程}
\include{chapters/00-hello}

\appendix
\volpart{参考}
\include{appendices/A-stub}

\end{document}
```

- [ ] **Step 2:** 写 `chapters/00-hello.tex`（精简版 smoke）

```latex
% docs/manual/developer/chapters/00-hello.tex
\chapter{Hello, Developer}

\section{占位章节}

本章为开发卷基础设施验收占位，正式撰写时由 \file{01-architecture.tex} 替换。

\noteinline{后续 plan 将填充：架构总览、开发环境、各子系统、扩展点、测试、贡献流程。}

\begin{minted}{python}
from openbench.config import load_config
from openbench.core.evaluation import run_evaluation

cfg = load_config("openbench.yaml")
run_evaluation(cfg)
\end{minted}
```

- [ ] **Step 3:** 写 `appendices/A-stub.tex`

```latex
% docs/manual/developer/appendices/A-stub.tex
\chapter{占位附录}

开发卷的真实附录由后续 plan 填充：包结构与依赖图、Public API、Registry schema、内部 interfaces、CONVENTIONS。
```

- [ ] **Step 4:** 编译开发卷

```bash
cd docs/manual && make dev
```

预期：生成 `developer/main_developer.pdf`。

- [ ] **Step 5:** Commit

```bash
git add docs/manual/developer/
git commit -m "docs: add developer volume scaffolding"
```

---

### Task 14: 写运维卷 main_operations.tex 与 smoke 文件

**Files:**
- Create: `docs/manual/operations/main_operations.tex`
- Create: `docs/manual/operations/chapters/00-hello.tex`
- Create: `docs/manual/operations/appendices/A-stub.tex`

- [ ] **Step 1:** 写 `main_operations.tex`

```latex
% docs/manual/operations/main_operations.tex
\documentclass[../manual]{subfiles}

\begin{document}

\ifSubfilesClassLoaded{%
  \frontmatter
  \title{\openbench{} 运维指南\\[1ex]\large Operations Guide}
  \author{OpenBench 项目组}
  \date{\today}
  \maketitle
  \tableofcontents
  \mainmatter
}{%
  \volumedivider{第三卷}{运维指南}
}

\volpart{教程}
\include{chapters/00-hello}

\appendix
\volpart{参考}
\include{appendices/A-stub}

\end{document}
```

- [ ] **Step 2:** 写 `chapters/00-hello.tex`

```latex
% docs/manual/operations/chapters/00-hello.tex
\chapter{Hello, Ops}

\section{占位章节}

本章为运维卷基础设施验收占位，正式撰写时由 \file{01-deployment.tex} 替换。

\noteinline{后续 plan 将填充：部署形态、HPC 安装、SSH 远程、性能调优、缓存策略、站点评估、监控、故障恢复。}

\begin{minted}{bash}
# 示例：HPC 上启动远程评估
openbench run config.yaml --remote-profile hpc01
\end{minted}
```

- [ ] **Step 3:** 写 `appendices/A-stub.tex`

```latex
% docs/manual/operations/appendices/A-stub.tex
\chapter{占位附录}

运维卷的真实附录由后续 plan 填充：Runtime 选项、SSH 配置、缓存目录布局、性能建议、日志索引。
```

- [ ] **Step 4:** 编译运维卷

```bash
cd docs/manual && make ops
```

预期：生成 `operations/main_operations.pdf`。

- [ ] **Step 5:** Commit

```bash
git add docs/manual/operations/
git commit -m "docs: add operations volume scaffolding"
```

---

## Phase 4: 总卷与端到端验收

### Task 15: 写总卷 manual.tex

**Files:**
- Create: `docs/manual/manual.tex`

- [ ] **Step 1:** 写 `manual.tex`

```latex
% docs/manual/manual.tex
% 总卷：使用 subfiles 合并三卷为单一 PDF

\documentclass{ctexbook}

% 共享配置
\input{common/preamble}
\input{common/styles}
\input{common/macros}

% PDF 元数据
\hypersetup{
  pdftitle={OpenBench 用户手册（合订总卷）},
  pdfauthor={OpenBench 项目组},
}

\title{\openbench{} 用户手册\\[1.5ex]\large 合订总卷}
\author{OpenBench 项目组}
\date{\today}

\begin{document}

\frontmatter
\maketitle
\tableofcontents

\mainmatter

\subfile{user/main_user}
\subfile{developer/main_developer}
\subfile{operations/main_operations}

\backmatter
% 后续可加 \printindex 或参考文献
% \bibliographystyle{plain}
% \bibliography{common/bibliography}

\end{document}
```

- [ ] **Step 2:** 编译总卷

```bash
cd docs/manual && make master
```

预期：生成 `manual.pdf`，三卷内容依次出现，目录中可见三个 \\part 层级。

- [ ] **Step 3:** 人工核对

```bash
open docs/manual/manual.pdf  # macOS
```

核对清单：
- [ ] 总目录最前面是三大卷分隔页（"第一卷 ─ 用户指南" / "第二卷 ─ 开发者指南" / "第三卷 ─ 运维指南"，由 `\volumedivider` 渲染，TOC 中以 chapter 级别条目出现）
- [ ] 每卷下有"教程"与"参考"两条不编号 part 条目（由 `\volpart` 渲染，避免跨卷 part 编号混乱）
- [ ] 章节本身编号在 book class 下沿用 1, 2, 3...；附录沿用 A, B, C...（`\appendix` 命令重置）
- [ ] PDF 书签：卷 → part → chapter 三层都可展开

- [ ] **Step 4:** Commit

```bash
git add docs/manual/manual.tex
git commit -m "docs: add master manual.tex combining three volumes"
```

---

### Task 16: 端到端 `make all` 验收

**Files:** 仅运行命令。

- [ ] **Step 1:** 清理后从零跑 `make all`

```bash
cd docs/manual && make clean && make all
```

预期退出码 0，编译总用时 < 3 分钟（首次运行 minted 缓存为空）。

- [ ] **Step 2:** 验证 4 个 PDF 都已产出

```bash
ls -la docs/manual/manual.pdf docs/manual/user/main_user.pdf \
       docs/manual/developer/main_developer.pdf \
       docs/manual/operations/main_operations.pdf
```

预期：四个 PDF 都存在，文件大小 > 0。

- [ ] **Step 3:** 检查所有 .log 没有 LaTeX fatal error

```bash
for f in docs/manual/manual.log \
         docs/manual/user/main_user.log \
         docs/manual/developer/main_developer.log \
         docs/manual/operations/main_operations.log; do
  echo "=== $f ==="
  grep -E "^\!|Undefined reference|Citation undefined" "$f" || echo "(clean)"
done
```

预期：每个文件都打印 `(clean)`。如有非 `(clean)` 输出，停下来排查。

- [ ] **Step 4:** 检查 overfull/underfull box 数量 ≤ 5/卷（spec §11 验收标准）

```bash
for f in docs/manual/manual.log \
         docs/manual/user/main_user.log \
         docs/manual/developer/main_developer.log \
         docs/manual/operations/main_operations.log; do
  echo "=== $f ==="
  grep -c -E "Overfull|Underfull" "$f" || echo "0"
done
```

预期：每文件 ≤ 5。

- [ ] **Step 5:** Commit 收尾（如有遗留改动）

```bash
git status
# 如有未提交的中间产物（不应该有，.gitignore 已覆盖）
```

如果一切干净，本 plan 完成。

---

## 自审清单（执行前再核一次）

- [ ] **Spec 覆盖**：spec §3 目录布局每个文件在本计划中都创建了占位（除 `_generated/*.tex` 由后续 plan 生成、`bibliography.bib` 已含骨架）
- [ ] **占位词扫描**：本计划无 `TBD` / `TODO` / `implement later`；所有 LaTeX 代码都是完整可编译的
- [ ] **类型一致性**：`\subfile{path}` 路径与目录布局一致；subfiles 类参数 `[../manual]` 在 user/dev/ops 三处都相同
- [ ] **新风险**：subfiles 在 ctexbook 上的兼容性已知良好；如 Task 12 编译失败，回退方案是改用 `\include` + 在 manual.tex 复制三卷的内容（不优雅但可行）—— 不预先 plan，遇到再改

---

## 完成后的状态

执行完本 plan 之后：

- `docs/manual/` 目录下有完整构建脚手架
- 4 个 PDF 都能编译产出（含占位内容）
- Bug 日志已就绪
- 后续 plan（生成器、三卷内容）可以并行/串行展开

下一份建议的 plan：**`2026-04-30-manual-generators.md`** —— 实现 spec §5 的 5 个自动生成器（config_schema / reference_table / model_table / registry_schema / internal_interfaces），把附录从手抄转为代码驱动。
