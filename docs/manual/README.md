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
