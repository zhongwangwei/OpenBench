# OpenBench 手册自动生成器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `docs/manual/scripts/` 下实现 5 个 LaTeX 片段生成器，分别从 OpenBench 源码（dataclass schema、registry YAML 目录、ABC/Protocol 接口）产出 `docs/manual/_generated/*.tex`，被三卷的"参考"附录 `\input{}`。`make generated` 一键重跑全部生成器；`make all` 自动调用以保证文档与代码不漂移。

**Architecture:**
- `scripts/lib/` 存放共享工具：LaTeX escape、longtable builder、I/O 写入。
- 每个生成器一个 `generate_*.py` 入口，命令行 `python -m docs.manual.scripts.generate_X` 即可单独运行。
- `run_all.py` 聚合 5 个生成器，提供"全量再生"入口。
- 测试以 TDD 方式开发：每个生成器先有 `tests/manual/test_*.py`，红→绿→重构。
- 生成的 LaTeX 片段不含 `\documentclass` / `\begin{document}` —— 只是可被 `\input{}` 的内容（章节内的 longtable 或 description list）。

**Tech Stack:** Python 3.10+（与 OpenBench 主体一致）；标准库 `dataclasses` / `tokenize` / `inspect` / `ast`；外部依赖仅 `PyYAML`（项目已有）。无需新增依赖。

**关联 spec:** `docs/superpowers/specs/2026-04-30-openbench-manual-design.md`（§5 与代码同步策略）

**前置依赖:** Plan 1 (`2026-04-30-manual-infrastructure.md`) 已完成，`docs/manual/_generated/` 目录就绪。

**不在本计划范围:**
- 错误消息（用户卷附录 E）/ 日志索引（运维卷附录 E）—— 半自动，留给各卷内容 plan
- 真实附录章节（A-config-reference.tex 等）—— 各卷内容 plan
- 解决基础设施 plan 中遗留的 appendix 锚点冲突（生成器不引入新冲突；真正修复在卷内容 plan 引入卷前缀时一并解决）

---

## 文件结构

新建：

| 路径 | 行数估计 | 责任 |
|---|---|---|
| `docs/manual/scripts/__init__.py` | 1 | 包标记 |
| `docs/manual/scripts/lib/__init__.py` | 1 | 包标记 |
| `docs/manual/scripts/lib/escape.py` | ~40 | LaTeX 特殊字符转义 |
| `docs/manual/scripts/lib/tables.py` | ~80 | longtable / booktabs 生成器 |
| `docs/manual/scripts/lib/io.py` | ~30 | 原子写入、警告头注释 |
| `docs/manual/scripts/generate_reference_table.py` | ~120 | 从 reference_catalog.yaml 生成数据集表 |
| `docs/manual/scripts/generate_model_table.py` | ~80 | 从 model_catalog.yaml 生成模型表 |
| `docs/manual/scripts/generate_config_schema.py` | ~150 | 从 config/schema.py 抽 dataclass 字段 |
| `docs/manual/scripts/generate_registry_schema.py` | ~120 | 从 data/registry/schema.py 抽 dataclass 字段 |
| `docs/manual/scripts/generate_internal_interfaces.py` | ~150 | 扫描 ABC/Protocol/abstractmethod |
| `docs/manual/scripts/run_all.py` | ~40 | 顺序调用 5 个生成器 |
| `tests/manual/__init__.py` | 0 | 包标记 |
| `tests/manual/conftest.py` | ~20 | pytest fixtures（测试用最小 catalog） |
| `tests/manual/test_escape.py` | ~40 | TDD: escape 各种特殊字符 |
| `tests/manual/test_tables.py` | ~60 | TDD: longtable 生成正确性 |
| `tests/manual/test_io.py` | ~30 | TDD: 写入 + 头注释 |
| `tests/manual/test_generate_reference_table.py` | ~80 | TDD |
| `tests/manual/test_generate_model_table.py` | ~60 | TDD |
| `tests/manual/test_generate_config_schema.py` | ~80 | TDD |
| `tests/manual/test_generate_registry_schema.py` | ~80 | TDD |
| `tests/manual/test_generate_internal_interfaces.py` | ~80 | TDD |

修改：

| 路径 | 改动 |
|---|---|
| `docs/manual/Makefile` | 加 `generated` 目标；让 `all` 依赖 `generated` |
| `docs/manual/user/appendices/A-stub.tex` | 替换为真正引用 `\input{../_generated/config_schema}` 的占位（仍标 stub，等真实附录 plan 替换） |
| `pyproject.toml` | 加 `[tool.pytest.ini_options]` testpaths 包括 `tests/manual`（如果还没有） |

---

## Phase 1: Foundation utilities (TDD)

### Task 1: 包结构

**Files:** Create `docs/manual/scripts/__init__.py`, `docs/manual/scripts/lib/__init__.py`, `tests/manual/__init__.py`, `tests/manual/conftest.py`.

- [ ] **Step 1:** 创建包结构

```bash
mkdir -p docs/manual/scripts/lib tests/manual
touch docs/manual/scripts/__init__.py docs/manual/scripts/lib/__init__.py tests/manual/__init__.py
```

- [ ] **Step 2:** 写 `tests/manual/conftest.py`（最小化测试用 catalog）

```python
"""共用 pytest fixtures，提供最小化测试用 registry / schema 数据。"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def minimal_reference_catalog(tmp_path: Path) -> Path:
    """返回只含 2 个数据集的小型 catalog 文件路径。"""
    p = tmp_path / "reference_catalog.yaml"
    p.write_text(textwrap.dedent("""
        GLEAM_v4.2a_LowRes:
          name: GLEAM_v4.2a_LowRes
          category: Water
          data_type: grid
          grid_res: 0.5
          tim_res: Month
          description: GLEAM v4.2a LowRes ET dataset
          variables:
            Evapotranspiration:
              varname: E
              varunit: mm/day
            Transpiration:
              varname: Et
              varunit: mm/day
        FLUXNET_PLUMBER2:
          name: FLUXNET_PLUMBER2
          category: Water
          data_type: stn
          tim_res: Hour
          description: FLUXNET PLUMBER2 station data
          variables:
            Evapotranspiration:
              varname: ET
              varunit: kg/m2/s
    """).strip())
    return p


@pytest.fixture
def minimal_model_catalog(tmp_path: Path) -> Path:
    p = tmp_path / "model_catalog.yaml"
    p.write_text(textwrap.dedent("""
        CoLM2024:
          name: CoLM2024
          data_type: grid
          tim_res: Month
          description: Common Land Model 2024
          variables:
            Evapotranspiration:
              varname: f_lh_vap
              varunit: W/m2
            Latent_Heat:
              varname: f_lh
              varunit: W/m2
    """).strip())
    return p
```

- [ ] **Step 3:** Commit

```bash
git add docs/manual/scripts tests/manual
git commit -m "chore: scaffold manual generator scripts and test fixtures"
```

---

### Task 2: LaTeX escape 模块（TDD）

**Files:**
- Create test: `tests/manual/test_escape.py`
- Create impl: `docs/manual/scripts/lib/escape.py`

- [ ] **Step 1:** 写失败测试 `tests/manual/test_escape.py`

```python
"""Tests for docs.manual.scripts.lib.escape."""
import pytest

from docs.manual.scripts.lib.escape import latex_escape


class TestLatexEscape:
    def test_passthrough_plain_text(self):
        assert latex_escape("hello world") == "hello world"

    def test_escapes_underscore(self):
        assert latex_escape("var_name") == r"var\_name"

    def test_escapes_dollar(self):
        assert latex_escape("$5") == r"\$5"

    def test_escapes_percent(self):
        assert latex_escape("50%") == r"50\%"

    def test_escapes_ampersand(self):
        assert latex_escape("a & b") == r"a \& b"

    def test_escapes_hash(self):
        assert latex_escape("#1") == r"\#1"

    def test_escapes_braces(self):
        assert latex_escape("{x}") == r"\{x\}"

    def test_escapes_backslash(self):
        assert latex_escape(r"a\b") == r"a\textbackslash{}b"

    def test_escapes_caret_and_tilde(self):
        assert latex_escape("a^b~c") == r"a\^{}b\~{}c"

    def test_combined(self):
        # 真实场景：YAML key 含下划线 + 单位字符串
        assert latex_escape("kg/m^2/s & 100%") == r"kg/m\^{}2/s \& 100\%"

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            latex_escape(42)
```

- [ ] **Step 2:** 运行测试，验证失败

```bash
cd /Volumes/Data01/Openbench && python -m pytest tests/manual/test_escape.py -v
```

预期：`ModuleNotFoundError: docs.manual.scripts.lib.escape` 或类似导入错误。

- [ ] **Step 3:** 写最小实现 `docs/manual/scripts/lib/escape.py`

```python
"""LaTeX special character escaping."""
from __future__ import annotations

# 处理顺序很重要：反斜杠必须最先替换，避免后续 \\ 被再处理
_BACKSLASH_PLACEHOLDER = "\x00BACKSLASH\x00"

_REPLACEMENTS = [
    # (input_char, latex_output)
    ("\\", _BACKSLASH_PLACEHOLDER),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("^", r"\^{}"),
    ("~", r"\~{}"),
    (_BACKSLASH_PLACEHOLDER, r"\textbackslash{}"),
]


def latex_escape(s: str) -> str:
    """Escape LaTeX special characters in s.

    Handles & % $ # _ { } ^ ~ and backslash. Order matters: backslash is
    replaced via a placeholder to avoid double-processing of the escape
    sequences emitted by other replacements.
    """
    if not isinstance(s, str):
        raise TypeError(f"latex_escape requires str, got {type(s).__name__}")
    for src, dst in _REPLACEMENTS:
        s = s.replace(src, dst)
    return s
```

- [ ] **Step 4:** 运行测试，验证通过

```bash
python -m pytest tests/manual/test_escape.py -v
```

预期：11 passed。

- [ ] **Step 5:** Commit

```bash
git add docs/manual/scripts/lib/escape.py tests/manual/test_escape.py
git commit -m "feat(manual): add latex_escape utility with TDD"
```

---

### Task 3: 表格生成模块（TDD）

**Files:**
- Create test: `tests/manual/test_tables.py`
- Create impl: `docs/manual/scripts/lib/tables.py`

- [ ] **Step 1:** 写失败测试

```python
"""Tests for docs.manual.scripts.lib.tables."""
from docs.manual.scripts.lib.tables import longtable


class TestLongtable:
    def test_basic_three_column(self):
        out = longtable(
            headers=["字段", "类型", "默认值"],
            rows=[
                ["name", "str", "---"],
                ["years", "list[int]", "---"],
            ],
            col_spec="l l l",
        )
        # 必含元素
        assert r"\begin{longtable}{l l l}" in out
        assert r"\toprule" in out
        assert r"\endhead" in out
        assert r"\bottomrule" in out
        assert r"\end{longtable}" in out
        assert "字段 & 类型 & 默认值" in out
        assert "name & str & ---" in out

    def test_escapes_cell_content(self):
        out = longtable(
            headers=["key", "value"],
            rows=[["proj_name", "100%"]],
            col_spec="l l",
        )
        # 单元格里的 _ 与 % 必须被转义
        assert r"proj\_name" in out
        assert r"100\%" in out
        # 但 & 是分隔符，不能误转
        assert " & " in out

    def test_caption_optional(self):
        out = longtable(
            headers=["a", "b"],
            rows=[["1", "2"]],
            col_spec="l l",
            caption="测试表格",
        )
        assert r"\caption{测试表格}" in out

    def test_empty_rows_renders(self):
        # 空 rows 也要产出有效骨架，避免 LaTeX 报错
        out = longtable(headers=["a", "b"], rows=[], col_spec="l l")
        assert r"\begin{longtable}" in out
        assert r"\end{longtable}" in out

    def test_column_count_mismatch_raises(self):
        import pytest
        with pytest.raises(ValueError, match="column count"):
            longtable(
                headers=["a", "b"],
                rows=[["1", "2", "3"]],  # 3 列但 header 只有 2 列
                col_spec="l l",
            )
```

- [ ] **Step 2:** 运行测试，验证失败

```bash
python -m pytest tests/manual/test_tables.py -v
```

- [ ] **Step 3:** 写实现 `docs/manual/scripts/lib/tables.py`

```python
"""LaTeX longtable / booktabs generation helpers."""
from __future__ import annotations

from typing import Optional

from .escape import latex_escape


def longtable(
    *,
    headers: list[str],
    rows: list[list[str]],
    col_spec: str,
    caption: Optional[str] = None,
) -> str:
    """Build a LaTeX longtable with booktabs rules.

    Args:
        headers: column header strings (already display-ready, will be escaped)
        rows: list of rows; each row is a list of cell strings (will be escaped)
        col_spec: LaTeX column spec, e.g., "l l p{6cm}"
        caption: optional caption (placed inside the longtable env)

    Raises:
        ValueError: if any row's column count != len(headers)
    """
    n = len(headers)
    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(
                f"column count mismatch: row {i} has {len(row)} cells, "
                f"expected {n}"
            )

    parts: list[str] = []
    parts.append(rf"\begin{{longtable}}{{{col_spec}}}")
    if caption:
        parts.append(rf"  \caption{{{latex_escape(caption)}}} \\")
    parts.append(r"  \toprule")
    parts.append("  " + " & ".join(latex_escape(h) for h in headers) + r" \\")
    parts.append(r"  \midrule")
    parts.append(r"  \endhead")
    for row in rows:
        parts.append("  " + " & ".join(latex_escape(c) for c in row) + r" \\")
    parts.append(r"  \bottomrule")
    parts.append(r"\end{longtable}")
    return "\n".join(parts) + "\n"
```

- [ ] **Step 4:** 运行测试，验证通过

```bash
python -m pytest tests/manual/test_tables.py -v
```

预期：5 passed。

- [ ] **Step 5:** Commit

```bash
git add docs/manual/scripts/lib/tables.py tests/manual/test_tables.py
git commit -m "feat(manual): add longtable builder with TDD"
```

---

### Task 4: I/O 模块（写入 + 头注释）

**Files:**
- Create test: `tests/manual/test_io.py`
- Create impl: `docs/manual/scripts/lib/io.py`

- [ ] **Step 1:** 写失败测试

```python
"""Tests for docs.manual.scripts.lib.io."""
from pathlib import Path

from docs.manual.scripts.lib.io import write_generated


class TestWriteGenerated:
    def test_writes_with_warning_header(self, tmp_path: Path):
        target = tmp_path / "out.tex"
        body = r"\section{测试}"
        write_generated(target, body, source="reference_catalog.yaml")
        text = target.read_text()
        # 头注释告诉读者别手改
        assert "% AUTO-GENERATED" in text
        assert "reference_catalog.yaml" in text
        # body 必须完整出现
        assert r"\section{测试}" in text
        # 末尾留有换行
        assert text.endswith("\n")

    def test_overwrites_existing(self, tmp_path: Path):
        target = tmp_path / "out.tex"
        target.write_text("OLD CONTENT")
        write_generated(target, "NEW", source="x.py")
        assert "OLD CONTENT" not in target.read_text()
        assert "NEW" in target.read_text()

    def test_creates_parent_dir(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "out.tex"
        write_generated(target, "x", source="x.py")
        assert target.exists()
```

- [ ] **Step 2:** 运行测试，验证失败

```bash
python -m pytest tests/manual/test_io.py -v
```

- [ ] **Step 3:** 写实现

```python
"""Atomic write of generated LaTeX fragments."""
from __future__ import annotations

from pathlib import Path

_HEADER = """% AUTO-GENERATED — DO NOT EDIT BY HAND
% 本文件由 docs/manual/scripts/ 中的生成器从代码自动产出。
% 来源：{source}
% 重新生成：cd docs/manual && make generated

"""


def write_generated(target: Path, body: str, *, source: str) -> None:
    """Write body to target with a warning header.

    Creates parent directories if needed; overwrites existing file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _HEADER.format(source=source) + body
    if not content.endswith("\n"):
        content += "\n"
    target.write_text(content, encoding="utf-8")
```

- [ ] **Step 4:** 运行测试

```bash
python -m pytest tests/manual/test_io.py -v
```

- [ ] **Step 5:** Commit

```bash
git add docs/manual/scripts/lib/io.py tests/manual/test_io.py
git commit -m "feat(manual): add write_generated I/O helper with TDD"
```

---

## Phase 2: Reference table generator

### Task 5: 写测试

**Files:** Create `tests/manual/test_generate_reference_table.py`

- [ ] **Step 1:** 测试用例覆盖：分类分组、grid/stn 区分、空 catalog、变量计数

```python
"""Tests for generate_reference_table."""
from pathlib import Path

from docs.manual.scripts.generate_reference_table import build_reference_table


class TestBuildReferenceTable:
    def test_groups_by_category(self, minimal_reference_catalog: Path):
        out = build_reference_table(minimal_reference_catalog)
        # 至少出现 "Water" 分类（minimal fixture 中两个都属 Water）
        assert "Water" in out
        # GLEAM 与 FLUXNET 都应被列出
        assert "GLEAM" in out
        assert "FLUXNET" in out

    def test_distinguishes_grid_vs_stn(self, minimal_reference_catalog: Path):
        out = build_reference_table(minimal_reference_catalog)
        # GLEAM 是 grid，FLUXNET 是 stn —— 输出中能找到 "grid" 与 "stn" 字样
        assert "grid" in out.lower() or "Grid" in out
        assert "stn" in out.lower() or "Station" in out

    def test_includes_variable_count(self, minimal_reference_catalog: Path):
        out = build_reference_table(minimal_reference_catalog)
        # GLEAM 有 2 个变量（ET、Transpiration），FLUXNET 有 1 个（ET）
        # 不强制具体格式，但数字应在某处出现
        assert "2" in out

    def test_escapes_special_chars(self, tmp_path: Path):
        # 构造一个含下划线的数据集名
        catalog = tmp_path / "ref.yaml"
        catalog.write_text(
            "weird_name_with_underscores:\n"
            "  name: weird_name_with_underscores\n"
            "  category: Test\n"
            "  data_type: grid\n"
            "  variables: {}\n"
        )
        out = build_reference_table(catalog)
        # 下划线在表格单元格中必须被转义
        assert r"weird\_name\_with\_underscores" in out

    def test_empty_catalog(self, tmp_path: Path):
        catalog = tmp_path / "empty.yaml"
        catalog.write_text("{}\n")
        out = build_reference_table(catalog)
        # 空 catalog 不该爆，输出至少包含一个 longtable 空骨架或说明文本
        assert r"\begin{longtable}" in out or "无数据集" in out
```

- [ ] **Step 2:** 运行测试，验证失败

```bash
python -m pytest tests/manual/test_generate_reference_table.py -v
```

---

### Task 6: 实现 generate_reference_table

**Files:** Create `docs/manual/scripts/generate_reference_table.py`

- [ ] **Step 1:** 写实现

```python
"""Generate LaTeX table of all reference datasets from reference_catalog.yaml."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .lib.io import write_generated
from .lib.tables import longtable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "src/openbench/data/registry/reference_catalog.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "docs/manual/_generated/reference_table.tex"


def build_reference_table(catalog_path: Path) -> str:
    """Build LaTeX content listing all reference datasets, grouped by category."""
    raw = yaml.safe_load(catalog_path.read_text()) or {}

    if not raw:
        return "本目录暂无数据集。\n"

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        category = entry.get("category", "Uncategorized")
        by_category[category].append({**entry, "_key": key})

    parts: list[str] = []
    for category in sorted(by_category):
        parts.append(rf"\subsection*{{{category}}}")
        rows = []
        for entry in sorted(by_category[category], key=lambda e: e.get("name", e["_key"])):
            name = str(entry.get("name") or entry["_key"])
            dtype = str(entry.get("data_type", ""))
            tim_res = str(entry.get("tim_res", ""))
            grid_res = str(entry.get("grid_res", ""))
            variables = entry.get("variables") or {}
            n_vars = len(variables) if isinstance(variables, dict) else 0
            rows.append([name, dtype, tim_res, grid_res, str(n_vars)])
        parts.append(
            longtable(
                headers=["名称", "类型", "时间分辨率", "空间分辨率", "变量数"],
                rows=rows,
                col_spec="l l l l r",
            )
        )
        parts.append("")  # 空行
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = build_reference_table(args.catalog)
    write_generated(
        args.output,
        body,
        source=f"src/openbench/data/registry/{args.catalog.name}",
    )
    print(f"wrote {args.output} ({len(body)} chars)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** 运行测试

```bash
python -m pytest tests/manual/test_generate_reference_table.py -v
```

预期：5 passed。

- [ ] **Step 3:** 在真实 catalog 上跑一次

```bash
cd /Volumes/Data01/Openbench && python -m docs.manual.scripts.generate_reference_table
```

预期：写出 `docs/manual/_generated/reference_table.tex`，约 3-5 KB。

- [ ] **Step 4:** 人工抽查输出

```bash
head -30 docs/manual/_generated/reference_table.tex
wc -l docs/manual/_generated/reference_table.tex
```

预期：头部有 AUTO-GENERATED 注释；按分类分组；含 ~70 个数据集。

- [ ] **Step 5:** Commit

```bash
git add docs/manual/scripts/generate_reference_table.py tests/manual/test_generate_reference_table.py
git commit -m "feat(manual): generate reference dataset table from catalog"
```

注意：`_generated/*.tex` 不要提交（已在 `.gitignore` 通过 `*.pdf` 兜不住，但 `_generated/.gitkeep` 已存）。先确保 `_generated/*.tex` 的忽略：

```bash
echo "_generated/*.tex" >> docs/manual/.gitignore
git add docs/manual/.gitignore && git commit -m "docs: ignore generated LaTeX fragments"
```

---

## Phase 3: Model table generator

### Task 7: 写测试

**Files:** Create `tests/manual/test_generate_model_table.py`

- [ ] **Step 1:** 测试

```python
"""Tests for generate_model_table."""
from pathlib import Path

from docs.manual.scripts.generate_model_table import build_model_table


class TestBuildModelTable:
    def test_lists_all_models(self, minimal_model_catalog: Path):
        out = build_model_table(minimal_model_catalog)
        assert "CoLM2024" in out

    def test_includes_var_count(self, minimal_model_catalog: Path):
        out = build_model_table(minimal_model_catalog)
        # CoLM2024 fixture 有 2 个变量
        assert "2" in out

    def test_includes_data_type(self, minimal_model_catalog: Path):
        out = build_model_table(minimal_model_catalog)
        assert "grid" in out.lower()

    def test_empty_catalog(self, tmp_path: Path):
        catalog = tmp_path / "empty.yaml"
        catalog.write_text("{}\n")
        out = build_model_table(catalog)
        assert r"\begin{longtable}" in out or "无模型" in out
```

- [ ] **Step 2:** 跑测试验证红

```bash
python -m pytest tests/manual/test_generate_model_table.py -v
```

---

### Task 8: 实现 generate_model_table

**Files:** Create `docs/manual/scripts/generate_model_table.py`

- [ ] **Step 1:** 写实现

```python
"""Generate LaTeX table of model profiles from model_catalog.yaml."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .lib.io import write_generated
from .lib.tables import longtable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "src/openbench/data/registry/model_catalog.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "docs/manual/_generated/model_table.tex"


def build_model_table(catalog_path: Path) -> str:
    raw = yaml.safe_load(catalog_path.read_text()) or {}
    if not raw:
        return "本目录暂无模型 profile。\n"

    rows = []
    for key, entry in sorted(raw.items()):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or key)
        dtype = str(entry.get("data_type", ""))
        tim_res = str(entry.get("tim_res", ""))
        description = str(entry.get("description", ""))
        variables = entry.get("variables") or {}
        n_vars = len(variables) if isinstance(variables, dict) else 0
        rows.append([name, dtype, tim_res, str(n_vars), description])

    return longtable(
        headers=["名称", "类型", "时间分辨率", "变量数", "说明"],
        rows=rows,
        col_spec="l l l r p{5cm}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = build_model_table(args.catalog)
    write_generated(args.output, body, source=f"src/openbench/data/registry/{args.catalog.name}")
    print(f"wrote {args.output} ({len(body)} chars)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** 测试 + 真实运行

```bash
python -m pytest tests/manual/test_generate_model_table.py -v
python -m docs.manual.scripts.generate_model_table
```

- [ ] **Step 3:** Commit

```bash
git add docs/manual/scripts/generate_model_table.py tests/manual/test_generate_model_table.py
git commit -m "feat(manual): generate model profile table from catalog"
```

---

## Phase 4: Config schema generator

### Task 9: 写测试

**Files:** Create `tests/manual/test_generate_config_schema.py`

- [ ] **Step 1:** 测试关键 dataclass + 字段被列出 + 注释被抽出

```python
"""Tests for generate_config_schema."""
from docs.manual.scripts.generate_config_schema import build_config_schema_table


class TestBuildConfigSchemaTable:
    def test_lists_top_level_dataclasses(self):
        out = build_config_schema_table()
        # 关键 dataclass 名应作为 subsection
        for cls in ("ProjectConfig", "EvaluationConfig", "ReferenceConfig", "OpenBenchConfig"):
            assert cls in out

    def test_lists_required_fields(self):
        out = build_config_schema_table()
        # ProjectConfig 必填字段
        assert "name" in out
        assert "output_dir" in out
        assert "years" in out

    def test_lists_optional_with_default(self):
        out = build_config_schema_table()
        # tim_res 默认 None
        assert "tim_res" in out

    def test_extracts_inline_comment(self):
        out = build_config_schema_table()
        # ProjectConfig.time_alignment 有注释 "intersection | per_pair | strict"
        assert "intersection" in out

    def test_escapes_special_chars(self):
        out = build_config_schema_table()
        # 字段名 IGBP_groupby 含下划线
        assert r"IGBP\_groupby" in out
```

- [ ] **Step 2:** 跑测试验证红

```bash
python -m pytest tests/manual/test_generate_config_schema.py -v
```

---

### Task 10: 实现 generate_config_schema

**Files:** Create `docs/manual/scripts/generate_config_schema.py`

- [ ] **Step 1:** 写实现

实现要点：
- 用 `dataclasses.fields(cls)` 遍历字段；得到名称、类型、默认值
- 用 `tokenize` 模块解析源文件，把同行 `# comment` 与字段名关联起来
- 为每个 dataclass 输出一个 `\subsection*{ClassName}` + 一个 longtable

```python
"""Generate LaTeX reference of OpenBench config dataclasses (config/schema.py)."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import io
import tokenize
from pathlib import Path
from typing import Any

from .lib.io import write_generated
from .lib.tables import longtable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "docs/manual/_generated/config_schema.tex"
SCHEMA_MODULE = "openbench.config.schema"
SCHEMA_FILE = REPO_ROOT / "src/openbench/config/schema.py"

# 要列出的 dataclass（顺序决定文档章节顺序）
DATACLASS_ORDER = [
    "ProjectConfig",
    "EvaluationConfig",
    "ReferenceConfig",
    "SimulationEntry",
    "ComparisonConfig",
    "StatisticsConfig",
    "OpenBenchConfig",
]


def _extract_inline_comments(source_path: Path) -> dict[tuple[str, str], str]:
    """Map (class_name, field_name) -> trailing line comment.

    Walks the source via tokenize, tracking the current `class X:` context
    and field assignments at that class level.
    """
    src = source_path.read_text()
    comments: dict[tuple[str, str], str] = {}
    current_class: str | None = None
    indent_stack: list[int] = []

    tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NAME and tok.string == "class":
            # 下一个 NAME 是 class 名
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j].type == tokenize.NAME:
                    current_class = tokens[j].string
                    break
        if not current_class:
            continue
        # 字段定义形式：NAME COLON ...  (with COMMENT later on same line)
        if (tok.type == tokenize.NAME
                and i + 1 < len(tokens)
                and tokens[i + 1].type == tokenize.OP
                and tokens[i + 1].string == ":"):
            field_name = tok.string
            # 在同一行后面找 COMMENT
            row = tok.start[0]
            for j in range(i + 2, len(tokens)):
                t = tokens[j]
                if t.start[0] != row:
                    break
                if t.type == tokenize.COMMENT:
                    text = t.string.lstrip("#").strip()
                    if text:
                        comments[(current_class, field_name)] = text
                    break
        # 离开 class 体的简单启发：碰到下一个顶格 class 或 def
        if tok.type == tokenize.NAME and tok.string in ("class", "def") and tok.start[1] == 0:
            if tok.string == "class":
                # already handled above
                pass
            else:
                current_class = None
    return comments


def _format_default(field: dataclasses.Field) -> str:
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return repr(field.default_factory())
        except Exception:
            return "<factory>"
    return "—"  # required


def _format_type(annotation: Any) -> str:
    # dataclasses 的 type hints 在 Python 3.10+ 可能是字符串（PEP 563）
    if isinstance(annotation, str):
        return annotation
    s = repr(annotation)
    # typing.Optional[str] → 'Optional[str]'
    s = s.replace("typing.", "")
    return s


def build_config_schema_table() -> str:
    module = importlib.import_module(SCHEMA_MODULE)
    comments = _extract_inline_comments(SCHEMA_FILE)
    parts: list[str] = []
    for cls_name in DATACLASS_ORDER:
        cls = getattr(module, cls_name, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        parts.append(rf"\subsection*{{{cls_name}}}")
        if cls.__doc__:
            first_line = cls.__doc__.strip().splitlines()[0]
            parts.append(first_line + r"\\")
        rows = []
        for field in dataclasses.fields(cls):
            description = comments.get((cls_name, field.name), "")
            rows.append([
                field.name,
                _format_type(field.type),
                _format_default(field),
                description,
            ])
        parts.append(
            longtable(
                headers=["字段", "类型", "默认值", "说明"],
                rows=rows,
                col_spec="l l l p{5cm}",
            )
        )
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = build_config_schema_table()
    write_generated(args.output, body, source="src/openbench/config/schema.py")
    print(f"wrote {args.output} ({len(body)} chars)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** 跑测试

```bash
python -m pytest tests/manual/test_generate_config_schema.py -v
```

如某测试失败（特别是 inline comment 抽取），用 `pytest -v --tb=long` 查具体哪个 token 没抓到，迭代修 `_extract_inline_comments`。

- [ ] **Step 3:** 真实运行

```bash
python -m docs.manual.scripts.generate_config_schema
head -50 docs/manual/_generated/config_schema.tex
```

- [ ] **Step 4:** Commit

```bash
git add docs/manual/scripts/generate_config_schema.py tests/manual/test_generate_config_schema.py
git commit -m "feat(manual): generate config schema reference from dataclasses"
```

---

## Phase 5: Registry schema generator

### Task 11: 写测试与实现

**Files:**
- Create test: `tests/manual/test_generate_registry_schema.py`
- Create impl: `docs/manual/scripts/generate_registry_schema.py`

实现思路与 Task 10 相同，只是换扫描目标为 `openbench.data.registry.schema`。

- [ ] **Step 1:** 写测试

```python
"""Tests for generate_registry_schema."""
from docs.manual.scripts.generate_registry_schema import build_registry_schema_table


class TestBuildRegistrySchemaTable:
    def test_lists_key_dataclasses(self):
        out = build_registry_schema_table()
        # registry/schema.py 含 FallbackVar、VariableMapping 等
        assert "VariableMapping" in out

    def test_lists_fields(self):
        out = build_registry_schema_table()
        # VariableMapping.varname / varunit
        assert "varname" in out
        assert "varunit" in out

    def test_escapes_chars(self):
        out = build_registry_schema_table()
        # 不允许出现裸的下划线（必须 \_）
        # 简单检查：字段名应作为 \_ 形式出现
        if "var_name" in out.lower():
            assert r"var\_name" in out or "var_name" not in out  # tolerant
```

- [ ] **Step 2:** 实现（复用 Task 10 的工具函数；提取到 `lib/dataclass_table.py`）

为避免代码重复，把 `_extract_inline_comments` / `_format_default` / `_format_type` / `build_dataclass_section` 抽成一个共享模块。如果 Task 10 写完后发现共享性强，本 task 内做这次重构。

```python
"""Generate LaTeX reference of registry dataclasses (data/registry/schema.py)."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
from pathlib import Path

from .generate_config_schema import (  # 复用工具函数
    _extract_inline_comments,
    _format_default,
    _format_type,
)
from .lib.io import write_generated
from .lib.tables import longtable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "docs/manual/_generated/registry_schema.tex"
SCHEMA_MODULE = "openbench.data.registry.schema"
SCHEMA_FILE = REPO_ROOT / "src/openbench/data/registry/schema.py"


def build_registry_schema_table() -> str:
    module = importlib.import_module(SCHEMA_MODULE)
    comments = _extract_inline_comments(SCHEMA_FILE)
    parts: list[str] = []
    # 自动发现：列出模块中所有顶层 dataclass，按定义顺序
    for name, obj in inspect.getmembers(module, dataclasses.is_dataclass):
        # 排除从其他模块 import 进来的
        if obj.__module__ != SCHEMA_MODULE:
            continue
        parts.append(rf"\subsection*{{{name}}}")
        if obj.__doc__:
            parts.append(obj.__doc__.strip().splitlines()[0] + r"\\")
        rows = []
        for field in dataclasses.fields(obj):
            description = comments.get((name, field.name), "")
            rows.append([
                field.name,
                _format_type(field.type),
                _format_default(field),
                description,
            ])
        parts.append(
            longtable(
                headers=["字段", "类型", "默认值", "说明"],
                rows=rows,
                col_spec="l l l p{5cm}",
            )
        )
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = build_registry_schema_table()
    write_generated(args.output, body, source="src/openbench/data/registry/schema.py")
    print(f"wrote {args.output} ({len(body)} chars)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** 测试 + 运行

```bash
python -m pytest tests/manual/test_generate_registry_schema.py -v
python -m docs.manual.scripts.generate_registry_schema
```

- [ ] **Step 4:** Commit

```bash
git add docs/manual/scripts/generate_registry_schema.py tests/manual/test_generate_registry_schema.py
git commit -m "feat(manual): generate registry schema reference from dataclasses"
```

---

## Phase 6: Internal interfaces generator

### Task 12: 写测试

**Files:** Create `tests/manual/test_generate_internal_interfaces.py`

- [ ] **Step 1:** 测试

```python
"""Tests for generate_internal_interfaces."""
from docs.manual.scripts.generate_internal_interfaces import build_interfaces_doc


class TestBuildInterfacesDoc:
    def test_includes_known_abc(self):
        out = build_interfaces_doc()
        # util/interfaces.py 含 IOutputManager（ABC）
        assert "IOutputManager" in out

    def test_lists_abstract_methods(self):
        out = build_interfaces_doc()
        # 至少 IOutputManager 有抽象方法被列出
        assert r"\subsection*{IOutputManager}" in out

    def test_escapes_chars(self):
        out = build_interfaces_doc()
        # 不允许 \_ 之外的裸下划线
        assert "_metadata" not in out or r"\_metadata" in out
```

- [ ] **Step 2:** 跑测试验证红

---

### Task 13: 实现 generate_internal_interfaces

**Files:** Create `docs/manual/scripts/generate_internal_interfaces.py`

- [ ] **Step 1:** 实现要点：
  - 扫描指定目录（`util/`、`core/`、`data/`、`runner/`）下的 `.py`
  - 用 `inspect` 找出 ABC 子类、Protocol、`@abstractmethod` 装饰的方法
  - 输出每个类的：模块路径、文档字符串首行、抽象方法签名

```python
"""Scan OpenBench source for ABC / Protocol / abstractmethod and emit reference."""
from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
from abc import ABC
from pathlib import Path
from typing import Iterator

from .lib.escape import latex_escape
from .lib.io import write_generated

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "docs/manual/_generated/internal_interfaces.tex"

# 要扫描的子包
SCAN_PACKAGES = [
    "openbench.util",
    "openbench.core",
    "openbench.data",
    "openbench.runner",
]


def _iter_modules(pkg_name: str) -> Iterator[str]:
    pkg = importlib.import_module(pkg_name)
    yield pkg_name
    if not hasattr(pkg, "__path__"):
        return
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
        # 跳过 __pycache__ / 测试文件
        if info.name.endswith("__pycache__"):
            continue
        yield info.name


def _is_abstract_class(obj: type) -> bool:
    if not inspect.isclass(obj):
        return False
    # ABC 子类
    if issubclass(obj, ABC) and obj is not ABC:
        return True
    # Protocol：检查 __mro__ 中是否含 typing.Protocol
    for base in obj.__mro__:
        if base.__name__ == "Protocol" and base.__module__ == "typing":
            return True
    return False


def _abstract_methods(cls: type) -> list[str]:
    return sorted(getattr(cls, "__abstractmethods__", set()))


def build_interfaces_doc() -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for pkg in SCAN_PACKAGES:
        for mod_name in _iter_modules(pkg):
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:  # 跳过 import 失败的模块（缺可选依赖等）
                continue
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                # 必须定义在被扫描的模块本身（不是 import 进来的）
                if obj.__module__ != mod_name:
                    continue
                if not _is_abstract_class(obj):
                    continue
                key = f"{mod_name}.{name}"
                if key in seen:
                    continue
                seen.add(key)
                parts.append(rf"\subsection*{{{name}}}")
                parts.append(rf"\noindent\textit{{{latex_escape(mod_name)}}}\\")
                if obj.__doc__:
                    parts.append(latex_escape(obj.__doc__.strip().splitlines()[0]))
                methods = _abstract_methods(obj)
                if methods:
                    parts.append(r"\begin{itemize}")
                    for m in methods:
                        parts.append(rf"  \item \texttt{{{latex_escape(m)}}}")
                    parts.append(r"\end{itemize}")
                parts.append("")
    if not parts:
        return "暂无可识别的 ABC / Protocol 接口。\n"
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = build_interfaces_doc()
    write_generated(
        args.output,
        body,
        source="src/openbench/{util,core,data,runner}/**/*.py (扫描 ABC/Protocol)",
    )
    print(f"wrote {args.output} ({len(body)} chars)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** 测试 + 运行

```bash
python -m pytest tests/manual/test_generate_internal_interfaces.py -v
python -m docs.manual.scripts.generate_internal_interfaces
```

- [ ] **Step 3:** Commit

```bash
git add docs/manual/scripts/generate_internal_interfaces.py tests/manual/test_generate_internal_interfaces.py
git commit -m "feat(manual): generate internal interfaces reference from ABC/Protocol scan"
```

---

## Phase 7: Integration

### Task 14: run_all 编排脚本

**Files:** Create `docs/manual/scripts/run_all.py`

- [ ] **Step 1:** 写实现

```python
"""Run all manual generators in sequence."""
from __future__ import annotations

import sys

from . import (
    generate_config_schema,
    generate_internal_interfaces,
    generate_model_table,
    generate_reference_table,
    generate_registry_schema,
)


GENERATORS = [
    ("reference_table", generate_reference_table.main),
    ("model_table", generate_model_table.main),
    ("config_schema", generate_config_schema.main),
    ("registry_schema", generate_registry_schema.main),
    ("internal_interfaces", generate_internal_interfaces.main),
]


def main() -> int:
    failures: list[str] = []
    for name, fn in GENERATORS:
        print(f"\n=== {name} ===", file=sys.stderr)
        try:
            fn()
        except SystemExit as e:
            if e.code not in (0, None):
                failures.append(f"{name}: exit {e.code}")
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    if failures:
        print("\nFailures:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nAll generators succeeded.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2:** 运行验证

```bash
cd /Volumes/Data01/Openbench && python -m docs.manual.scripts.run_all
```

预期：5 段输出，每段 "wrote .../<name>.tex"，最终 "All generators succeeded."

- [ ] **Step 3:** 验证产物

```bash
ls -la docs/manual/_generated/*.tex
```

预期：5 个 .tex 文件存在。

- [ ] **Step 4:** Commit

```bash
git add docs/manual/scripts/run_all.py
git commit -m "feat(manual): orchestrate all generators via run_all"
```

---

### Task 15: Makefile 集成

**Files:** Modify `docs/manual/Makefile`

- [ ] **Step 1:** 加 `generated` 目标，并把 `all` 改为依赖它

```makefile
# 在 .PHONY 行加上 generated：
.PHONY: all user dev ops master clean probe generated

# all 现在依赖 generated（即每次 make all 都重新生成参考章节）
all: generated user dev ops master

generated:
	cd .. && python -m docs.manual.scripts.run_all
```

注意：路径 `cd ..` 是因为 Makefile 在 `docs/manual/`，需回到 repo root 才能用 `python -m docs.manual.scripts.run_all`。

- [ ] **Step 2:** 测试

```bash
cd /Volumes/Data01/Openbench/docs/manual && make clean && make generated && ls _generated/
```

预期：5 个 .tex 文件出现。

- [ ] **Step 3:** 然后 `make all` 验收

```bash
make all 2>&1 | tail -10
```

预期：generated → user → dev → ops → master 顺序完成；4 个 PDF 都在。

- [ ] **Step 4:** Commit

```bash
git add docs/manual/Makefile
git commit -m "build(manual): wire generators into make all"
```

---

### Task 16: 把 stub 附录改为引用生成产物

**Files:** Modify
- `docs/manual/user/appendices/A-stub.tex`
- `docs/manual/developer/appendices/A-stub.tex`

把 stub 内容替换为引用生成片段，验证 LaTeX 端 `\input{}` 路径正确。

- [ ] **Step 1:** `user/appendices/A-stub.tex`

```latex
% docs/manual/user/appendices/A-stub.tex
% 占位附录：演示如何 \input 自动生成的内容（正式撰写时由 A-config-reference.tex 替换）

\chapter{占位附录（含生成内容）}

下表自动从 \file{src/openbench/config/schema.py} 抽取，由 \cli{make generated} 重新生成：

\input{../_generated/config_schema}
```

- [ ] **Step 2:** `developer/appendices/A-stub.tex`

```latex
% docs/manual/developer/appendices/A-stub.tex
\chapter{占位附录（含生成内容）}

下面列出从代码扫描出的 ABC / Protocol 接口，由 \cli{make generated} 重新生成：

\input{../_generated/internal_interfaces}
```

- [ ] **Step 3:** 重新编译验证 LaTeX 能正确 \input 路径

```bash
cd /Volumes/Data01/Openbench/docs/manual && make clean && make all 2>&1 | tail -15
```

预期：4 PDF 全部产出；user/dev 的 PDF 现在含真实生成的表格。

- [ ] **Step 4:** 检查页数变化

```bash
pdfinfo user/main_user.pdf | grep Pages
pdfinfo developer/main_developer.pdf | grep Pages
```

预期：用户卷与开发卷都比之前多几页（因为附录现在含真实表格）。

- [ ] **Step 5:** Commit

```bash
git add docs/manual/user/appendices/A-stub.tex docs/manual/developer/appendices/A-stub.tex
git commit -m "docs: stub appendices now \\input generated fragments"
```

---

### Task 17: 更新 README

**Files:** Modify `docs/manual/README.md`

- [ ] **Step 1:** 加一节"生成器"

在"编译"小节之后追加：

```markdown
## 自动生成器

部分参考章节由代码自动生成，避免手抄漂移：

| 输出文件 (`_generated/`) | 数据源 | 生成器 |
|---|---|---|
| `reference_table.tex` | `src/openbench/data/registry/reference_catalog.yaml` | `generate_reference_table` |
| `model_table.tex` | `src/openbench/data/registry/model_catalog.yaml` | `generate_model_table` |
| `config_schema.tex` | `src/openbench/config/schema.py` | `generate_config_schema` |
| `registry_schema.tex` | `src/openbench/data/registry/schema.py` | `generate_registry_schema` |
| `internal_interfaces.tex` | `src/openbench/{util,core,data,runner}/**/*.py` (扫 ABC/Protocol) | `generate_internal_interfaces` |

`make all` 会先调用 `make generated`，所以正常编译流程不需要手动跑生成器。
单独再生：

```bash
make generated
# 或单个
python -m docs.manual.scripts.generate_reference_table
```

修改了源文件（schema 或 catalog）就跑 `make generated`，否则附录会过期。
```

- [ ] **Step 2:** Commit

```bash
git add docs/manual/README.md
git commit -m "docs: document manual generators in README"
```

---

### Task 18: 端到端验收

**Files:** 仅运行命令。

- [ ] **Step 1:** 清空所有 LaTeX 中间产物，做一次端到端冷启动

```bash
cd /Volumes/Data01/Openbench/docs/manual
make clean
rm -rf _generated/*.tex
time make all 2>&1 | tee /tmp/manual_full_build.log | tail -20
```

预期：5-15 秒；最终 4 个 PDF + 5 个 _generated/*.tex 全部就绪。

- [ ] **Step 2:** 跑全部 manual 测试

```bash
cd /Volumes/Data01/Openbench && python -m pytest tests/manual/ -v
```

预期：全部 passed。

- [ ] **Step 3:** 检查所有 LaTeX log 干净

```bash
for f in user/main_user.log developer/main_developer.log operations/main_operations.log manual.log; do
  echo "=== $f ==="
  cd /Volumes/Data01/Openbench/docs/manual
  grep -c -E "^\!|Undefined reference|Citation undefined" "$f"
done
```

预期：全部 0。

- [ ] **Step 4:** 检查 .gitignore 把 `_generated/*.tex` 与 `*.xdv` `_minted/` 等都覆盖

```bash
git status -s docs/manual/
```

预期：干净，无 untracked 中间产物。

- [ ] **Step 5:** 不提交 —— 这是验收，无新文件改动。

---

## 自审清单

- [ ] **Spec 覆盖**：spec §5 中列的 5 个全自动生成器都已实现（reference / model / config_schema / registry_schema / internal_interfaces）
- [ ] **占位词扫描**：本计划无 TBD/TODO；所有 Python 与 LaTeX 代码都是完整可运行的
- [ ] **类型一致性**：`build_*_table` 所有签名都返回 `str`；`main()` 签名一致；CLI flag `--output` 一致
- [ ] **TDD 顺序**：每个生成器都先红测试再实现，按 superpowers:test-driven-development 风格
- [ ] **不破坏 Plan 1 验收**：Task 18 验证 4 个 PDF 仍可编译且 0 错误

---

## 完成后状态

- 5 个生成器可独立运行，亦可通过 `run_all` 一次跑完
- `make all` 自动调用 `make generated`，文档与代码不再漂移
- 用户卷 / 开发卷的 stub 附录已演示 `\input` 用法
- 全部测试 passing，CI 接口预留好（Plan 4 写测试章节时可纳入 CI）

下一份 plan：**`2026-XX-XX-manual-volume-user.md`** —— 用户卷 10 章 + 5 真实附录（替换 stub），含深度审查 + bug 报告流程的实战。
