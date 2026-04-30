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

    tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, tok in enumerate(tokens):
        # 检测 class 定义开始
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
    return comments


def _format_default(field: dataclasses.Field) -> str:
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            value = field.default_factory()
            # Avoid overfull hbox: dataclass repr like
            # "ComparisonConfig(enabled=False, items=None)" is too wide for
            # the 1.8cm default column. Show short form; field semantics
            # already covered in their own dataclass section.
            if dataclasses.is_dataclass(value):
                return f"{type(value).__name__}(...)"
            return repr(value)
        except Exception:
            return "<factory>"
    return "—"  # required


def _format_type(annotation: Any) -> str:
    # dataclasses 的 type hints 在 Python 3.10+ 可能是字符串（PEP 563）
    if isinstance(annotation, str):
        return annotation
    s = repr(annotation)
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
                col_spec="p{2.5cm} p{3.5cm} p{1.8cm} p{4.5cm}",
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
