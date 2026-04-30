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
