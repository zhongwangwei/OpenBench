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
