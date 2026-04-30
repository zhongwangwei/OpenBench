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
        if info.name.endswith("__pycache__"):
            continue
        yield info.name


def _is_abstract_class(obj: type) -> bool:
    if not inspect.isclass(obj):
        return False
    # ABC 子类
    try:
        if issubclass(obj, ABC) and obj is not ABC:
            return True
    except TypeError:
        pass
    # Protocol：检查 __mro__ 中是否含 typing.Protocol
    for base in getattr(obj, "__mro__", ()):
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
            except Exception:
                # 跳过 import 失败的模块（缺可选依赖等）
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
