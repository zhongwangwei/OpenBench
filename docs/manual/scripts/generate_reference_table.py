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
