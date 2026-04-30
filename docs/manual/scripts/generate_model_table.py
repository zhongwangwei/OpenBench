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
