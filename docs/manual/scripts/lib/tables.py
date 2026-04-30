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
