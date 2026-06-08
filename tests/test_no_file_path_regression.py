"""Guard against `Path(__file__)` / `dirname(__file__)` regressions.

We migrated package-resource lookups to ``importlib.resources.files()``
Traversable objects. The migration is what makes wheel + zipimport
loads work — see ``.github/workflows/ci.yml`` job ``wheel-zip-smoke``.

Whenever a developer adds a new module that does
``Path(__file__).parent / "data.yaml"`` or
``os.path.dirname(__file__) + "/foo"``, the GUI / CLI smoke tests
silently keep passing on a normal pip install but break under
zipimport. This test catches that at PR time, before it ships.

If you legitimately need ``__file__`` for something other than
package-resource resolution (e.g. logging the source file for
debugging), add the affected line / file to ``_ALLOWED`` below with a
brief justification.
"""

from __future__ import annotations

import re
from pathlib import Path

# Patterns that indicate a likely package-data lookup using __file__.
_FORBIDDEN = [
    re.compile(r"Path\(\s*__file__\s*\)"),
    re.compile(r"os\.path\.dirname\(\s*__file__\s*\)"),
    re.compile(r"os\.path\.abspath\(\s*__file__\s*\)"),
    re.compile(r"\b__file__\s*\.\s*parent\b"),
]

# Files that may legitimately reference these patterns (docstrings or
# educational comments). Each entry is the path relative to the repo root.
_ALLOWED = {
    # Generator template (NOT shipped as code — used by setup.py to
    # write the generated cmaps.py).
    "src/openbench/visualization/cmaps/cmaps.template",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _python_sources() -> list[Path]:
    root = _project_root() / "src" / "openbench"
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _is_in_string_or_comment(line: str, match_start: int) -> bool:
    """Heuristic: ignore matches inside docstrings or `# ...` comments.

    Accurate enough for our needs — we only want to catch real code
    that performs a package-data lookup. Imperfect for matches inside
    triple-quoted docstrings spanning multiple lines, but those are
    rare and the explicit ``_ALLOWED`` list covers the
    intentional-doc cases (``_resources.py``).
    """
    leading = line[:match_start]
    if "#" in leading:
        # Match appears after a `#` on the same line → comment.
        return True
    quote_count = leading.count('"') + leading.count("'")
    if quote_count % 2 == 1:
        # Odd number of quote chars before the match → match is inside
        # a string literal.
        return True
    return False


def _scan_file(path: Path) -> list[tuple[int, str]]:
    rel = path.relative_to(_project_root()).as_posix()
    if rel in _ALLOWED:
        return []
    text = path.read_text(encoding="utf-8")
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in _FORBIDDEN:
            for m in pattern.finditer(line):
                if _is_in_string_or_comment(line, m.start()):
                    continue
                hits.append((lineno, line.strip()))
                break
    return hits


def test_no_file_path_resource_lookups() -> None:
    """No active `__file__`-based path resolution outside the allowlist."""
    violations: list[str] = []
    for src in _python_sources():
        rel = src.relative_to(_project_root()).as_posix()
        for lineno, line in _scan_file(src):
            violations.append(f"{rel}:{lineno}: {line}")
    assert not violations, (
        "Use importlib.resources.files() instead of __file__-based path "
        "resolution — the latter breaks under "
        "zipimport and PyInstaller bundles. Offending lines:\n  " + "\n  ".join(violations[:30])
    )
