"""Shared helpers for the Taylor and Target diagram modules.

Both `Fig_taylor_diagram.py` and `Fig_target_diagram.py` historically
carried verbatim copies of these tiny utilities (one prefixed with `_`,
one without). Centralising them here removes ~100 lines of duplication
and ensures any future bug fix propagates to both.
"""

from __future__ import annotations

import ast
import re


def is_int(element) -> bool:
    """Check if value can be parsed as an integer."""
    try:
        int(element)
        return True
    except (ValueError, TypeError):
        return False


def is_float(element) -> bool:
    """Check if value can be parsed as a float."""
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def is_list_in_string(element: str) -> bool:
    """Check if a string contains list-like brackets."""
    return bool(re.search(r"\[|\]", element))


def disp(text: str = "") -> None:
    """Print a single line — used by the diagram-options help text."""
    print(text)


def dispopt(name: str, description: str) -> None:
    """Print an `option_name: description` line indented two spaces."""
    print(f"  {name}: {description}")


def parse_literal_option(value: str, option_name: str):
    """Parse a CSV option value with ``ast.literal_eval``.

    Diagram option files historically used Python tuple syntax for color
    triples, e.g. ``(0, 0.6, 0)``.  Use literal parsing rather than
    ``eval`` so option files cannot execute code.
    """
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {option_name}: {value}") from exc
