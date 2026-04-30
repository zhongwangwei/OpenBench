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
