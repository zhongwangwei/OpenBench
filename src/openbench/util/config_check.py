"""Terminal color helper retained for legacy utility consumers."""

from __future__ import annotations

import os
import platform
import sys


def get_platform_colors() -> dict[str, str]:
    """Return terminal color codes plus a unicode-support flag."""
    system = platform.system().lower()
    supports_color = (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("TERM", "") != "dumb"
        and ("COLORTERM" in os.environ or os.environ.get("TERM", "").endswith(("color", "256color", "truecolor")))
    )

    if system == "windows" and not os.environ.get("WT_SESSION"):
        supports_color = False

    supports_unicode = True
    if system == "windows" and not os.environ.get("WT_SESSION"):
        supports_unicode = False

    try:
        if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
            supports_unicode = False
    except (AttributeError, TypeError):
        pass

    if supports_color:
        return {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "cyan": "\033[1;36m",
            "green": "\033[1;32m",
            "yellow": "\033[1;33m",
            "blue": "\033[1;34m",
            "magenta": "\033[1;35m",
            "red": "\033[1;31m",
            "unicode_support": supports_unicode,
        }

    return {
        "reset": "",
        "bold": "",
        "cyan": "",
        "green": "",
        "yellow": "",
        "blue": "",
        "magenta": "",
        "red": "",
        "unicode_support": supports_unicode,
    }
