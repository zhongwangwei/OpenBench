"""Lazy bridge from core algorithms to visualization renderers.

Core modules must not import :mod:`openbench.visualization` at module import
time: visualization modules import core classes for the read-only drawing path,
so eager imports create a fragile package-level cycle.  Keep the dependency as a
runtime call boundary instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def visualization_callable(name: str) -> Callable[..., Any]:
    """Return a callable that resolves a visualization function only on use."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        from openbench import visualization

        return getattr(visualization, name)(*args, **kwargs)

    return _call
