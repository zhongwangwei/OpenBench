"""Helpers for isolating matplotlib rcParams mutations to a single call.

Many `make_*` plotting functions in this package set global rcParams
(`rcParams.update(...)`) at function entry to control axes width, font,
ticks, etc. Without restoration, these settings leak into all subsequent
matplotlib code in the same process — when the runner produces dozens of
plots in sequence, later figures inherit settings the user never asked
for, and tests that exercise plotting can flake based on call order.

This module provides two equivalent ways to fix that:

  * `isolated_rc(params)` — a context manager. Use as
        with isolated_rc(params):
            ...plot code...

  * `@with_isolated_rc` — a decorator. Annotate a make_* function and
    its rcParams.update call (still inside the function body) will be
    automatically reverted on exit, regardless of which path the
    function takes.

The decorator is the recommended fit for the existing code shape, since
it does not require indenting the function body.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import matplotlib
import matplotlib.pyplot as plt


def _close_figures_opened_after(open_figures: set[int]) -> None:
    """Close figures opened after *open_figures* was captured."""
    for number in set(plt.get_fignums()) - open_figures:
        plt.close(number)


@contextmanager
def isolated_rc(params: dict | None = None) -> Iterator[None]:
    """Snapshot rcParams, optionally apply *params*, restore on exit."""
    saved = dict(matplotlib.rcParams)
    open_figures = set(plt.get_fignums())
    if params:
        matplotlib.rcParams.update(params)
    try:
        yield
    except BaseException:
        _close_figures_opened_after(open_figures)
        raise
    finally:
        matplotlib.rcParams.update(saved)


def with_isolated_rc(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: snapshot+restore rcParams around the wrapped call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        saved = dict(matplotlib.rcParams)
        open_figures = set(plt.get_fignums())
        try:
            return func(*args, **kwargs)
        except BaseException:
            _close_figures_opened_after(open_figures)
            raise
        finally:
            matplotlib.rcParams.update(saved)

    return wrapper
