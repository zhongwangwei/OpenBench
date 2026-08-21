"""Shared navigation helpers for interactive CLI workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import click


class BackRequested(Exception):
    """Return control to the previous interactive step."""


class _BackAwareType(click.ParamType):
    def __init__(self, inner: click.ParamType):
        self.inner = inner
        self.name = inner.name

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.strip().lower() == "back":
            raise BackRequested
        return self.inner.convert(value, param, ctx)


def prompt(text: str, *, type=None, **kwargs):
    """Prompt like Click, reserving ``back`` for navigation."""
    inner = click.types.convert_type(type, kwargs.get("default"))
    return click.prompt(text, type=_BackAwareType(inner), **kwargs)


def confirm(text: str, *, default: bool) -> bool:
    """Confirm while accepting the shared back command."""
    options = "Y/n/back" if default else "y/N/back"
    while True:
        answer = prompt(f"{text} [{options}]", default="", show_default=False).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        click.secho("  Enter yes, no, or back.", fg="red")


def prompt_fields(fields: Sequence[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    """Collect a fixed field sequence, moving back one field on request."""
    return prompt_steps(
        [(key, text, lambda text=text, kwargs=kwargs: prompt(text, **kwargs)) for key, text, kwargs in fields]
    )


def prompt_steps(steps: Sequence[tuple[str, str, Callable[[], Any]]]) -> dict[str, Any]:
    """Collect arbitrary prompt steps, moving back one step on request."""
    values: dict[str, Any] = {}
    index = 0
    while index < len(steps):
        key, text, ask = steps[index]
        try:
            values[key] = ask()
            index += 1
        except BackRequested:
            if index == 0:
                raise
            index -= 1
            click.secho(f"  Returning to: {steps[index][1].strip()}", fg="yellow")
    return values


def navigation_hint() -> None:
    click.echo("  Type 'back' to return to the previous step.")
