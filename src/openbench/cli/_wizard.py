"""Shared navigation helpers for interactive CLI workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import click


class BackRequested(Exception):
    """Return control to the previous interactive step."""


def is_back(value, *, short_back: bool = True) -> bool:
    if not isinstance(value, str):
        return False
    word = value.strip().lower()
    return word == "back" or (short_back and word == "b")


class _BackAwareType(click.ParamType):
    def __init__(self, inner: click.ParamType, short_back: bool):
        self.inner = inner
        self.name = inner.name
        self.short_back = short_back

    def convert(self, value, param, ctx):
        if is_back(value, short_back=self.short_back):
            raise BackRequested
        return self.inner.convert(value, param, ctx)


def prompt(text: str, *, type=None, short_back: bool | None = None, **kwargs):
    """Prompt like Click, reserving ``back`` (and optionally ``b``) for navigation.

    ``b`` is accepted by default only for typed prompts; free-text prompts
    (NetCDF variable names, units, globs, ...) need ``short_back=True`` because
    ``b`` may be a legitimate answer there.
    """
    inner = click.types.convert_type(type, kwargs.get("default"))
    if short_back is None:
        short_back = inner is not click.STRING
    return click.prompt(text, type=_BackAwareType(inner, short_back), **kwargs)


def confirm(text: str, *, default: bool) -> bool:
    """Confirm while accepting the shared back command; Enter keeps the default."""
    options = f"yes/no/back, Enter = {'yes' if default else 'no'}"
    while True:
        answer = prompt(f"{text} [{options}]", default="", show_default=False, short_back=True).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        click.secho("  Enter yes (y), no (n), or back (b).", fg="red")


def prompt_fields(fields: Sequence[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    """Collect a fixed field sequence, moving back one field on request."""
    steps = []
    for key, text, kwargs in fields:
        prompt_kwargs = dict(kwargs)

        def ask(text=text, prompt_kwargs=prompt_kwargs):
            value = prompt(text, **prompt_kwargs)
            prompt_kwargs["default"] = value
            return value

        steps.append((key, text, ask))
    return prompt_steps(steps)


def prompt_steps(steps: Sequence[tuple]) -> dict[str, Any]:
    """Collect arbitrary prompt steps, moving back one step on request.

    Each step is ``(key, text, ask)`` or ``(key, text, ask, when)``. A step
    whose ``when()`` is false is skipped in both directions and leaves no value.
    """

    def active(index: int) -> bool:
        return len(steps[index]) < 4 or steps[index][3]()

    values: dict[str, Any] = {}
    index = 0
    while index < len(steps):
        key, text, ask = steps[index][:3]
        if not active(index):
            values.pop(key, None)
            index += 1
            continue
        try:
            values[key] = ask()
            index += 1
        except BackRequested:
            previous = index - 1
            while previous >= 0 and not active(previous):
                previous -= 1
            if previous < 0:
                raise
            index = previous
            click.secho(f"  Returning to: {steps[index][1].strip()}", fg="yellow")
    return values


def navigation_hint(*, short_back: bool = False) -> None:
    click.echo("  Press Enter to accept the default shown in [brackets].")
    if short_back:
        click.echo("  Type 'b' (back) to return to the previous step.")
    else:
        click.echo("  Type 'back' to return to the previous step.")
