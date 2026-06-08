"""Shared CLI option helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

import click

TIM_RES_VALUES = (
    "Month",
    "Day",
    "Hour",
    "Year",
    "3Hour",
    "6Hour",
    "8Day",
    "climatology-month",
    "climatology-year",
)
_TIM_RES_CANONICAL = {value.lower(): value for value in TIM_RES_VALUES}
_MULTI_MONTH_RE = re.compile(r"[1-9]\d*month")


class TimResParamType(click.ParamType):
    name = "Month|Day|Hour|Year|3Hour|6Hour|8Day|Nmonth|climatology-month|climatology-year"

    def convert(self, value, param, ctx):
        text = str(value).strip()
        lowered = text.lower()
        if lowered in _TIM_RES_CANONICAL:
            return _TIM_RES_CANONICAL[lowered]
        if _MULTI_MONTH_RE.fullmatch(lowered):
            return lowered
        self.fail(f"{value!r} is not a supported time resolution", param, ctx)


TIM_RES_TYPE = TimResParamType()


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def expand_existing_directory(value: str | Path, label: str) -> Path:
    path = expand_path(value).resolve()
    if not path.exists():
        raise click.ClickException(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise click.ClickException(f"{label} must be a directory: {path}")
    return path


def remote_not_implemented_message(profile: str | None = None) -> str:
    target = f" for '{profile}'" if profile else ""
    return (
        f"Remote execution not yet implemented{target}. "
        "Remote execution is not implemented in the CLI yet. "
        "Install colm-openbench[remote] and use openbench gui for remote execution."
    )
