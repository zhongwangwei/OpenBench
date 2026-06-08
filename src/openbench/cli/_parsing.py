"""Shared CLI parsing helpers for variable and fallback definitions.

Used by both ``ref register`` and ``model register`` commands.
"""

from __future__ import annotations

import shlex
from typing import Any

import click

from openbench.util.names import (
    AmbiguousNameError,
    get_mapping_key_case_insensitive,
)

_NAMED_VAR_KEYS = {
    "name": "varname",
    "nc": "varname",
    "nc_name": "varname",
    "varname": "varname",
    "unit": "varunit",
    "varunit": "varunit",
    "prefix": "prefix",
    "suffix": "suffix",
    "sub_dir": "sub_dir",
    "prefix_fallback": "prefix_fallback",
}


def _parse_named_variable(raw: str) -> tuple[str, dict[str, Any]]:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise click.ClickException(f"Invalid variable definition '{raw}': {exc}") from exc
    if len(tokens) < 2:
        raise click.ClickException(f"Invalid format: '{raw}'. Use 'StdName name=nc_var unit=unit [prefix=...]'")
    std_name = tokens[0].strip()
    if not std_name:
        raise click.ClickException(f"Invalid format: '{raw}'. Standard variable name must not be empty.")

    entry: dict[str, Any] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise click.ClickException(f"Invalid variable attribute '{token}'. Use key=value in named -v syntax.")
        key, value = token.split("=", 1)
        key = key.strip()
        mapped_key = _NAMED_VAR_KEYS.get(key)
        if mapped_key is None:
            valid = ", ".join(sorted(_NAMED_VAR_KEYS))
            raise click.ClickException(f"Invalid variable attribute '{key}'. Valid keys: {valid}")
        value = value.strip()
        if mapped_key == "prefix_fallback":
            entry[mapped_key] = [part.strip() for part in value.split(",") if part.strip()]
        else:
            entry[mapped_key] = value

    if not entry.get("varname"):
        raise click.ClickException(f"Invalid format: '{raw}'. Named -v syntax requires name= or varname=.")
    entry.setdefault("varunit", "")
    return std_name, entry


def parse_variables(raw_vars: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Parse variable options into registry variable mappings.

    Supports the legacy ``StdName:ncname:unit[:prefix[:suffix]]`` form and the
    named form ``StdName name=ncname unit="mm day-1" prefix="C:/case/"`` for
    values that contain colons or other delimiter-sensitive text.

    Returns:
        ``{std_name: {"varname": nc_name, "varunit": unit, ...}}``
    """
    result: dict[str, dict[str, Any]] = {}

    def add_variable(std_name: str, entry: dict[str, Any], raw: str) -> None:
        try:
            existing_key = get_mapping_key_case_insensitive(result, std_name)
        except AmbiguousNameError as exc:
            raise click.ClickException(f"Invalid variable definition '{raw}': {exc}") from exc
        if existing_key is not None:
            raise click.ClickException(f"Duplicate variable '{std_name}' conflicts with '{existing_key}' ignoring case")
        result[std_name] = entry

    for v in raw_vars:
        try:
            tokens = shlex.split(v)
        except ValueError as exc:
            raise click.ClickException(f"Invalid variable definition '{v}': {exc}") from exc
        if len(tokens) > 1 and any("=" in token for token in tokens[1:]):
            std_name, entry = _parse_named_variable(v)
            add_variable(std_name, entry, v)
            continue

        parts = v.split(":", 4)
        if len(parts) < 2:
            raise click.ClickException(
                f"Invalid format: '{v}'. Use 'StdName:ncname:unit[:prefix[:suffix]]' or 'StdName name=nc_var unit=unit'"
            )
        std_name = parts[0].strip()
        nc_name = parts[1].strip()
        if not std_name:
            raise click.ClickException(f"Invalid format: '{v}'. Standard variable name must not be empty.")
        if not nc_name:
            raise click.ClickException(f"Invalid format: '{v}'. NetCDF variable name must not be empty.")
        unit = parts[2].strip() if len(parts) > 2 else ""
        entry: dict[str, Any] = {"varname": nc_name, "varunit": unit}
        if len(parts) > 3 and parts[3].strip():
            entry["prefix"] = parts[3].strip()
        if len(parts) > 4 and parts[4].strip():
            entry["suffix"] = parts[4].strip()
        add_variable(std_name, entry, v)
    return result


def parse_fallbacks(
    raw_fallbacks: tuple[str, ...],
    new_vars: dict[str, dict[str, Any]],
    existing_vars: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Parse ``-f 'StdName:fb_name:fb_unit[:conversion]'`` and attach to *new_vars* in-place.

    If a fallback references a variable not yet in *new_vars* but present in
    *existing_vars*, the existing entry is copied into *new_vars* first so the
    fallback can be attached.
    """
    if existing_vars is None:
        existing_vars = {}

    for fb_def in raw_fallbacks:
        parts = fb_def.split(":", 3)
        if len(parts) < 3:
            raise click.ClickException(f"Invalid fallback: '{fb_def}'. Use 'StdName:fb_name:fb_unit[:conversion]'")

        std_name = parts[0].strip()
        fb_entry: dict[str, str] = {
            "varname": parts[1].strip(),
            "varunit": parts[2].strip(),
        }
        if len(parts) > 3:
            fb_entry["convert"] = parts[3].strip()

        # Find or copy the target variable.  Standard OpenBench variable
        # names are user-facing identifiers, so resolve them case-insensitively
        # while preserving the canonical key already present in the mapping.
        try:
            new_key = get_mapping_key_case_insensitive(new_vars, std_name)
            existing_key = get_mapping_key_case_insensitive(existing_vars, std_name)
        except AmbiguousNameError as exc:
            raise click.ClickException(f"Fallback target '{std_name}' is ambiguous: {exc}") from exc

        if new_key is not None:
            target = new_vars[new_key]
        elif existing_key is not None:
            new_vars[existing_key] = dict(existing_vars[existing_key])
            target = new_vars[existing_key]
        else:
            raise click.ClickException(
                f"Fallback target '{std_name}' has no primary variable. "
                "Define -v first or attach fallback to an existing variable."
            )

        if "fallbacks" not in target:
            target["fallbacks"] = []
        target["fallbacks"].append(fb_entry)
