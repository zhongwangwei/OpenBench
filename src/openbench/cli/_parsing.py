"""Shared CLI parsing helpers for variable and fallback definitions.

Used by both ``data register`` and ``model register`` commands.
"""

from __future__ import annotations

from typing import Any

import click


def parse_variables(raw_vars: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Parse ``-v 'VarName:ncname:unit'`` options into a variables dict.

    Returns:
        ``{std_name: {"varname": nc_name, "varunit": unit}}``
    """
    result: dict[str, dict[str, Any]] = {}
    for v in raw_vars:
        parts = v.split(":")
        if len(parts) < 2:
            click.secho(f"Invalid format: '{v}'. Use 'VarName:ncname:unit'", fg="red")
            raise SystemExit(1)
        std_name = parts[0].strip()
        nc_name = parts[1].strip()
        unit = parts[2].strip() if len(parts) > 2 else ""
        result[std_name] = {"varname": nc_name, "varunit": unit}
    return result


def parse_fallbacks(
    raw_fallbacks: tuple[str, ...],
    new_vars: dict[str, dict[str, Any]],
    existing_vars: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Parse ``-f 'VarName:fb_name:fb_unit[:conversion]'`` and attach to *new_vars* in-place.

    If a fallback references a variable not yet in *new_vars* but present in
    *existing_vars*, the existing entry is copied into *new_vars* first so the
    fallback can be attached.
    """
    if existing_vars is None:
        existing_vars = {}

    for fb_def in raw_fallbacks:
        parts = fb_def.split(":")
        if len(parts) < 3:
            click.secho(
                f"Invalid fallback: '{fb_def}'. Use 'VarName:fb_name:fb_unit[:conversion]'",
                fg="red",
            )
            raise SystemExit(1)

        std_name = parts[0].strip()
        fb_entry: dict[str, str] = {
            "varname": parts[1].strip(),
            "varunit": parts[2].strip(),
        }
        if len(parts) > 3:
            fb_entry["convert"] = parts[3].strip()

        # Find or copy the target variable
        if std_name in new_vars:
            target = new_vars[std_name]
        elif std_name in existing_vars:
            new_vars[std_name] = dict(existing_vars[std_name])
            target = new_vars[std_name]
        else:
            click.secho(
                f"Warning: fallback for '{std_name}' but no primary variable defined. Use -v first.",
                fg="yellow",
            )
            continue

        if "fallbacks" not in target:
            target["fallbacks"] = []
        target["fallbacks"].append(fb_entry)
