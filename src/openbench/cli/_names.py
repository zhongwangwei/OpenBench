"""CLI-specific wrappers for OpenBench name resolution."""

from __future__ import annotations

import click

from openbench.util.names import (
    AmbiguousNameError,
    NameResolutionError,
    normalize_name,
    resolve_name_case_insensitive,
)


def resolve_variable_filters(
    variables: tuple[str, ...],
    configured_variables: list[str],
) -> list[str]:
    """Resolve ``--variable`` filters ignoring case while preserving CLI errors."""

    resolved: list[str] = []
    unknown: list[str] = []
    for var in variables:
        try:
            resolved.append(
                str(
                    resolve_name_case_insensitive(
                        var,
                        configured_variables,
                        label="--variable value",
                    )
                )
            )
        except AmbiguousNameError as exc:
            raise click.ClickException(str(exc)) from exc
        except NameResolutionError:
            unknown.append(var)

    seen: set[str] = set()
    duplicate_variables: list[str] = []
    for var in resolved:
        key = normalize_name(var)
        if key in seen and var not in duplicate_variables:
            duplicate_variables.append(var)
        seen.add(key)
    errors: list[str] = []
    if duplicate_variables:
        errors.append("--variable values must be unique: " + ", ".join(duplicate_variables))
    if unknown:
        errors.append("--variable value not in evaluation.variables: " + ", ".join(unknown))
    if errors:
        raise click.ClickException("; ".join(errors))
    return resolved
