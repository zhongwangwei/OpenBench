"""Shared CLI helpers for reference-resolution failures."""

import click

_AMBIGUOUS_TARGET_RESOLUTION = "Reference resolution is ambiguous across simulations:"


def emit_reference_resolution_error(message: str, *, prefix: str = "") -> None:
    """Render a resolver failure with optional user-facing remediation."""
    click.secho(f"{prefix}{message}", fg="red")

    if _AMBIGUOUS_TARGET_RESOLUTION not in message:
        return

    click.echo("  Simulations imply different target resolutions, so reference auto-selection is ambiguous.")
    click.echo("  Set project.tim_res / project.grid_res explicitly so all references bind to one target resolution.")
    click.echo("  Example:")
    click.echo("    project:")
    click.echo("      tim_res: Month")
    click.echo("      grid_res: 0.25")
