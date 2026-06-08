"""openbench cache commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GiB"


@click.group()
def cache():
    """Manage OpenBench runtime caches."""


@cache.command("status")
@click.option("--regrid", "include_regrid", is_flag=True, help="Show conservative regrid weight cache status.")
@click.option("--dir", "cache_dir", type=click.Path(file_okay=False), default=None, help="Regrid cache directory.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def status(include_regrid, cache_dir, as_json):
    """Show cache status."""
    if not include_regrid:
        raise click.ClickException("Choose a cache family, e.g. --regrid")

    from openbench.data.regrid.methods.conservative import _weights_disk_cache_dir, prune_weight_disk_cache

    summary = prune_weight_disk_cache(cache_dir)
    if as_json:
        click.echo(json.dumps({"regrid": summary}, indent=2, sort_keys=True))
        return

    directory = str(Path(cache_dir).expanduser()) if cache_dir else _weights_disk_cache_dir()
    directory = directory or "(unset)"
    click.echo(f"Regrid weight cache: {directory}")
    click.echo(f"  Files: {summary['files']}")
    click.echo(f"  Size: {_format_bytes(summary['bytes'])}")
    if summary["removed_files"]:
        click.echo(f"  Pruned: {summary['removed_files']} file(s), {_format_bytes(summary['removed_bytes'])}")


@cache.command("clear")
@click.option("--regrid", "clear_regrid", is_flag=True, help="Clear conservative regrid weight disk cache.")
@click.option("--dir", "cache_dir", type=click.Path(file_okay=False), default=None, help="Regrid cache directory.")
@click.option("--yes", "assume_yes", "-y", is_flag=True, help="Delete without confirmation.")
def clear(clear_regrid, cache_dir, assume_yes):
    """Clear selected cache files."""
    if not clear_regrid:
        raise click.ClickException("Choose a cache family, e.g. --regrid")

    from openbench.data.regrid.methods.conservative import _weights_disk_cache_dir, clear_weight_cache

    target_path = str(Path(cache_dir).expanduser()) if cache_dir else _weights_disk_cache_dir()
    target_label = target_path or "(unset)"
    if not assume_yes:
        click.confirm(f"Delete regrid weight cache files in {target_label}?", abort=True)
    clear_weight_cache(clear_disk=True, cache_dir=target_path)
    click.echo("Regrid weight cache cleared")
