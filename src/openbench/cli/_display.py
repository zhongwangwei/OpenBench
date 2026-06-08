"""Display helpers for reference CLI commands."""

from __future__ import annotations

import json
from typing import Callable

import click
import yaml


def show_reference(name: str, fmt: str, *, reference_to_dict_fn: Callable) -> None:
    """Show details of a dataset. Supports base name to show all resolutions."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    variants = mgr.get_resolution_variants(name)

    if variants and len(variants) > 1:
        if fmt != "text":
            data = {res_label: reference_to_dict_fn(ref) for res_label, ref in sorted(variants.items())}
            if fmt == "json":
                click.echo(json.dumps(data, indent=2, sort_keys=True))
            else:
                click.echo(yaml.safe_dump(data, sort_keys=False).rstrip())
            return
        click.secho(f"{name}", bold=True)
        click.echo(f"Available at {len(variants)} resolution(s):\n")

        for res_label, ref in sorted(variants.items()):
            status = click.style(f"[{res_label}]", bold=True)
            click.echo(f"  {status} {ref.name}")
            click.echo(f"    Type: {ref.data_type}, Grid: {ref.grid_res or 'N/A'}°, Time: {ref.tim_res}")
            years = f"{ref.years[0]}-{ref.years[1]}" if ref.years else "N/A"
            click.echo(f"    Years: {years}, Variables: {len(ref.variables)}")
            if ref.root_dir:
                click.echo(f"    Path: {ref.root_dir}")
            click.echo()

        click.echo("Base-name references are resolved using target resolution context:")
        click.echo("  - project.tim_res / project.grid_res when set")
        click.echo("  - otherwise the shared simulation resolution, if all simulations agree")
        click.echo("  - otherwise OpenBench asks you to specify a full variant or set comparison.* explicitly")
        click.echo()
        click.echo("In openbench.yaml, use either:")
        click.echo("  reference:")
        click.echo(f"    Evapotranspiration: {name}            # select by target resolution context")
        for _res_label, ref in sorted(variants.items()):
            click.echo(f"    Evapotranspiration: {ref.name}   # force {_res_label}")
        return

    if variants:
        ref = next(iter(variants.values()))
    else:
        ref = mgr.get_reference(name)
    if ref is None:
        raise click.ClickException(f"Dataset not found: {name}")
    if fmt != "text":
        data = reference_to_dict_fn(ref)
        if fmt == "json":
            click.echo(json.dumps(data, indent=2, sort_keys=True))
        else:
            click.echo(yaml.safe_dump(data, sort_keys=False).rstrip())
        return

    click.secho(f"{ref.name}", bold=True)
    click.echo(f"Description: {ref.description}")
    click.echo(f"Category: {ref.category}")
    click.echo(f"Type: {ref.data_type}, Resolution: {ref.grid_res or 'N/A'}°, Time: {ref.tim_res}")
    click.echo(f"Years: {ref.years[0]}-{ref.years[1]}" if ref.years else "Years: N/A")
    if ref.root_dir:
        click.echo(f"Path: {ref.root_dir}")
    click.echo()
    click.secho(f"{'Variable':<35} {'NetCDF name':<20} {'Unit':<20} {'Fallback'}", bold=True)
    click.echo("─" * 100)
    for var_name, mapping in sorted(ref.variables.items()):
        vn = mapping.varname if isinstance(mapping.varname, str) else mapping.varname[0]
        fb_str = ""
        if mapping.fallbacks:
            fb_parts = []
            for fb in mapping.fallbacks:
                conv = f" ({fb.convert})" if fb.convert else ""
                fb_parts.append(f"{fb.varname} [{fb.varunit}]{conv}")
            fb_str = " → ".join(fb_parts)
        click.echo(f"{var_name:<35} {vn:<20} {mapping.varunit:<20} {fb_str}")
