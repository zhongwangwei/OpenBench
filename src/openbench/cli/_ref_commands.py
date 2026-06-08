"""Implementation helpers for simple ``openbench ref`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import click

from openbench.util.names import get_mapping_key_case_insensitive


def reference_to_dict(ref) -> dict:
    def convert(value):
        if hasattr(value, "to_dict"):
            return convert(value.to_dict())
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "__dict__"):
            return {str(k): convert(v) for k, v in vars(value).items() if not str(k).startswith("_")}
        return str(value)

    converted = convert(ref)
    return converted if isinstance(converted, dict) else {}


def list_datasets(variable, fmt, *, reference_to_dict_fn: Callable = reference_to_dict):
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()

    if variable:
        refs = mgr.references_for_variable(variable)
        if not refs:
            click.echo(f"No datasets found for variable: {variable}")
            return
    else:
        refs = mgr.list_references()

    if fmt == "json":
        click.echo(json.dumps([reference_to_dict_fn(ref) for ref in refs], indent=2, sort_keys=True))
        return

    click.secho(f"{'Name':<30} {'Category':<12} {'Type':<6} {'Res':<8} {'Years':<14} {'Variables'}", bold=True)
    click.echo("─" * 100)
    for r in refs:
        res = f"{r.grid_res}°" if r.grid_res else "stn"
        years = f"{r.years[0]}-{r.years[1]}" if r.years else "?"
        nvars = len(r.variables)
        click.echo(f"{r.name:<30} {r.category:<12} {r.data_type:<6} {res:<8} {years:<14} {nvars}")

    click.echo(f"\nTotal: {len(refs)} datasets")


def download(names):
    click.echo("Dataset download not yet implemented (requires hosted data repository).")
    click.echo(f"Requested: {', '.join(names)}")
    raise click.ClickException("Dataset download is not yet implemented.")


def status():
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    refs = mgr.list_references()
    click.echo(f"Registry: {len(refs)} datasets available")
    click.echo("Local cache: not yet implemented")


def path(name):
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)

    if ref is None:
        variants = mgr.get_resolution_variants(name)
        if variants:
            for _res_label, variant in sorted(variants.items()):
                path_str = variant.root_dir or "(no local path)"
                click.echo(f"{variant.name}: {path_str}")
            return
        raise click.ClickException(f"Dataset not found: {name}")

    if ref.root_dir:
        click.echo(ref.root_dir)
    else:
        click.echo(f"No local path configured for {name}")


def convert_old(old_path: Path, output_path: Path, name: str, category: str, description: str):
    from openbench.data.registry.converter import convert_old_reference

    try:
        convert_old_reference(
            old_path=old_path,
            output_path=output_path,
            name=name,
            category=category,
            description=description,
        )
    except Exception as exc:
        raise click.ClickException(f"Failed to convert old reference descriptor: {exc}") from exc
    click.secho(f"✓ Converted {old_path} -> {output_path}", fg="green")


def delete_reference(name, yes, *, load_catalog_for_cli_fn: Callable):
    from openbench.data.registry.manager import RegistryManager, get_writable_reference_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    catalog_path = get_writable_reference_catalog_path()
    catalog = load_catalog_for_cli_fn(catalog_path)
    key = get_mapping_key_case_insensitive(catalog, name)
    if key is None:
        if RegistryManager().get_reference(name) is not None:
            raise click.ClickException(
                f"'{name}' is a bundled reference entry. Delete only removes user entries/overlays."
            )
        raise click.ClickException(f"Reference not found in user catalog: {name}")
    if not yes and not click.confirm(f"Delete reference entry '{key}' from {catalog_path}?"):
        return
    with _catalog_write_lock(catalog_path):
        catalog = load_catalog_for_cli_fn(catalog_path)
        key = get_mapping_key_case_insensitive(catalog, name)
        if key is None:
            raise click.ClickException(f"Reference no longer exists in user catalog: {name}")
        catalog.pop(key, None)
        backup_path = _backup_then_write(catalog_path, catalog)
    _invalidate_registry_caches()
    click.secho(f"✓ Deleted reference entry '{key}'", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")


def generate_station_list(dataset_dir, output, *, expand_existing_directory_fn: Callable, expand_path_fn: Callable):
    from openbench.data.registry.scanner import generate_station_list as gen_list

    dataset_path = expand_existing_directory_fn(dataset_dir, "DATASET_DIR")
    output_path = expand_path_fn(output) if output else None

    try:
        csv_path = gen_list(dataset_path, output_path)
        import pandas as pd

        df = pd.read_csv(csv_path)
        click.secho(f"✓ Generated {csv_path}", fg="green")
        click.echo(f"  Stations: {len(df)}")
        if "LON" in df.columns and "LAT" in df.columns:
            click.echo(f"  Lon range: [{df['LON'].min():.2f}, {df['LON'].max():.2f}]")
            click.echo(f"  Lat range: [{df['LAT'].min():.2f}, {df['LAT'].max():.2f}]")
        if "SYEAR" in df.columns and "EYEAR" in df.columns:
            click.echo(f"  Year range: [{df['SYEAR'].min()}, {df['EYEAR'].max()}]")
        elif "SYEAR" in df.columns:
            click.echo(f"  Start year: {df['SYEAR'].min()}-{df['SYEAR'].max()}")
    except Exception as e:
        raise click.ClickException(f"Failed: {e}") from e
