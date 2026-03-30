"""openbench data commands."""

import click


@click.group()
def data():
    """Manage reference datasets."""


@data.command("list")
@click.option("--variable", default=None, help="Filter by variable name.")
def list_datasets(variable):
    """List all available reference datasets."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()

    if variable:
        refs = mgr.references_for_variable(variable)
        if not refs:
            click.echo(f"No datasets found for variable: {variable}")
            return
    else:
        refs = mgr.list_references()

    click.secho(f"{'Name':<30} {'Category':<12} {'Type':<6} {'Res':<8} {'Years':<14} {'Variables'}", bold=True)
    click.echo("─" * 100)
    for r in refs:
        res = f"{r.grid_res}°" if r.grid_res else "stn"
        years = f"{r.years[0]}-{r.years[1]}" if r.years else "?"
        nvars = len(r.variables)
        click.echo(f"{r.name:<30} {r.category:<12} {r.data_type:<6} {res:<8} {years:<14} {nvars}")

    click.echo(f"\nTotal: {len(refs)} datasets")


@data.command()
@click.argument("names", nargs=-1, required=True)
def download(names):
    """Download reference datasets by name."""
    click.echo("Dataset download not yet implemented (requires hosted data repository).")
    click.echo(f"Requested: {', '.join(names)}")


@data.command()
def status():
    """Show local dataset cache status."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    refs = mgr.list_references()
    click.echo(f"Registry: {len(refs)} datasets available")
    click.echo("Local cache: not yet implemented")


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)
    if ref is None:
        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)
    if ref.root_dir:
        click.echo(ref.root_dir)
    else:
        click.echo(f"No local path configured for {name}")


@data.command()
@click.argument("name")
def optimize(name):
    """Convert dataset to zarr for faster reads."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)
    if ref is None:
        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)

    if not ref.root_dir:
        click.secho(f"No local path configured for {name}. Download it first.", fg="red")
        raise SystemExit(1)

    try:
        import xarray as xr
    except ImportError:
        click.secho("xarray is required for optimization.", fg="red")
        raise SystemExit(1)

    import glob
    from pathlib import Path

    root = Path(ref.root_dir)
    if not root.exists():
        click.secho(f"Data directory not found: {root}", fg="red")
        raise SystemExit(1)

    zarr_dir = root.parent / f"{root.name}.zarr"
    if zarr_dir.exists():
        click.echo(f"Zarr store already exists: {zarr_dir}")
        if not click.confirm("Overwrite?"):
            return

    # Find all NetCDF files
    nc_files = sorted(glob.glob(str(root / "**" / "*.nc"), recursive=True))
    nc_files += sorted(glob.glob(str(root / "**" / "*.nc4"), recursive=True))

    if not nc_files:
        click.secho(f"No NetCDF files found in {root}", fg="yellow")
        return

    click.echo(f"Found {len(nc_files)} NetCDF files")
    click.echo(f"Converting to zarr: {zarr_dir}")

    try:
        # Open all files as a single dataset and save as zarr
        ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
        ds.to_zarr(str(zarr_dir), mode="w")
        ds.close()

        # Report size comparison
        import os

        nc_size = sum(os.path.getsize(f) for f in nc_files)
        zarr_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(zarr_dir)
            for f in fns
        )
        click.secho("✓ Converted to zarr", fg="green")
        click.echo(f"  NetCDF: {nc_size / 1e9:.1f} GB")
        click.echo(f"  Zarr:   {zarr_size / 1e9:.1f} GB")
    except Exception as e:
        click.secho(f"Conversion failed: {e}", fg="red")
        raise SystemExit(1)
