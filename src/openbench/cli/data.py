"""openbench data commands."""

import click


@click.group()
def data():
    """Manage reference datasets."""


@data.command("list")
def list_datasets():
    """List all available reference datasets."""
    click.echo("Not yet implemented.")


@data.command()
@click.argument("names", nargs=-1, required=True)
def download(names):
    """Download reference datasets by name."""
    click.echo(f"Not yet implemented. Datasets: {', '.join(names)}")


@data.command()
def status():
    """Show local dataset cache status."""
    click.echo("Not yet implemented.")


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset."""
    click.echo(f"Not yet implemented. Dataset: {name}")


@data.command()
@click.argument("name")
def optimize(name):
    """Convert dataset to zarr for faster reads."""
    click.echo(f"Not yet implemented. Dataset: {name}")
