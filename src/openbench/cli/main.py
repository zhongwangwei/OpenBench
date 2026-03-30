"""OpenBench CLI entry point."""

import click

from openbench import __version__


@click.group()
@click.version_option(version=__version__, prog_name="openbench")
def cli():
    """OpenBench: Land Surface Model Benchmarking System."""


@cli.command()
def version():
    """Show version information."""
    click.echo(f"openbench {__version__}")
