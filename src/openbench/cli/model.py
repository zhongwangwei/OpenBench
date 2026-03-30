"""openbench model commands."""

import click


@click.group()
def model():
    """Manage model profiles."""


@model.command("list")
def list_models():
    """List all available model profiles."""
    click.echo("Not yet implemented.")


@model.command()
@click.argument("name")
def show(name):
    """Show variable mappings for a model profile."""
    click.echo(f"Not yet implemented. Model: {name}")


@model.command()
def create():
    """Interactively create a new model profile."""
    click.echo("Not yet implemented.")
