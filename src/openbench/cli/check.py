"""openbench check command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
def check(config):
    """Validate config file and check data availability."""
    click.echo(f"Not yet implemented. Config: {config}")
