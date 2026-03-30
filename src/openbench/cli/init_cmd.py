"""openbench init command."""

import click


@click.command("init")
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def init_cmd(output):
    """Interactively generate an openbench.yaml config file."""
    click.echo(f"Not yet implemented. Output: {output}")
