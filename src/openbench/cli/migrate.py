"""openbench migrate command."""

import click


@click.command()
@click.argument("old_config", type=click.Path(exists=True))
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def migrate(old_config, output):
    """Convert old JSON/NML config to unified YAML."""
    click.echo(f"Not yet implemented. Input: {old_config}, Output: {output}")
