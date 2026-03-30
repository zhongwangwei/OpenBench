"""openbench run command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Check only, don't execute.")
@click.option("--cores", type=int, default=None, help="Override number of CPU cores.")
@click.option("--variables", multiple=True, help="Run only specified variables.")
@click.option("--remote", default=None, help="Remote host or saved profile name.")
def run(config, dry_run, cores, variables, remote):
    """Run evaluation from a config file."""
    click.echo(f"Not yet implemented. Config: {config}")
