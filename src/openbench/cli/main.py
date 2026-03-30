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


# Register sub-commands
from openbench.cli.check import check  # noqa: E402
from openbench.cli.data import data  # noqa: E402
from openbench.cli.gui import gui  # noqa: E402
from openbench.cli.init_cmd import init_cmd  # noqa: E402
from openbench.cli.migrate import migrate  # noqa: E402
from openbench.cli.model import model  # noqa: E402
from openbench.cli.run import run  # noqa: E402

cli.add_command(run)
cli.add_command(check)
cli.add_command(data)
cli.add_command(model)
cli.add_command(migrate)
cli.add_command(init_cmd)
cli.add_command(gui)
