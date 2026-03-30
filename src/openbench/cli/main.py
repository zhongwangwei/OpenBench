"""OpenBench CLI entry point.

Uses lazy command loading to avoid importing all submodules at startup.
"""

import importlib

import click

from openbench import __version__


class LazyGroup(click.Group):
    """Click group that lazily loads subcommands on first use."""

    COMMAND_MAP = {
        "run": "openbench.cli.run:run",
        "check": "openbench.cli.check:check",
        "data": "openbench.cli.data:data",
        "model": "openbench.cli.model:model",
        "migrate": "openbench.cli.migrate:migrate",
        "init": "openbench.cli.init_cmd:init_cmd",
        "gui": "openbench.cli.gui:gui",
    }

    def list_commands(self, ctx):
        return ["run", "check", "init", "data", "model", "migrate", "gui", "version"]

    def get_command(self, ctx, cmd_name):
        if cmd_name == "version":
            return version

        if cmd_name in self.COMMAND_MAP:
            module_path, attr = self.COMMAND_MAP[cmd_name].rsplit(":", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, attr)

        return None


@click.group(cls=LazyGroup)
@click.version_option(version=__version__, prog_name="openbench")
def cli():
    """OpenBench: Land Surface Model Benchmarking System."""


@cli.command()
def version():
    """Show version information."""
    click.echo(f"openbench {__version__}")
