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
        "ref": "openbench.cli.data:data",
        "sim": "openbench.cli.sim:sim",
        "model": "openbench.cli.model:model",
        "migrate": "openbench.cli.migrate:migrate",
        "init": "openbench.cli.init_cmd:init_cmd",
        "gui": "openbench.cli.gui:gui",
        "cache": "openbench.cli.cache:cache",
        "smoke-test": "openbench.cli.smoke:smoke_test",
    }

    def list_commands(self, ctx):
        preferred = [
            "run",
            "check",
            "smoke-test",
            "init",
            "ref",
            "sim",
            "model",
            "migrate",
            "cache",
            "gui",
            "version",
        ]
        names = set(super().list_commands(ctx)) | set(self.COMMAND_MAP)
        return [name for name in preferred if name in names] + sorted(names - set(preferred))

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.COMMAND_MAP:
            module_path, attr = self.COMMAND_MAP[cmd_name].rsplit(":", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, attr)

        return super().get_command(ctx, cmd_name)


@click.group(cls=LazyGroup)
@click.version_option(version=__version__, prog_name="openbench")
def cli():
    """OpenBench: Land Surface Model Benchmarking System."""


@cli.command()
def version():
    """Show version information."""
    click.echo(f"openbench {__version__}")
