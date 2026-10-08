"""OpenBench CLI entry point.

Uses lazy command loading to avoid importing all submodules at startup.
"""

import importlib
import io
import sys

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
        "registry": "openbench.cli.registry_cmd:registry",
    }

    def main(self, *args, **kwargs):
        # Redirected Windows streams may use a codec without our status glyphs.
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, io.TextIOWrapper) and stream.errors in {"strict", "surrogateescape", "surrogatepass"}:
                stream.reconfigure(errors="backslashreplace")
        return super().main(*args, **kwargs)

    def list_commands(self, ctx):
        preferred = [
            "run",
            "check",
            "smoke-test",
            "init",
            "ref",
            "sim",
            "model",
            "registry",
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

    def parse_args(self, ctx, args):
        # The group callback runs before subcommand arguments are parsed; keep
        # them so it can tell read-only invocations apart.
        ctx.meta["openbench.argv"] = list(args)
        return super().parse_args(ctx, args)


# Only commands that may write the registry overlay compact it first; help, dry
# runs, inspection, bare groups and mistyped subcommands leave it untouched.
_OVERLAY_WRITING_COMMANDS = frozenset({"gui", "init", "run"})
_OVERLAY_WRITING_SUBCOMMANDS = {
    "model": frozenset({"delete", "import", "register", "remove-var", "rename"}),
    "ref": frozenset({"delete", "register", "register-profile", "scan"}),
    "sim": frozenset({"scan"}),
}


def _may_compact_registry(argv: list[str]) -> bool:
    """True only for commands that may write the registry overlay."""
    if not argv or any(arg in ("--help", "-h") or arg.split("=", 1)[0] == "--dry-run" for arg in argv):
        return False
    command = argv[0]
    if command in _OVERLAY_WRITING_COMMANDS:
        return True
    subcommand = next((arg for arg in argv[1:] if not arg.startswith("-")), None)
    return subcommand in _OVERLAY_WRITING_SUBCOMMANDS.get(command, frozenset())


@click.group(cls=LazyGroup)
@click.version_option(version=__version__, prog_name="openbench", message="openbench %(version)s")
def cli():
    """OpenBench: Land Surface Model Benchmarking System."""
    ctx = click.get_current_context()
    if not _may_compact_registry(ctx.meta.get("openbench.argv", [])):
        # Help, dry runs and inspection observe the file as it is: no compaction
        # here or through get_registry() during this command.
        try:
            from openbench.data.registry.overlay_audit import suspend_sync

            ctx.call_on_close(suspend_sync())
        except Exception:
            pass
        return
    # Once per overlay file, remove redundant current defaults without changing
    # user overrides; then a throttled, silent-when-clean notice
    # if the overlay still shadows the bundled catalog. Never raises.
    try:
        from openbench.data.registry.overlay_audit import (
            format_sync_notice,
            maybe_emit_overlay_notice,
            sync_legacy_overlays,
        )

        notice = format_sync_notice(sync_legacy_overlays())
        if notice:
            click.echo(notice, err=True)
        maybe_emit_overlay_notice()
    except Exception:
        pass


@cli.command()
def version():
    """Show version information."""
    click.echo(f"openbench {__version__}")
