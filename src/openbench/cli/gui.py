"""openbench gui command."""

import click

from openbench.cli._options import remote_not_implemented_message


@click.command()
@click.argument("config", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False)
@click.option("--remote", default=None, help="[NOT IMPLEMENTED] Remote host or saved profile name.")
def gui(config, remote):
    """Launch the OpenBench graphical interface."""
    if remote:
        raise click.ClickException(remote_not_implemented_message(remote))

    try:
        from openbench.gui import _check_gui_deps

        _check_gui_deps()
    except ImportError as e:
        raise click.ClickException(str(e))

    from openbench.gui.app import launch

    launch(config_path=config)
