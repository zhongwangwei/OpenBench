"""openbench gui command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option("--remote", is_flag=True, help="Start in remote mode.")
def gui(config, remote):
    """Launch the OpenBench graphical interface."""
    try:
        from openbench.gui import _check_gui_deps

        _check_gui_deps()
    except ImportError as e:
        raise click.ClickException(str(e))

    click.echo(f"Not yet implemented. Config: {config}")
