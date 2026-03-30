"""openbench model commands."""

import click


@click.group()
def model():
    """Manage model profiles."""


@model.command("list")
def list_models():
    """List all available model profiles."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    models = mgr.list_models()

    click.secho(f"{'Name':<20} {'Type':<6} {'Res':<8} {'Variables'}", bold=True)
    click.echo("─" * 60)
    for m in models:
        res = f"{m.grid_res}°" if m.grid_res else "?"
        click.echo(f"{m.name:<20} {m.data_type:<6} {res:<8} {len(m.variables)}")

    click.echo(f"\nTotal: {len(models)} model profiles")


@model.command()
@click.argument("name")
def show(name):
    """Show variable mappings for a model profile."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    m = mgr.get_model(name)
    if m is None:
        click.secho(f"Model profile not found: {name}", fg="red")
        raise SystemExit(1)

    click.secho(f"{m.name}", bold=True)
    click.echo(f"Description: {m.description}")
    click.echo(f"Type: {m.data_type}, Resolution: {m.grid_res}°, Time: {m.tim_res}")
    click.echo()
    click.secho(f"{'Variable':<35} {'File varname':<20} {'Unit'}", bold=True)
    click.echo("─" * 70)
    for var_name, mapping in sorted(m.variables.items()):
        click.echo(f"{var_name:<35} {mapping.varname:<20} {mapping.varunit}")


@model.command()
def create():
    """Interactively create a new model profile."""
    click.echo("Not yet implemented.")
