"""openbench migrate command."""

import click


@click.command()
@click.argument("old_config", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
@click.option("--force", "-f", is_flag=True, help="Overwrite output file without prompting.")
def migrate(old_config, output, force):
    """Convert old JSON/NML config to unified YAML."""
    from pathlib import Path

    from openbench.config.migration import migrate_config

    output_path = Path(output)
    # Prevent silent destruction of an existing config. The default
    # output path is `openbench.yaml` in CWD, which is exactly the file
    # a user is most likely to have already created via `openbench init`.
    if output_path.exists() and not force:
        click.confirm(
            f"{output_path} already exists. Overwrite?",
            abort=True,
        )

    try:
        result = migrate_config(Path(old_config), output_path)
    except Exception as e:
        raise click.ClickException(f"Migration failed: {e}") from e

    click.secho(f"✓ Read {result['files_read']} config files", fg="green")
    click.secho(f"✓ {len(result['variables'])} evaluation variables", fg="green")
    click.secho(f"✓ {len(result['simulations'])} simulation models", fg="green")
    click.secho(f"✓ Written to {output}", fg="green", bold=True)
    click.echo(f"\nNext: openbench check {output}")
