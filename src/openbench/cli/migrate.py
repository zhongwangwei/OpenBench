"""openbench migrate command."""

import click


@click.command()
@click.argument("old_config", type=click.Path(exists=True))
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def migrate(old_config, output):
    """Convert old JSON/NML config to unified YAML."""
    from pathlib import Path

    from openbench.config.migration import migrate_config

    try:
        result = migrate_config(Path(old_config), Path(output))
    except Exception as e:
        click.secho(f"Migration failed: {e}", fg="red")
        raise SystemExit(1)

    click.secho(f"✓ Read {result['files_read']} config files", fg="green")
    click.secho(f"✓ {len(result['variables'])} evaluation variables", fg="green")
    click.secho(f"✓ {len(result['simulations'])} simulation models", fg="green")
    click.secho(f"✓ Written to {output}", fg="green", bold=True)
    click.echo(f"\nNext: openbench check {output}")
