"""openbench check command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
def check(config):
    """Validate config file and check data availability."""
    from openbench.config import ConfigError, load_config

    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"  ✗ {e}", fg="red")
        raise SystemExit(1)

    click.secho("Config validation:", bold=True)
    click.secho("  ✓ YAML syntax valid", fg="green")
    click.secho("  ✓ Schema validation passed", fg="green")
    click.secho(
        f"  ✓ Year range [{cfg.project.years[0]}, {cfg.project.years[1]}] valid",
        fg="green",
    )

    click.secho(f"\nReference data ({len(cfg.reference)} sources):", bold=True)
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    has_errors = False

    for var, source in cfg.reference.items():
        ref = mgr.get_reference(source)
        if ref is not None:
            click.secho(f"  ✓ {var} → {source} ({ref.data_type}, {ref.tim_res})", fg="green")
        else:
            # Check if it's a base name with resolution variants
            variants = mgr.get_resolution_variants(source)
            if variants:
                click.secho(f"  ✗ {var} → {source}", fg="red")
                click.echo(f"    '{source}' has multiple resolutions. Please specify one:")
                for label, v in sorted(variants.items()):
                    click.echo(f"      {v.name}  ({v.data_type}, {v.tim_res}, {v.grid_res}°)")
                has_errors = True
            else:
                click.secho(f"  ⚠ {var} → {source} (not in registry, will use inline config)", fg="yellow")

    click.secho(f"\nSimulation data ({len(cfg.simulation)} models):", bold=True)
    for label, entry in cfg.simulation.items():
        click.secho(f"  ✓ {label} (model: {entry.model}, root: {entry.root_dir})", fg="green")

    if cfg.metrics:
        click.secho(f"\nMetrics: {', '.join(cfg.metrics)}", bold=True)
    if cfg.scores:
        click.secho(f"Scores: {', '.join(cfg.scores)}", bold=True)

    click.secho("\nOptions:", bold=True)
    click.secho(f"  Time alignment: {cfg.options.time_alignment}")
    click.secho(f"  Unified mask: {cfg.options.unified_mask}")
    click.secho(f"  Comparison: {cfg.comparison.enabled}")
    click.secho(f"  Statistics: {cfg.statistics.enabled}")

    if has_errors:
        click.secho("\n✗ Config has errors. Please fix and re-check.", fg="red", bold=True)
        raise SystemExit(1)

    click.secho("\n✓ Config valid. Ready to run.", fg="green", bold=True)
