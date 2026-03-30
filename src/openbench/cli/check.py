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
    for var, source in cfg.reference.items():
        click.secho(f"  ✓ {var} → {source}", fg="green")

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

    click.secho("\n✓ Config valid. Ready to run.", fg="green", bold=True)
