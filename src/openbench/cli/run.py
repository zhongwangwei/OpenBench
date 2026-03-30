"""openbench run command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Check only, don't execute.")
@click.option("--cores", type=int, default=None, help="Override number of CPU cores.")
@click.option("--variables", multiple=True, help="Run only specified variables.")
@click.option("--remote", default=None, help="Remote host or saved profile name.")
def run(config, dry_run, cores, variables, remote):
    """Run evaluation from a config file."""
    from openbench.config import ConfigError, load_config

    # Load and validate config
    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"Config error: {e}", fg="red")
        raise SystemExit(1)

    # Apply CLI overrides
    if cores:
        cfg.options.num_cores = cores
    if variables:
        cfg.evaluation.variables = list(variables)

    if dry_run:
        click.secho("Dry run — config valid, would evaluate:", bold=True)
        click.echo(f"  Project: {cfg.project.name}")
        click.echo(f"  Variables: {', '.join(cfg.evaluation.variables)}")
        click.echo(f"  Simulations: {', '.join(cfg.simulation.keys())}")
        click.echo(f"  Metrics: {cfg.metrics or 'all'}")
        return

    if remote:
        click.echo("Remote execution not yet implemented.")
        click.echo("Install openbench[remote] and use openbench gui for remote execution.")
        raise SystemExit(1)

    # Run evaluation
    from openbench.runner.local import run_evaluation

    click.secho(f"Running evaluation: {cfg.project.name}", bold=True)
    results = run_evaluation(cfg)

    click.secho("\n✓ Evaluation complete", fg="green", bold=True)
    click.echo(f"  Output: {results['output_dir']}")
    click.echo(f"  Variables: {len(results['variables'])}")
    click.echo(f"  Simulations: {len(results['simulations'])}")
