"""openbench run command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Check only, don't execute.")
@click.option("--cores", type=int, default=None, help="Override number of CPU cores.")
@click.option("--variables", multiple=True, help="Run only specified variables.")
@click.option("--remote", default=None, help="Remote host or saved profile name.")
@click.option("--dump-config", is_flag=True, help="Write intermediate legacy configs to output dir for debugging.")
def run(config, dry_run, cores, variables, remote, dump_config):
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

    # Dump intermediate config if requested (works with --dry-run too)
    if dump_config:
        _dump_legacy_config(cfg)

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


def _dump_legacy_config(cfg):
    """Write intermediate legacy namelists to output dir for debugging."""
    import os

    import yaml

    from openbench.config.adapter import build_fig_nml, build_legacy_namelists, to_legacy_config

    legacy = to_legacy_config(cfg)
    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)
    fig_nml = build_fig_nml()

    output_dir = os.path.join(legacy["general"]["basedir"], legacy["general"]["basename"])
    dump_dir = os.path.join(output_dir, "debug")
    os.makedirs(dump_dir, exist_ok=True)

    files = {
        "main_nl.yaml": main_nl,
        "ref_nml.yaml": ref_nml,
        "sim_nml.yaml": sim_nml,
        "fig_nml.yaml": fig_nml,
        "legacy_config.yaml": legacy,
    }

    for filename, data in files.items():
        path = os.path.join(dump_dir, filename)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.secho(f"Debug configs written to {dump_dir}/", fg="cyan")
    for filename in files:
        click.echo(f"  {filename}")
