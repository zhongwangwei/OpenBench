"""openbench run command."""

from dataclasses import asdict

import click

from openbench.cli._reference_errors import emit_reference_resolution_error


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Check only, don't execute.")
@click.option("--cores", type=int, default=None, help="Override number of CPU cores.")
@click.option("--variables", multiple=True, help="Run only specified variables.")
@click.option("--remote", default=None,
              help="[NOT IMPLEMENTED in CLI — use 'openbench gui' for remote runs] Remote host or saved profile name.")
@click.option("--dump-config", is_flag=True, help="Write intermediate runner/debug configs to output dir.")
@click.option("--comparison-only", is_flag=True, help="Skip evaluation, only run comparisons on existing results.")
def run(config, dry_run, cores, variables, remote, dump_config, comparison_only):
    """Run evaluation from a config file."""
    from openbench.config import ConfigError, load_config

    # Load and validate config
    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"Config error: {e}", fg="red")
        raise SystemExit(1)

    # Apply CLI overrides
    if cores is not None:
        if cores < 1:
            click.secho(f"--cores must be >= 1, got {cores}", fg="red")
            raise SystemExit(1)
        cfg.project.num_cores = cores
    if variables:
        cfg.evaluation.variables = list(variables)

    # Dump intermediate config if requested (works with --dry-run too)
    if dump_config:
        _dump_debug_config(cfg)

    if dry_run:
        # Run the same resolver path as real execution to catch binding errors early
        from openbench.config.resolver import resolve_all_references
        from openbench.data.registry.manager import get_registry

        try:
            resolved = resolve_all_references(cfg, get_registry(), strict=cfg.project.strict_reference)
        except Exception as e:
            emit_reference_resolution_error(str(e), prefix="Reference resolution failed: ")
            raise SystemExit(1)

        click.secho("Dry run — config valid, would evaluate:", bold=True)
        click.echo(f"  Project: {cfg.project.name}")
        click.echo(f"  Variables: {', '.join(cfg.evaluation.variables)}")
        click.echo(f"  Simulations: {', '.join(cfg.simulation.keys())}")
        click.echo(f"  Metrics: {cfg.metrics or 'all'}")
        for r in resolved:
            status_icon = "✓" if r.status == "ok" else "⚠" if r.status == "not_found" else "✗"
            click.echo(f"  {status_icon} {r.var_name} → {r.resolved_name} [{r.provenance or r.status}]")
        return

    if remote:
        click.echo("Remote execution not yet implemented.")
        click.echo("Install openbench[remote] and use openbench gui for remote execution.")
        raise SystemExit(1)

    # Run evaluation
    from openbench.runner.local import run_evaluation

    if comparison_only:
        click.secho(f"Running comparisons only: {cfg.project.name}", bold=True)
    else:
        click.secho(f"Running evaluation: {cfg.project.name}", bold=True)
    results = run_evaluation(cfg, comparison_only=comparison_only)

    status = results.get("status", "success")
    if status == "success":
        click.secho("\n✓ Evaluation complete", fg="green", bold=True)
        click.echo(f"  Output: {results['output_dir']}")
        click.echo(f"  Variables: {len(results['variables'])}")
        click.echo(f"  Simulations: {len(results['simulations'])}")
        return

    heading = "Evaluation completed with errors" if status == "partial" else "Evaluation failed"
    click.secho(f"\n✗ {heading}", fg="red", bold=True)
    click.echo(f"  Output: {results['output_dir']}")
    for error in results.get("errors", []):
        phase = error.get("phase", "evaluation")
        message = error.get("message", "unknown error")
        click.echo(f"  - [{phase}] {message}")
    raise SystemExit(1)


def _dump_debug_config(cfg):
    """Write intermediate runner/debug configs to output dir."""
    import os

    import yaml

    from openbench.config.adapter import build_runner_bindings

    bindings = build_runner_bindings(cfg)
    runner_cfg = bindings.runner_cfg
    namelists = bindings.namelists
    figures = bindings.figures

    output_dir = os.path.join(runner_cfg.basedir, runner_cfg.basename)
    dump_dir = os.path.join(output_dir, "debug")
    os.makedirs(dump_dir, exist_ok=True)

    files = {
        "runner_config.yaml": asdict(runner_cfg),
        "main_nl.yaml": namelists.main,
        "ref_nml.yaml": namelists.reference,
        "sim_nml.yaml": namelists.simulation,
        "fig_nml.yaml": figures.raw,
    }

    for filename, data in files.items():
        path = os.path.join(dump_dir, filename)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.secho(f"Debug configs written to {dump_dir}/", fg="cyan")
    for filename in files:
        click.echo(f"  {filename}")
