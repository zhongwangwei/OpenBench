"""openbench run command."""

import logging
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from openbench.cli._names import resolve_variable_filters
from openbench.cli._options import remote_not_implemented_message
from openbench.cli._reference_errors import emit_reference_resolution_error
from openbench.cli._simulation_validation import simulation_root_errors


@click.command()
@click.argument("config", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Check only, don't execute.")
@click.option("--cores", type=int, default=None, help="Override number of CPU cores.")
@click.option(
    "--variable",
    "--variables",
    "variables",
    multiple=True,
    help="Run only specified variable (repeatable). --variables retained as alias.",
)
@click.option(
    "--remote",
    default=None,
    help="[NOT IMPLEMENTED in CLI — use 'openbench gui' for remote runs] Remote host or saved profile name.",
)
@click.option("--dump-config", is_flag=True, help="Write intermediate runner/debug configs to output dir.")
@click.option("--comparison-only", is_flag=True, help="Skip evaluation, only run comparisons on existing results.")
@click.option("--force", is_flag=True, help="Bypass incremental cache and re-run evaluations.")
@click.option("--output-dir", type=click.Path(file_okay=False), default=None, help="Override project.output_dir.")
def run(config, dry_run, cores, variables, remote, dump_config, comparison_only, force, output_dir):
    """Run evaluation from a config file.

    Performance settings live in ``project.io`` and ``project.dask``:
    NetCDF compression, multi-file NetCDF batch combine, and optional
    dask.distributed workers can be configured in YAML. Environment variables
    such as OPENBENCH_NETCDF_COMPRESSION, OPENBENCH_MFDATASET_BATCH_SIZE, and
    OPENBENCH_DASK override YAML for temporary benchmarking.
    """
    from openbench.config import ConfigError, load_config

    # Load and validate config
    try:
        cfg = load_config(config)
    except ConfigError as e:
        raise click.ClickException(f"Config error: {e}") from e

    if remote:
        raise click.ClickException(remote_not_implemented_message(remote))

    _expand_config_paths(cfg)

    # Apply CLI overrides
    if cores is not None:
        if cores < 1:
            raise click.ClickException(f"--cores must be >= 1, got {cores}")
        cfg.project.num_cores = cores
    if output_dir is not None:
        cfg.project.output_dir = _expand_path_value(output_dir)
    if variables:
        cfg.evaluation.variables = resolve_variable_filters(variables, cfg.evaluation.variables)

    missing_refs = [var for var in cfg.evaluation.variables if not cfg.reference.sources.get(var)]
    if missing_refs:
        raise click.ClickException("Config error:\n  missing reference source for: " + ", ".join(missing_refs))

    if comparison_only and not cfg.comparison.enabled:
        raise click.ClickException("--comparison-only requires comparison.enabled: true in the config")
    if comparison_only and cfg.project.only_drawing:
        raise click.ClickException("--comparison-only conflicts with project.only_drawing=true; choose one mode")

    _run_static_preflight(cfg)

    output_only = comparison_only or cfg.project.only_drawing
    if not output_only:
        sim_errors = simulation_root_errors(cfg)
        if sim_errors:
            details = "\n  ".join(f"{label}: {message}" for label, message in sim_errors)
            raise click.ClickException(f"Config error:\n  {details}")
        fulllist_errors = _simulation_fulllist_errors(cfg)
        if fulllist_errors:
            details = "\n  ".join(fulllist_errors)
            raise click.ClickException(f"Config error:\n  {details}")

    resolved = _resolve_references_for_run(cfg)
    _run_model_preflight(cfg)

    if dry_run and comparison_only:
        _validate_comparison_only_dry_run(cfg)
    elif dry_run and cfg.project.only_drawing:
        _validate_existing_outputs_dry_run(cfg)

    if dry_run:
        click.secho("Dry run — config valid, would evaluate:", bold=True)
        click.echo(f"  Project: {cfg.project.name}")
        click.echo(f"  Variables: {', '.join(cfg.evaluation.variables)}")
        click.echo(f"  Simulations: {', '.join(cfg.simulation.keys())}")
        click.echo(f"  Metrics: {cfg.metrics or 'all'}")
        click.echo(f"  Force: {force or cfg.project.force}")
        click.echo(f"  Time alignment: {cfg.project.time_alignment}")
        click.echo(f"  Unified mask: {cfg.project.unified_mask}")
        for r in resolved or []:
            status_icon = "✓" if r.status == "ok" else "⚠" if r.status == "not_found" else "✗"
            click.echo(f"  {status_icon} {r.var_name} → {r.resolved_name} [{r.provenance or r.status}]")
        if dump_config:
            click.echo("  Debug config dump skipped in dry-run mode.")
        return

    # Dump intermediate config only after the same reference resolver path has
    # succeeded, so failed configs and dry-runs do not leave debug artifacts behind.
    if dump_config:
        _dump_debug_config(cfg)

    # Run evaluation
    from openbench.runner.local import run_evaluation

    if comparison_only:
        click.secho(f"Running comparisons only: {cfg.project.name}", bold=True)
    else:
        click.secho(f"Running evaluation: {cfg.project.name}", bold=True)
    with _run_file_logging(cfg):
        try:
            results = run_evaluation(cfg, force=force, comparison_only=comparison_only)
        except Exception as e:
            logging.getLogger(__name__).exception("Evaluation failed")
            raise click.ClickException(f"Evaluation failed: {e}") from e

    status = results.get("status", "success")
    if status == "success":
        click.secho("\n✓ Evaluation complete", fg="green", bold=True)
        click.echo(f"  Output: {results.get('output_dir', 'unknown')}")
        click.echo(f"  Variables: {len(results.get('variables', []))}")
        click.echo(f"  Simulations: {len(results.get('simulations', []))}")
        return

    heading = "Evaluation completed with errors" if status == "partial" else "Evaluation failed"
    click.secho(f"\n✗ {heading}", fg="red", bold=True)
    click.echo(f"  Output: {results.get('output_dir', 'unknown')}")
    for error in results.get("errors", []):
        phase = error.get("phase", "evaluation")
        message = error.get("message", "unknown error")
        click.echo(f"  - [{phase}] {message}")
    # Per-error context emitted above; exit silently with non-zero status.
    raise SystemExit(1)


def _expand_path_value(value):
    if value is None:
        return None
    return os.path.expandvars(os.path.expanduser(str(value)))


def _expand_config_paths(cfg):
    """Expand shell-style env vars and user paths before preflight/runner use."""
    cfg.project.output_dir = _expand_path_value(cfg.project.output_dir)
    if cfg.reference.data_root:
        cfg.reference.data_root = _expand_path_value(cfg.reference.data_root)
    for entry in cfg.simulation.values():
        entry.root_dir = _expand_path_value(entry.root_dir)
        if entry.fulllist:
            entry.fulllist = _expand_path_value(entry.fulllist)
        for inline in (entry.variables or {}).values():
            if isinstance(inline, dict) and inline.get("fulllist"):
                inline["fulllist"] = _expand_path_value(inline["fulllist"])


def _run_static_preflight(cfg):
    """Run non-data-mutating checks shared with openbench check."""
    from openbench.cli.check import _config_findings, _groupby_static_dataset_findings

    errors, _warnings = _config_findings(cfg)
    errors.extend(_groupby_static_dataset_findings(cfg))
    if errors:
        details = "\n  ".join(errors)
        raise click.ClickException(f"Config error:\n  {details}")


def _run_model_preflight(cfg):
    """Run registry-backed simulation model checks shared with openbench check."""
    from openbench.cli.check import _simulation_model_error_messages
    from openbench.data.registry.manager import get_registry

    errors = _simulation_model_error_messages(cfg, get_registry())
    if errors:
        details = "\n  ".join(errors)
        raise click.ClickException(f"Config error:\n  {details}")


def _simulation_fulllist_errors(cfg):
    """Validate station simulation list files without requiring registry access."""
    from openbench.cli.check import _fulllist_path_findings

    errors = []
    for label, entry in cfg.simulation.items():
        if entry.fulllist:
            list_errors, _warnings = _fulllist_path_findings(
                str(entry.fulllist),
                f"simulation.{label}.fulllist",
                entry.root_dir,
            )
            errors.extend(list_errors)
        for var_name, inline in (entry.variables or {}).items():
            if not isinstance(inline, dict) or not inline.get("fulllist"):
                continue
            list_errors, _warnings = _fulllist_path_findings(
                str(inline["fulllist"]),
                f"simulation.{label}.variables.{var_name}.fulllist",
                entry.root_dir,
            )
            errors.extend(list_errors)
    return errors


def _resolve_references_for_run(cfg):
    """Run the resolver path used by real execution and format failures for CLI."""
    from openbench.config.resolver import resolve_all_references
    from openbench.data.registry.manager import get_registry

    try:
        resolved = resolve_all_references(cfg, get_registry(), strict=cfg.project.strict_reference)
    except Exception as e:
        # Multi-line context (resolver hint + remediation) already emitted; exit silently.
        emit_reference_resolution_error(str(e), prefix="Reference resolution failed: ")
        raise SystemExit(1) from e

    problems = [r for r in resolved if getattr(r, "status", "ok") != "ok"]
    if problems:
        lines = []
        for r in problems:
            if r.status == "not_found" and r.source_name:
                lines.append(
                    f"{r.var_name} → {r.source_name} "
                    "(not in registry; runtime fallback is disabled for CLI run preflight)"
                )
            elif r.status == "not_found":
                lines.append(f"{r.var_name}: no reference configured")
            elif r.status == "no_variable":
                lines.append(f"{r.var_name} → {r.resolved_name}: {r.message}")
            else:
                lines.append(f"{r.var_name} → {r.source_name}: {r.message}")
        emit_reference_resolution_error("Reference resolution errors:\n  - " + "\n  - ".join(lines))
        raise SystemExit(1)

    return resolved


def _validate_comparison_only_dry_run(cfg):
    from openbench.runner.local import comparison_only_preflight_errors

    errors = comparison_only_preflight_errors(cfg)
    if errors:
        details = "\n  ".join(error.get("message", "unknown error") for error in errors)
        raise click.ClickException(f"Config error:\n  {details}")


def _validate_existing_outputs_dry_run(cfg):
    from openbench.runner.local import existing_output_preflight_errors

    errors = existing_output_preflight_errors(cfg)
    if errors:
        details = "\n  ".join(error.get("message", "unknown error") for error in errors)
        raise click.ClickException(f"Config error:\n  {details}")


@contextmanager
def _run_file_logging(cfg):
    """Attach a DEBUG file log for one real run invocation."""
    log_dir = (Path(cfg.project.output_dir) / cfg.project.name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    previous_handler_levels = {handler: handler.level for handler in root_logger.handlers}

    handler = None
    handler_added = False
    try:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== run started at {datetime.now().isoformat(timespec='seconds')} ===\n")
        for existing in root_logger.handlers:
            if existing.level == logging.NOTSET:
                existing.setLevel(previous_root_level or logging.INFO)
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(handler)
        handler_added = True
        logging.getLogger(__name__).debug("Run log started: %s", log_path)
        yield log_path
    finally:
        if handler is not None:
            if handler_added:
                root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(previous_root_level)
        for existing, level in previous_handler_levels.items():
            existing.setLevel(level)


def _dump_debug_config(cfg):
    """Write intermediate runner/debug configs to output dir."""
    import os

    import yaml

    from openbench.config.adapter import build_runner_bindings

    bindings = build_runner_bindings(cfg)
    runner_cfg = bindings.runner_cfg
    namelists = bindings.namelists
    figures = bindings.figures

    basename = str(runner_cfg.basename)
    output_dir = Path(runner_cfg.basedir) / basename
    dump_dir = output_dir / "debug"
    os.makedirs(dump_dir, exist_ok=True)

    files = {
        "runner_config.yaml": asdict(runner_cfg),
        "main_nl.yaml": namelists.main,
        "ref_nml.yaml": namelists.reference,
        "sim_nml.yaml": namelists.simulation,
        "fig_nml.yaml": figures.raw,
    }

    for filename, data in files.items():
        path = dump_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.secho(f"Debug configs written to {dump_dir}/", fg="cyan")
    for filename in files:
        click.echo(f"  {filename}")
