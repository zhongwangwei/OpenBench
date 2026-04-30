"""openbench check command."""

import click

from openbench.cli._reference_errors import emit_reference_resolution_error


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

    # Count total resolved entries (sum of single + list values per variable)
    _n_ref_total = sum(
        1 if isinstance(v, str) else len(v) for v in cfg.reference.sources.values()
    )
    _n_ref_vars = len(cfg.reference.sources)
    _ref_summary = (
        f"{_n_ref_total} sources for {_n_ref_vars} variables"
        if _n_ref_total != _n_ref_vars else f"{_n_ref_vars} sources"
    )
    click.secho(f"\nReference data ({_ref_summary}):", bold=True)
    from openbench.config.resolver import resolve_all_references
    from openbench.data.registry.manager import get_registry

    mgr = get_registry()
    strict = cfg.project.strict_reference
    has_errors = False

    try:
        resolved = resolve_all_references(cfg, mgr, strict=strict)
    except Exception as e:
        emit_reference_resolution_error(str(e), prefix="  ✗ ")
        raise SystemExit(1)

    for r in resolved:
        if r.status == "ok":
            if r.resolved_name != r.source_name:
                click.secho(
                    f"  ✓ {r.var_name} → {r.source_name} → {r.resolved_name} "
                    f"({r.ref_ds.data_type}, {r.ref_ds.tim_res}, {r.ref_ds.grid_res}°)",
                    fg="cyan",
                )
            else:
                click.secho(
                    f"  ✓ {r.var_name} → {r.source_name} ({r.ref_ds.data_type}, {r.ref_ds.tim_res})",
                    fg="green",
                )
            # Warn about low/medium-confidence time-spatial fields
            from openbench.config.resolver import PROVENANCE_LOW, PROVENANCE_MEDIUM
            ds_prov = getattr(r.ref_ds, "_provenance", None) or {}
            for fld in ("tim_res", "grid_res"):
                source = ds_prov.get(fld)
                if not source:
                    continue
                value = getattr(r.ref_ds, fld, "?")
                if source in PROVENANCE_LOW:
                    if strict:
                        click.secho(
                            f"    ✗ {fld}: {value} (unconfirmed default)",
                            fg="red",
                        )
                        has_errors = True
                    else:
                        click.secho(
                            f"    ⚠ {fld}: {value} (default — not confirmed from NC or profile)",
                            fg="yellow",
                        )
                elif source in PROVENANCE_MEDIUM:
                    click.secho(
                        f"    ~ {fld}: {value} (inferred from directory structure)",
                        fg="cyan",
                    )
        elif r.status == "no_variable":
            click.secho(f"  ✗ {r.var_name} → {r.resolved_name}: {r.message}", fg="red")
            has_errors = True
        elif r.status == "ambiguous":
            click.secho(f"  ✗ {r.var_name} → {r.source_name}", fg="red")
            click.echo(f"    {r.message}")
            has_errors = True
        elif r.status == "not_found":
            if r.source_name:
                click.secho(
                    f"  ⚠ {r.var_name} → {r.source_name} "
                    "(not in registry, runtime will fall back to minimal defaults)",
                    fg="yellow",
                )
            else:
                click.secho(f"  ✗ {r.var_name}: no reference configured", fg="red")
                has_errors = True

    click.secho(f"\nSimulation data ({len(cfg.simulation)} models):", bold=True)
    for label, entry in cfg.simulation.items():
        click.secho(f"  ✓ {label} (model: {entry.model}, root: {entry.root_dir})", fg="green")

    if cfg.metrics:
        click.secho(f"\nMetrics: {', '.join(cfg.metrics)}", bold=True)
    if cfg.scores:
        click.secho(f"Scores: {', '.join(cfg.scores)}", bold=True)

    click.secho("\nOptions:", bold=True)
    click.secho(f"  Time alignment: {cfg.project.time_alignment}")
    click.secho(f"  Unified mask: {cfg.project.unified_mask}")
    click.secho(f"  Comparison: {cfg.comparison.enabled}")
    click.secho(f"  Statistics: {cfg.statistics.enabled}")

    if has_errors:
        click.secho("\n✗ Config has errors. Please fix and re-check.", fg="red", bold=True)
        raise SystemExit(1)

    n_refs = len(resolved) if resolved else 0
    n_sims = len(cfg.simulation) if cfg.simulation else 0
    n_vars = len(cfg.evaluation.variables) if cfg.evaluation.variables else 0
    click.secho(
        f"\n✓ Config valid ({n_vars} variables, {n_refs} references, {n_sims} simulations). Ready to run.",
        fg="green", bold=True,
    )
