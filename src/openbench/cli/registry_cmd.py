"""``openbench registry`` — inspect and re-sparse the user registry overlay.

The user overlay (``~/.openbench/...``) deep-merges on top of the bundled
catalog and should contain only sparse deltas. A bloated overlay (e.g. a legacy
full snapshot) silently shadows bundled fixes. These commands make that visible
and let you re-sparse it safely.
"""

from __future__ import annotations

import click

from openbench.data.registry import overlay_audit as oa


@click.group()
def registry() -> None:
    """Inspect and re-sparse the user registry overlay."""


def _print_catalog_report(cat) -> bool:
    """Print one catalog's classification. Return True if it has bloat."""
    redundant = cat.by_kind(oa.REDUNDANT)
    stale = cat.by_kind(oa.STALE_FULLCOPY)
    delta = cat.by_kind(oa.DELTA)
    custom = cat.by_kind(oa.CUSTOM)

    click.secho(f"\n{cat.label}: {cat.overlay_path}", bold=True)
    if not cat.entries:
        click.echo("  (overlay empty — fully tracking bundled catalog) ✓")
        return False

    if redundant:
        click.secho(
            f"  ● {len(redundant)} redundant (identical to bundled — safe to drop)",
            fg="yellow",
        )
    if stale:
        click.secho(
            f"  ● {len(stale)} stale full-cop{'y' if len(stale) == 1 else 'ies'} "
            f"(shadow bundled and differ — likely outdated):",
            fg="red",
        )
        for e in stale:
            fields = ", ".join(sorted(e.minimal.get("variables", e.minimal).keys()))
            click.echo(f"      - {e.name}  (overrides: {fields})")
    if delta:
        click.secho(f"  ● {len(delta)} deliberate delta(s) (minimal overrides — kept):", fg="green")
        for e in delta:
            fields = ", ".join(sorted(e.minimal.get("variables", e.minimal).keys()))
            click.echo(f"      - {e.name}  (overrides: {fields})")
    if custom:
        noun = "entry" if len(custom) == 1 else "entries"
        click.secho(f"  ● {len(custom)} custom {noun} (not in bundled — kept):", fg="cyan")
        for e in custom:
            click.echo(f"      - {e.name}")

    return bool(redundant or stale)


@registry.command(name="diff")
def diff_cmd() -> None:
    """Show how the user overlay diverges from the bundled catalog."""
    audit = oa.audit_overlays()
    has_bloat = False
    for cat in audit.catalogs:
        has_bloat = _print_catalog_report(cat) or has_bloat

    n_delta = sum(len(c.by_kind(oa.DELTA)) for c in audit.catalogs)
    click.echo()
    if has_bloat:
        click.secho(
            "Your overlay shadows bundled entries. Run `openbench registry prune` to "
            "re-sparse it (behavior-preserving: drops redundant copies, reduces stale "
            "ones to minimal overrides). Review the remaining overrides afterwards.",
            fg="yellow",
        )
    elif n_delta:
        click.secho(
            f"No snapshot bloat. {n_delta} deliberate override(s) still shadow bundled "
            "(listed above) — delete an entry from the overlay to take the bundled value.",
            fg="green",
        )
    else:
        click.secho("Overlay is clean — fully tracking the bundled catalog. ✓", fg="green")


# Alias: `registry status` behaves like `registry diff`.
registry.add_command(diff_cmd, name="status")


@registry.command(name="prune")
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def prune_cmd(dry_run: bool, yes: bool) -> None:
    """Re-sparse the overlay: drop redundant entries, minimize stale full-copies.

    Behavior-preserving — the merged registry is identical before and after.
    Each modified overlay file is backed up first.
    """
    audit = oa.audit_overlays()
    if audit.bloat_count == 0:
        click.secho("Overlay already sparse — nothing to prune. ✓", fg="green")
        return

    if not dry_run and not yes:
        click.echo(
            f"This will re-sparse {audit.bloat_count} overlay entr"
            f"{'y' if audit.bloat_count == 1 else 'ies'} (a backup is made first)."
        )
        click.confirm("Proceed?", abort=True)

    results = oa.prune_overlays(dry_run=dry_run)
    verb = "Would remove" if dry_run else "Removed"
    verb2 = "would minimize" if dry_run else "minimized"
    for res in results:
        if not (res.removed or res.minimized):
            continue
        click.secho(f"\n{res.label}:", bold=True)
        click.echo(
            f"  {verb} {len(res.removed)} redundant; {verb2} {len(res.minimized)} stale full-cop"
            f"{'y' if len(res.minimized) == 1 else 'ies'}."
        )
        if res.minimized:
            click.echo(f"    minimized: {', '.join(res.minimized)}")
        if res.backup:
            click.echo(f"    backup: {res.backup}")

    click.echo()
    if dry_run:
        click.secho("Dry run — no files written. Re-run without --dry-run to apply.", fg="yellow")
    else:
        click.secho("Done. Run `openbench registry diff` to review remaining overrides.", fg="green")
