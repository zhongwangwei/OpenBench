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


_PER_FILE_DETAIL = {
    oa.REDUNDANT: "identical to the bundled entry; hides its future fixes",
    oa.STALE_FULLCOPY: "copies the bundled entry and differs",
    oa.CUSTOM: "not in bundled",
}


def _print_catalog_report(cat) -> bool:
    """Print one catalog's classification. Return True if it has bloat."""
    redundant = cat.by_kind(oa.REDUNDANT)
    stale = cat.by_kind(oa.STALE_FULLCOPY)
    delta = cat.by_kind(oa.DELTA)
    custom = cat.by_kind(oa.CUSTOM)
    duplicates = cat.duplicates
    per_file = cat.per_file

    click.secho(f"\n{cat.label}: {cat.overlay_path}", bold=True)
    if cat.error:
        click.secho(f"  ✗ The overlay {cat.error}; the registry cannot use it. Fix or remove the file.", fg="red")
    if not cat.entries and not per_file:
        if not cat.error:
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
        click.secho(f"  ● {len(delta)} delta(s) (minimal overrides — kept):", fg="green")
        for e in delta:
            fields = ", ".join(sorted(e.minimal.get("variables", e.minimal).keys()))
            click.echo(f"      - {e.name}  (overrides: {fields})")
    if custom:
        noun = "entry" if len(custom) == 1 else "entries"
        click.secho(f"  ● {len(custom)} custom {noun} (not in bundled — kept):", fg="cyan")
        for e in custom:
            note = (
                "  (shipped by an older OpenBench, no longer bundled)"
                if oa.previously_bundled(cat.label, e.name, e.minimal)
                else ""
            )
            click.echo(f"      - {e.name}{note}")
    if duplicates:
        click.secho(
            f"  ● {len(duplicates)} keys differ only in case (merged in file order, later ones win — keep one):",
            fg="red",
        )
        for e in duplicates:
            click.echo(f"      - {e.name}")
    if per_file:
        click.secho(
            f"  ● {len(per_file)} entr{'y' if len(per_file) == 1 else 'ies'} in separate files "
            "(merged after the catalog — edit or remove the file to change them):",
            fg="cyan",
        )
        for e in per_file:
            detail = _PER_FILE_DETAIL.get(e.kind)
            if detail is None:
                detail = "overrides: " + ", ".join(sorted(e.minimal.get("variables", e.minimal).keys()))
            click.echo(f"      - {e.name}  ({e.path}; {detail})")

    for entry in cat.entries:
        if entry.historical_fields:
            verb = "matches" if len(entry.historical_fields) == 1 else "match"
            click.secho(
                f"  ⚠ {entry.name}: {', '.join(entry.historical_fields)} {verb} older bundled defaults; "
                "may be outdated. Values are preserved; review or run "
                f"{_reset_command(entry.name, cat.label)}.",
                fg="yellow",
            )
    for entry in per_file:
        if entry.historical_fields:
            verb = "matches" if len(entry.historical_fields) == 1 else "match"
            click.secho(
                f"  ⚠ {entry.name} in {entry.path}: {', '.join(entry.historical_fields)} {verb} older bundled "
                "defaults; may be outdated. Values are preserved; edit or remove the entry in that file.",
                fg="yellow",
            )

    return bool(redundant or stale)


def _reset_command(name: str, kind: str) -> str:
    """``openbench registry reset NAME``, with ``--kind`` only when the name is ambiguous."""
    from openbench.util.names import normalize_name

    audit_paths = oa._paths()
    in_both = all(
        any(normalize_name(key) == normalize_name(name) for key in oa._load_yaml(path))
        for path in (audit_paths[1], audit_paths[3])
    )
    return f"openbench registry reset {name}" + (f" --kind {kind}" if in_both else "")


@registry.command(name="diff")
def diff_cmd() -> None:
    """Show how the user overlay diverges from the bundled catalog."""
    audit = oa.audit_overlays()
    has_bloat = False
    for cat in audit.catalogs:
        has_bloat = _print_catalog_report(cat) or has_bloat

    n_delta = sum(len(c.by_kind(oa.DELTA)) for c in audit.catalogs)
    n_duplicates = audit.duplicate_count
    # Custom entries in separate files add datasets; only the others shadow bundled.
    n_files = sum(1 for c in audit.catalogs for e in c.per_file if e.kind != oa.CUSTOM)
    click.echo()
    if any(c.error for c in audit.catalogs):
        click.secho("An overlay file cannot be read (see ✗ above); fix or remove it.", fg="red")
    if n_duplicates:
        click.secho(
            "Some overlay keys differ only in case: keep one of each (edit the overlay), "
            "or `openbench registry reset NAME` to drop them all.",
            fg="yellow",
        )
    if has_bloat:
        click.secho(
            "Your overlay shadows bundled entries. Run `openbench registry prune` to "
            "re-sparse it (behavior-preserving: drops redundant copies, reduces stale "
            "ones to minimal overrides). Review the remaining overrides afterwards.",
            fg="yellow",
        )
    elif n_delta or n_duplicates:
        click.secho(
            f"No snapshot bloat. {n_delta} override(s) still shadow bundled (listed above); "
            "`openbench registry reset NAME` restores an entry's bundled values.",
            fg="green",
        )
    if n_files:
        attention = sum(len(c.per_file_attention) for c in audit.catalogs)
        click.secho(
            f"{n_files} entr{'y' if n_files == 1 else 'ies'} in separate overlay files still override "
            "bundled entries (listed above); prune and reset do not edit those files — edit or "
            "remove them to follow the bundled catalog.",
            fg="yellow" if attention else "cyan",
        )
    if not (has_bloat or n_delta or n_duplicates or n_files or any(c.error for c in audit.catalogs)):
        click.secho("Overlay is clean — fully tracking the bundled catalog. ✓", fg="green")


# Alias: `registry status` behaves like `registry diff`.
registry.add_command(diff_cmd, name="status")


@registry.command(name="reset")
@click.argument("name")
@click.option(
    "--kind",
    type=click.Choice(["references", "models"]),
    default=None,
    help="Catalog to reset; detected from the bundled catalogs when omitted.",
)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def reset_cmd(name: str, kind: str | None, yes: bool) -> None:
    """Remove an entry's user overrides and restore its current bundled values."""
    import yaml

    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
        _safe_load_catalog,
    )
    from openbench.util.names import normalize_name

    audit = oa.audit_overlays()
    wanted = normalize_name(name)
    in_bundled = {
        catalog.label: any(normalize_name(key) == wanted for key in oa._load_yaml(catalog.bundled_path))
        for catalog in audit.catalogs
    }
    if kind is None:
        matches = [label for label, present in in_bundled.items() if present]
        if not matches:
            # No longer bundled: the user's own files say which catalog it lives in.
            matches = [
                catalog.label
                for catalog in audit.catalogs
                if any(normalize_name(entry.name) == wanted for entry in catalog.entries)
                or any(normalize_name(entry_name) == wanted for entry_name, _ in oa.per_file_overrides(catalog.label))
            ]
        if len(matches) > 1:
            raise click.ClickException(f"{name!r} is both a reference and a model; pass --kind.")
        kind = matches[0] if matches else "references"
    if kind == "models":
        from openbench.data.registry.manager import RegistryManager

        target = RegistryManager().model_write_target(name)
        if target is not None and normalize_name(target) != wanted:
            click.echo(f"{name} is an alias of {target}; resetting {target}.")
            name, wanted = target, normalize_name(target)
    catalog_audit = getattr(audit, kind)
    path = catalog_audit.overlay_path
    if path.exists() and not oa._automatically_writable(path):
        detail = f"links to {path.resolve()}" if path.resolve() != path.absolute() else "is read-only"
        raise click.ClickException(
            f"{path} {detail}; resetting would replace it with a private copy. "
            "Edit that catalog directly, or replace the link with a copy first."
        )
    try:
        catalog = _safe_load_catalog(path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    if not isinstance(catalog, dict):
        raise click.ClickException(f"{path} is not a mapping of entries; fix or remove the file.")
    keys = [key for key in catalog if normalize_name(key) == wanted]
    other_files = [file for entry_name, file in oa.per_file_overrides(kind) if normalize_name(entry_name) == wanted]

    if not in_bundled[kind]:
        if keys and oa.previously_bundled(kind, name, catalog[keys[0]] if isinstance(catalog[keys[0]], dict) else {}):
            raise click.ClickException(
                f"{name!r} is no longer in the bundled catalog, so there are no bundled values to restore; "
                f"delete the entry instead if you no longer need it."
            )
        raise click.ClickException(f"{name!r} has no bundled {kind} entry to restore.")
    if not keys and not other_files:
        click.echo(f"{name} already follows the bundled catalog.")
        return
    if keys:
        bundled_key, bundled_entry = oa._bundled_lookup(oa._load_yaml(catalog_audit.bundled_path)).get(
            wanted, (name, {})
        )
        effective = {
            key: (
                oa._sparse_delta(bundled_entry, catalog[key], kind=kind, name=bundled_key)
                if isinstance(catalog[key], dict) and not catalog[key].get("_deleted")
                else catalog[key]
            )
            for key in keys
        }
        click.echo(f"Overrides of {name} in {path} that will be removed:")
        click.echo(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True).rstrip())
        if not yes:
            click.confirm(f"Restore bundled values for {name}? The overlay will be backed up.", abort=True)
        backups_before = set(path.parent.glob(f"{path.name}.*.bak"))
        try:
            with _catalog_write_lock(path):
                catalog = _safe_load_catalog(path)
                for key in [key for key in catalog if normalize_name(key) == wanted]:
                    catalog.pop(key)
                compacted = oa.sparsify_overlay_catalog(kind, catalog) != catalog
                # reset reports overrides kept in separate files itself
                backup = _backup_then_write(path, catalog, separate_files="ignore")
        except (OSError, RuntimeError) as exc:
            raise click.ClickException(f"Could not update {path}: {exc}") from exc
        _invalidate_registry_caches()
        new_backups = sorted(set(path.parent.glob(f"{path.name}.*.bak")) - backups_before)
        click.echo(f"Removed the overrides of {name}. Backup: {new_backups[-1] if new_backups else backup}")
        if compacted:
            click.echo("Other entries were compacted to their minimal overrides; their effect is unchanged.")
    if other_files:
        listed = ", ".join(str(file) for file in other_files)
        raise click.ClickException(
            f"{name} is still overridden by {listed}; edit or remove that file to use the bundled values."
        )
    click.echo(f"{name} now follows the bundled catalog.")


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
