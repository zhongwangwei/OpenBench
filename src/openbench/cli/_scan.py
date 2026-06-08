"""Implementation for ``openbench ref scan`` kept out of the Click command module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import click

ExpandDirectory = Callable[[str | Path, str], str]
SkipHandler = Callable[[object], None]


def run_scan(
    ref_root,
    auto,
    dry_run,
    *,
    rescan=False,
    only_pattern=None,
    allow_skip=False,
    pick_first=False,
    expand_existing_directory_fn: ExpandDirectory,
    filter_scan_groups_fn: Callable,
    filter_scan_skips_fn: Callable,
    print_scan_skip_report_fn: Callable,
    print_profile_rescue_preview_fn: Callable,
    prompt_scan_skip_action_fn: Callable,
    profile_rescue_supported_fn: Callable,
    create_ignore_profiles_for_scan_skips_fn: Callable,
    create_profiles_for_scan_skips_fn: Callable,
    scan_skip_keys_fn: Callable,
    format_scan_skip_key_fn: Callable,
) -> None:
    """Scan a directory for reference datasets and register new ones."""
    from openbench.config.user_settings import remember_reference_root
    from openbench.data.registry.manager import get_writable_reference_catalog_path
    from openbench.data.registry.scanner import (
        find_new_datasets,
        register_scanned_datasets_batch,
        scan_reference_directory,
    )

    if pick_first and not auto:
        raise click.ClickException("--pick-first requires --auto.")

    def _progress(msg):
        click.echo(msg)

    ref_root = expand_existing_directory_fn(ref_root, "Reference root")
    ref_root_path = Path(ref_root)
    had_ref_root_env = "OPENBENCH_REF_ROOT" in os.environ
    previous_ref_root_env = os.environ.get("OPENBENCH_REF_ROOT")
    os.environ["OPENBENCH_REF_ROOT"] = ref_root

    try:
        catalog_path = get_writable_reference_catalog_path()

        def _scan_once():
            skipped_now = []
            click.secho(f"Scanning {ref_root}...", bold=True)
            groups = (
                scan_reference_directory(ref_root, on_progress=_progress, on_skip=skipped_now.append)
                if rescan
                else find_new_datasets(ref_root, on_progress=_progress, on_skip=skipped_now.append)
            )
            if only_pattern:
                groups = filter_scan_groups_fn(groups, only_pattern)
                skipped_now = filter_scan_skips_fn(skipped_now, only_pattern)
            return groups, skipped_now

        profiles_updated_during_scan = False
        new_groups, skipped = _scan_once()
        final_skipped_count = len(skipped)
        click.echo()
        while skipped:
            final_skipped_count = len(skipped)
            print_scan_skip_report_fn(skipped)
            if dry_run:
                print_profile_rescue_preview_fn(skipped)
                break
            if allow_skip:
                click.secho("Continuing because --allow-skip was set.", fg="yellow")
                break
            if auto:
                raise click.ClickException(
                    f"ref scan found {len(skipped)} unsupported folder(s). "
                    "Re-run with --allow-skip to skip them explicitly."
                )

            action = prompt_scan_skip_action_fn(
                len(skipped),
                can_profile=any(profile_rescue_supported_fn(item) for item in skipped),
            )
            if action == "s":
                click.secho("Continuing after skipping unsupported folders.", fg="yellow")
                break
            if action == "a":
                raise click.ClickException("Scan cancelled because unsupported folders were not skipped.")
            if action == "i":
                before_skip_keys = scan_skip_keys_fn(skipped)
                updated = create_ignore_profiles_for_scan_skips_fn(skipped)
                if updated == 0:
                    raise click.ClickException("No ignore profiles were created from the skipped folders.")
                profiles_updated_during_scan = True
                click.secho(f"Updated {updated} ignore profile(s). Rescanning...", fg="green")
                click.echo()
                new_groups, skipped = _scan_once()
                final_skipped_count = len(skipped)
                if skipped and scan_skip_keys_fn(skipped) == before_skip_keys:
                    unresolved = ", ".join(sorted(format_scan_skip_key_fn(key) for key in before_skip_keys))
                    raise click.ClickException(
                        f"Ignore profile creation did not resolve unsupported folder(s): {unresolved}"
                    )
                click.echo()
                continue

            before_skip_keys = scan_skip_keys_fn(skipped)
            updated = create_profiles_for_scan_skips_fn(skipped, ref_root_path)
            if updated == 0:
                if any(profile_rescue_supported_fn(item) for item in skipped):
                    raise click.ClickException("Reference profile creation was cancelled or produced no updates.")
                raise click.ClickException(
                    "None of the skipped folders support automatic reference profile creation. "
                    "Use [i] to ignore them, [s] to skip for now, or register a profile manually."
                )
            profiles_updated_during_scan = True
            click.secho(
                f"Updated {updated} reference profile(s). Rescanning...",
                fg="green",
            )
            click.echo()
            new_groups, skipped = _scan_once()
            final_skipped_count = len(skipped)
            if skipped and scan_skip_keys_fn(skipped) == before_skip_keys:
                unresolved = ", ".join(sorted(format_scan_skip_key_fn(key) for key in before_skip_keys))
                raise click.ClickException(f"Profile creation did not resolve unsupported folder(s): {unresolved}")
            click.echo()

        if not new_groups:
            message = (
                "No datasets found to rescan." if rescan else "No new datasets found. All datasets already registered."
            )
            click.secho(message, fg="yellow")
            if not dry_run:
                settings_path = remember_reference_root(ref_root_path)
                click.echo(f"Saved reference root: {ref_root_path} ({settings_path})")
                if profiles_updated_during_scan and not rescan:
                    click.secho(
                        "Profile changes were saved. Use --rescan if matching datasets were "
                        "already registered and should be refreshed.",
                        fg="yellow",
                    )
                if final_skipped_count:
                    click.secho(f"Skipped {final_skipped_count} folder(s) during scan.", fg="yellow")
            return

        action_label = "dataset group(s) to register/update" if rescan else "new dataset(s)"
        click.secho(f"Found {len(new_groups)} {action_label}:", bold=True)
        click.echo()

        to_register = []
        for group in new_groups:
            for _res_name, variant in sorted(group.variants.items()):
                label = (
                    f"  {variant.registry_name:<35} {variant.data_type:<5} "
                    f"{variant.category:<10} {len(variant.variables)} vars, "
                    f"{variant.file_count} files"
                )
                click.echo(label)
                to_register.append(variant)

        click.echo()

        if dry_run:
            action = "register/update" if rescan else "register"
            click.secho(
                f"[DRY RUN] Would {action} {len(to_register)} dataset(s). No catalog changes made.",
                fg="cyan",
                bold=True,
            )
            if skipped:
                if allow_skip:
                    click.echo(
                        f"{len(skipped)} unsupported folder(s) would be skipped "
                        "when committing because --allow-skip is set."
                    )
                else:
                    click.echo(
                        f"{len(skipped)} unsupported folder(s) would still need "
                        "--allow-skip or interactive profile handling when committing."
                    )
            click.echo("Re-run without --dry-run to commit. Use --auto to skip the confirmation prompt.")
            return

        if not auto:
            action = "Register/update" if rescan else "Register"
            if not click.confirm(f"{action} {len(to_register)} dataset(s)?"):
                return

        def _multi_var_handler(var_name, sub_dir, all_vars):
            """Pick a variable when NC file has multiple data variables."""
            click.echo()
            click.secho(f"  Multiple variables in {sub_dir}/ (evaluating: {var_name}):", fg="yellow")
            if not all_vars:
                raise click.ClickException(f"No NetCDF variables found in {sub_dir}/ while evaluating {var_name}.")
            for i, v in enumerate(all_vars, 1):
                desc = v.get("long_name") or v.get("standard_name") or ""
                if desc:
                    desc = f"  — {desc}"
                click.echo(f"    [{i}] {v['name']:<20} {v['unit']:<15} {v['dims']}{desc}")
            if auto:
                if pick_first:
                    click.echo(f"    → Auto-selected: {all_vars[0]['name']}")
                    return all_vars[0]["name"]
                candidates = ", ".join(v["name"] for v in all_vars)
                raise click.ClickException(
                    f"--auto cannot pick a variable in {sub_dir}/ "
                    f"(evaluating: {var_name}); got {len(all_vars)} candidates "
                    f"({candidates}). Re-run interactively, or pass --pick-first "
                    "to silently choose the first candidate."
                )
            while True:
                choice = click.prompt("  Select variable number", type=int, default=1)
                if 1 <= choice <= len(all_vars):
                    return all_vars[choice - 1]["name"]
                click.secho(
                    f"Variable choice out of range for {sub_dir}/ "
                    f"(evaluating: {var_name}): {choice} (expected 1-{len(all_vars)})",
                    fg="yellow",
                )

        def _register_progress(msg):
            click.secho(f"  ✓{msg.lstrip()}", fg="green")

        written_catalog_path = register_scanned_datasets_batch(
            to_register,
            catalog_path=catalog_path,
            on_multi_var=_multi_var_handler,
            on_progress=_register_progress,
        )
        registered = len(to_register)
        settings_path = remember_reference_root(ref_root_path)

        # Clear registry cache so subsequent lookups see newly registered datasets
        from openbench.data.registry.manager import clear_registry_cache

        clear_registry_cache()

        click.echo()
        action_done = "Registered/updated" if rescan else "Registered"
        click.secho(f"{action_done} {registered} dataset(s).", fg="green", bold=True)
        click.echo(f"Catalog: {written_catalog_path}")
        if catalog_path.exists():
            backup_path = Path(str(written_catalog_path) + ".bak")
            if backup_path.exists():
                click.echo(f"Backup: {backup_path}")
        click.echo(f"Saved reference root: {ref_root_path} ({settings_path})")
        if final_skipped_count:
            click.secho(f"Skipped {final_skipped_count} folder(s) during scan.", fg="yellow")
        click.echo("Verify: openbench ref list")

        # Warn about unverified defaults
        click.echo()
        click.secho(
            "Note: Some fields (tim_res, grid_res, years) may be defaults and not verified from data.",
            fg="yellow",
        )
        click.echo("Use 'openbench ref show <name>' to check, and 'openbench ref register <name>' to fix.")
    finally:
        if had_ref_root_env:
            os.environ["OPENBENCH_REF_ROOT"] = previous_ref_root_env or ""
        else:
            os.environ.pop("OPENBENCH_REF_ROOT", None)
