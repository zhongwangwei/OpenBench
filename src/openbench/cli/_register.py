"""Implementation helpers for ``openbench ref register`` commands."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import click

from openbench.util.names import get_mapping_key_case_insensitive


def register_reference(
    name,
    root_dir,
    data_type,
    tim_res,
    grid_res,
    category,
    years,
    fulllist,
    variable,
    fallback,
    *,
    expand_existing_directory_fn: Callable,
    normalize_fulllist_path_fn: Callable,
    load_catalog_for_cli_fn: Callable[[Path], dict],
) -> None:
    """Register or update a reference dataset in the registry."""
    from openbench.cli._parsing import parse_fallbacks, parse_variables
    from openbench.data.registry import RegistryManager
    from openbench.data.registry.manager import get_writable_reference_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    catalog_path = get_writable_reference_catalog_path()
    if root_dir:
        root_dir = expand_existing_directory_fn(root_dir, "--root-dir")

    existing_catalog = load_catalog_for_cli_fn(catalog_path)

    def _resolve_existing(catalog):
        catalog_name = name
        existing_key = get_mapping_key_case_insensitive(catalog, name)
        existing = catalog.get(existing_key) if existing_key is not None else None
        if existing is not None:
            return existing_key, existing
        existing_ref = RegistryManager().get_reference(name)
        if existing_ref is not None:
            return existing_ref.name, existing_ref.to_dict()
        return catalog_name, {}

    _, existing = _resolve_existing(existing_catalog)
    is_new = not bool(existing)

    if grid_res is not None:
        if grid_res <= 0:
            raise click.ClickException("--grid-res must be a positive value")
        effective_data_type = data_type or existing.get("data_type")
        if effective_data_type == "stn":
            raise click.ClickException("--grid-res is not valid for station reference datasets")
    if years and years[1] < years[0]:
        raise click.ClickException("--years start year must be <= end year")

    new_vars = parse_variables(variable)

    if is_new and not root_dir:
        raise click.ClickException("--root-dir is required for new datasets.")

    if not variable and not fallback and is_new:
        click.echo("\nAdd variables (empty name to finish):")
        while True:
            var_name = click.prompt("  Standard variable name (e.g., Evapotranspiration)", default="")
            if not var_name:
                break
            nc_name = click.prompt("  Variable name in NetCDF file", default=var_name)
            unit = click.prompt("  Unit", default="")
            prefix = click.prompt("  File prefix", default="")
            suffix = click.prompt("  File suffix", default="")
            entry = {"varname": nc_name, "varunit": unit}
            if prefix:
                entry["prefix"] = prefix
            if suffix:
                entry["suffix"] = suffix
            new_vars[var_name] = entry

    if not new_vars and is_new and not fallback:
        click.secho("No variables defined. Registration cancelled.", fg="yellow")
        return

    data_type_override = data_type

    if data_type is None and root_dir:
        from openbench.data.coordinates import glob_nc
        from openbench.data.registry.scanner import _detect_data_type_from_nc

        root_path = Path(root_dir)
        nc_files = glob_nc(root_path)
        if not nc_files:
            try:
                children = sorted(root_path.iterdir())
            except OSError as exc:
                click.secho(
                    f"  Could not inspect subdirectories for data_type auto-detection ({exc}); using default: grid",
                    fg="yellow",
                )
                children = []
            for child in children:
                if child.is_dir():
                    nc_files = glob_nc(child)
                    if nc_files:
                        break
        nc_file = nc_files[0] if nc_files else None
        if nc_file is not None:
            detected = _detect_data_type_from_nc(nc_file)
            if detected:
                data_type = detected
                data_type_override = detected
                click.echo(f"  Auto-detected data_type: {data_type}")
            else:
                click.echo("  Could not auto-detect data_type from NC file, using default: grid")
    if is_new and data_type is None:
        data_type = "grid"
    if grid_res is not None and data_type == "stn":
        raise click.ClickException("--grid-res is not valid for station reference datasets")

    cancellation_reason = None
    latest_is_new = is_new
    merged_vars = {}
    updated = []
    added = []
    registration_race = False
    wrote_catalog = False
    backup_path = None
    with _catalog_write_lock(catalog_path):
        latest_catalog = load_catalog_for_cli_fn(catalog_path)
        catalog_name, existing = _resolve_existing(latest_catalog)
        latest_is_new = not bool(existing)
        registration_race = latest_is_new != is_new
        latest_effective_data_type = data_type_override or existing.get("data_type") or data_type
        if grid_res is not None and latest_effective_data_type == "stn":
            raise click.ClickException("--grid-res is not valid for station reference datasets")
        vars_to_merge = deepcopy(new_vars)
        parse_fallbacks(fallback, vars_to_merge, existing.get("variables") or {})
        if latest_is_new and not root_dir:
            cancellation_reason = "no_root_dir"
        elif not vars_to_merge and latest_is_new:
            cancellation_reason = "no_vars"
        else:
            descriptor = existing.copy()
            descriptor["name"] = catalog_name
            if latest_is_new or root_dir:
                descriptor["root_dir"] = root_dir
            if latest_is_new:
                descriptor.setdefault("description", f"{name} reference dataset")
                descriptor.setdefault("category", category)
                descriptor.setdefault("data_type", data_type)
                descriptor.setdefault("tim_res", tim_res or "Month")
                descriptor.setdefault("data_groupby", "Year")
                descriptor.setdefault("timezone", 0)
                descriptor.setdefault("years", list(years) if years else [2000, 2020])
            else:
                if category != "Other":
                    descriptor["category"] = category
                if years:
                    descriptor["years"] = list(years)
                if data_type_override is not None:
                    descriptor["data_type"] = data_type_override
                if tim_res is not None:
                    descriptor["tim_res"] = tim_res

            if grid_res is not None:
                descriptor["grid_res"] = grid_res
            elif descriptor.get("data_type") == "stn":
                descriptor.pop("grid_res", None)
            if fulllist is not None:
                descriptor["fulllist"] = normalize_fulllist_path_fn(
                    fulllist,
                    descriptor.get("root_dir") or root_dir,
                )

            merged_vars = deepcopy(descriptor.get("variables") or {})
            for var_name, var_descriptor in vars_to_merge.items():
                existing_key = get_mapping_key_case_insensitive(merged_vars, var_name)
                target_key = existing_key or var_name
                if existing_key is None:
                    added.append(var_name)
                else:
                    updated.append(target_key)
                merged_vars[target_key] = var_descriptor
            descriptor["variables"] = merged_vars

            latest_catalog[catalog_name] = descriptor
            backup_path = _backup_then_write(catalog_path, latest_catalog)
            wrote_catalog = True
    if wrote_catalog:
        _invalidate_registry_caches()
    if cancellation_reason == "no_root_dir":
        raise click.ClickException("--root-dir is required for new datasets.")
    if cancellation_reason == "no_vars":
        click.secho("No variables defined. Registration cancelled.", fg="yellow")
        return
    if registration_race:
        click.secho(
            "Dataset state changed while registering; merged against the latest catalog snapshot.",
            fg="yellow",
        )

    if latest_is_new:
        click.secho(f"✓ Created '{name}' ({len(merged_vars)} variables)", fg="green")
    else:
        parts = []
        if added:
            parts.append(f"{len(added)} added")
        if updated:
            parts.append(f"{len(updated)} updated")
        summary = ", ".join(parts) if parts else "metadata updated"
        click.secho(f"✓ Updated '{name}': {summary} ({len(merged_vars)} total)", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")
    click.echo(f"Verify: openbench ref show {name}")


def register_reference_profile(
    name,
    variable,
    *,
    fallback=(),
    tim_res=None,
    category=None,
    data_groupby=None,
    fulllist=None,
    description=None,
    load_catalog_for_cli_fn: Callable[[Path], dict],
) -> None:
    """Register a reference profile (variable mappings for a dataset type)."""
    from openbench.cli._parsing import parse_fallbacks, parse_variables
    from openbench.data.registry.manager import get_writable_reference_profiles_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    profile_path = get_writable_reference_profiles_path()
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = load_catalog_for_cli_fn(profile_path)

    profile_key = get_mapping_key_case_insensitive(profiles, name)
    existing = profiles.get(profile_key, {}) if profile_key is not None else {}
    is_new = not bool(existing)

    new_vars = parse_variables(variable)

    if not variable and not fallback and not existing:
        raise click.ClickException("No variables specified. Use -v 'StdName:ncname:unit[:prefix[:suffix]]'")

    latest_is_new = is_new
    merged = {}
    wrote_profile = False
    backup_path = None
    with _catalog_write_lock(profile_path):
        latest_profiles = load_catalog_for_cli_fn(profile_path)
        profile_key = get_mapping_key_case_insensitive(latest_profiles, name)
        profile_name = profile_key or name
        existing = latest_profiles.get(profile_key, {}) if profile_key is not None else {}
        latest_is_new = not bool(existing)
        vars_to_merge = deepcopy(new_vars)
        parse_fallbacks(fallback, vars_to_merge, existing.get("variables") or {})

        profile = existing.copy()
        if description:
            profile["description"] = description
        elif "description" not in profile:
            profile["description"] = f"{name} reference dataset"
        if tim_res:
            profile["tim_res"] = tim_res
        if category:
            profile["category"] = category
        if data_groupby:
            profile["data_groupby"] = data_groupby
        if fulllist:
            profile["fulllist"] = fulllist

        merged = deepcopy(profile.get("variables") or {})
        for var_name, var_descriptor in vars_to_merge.items():
            existing_key = get_mapping_key_case_insensitive(merged, var_name)
            merged[existing_key or var_name] = var_descriptor
        profile["variables"] = merged

        latest_profiles[profile_name] = profile
        backup_path = _backup_then_write(profile_path, latest_profiles)
        wrote_profile = True

    if wrote_profile:
        _invalidate_registry_caches()

    action = "Created" if latest_is_new else "Updated"
    click.secho(f"✓ {action} profile '{name}' ({len(merged)} variables)", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")
    click.echo("Re-scan to apply: openbench ref scan /path/to/reference --rescan --auto")
