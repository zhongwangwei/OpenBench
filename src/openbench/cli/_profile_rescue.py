"""Reference scan profile-rescue helpers for the ref CLI."""

from __future__ import annotations

from pathlib import Path

import click
import yaml


def _load_catalog_for_cli(path: Path) -> dict:
    from openbench.data.registry.scanner import _safe_load_catalog

    try:
        return _safe_load_catalog(path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _print_profile_rescue_preview(skipped) -> None:
    previews = {}
    for item in skipped:
        if not _profile_rescue_supported(item):
            continue
        skip_path = getattr(item, "path", str(item))
        parts = Path(skip_path).parts
        profile_name = _default_profile_name_for_skip(parts)
        if getattr(item, "reason", "") == "ambiguous_nc_subdirectories" and _is_grid_dataset_skip(parts):
            layout = "grid_dataset_choice"
        elif _is_grid_composite_root_skip(parts):
            layout = "grid_composite_files"
        elif _is_station_dataset_skip(parts):
            layout = "station_direct"
        else:
            layout = "grid_nested_root"
        previews[profile_name] = {
            "scan": {
                "layout": layout,
                "root_sub_dir": skip_path,
            },
            "variables": {},
        }
    if not previews:
        return

    click.echo()
    click.secho("Profile rescue preview (dry-run; no profile file written):", fg="cyan", bold=True)
    click.echo(yaml.safe_dump(previews, sort_keys=False).rstrip())


def _prompt_scan_skip_action(skip_count: int, *, can_profile: bool) -> str:
    click.echo(f"Skip {skip_count} unsupported folder(s), add a profile, ignore them, or abort:")
    click.echo("  [s] skip for now")
    if can_profile:
        click.echo("  [p] create/update reference profile and rescan")
    click.echo("  [i] add ignore profile and rescan")
    click.echo("  [a] abort")
    while True:
        choice = click.prompt("Action", default="s").strip().lower()
        if choice in {"s", "skip", "y", "yes"}:
            return "s"
        if can_profile and choice in {"p", "profile"}:
            return "p"
        if choice in {"i", "ignore"}:
            return "i"
        if choice in {"a", "abort", "n", "no"}:
            return "a"
        choices = "s, p, i, or a" if can_profile else "s, i, or a"
        click.secho(f"Choose {choices}.", fg="yellow")


def _profile_rescue_supported(skip) -> bool:
    skip_path = getattr(skip, "path", str(skip))
    reason = getattr(skip, "reason", "unsupported_layout")
    parts = Path(skip_path).parts
    if reason == "ambiguous_nc_subdirectories":
        return _is_grid_composite_root_skip(parts) or _is_grid_dataset_skip(parts) or _is_station_dataset_skip(parts)
    if reason == "nc_files_too_deep":
        return _is_grid_dataset_skip(parts) or _is_station_dataset_skip(parts)
    return False


def _create_profiles_for_scan_skips(skipped, ref_root: Path) -> int:
    pending = []
    for item in skipped:
        if not _profile_rescue_supported(item):
            path = getattr(item, "path", str(item))
            reason = getattr(item, "reason", "unsupported_layout")
            click.secho(
                f"Cannot infer a reference profile for {path}: {reason}. "
                "Use [i] to ignore it, [s] to skip for now, or register manually.",
                fg="yellow",
            )
            continue
        profile_name, profile = _prompt_reference_profile_for_scan_skip(item, ref_root)
        pending.append((profile_name, profile))
    updated = _write_reference_profiles(pending)
    for profile_name, _profile in pending:
        click.secho(f"Updated reference profile: {profile_name}", fg="green")
    return updated


def _create_ignore_profiles_for_scan_skips(skipped) -> int:
    pending = []
    for item in skipped:
        skip_path = getattr(item, "path", str(item))
        pending.append(
            (
                _ignore_profile_name(skip_path),
                {"scan": {"layout": "ignore", "root_sub_dir": skip_path}},
            )
        )
    updated = _write_reference_profiles(pending)
    for profile_name, _profile in pending:
        click.secho(f"Updated ignore profile: {profile_name}", fg="green")
    return updated


def _prompt_reference_profile_for_scan_skip(skip, ref_root: Path) -> tuple[str, dict]:
    skip_path = getattr(skip, "path", str(skip))
    reason = getattr(skip, "reason", "unsupported_layout")
    parts = Path(skip_path).parts
    if reason == "ambiguous_nc_subdirectories" and _is_grid_composite_root_skip(parts):
        return _prompt_grid_composite_profile(skip_path, parts, ref_root)
    if reason == "ambiguous_nc_subdirectories" and _is_grid_dataset_skip(parts):
        return _prompt_grid_dataset_choice_profile(skip_path, parts, ref_root)
    if reason == "ambiguous_nc_subdirectories" and _is_station_dataset_skip(parts):
        return _prompt_station_direct_profile(skip_path, parts, ref_root)
    if reason == "nc_files_too_deep" and _is_grid_dataset_skip(parts):
        return _prompt_grid_nested_profile(skip_path, parts, ref_root)
    if reason == "nc_files_too_deep" and _is_station_dataset_skip(parts):
        return _prompt_station_direct_profile(skip_path, parts, ref_root)
    if reason == "mixed_grid_resolutions_in_profile":
        raise click.ClickException(
            f"Cannot auto-repair {skip_path}: one profile cannot mix grid resolutions. "
            "Create separate profiles for each resolution, or use [i] to ignore this raw folder."
        )
    raise click.ClickException(f"Cannot infer a profile for {skip_path}: unsupported reason '{reason}'.")


def _prompt_grid_composite_profile(skip_path: str, parts: tuple[str, ...], ref_root: Path) -> tuple[str, dict]:
    dataset_root = ref_root / Path(skip_path)
    if not dataset_root.is_dir():
        raise click.ClickException(f"Cannot create profile for missing folder: {dataset_root}")

    profile_default = _default_profile_name_for_skip(parts)
    click.echo()
    click.secho(f"Create reference profile for {skip_path}", bold=True)
    profile_name = click.prompt("Profile name", default=profile_default).strip()
    if not profile_name:
        raise click.ClickException("Profile name cannot be empty.")

    child_specs = _profile_child_specs(dataset_root, ref_root)
    if not child_specs:
        raise click.ClickException(f"No NC-bearing child folders found under {skip_path}.")

    variables: dict[str, dict] = {}
    for child, nc_dir, nc_info in child_specs:
        child_rel = _ref_relative_path(child, ref_root)
        nc_rel = _ref_relative_path(nc_dir, ref_root)
        all_vars = nc_info.get("all_data_vars") or []
        default_nc = nc_info.get("varname") or (all_vars[0]["name"] if all_vars else child.name)
        default_unit = nc_info.get("varunit") or (all_vars[0].get("unit", "") if all_vars else "")
        file_glob = _prompt_file_glob(child_rel)

        if len(all_vars) > 1:
            click.secho("    Multiple NetCDF variables detected:", fg="yellow")
            for idx, var in enumerate(all_vars, 1):
                click.echo(f"      [{idx}] {var['name']} {var.get('unit', '')}")

        variables.update(
            _prompt_profile_variables_for_child(
                child_rel=child_rel,
                nc_rel=nc_rel,
                child_name=child.name,
                all_vars=all_vars,
                default_nc=default_nc,
                default_unit=default_unit,
                existing_names=set(variables),
                file_glob=file_glob,
                default_std=None,
                allow_skip_single=True,
                include_root_sub_dir=True,
            )
        )

    if not variables:
        raise click.ClickException(f"No variables were added to profile '{profile_name}'.")

    return profile_name, {
        "dataset_name": _grid_dataset_name_for_skip(parts),
        "scan": {
            "layout": "grid_composite_files",
            "root_sub_dir": skip_path,
        },
        "variables": variables,
    }


def _prompt_grid_dataset_choice_profile(skip_path: str, parts: tuple[str, ...], ref_root: Path) -> tuple[str, dict]:
    dataset_root = ref_root / Path(skip_path)
    if not dataset_root.is_dir():
        raise click.ClickException(f"Cannot create profile for missing folder: {dataset_root}")

    profile_default = _default_profile_name_for_skip(parts)
    default_std = parts[3] if len(parts) >= 5 else profile_default
    click.echo()
    click.secho(f"Create grid dataset choice profile for {skip_path}", bold=True)
    profile_name = click.prompt("Profile name", default=profile_default).strip()
    if not profile_name:
        raise click.ClickException("Profile name cannot be empty.")

    child_specs = _profile_child_specs(dataset_root, ref_root)
    if not child_specs:
        raise click.ClickException(f"No NC-bearing child folders found under {skip_path}.")
    click.echo("  Choose the child directory to register:")
    for idx, (_child, nc_dir, _info) in enumerate(child_specs, 1):
        click.echo(f"    [{idx}] {_ref_relative_path(nc_dir, ref_root)}")
    choice = click.prompt("  Child number", type=int, default=1)
    if not 1 <= choice <= len(child_specs):
        raise click.ClickException(f"Child choice out of range: {choice}")
    child, nc_dir, nc_info = child_specs[choice - 1]
    all_vars = nc_info.get("all_data_vars") or []
    default_nc = nc_info.get("varname") or (all_vars[0]["name"] if all_vars else default_std)
    default_unit = nc_info.get("varunit") or (all_vars[0].get("unit", "") if all_vars else "")
    file_glob = _prompt_file_glob(_ref_relative_path(nc_dir, ref_root))
    variables = _prompt_profile_variables_for_child(
        child_rel=_ref_relative_path(nc_dir, ref_root),
        nc_rel=str(nc_dir.relative_to(ref_root / "Grid" / parts[1])),
        child_name=default_std,
        all_vars=all_vars,
        default_nc=default_nc,
        default_unit=default_unit,
        existing_names=set(),
        file_glob=file_glob,
        default_std=default_std,
        allow_skip_single=False,
        include_root_sub_dir=False,
    )
    for entry in variables.values():
        entry["sub_dir"] = str(nc_dir.relative_to(ref_root / "Grid" / parts[1]))
    return profile_name, {
        "dataset_name": _grid_dataset_name_for_skip(parts),
        "scan": {
            "layout": "grid_dataset_choice",
            "root_sub_dir": skip_path,
            "nc_sub_dir": str(child.relative_to(dataset_root)),
        },
        "variables": variables,
    }


def _prompt_station_direct_profile(skip_path: str, parts: tuple[str, ...], ref_root: Path) -> tuple[str, dict]:
    dataset_root = ref_root / Path(skip_path)
    if not dataset_root.is_dir():
        raise click.ClickException(f"Cannot create profile for missing folder: {dataset_root}")

    profile_default = _default_profile_name_for_skip(parts)
    default_std = _station_standard_variable_default(parts)
    click.echo()
    click.secho(f"Create station reference profile for {skip_path}", bold=True)
    profile_name = click.prompt("Profile name", default=profile_default).strip()
    if not profile_name:
        raise click.ClickException("Profile name cannot be empty.")

    file_glob = click.prompt("  File glob", default="**/*.nc").strip()
    nc_info = _inspect_first_nc_under(dataset_root)
    all_vars = nc_info.get("all_data_vars") or []
    default_nc = nc_info.get("varname") or (all_vars[0]["name"] if all_vars else default_std)
    default_unit = nc_info.get("varunit") or (all_vars[0].get("unit", "") if all_vars else "")

    if len(all_vars) > 1:
        click.secho("  Multiple NetCDF variables detected:", fg="yellow")
        for idx, var in enumerate(all_vars, 1):
            click.echo(f"    [{idx}] {var['name']} {var.get('unit', '')}")

    variables = _prompt_profile_variables_for_child(
        child_rel=skip_path,
        nc_rel="",
        child_name=default_std,
        all_vars=all_vars,
        default_nc=default_nc,
        default_unit=default_unit,
        existing_names=set(),
        file_glob=None,
        default_std=default_std,
        allow_skip_single=False,
        include_root_sub_dir=False,
    )
    if not variables:
        raise click.ClickException(f"No variables were added to profile '{profile_name}'.")

    scan = {
        "layout": "station_direct",
        "root_sub_dir": skip_path,
    }
    if file_glob:
        scan["file_glob"] = file_glob
    return profile_name, {"scan": scan, "variables": variables}


def _prompt_grid_nested_profile(skip_path: str, parts: tuple[str, ...], ref_root: Path) -> tuple[str, dict]:
    dataset_root = ref_root / Path(skip_path)
    if not dataset_root.is_dir():
        raise click.ClickException(f"Cannot create profile for missing folder: {dataset_root}")

    profile_default = _default_profile_name_for_skip(parts)
    default_std = parts[3] if len(parts) >= 5 else profile_default
    click.echo()
    click.secho(f"Create nested grid reference profile for {skip_path}", bold=True)
    profile_name = click.prompt("Profile name", default=profile_default).strip()
    if not profile_name:
        raise click.ClickException("Profile name cannot be empty.")

    nc_info = _inspect_first_nc_under(dataset_root)
    from openbench.data.coordinates import glob_nc

    nc_files = sorted(glob_nc(dataset_root, recursive=True))
    concrete_nc_dir = nc_files[0].parent if nc_files else dataset_root
    res_dir = ref_root / "Grid" / parts[1]
    concrete_sub_dir = str(concrete_nc_dir.relative_to(res_dir))
    all_vars = nc_info.get("all_data_vars") or []
    default_nc = nc_info.get("varname") or (all_vars[0]["name"] if all_vars else default_std)
    default_unit = nc_info.get("varunit") or (all_vars[0].get("unit", "") if all_vars else "")

    if len(all_vars) > 1:
        click.secho("  Multiple NetCDF variables detected:", fg="yellow")
        for idx, var in enumerate(all_vars, 1):
            click.echo(f"    [{idx}] {var['name']} {var.get('unit', '')}")

    variables = _prompt_profile_variables_for_child(
        child_rel=skip_path,
        nc_rel=skip_path,
        child_name=default_std,
        all_vars=all_vars,
        default_nc=default_nc,
        default_unit=default_unit,
        existing_names=set(),
        file_glob=None,
        default_std=default_std,
        allow_skip_single=False,
        include_root_sub_dir=False,
    )
    if not variables:
        raise click.ClickException(f"No variables were added to profile '{profile_name}'.")
    for entry in variables.values():
        entry["sub_dir"] = concrete_sub_dir

    return profile_name, {
        "dataset_name": _grid_dataset_name_for_skip(parts),
        "scan": {
            "layout": "grid_nested_root",
            "root_sub_dir": skip_path,
        },
        "variables": variables,
    }


def _prompt_profile_variables_for_child(
    *,
    child_rel: str,
    nc_rel: str,
    child_name: str,
    all_vars: list[dict],
    default_nc: str,
    default_unit: str,
    existing_names: set[str],
    file_glob: str | None,
    default_std: str | None,
    allow_skip_single: bool,
    include_root_sub_dir: bool,
) -> dict[str, dict]:
    variables: dict[str, dict] = {}
    if len(all_vars) > 1:
        used_nc_names: set[str] = set()
        while True:
            std_name = _prompt_standard_variable_name(
                child_rel,
                existing_names | set(variables),
                default=None,
                allow_empty=True,
            )
            if not std_name:
                return variables

            nc_default = _next_nc_default(all_vars, used_nc_names, default_nc)
            nc_name, detected_unit = _prompt_nc_variable_name(child_rel, all_vars, nc_default)
            used_nc_names.add(nc_name)
            unit = click.prompt(
                "    Unit",
                default=detected_unit or default_unit,
                show_default=bool(detected_unit or default_unit),
            ).strip()
            variables[std_name] = _profile_variable_entry(
                nc_rel=nc_rel,
                nc_name=nc_name,
                unit=unit,
                file_glob=file_glob,
                include_root_sub_dir=include_root_sub_dir,
            )

    std_name = _prompt_standard_variable_name(
        child_rel,
        existing_names,
        default=default_std,
        allow_empty=allow_skip_single,
    )
    if not std_name:
        return {}
    nc_name, detected_unit = _prompt_nc_variable_name(child_rel, all_vars, default_nc)
    unit = click.prompt(
        "    Unit",
        default=detected_unit or default_unit,
        show_default=bool(detected_unit or default_unit),
    ).strip()
    return {
        std_name: _profile_variable_entry(
            nc_rel=nc_rel,
            nc_name=nc_name,
            unit=unit,
            file_glob=file_glob,
            include_root_sub_dir=include_root_sub_dir,
        )
    }


def _profile_variable_entry(
    *,
    nc_rel: str,
    nc_name: str,
    unit: str,
    file_glob: str | None,
    include_root_sub_dir: bool,
) -> dict:
    entry = {
        "varname": nc_name,
        "varunit": unit,
    }
    if include_root_sub_dir:
        entry = {"root_sub_dir": nc_rel, **entry}
    if file_glob:
        entry["file_glob"] = file_glob
    return entry


def _prompt_standard_variable_name(
    child_rel: str,
    existing_names: set[str],
    *,
    default: str | None,
    allow_empty: bool,
) -> str:
    while True:
        if default is None:
            value = click.prompt(
                "    Standard variable name (blank to skip/finish)",
                default="",
                show_default=False,
            ).strip()
        else:
            value = click.prompt("    Standard variable name", default=default).strip()
        if value.lower() in {"skip", "-"} and allow_empty:
            return ""
        if not value:
            if allow_empty:
                return ""
            click.secho(f"    Standard variable name cannot be empty for {child_rel}.", fg="yellow")
            continue
        if value in existing_names:
            click.secho(f"    Standard variable '{value}' already exists in this profile.", fg="yellow")
            default = None
            continue
        return value


def _prompt_nc_variable_name(child_rel: str, all_vars: list[dict], default_name: str) -> tuple[str, str]:
    while True:
        value = click.prompt("    NetCDF variable name", default=default_name).strip()
        if not value:
            click.secho(f"    NetCDF variable name cannot be empty for {child_rel}.", fg="yellow")
            continue
        if not all_vars:
            return value, ""
        selected = _resolve_nc_variable_choice(value, all_vars)
        if selected is not None:
            return selected["name"], selected.get("unit", "")


def _resolve_nc_variable_choice(value: str, all_vars: list[dict]) -> dict | None:
    text = value.strip()
    if text.isdigit():
        choice = int(text)
        if 1 <= choice <= len(all_vars):
            return all_vars[choice - 1]
        click.secho(
            f"    Variable choice out of range: {choice} (expected 1-{len(all_vars)}).",
            fg="yellow",
        )
        return None

    for var in all_vars:
        if var["name"] == text:
            return var

    for var in all_vars:
        if var["name"].lower() == text.lower():
            click.secho(
                f"    Invalid NetCDF variable '{text}'. Did you mean '{var['name']}'?",
                fg="yellow",
            )
            return None

    candidates = ", ".join(var["name"] for var in all_vars)
    click.secho(
        f"    Invalid NetCDF variable '{text}'. Choose one of: {candidates}.",
        fg="yellow",
    )
    return None


def _next_nc_default(all_vars: list[dict], used_nc_names: set[str], fallback: str) -> str:
    for var in all_vars:
        if var["name"] not in used_nc_names:
            return var["name"]
    return fallback


def _prompt_file_glob(child_rel: str) -> str:
    click.echo(f"  {child_rel}")
    return click.prompt(
        "    File glob (blank for all *.nc)",
        default="",
        show_default=False,
    ).strip()


def _is_grid_composite_root_skip(parts: tuple[str, ...]) -> bool:
    return len(parts) == 4 and parts[0] == "Grid" and parts[2] == "Composite"


def _is_grid_composite_skip(parts: tuple[str, ...]) -> bool:
    return _is_grid_composite_root_skip(parts)


def _is_grid_dataset_skip(parts: tuple[str, ...]) -> bool:
    return len(parts) >= 5 and parts[0] == "Grid"


def _is_station_dataset_skip(parts: tuple[str, ...]) -> bool:
    return len(parts) >= 4 and parts[0] == "Station"


def _default_profile_name_for_skip(parts: tuple[str, ...]) -> str:
    if _is_grid_composite_root_skip(parts):
        return _sanitize_profile_name(f"{parts[3]}_{parts[1]}")
    if _is_grid_dataset_skip(parts):
        return _sanitize_profile_name(f"{parts[4]}_{parts[1]}")
    if _is_station_dataset_skip(parts):
        return _sanitize_profile_name(f"{parts[3]}_Station")
    return _sanitize_profile_name("_".join(parts) or "ReferenceProfile")


def _grid_dataset_name_for_skip(parts: tuple[str, ...]) -> str:
    if _is_grid_composite_root_skip(parts):
        return parts[3]
    if _is_grid_dataset_skip(parts):
        return parts[4]
    return _default_profile_name_for_skip(parts)


def _station_standard_variable_default(parts: tuple[str, ...]) -> str:
    if len(parts) >= 4 and parts[0] == "Station":
        return parts[2]
    return "Station_Variable"


def _ignore_profile_name(skip_path: str) -> str:
    return "_Ignore_" + _sanitize_profile_name(skip_path.replace("/", "_"))


def _sanitize_profile_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "ReferenceProfile"


def _inspect_first_nc_under(dataset_root: Path) -> dict:
    from openbench.data.coordinates import glob_nc
    from openbench.data.registry.scanner import inspect_nc_file

    nc_files = sorted(glob_nc(dataset_root, recursive=True))
    if not nc_files:
        return {}
    return inspect_nc_file(nc_files[0].parent)


def _profile_child_specs(dataset_root: Path, ref_root: Path) -> list[tuple[Path, Path, dict]]:
    from openbench.data.registry.scanner import (
        DEFAULT_NC_DESCENT,
        _find_nc_dir_with_descent,
        inspect_nc_file,
    )

    specs = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        nc_dir, _count, status = _find_nc_dir_with_descent(
            child,
            max_descent=DEFAULT_NC_DESCENT,
        )
        if status == "ambiguous":
            raise click.ClickException(
                f"Cannot infer a single variable folder for {_ref_relative_path(child, ref_root)}."
            )
        if status != "found" or nc_dir is None:
            continue
        specs.append((child, nc_dir, inspect_nc_file(nc_dir)))
    return specs


def _ref_relative_path(path: Path, ref_root: Path) -> str:
    return path.relative_to(ref_root).as_posix()


def _write_reference_profile(profile_name: str, profile: dict) -> Path:
    _write_reference_profiles([(profile_name, profile)])
    from openbench.data.registry.manager import get_writable_reference_profiles_path

    return get_writable_reference_profiles_path()


def _write_reference_profiles(profile_defs: list[tuple[str, dict]]) -> int:
    import copy

    from openbench.data.registry.manager import get_writable_reference_profiles_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    if not profile_defs:
        return 0

    profile_path = get_writable_reference_profiles_path()
    with _catalog_write_lock(profile_path):
        profiles = _load_catalog_for_cli(profile_path)
        updated_profiles = copy.deepcopy(profiles)
        updated_names: set[str] = set()
        for profile_name, profile in profile_defs:
            existing = updated_profiles.get(profile_name)
            merged = copy.deepcopy(existing) if isinstance(existing, dict) else {}
            new_scan = profile.get("scan")
            existing_scan = merged.get("scan")
            if isinstance(existing_scan, dict) and isinstance(new_scan, dict) and existing_scan != new_scan:
                click.echo(f"Profile '{profile_name}' already has a different scan config:")
                click.echo(f"  existing: {existing_scan}")
                click.echo(f"  new:      {new_scan}")
                if not click.confirm("Overwrite this profile scan config?", default=False):
                    raise click.ClickException(f"Profile scan overwrite cancelled for '{profile_name}'.")
            if isinstance(new_scan, dict):
                merged["scan"] = copy.deepcopy(new_scan)
            for key, value in profile.items():
                if key in {"scan", "variables"}:
                    continue
                merged[key] = copy.deepcopy(value)
            variables = copy.deepcopy(merged.get("variables") or {})
            variables.update(copy.deepcopy(profile.get("variables") or {}))
            merged["variables"] = variables
            updated_profiles[profile_name] = merged
            updated_names.add(profile_name)
        backup_path = _backup_then_write(profile_path, updated_profiles)
    _invalidate_registry_caches()
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")
    return len(updated_names)
