"""openbench model commands."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import click
import yaml

from openbench.cli._options import TIM_RES_TYPE
from openbench.util.names import (
    AmbiguousNameError,
    get_mapping_key_case_insensitive,
    normalize_name,
)

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESERVED_MODEL_NAMES = {"_defaults", "default", "general", "variables", "models", "aliases"}
_VAR_ATTR_KEYS = {"varname", "varunit", "prefix", "suffix", "sub_dir", "compute", "prefix_fallback"}


def _validate_model_name(name: str) -> str:
    text = str(name).strip()
    if not text or normalize_name(text) in _RESERVED_MODEL_NAMES or not _MODEL_NAME_RE.fullmatch(text):
        raise click.ClickException("model name must be a simple identifier using letters, numbers, '.', '_' or '-'")
    return text


def _case_insensitive_catalog_entry(catalog: dict[str, Any], name: str) -> tuple[str, dict] | tuple[None, None]:
    try:
        key = get_mapping_key_case_insensitive(catalog, name)
    except AmbiguousNameError as exc:
        raise click.ClickException(str(exc)) from exc
    return (key, catalog[key]) if key is not None else (None, None)


def _model_lookup_key(name: str) -> str:
    from openbench.data.registry.manager import RegistryManager, canonical_model_key

    key = canonical_model_key(name)
    return getattr(RegistryManager, "MODEL_ALIASES", {}).get(key, key)


def _resolve_write_name(name: str, mgr) -> tuple[str, str | None]:
    requested = _validate_model_name(name)
    lookup_key = _model_lookup_key(requested)
    model = mgr.get_model(lookup_key)
    if model is not None:
        alias_message = None
        if normalize_name(requested) != normalize_name(model.name):
            alias_message = f"model alias '{requested}' resolved to '{model.name}'"
        return model.name, alias_message
    if normalize_name(lookup_key) != normalize_name(requested):
        return lookup_key, f"model alias '{requested}' resolved to '{lookup_key}'"
    return requested, None


def _load_catalog(path: Path) -> dict[str, Any]:
    from openbench.data.registry.scanner import _safe_load_catalog

    try:
        return _safe_load_catalog(path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _read_yaml_mapping(path: Path, *, label: str, missing_ok: bool = True) -> dict[str, Any]:
    if not path.exists():
        if missing_ok:
            return {}
        raise click.ClickException(f"{label} YAML not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise click.ClickException(f"Failed to read {label} YAML at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise click.ClickException(f"Failed to read {label} YAML at {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise click.ClickException(f"{label} YAML at {path} must be a mapping")
    return raw


def _source_var_names_from_mapping(mapping: Any) -> set[str]:
    """Return NetCDF/source variable names exposed to conversion snippets."""

    names: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value:
            names.add(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)

    if isinstance(mapping, dict):
        add(mapping.get("varname"))
        for fb in mapping.get("fallbacks") or []:
            if isinstance(fb, dict):
                add(fb.get("varname"))
            else:
                add(getattr(fb, "varname", None))
    else:
        add(getattr(mapping, "varname", None))
        for fb in getattr(mapping, "fallbacks", None) or []:
            add(getattr(fb, "varname", None))

    return names


def _source_var_names(variables: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for mapping in (variables or {}).values():
        names.update(_source_var_names_from_mapping(mapping))
    return names


def _validate_expression(expr: str, *, label: str, allowed_names: set[str] | None = None) -> None:
    if not expr:
        return
    from openbench.data.compute import (
        ComputeError,
    )
    from openbench.data.compute import (
        _validate_expression as validate_one,
    )

    try:
        parts = [part.strip() for part in str(expr).split(";") if part.strip()]
        # CLI registration historically accepted `value` for conversion-style
        # snippets. Keep accepting it at ingest time while still rejecting
        # unknown/dangerous identifiers.
        active_allowed_names = {"value", "np", *(allowed_names or set())}
        for part in parts:
            if "=" in part and not part.startswith("("):
                target, rhs = part.split("=", 1)
                validate_one(rhs.strip(), allowed_names=active_allowed_names)
                target = target.strip()
                if target.isidentifier():
                    active_allowed_names.add(target)
            else:
                validate_one(part, allowed_names=active_allowed_names)
    except ComputeError as exc:
        raise click.ClickException(f"Invalid {label}: {exc}") from exc


def _parse_var_attrs(
    raw_attrs: tuple[str, ...],
    target_vars: dict[str, dict[str, Any]],
    existing_vars: dict[str, dict[str, Any]],
) -> None:
    for raw in raw_attrs:
        marker_matches = []
        for candidate_key in _VAR_ATTR_KEYS:
            marker = f":{candidate_key}="
            idx = raw.rfind(marker)
            if idx >= 0:
                marker_matches.append((idx, candidate_key, marker))
        if not marker_matches:
            if ":" not in raw or "=" not in raw:
                raise click.ClickException("Invalid --var-attr. Use 'StdName:key=value'.")
            valid_keys = ", ".join(sorted(_VAR_ATTR_KEYS))
            raise click.ClickException(f"Invalid --var-attr key. Valid keys: {valid_keys}")
        idx, key, marker = max(marker_matches, key=lambda item: item[0])
        std_name = raw[:idx].strip()
        value = raw[idx + len(marker) :].strip()
        if not std_name:
            raise click.ClickException("Invalid --var-attr: variable name is empty")
        if not key:
            raise click.ClickException("Invalid --var-attr: key is empty")
        if key not in _VAR_ATTR_KEYS:
            valid_keys = ", ".join(sorted(_VAR_ATTR_KEYS))
            raise click.ClickException(f"Invalid --var-attr key '{key}'. Valid keys: {valid_keys}")
        if value == "":
            click.secho(
                f"⚠ --var-attr {std_name}:{key}= has an empty value; this will clear the field.",
                fg="yellow",
            )
        target_key = get_mapping_key_case_insensitive(target_vars, std_name)
        existing_key = get_mapping_key_case_insensitive(existing_vars, std_name)
        if target_key is None:
            target_key = existing_key or std_name
            target_vars[target_key] = deepcopy(existing_vars.get(existing_key, {"varname": target_key, "varunit": ""}))
        if key == "prefix_fallback":
            target_vars[target_key][key] = [part.strip() for part in value.split(",") if part.strip()]
        else:
            if key == "compute":
                allowed_names = _source_var_names(existing_vars) | _source_var_names(target_vars)
                allowed_names.update(_source_var_names_from_mapping(target_vars[target_key]))
                _validate_expression(
                    value,
                    label=f"compute expression for {target_key}",
                    allowed_names=allowed_names,
                )
            target_vars[target_key][key] = value


def _parse_time_offsets(raw_offsets: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in raw_offsets:
        if "=" not in raw:
            raise click.ClickException("Invalid --time-offset. Use 'Resolution=offset' or 'Resolution:Vars=offset'.")
        lhs, value = raw.split("=", 1)
        lhs = lhs.strip()
        value = value.strip()
        if not lhs or not value:
            raise click.ClickException("Invalid --time-offset: resolution and offset are required")
        if ":" in lhs:
            resolution, variables = lhs.split(":", 1)
            resolution = resolution.strip()
            variables = variables.strip()
            if not resolution or not variables:
                raise click.ClickException("Invalid --time-offset: resolution and variable names are required")
            bucket = result.setdefault(resolution, {})
            if not isinstance(bucket, dict):
                bucket = {"default": bucket}
                result[resolution] = bucket
            bucket[variables] = value
        else:
            existing = result.get(lhs)
            if isinstance(existing, dict):
                existing["default"] = value
            else:
                result[lhs] = value
    return result


def _merge_time_offset_value(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, dict) or isinstance(incoming, dict):
        merged: dict[str, Any] = {}
        if isinstance(existing, dict):
            merged.update(existing)
        elif existing not in (None, ""):
            merged["default"] = existing
        if isinstance(incoming, dict):
            merged.update(incoming)
        elif incoming not in (None, ""):
            merged["default"] = incoming
        return merged
    return incoming


def _append_unique_fallbacks(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    incoming_fallbacks = incoming.get("fallbacks") or []
    if not incoming_fallbacks:
        return False
    current = list(existing.get("fallbacks") or [])
    seen = {(fb.get("varname"), fb.get("varunit"), fb.get("convert")) for fb in current}
    changed = False
    for fb in incoming_fallbacks:
        marker = (fb.get("varname"), fb.get("varunit"), fb.get("convert"))
        if marker in seen:
            continue
        current.append(fb)
        seen.add(marker)
        changed = True
    if changed:
        existing["fallbacks"] = current
    return changed


def _variable_completeness_notes(mapping) -> list[str]:
    notes: list[str] = []
    if not getattr(mapping, "varname", None):
        notes.append("missing varname")
    if not getattr(mapping, "varunit", None):
        notes.append("missing varunit")
    return notes


def _model_completeness_label(model_obj) -> str:
    missing = sum(1 for mapping in model_obj.variables.values() if _variable_completeness_notes(mapping))
    return "✓" if missing == 0 else f"⚠ {missing}"


def _model_source_label(name: str) -> str:
    from openbench.data.registry.manager import get_writable_model_catalog_path

    catalog_path = get_writable_model_catalog_path()
    catalog = _read_yaml_mapping(catalog_path, label="model catalog")
    key, _ = _case_insensitive_catalog_entry(catalog or {}, name)
    return f"user({catalog_path})" if key else "bundled"


def _emit_model(model_obj, *, fmt: str, source: str | None = None) -> None:
    data = model_obj.to_dict()
    if source:
        data["_source"] = source
    if fmt == "json":
        click.echo(json.dumps(data, indent=2, sort_keys=True))
    elif fmt == "yaml":
        click.echo(yaml.safe_dump(data, sort_keys=False).rstrip())
    else:
        raise click.ClickException(f"Unsupported format: {fmt}")


def _model_history_entries(name: str) -> list[dict[str, Any]]:
    from openbench.data.registry.manager import get_writable_model_catalog_path

    catalog_path = get_writable_model_catalog_path()
    candidates = []
    single_slot = catalog_path.with_suffix(catalog_path.suffix + ".bak")
    if single_slot.exists():
        candidates.append(single_slot)
    candidates.extend(sorted(catalog_path.parent.glob(f"{catalog_path.name}.*.bak")))

    entries: list[dict[str, Any]] = []
    for path in candidates:
        catalog = _load_catalog(path)
        key, profile = _case_insensitive_catalog_entry(catalog, name)
        if key is None:
            continue
        entries.append(
            {
                "backup": str(path),
                "catalog_name": key,
                "description": profile.get("description", ""),
                "variables": len(profile.get("variables") or {}),
                "mtime": path.stat().st_mtime,
            }
        )
    entries.sort(key=lambda item: (item["mtime"], item["backup"]))
    return entries


def _merge_model_profile_descriptor(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    append_only: bool = False,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    """Merge a model profile descriptor using the same variable semantics as register."""
    merged = deepcopy(existing or {})
    added_keys: list[str] = []
    updated_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, value in (incoming or {}).items():
        if key != "variables":
            merged[key] = value

    variables = deepcopy(merged.get("variables") or {})
    incoming_variables = (incoming or {}).get("variables")
    if incoming_variables is not None:
        for var_name, var_descriptor in incoming_variables.items():
            existing_key = get_mapping_key_case_insensitive(variables, var_name)
            target_key = existing_key or var_name
            if existing_key is not None:
                if append_only:
                    if _append_unique_fallbacks(variables[target_key], var_descriptor):
                        updated_keys.append(f"{target_key} fallback")
                    skipped_keys.append(target_key)
                    continue
                updated_keys.append(target_key)
            else:
                added_keys.append(var_name)
            variables[target_key] = var_descriptor

    merged["variables"] = variables
    return merged, added_keys, updated_keys, skipped_keys


def _write_model_profile_descriptor(
    name: str,
    profile: dict[str, Any],
    *,
    overwrite: bool = False,
    exists_message: str | None = None,
) -> Path:
    """Write a complete model profile descriptor to the user catalog.

    This shared writer is used by ``model register``-adjacent flows and
    ``sim scan --register-model`` so both paths agree on case handling,
    bundled overlays, backups, cache invalidation, and variable merge behavior.
    """
    from openbench.data.registry.manager import RegistryManager, get_writable_model_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    registry = RegistryManager()
    write_name, _ = _resolve_write_name(name, registry)
    catalog_path = get_writable_model_catalog_path()
    with _catalog_write_lock(catalog_path):
        catalog = _load_catalog(catalog_path)
        key, existing = _case_insensitive_catalog_entry(catalog, write_name)
        catalog_name = key or write_name
        if existing is None:
            existing_model = RegistryManager().get_model(write_name)
            if existing_model is not None:
                existing = existing_model.to_dict()
                catalog_name = existing_model.name

        if existing is not None and not overwrite:
            hint = exists_message or "Use `openbench model register` to update variable mappings."
            existing_name = existing.get("name", catalog_name)
            raise click.ClickException(f"Model profile already exists: {existing_name}. {hint}")

        incoming = deepcopy(profile)
        incoming["name"] = incoming.get("name") or catalog_name
        if existing is not None:
            descriptor, _, _, _ = _merge_model_profile_descriptor(existing, incoming)
        else:
            descriptor = incoming
            descriptor.setdefault("name", catalog_name)

        if descriptor.get("data_type") == "stn":
            descriptor.pop("grid_res", None)
        descriptor.pop("_deleted", None)
        catalog[catalog_name] = descriptor
        backup_path = _backup_then_write(catalog_path, catalog)
    _invalidate_registry_caches()
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")
    return catalog_path


@click.group()
def model():
    """Manage model profiles."""


@model.command("list")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def list_models(fmt):
    """List all available model profiles."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    models = mgr.list_models()

    if fmt == "json":
        click.echo(json.dumps([m.to_dict() for m in models], indent=2, sort_keys=True))
        return
    if not models:
        click.echo("No model profiles registered.")
        return

    name_width = max([20, *(len(m.name) for m in models)])
    click.secho(f"{'Name':<{name_width}} {'Type':<6} {'Res':<8} {'Variables':<9} {'Complete'}", bold=True)
    click.echo("─" * (name_width + 48))
    for m in models:
        res = "—" if m.data_type == "stn" else (f"{m.grid_res}°" if m.grid_res else "?")
        complete = _model_completeness_label(m)
        click.echo(f"{m.name:<{name_width}} {m.data_type:<6} {res:<8} {len(m.variables):<9} {complete}")

    click.echo(f"\nTotal: {len(models)} model profiles")


@model.command()
@click.argument("name")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "yaml"]), default="text")
@click.option("--history", is_flag=True, help="Show user catalog backup history for this model.")
def show(name, fmt, history):
    """Show variable mappings for a model profile."""
    from openbench.data.registry import RegistryManager

    if history:
        entries = _model_history_entries(name)
        if fmt == "json":
            click.echo(json.dumps(entries, indent=2, sort_keys=True))
        elif fmt == "yaml":
            click.echo(yaml.safe_dump(entries, sort_keys=False).rstrip())
        else:
            if not entries:
                click.echo(f"No backup history found for model profile: {name}")
                return
            click.secho(f"Backup history for {name}", bold=True)
            click.secho(f"{'Variables':<9} {'Catalog name':<24} Backup", bold=True)
            click.echo("-" * 80)
            for entry in entries:
                click.echo(f"{entry['variables']:<9} {entry['catalog_name']:<24} {entry['backup']}")
        return

    mgr = RegistryManager()
    m = mgr.get_model(name)
    if m is None:
        raise click.ClickException(f"Model profile not found: {name}")

    source = _model_source_label(m.name)
    if fmt != "text":
        _emit_model(m, fmt=fmt, source=source)
        return

    click.secho(f"{m.name}", bold=True)
    click.echo(f"Description: {m.description}")
    resolution = f"{m.grid_res}°" if m.grid_res is not None else "N/A"
    click.echo(f"Type: {m.data_type}, Resolution: {resolution}, Time: {m.tim_res}")
    if m.time_offset:
        click.echo(f"Time offset: {m.time_offset}")
    click.echo(f"Source: {source}")
    click.echo()
    click.secho(f"{'Variable':<35} {'Source':<25} {'Unit':<15} {'Path/Notes'}", bold=True)
    click.echo("─" * 100)
    for var_name, mapping in sorted(m.variables.items()):
        # Determine source type and display
        vn_str = str(mapping.varname)

        notes_parts = []
        if mapping.fallbacks:
            for fb in mapping.fallbacks:
                conv = f", {fb.convert}" if fb.convert else ""
                notes_parts.append(f"fallback: {fb.varname} [{fb.varunit}{conv}]")
        if mapping.compute:
            if len(mapping.compute) < 40:
                expr = mapping.compute
            else:
                expr = mapping.compute[:37] + "... (use --format json/yaml to see full)"
            notes_parts.append(f"compute: {expr}")
        if mapping.sub_dir:
            notes_parts.append(f"sub_dir: {mapping.sub_dir}")
        if mapping.prefix or mapping.suffix:
            notes_parts.append(f"pattern: {mapping.prefix}*{mapping.suffix}")
        if mapping.prefix_fallback:
            notes_parts.append(f"file: *{'/'.join(mapping.prefix_fallback)}*")
        notes_parts.extend(_variable_completeness_notes(mapping))
        notes = " | ".join(notes_parts)

        click.echo(f"{var_name:<35} {vn_str:<25} {mapping.varunit:<15} {notes}")


@model.command()
@click.argument("name")
@click.option("--data-type", type=click.Choice(["grid", "stn"]), default=None)
@click.option("--grid-res", type=float, default=None)
@click.option("--tim-res", type=TIM_RES_TYPE, default=None)
@click.option("--description", default=None)
@click.option(
    "-v",
    "--variable",
    multiple=True,
    help="'StdName:ncname:unit[:prefix[:suffix]]' or 'StdName name=nc unit=unit' (repeatable).",
)
@click.option("-f", "--fallback", multiple=True, help="'StdName:fallback_name:fallback_unit:conversion' (repeatable).")
@click.option("--var-attr", multiple=True, help="'StdName:key=value' for compute, sub_dir, prefix_fallback, etc.")
@click.option("--time-offset", multiple=True, help="'Resolution=offset' or 'Resolution:Var1,Var2=offset' (repeatable).")
@click.option("--append-only", is_flag=True, help="Only add new variables, skip existing ones.")
def register(
    name,
    data_type,
    grid_res,
    tim_res,
    description,
    variable,
    fallback,
    var_attr=(),
    time_offset=(),
    append_only=False,
):
    """Register or update a model profile.

\b
Examples:
openbench model register MyModel --data-type grid --grid-res 0.5 \\
  -v "Evapotranspiration:ET:mm day-1"

\b
openbench model register CoLM2024 \\
  -v "Gross_Primary_Productivity:f_gpp:g m-2 s-1" \\
  -f "Gross_Primary_Productivity:f_assim:mol m-2 s-1:value * 12.011"

\b
openbench model register CoLM2024 -v "Snow_Depth:f_snowdp:m"
    """
    from openbench.data.registry.manager import RegistryManager, get_writable_model_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    if grid_res is not None and grid_res <= 0:
        raise click.ClickException("--grid-res must be a positive value")
    if data_type == "stn" and grid_res is not None:
        raise click.ClickException("--grid-res is not valid for station model profiles")

    catalog_path = get_writable_model_catalog_path()
    registry = RegistryManager()
    write_name, alias_message = _resolve_write_name(name, registry)
    if alias_message:
        click.secho(f"~ {alias_message}", fg="cyan")

    # Hardened load: corrupted YAML raises rather than silently resetting
    catalog = _load_catalog(catalog_path)

    def _resolve_existing(catalog):
        catalog_name = write_name
        key, existing = _case_insensitive_catalog_entry(catalog, catalog_name)
        if existing is not None:
            return key, existing
        existing_model = RegistryManager().get_model(catalog_name)
        if existing_model is None and normalize_name(catalog_name) != normalize_name(name):
            existing_model = RegistryManager().get_model(name)
        if existing_model is not None:
            return existing_model.name, existing_model.to_dict()
        return catalog_name, {}

    # Load existing profile if updating. User overlay writes should still be
    # able to extend bundled default profiles without rewriting package files.
    catalog_key, existing = _resolve_existing(catalog)
    is_new = not bool(existing)

    effective_data_type = data_type or existing.get("data_type")
    if grid_res is not None and effective_data_type == "stn":
        raise click.ClickException("--grid-res is not valid for station model profiles")

    if is_new and not data_type:
        data_type = click.prompt("  Data type", type=click.Choice(["grid", "stn"]), default="grid")
    if is_new and grid_res is None and (data_type or existing.get("data_type")) == "grid":
        grid_res = click.prompt("  Grid resolution (degrees)", type=float, default=0.5)
    if is_new and tim_res is None:
        tim_res = click.prompt("  Time resolution", type=TIM_RES_TYPE, default="Month")

    # Parse primary variables. Fallbacks are attached inside the write lock so
    # they see the latest catalog variables in concurrent register calls.
    from openbench.cli._parsing import parse_fallbacks, parse_variables

    new_vars = parse_variables(variable)

    if not variable and not fallback and not var_attr and is_new:
        if description is None:
            prompted_description = click.prompt("  Description", default="", show_default=False)
            if prompted_description:
                description = prompted_description
        if not time_offset:
            prompted_offset = click.prompt("  Time offset (optional)", default="", show_default=False)
            if prompted_offset:
                time_offset = (prompted_offset,)

        # Interactive variable entry
        click.echo("\nAdd variables (empty name to finish):")
        while True:
            std_name = click.prompt("  Standard variable name (e.g., Evapotranspiration)", default="")
            if not std_name:
                break
            nc_name = click.prompt("  NetCDF variable name(s), comma-separated for fallback", default=std_name)
            unit = click.prompt("  Unit", default="")
            sub_dir = click.prompt("  Sub-directory (optional)", default="", show_default=False)
            prefix = click.prompt("  File prefix (optional)", default="", show_default=False)
            suffix = click.prompt("  File suffix (optional)", default="", show_default=False)
            nc_names = [n.strip() for n in nc_name.split(",") if n.strip()]
            primary = nc_names[0] if nc_names else std_name
            entry = {"varname": primary, "varunit": unit}
            if sub_dir:
                entry["sub_dir"] = sub_dir
            if prefix:
                entry["prefix"] = prefix
            if suffix:
                entry["suffix"] = suffix
            if len(nc_names) > 1:
                entry["fallbacks"] = [{"varname": nc, "varunit": unit} for nc in nc_names[1:]]
            new_vars[std_name] = entry

    # Backup previous catalog before write (recovery path) + invalidate
    # singleton cache so subsequent get_registry() reads see the new entry.
    latest_is_new = is_new
    merged_vars = {}
    updated_keys = []
    added_keys = []
    skipped_keys = []
    cancellation_reason = None
    creating_user_overlay = False
    backup_path = None
    with _catalog_write_lock(catalog_path):
        latest_catalog = _load_catalog(catalog_path)
        latest_user_key, _ = _case_insensitive_catalog_entry(latest_catalog, write_name)
        creating_user_overlay = latest_user_key is None and RegistryManager().get_model(write_name) is not None
        catalog_name, existing = _resolve_existing(latest_catalog)
        latest_is_new = not bool(existing)

        # Build/update profile and merge variables against the locked snapshot.
        profile = existing.copy()
        profile["name"] = profile.get("name") or catalog_name
        if description:
            profile["description"] = description
        elif latest_is_new and "description" not in profile:
            profile["description"] = f"{catalog_name} model profile"
        if data_type:
            profile["data_type"] = data_type
        elif "data_type" not in profile:
            profile["data_type"] = "grid"
        if grid_res is not None:
            if profile.get("data_type") == "stn":
                raise click.ClickException("--grid-res is not valid for station model profiles")
            profile["grid_res"] = grid_res
        elif profile.get("data_type") == "stn":
            profile.pop("grid_res", None)
        if tim_res:
            profile["tim_res"] = tim_res
        elif "tim_res" not in profile:
            profile["tim_res"] = "Month"
        if description is None and latest_is_new and "description" not in profile:
            profile["description"] = f"{catalog_name} model profile"
        if time_offset:
            offsets = deepcopy(profile.get("time_offset") or {})
            for key, value in _parse_time_offsets(time_offset).items():
                offsets[key] = _merge_time_offset_value(offsets.get(key), value)
            profile["time_offset"] = offsets

        vars_to_merge = deepcopy(new_vars)
        existing_vars = profile.get("variables") or {}
        _parse_var_attrs(var_attr, vars_to_merge, existing_vars)
        parse_fallbacks(fallback, vars_to_merge, existing_vars)
        source_names = _source_var_names({**existing_vars, **vars_to_merge})
        for std_name, entry in vars_to_merge.items():
            for fb in entry.get("fallbacks") or []:
                _validate_expression(
                    fb.get("convert", ""),
                    label=f"fallback conversion for {std_name}",
                    allowed_names=source_names,
                )
        if latest_is_new and not vars_to_merge:
            cancellation_reason = "no_vars"
        else:
            profile_updates = {k: v for k, v in profile.items() if k != "variables"}
            profile_updates["variables"] = vars_to_merge
            profile, added_keys, updated_keys, skipped_keys = _merge_model_profile_descriptor(
                existing,
                profile_updates,
                append_only=append_only,
            )
            merged_vars = profile["variables"]
            profile.pop("_deleted", None)
            latest_catalog[catalog_name] = profile
            backup_path = _backup_then_write(catalog_path, latest_catalog)
    if cancellation_reason == "no_vars":
        click.secho("No variables defined. Registration cancelled.", fg="yellow")
        return
    _invalidate_registry_caches()

    # Report
    if creating_user_overlay:
        click.secho(f"~ Creating user overlay over bundled model profile '{catalog_name}'", fg="cyan")
    if latest_is_new:
        click.secho(f"✓ Created model profile '{catalog_name}' ({len(merged_vars)} variables)", fg="green")
    else:
        parts = []
        if added_keys:
            parts.append(f"{len(added_keys)} added")
        if updated_keys:
            parts.append(f"{len(updated_keys)} updated")
        if skipped_keys:
            parts.append(f"{len(skipped_keys)} skipped (use without --append-only to update)")
        if not parts:
            parts.append("metadata updated")
        click.secho(f"✓ Updated '{catalog_name}': {', '.join(parts)} ({len(merged_vars)} total)", fg="green")

    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")
    click.echo(f"Verify: openbench model show {catalog_name}")


@model.command()
@click.argument("name")
@click.argument("variable_name")
def remove_var(name, variable_name):
    """Remove a variable from a model profile.

    Example: openbench model remove-var CoLM2024 Snow_Depth
    """
    from openbench.data.registry.manager import RegistryManager, get_writable_model_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
        _safe_load_catalog,
    )

    catalog_path = get_writable_model_catalog_path()
    with _catalog_write_lock(catalog_path):
        catalog = _safe_load_catalog(catalog_path)

        write_name, alias_message = _resolve_write_name(name, RegistryManager())
        if alias_message:
            click.secho(f"~ {alias_message}", fg="cyan")

        catalog_name = write_name
        key, profile = _case_insensitive_catalog_entry(catalog, catalog_name)
        if key:
            catalog_name = key
        if profile is None:
            existing = RegistryManager().get_model(catalog_name)
            if existing is not None:
                catalog_name = existing.name
                profile = existing.to_dict()
        if profile is None:
            raise click.ClickException(f"Model not found: {name}")

        variables = profile.get("variables") or {}
        variable_key = get_mapping_key_case_insensitive(variables, variable_name)
        if variable_key is None:
            raise click.ClickException(f"Variable '{variable_name}' not in {name}")

        del variables[variable_key]
        profile["variables"] = variables
        profile["name"] = profile.get("name") or catalog_name
        deleted = list(profile.get("_delete_variables") or [])
        if variable_key not in deleted:
            deleted.append(variable_key)
        profile["_delete_variables"] = deleted
        catalog[catalog_name] = profile
        backup_path = _backup_then_write(catalog_path, catalog)
    _invalidate_registry_caches()

    click.secho(f"✓ Removed '{variable_key}' from {name} ({len(variables)} remaining)", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")
    if not variables:
        click.secho("  ⚠ Profile has no variables left; consider `openbench model delete`.", fg="yellow")


@model.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Delete without confirmation.")
def delete(name, yes):
    """Delete a user model profile or user overlay."""
    from openbench.data.registry.manager import RegistryManager, get_writable_model_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
        _safe_load_catalog,
    )

    write_name, alias_message = _resolve_write_name(name, RegistryManager())
    catalog_path = get_writable_model_catalog_path()
    catalog = _safe_load_catalog(catalog_path)
    key, _ = _case_insensitive_catalog_entry(catalog, write_name)
    if key is None:
        if RegistryManager().get_model(write_name) is not None:
            raise click.ClickException(
                f"'{write_name}' is a bundled model profile. Delete only removes user profiles/overlays."
            )
        raise click.ClickException(f"Model not found in user catalog: {name}")
    if not yes and not click.confirm(f"Delete model profile '{key}' from {catalog_path}?"):
        return
    with _catalog_write_lock(catalog_path):
        catalog = _safe_load_catalog(catalog_path)
        locked_key, _ = _case_insensitive_catalog_entry(catalog, key)
        if locked_key is None:
            raise click.ClickException(f"Model no longer exists in user catalog: {key}")
        key = locked_key
        catalog.pop(key, None)
        backup_path = _backup_then_write(catalog_path, catalog)
    _invalidate_registry_caches()
    if alias_message:
        click.secho(f"~ {alias_message}", fg="cyan")
    click.secho(f"✓ Deleted model profile '{key}'", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")


@model.command("export")
@click.argument("name")
@click.option("-o", "--output", type=click.Path(file_okay=True, dir_okay=False), default=None)
def export_model(name, output):
    """Export a resolved model profile as a standalone YAML file."""
    from openbench.data.registry.manager import RegistryManager

    model_obj = RegistryManager().get_model(name)
    if model_obj is None:
        raise click.ClickException(f"Model profile not found: {name}")
    data = yaml.safe_dump(model_obj.to_dict(), sort_keys=False)
    if output:
        path = Path(output).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        except OSError as exc:
            raise click.ClickException(f"Failed to export model profile to {path}: {exc}") from exc
        click.secho(f"✓ Exported {model_obj.name} to {path}", fg="green")
    else:
        click.echo(data.rstrip())


@model.command("import")
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--yes", "-y", is_flag=True, help="Overwrite existing user profile without confirmation.")
def import_model(path, yes):
    """Import a standalone model profile YAML into the user catalog."""
    from openbench.data.registry.manager import _build_model, get_writable_model_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
    )

    src = Path(path).expanduser()
    data = _read_yaml_mapping(src, label="model profile", missing_ok=False)
    if not isinstance(data, dict) or "name" not in data:
        raise click.ClickException("Imported model YAML must be a mapping with a 'name' field")
    name = _validate_model_name(data["name"])
    try:
        _build_model(data)
    except Exception as exc:
        raise click.ClickException(f"Invalid model profile: {exc}") from exc
    catalog_path = get_writable_model_catalog_path()
    catalog = _load_catalog(catalog_path)
    key, _ = _case_insensitive_catalog_entry(catalog, name)
    write_name = key or name
    if key is not None and not yes and not click.confirm(f"Overwrite existing profile '{key}'?"):
        return
    data["name"] = write_name
    with _catalog_write_lock(catalog_path):
        catalog = _load_catalog(catalog_path)
        locked_key, _ = _case_insensitive_catalog_entry(catalog, write_name)
        final_name = locked_key or write_name
        data["name"] = final_name
        catalog[final_name] = data
        backup_path = _backup_then_write(catalog_path, catalog)
    _invalidate_registry_caches()
    click.secho(f"✓ Imported model profile '{name}'", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")


@model.command("rename")
@click.argument("old")
@click.argument("new")
@click.option("--yes", "-y", is_flag=True, help="Rename without confirmation.")
def rename(old, new, yes):
    """Rename a user model profile in the user catalog."""
    from openbench.data.registry.manager import get_writable_model_catalog_path
    from openbench.data.registry.scanner import (
        _backup_then_write,
        _catalog_write_lock,
        _invalidate_registry_caches,
        _safe_load_catalog,
    )

    new_name = _validate_model_name(new)
    catalog_path = get_writable_model_catalog_path()
    catalog = _safe_load_catalog(catalog_path)
    old_key, _profile = _case_insensitive_catalog_entry(catalog, old)
    if old_key is None:
        raise click.ClickException(f"Model not found in user catalog: {old}")
    new_key, _ = _case_insensitive_catalog_entry(catalog, new_name)
    if new_key is not None:
        raise click.ClickException(f"Model already exists in user catalog: {new_key}")
    if not yes and not click.confirm(f"Rename '{old_key}' to '{new_name}'?"):
        return
    with _catalog_write_lock(catalog_path):
        catalog = _safe_load_catalog(catalog_path)
        old_key, profile = _case_insensitive_catalog_entry(catalog, old)
        if old_key is None:
            raise click.ClickException(f"Model not found in user catalog: {old}")
        new_key, _ = _case_insensitive_catalog_entry(catalog, new_name)
        if new_key is not None:
            raise click.ClickException(f"Model already exists in user catalog: {new_key}")
        profile = dict(profile)
        profile["name"] = new_name
        catalog.pop(old_key, None)
        catalog[new_name] = profile
        backup_path = _backup_then_write(catalog_path, catalog)
    _invalidate_registry_caches()
    click.secho(f"✓ Renamed '{old_key}' to '{new_name}'", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")


@model.command("alias")
@click.argument("alias_name", required=False)
@click.argument("canonical_name", required=False)
def alias(alias_name, canonical_name):
    """List or create user model aliases."""
    from openbench.data.registry.manager import RegistryManager, get_writable_model_catalog_path
    from openbench.data.registry.scanner import _backup_then_write, _catalog_write_lock, _invalidate_registry_caches

    alias_path = get_writable_model_catalog_path().parent / "aliases.yaml"
    if not alias_name and not canonical_name:
        aliases = _read_yaml_mapping(alias_path, label="model aliases")
        for key, value in sorted({**RegistryManager.MODEL_ALIASES, **aliases}.items()):
            click.echo(f"{key} -> {value}")
        return
    if not alias_name or not canonical_name:
        raise click.ClickException("Usage: openbench model alias ALIAS CANONICAL")
    alias_key = normalize_name(_validate_model_name(alias_name))
    canonical = _validate_model_name(canonical_name)
    if RegistryManager().get_model(canonical) is None:
        raise click.ClickException(f"Canonical model not found: {canonical}")
    with _catalog_write_lock(alias_path):
        aliases = _read_yaml_mapping(alias_path, label="model aliases")
        aliases[alias_key] = canonical
        backup_path = _backup_then_write(alias_path, aliases)
    _invalidate_registry_caches()
    click.secho(f"✓ Added model alias '{alias_key}' -> '{canonical}'", fg="green")
    if backup_path is not None:
        click.echo(f"Backup: {backup_path}")


@model.command("status")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def status(fmt):
    """Show model registry status."""
    from openbench.data.registry.manager import RegistryManager, get_writable_model_catalog_path

    models = RegistryManager().list_models()
    if fmt == "json":
        click.echo(
            json.dumps(
                {
                    "model_profiles": len(models),
                    "user_catalog": str(get_writable_model_catalog_path()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"Registry: {len(models)} model profiles available")
    click.echo(f"User catalog: {get_writable_model_catalog_path()}")


@model.command("path")
@click.argument("name")
def path(name):
    """Print whether a model profile is bundled or from the user catalog."""
    from openbench.data.registry.manager import RegistryManager

    model_obj = RegistryManager().get_model(name)
    if model_obj is None:
        raise click.ClickException(f"Model profile not found: {name}")
    click.echo(_model_source_label(model_obj.name))


@model.command("validate")
@click.argument("name")
def validate(name):
    """Validate a model profile for common completeness issues."""
    from openbench.data.registry.manager import RegistryManager

    model_obj = RegistryManager().get_model(name)
    if model_obj is None:
        raise click.ClickException(f"Model profile not found: {name}")
    warnings = []
    if not model_obj.variables:
        warnings.append("profile has no variables")
    source_names = _source_var_names(model_obj.variables)
    for var_name, mapping in sorted(model_obj.variables.items()):
        if not mapping.varname:
            warnings.append(f"{var_name}: missing varname")
        if not mapping.varunit:
            warnings.append(f"{var_name}: missing varunit")
        if mapping.compute:
            _validate_expression(
                mapping.compute,
                label=f"compute expression for {var_name}",
                allowed_names=source_names,
            )
        for fb in mapping.fallbacks or []:
            _validate_expression(
                fb.convert or "",
                label=f"fallback conversion for {var_name}",
                allowed_names=source_names,
            )
    if warnings:
        for message in warnings:
            click.secho(f"⚠ {message}", fg="yellow")
        raise SystemExit(1)
    click.secho(f"✓ Model profile '{model_obj.name}' looks valid", fg="green")
