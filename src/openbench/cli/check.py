"""openbench check command."""

from __future__ import annotations

import os
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import click

from openbench.cli._names import resolve_variable_filters
from openbench.cli._reference_errors import emit_reference_resolution_error
from openbench.cli._simulation_validation import simulation_root_errors
from openbench.config.provenance import PROVENANCE_FIELDS
from openbench.util.names import get_mapping_key_case_insensitive
from openbench.util.static_datasets import static_dataset_candidates, static_dataset_exists

_UNRESOLVED_ENV_RE = re.compile(r"(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%)")
_VALID_DATA_GROUPBY = {"year", "month", "day", "single"}
_VALID_WEIGHTS = {"area", "mass", "none"}
_BASIC_ANALYSIS_ITEMS = {"Basic", "Mean", "Median", "Max", "Min", "Sum"}


def _reference_root_value(cfg, ref_ds) -> str | None:
    root_dir = getattr(ref_ds, "root_dir", None)
    if getattr(ref_ds, "data_type", None) == "stn":
        return root_dir or cfg.reference.data_root
    return cfg.reference.data_root or root_dir


def _expanded_path(raw: str, what: str) -> tuple[Path | None, str | None]:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if _UNRESOLVED_ENV_RE.search(expanded):
        return None, f"{what} contains unresolved environment variable: {raw}"
    return Path(expanded), None


def _expanded_reference_path(raw: str) -> tuple[Path | None, str | None]:
    return _expanded_path(raw, "Reference root")


def _has_nearby_netcdf_files(path: Path) -> bool:
    from openbench.data.coordinates import glob_nc

    if glob_nc(path):
        return True
    try:
        children = sorted(child for child in path.iterdir() if child.is_dir())
    except OSError:
        return False
    return any(glob_nc(child) for child in children)


def _figlib_names(section: str) -> set[str]:
    from importlib.resources import files

    import yaml

    # Keep this as a Traversable so it works when OpenBench is imported
    # directly from a zipped wheel.
    figlib = files("openbench.data.fignml") / "figlib.yaml"
    try:
        data = yaml.safe_load(figlib.read_text(encoding="utf-8")) or {}
    except OSError:
        return set()
    names = set()
    for key in data.get(section) or {}:
        names.add(key[:-7] if key.endswith("_source") else key)
    return names


def _suggestion(name: str, valid: set[str]) -> str:
    matches = get_close_matches(name, sorted(valid), n=1, cutoff=0.55)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def _validate_string_list(raw: Any, label: str) -> tuple[list[str], list[str]]:
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [f"{label} must be a list of strings, got {type(raw).__name__}"]
    errors = []
    values = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            errors.append(f"{label}[{idx}] must be a string, got {type(item).__name__}")
        else:
            values.append(item)
    return values, errors


def _validate_names(kind: str, values: list[str], valid: set[str]) -> list[str]:
    errors = []
    for name in values:
        if name not in valid:
            errors.append(f"Unknown {kind} '{name}'.{_suggestion(name, valid)}")
    return errors


def _timezone_findings(value: Any, label: str) -> tuple[list[str], list[str]]:
    if value is None:
        return [], []
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{label} must be a numeric hour offset, got {type(value).__name__}"], []
    if value < -12 or value > 14:
        return [], [f"{label} timezone {value} is outside the usual [-12, 14] hour range"]
    return [], []


def _data_groupby_error(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.lower() not in _VALID_DATA_GROUPBY:
        return f"{label} data_groupby '{value}' is invalid; expected one of Year, Month, Day, single"
    return None


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path if path.exists() else path.parent
    while current != current.parent:
        if current.exists():
            return current
        current = current.parent
    return current if current.exists() else None


def _config_findings(cfg) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    output_path, output_error = _expanded_path(str(cfg.project.output_dir), "project.output_dir")
    if output_error:
        errors.append(output_error)
    elif output_path is not None:
        if output_path.exists() and not output_path.is_dir():
            errors.append(f"project.output_dir exists but is not a directory: {output_path}")
        else:
            writable_target = output_path if output_path.exists() else _nearest_existing_parent(output_path)
            if writable_target is None:
                warnings.append(f"project.output_dir has no existing parent to check: {output_path}")
            elif output_path.exists() and not os.access(writable_target, os.W_OK):
                errors.append(f"project.output_dir parent is not writable: {writable_target}")
            elif not os.access(writable_target, os.W_OK):
                warnings.append(f"project.output_dir parent may not be writable: {writable_target}")

    weight = cfg.project.weight
    if weight is not None and str(weight).lower() not in _VALID_WEIGHTS:
        errors.append(f"project.weight must be one of {sorted(_VALID_WEIGHTS)}, got '{weight}'")

    tz_errors, tz_warnings = _timezone_findings(cfg.project.timezone, "project")
    errors.extend(tz_errors)
    warnings.extend(tz_warnings)

    metric_values, metric_type_errors = _validate_string_list(cfg.metrics, "metrics")
    score_values, score_type_errors = _validate_string_list(cfg.scores, "scores")
    errors.extend(metric_type_errors)
    errors.extend(score_type_errors)
    if metric_values:
        from openbench.core.registry import IMPLEMENTED_METRICS

        errors.extend(_validate_names("metric", metric_values, IMPLEMENTED_METRICS))
    if score_values:
        from openbench.core.registry import IMPLEMENTED_SCORES

        errors.extend(_validate_names("score", score_values, IMPLEMENTED_SCORES))

    comparison_items, comparison_type_errors = _validate_string_list(
        cfg.comparison.items,
        "comparison.items",
    )
    statistic_items, statistic_type_errors = _validate_string_list(
        cfg.statistics.items,
        "statistics.items",
    )
    errors.extend(comparison_type_errors)
    errors.extend(statistic_type_errors)
    if comparison_items:
        valid = _figlib_names("comparison_nml") | _BASIC_ANALYSIS_ITEMS
        errors.extend(_validate_names("comparison item", comparison_items, valid))
    if statistic_items:
        valid = _figlib_names("statistic_nml") | _BASIC_ANALYSIS_ITEMS | {"False_Discovery_Rate"}
        errors.extend(_validate_names("statistics item", statistic_items, valid))

    for label, entry in cfg.simulation.items():
        err = _data_groupby_error(entry.data_groupby, f"simulation.{label}")
        if err:
            errors.append(err)
        for var_name, inline in (entry.variables or {}).items():
            if not isinstance(inline, dict):
                errors.append(f"simulation.{label}.variables.{var_name} must be a mapping, got {type(inline).__name__}")
                continue
            err = _data_groupby_error(inline.get("data_groupby"), f"simulation.{label}.variables.{var_name}")
            if err:
                errors.append(err)
            tz_errors, tz_warnings = _timezone_findings(
                inline.get("timezone"),
                f"simulation.{label}.variables.{var_name}",
            )
            errors.extend(tz_errors)
            warnings.extend(tz_warnings)

    return errors, warnings


def _groupby_static_dataset_findings(cfg) -> list[str]:
    errors: list[str] = []
    groupby_requirements = {
        "IGBP_groupby": ("IGBP.nc", cfg.project.IGBP_groupby),
        "PFT_groupby": ("PFT.nc", cfg.project.PFT_groupby),
        "climate_zone_groupby": ("Climate_zone.nc", cfg.project.climate_zone_groupby),
    }
    for label, (filename, enabled) in groupby_requirements.items():
        if not enabled:
            continue
        candidates = static_dataset_candidates(filename)
        if not static_dataset_exists(filename):
            errors.append(
                f"{label} requires static dataset {filename} (checked: {', '.join(str(p) for p in candidates)})"
            )
    return errors


def _reference_data_findings(cfg, resolved_ref) -> tuple[list[str], list[str], list[str]]:
    if resolved_ref.status != "ok" or resolved_ref.ref_ds is None:
        return [], [], []

    raw_root = _reference_root_value(cfg, resolved_ref.ref_ds)
    if not raw_root or not str(raw_root).strip():
        return (
            [
                f"Reference root is not configured for {resolved_ref.resolved_name}; "
                "set reference.data_root or register the reference with --root-dir."
            ],
            [],
            [],
        )

    root_path, error = _expanded_reference_path(str(raw_root).strip())
    if error:
        return [error], [], []
    if root_path is None:
        return ["Reference root could not be resolved."], [], []

    from openbench.config.adapter import _find_nc_dir

    warnings: list[str] = []
    info: list[str] = [f"effective root: {root_path}"]
    sub_dir = getattr(resolved_ref.var_map, "sub_dir", None)

    if sub_dir:
        direct_path = root_path / sub_dir
        candidate = Path(_find_nc_dir(str(direct_path), str(root_path), str(sub_dir)))
        if candidate.exists() and candidate.is_dir():
            if candidate != direct_path:
                warnings.append(f"Reference data path uses fallback: {candidate}")
            if not _has_nearby_netcdf_files(candidate):
                warnings.append(f"Reference root has no NetCDF files found near: {candidate}")
            return [], warnings, info
        if direct_path.exists() and not direct_path.is_dir():
            return [f"Reference data path is not a directory: {direct_path}"], warnings, info
        return [f"Reference data path does not exist: {direct_path}"], warnings, info

    if not root_path.exists() or not root_path.is_dir():
        return [f"Reference root does not exist: {root_path}"], warnings, info
    if not _has_nearby_netcdf_files(root_path):
        warnings.append(f"Reference root has no NetCDF files found near: {root_path}")
    return [], warnings, info


def _tim_res_rank(value: str | None) -> int:
    from openbench.data.registry.scanner import _tim_res_rank as scanner_tim_res_rank

    return scanner_tim_res_rank(value or "")


def _years_findings(ref_name: str, ref_years: Any, project_years: list[int]) -> tuple[list[str], list[str]]:
    if not ref_years:
        return [], [f"Reference '{ref_name}' has no registered years; using project years at runtime"]
    if not isinstance(ref_years, list) or len(ref_years) < 2:
        return [], [f"Reference '{ref_name}' years metadata is incomplete: {ref_years}"]
    ref_start, ref_end = ref_years[0], ref_years[1]
    proj_start, proj_end = project_years[0], project_years[1]
    if ref_end < proj_start or ref_start > proj_end:
        return [
            f"Reference '{ref_name}' years [{ref_start}, {ref_end}] do not overlap "
            f"project years [{proj_start}, {proj_end}]"
        ], []
    if ref_start > proj_start or ref_end < proj_end:
        return [], [
            f"Reference '{ref_name}' years [{ref_start}, {ref_end}] only partially cover "
            f"project years [{proj_start}, {proj_end}]"
        ]
    return [], []


def _fulllist_path_findings(raw: str, label: str, root_dir: str | None = None) -> tuple[list[str], list[str]]:
    from openbench.config.adapter import _resolve_root_relative_path

    resolved = _resolve_root_relative_path(raw, root_dir)
    path, error = _expanded_path(resolved, label)
    if error:
        return [error], []
    if path is None or not path.exists() or not path.is_file():
        return [f"Station fulllist does not exist: {path or resolved}"], []
    if not os.access(path, os.R_OK):
        return [f"Station fulllist is not readable: {path}"], []
    return [], []


def _reference_metadata_findings(
    cfg,
    resolved_ref,
    target_tim_res: str | None,
    *,
    file_checks: bool = True,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if resolved_ref.status != "ok" or resolved_ref.ref_ds is None:
        return errors, warnings

    ref_ds = resolved_ref.ref_ds
    ref_name = resolved_ref.resolved_name
    ref_tim_res = getattr(ref_ds, "tim_res", None)
    if ref_tim_res and target_tim_res:
        ref_rank = _tim_res_rank(ref_tim_res)
        target_rank = _tim_res_rank(target_tim_res)
        if ref_rank < 0:
            warnings.append(f"Reference '{ref_name}' time resolution '{ref_tim_res}' is not recognized")
        elif target_rank >= 0 and ref_rank < target_rank:
            errors.append(
                f"Reference '{ref_name}' time resolution {ref_tim_res} is coarser than target {target_tim_res}"
            )

    err = _data_groupby_error(getattr(ref_ds, "data_groupby", None), f"reference.{ref_name}")
    if err:
        errors.append(err)

    tz_errors, tz_warnings = _timezone_findings(getattr(ref_ds, "timezone", None), f"reference.{ref_name}")
    errors.extend(tz_errors)
    warnings.extend(tz_warnings)

    year_errors, year_warnings = _years_findings(ref_name, getattr(ref_ds, "years", None), cfg.project.years)
    errors.extend(year_errors)
    warnings.extend(year_warnings)

    if file_checks and getattr(ref_ds, "data_type", None) == "stn":
        var_map = resolved_ref.var_map
        fulllist = getattr(var_map, "fulllist", None) or getattr(ref_ds, "fulllist", None)
        root_dir = getattr(ref_ds, "root_dir", None) or _reference_root_value(cfg, ref_ds)
        if fulllist:
            list_errors, list_warnings = _fulllist_path_findings(
                str(fulllist),
                f"reference.{ref_name}.fulllist",
                root_dir,
            )
            errors.extend(list_errors)
            warnings.extend(list_warnings)
        elif not getattr(ref_ds, "station_matching", None):
            warnings.append(
                f"Station reference '{ref_name}' has no fulllist; "
                "runtime will rely on station matching or custom filters"
            )

    return errors, warnings


def _effective_sim_values(entry, model_profile, var_name: str) -> dict[str, Any]:
    inline_variables = entry.variables or {}
    inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
    inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
    return {
        "inline": inline,
        "data_type": inline.get("data_type")
        or entry.data_type
        or (getattr(model_profile, "data_type", None) if model_profile else "grid"),
        "tim_res": inline.get("tim_res")
        or entry.tim_res
        or (getattr(model_profile, "tim_res", None) if model_profile else None),
        "grid_res": inline.get("grid_res")
        if inline.get("grid_res") is not None
        else (
            entry.grid_res
            if entry.grid_res is not None
            else (getattr(model_profile, "grid_res", None) if model_profile else None)
        ),
        "fulllist": inline.get("fulllist") if "fulllist" in inline else entry.fulllist,
    }


def _append_simulation_model_findings(findings: dict[str, dict[str, list[str]]], cfg, registry) -> None:
    for label, entry in cfg.simulation.items():
        can_validate_model = hasattr(registry, "get_model")
        model_profile = registry.get_model(entry.model) if can_validate_model else None
        if can_validate_model and model_profile is None:
            inline_variables = entry.variables or {}
            missing_inline = []
            for var_name in cfg.evaluation.variables:
                inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
                inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
                if not inline.get("varname"):
                    missing_inline.append(var_name)
            if missing_inline:
                findings[label]["errors"].append(
                    f"Model '{entry.model}' is not registered and lacks inline varname for: {', '.join(missing_inline)}"
                )
            else:
                findings[label]["info"].append(
                    f"Model '{entry.model}' has no registry profile; using inline variable mappings"
                )
        elif model_profile is not None:
            profile_name = getattr(model_profile, "name", entry.model)
            if str(profile_name).lower() != str(entry.model).lower():
                findings[label]["info"].append(f"model alias '{entry.model}' resolved to '{profile_name}'")

        for var_name in cfg.evaluation.variables:
            inline_variables = entry.variables or {}
            inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
            inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
            profile_variables = getattr(model_profile, "variables", {}) if model_profile is not None else {}
            profile_key = (
                get_mapping_key_case_insensitive(profile_variables, var_name) if model_profile is not None else None
            )
            if model_profile is not None and profile_key is None and not inline:
                findings[label]["errors"].append(
                    f"Variable '{var_name}' is not defined in model profile "
                    f"'{getattr(model_profile, 'name', entry.model)}'"
                )


def _simulation_model_error_messages(cfg, registry) -> list[str]:
    findings: dict[str, dict[str, list[str]]] = {
        label: {"errors": [], "warnings": [], "info": []} for label in cfg.simulation
    }
    _append_simulation_model_findings(findings, cfg, registry)
    return [message for label in cfg.simulation for message in findings[label]["errors"]]


def _simulation_findings(
    cfg,
    registry,
    *,
    comparison_only: bool,
    only_drawing: bool = False,
) -> dict[str, dict[str, list[str]]]:
    findings: dict[str, dict[str, list[str]]] = {
        label: {"errors": [], "warnings": [], "info": []} for label in cfg.simulation
    }

    output_only = comparison_only or only_drawing
    if not output_only:
        for label, message in simulation_root_errors(cfg):
            findings[label]["errors"].append(message)

    _append_simulation_model_findings(findings, cfg, registry)

    if output_only:
        return findings

    for label, entry in cfg.simulation.items():
        can_validate_model = hasattr(registry, "get_model")
        model_profile = registry.get_model(entry.model) if can_validate_model else None
        for var_name in cfg.evaluation.variables:
            values = _effective_sim_values(entry, model_profile, var_name)
            if str(values["data_type"]).lower() == "stn":
                fulllist = values["fulllist"]
                if fulllist:
                    inline_variables = entry.variables or {}
                    inline_key = get_mapping_key_case_insensitive(inline_variables, var_name)
                    inline = inline_variables.get(inline_key, {}) if inline_key is not None else {}
                    if isinstance(inline, dict) and inline.get("fulllist"):
                        fulllist_label = f"simulation.{label}.variables.{var_name}.fulllist"
                    else:
                        fulllist_label = f"simulation.{label}.fulllist"
                    list_errors, list_warnings = _fulllist_path_findings(
                        str(fulllist),
                        fulllist_label,
                        entry.root_dir,
                    )
                    findings[label]["errors"].extend(list_errors)
                    findings[label]["warnings"].extend(list_warnings)
                else:
                    findings[label]["warnings"].append(
                        f"Station simulation '{label}' variable '{var_name}' has no fulllist; "
                        "runtime will rely on station auto-scan or custom filters"
                    )

    return findings


def _format_optional_list(value: Any) -> str:
    if value is None:
        return "all"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


@click.command()
@click.option(
    "--comparison-only",
    is_flag=True,
    help="Validate for comparison-only runs and skip local simulation root checks.",
)
@click.option(
    "--strict-reference",
    "--strict",
    is_flag=True,
    help="Treat low-confidence reference metadata as errors without editing the YAML.",
)
@click.option(
    "--variable",
    "--variables",
    "variables",
    multiple=True,
    help="Validate only specified evaluation variable (repeatable). --variables retained as alias.",
)
@click.argument("config", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def check(config, comparison_only=False, strict_reference=False, variables=()):
    """Validate config file and check data availability."""
    from openbench.cli.run import _expand_config_paths
    from openbench.config import ConfigError, load_config

    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"✗ Config error: {e}", fg="red", bold=True)
        raise SystemExit(1) from e
    _expand_config_paths(cfg)
    if variables:
        cfg.evaluation.variables = resolve_variable_filters(variables, cfg.evaluation.variables)

    click.secho("Config validation:", bold=True)
    click.secho("  ✓ YAML syntax valid", fg="green")
    click.secho("  ✓ Schema validation passed", fg="green")
    click.secho(
        f"  ✓ Year range [{cfg.project.years[0]}, {cfg.project.years[1]}] valid",
        fg="green",
    )

    has_errors = False
    config_errors, config_warnings = _config_findings(cfg)
    for message in config_errors:
        click.secho(f"  ✗ {message}", fg="red")
        has_errors = True
    for message in config_warnings:
        click.secho(f"  ⚠ {message}", fg="yellow")

    # Count total resolved entries (sum of single + list values per variable)
    _n_ref_total = sum(1 if isinstance(v, str) else len(v) for v in cfg.reference.sources.values())
    _n_ref_vars = len(cfg.reference.sources)
    _ref_summary = (
        f"{_n_ref_total} sources for {_n_ref_vars} variables"
        if _n_ref_total != _n_ref_vars
        else f"{_n_ref_vars} sources"
    )
    click.secho(f"\nReference data ({_ref_summary}):", bold=True)
    from openbench.config.resolver import (
        PROVENANCE_LOW,
        PROVENANCE_MEDIUM,
        derive_target_resolution_context,
        resolve_all_references,
    )
    from openbench.data.registry.manager import get_registry

    mgr = get_registry()
    strict = cfg.project.strict_reference or strict_reference

    try:
        resolved = resolve_all_references(cfg, mgr, strict=strict)
    except Exception as e:
        # Multi-line context (resolver hint + remediation) already emitted; exit silently.
        emit_reference_resolution_error(str(e), prefix="  ✗ ")
        raise SystemExit(1) from e

    for r in resolved:
        if r.status == "ok":
            if r.resolved_name != r.source_name:
                click.secho(
                    f"  ✓ {r.var_name} → {r.source_name} → {r.resolved_name} "
                    f"({r.ref_ds.data_type}, {r.ref_ds.tim_res}, "
                    f"{f'{r.ref_ds.grid_res}°' if r.ref_ds.grid_res is not None else 'N/A'})",
                    fg="cyan",
                )
            else:
                click.secho(
                    f"  ✓ {r.var_name} → {r.source_name} ({r.ref_ds.data_type}, {r.ref_ds.tim_res})",
                    fg="green",
                )
            ds_prov = getattr(r.ref_ds, "_provenance", None) or {}
            for fld in PROVENANCE_FIELDS:
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
                            f"    ⚠ {fld}: {value} (default - not confirmed from NC or profile)",
                            fg="yellow",
                        )
                elif source in PROVENANCE_MEDIUM:
                    click.secho(
                        f"    ~ {fld}: {value} (inferred from directory structure)",
                        fg="cyan",
                    )

            try:
                target_ctx = derive_target_resolution_context(cfg, mgr, var_name=r.var_name)
            except ConfigError:
                target_ctx = None
            target_tim_res = target_ctx.tim_res if target_ctx is not None else None
            ref_meta_errors, ref_meta_warnings = _reference_metadata_findings(
                cfg,
                r,
                target_tim_res,
                file_checks=not comparison_only,
            )
            if comparison_only:
                ref_errors, ref_warnings, ref_info = [], [], []
            else:
                ref_errors, ref_warnings, ref_info = _reference_data_findings(cfg, r)
            for message in ref_info:
                click.echo(f"    {message}")
            for message in [*ref_meta_errors, *ref_errors]:
                click.secho(f"    ✗ {message}", fg="red")
                has_errors = True
            for message in [*ref_meta_warnings, *ref_warnings]:
                click.secho(f"    ⚠ {message}", fg="yellow")
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
                    f"  ✗ {r.var_name} → {r.source_name} "
                    "(not in registry; runtime fallback would use minimal defaults)",
                    fg="red",
                )
                has_errors = True
            else:
                click.secho(f"  ✗ {r.var_name}: no reference configured", fg="red")
                has_errors = True

    n_tasks = len(resolved) * len(cfg.simulation)
    if n_tasks:
        click.echo(f"  Evaluation tasks: {n_tasks} ({len(resolved)} references × {len(cfg.simulation)} simulations)")

    click.secho(f"\nSimulation data ({len(cfg.simulation)} models):", bold=True)
    sim_findings = _simulation_findings(
        cfg,
        mgr,
        comparison_only=comparison_only,
        only_drawing=cfg.project.only_drawing,
    )
    for label, entry in cfg.simulation.items():
        label_findings = sim_findings[label]
        if label_findings["errors"]:
            click.secho(
                f"  ✗ {label} (model: {entry.model}, root: {entry.root_dir})",
                fg="red",
            )
            has_errors = True
        else:
            click.secho(f"  ✓ {label} (model: {entry.model}, root: {entry.root_dir})", fg="green")
        for message in label_findings["info"]:
            click.secho(f"    ~ {message}", fg="cyan")
        for message in label_findings["errors"]:
            click.echo(f"    {message}")
        for message in label_findings["warnings"]:
            click.secho(f"    ⚠ {message}", fg="yellow")

    if cfg.metrics is not None:
        click.secho(f"\nMetrics: {_format_optional_list(cfg.metrics)}", bold=True)
    if cfg.scores is not None:
        click.secho(f"Scores: {_format_optional_list(cfg.scores)}", bold=True)

    click.secho("\nOptions:", bold=True)
    click.secho(f"  Time alignment: {cfg.project.time_alignment}")
    click.secho(f"  Unified mask: {cfg.project.unified_mask}")
    click.secho(f"  Comparison: {cfg.comparison.enabled}")
    click.secho(f"  Statistics: {cfg.statistics.enabled}")
    if comparison_only and cfg.project.only_drawing:
        click.secho(
            "  ✗ --comparison-only conflicts with project.only_drawing=true; choose one mode",
            fg="red",
        )
        has_errors = True
    if comparison_only:
        click.secho("  Check mode: comparison-only")
        if not cfg.comparison.enabled:
            click.secho("  ✗ comparison-only mode requires comparison.enabled: true", fg="red")
            has_errors = True
        else:
            from openbench.runner.local import comparison_only_preflight_errors

            for error in comparison_only_preflight_errors(cfg):
                click.secho(f"  ✗ {error.get('message', 'comparison-only preflight failed')}", fg="red")
                has_errors = True
    elif cfg.project.only_drawing:
        click.secho("  Check mode: only-drawing")
        from openbench.runner.local import existing_output_preflight_errors

        for error in existing_output_preflight_errors(cfg):
            click.secho(f"  ✗ {error.get('message', 'only-drawing preflight failed')}", fg="red")
            has_errors = True

    for message in _groupby_static_dataset_findings(cfg):
        click.secho(f"  ✗ {message}", fg="red")
        has_errors = True

    if has_errors:
        # Per-error context emitted above; exit silently with non-zero status.
        click.secho("\n✗ Config has errors. Please fix and re-check.", fg="red", bold=True)
        raise SystemExit(1)

    n_refs = len(resolved) if resolved else 0
    n_sims = len(cfg.simulation) if cfg.simulation else 0
    n_vars = len(cfg.evaluation.variables) if cfg.evaluation.variables else 0
    click.secho(
        f"\n✓ Config valid ({n_vars} variables, {n_refs} references, {n_sims} simulations). Ready to run.",
        fg="green",
        bold=True,
    )
