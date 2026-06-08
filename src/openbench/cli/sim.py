"""openbench sim commands."""

import csv
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import click

from openbench.cli._options import expand_existing_directory, expand_path

_REGISTER_MODEL_EXISTS_HINT = (
    "Use `openbench model register` to update variable mappings, "
    "pass --overwrite-model to merge the scanned draft, or omit --register-model."
)


@click.group()
def sim():
    """Manage simulation outputs."""


@sim.command()
@click.argument("roots", nargs=-1, required=True)
@click.option("--model", "model_name", default="auto", help="Model profile name, or 'auto'.")
@click.option("--case-depth", type=click.IntRange(0, 10), default=5, show_default=True)
@click.option("--case-pattern", default=None, help="Only include case labels matching this glob.")
@click.option("--exclude", multiple=True, help="Directory name or glob to exclude. Repeatable.")
@click.option(
    "--climatology",
    type=click.Choice(["auto", "off", "year", "month"]),
    default="auto",
    show_default=True,
    help="Detect or force climatology tim_res handling.",
)
@click.option("--dry-run", is_flag=True, help="Preview discovered cases without writing files.")
@click.option("--auto", "--yes", "-y", "auto", is_flag=True, help="Write output files without confirmation.")
@click.option("--output", default=None, help="Simulation YAML output path.")
@click.option("--report", default=None, help="Scan report YAML output path.")
@click.option("--station-output", default=None, help="Directory for generated station lists and merged station files.")
@click.option("--station-workers", type=click.IntRange(1, 64), default=4, show_default=True)
@click.option("--allow-unresolved", is_flag=True, help="Allow unresolved model inference in generated YAML.")
@click.option(
    "--allow-partial-stations",
    is_flag=True,
    help="Allow station materialization to drop sites silently. Default fails on dropped sites.",
)
@click.option(
    "--register-model",
    is_flag=True,
    help="Register a draft model profile from scanned metadata. Use with --model NAME.",
)
@click.option(
    "--overwrite-model",
    is_flag=True,
    help="Allow --register-model to merge into an existing model profile.",
)
def scan(
    roots,
    model_name,
    case_depth,
    case_pattern,
    exclude,
    climatology,
    dry_run,
    auto,
    output,
    report,
    station_output,
    station_workers,
    allow_unresolved,
    allow_partial_stations,
    register_model,
    overwrite_model,
):
    """Scan directories for simulation cases and generate config fragments."""
    from openbench.data.sim_scanner import materialize_station_cases, scan_simulation_roots

    model_name = str(model_name).strip()
    if model_name.casefold() == "auto":
        model_name = "auto"
    roots = tuple(str(expand_existing_directory(root, "Simulation root")) for root in roots)

    result = scan_simulation_roots(
        list(roots),
        model_name=model_name,
        case_depth=case_depth,
        case_pattern=case_pattern,
        exclude=exclude,
        climatology=climatology,
    )

    if not result.cases:
        raise click.ClickException("No simulation cases found.")

    register_model_name = None
    draft_model_profile = None
    if register_model:
        register_model_name = _resolve_registration_model_name(
            model_name,
            result,
            auto=auto,
            dry_run=dry_run,
        )
        if register_model_name != model_name:
            result = scan_simulation_roots(
                list(roots),
                model_name=register_model_name,
                case_depth=case_depth,
                case_pattern=case_pattern,
                exclude=exclude,
                climatology=climatology,
            )
            if not result.cases:
                raise click.ClickException("No simulation cases found.")
        draft_model_profile = _draft_model_profile_from_scan(register_model_name, result)
        if not dry_run:
            _ensure_model_profile_can_register(register_model_name, overwrite=overwrite_model)

    if result.unresolved and not allow_unresolved:
        _print_scan_summary(result, dry_run=dry_run)
        raise click.ClickException(_unresolved_message_for_cases(result.unresolved))

    _handle_climatology_candidates(
        result,
        climatology=climatology,
        auto=auto,
        dry_run=dry_run,
    )
    _print_scan_summary(result, dry_run=dry_run)

    if dry_run:
        if register_model_name:
            _print_model_registration_preview(result, register_model_name)
        click.secho("[DRY RUN] No files written.", fg="cyan", bold=True)
        return

    confirm_message = f"Write simulation config for {len(result.cases)} case(s)?"
    if register_model_name:
        confirm_message = (
            f"Register model profile '{register_model_name}' and write simulation config "
            f"for {len(result.cases)} case(s)?"
        )
    if not auto and not click.confirm(confirm_message):
        return

    sim_path, report_path, station_output_path = _resolve_output_paths(output, report, station_output)
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    station_output_path.parent.mkdir(parents=True, exist_ok=True)

    from openbench.data.registry.scanner import _atomic_yaml_write

    station_staging_path: Path | None = None
    try:
        if any(case.station_layout for case in result.cases):
            station_staging_path = Path(
                tempfile.mkdtemp(
                    prefix=f".{station_output_path.name}.",
                    suffix=".tmp",
                    dir=str(station_output_path.parent),
                )
            )
            materialize_station_cases(
                result,
                station_staging_path,
                num_workers=station_workers,
                allow_partial=allow_partial_stations,
            )
        else:
            materialize_station_cases(
                result,
                station_output_path,
                num_workers=station_workers,
                allow_partial=allow_partial_stations,
            )

        partial_station_cases = [case for case in result.cases if case.station_dropped_sites]
        if partial_station_cases and not allow_partial_stations:
            labels = ", ".join(case.label for case in partial_station_cases)
            _atomic_yaml_write(report_path, _report_yaml(result, case_depth=case_depth))
            raise click.ClickException(
                f"Station materialization dropped sites for: {labels}.\n"
                + _unresolved_message_for_cases(partial_station_cases)
                + "\nDropped site IDs are listed in `station_dropped_sites` of the "
                + f"report YAML for inspection: {report_path}"
            )

        if station_staging_path is not None:
            if station_output_path.exists():
                shutil.rmtree(station_output_path)
            os.replace(station_staging_path, station_output_path)
            _rebase_station_artifacts(result, station_staging_path, station_output_path)
            station_staging_path = None

        if register_model_name and draft_model_profile:
            catalog_path = _register_model_profile_from_scan(
                register_model_name,
                draft_model_profile,
                overwrite=overwrite_model,
            )
            click.secho(f"Wrote draft model profile {catalog_path}", fg="green", bold=True)

        _atomic_yaml_write(sim_path, _simulation_yaml(result, sim_path=sim_path))
        _atomic_yaml_write(report_path, _report_yaml(result, case_depth=case_depth))

    finally:
        if station_staging_path is not None and station_staging_path.exists():
            shutil.rmtree(station_staging_path, ignore_errors=True)

    click.secho(f"Wrote {sim_path}", fg="green", bold=True)
    click.secho(f"Wrote {report_path}", fg="green", bold=True)


def _rebase_station_artifacts(result, old_root: Path, new_root: Path) -> None:
    """Move station artifact paths from a staging directory to the final directory."""
    old_root = old_root.expanduser().resolve()
    new_root = new_root.expanduser().resolve()

    def rebase_path(value):
        if value is None:
            return None
        try:
            target = Path(value).expanduser().resolve()
            rel = target.relative_to(old_root)
        except (OSError, ValueError):
            return value
        return new_root / rel

    for case in result.cases:
        if getattr(case, "fulllist", None):
            case.fulllist = rebase_path(case.fulllist)
            _rebase_station_fulllist_sim_dirs(case.fulllist, old_root, new_root)
        if getattr(case, "merged_dir", None):
            case.merged_dir = rebase_path(case.merged_dir)


def _rebase_station_fulllist_sim_dirs(fulllist: Path | None, old_root: Path, new_root: Path) -> None:
    if fulllist is None or not Path(fulllist).exists():
        return
    path = Path(fulllist)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames or "sim_dir" not in fieldnames:
        return
    changed = False
    for row in rows:
        raw = row.get("sim_dir")
        if not raw:
            continue
        try:
            target = Path(raw).expanduser().resolve()
            rel = target.relative_to(old_root)
        except (OSError, ValueError):
            continue
        row["sim_dir"] = str(new_root / rel)
        changed = True
    if not changed:
        return
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _resolve_registration_model_name(
    model_name: str,
    result,
    *,
    auto: bool,
    dry_run: bool,
) -> str:
    name = str(model_name).strip()
    if name and name.lower() != "auto":
        return name

    if not result.unresolved:
        raise click.ClickException(
            "--register-model needs a new model name. Re-run with "
            "--model NewModel --register-model, or omit --register-model for an existing model."
        )

    if auto or dry_run:
        raise click.ClickException(
            "Cannot register an inferred model profile without a name. "
            "Re-run with --model NewModel --register-model, then review mappings with "
            "`openbench model show NewModel` or update them with "
            "`openbench model register NewModel -v StdName:nc_var:unit`."
        )

    name = click.prompt("  New model profile name", type=str).strip()
    if not name or name.lower() == "auto":
        raise click.ClickException("New model profile name must not be empty or 'auto'.")
    from openbench.config.schema import is_simple_project_name

    if not is_simple_project_name(name):
        raise click.ClickException("New model profile name must be a simple name, not a path.")
    return name


_UNRESOLVED_REASONS: dict[str, tuple[str, str]] = {
    "model": (
        "model inference",
        "Specify --model MODEL for an existing profile, create a draft with "
        "--model NewModel --register-model, or re-run with --allow-unresolved. "
        "For exact mappings, use `openbench model register NewModel -v StdName:nc_var:unit`.",
    ),
    "multi_undated_files": (
        "multiple NetCDF files without a date token",
        "OpenBench's single-file lookup needs `prefix + suffix.nc` to be unique. "
        "Either rename the files to include a date stamp, split them into "
        "subdirectories, or filter via --case-pattern. The longest common "
        "prefix is reported in YAML so you can validate manually.",
    ),
    "variable_stream_inconsistent": (
        "variables that span multiple file streams with different tim_res / data_type / data_groupby",
        "Per-variable overrides have been written; review them in the report "
        "YAML. If the inconsistency is intentional, re-run with "
        "--allow-unresolved; otherwise consider splitting the case into "
        "homogeneous subdirectories or fixing the model profile.",
    ),
    "station_partial": (
        "station materialization dropped sites silently",
        "Inspect `station_dropped_sites` in the report YAML for the failed IDs. "
        "Re-run with --allow-partial-stations once you've confirmed the drops "
        "are expected, or fix the inputs (missing coordinates, unreadable NC).",
    ),
}


def _unresolved_message_for_cases(unresolved_cases) -> str:
    by_reason: dict[str, list[str]] = {}
    for case in unresolved_cases:
        reasons = case.unresolved or ["model"]
        for reason in reasons:
            by_reason.setdefault(reason, []).append(case.label)

    sections: list[str] = []
    for reason, labels in by_reason.items():
        title, hint = _UNRESOLVED_REASONS.get(
            reason,
            (f"unresolved ({reason})", "Re-run with --allow-unresolved to ignore."),
        )
        labels_str = ", ".join(sorted(set(labels)))
        sections.append(f"- {title}: {labels_str}\n  {hint}")
    return (
        "Simulation scan reported unresolved cases:\n"
        + "\n".join(sections)
        + "\nPass --allow-unresolved to publish anyway."
    )


def _ensure_model_profile_can_register(name: str, *, overwrite: bool = False) -> None:
    from openbench.data.registry.manager import RegistryManager

    existing = RegistryManager().get_model(name)
    if existing is not None and not overwrite:
        raise click.ClickException(f"Model profile already exists: {existing.name}. {_REGISTER_MODEL_EXISTS_HINT}")


def _register_model_profile_from_scan(name: str, profile: dict, *, overwrite: bool = False) -> Path:
    from openbench.cli.model import _write_model_profile_descriptor

    return _write_model_profile_descriptor(
        name,
        profile,
        overwrite=overwrite,
        exists_message=_REGISTER_MODEL_EXISTS_HINT,
    )


def _draft_model_profile_from_scan(name: str, result) -> dict:
    # Use the raw scan metadata before climatology normalization; the profile
    # describes model output frequency, not a per-case climatology interpretation.
    _warn_on_inconsistent_case_variables(result)
    profile = {
        "name": name,
        "description": (
            "Draft model profile generated by `openbench sim scan --register-model`. "
            "Review raw variable mappings before evaluation."
        ),
        "data_type": _required_common_case_value(result.cases, "data_type") or "grid",
        "tim_res": _required_common_case_value(result.cases, "tim_res") or "Month",
        "variables": _draft_model_variables(result),
    }
    grid_res = _required_common_case_value(result.cases, "grid_res")
    if grid_res is not None:
        profile["grid_res"] = grid_res
    return profile


def _warn_on_inconsistent_case_variables(result) -> None:
    """Warn when cases expose disjoint variable sets — a likely profile mistake."""
    case_vars: list[tuple[str, set[str]]] = []
    for case in result.cases:
        names: set[str] = set()
        for item in getattr(case, "variable_metadata", None) or []:
            raw = str(item.get("name", "")).strip()
            if raw:
                names.add(raw)
        for raw in getattr(case, "variables", None) or []:
            text = str(raw).strip()
            if text:
                names.add(text)
        case_vars.append((case.label, names))
    if len(case_vars) < 2:
        return
    union = set().union(*(names for _label, names in case_vars))
    missing_per_case = {label: union - names for label, names in case_vars if names != union}
    if not missing_per_case:
        return
    click.secho(
        "Warning: cases report different variable sets — draft model profile will "
        "include the union, which may map two raw names to one logical variable.",
        fg="yellow",
    )
    for label, missing in sorted(missing_per_case.items()):
        click.secho(
            f"  {label} is missing: {', '.join(sorted(missing))}",
            fg="yellow",
        )
    click.secho(
        "Consider re-running with --case-pattern to limit the scan, or use "
        "`openbench model register` to fine-tune mappings after writing the draft.",
        fg="yellow",
    )


def _draft_model_variables(result) -> dict[str, dict[str, str]]:
    variables: dict[str, dict[str, str]] = {}
    for case in result.cases:
        for item in getattr(case, "variable_metadata", None) or []:
            raw_name = str(item.get("name", "")).strip()
            if not raw_name or raw_name in variables:
                continue
            variables[raw_name] = {
                "varname": raw_name,
                "varunit": str(item.get("unit") or ""),
            }
        for raw in getattr(case, "variables", None) or []:
            raw_name = str(raw).strip()
            if raw_name and raw_name not in variables:
                variables[raw_name] = {"varname": raw_name, "varunit": ""}
    return dict(sorted(variables.items()))


def _common_case_value(cases, attr: str):
    values = [getattr(case, attr) for case in cases if getattr(case, attr, None) not in (None, "")]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values) else None


def _required_common_case_value(cases, attr: str):
    values = [getattr(case, attr) for case in cases if getattr(case, attr, None) not in (None, "")]
    unique = sorted({str(value) for value in values})
    if len(unique) > 1:
        raise click.ClickException(
            f"Cannot register one draft model profile from mixed {attr}: "
            f"{', '.join(unique)}. Split the scan into separate model profiles."
        )
    return values[0] if values else None


def _print_model_registration_preview(result, name: str) -> None:
    variable_count = len(_draft_model_variables(result))
    click.secho(
        f"[DRY RUN] Would register draft model profile '{name}' with {variable_count} raw variable(s).",
        fg="cyan",
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_output_paths(
    output: str | None,
    report: str | None,
    station_output: str | None,
) -> tuple[Path, Path, Path]:
    stamp = _timestamp()
    sim_path = expand_path(output) if output else Path(f"openbench_sim_scan_{stamp}.yaml")
    report_path = expand_path(report) if report else Path(f"openbench_sim_scan_report_{stamp}.yaml")
    if station_output:
        station_output_path = expand_path(station_output)
    elif output:
        station_output_path = sim_path.with_name(f"{sim_path.stem}_sim_station_lists")
    else:
        station_output_path = Path(f"openbench_sim_scan_station_lists_{stamp}")
    return sim_path, report_path, station_output_path


def _print_scan_summary(result, *, dry_run: bool) -> None:
    header = "[DRY RUN] " if dry_run else ""
    click.secho(f"{header}Found {len(result.cases)} simulation case(s):", bold=True)
    for case in result.cases:
        unresolved = " unresolved" if case.unresolved else ""
        res = f"{case.grid_res} deg" if case.grid_res is not None else "?"
        click.echo(
            f"  {case.label:<24} model={case.model:<16} "
            f"type={case.data_type or '?':<4} time={case.tim_res or '?':<6} "
            f"grid={res:<8} groupby={case.data_groupby or '?'}{unresolved}"
        )
    if result.unresolved:
        click.secho(
            f"Unresolved cases: {', '.join(case.label for case in result.unresolved)}",
            fg="yellow",
        )


def _handle_climatology_candidates(result, *, climatology: str, auto: bool, dry_run: bool) -> None:
    if str(climatology).lower() != "auto":
        return

    candidates = [case for case in result.cases if getattr(case, "temporal_kind_candidate", None)]
    if not candidates:
        return

    lines = [f"{case.label} -> {case.temporal_kind_candidate}" for case in candidates]
    message = (
        "Monthly climatology candidates require explicit confirmation: "
        f"{'; '.join(lines)}. Re-run with --climatology month to accept, "
        "or --climatology off to keep the NC time resolution."
    )

    if dry_run:
        click.secho(message, fg="yellow")
        return

    if auto:
        raise click.ClickException(message)

    for case in candidates:
        if click.confirm(
            f"Treat {case.label} as {case.temporal_kind_candidate}?",
            default=False,
        ):
            _apply_temporal_kind_candidate(case)
        else:
            case.provenance["temporal_kind"] = "user-declined"


def _apply_temporal_kind_candidate(case) -> None:
    candidate = case.temporal_kind_candidate
    if not candidate:
        return
    case.temporal_kind = candidate
    case.tim_res = candidate
    case.temporal_kind_candidate = None
    case.provenance["temporal_kind"] = "user-confirmed"
    case.provenance["tim_res"] = "climatology"


def _simulation_yaml(result, *, sim_path: Path | None = None) -> dict:
    entries = {case.label: _case_simulation_entry(case, sim_path=sim_path) for case in result.cases}
    defaults = _common_defaults(entries)
    if defaults:
        for entry in entries.values():
            for key in defaults:
                entry.pop(key, None)
        entries = {"_defaults": defaults, **entries}
    return {"simulation": entries}


def _portable_artifact_path(path: Path | None, sim_path: Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if sim_path is not None:
        try:
            base = Path(sim_path).expanduser().resolve().parent
            rel = Path(os.path.relpath(target.expanduser().resolve(), start=base)).as_posix()
            if not rel.startswith(".."):
                return rel
        except (OSError, ValueError):
            pass
    return str(target)


def _case_simulation_entry(case, *, sim_path: Path | None = None) -> dict:
    entry = {
        "model": case.model,
        "root_dir": str(case.root_dir),
    }
    if case.data_type:
        entry["data_type"] = case.data_type
    if case.tim_res:
        entry["tim_res"] = case.tim_res
    if case.grid_res is not None:
        entry["grid_res"] = case.grid_res
    if case.data_groupby:
        entry["data_groupby"] = case.data_groupby
    if case.data_type != "stn" and case.prefix and _case_prefix_is_safe_to_write(case):
        entry["prefix"] = case.prefix
    if case.data_type != "stn" and case.suffix and _case_prefix_is_safe_to_write(case):
        entry["suffix"] = case.suffix
    if getattr(case, "variable_overrides", None):
        entry["variables"] = case.variable_overrides
    if getattr(case, "fulllist", None):
        entry["fulllist"] = _portable_artifact_path(case.fulllist, sim_path)
    return entry


def _case_prefix_is_safe_to_write(case) -> bool:
    """Skip case-level prefix/suffix when overrides reveal multiple file streams.

    A case-level prefix only helps variables that lack a per-variable override
    *and* actually share that file pattern. When ``variable_overrides`` carries
    more than one distinct prefix/suffix the case is multi-stream; writing a
    case-level prefix would silently apply the alphabetically-first stream's
    pattern to every unmapped variable.
    """
    overrides = getattr(case, "variable_overrides", None) or {}
    if not overrides:
        return True
    case_prefix = case.prefix or ""
    case_suffix = case.suffix or ""
    seen_prefixes = {case_prefix}
    seen_suffixes = {case_suffix}
    for override in overrides.values():
        if not isinstance(override, dict):
            continue
        seen_prefixes.add(override.get("prefix", case_prefix) or "")
        seen_suffixes.add(override.get("suffix", case_suffix) or "")
    return len(seen_prefixes) <= 1 and len(seen_suffixes) <= 1


def _common_defaults(entries: dict[str, dict]) -> dict:
    if len(entries) < 2:
        return {}
    candidate_keys = ("model", "data_type", "tim_res", "grid_res", "data_groupby")
    defaults = {}
    for key in candidate_keys:
        values = [entry.get(key) for entry in entries.values()]
        if values and values[0] is not None and all(value == values[0] for value in values):
            defaults[key] = values[0]
    return defaults


def _report_yaml(result, *, case_depth: int) -> dict:
    return {
        "summary": {
            "roots": [str(root) for root in result.roots],
            "cases": len(result.cases),
            "unresolved": len(result.unresolved),
            "case_depth": case_depth,
        },
        "cases": [
            {
                "label": case.label,
                "root_dir": str(case.root_dir),
                "source_root": str(case.source_root) if case.source_root else None,
                "model": case.model,
                "data_type": case.data_type,
                "tim_res": case.tim_res,
                "grid_res": case.grid_res,
                "data_groupby": case.data_groupby,
                "fulllist": str(case.fulllist) if case.fulllist else None,
                "station_layout": case.station_layout,
                "station_count": case.station_count,
                "station_dropped_sites": case.station_dropped_sites,
                "merged_dir": str(case.merged_dir) if case.merged_dir else None,
                "temporal_kind": case.temporal_kind,
                "temporal_kind_candidate": case.temporal_kind_candidate,
                "years": case.years,
                "time_start": case.time_start,
                "time_end": case.time_end,
                "time_count": case.time_count,
                "time_span_days": case.time_span_days,
                "prefix": case.prefix,
                "suffix": case.suffix,
                "variables": case.variables,
                "variable_overrides": case.variable_overrides,
                "unresolved": case.unresolved,
                "provenance": case.provenance,
            }
            for case in result.cases
        ],
    }
