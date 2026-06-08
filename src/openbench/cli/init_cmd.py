"""openbench init command — interactive config generator."""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import click
import yaml

from openbench.core.registry import IMPLEMENTED_METRIC_NAMES, IMPLEMENTED_SCORE_NAMES

_SEED_MANIFEST_NAME = ".seeded_defaults.yaml"
_UNRESOLVED_ENV_RE = re.compile(r"(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%)")

DEFAULT_METRICS = ["bias", "RMSE", "correlation"]
DEFAULT_SCORES = ["Overall_Score"]
DEFAULT_COMPARISONS = ["Taylor_Diagram", "HeatMap"]
DEFAULT_STATISTICS = ["Mean", "Median", "Min", "Max", "Sum"]

METRIC_OPTIONS = list(IMPLEMENTED_METRIC_NAMES)

SCORE_OPTIONS = list(IMPLEMENTED_SCORE_NAMES)

COMPARISON_OPTIONS = [
    "Taylor_Diagram",
    "Target_Diagram",
    "Whisker_Plot",
    "Parallel_Coordinates",
    "Portrait_Plot_seasonal",
    "Ridgeline_Plot",
    "HeatMap",
    "Kernel_Density_Estimate",
    "Diff_Plot",
    "RadarMap",
    "Single_Model_Performance_Index",
    "Relative_Score",
    "Mann_Kendall_Trend_Test",
    "Correlation",
    "Standard_Deviation",
    "Functional_Response",
    "Mean",
    "Median",
    "Min",
    "Max",
    "Sum",
]

STATISTICS_OPTIONS = [
    "Mean",
    "Median",
    "Min",
    "Max",
    "Sum",
    "Standard_Deviation",
    "Mann_Kendall_Trend_Test",
    "Correlation",
    "Functional_Response",
    "Z_Score",
    "Hellinger_Distance",
    "Three_Cornered_Hat",
    "Partial_Least_Squares_Regression",
    "False_Discovery_Rate",
    "ANOVA",
]


@dataclass(frozen=True)
class ReferenceCatalogStatus:
    catalog_path: Path
    exists: bool
    empty: bool
    corrupted: bool = False


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_sha256(resource: Any) -> str:
    digest = sha256()
    digest.update(resource.read_bytes())
    return digest.hexdigest()


def _read_seed_manifest(base_dir: Path) -> dict:
    path = base_dir / _SEED_MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_seed_manifest(base_dir: Path, manifest: dict) -> None:
    if manifest:
        (base_dir / _SEED_MANIFEST_NAME).write_text(yaml.safe_dump(manifest, sort_keys=True))


def _copy_default_file_if_missing(
    source: Any,
    target: Path,
    base_dir: Path,
    manifest: dict,
) -> None:
    """Seed target from a bundled default file without overwriting user edits."""
    relative_target = target.relative_to(base_dir).as_posix()
    seed_info = manifest.get(relative_target, {})
    seeded_hash = seed_info.get("sha256") if isinstance(seed_info, dict) else None
    source_hash = _resource_sha256(source) if source.is_file() else None

    if target.exists():
        target_hash = _file_sha256(target)
        if source_hash is not None and target_hash == source_hash:
            manifest[relative_target] = {"sha256": target_hash}
            return
        if not seeded_hash or target_hash != seeded_hash:
            return

    if not source.is_file():
        manifest.pop(relative_target, None)
        return

    target.write_bytes(source.read_bytes())
    manifest[relative_target] = {"sha256": _file_sha256(target)}


def _ensure_empty_yaml_overlay(target: Path, base_dir: Path, manifest: dict) -> None:
    """Create an empty user overlay, replacing old untouched seeded copies."""
    relative_target = target.relative_to(base_dir).as_posix()
    seed_info = manifest.get(relative_target, {})
    seeded_hash = seed_info.get("sha256") if isinstance(seed_info, dict) else None
    seed_kind = seed_info.get("kind") if isinstance(seed_info, dict) else None

    if target.exists():
        if seed_kind != "empty-overlay" and seeded_hash and _file_sha256(target) == seeded_hash:
            target.write_text("{}\n")
            manifest[relative_target] = {
                "sha256": _file_sha256(target),
                "kind": "empty-overlay",
            }
        return

    target.write_text("{}\n")
    manifest[relative_target] = {
        "sha256": _file_sha256(target),
        "kind": "empty-overlay",
    }


def ensure_user_registry_overlays(user_dir: str | Path | None = None) -> Path:
    """Create the per-user registry overlay skeleton and seed custom filters."""
    base_dir = Path(user_dir) if user_dir is not None else Path.home() / ".openbench"
    references_dir = base_dir / "references"
    models_dir = base_dir / "models"
    custom_dir = base_dir / "custom"

    references_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    custom_dir.mkdir(parents=True, exist_ok=True)

    seed_manifest = _read_seed_manifest(base_dir)
    _ensure_empty_yaml_overlay(
        references_dir / "reference_catalog.yaml",
        base_dir,
        seed_manifest,
    )
    _ensure_empty_yaml_overlay(
        references_dir / "reference_profiles.yaml",
        base_dir,
        seed_manifest,
    )
    _ensure_empty_yaml_overlay(
        models_dir / "model_catalog.yaml",
        base_dir,
        seed_manifest,
    )
    custom_source_dir = files("openbench.data.custom")
    custom_filters = sorted(
        (child for child in custom_source_dir.iterdir() if child.name.endswith("_filter.py") and child.is_file()),
        key=lambda child: child.name,
    )
    for custom_filter in custom_filters:
        _copy_default_file_if_missing(
            custom_filter,
            custom_dir / custom_filter.name,
            base_dir,
            seed_manifest,
        )
    _write_seed_manifest(base_dir, seed_manifest)

    return base_dir


def _default_init_output_path() -> Path:
    return Path(f"openbench_init_{datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml")


def _expand_output_file_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if _UNRESOLVED_ENV_RE.search(expanded):
        raise click.ClickException(f"Output path contains unresolved environment variable: {value}")
    path = Path(expanded)
    if path.exists() and path.is_dir():
        raise click.ClickException(f"Output path must be a file, got directory: {path}")
    if path.exists() and not path.is_file():
        raise click.ClickException(f"Output path must be a regular file: {path}")
    return path


def _expand_project_output_dir(value: str | Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if _UNRESOLVED_ENV_RE.search(expanded):
        raise click.ClickException(f"project.output_dir contains unresolved environment variable: {value}")
    return expanded


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _default_min_year_threshold_for_span(syear: int, eyear: int) -> int:
    from openbench.config.schema import ProjectConfig

    default = ProjectConfig(name="_", output_dir=".", years=[syear, eyear]).min_year_threshold
    span_years = max(1, int(eyear) - int(syear) + 1)
    return min(default, span_years)


def _coerce_year_span(years) -> tuple[int, int] | None:
    if not years or not isinstance(years, (list, tuple)) or len(years) < 2:
        return None
    try:
        start = int(years[0])
        end = int(years[1])
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def _year_spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _format_year_span(span: tuple[int, int]) -> str:
    return f"{span[0]}-{span[1]}"


def _warn_reference_year_coverage(selected_refs: dict, project_years: tuple[int, int]) -> None:
    for variable, ref in selected_refs.items():
        ref_years = _coerce_year_span(getattr(ref, "years", None))
        if ref_years is None or _year_spans_overlap(project_years, ref_years):
            continue
        click.secho(
            "  Warning: reference "
            f"{getattr(ref, 'name', '<unknown>')} for {variable} covers "
            f"{_format_year_span(ref_years)}, outside project years "
            f"{_format_year_span(project_years)}.",
            fg="yellow",
        )


def _warn_simulation_case_year_coverage(cases: list, project_years: tuple[int, int]) -> None:
    for case in cases:
        case_years = _coerce_year_span(getattr(case, "years", None))
        if case_years is None or _year_spans_overlap(project_years, case_years):
            continue
        click.secho(
            "  Warning: simulation case "
            f"{getattr(case, 'label', '<unknown>')} covers "
            f"{_format_year_span(case_years)}, outside project years "
            f"{_format_year_span(project_years)}.",
            fg="yellow",
        )


def _expand_existing_directory(value: str | Path, label: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.exists() or not path.is_dir():
        raise click.ClickException(f"{label} does not exist: {path}")
    return path


def _reference_catalog_status(user_dir: str | Path | None = None) -> ReferenceCatalogStatus:
    base_dir = Path(user_dir) if user_dir is not None else Path.home() / ".openbench"
    catalog_path = base_dir / "references" / "reference_catalog.yaml"
    if not catalog_path.exists():
        return ReferenceCatalogStatus(catalog_path=catalog_path, exists=False, empty=True)

    try:
        content = yaml.safe_load(catalog_path.read_text()) or {}
    except yaml.YAMLError:
        return ReferenceCatalogStatus(
            catalog_path=catalog_path,
            exists=True,
            empty=False,
            corrupted=True,
        )

    return ReferenceCatalogStatus(
        catalog_path=catalog_path,
        exists=True,
        empty=not bool(content),
    )


def _resolve_reference_root(ref_root: str | Path | None = None) -> Path:
    from openbench.config.user_settings import resolve_reference_root

    if ref_root:
        return _expand_existing_directory(ref_root, "Reference data root")

    env_root = resolve_reference_root()
    default = None
    if env_root:
        expanded = Path(os.path.expandvars(os.path.expanduser(env_root)))
        if expanded.exists() and expanded.is_dir():
            default = str(expanded)
        else:
            click.secho(
                f"Configured reference root is not available: {expanded}",
                fg="yellow",
            )

    if default is None:
        local_reference = Path.cwd() / "Reference"
        if local_reference.exists() and local_reference.is_dir():
            default = str(local_reference)

    if default:
        value = click.prompt("  Reference data root", default=default)
    else:
        value = click.prompt("  Reference data root")
    return _expand_existing_directory(value, "Reference data root")


def _scan_reference_variants(ref_root: Path) -> list:
    from openbench.data.registry.scanner import scan_reference_directory

    def _progress(msg):
        click.echo(msg)

    def _scan_once():
        old_ref_root = os.environ.get("OPENBENCH_REF_ROOT")
        os.environ["OPENBENCH_REF_ROOT"] = str(ref_root.expanduser().resolve())
        click.secho(f"Scanning {ref_root} (dry run)...", bold=True)
        skipped_now = []
        try:
            groups_now = scan_reference_directory(
                ref_root,
                on_progress=_progress,
                on_skip=skipped_now.append,
            )
        finally:
            if old_ref_root is None:
                os.environ.pop("OPENBENCH_REF_ROOT", None)
            else:
                os.environ["OPENBENCH_REF_ROOT"] = old_ref_root
        return groups_now, skipped_now

    groups, skipped = _scan_once()

    while skipped:
        from openbench.cli.data import (
            _create_ignore_profiles_for_scan_skips,
            _create_profiles_for_scan_skips,
            _format_scan_skip_key,
            _print_scan_skip_report,
            _profile_rescue_supported,
            _prompt_scan_skip_action,
            _scan_skip_keys,
        )

        click.echo()
        _print_scan_skip_report(skipped)
        action = _prompt_scan_skip_action(
            len(skipped),
            can_profile=any(_profile_rescue_supported(item) for item in skipped),
        )
        if action == "s":
            click.secho("Continuing after skipping unsupported folders.", fg="yellow")
            break
        if action == "a":
            raise click.ClickException("Reference scan cancelled because unsupported folders were not skipped.")
        if action == "i":
            before_skip_keys = _scan_skip_keys(skipped)
            updated = _create_ignore_profiles_for_scan_skips(skipped)
            if updated == 0:
                raise click.ClickException("No ignore profiles were created from the skipped folders.")
            click.secho(f"Updated {updated} ignore profile(s). Rescanning...", fg="green")
        else:
            before_skip_keys = _scan_skip_keys(skipped)
            updated = _create_profiles_for_scan_skips(skipped, ref_root)
            if updated == 0:
                raise click.ClickException("No reference profiles were created from the skipped folders.")
            click.secho(f"Updated {updated} reference profile(s). Rescanning...", fg="green")

        click.echo()
        groups, skipped = _scan_once()
        if skipped and _scan_skip_keys(skipped) == before_skip_keys:
            unresolved = ", ".join(sorted(_format_scan_skip_key(key) for key in before_skip_keys))
            raise click.ClickException(f"Profile handling did not resolve unsupported folder(s): {unresolved}")

    variants = []
    for group in groups:
        variants.extend(variant for _, variant in sorted(group.variants.items()))

    click.echo()
    if not variants:
        click.secho("[DRY RUN] No reference datasets found.", fg="yellow")
        return variants

    click.secho(
        f"[DRY RUN] Found {len(variants)} dataset(s) to register/update. No catalog changes made yet.",
        fg="cyan",
        bold=True,
    )
    for variant in variants[:30]:
        click.echo(
            f"  {variant.registry_name:<35} {variant.data_type:<5} "
            f"{variant.category:<10} {len(variant.variables)} vars, "
            f"{variant.file_count} files"
        )
    if len(variants) > 30:
        click.echo(f"  ... {len(variants) - 30} more")
    click.echo()
    return variants


def _register_reference_variants(variants: list, catalog_path: Path, ref_root: Path | None = None) -> Path:
    from openbench.data.registry.manager import clear_registry_cache
    from openbench.data.registry.scanner import register_scanned_datasets_batch

    def _multi_var_handler(var_name, sub_dir, all_vars):
        if not all_vars:
            click.secho(f"  No selectable NC variables found for {sub_dir}/ ({var_name}); skipping.", fg="yellow")
            return None
        click.echo()
        click.secho(f"  Multiple variables in {sub_dir}/ (evaluating: {var_name}):", fg="yellow")
        for i, var in enumerate(all_vars, 1):
            desc = var.get("long_name") or var.get("standard_name") or ""
            suffix = f"  - {desc}" if desc else ""
            click.echo(f"    [{i}] {var['name']:<20} {var['unit']:<15} {var['dims']}{suffix}")
        choice = click.prompt("  Select variable number", type=int, default=1)
        if choice < 1 or choice > len(all_vars):
            raise click.ClickException(
                f"Variable choice out of range for {sub_dir}/ ({var_name}): {choice} (expected 1-{len(all_vars)})"
            )
        idx = choice - 1
        selected = all_vars[idx]["name"]
        click.echo(f"  Selected {selected} for {sub_dir}/ ({var_name})")
        return selected

    def _progress(msg):
        click.secho(f"  {msg.strip()}", fg="green")

    old_ref_root = os.environ.get("OPENBENCH_REF_ROOT")
    if ref_root is not None:
        os.environ["OPENBENCH_REF_ROOT"] = str(ref_root.expanduser().resolve())
    try:
        written_path = register_scanned_datasets_batch(
            variants,
            catalog_path=catalog_path,
            on_multi_var=_multi_var_handler,
            on_progress=_progress,
        )
    finally:
        if ref_root is not None:
            if old_ref_root is None:
                os.environ.pop("OPENBENCH_REF_ROOT", None)
            else:
                os.environ["OPENBENCH_REF_ROOT"] = old_ref_root
    clear_registry_cache()
    return written_path


def _init_reference_registry_preflight(
    status: ReferenceCatalogStatus,
    ref_root: str | Path | None = None,
    refresh_ref: bool = False,
) -> None:
    from openbench.config.user_settings import remember_reference_root

    if status.corrupted:
        raise click.ClickException(
            f"Reference catalog is corrupt: {status.catalog_path}. "
            "Restore it from a backup, delete it and rescan, or run "
            "`openbench ref scan ROOT --auto` after moving the broken file aside."
        )

    must_scan = (not status.exists) or status.empty
    if refresh_ref:
        if must_scan:
            click.echo("Skipping reference scan confirmation prompt because --refresh-ref was set.")
        else:
            click.echo("Refreshing reference catalog without prompt because --refresh-ref was set.")

    if must_scan:
        click.secho(
            "Reference catalog is missing or empty. A reference scan is required before init can continue.",
            fg="yellow",
        )
        root = _resolve_reference_root(ref_root)
        variants = _scan_reference_variants(root)
        if not variants:
            raise click.ClickException(
                "Reference scan found no datasets. Add reference data or pass --no-ref-check to skip."
            )
        if not refresh_ref and not click.confirm(
            f"Register/update {len(variants)} reference dataset(s) now?",
            default=True,
        ):
            raise click.ClickException("Reference scan is required before init can continue.")
        written_path = _register_reference_variants(variants, status.catalog_path, ref_root=root)
        settings_path = remember_reference_root(root)
        click.secho(f"Reference catalog updated: {written_path}", fg="green")
        click.echo(f"Saved reference root: {root} ({settings_path})")
        click.echo()
        return

    if not refresh_ref:
        if not click.confirm(
            "Reference catalog exists. Refresh it from the data root now?",
            default=False,
        ):
            return

    root = _resolve_reference_root(ref_root)
    variants = _scan_reference_variants(root)
    if not variants:
        click.secho("Reference scan found no datasets to register/update.", fg="yellow")
        settings_path = remember_reference_root(root)
        click.echo(f"Saved reference root: {root} ({settings_path})")
        click.echo()
        return
    if not refresh_ref and not click.confirm(
        f"Register/update {len(variants)} reference dataset(s) now?",
        default=True,
    ):
        click.echo("Reference catalog left unchanged.")
        click.echo()
        return
    written_path = _register_reference_variants(variants, status.catalog_path, ref_root=root)
    settings_path = remember_reference_root(root)
    click.secho(f"Reference catalog updated: {written_path}", fg="green")
    click.echo(f"Saved reference root: {root} ({settings_path})")
    click.echo()


def _parse_simulation_roots(value: str) -> list[str]:
    roots: list[str] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(text)))
        if not path.exists() or not path.is_dir():
            raise click.ClickException(f"Simulation data root does not exist: {path}")
        roots.append(str(path))
    return roots


def _parse_simulation_root_values(values: tuple[str, ...] | list[str]) -> list[str]:
    roots: list[str] = []
    for value in values:
        roots.extend(_parse_simulation_roots(value))
    return roots


def _scan_simulation_config(
    roots: list[str],
    *,
    model_name: str,
    output_path: str | Path,
    project_years: tuple[int, int] | None = None,
    case_depth: int,
    case_pattern: str | None,
    exclude: tuple[str, ...],
    climatology: str,
) -> dict:
    from openbench.cli.sim import (
        _handle_climatology_candidates,
        _print_scan_summary,
        _simulation_yaml,
    )
    from openbench.data.sim_scanner import materialize_station_cases, scan_simulation_roots

    def _run_scan(current_model_name: str):
        result = scan_simulation_roots(
            roots,
            model_name=current_model_name,
            case_depth=case_depth,
            case_pattern=case_pattern,
            exclude=exclude,
            climatology=climatology,
        )
        if not result.cases:
            raise click.ClickException("Simulation scan found no cases.")

        _handle_climatology_candidates(
            result,
            climatology=climatology,
            auto=False,
            dry_run=False,
        )
        _print_scan_summary(result, dry_run=False)
        return result

    result = _run_scan(model_name)

    if result.unresolved:
        labels = ", ".join(case.label for case in result.unresolved)
        if model_name == "auto":
            retry_model = click.prompt(
                "  Model profile for unresolved simulation case(s)",
                default="",
                show_default=False,
            ).strip()
            if retry_model:
                result = _run_scan(retry_model)
                if not result.unresolved:
                    model_name = retry_model
            labels = ", ".join(case.label for case in result.unresolved)
        if result.unresolved:
            hint = (
                "Pass --sim-model MODEL for an existing profile, or register the model first."
                if model_name != "auto"
                else "Register the model first or re-run with --sim-model MODEL."
            )
            raise click.ClickException(f"Simulation scan has unresolved model inference for: {labels}. {hint}")

    output = Path(output_path)
    station_output = output.with_name(f"{output.stem}_sim_station_lists")
    materialize_station_cases(result, station_output, allow_partial=False)
    partial_station_cases = [case for case in result.cases if case.station_dropped_sites]
    if partial_station_cases:
        labels = ", ".join(case.label for case in partial_station_cases)
        raise click.ClickException(
            f"Station materialization dropped sites for: {labels}. "
            "Run `openbench sim scan ... --allow-partial-stations` to inspect and publish "
            "a simulation fragment explicitly."
        )
    if project_years is not None:
        _warn_simulation_case_year_coverage(result.cases, project_years)
    return _simulation_yaml(result, sim_path=output)["simulation"]


def _prompt_manual_simulations(mgr) -> dict:
    models = mgr.list_models()
    known_model_names = {m.name.lower() for m in models}
    if models:
        click.echo("  Known models:")
        for i, m in enumerate(models, 1):
            click.echo(f"    [{i}] {m.name} ({len(m.variables)} variables)")

    simulation = {}
    while True:
        click.echo()
        model_input = click.prompt(
            "  Model name (or number from list, empty to finish)",
            default="",
        )
        if not model_input:
            break

        if model_input.isdigit() and models:
            midx = int(model_input) - 1
            if 0 <= midx < len(models):
                model_name = models[midx].name
            else:
                model_name = model_input
        else:
            model_name = model_input

        root_dir = click.prompt(f"  Data root directory for {model_name}")
        if known_model_names and model_name.lower() not in known_model_names:
            click.secho(
                f"  Warning: model is not registered: {model_name}",
                fg="yellow",
            )
        root_path = Path(os.path.expandvars(os.path.expanduser(root_dir)))
        if _UNRESOLVED_ENV_RE.search(str(root_path)):
            click.secho(
                f"  Warning: simulation root contains an unresolved environment variable: {root_dir}",
                fg="yellow",
            )
        elif not root_path.exists() or not root_path.is_dir():
            click.secho(
                f"  Warning: simulation root does not exist yet: {root_path}",
                fg="yellow",
            )
        label = click.prompt("  Label for this run", default=model_name)
        if label in simulation:
            click.secho(f"  Label already exists: {label}. Choose a different label.", fg="red")
            continue

        simulation[label] = {"model": model_name, "root_dir": root_dir}
        click.echo(f"  Added: {label}")

    if not simulation:
        click.secho("  No simulations added. Adding placeholder.", fg="yellow")
        simulation["MyModel"] = {"model": "MyModel", "root_dir": "/path/to/data"}

    return simulation


def _common_non_null_value(values) -> object | None:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    first = non_null[0]
    if all(value == first for value in non_null):
        return first
    return None


def _simulation_entries_with_defaults(simulation: dict) -> list[dict]:
    defaults = simulation.get("_defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    entries = []
    for label, entry in simulation.items():
        if label == "_defaults" or not isinstance(entry, dict):
            continue
        entries.append({**defaults, **entry})
    return entries


def _infer_project_resolution_fields(selected_refs: dict, simulation: dict) -> dict:
    sim_entries = _simulation_entries_with_defaults(simulation)
    sim_tim_res = _common_non_null_value(entry.get("tim_res") for entry in sim_entries)
    sim_grid_res = _common_non_null_value(entry.get("grid_res") for entry in sim_entries)
    ref_tim_res = _common_non_null_value(getattr(ref, "tim_res", None) for ref in selected_refs.values())
    ref_grid_res = _common_non_null_value(getattr(ref, "grid_res", None) for ref in selected_refs.values())

    fields = {}
    # Time resolution follows the simulation outputs when available because
    # they define the evaluation cadence. Spatial resolution follows selected
    # references first because target grids are usually chosen from reference
    # products; simulation resolution is only a fallback.
    tim_res = sim_tim_res if sim_tim_res is not None else ref_tim_res
    grid_res = ref_grid_res if ref_grid_res is not None else sim_grid_res
    if tim_res is not None:
        fields["tim_res"] = tim_res
    if grid_res is not None:
        fields["grid_res"] = grid_res
    return fields


def _unique_non_null_values(values) -> list:
    result = []
    for value in values:
        if value in (None, ""):
            continue
        if value in result:
            continue
        result.append(value)
    return result


def _prompt_missing_project_resolution_fields(project_resolution: dict, simulation: dict) -> dict:
    fields = dict(project_resolution)
    sim_entries = _simulation_entries_with_defaults(simulation)

    if "tim_res" not in fields:
        tim_values = _unique_non_null_values(entry.get("tim_res") for entry in sim_entries)
        if len(tim_values) > 1:
            fields["tim_res"] = click.prompt(
                "  Target tim_res for mixed simulation resolutions",
                default=str(tim_values[0]),
            )

    if "grid_res" not in fields:
        grid_values = _unique_non_null_values(entry.get("grid_res") for entry in sim_entries)
        if len(grid_values) > 1:
            fields["grid_res"] = click.prompt(
                "  Target grid_res for mixed simulation resolutions",
                type=float,
                default=float(grid_values[0]),
            )

    return fields


def _parse_variable_selection(selection: str, var_list: list[str]) -> list[str]:
    if selection.strip().lower() == "all":
        return list(var_list)

    selected = []
    invalid = []
    by_lower = {var.lower(): var for var in var_list}
    for token in selection.split(","):
        item = token.strip()
        if not item:
            continue
        if item.isdigit():
            index = int(item)
            if index < 1 or index > len(var_list):
                raise click.ClickException(f"Variable selection out of range: {item} (expected 1-{len(var_list)})")
            selected.append(var_list[index - 1])
            continue
        if item in var_list:
            selected.append(item)
            continue
        match = by_lower.get(item.lower())
        if match:
            selected.append(match)
            continue
        invalid.append(item)

    if invalid:
        raise click.ClickException(
            "Invalid variable selection: "
            + ", ".join(invalid)
            + ". Use comma-separated numbers, variable names, or 'all'."
        )
    if not selected:
        raise click.ClickException("No variables selected.")
    return _unique_preserving_order(selected)


def _parse_freeform_variable_selection(selection: str) -> list[str]:
    variables = [token.strip() for token in selection.split(",") if token.strip()]
    variables = _unique_preserving_order(variables)
    if not variables:
        raise click.ClickException("No variables selected.")
    return variables


def _format_reference_choice(ref) -> str:
    details = []
    data_type = getattr(ref, "data_type", None)
    tim_res = getattr(ref, "tim_res", None)
    if data_type:
        details.append(str(data_type))
    if tim_res:
        details.append(str(tim_res))
    grid_res = getattr(ref, "grid_res", None)
    if grid_res is not None:
        details.append(f"grid_res={grid_res}")
    years = getattr(ref, "years", None)
    if years and len(years) == 2:
        details.append(f"years={years[0]}-{years[1]}")
    category = getattr(ref, "category", None)
    if category:
        details.append(f"category={category}")
    root_dir = getattr(ref, "root_dir", None)
    if root_dir:
        details.append(str(root_dir))
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{ref.name}{suffix}"


def _parse_reference_selection(selection: str, available: list, variable: str):
    item = selection.strip()
    if not item:
        item = "1"
    if item == "0":
        return None
    if item.isdigit():
        choice = int(item)
        if choice < 1 or choice > len(available):
            raise click.ClickException(
                f"Reference choice out of range for {variable}: {choice} (expected 0-{len(available)})"
            )
        return available[choice - 1]

    for ref in available:
        if item == ref.name:
            return ref
    lowered = item.lower()
    for ref in available:
        if lowered == ref.name.lower():
            return ref
    raise click.ClickException(
        f"Invalid reference selection for {variable}: {item}. Use 0, a number, or a reference name."
    )


def _unique_preserving_order(values) -> list:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _yaml_inline(value) -> str:
    text = yaml.safe_dump(
        value,
        default_flow_style=True,
        sort_keys=False,
        allow_unicode=True,
    )
    lines = [line for line in text.strip().splitlines() if line != "..."]
    return " ".join(lines) if lines else "null"


def _yaml_key(value) -> str:
    text = str(value)
    if text and all(ch.isalnum() or ch in "_.-" for ch in text):
        return text
    return _yaml_inline(text)


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _append_key_value(lines: list[str], key: str, value, *, indent: int = 0, commented: bool = False) -> None:
    prefix = " " * indent + ("# " if commented else "")
    key_text = _yaml_key(key)

    if isinstance(value, dict):
        lines.append(f"{prefix}{key_text}:")
        for child_key, child_value in value.items():
            _append_key_value(
                lines,
                child_key,
                child_value,
                indent=indent + 2,
                commented=commented,
            )
        return

    if isinstance(value, list):
        if all(_is_scalar(item) for item in value):
            lines.append(f"{prefix}{key_text}: {_yaml_inline(value)}")
            return
        lines.append(f"{prefix}{key_text}:")
        item_prefix = " " * (indent + 2) + ("# " if commented else "")
        for item in value:
            if _is_scalar(item):
                lines.append(f"{item_prefix}- {_yaml_inline(item)}")
            else:
                dumped = yaml.safe_dump(item, sort_keys=False, allow_unicode=True).rstrip()
                item_lines = dumped.splitlines()
                if item_lines:
                    lines.append(f"{item_prefix}- {item_lines[0]}")
                    for extra in item_lines[1:]:
                        lines.append(f"{' ' * (indent + 4)}{'# ' if commented else ''}{extra}")
        return

    lines.append(f"{prefix}{key_text}: {_yaml_inline(value)}")


def _append_sequence_with_commented_options(
    lines: list[str],
    key: str,
    active_items: list[str],
    all_items: list[str],
    *,
    indent: int = 0,
) -> None:
    lines.append(f"{' ' * indent}{_yaml_key(key)}:")
    active = _unique_preserving_order(active_items)
    active_set = set(active)
    for item in active:
        lines.append(f"{' ' * (indent + 2)}- {_yaml_inline(item)}")
    for item in _unique_preserving_order(all_items):
        if item not in active_set:
            lines.append(f"{' ' * (indent + 2)}# - {_yaml_inline(item)}")


def _reference_candidates_by_variable(all_refs: list) -> dict[str, list]:
    candidates: dict[str, list] = {}
    seen: set[tuple[str, str]] = set()
    for ref in all_refs:
        for variable in getattr(ref, "variables", {}) or {}:
            key = (variable, ref.name)
            if key in seen:
                continue
            seen.add(key)
            candidates.setdefault(variable, []).append(ref)
    return candidates


def _render_project_section(lines: list[str], project: dict) -> None:
    lines.append("project:")
    active_order = [
        "name",
        "output_dir",
        "years",
        "min_year_threshold",
        "tim_res",
        "grid_res",
    ]
    for key in active_order:
        if key in project:
            _append_key_value(lines, key, project[key], indent=2)

    lines.append("  # Optional project controls:")
    examples = {
        "lat_range": [-90.0, 90.0],
        "lon_range": [-180.0, 180.0],
        "timezone": 0,
        "weight": "area",
        "num_cores": 4,
        "time_alignment": "intersection",
        "unified_mask": True,
        "generate_report": True,
        "IGBP_groupby": False,
        "PFT_groupby": False,
        "climate_zone_groupby": False,
        "debug_mode": False,
        "only_drawing": False,
        "force": False,
        "strict_reference": False,
    }
    if "tim_res" not in project:
        examples = {"tim_res": "Month", **examples}
    if "grid_res" not in project:
        examples = {"grid_res": 0.5, **examples}
    for key, value in examples.items():
        if key not in project:
            _append_key_value(lines, key, value, indent=2, commented=True)


def _render_evaluation_section(lines: list[str], selected_vars: list[str], all_vars: list[str]) -> None:
    lines.append("")
    lines.append("evaluation:")
    _append_sequence_with_commented_options(
        lines,
        "variables",
        selected_vars,
        all_vars,
        indent=2,
    )


def _render_reference_section(lines: list[str], reference: dict, all_refs: list, all_vars: list[str]) -> None:
    candidates = _reference_candidates_by_variable(all_refs)
    lines.append("")
    if not reference:
        lines.append("reference: {}")
        lines.append("# data_root: /path/to/reference/root")
        lines.append("# Variable -> reference dataset. Uncomment and fill one source per evaluation variable.")
        variables = _unique_preserving_order(all_vars)
        for variable in variables:
            lines.append(f"# {_yaml_key(variable)}: <reference_dataset>")
        return

    lines.append("reference:")
    lines.append("  # data_root: /path/to/reference/root")
    lines.append("  # Variable -> reference dataset. Uncomment an alternative source to switch.")

    variables = _unique_preserving_order([*reference.keys(), *all_vars])
    for variable in variables:
        active_source = reference.get(variable)
        if active_source:
            _append_key_value(lines, variable, active_source, indent=2)
        elif not candidates.get(variable):
            lines.append(f"  # {_yaml_key(variable)}: <reference_dataset>")
        for ref in candidates.get(variable, []):
            if ref.name == active_source:
                continue
            details = ", ".join(
                str(value)
                for value in (
                    getattr(ref, "data_type", None),
                    getattr(ref, "tim_res", None),
                    getattr(ref, "grid_res", None),
                )
                if value is not None
            )
            suffix = f"  # {details}" if details else ""
            lines.append(f"  # {_yaml_key(variable)}: {_yaml_inline(ref.name)}{suffix}")


def _render_simulation_section(lines: list[str], simulation: dict) -> None:
    lines.append("")
    lines.append("simulation:")
    optional_examples = {
        "data_type": "grid",
        "grid_res": 0.5,
        "tim_res": "Month",
        "data_groupby": "Month",
        "prefix": "Case01_hist_",
        "suffix": ".nc",
        "fulllist": "/path/to/station_list.csv",
    }

    defaults = simulation.get("_defaults", {})
    for label, entry in simulation.items():
        if not isinstance(entry, dict):
            continue
        lines.append(f"  {_yaml_key(label)}:")
        for key, value in entry.items():
            _append_key_value(lines, key, value, indent=4)

        if label == "_defaults":
            continue

        merged = {**defaults, **entry} if isinstance(defaults, dict) else dict(entry)
        lines.append("    # Optional per-case overrides:")
        for key, value in optional_examples.items():
            if key not in merged:
                _append_key_value(lines, key, value, indent=4, commented=True)
        if "variables" not in merged:
            lines.append("    # variables:")
            lines.append("    #   Latent_Heat:")
            lines.append("    #     varname: f_lfevpa")
            lines.append("    #     varunit: W m-2")


def _render_metric_score_option_sections(lines: list[str], config: dict) -> None:
    lines.append("")
    _append_sequence_with_commented_options(
        lines,
        "metrics",
        config.get("metrics", DEFAULT_METRICS),
        METRIC_OPTIONS,
    )

    lines.append("")
    _append_sequence_with_commented_options(
        lines,
        "scores",
        config.get("scores", DEFAULT_SCORES),
        SCORE_OPTIONS,
    )

    lines.append("")
    lines.append("comparison:")
    _append_key_value(lines, "enabled", config["comparison"]["enabled"], indent=2)
    if config["comparison"].get("items"):
        _append_sequence_with_commented_options(
            lines,
            "items",
            config["comparison"]["items"],
            COMPARISON_OPTIONS,
            indent=2,
        )
    else:
        lines.append("  # items:")
        for item in COMPARISON_OPTIONS:
            lines.append(f"  #   - {_yaml_inline(item)}")

    lines.append("")
    lines.append("statistics:")
    _append_key_value(lines, "enabled", config["statistics"]["enabled"], indent=2)
    if config["statistics"].get("items"):
        _append_sequence_with_commented_options(
            lines,
            "items",
            config["statistics"]["items"],
            STATISTICS_OPTIONS,
            indent=2,
        )
    else:
        lines.append("  # items:")
        for item in STATISTICS_OPTIONS:
            lines.append(f"  #   - {_yaml_inline(item)}")


def _render_init_config_template(config: dict, *, all_refs: list, all_vars: list[str]) -> str:
    lines = [
        "# OpenBench configuration generated by `openbench init`.",
        "# Active lines are used by OpenBench; commented lines are editable alternatives.",
    ]
    _render_project_section(lines, config["project"])
    _render_evaluation_section(lines, config["evaluation"]["variables"], all_vars)
    _render_reference_section(lines, config["reference"], all_refs, all_vars)
    _render_simulation_section(lines, config["simulation"])
    _render_metric_score_option_sections(lines, config)
    return "\n".join(lines) + "\n"


@click.command("init")
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output file path. Default: openbench_init_YYYYmmdd-HHMMSS.yaml.",
)
@click.option(
    "--ref-root",
    type=str,
    default=None,
    help="Reference data root for the init scan/update preflight.",
)
@click.option(
    "--refresh-ref",
    is_flag=True,
    help="Refresh the reference catalog during init without asking first.",
)
@click.option(
    "--no-ref-check",
    is_flag=True,
    help="Skip the reference catalog scan/update preflight.",
)
@click.option(
    "--sim-root",
    "sim_roots",
    multiple=True,
    type=str,
    help="Simulation data root to scan during init. Repeat for multiple roots.",
)
@click.option("--sim-model", default="auto", show_default=True, help="Simulation model profile name, or 'auto'.")
@click.option("--sim-case-depth", type=click.IntRange(0, 10), default=5, show_default=True)
@click.option("--sim-case-pattern", default=None, help="Only include simulation case labels matching this glob.")
@click.option("--sim-exclude", multiple=True, help="Simulation directory name or glob to exclude. Repeatable.")
@click.option(
    "--sim-climatology",
    type=click.Choice(["auto", "off", "year", "month"]),
    default="auto",
    show_default=True,
    help="Detect or force simulation climatology tim_res handling.",
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="[NOT IMPLEMENTED] Generate without prompts.",
    hidden=True,
)
def init_cmd(
    output,
    ref_root,
    refresh_ref,
    no_ref_check,
    sim_roots,
    sim_model,
    sim_case_depth,
    sim_case_pattern,
    sim_exclude,
    sim_climatology,
    no_interactive,
):
    """Interactively generate an openbench.yaml config file."""
    from openbench.data.registry import RegistryManager

    if no_interactive:
        raise click.ClickException(
            "init --no-interactive is not implemented yet. "
            "Use explicit scan/register commands for non-interactive setup."
        )

    if output is None:
        output = _default_init_output_path()
    output = _expand_output_file_path(output)
    if output.exists() and not click.confirm(f"Output file already exists: {output}. Overwrite?", default=False):
        raise click.ClickException(f"Init cancelled by user; output file was not overwritten: {output}")

    ensure_user_registry_overlays()
    reference_status = _reference_catalog_status()
    if not no_ref_check:
        _init_reference_registry_preflight(
            reference_status,
            ref_root=ref_root,
            refresh_ref=refresh_ref,
        )
    mgr = RegistryManager()

    click.secho("OpenBench Configuration Wizard", bold=True)
    click.echo()

    # Project settings
    click.secho("1. Project Settings", bold=True)
    name = click.prompt("  Project name", default="my-evaluation")
    from openbench.config.schema import is_simple_project_name

    if not is_simple_project_name(name):
        raise click.ClickException("project.name must be a simple directory name, not a path.")
    output_dir = _expand_project_output_dir(click.prompt("  Output directory", default="./output"))
    syear = click.prompt("  Start year", type=int, default=2004)
    eyear = click.prompt("  End year", type=int, default=2010)
    if eyear < syear:
        raise click.ClickException(f"Start year must be <= end year (got {syear} > {eyear}).")

    # Variable selection
    click.echo()
    click.secho("2. Evaluation Variables", bold=True)
    click.echo("  Available variables (by category):")

    all_refs = mgr.list_references()

    # Group by category
    categories = {}
    for ref in all_refs:
        cat = ref.category or "Other"
        if cat not in categories:
            categories[cat] = set()
        categories[cat].update(ref.variables.keys())

    var_list = []
    seen_vars = set()
    idx = 1
    for cat in sorted(categories.keys()):
        category_vars = []
        for var in sorted(categories[cat]):
            if var in seen_vars:
                continue
            seen_vars.add(var)
            category_vars.append(var)
        if not category_vars:
            continue
        click.echo(f"  {cat}:")
        for var in category_vars:
            click.echo(f"    [{idx}] {var}")
            var_list.append(var)
            idx += 1

    template_reference_mode = False
    if not var_list:
        if not no_ref_check:
            raise click.ClickException(
                "Reference registry has no variables. Run `openbench ref scan ROOT --auto` first."
            )
        template_reference_mode = True
        click.secho(
            "  Reference registry has no variables. Continuing in template mode because --no-ref-check was set.",
            fg="yellow",
        )
        selection = click.prompt(
            "  Enter evaluation variables (comma-separated names)",
            default="",
            show_default=False,
        )
        selected_vars = _parse_freeform_variable_selection(selection)
        var_list = list(selected_vars)
    else:
        click.echo()
        selection = click.prompt(
            "  Select variables (comma-separated session-local numbers, names, or 'all')",
            default="all",
        )
        selected_vars = _parse_variable_selection(selection, list(var_list))

    # Reference selection
    click.echo()
    click.secho("3. Reference Data Sources", bold=True)
    reference = {}
    selected_reference_objects = {}
    if template_reference_mode:
        click.secho(
            "  No reference registry entries are available; generated YAML will contain placeholders.",
            fg="yellow",
        )
    else:
        for var in list(selected_vars):
            available = mgr.references_for_variable(var)
            if len(available) == 1:
                chosen = available[0]
                reference[var] = chosen.name
                selected_reference_objects[var] = chosen
                click.echo(f"  {var} -> {chosen.name} (only option)")
            elif len(available) > 1:
                click.echo(f"  {var} - available sources:")
                click.echo("    [0] skip this variable")
                for i, ref in enumerate(available, 1):
                    click.echo(f"    [{i}] {_format_reference_choice(ref)}")
                choice = click.prompt(f"  Select for {var}", default="1")
                chosen = _parse_reference_selection(choice, available, var)
                if chosen is None:
                    selected_vars.remove(var)
                    click.secho(f"  {var} skipped", fg="yellow")
                    continue
                reference[var] = chosen.name
                selected_reference_objects[var] = chosen
            else:
                click.secho(f"  {var} - no reference data available, skipping", fg="yellow")
                selected_vars.remove(var)
        if not selected_vars:
            raise click.ClickException("No reference data selected for evaluation variables.")
        _warn_reference_year_coverage(selected_reference_objects, (syear, eyear))

    # Simulation selection
    click.echo()
    click.secho("4. Simulation Data", bold=True)
    roots = _parse_simulation_root_values(list(sim_roots))
    if not roots:
        root_input = click.prompt(
            "  Simulation data root(s) to scan (comma-separated, empty for manual entry)",
            default="",
            show_default=False,
        )
        roots = _parse_simulation_roots(root_input)

    if roots:
        simulation = _scan_simulation_config(
            roots,
            model_name=sim_model,
            output_path=output,
            project_years=(syear, eyear),
            case_depth=sim_case_depth,
            case_pattern=sim_case_pattern,
            exclude=tuple(sim_exclude),
            climatology=sim_climatology,
        )
    else:
        simulation = _prompt_manual_simulations(mgr)

    # Options
    click.echo()
    click.secho("5. Options", bold=True)
    comparison = click.confirm("  Enable comparison?", default=True)
    statistics = click.confirm("  Enable statistics?", default=False)

    project_resolution = _infer_project_resolution_fields(
        selected_reference_objects,
        simulation,
    )
    project_resolution = _prompt_missing_project_resolution_fields(
        project_resolution,
        simulation,
    )
    if project_resolution:
        click.echo(
            "  Inferred target resolution: " + ", ".join(f"{key}={value}" for key, value in project_resolution.items())
        )

    # Build config
    # NOTE: reference uses flat var→source mapping (matches loader._build_reference);
    # not {"sources": {...}} which loader rejects as "reference.sources must be a string".
    project = {
        "name": name,
        "output_dir": output_dir,
        "years": [syear, eyear],
        "min_year_threshold": _default_min_year_threshold_for_span(syear, eyear),
        **project_resolution,
    }
    config = {
        "project": project,
        "evaluation": {"variables": selected_vars},
        "reference": reference,
        "simulation": simulation,
        "metrics": DEFAULT_METRICS.copy(),
        "scores": DEFAULT_SCORES.copy(),
        "comparison": {
            "enabled": comparison,
            "items": DEFAULT_COMPARISONS.copy() if comparison else None,
        },
        "statistics": {
            "enabled": statistics,
            "items": DEFAULT_STATISTICS.copy() if statistics else None,
        },
    }

    # Write
    rendered = _render_init_config_template(
        config,
        all_refs=all_refs,
        all_vars=_unique_preserving_order(var_list),
    )
    _atomic_text_write(output, rendered)

    click.echo()
    click.secho(f"Generated {output}", fg="green", bold=True)
    if template_reference_mode:
        click.echo(f"  Next: Fill in reference placeholders in {output}, then run openbench check.")
    else:
        click.echo(f"  Next: openbench check {output}")
