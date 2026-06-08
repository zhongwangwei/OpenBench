"""openbench ref commands."""

from pathlib import Path

import click

from openbench.cli import _display, _optimize, _profile_rescue, _ref_commands, _register, _scan, _scan_support
from openbench.cli._options import TIM_RES_TYPE, expand_existing_directory, expand_path

DATA_GROUPBY_TYPE = click.Choice(
    ["single", "Year", "Day", "Month"],
    case_sensitive=False,
)


def _expand_existing_directory(value: str | Path, label: str) -> str:
    return str(expand_existing_directory(value, label))


def _normalize_fulllist_path(value: str | Path, root_dir: str | Path | None) -> str:
    path = expand_path(value)
    if path.is_absolute():
        return str(path)
    if root_dir:
        return str((Path(root_dir) / path).resolve())
    return str(path.resolve())


def _load_catalog_for_cli(path: Path) -> dict:
    from openbench.data.registry.scanner import _safe_load_catalog

    try:
        return _safe_load_catalog(path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


_reference_to_dict = _ref_commands.reference_to_dict


@click.group()
def data():
    """Manage reference datasets."""


@data.command("list")
@click.option("--variable", default=None, help="Filter by variable name.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def list_datasets(variable, fmt):
    """List all available reference datasets."""
    return _ref_commands.list_datasets(variable, fmt, reference_to_dict_fn=_reference_to_dict)


@data.command(hidden=True)
@click.argument("names", nargs=-1, required=True)
def download(names):
    """[NOT IMPLEMENTED] Download reference datasets by name (planned for v3.0)."""
    return _ref_commands.download(names)


@data.command(hidden=True)
def status():
    """Show local dataset cache status. [Cache reporting NOT IMPLEMENTED in v3.0a1; only registry count is shown.]"""
    return _ref_commands.status()


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset.  Supports base names (e.g. ERA5LAND)."""
    return _ref_commands.path(name)


@data.command("convert-old")
@click.argument("old_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--name", required=True, help="Dataset name for the converted registry entry.")
@click.option("--category", default="", help="Dataset category for the converted registry entry.")
@click.option("--description", default="", help="Description for the converted registry entry.")
def convert_old(old_path, output_path, name, category, description):
    """Convert an old-format reference YAML into a v3 registry descriptor."""
    return _ref_commands.convert_old(old_path, output_path, name, category, description)


@data.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Delete without confirmation.")
def delete(name, yes):
    """Delete a user reference dataset entry or overlay."""
    return _ref_commands.delete_reference(name, yes, load_catalog_for_cli_fn=_load_catalog_for_cli)


@data.command()
@click.argument("name")
@click.option(
    "--root-dir",
    default=None,
    help="Root directory containing data files (required for new).",
)
@click.option(
    "--data-type",
    type=click.Choice(["grid", "stn"]),
    default=None,
    help="Data type (auto-detected from NC if omitted).",
)
@click.option("--tim-res", type=TIM_RES_TYPE, default=None)
@click.option("--grid-res", type=float, default=None, help="Grid resolution in degrees.")
@click.option("--category", default="Other", help="Category: Water, Carbon, Energy, etc.")
@click.option("--years", nargs=2, type=int, default=None, help="Start and end year.")
@click.option("--fulllist", default=None, type=click.Path(), help="Station list CSV path (for stn data).")
@click.option(
    "-v",
    "--variable",
    multiple=True,
    help="'StdName:ncname:unit[:prefix[:suffix]]' (repeatable). Overwrites if exists.",
)
@click.option("-f", "--fallback", multiple=True, help="'StdName:fallback_name:fallback_unit:conversion' (repeatable).")
def register(name, root_dir, data_type, tim_res, grid_res, category, years, fulllist, variable, fallback):
    """Register or update a reference dataset in the registry.

    Creates a new entry or updates an existing one. Variables are overwritten
    by default when names match.


Examples:
openbench ref register MyData --root-dir /data/myref \
  --data-type grid --grid-res 0.5 --tim-res Month \
  --years 2000 2020 --category Water \
  -v "Evapotranspiration:ET:mm day-1"


openbench ref register ERA5 --root-dir /data/era5 \
  -v "Latent_Heat:slhf:W m-2" \
  -f "Latent_Heat:surface_latent_heat_flux:J m-2:value / 3600"


openbench ref register PLUMBER2 --root-dir /data/PLUMBER2/dataset \
  --data-type stn --tim-res Day \
  --fulllist /data/PLUMBER2/list/PLUMBER2.csv \
  -v "Latent_Heat:Qle_cor:W/m2" \
  -v "Sensible_Heat:Qh_cor:W/m2"


openbench ref register MyData -v "Runoff:RNOF:mm day-1"
    """
    return _register.register_reference(
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
        expand_existing_directory_fn=_expand_existing_directory,
        normalize_fulllist_path_fn=_normalize_fulllist_path,
        load_catalog_for_cli_fn=_load_catalog_for_cli,
    )


@data.command("register-profile")
@click.argument("name")
@click.option("-v", "--variable", multiple=True, help="'StdName:ncname:unit[:prefix[:suffix]]' (repeatable).")
@click.option("-f", "--fallback", multiple=True, help="'StdName:fallback_name:fallback_unit:conversion' (repeatable).")
@click.option("--tim-res", type=TIM_RES_TYPE, default=None, help="Time resolution override.")
@click.option("--category", default=None, help="Category override: Water, Carbon, Energy, etc.")
@click.option("--data-groupby", type=DATA_GROUPBY_TYPE, default=None, help="Data groupby override.")
@click.option("--fulllist", default=None, help="Station list CSV pattern (relative to root_dir).")
@click.option("--description", default=None, help="Dataset description.")
def register_profile(
    name,
    variable,
    fallback=(),
    tim_res=None,
    category=None,
    data_groupby=None,
    fulllist=None,
    description=None,
):
    """Register a reference profile (variable mappings for a dataset type).

    Profiles are shared across resolutions. E.g., registering 'GLEAM_v4.2a'
    applies to GLEAM_v4.2a_LowRes, GLEAM_v4.2a_MidRes, etc.


Examples:
openbench ref register-profile MyData \
  -v "Latent_Heat:LE:W m-2" \
  -v "Sensible_Heat:H:W m-2"


openbench ref register-profile PLUMBER2_new \
  --tim-res Day --data-groupby single \
  --fulllist "../list/stations.csv" \
  -v "Latent_Heat:Qle_cor:W m-2"
    """
    return _register.register_reference_profile(
        name,
        variable,
        fallback=fallback,
        tim_res=tim_res,
        category=category,
        data_groupby=data_groupby,
        fulllist=fulllist,
        description=description,
        load_catalog_for_cli_fn=_load_catalog_for_cli,
    )


@data.command()
@click.argument("name")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "yaml"]), default="text")
def show(name, fmt):
    """Show details of a dataset. Supports base name to show all resolutions.

    Examples:
        openbench ref show GLEAM_v4.2a          # shows all resolutions
        openbench ref show GLEAM_v4.2a_LowRes   # shows specific one
    """
    return _display.show_reference(name, fmt, reference_to_dict_fn=_reference_to_dict)


@data.command()
@click.argument("ref_root")
@click.option(
    "--auto",
    "--yes",
    "-y",
    "auto",
    is_flag=True,
    help="Register all found datasets without prompting; fail on ambiguous NC variables.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be registered without writing the catalog.")
@click.option("--rescan", is_flag=True, help="Refresh already registered scanned datasets too.")
@click.option(
    "--only",
    "only_pattern",
    default=None,
    help="Only register/rescan dataset names matching this shell pattern.",
)
@click.option(
    "--allow-skip",
    is_flag=True,
    help="Continue when scanner finds unsupported folders that cannot be auto-registered.",
)
@click.option(
    "--pick-first",
    is_flag=True,
    help="With --auto, silently pick the first variable when an NC file is ambiguous "
    "(unsafe; matches pre-3.x behavior).",
)
def scan(ref_root, auto, dry_run, rescan=False, only_pattern=None, allow_skip=False, pick_first=False):
    """Scan a directory for reference datasets and register new ones.

        REF_ROOT is the reference data root (e.g., /Volumes/work/Reference).
        Expected structure: Grid/{LowRes,MidRes,HigRes}/<category>/<variable>/<dataset>/

    
    Examples:
    openbench ref scan /Volumes/work/Reference

    
    openbench ref scan /Volumes/work/Reference --dry-run

    
    openbench ref scan /Volumes/work/Reference --rescan --auto
    """
    return _scan.run_scan(
        ref_root,
        auto,
        dry_run,
        rescan=rescan,
        only_pattern=only_pattern,
        allow_skip=allow_skip,
        pick_first=pick_first,
        expand_existing_directory_fn=_expand_existing_directory,
        filter_scan_groups_fn=_filter_scan_groups,
        filter_scan_skips_fn=_filter_scan_skips,
        print_scan_skip_report_fn=_print_scan_skip_report,
        print_profile_rescue_preview_fn=_print_profile_rescue_preview,
        prompt_scan_skip_action_fn=_prompt_scan_skip_action,
        profile_rescue_supported_fn=_profile_rescue_supported,
        create_ignore_profiles_for_scan_skips_fn=_create_ignore_profiles_for_scan_skips,
        create_profiles_for_scan_skips_fn=_create_profiles_for_scan_skips,
        scan_skip_keys_fn=_scan_skip_keys,
        format_scan_skip_key_fn=_format_scan_skip_key,
    )


# Compatibility names imported by init_cmd/tests and passed into _scan.run_scan.
_print_scan_skip_report = _scan_support.print_scan_skip_report
_scan_skip_keys = _scan_support.scan_skip_keys
_filter_scan_groups = _scan_support.filter_scan_groups
_filter_scan_skips = _scan_support.filter_scan_skips
_format_scan_skip_key = _scan_support.format_scan_skip_key
_create_profiles_for_scan_skips = _scan_support.create_profiles_for_scan_skips
_create_ignore_profiles_for_scan_skips = _scan_support.create_ignore_profiles_for_scan_skips

_print_profile_rescue_preview = _profile_rescue._print_profile_rescue_preview
_prompt_scan_skip_action = _profile_rescue._prompt_scan_skip_action
_profile_rescue_supported = _profile_rescue._profile_rescue_supported
_prompt_reference_profile_for_scan_skip = _profile_rescue._prompt_reference_profile_for_scan_skip
_prompt_grid_composite_profile = _profile_rescue._prompt_grid_composite_profile
_prompt_grid_dataset_choice_profile = _profile_rescue._prompt_grid_dataset_choice_profile
_prompt_station_direct_profile = _profile_rescue._prompt_station_direct_profile
_prompt_grid_nested_profile = _profile_rescue._prompt_grid_nested_profile
_prompt_profile_variables_for_child = _profile_rescue._prompt_profile_variables_for_child
_profile_variable_entry = _profile_rescue._profile_variable_entry
_prompt_standard_variable_name = _profile_rescue._prompt_standard_variable_name
_prompt_nc_variable_name = _profile_rescue._prompt_nc_variable_name
_resolve_nc_variable_choice = _profile_rescue._resolve_nc_variable_choice
_next_nc_default = _profile_rescue._next_nc_default
_prompt_file_glob = _profile_rescue._prompt_file_glob
_is_grid_composite_root_skip = _profile_rescue._is_grid_composite_root_skip
_is_grid_composite_skip = _profile_rescue._is_grid_composite_skip
_is_grid_dataset_skip = _profile_rescue._is_grid_dataset_skip
_is_station_dataset_skip = _profile_rescue._is_station_dataset_skip
_default_profile_name_for_skip = _profile_rescue._default_profile_name_for_skip
_grid_dataset_name_for_skip = _profile_rescue._grid_dataset_name_for_skip
_station_standard_variable_default = _profile_rescue._station_standard_variable_default
_ignore_profile_name = _profile_rescue._ignore_profile_name
_sanitize_profile_name = _profile_rescue._sanitize_profile_name
_inspect_first_nc_under = _profile_rescue._inspect_first_nc_under
_profile_child_specs = _profile_rescue._profile_child_specs
_ref_relative_path = _profile_rescue._ref_relative_path
_write_reference_profile = _profile_rescue._write_reference_profile
_write_reference_profiles = _profile_rescue._write_reference_profiles


@data.command()
@click.argument("name")
def optimize(name):
    """Convert dataset to zarr for faster reads."""
    return _optimize.optimize_reference(name)


@data.command("generate-station-list")
@click.argument("dataset_dir")
@click.option("-o", "--output", default=None, help="Output CSV path. Default: dataset_dir/station_list.csv")
def generate_station_list(dataset_dir, output):
    """Auto-generate a station list CSV from NC files.

    Scans NC files in DATASET_DIR, extracts station ID, lat, lon,
    time range, and writes a fulllist CSV.

    Supports:
      - One-file-per-station (e.g., PLUMBER2: 90 NC files)
      - Single merged file (e.g., GRDC: 1 NC with station dimension)

    \b
    Example:
      openbench ref generate-station-list /data/PLUMBER2/dataset/
      openbench ref generate-station-list /data/GRDC/ -o grdc_stations.csv
    """
    return _ref_commands.generate_station_list(
        dataset_dir,
        output,
        expand_existing_directory_fn=expand_existing_directory,
        expand_path_fn=expand_path,
    )
