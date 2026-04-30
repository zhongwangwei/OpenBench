"""openbench data commands."""

import click


@click.group()
def data():
    """Manage reference datasets."""


@data.command("list")
@click.option("--variable", default=None, help="Filter by variable name.")
def list_datasets(variable):
    """List all available reference datasets."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()

    if variable:
        refs = mgr.references_for_variable(variable)
        if not refs:
            click.echo(f"No datasets found for variable: {variable}")
            return
    else:
        refs = mgr.list_references()

    click.secho(f"{'Name':<30} {'Category':<12} {'Type':<6} {'Res':<8} {'Years':<14} {'Variables'}", bold=True)
    click.echo("─" * 100)
    for r in refs:
        res = f"{r.grid_res}°" if r.grid_res else "stn"
        years = f"{r.years[0]}-{r.years[1]}" if r.years else "?"
        nvars = len(r.variables)
        click.echo(f"{r.name:<30} {r.category:<12} {r.data_type:<6} {res:<8} {years:<14} {nvars}")

    click.echo(f"\nTotal: {len(refs)} datasets")


@data.command()
@click.argument("names", nargs=-1, required=True)
def download(names):
    """[NOT IMPLEMENTED] Download reference datasets by name (planned for v3.0)."""
    click.echo("Dataset download not yet implemented (requires hosted data repository).")
    click.echo(f"Requested: {', '.join(names)}")


@data.command()
def status():
    """Show local dataset cache status. [Cache reporting NOT IMPLEMENTED in v3.0a1; only registry count is shown.]"""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    refs = mgr.list_references()
    click.echo(f"Registry: {len(refs)} datasets available")
    click.echo("Local cache: not yet implemented")


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset.  Supports base names (e.g. ERA5LAND)."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)

    # Fall back to resolution variants if exact match fails
    if ref is None:
        variants = mgr.get_resolution_variants(name)
        if variants:
            for res_label, v in sorted(variants.items()):
                path_str = v.root_dir or "(no local path)"
                click.echo(f"{v.name}: {path_str}")
            return

        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)

    if ref.root_dir:
        click.echo(ref.root_dir)
    else:
        click.echo(f"No local path configured for {name}")


@data.command()
@click.argument("name")
@click.option("--root-dir", default=None, help="Root directory containing data files (required for new).")
@click.option("--data-type", type=click.Choice(["grid", "stn"]), default=None, help="Data type (auto-detected from NC if omitted).")
@click.option("--tim-res", type=click.Choice(["Month", "Day", "Hour", "Year"]), default="Month")
@click.option("--grid-res", type=float, default=None, help="Grid resolution in degrees.")
@click.option("--category", default="Other", help="Category: Water, Carbon, Energy, etc.")
@click.option("--years", nargs=2, type=int, default=None, help="Start and end year.")
@click.option("--fulllist", default=None, type=click.Path(), help="Station list CSV path (for stn data).")
@click.option("-v", "--variable", multiple=True,
              help="'VarName:ncname:unit' (repeatable). Overwrites if exists.")
@click.option("-f", "--fallback", multiple=True,
              help="'VarName:fallback_name:fallback_unit:conversion' (repeatable).")
def register(name, root_dir, data_type, tim_res, grid_res, category, years, fulllist, variable, fallback):
    """Register or update a reference dataset in the registry.

    Creates a new entry or updates an existing one. Variables are overwritten
    by default when names match.

    \b
    Examples:
        openbench data register MyData --root-dir /data/myref \\
            --data-type grid --grid-res 0.5 --tim-res Month \\
            --years 2000 2020 --category Water \\
            -v "Evapotranspiration:ET:mm day-1"

        # With fallback + conversion
        openbench data register ERA5 --root-dir /data/era5 \\
            -v "Latent_Heat:slhf:W m-2" \\
            -f "Latent_Heat:surface_latent_heat_flux:J m-2:value / 3600"

        # Station data with fulllist CSV
        openbench data register PLUMBER2 --root-dir /data/PLUMBER2/dataset \\
            --data-type stn --tim-res Day \\
            --fulllist /data/PLUMBER2/list/PLUMBER2.csv \\
            -v "Latent_Heat:Qle_cor:W/m2" \\
            -v "Sensible_Heat:Qh_cor:W/m2"

        # Update existing: add a variable
        openbench data register MyData -v "Runoff:RNOF:mm day-1"
    """
    import yaml
    from pathlib import Path

    from openbench.data.registry.manager import get_writable_reference_catalog_path

    catalog_path = get_writable_reference_catalog_path()

    existing_catalog = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            existing_catalog = yaml.safe_load(f) or {}

    existing = existing_catalog.get(name, {})
    is_new = name not in existing_catalog

    # Parse primary variables and fallbacks
    from openbench.cli._parsing import parse_fallbacks, parse_variables

    new_vars = parse_variables(variable)
    parse_fallbacks(fallback, new_vars, existing.get("variables", {}))

    # Require root_dir for new entries
    if is_new and not root_dir:
        click.secho("Error: --root-dir is required for new datasets.", fg="red")
        raise SystemExit(1)

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

    if not new_vars and is_new:
        click.secho("No variables defined. Registration cancelled.", fg="yellow")
        return

    # Auto-detect data_type from NC file if not specified
    if data_type is None and root_dir:
        from openbench.data.coordinates import glob_nc
        from openbench.data.registry.scanner import _detect_data_type_from_nc
        # Search root_dir and up to one level of subdirs for a NC file
        root_path = Path(root_dir)
        nc_files = glob_nc(root_path)
        # Try one level deeper if nothing at root
        if not nc_files:
            for child in root_path.iterdir():
                if child.is_dir():
                    nc_files = glob_nc(child)
                    if nc_files:
                        break
        nc_file = nc_files[0] if nc_files else None
        if nc_file is not None:
            detected = _detect_data_type_from_nc(nc_file)
            if detected:
                data_type = detected
                click.echo(f"  Auto-detected data_type: {data_type}")
            else:
                click.echo("  Could not auto-detect data_type from NC file, using default: grid")
    if data_type is None:
        data_type = "grid"  # final fallback

    # Build/update descriptor
    descriptor = existing.copy()
    descriptor["name"] = name
    if is_new or root_dir:
        descriptor["root_dir"] = root_dir
    if is_new:
        descriptor.setdefault("description", f"{name} reference dataset")
        descriptor.setdefault("category", category)
        descriptor.setdefault("data_type", data_type)
        descriptor.setdefault("tim_res", tim_res)
        descriptor.setdefault("data_groupby", "Year")
        descriptor.setdefault("timezone", 0)
        descriptor.setdefault("years", list(years) if years else [2000, 2020])
    else:
        if category != "Other":
            descriptor["category"] = category
        if years:
            descriptor["years"] = list(years)

    if grid_res is not None:
        descriptor["grid_res"] = grid_res
    if fulllist is not None:
        descriptor["fulllist"] = str(Path(fulllist).resolve())

    # Merge variables (overwrite existing)
    merged_vars = descriptor.get("variables", {})
    updated = [k for k in new_vars if k in merged_vars]
    added = [k for k in new_vars if k not in merged_vars]
    merged_vars.update(new_vars)
    descriptor["variables"] = merged_vars

    existing_catalog[name] = descriptor
    from openbench.data.registry.scanner import _atomic_yaml_write
    _atomic_yaml_write(catalog_path, existing_catalog)

    if is_new:
        click.secho(f"✓ Created '{name}' ({len(merged_vars)} variables)", fg="green")
    else:
        parts = []
        if added:
            parts.append(f"{len(added)} added")
        if updated:
            parts.append(f"{len(updated)} updated")
        click.secho(f"✓ Updated '{name}': {', '.join(parts)} ({len(merged_vars)} total)", fg="green")
    click.echo(f"Verify: openbench data show {name}")


@data.command("register-profile")
@click.argument("name")
@click.option("-v", "--variable", multiple=True,
              help="'VarName:ncname:unit' (repeatable).")
@click.option("--tim-res", default=None, help="Time resolution override (Day, Month, Hour).")
@click.option("--data-groupby", default=None, help="Data groupby override (single, Year).")
@click.option("--fulllist", default=None, help="Station list CSV pattern (relative to root_dir).")
@click.option("--description", default=None, help="Dataset description.")
def register_profile(name, variable, tim_res, data_groupby, fulllist, description):
    """Register a reference profile (variable mappings for a dataset type).

    Profiles are shared across resolutions. E.g., registering 'GLEAM_v4.2a'
    applies to GLEAM_v4.2a_LowRes, GLEAM_v4.2a_MidRes, etc.

    \b
    Examples:
        openbench data register-profile MyData \\
            -v "Latent_Heat:LE:W m-2" \\
            -v "Sensible_Heat:H:W m-2"

        openbench data register-profile PLUMBER2_new \\
            --tim-res Day --data-groupby single \\
            --fulllist "../list/stations.csv" \\
            -v "Latent_Heat:Qle_cor:W m-2"
    """
    import yaml

    from openbench.data.registry.manager import clear_registry_cache, get_writable_registry_dir
    from openbench.data.registry.scanner import clear_reference_profile_cache

    profile_path = get_writable_registry_dir() / "reference_profiles.yaml"

    profiles = {}
    if profile_path.exists():
        with open(profile_path) as f:
            profiles = yaml.safe_load(f) or {}

    existing = profiles.get(name, {})

    # Parse variables
    new_vars = {}
    for v in variable:
        parts = v.split(":")
        if len(parts) < 2:
            click.secho(f"Invalid format: '{v}'. Use 'VarName:ncname:unit'", fg="red")
            raise SystemExit(1)
        var_name = parts[0].strip()
        nc_name = parts[1].strip()
        unit = parts[2].strip() if len(parts) > 2 else ""
        new_vars[var_name] = {"varname": nc_name, "varunit": unit}

    if not variable and not existing:
        click.secho("No variables specified. Use -v 'VarName:ncname:unit'", fg="red")
        raise SystemExit(1)

    # Build profile
    profile = existing.copy()
    if description:
        profile["description"] = description
    elif "description" not in profile:
        profile["description"] = f"{name} reference dataset"
    if tim_res:
        profile["tim_res"] = tim_res
    if data_groupby:
        profile["data_groupby"] = data_groupby
    if fulllist:
        profile["fulllist"] = fulllist

    # Merge variables
    merged = profile.get("variables", {})
    merged.update(new_vars)
    profile["variables"] = merged

    profiles[name] = profile

    from openbench.data.registry.scanner import _atomic_yaml_write
    _atomic_yaml_write(profile_path, profiles)

    clear_registry_cache()
    clear_reference_profile_cache()

    is_new = name not in existing or not existing
    action = "Created" if is_new else "Updated"
    click.secho(f"✓ {action} profile '{name}' ({len(merged)} variables)", fg="green")
    click.echo("Re-scan to apply: openbench data scan /path/to/reference")


@data.command()
@click.argument("name")
def show(name):
    """Show details of a dataset. Supports base name to show all resolutions.

    Examples:
        openbench data show GLEAM_v4.2a          # shows all resolutions
        openbench data show GLEAM_v4.2a_LowRes   # shows specific one
    """
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()

    # Check for resolution variants first
    variants = mgr.get_resolution_variants(name)

    if variants and len(variants) > 1:
        # Multi-resolution: show summary of all variants
        click.secho(f"{name}", bold=True)
        click.echo(f"Available at {len(variants)} resolution(s):\n")

        for res_label, ref in sorted(variants.items()):
            status = click.style(f"[{res_label}]", bold=True)
            click.echo(f"  {status} {ref.name}")
            click.echo(f"    Type: {ref.data_type}, Grid: {ref.grid_res or 'N/A'}°, Time: {ref.tim_res}")
            years = f"{ref.years[0]}-{ref.years[1]}" if ref.years else "N/A"
            click.echo(f"    Years: {years}, Variables: {len(ref.variables)}")
            if ref.root_dir:
                click.echo(f"    Path: {ref.root_dir}")
            click.echo()

        click.echo("Base-name references are resolved using target resolution context:")
        click.echo("  - project.tim_res / project.grid_res when set")
        click.echo("  - otherwise the shared simulation resolution, if all simulations agree")
        click.echo("  - otherwise OpenBench asks you to specify a full variant or set comparison.* explicitly")
        click.echo()
        click.echo("In openbench.yaml, use either:")
        click.echo("  reference:")
        click.echo(f"    Evapotranspiration: {name}            # select by target resolution context")
        for res_label, ref in sorted(variants.items()):
            click.echo(f"    Evapotranspiration: {ref.name}   # force {res_label}")
        return

    # Single dataset (exact match or auto-resolved)
    ref = mgr.get_reference(name)
    if ref is None:
        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)

    click.secho(f"{ref.name}", bold=True)
    click.echo(f"Description: {ref.description}")
    click.echo(f"Category: {ref.category}")
    click.echo(f"Type: {ref.data_type}, Resolution: {ref.grid_res or 'N/A'}°, Time: {ref.tim_res}")
    click.echo(f"Years: {ref.years[0]}-{ref.years[1]}" if ref.years else "Years: N/A")
    if ref.root_dir:
        click.echo(f"Path: {ref.root_dir}")
    click.echo()
    click.secho(f"{'Variable':<35} {'NetCDF name':<20} {'Unit':<20} {'Fallback'}", bold=True)
    click.echo("─" * 100)
    for var_name, mapping in sorted(ref.variables.items()):
        vn = mapping.varname if isinstance(mapping.varname, str) else mapping.varname[0]
        fb_str = ""
        if mapping.fallbacks:
            fb_parts = []
            for fb in mapping.fallbacks:
                conv = f" ({fb.convert})" if fb.convert else ""
                fb_parts.append(f"{fb.varname} [{fb.varunit}]{conv}")
            fb_str = " → ".join(fb_parts)
        click.echo(f"{var_name:<35} {vn:<20} {mapping.varunit:<20} {fb_str}")


@data.command()
@click.argument("ref_root", type=click.Path(exists=True))
@click.option("--auto", is_flag=True, help="Register all found datasets without prompting.")
def scan(ref_root, auto):
    """Scan a directory for reference datasets and register new ones.

    REF_ROOT is the reference data root (e.g., /Volumes/work/Reference).
    Expected structure: Grid/{LowRes,MidRes,HigRes}/<category>/<variable>/<dataset>/

    Example:
        openbench data scan /Volumes/work/Reference
    """
    from openbench.data.registry.scanner import find_new_datasets, register_scanned_datasets_batch

    def _progress(msg):
        click.echo(msg)

    click.secho(f"Scanning {ref_root}...", bold=True)
    new_groups = find_new_datasets(ref_root, on_progress=_progress)

    click.echo()
    if not new_groups:
        click.secho("No new datasets found. All datasets already registered.", fg="yellow")
        return

    click.secho(f"Found {len(new_groups)} new dataset(s):", bold=True)
    click.echo()

    to_register = []
    for group in new_groups:
        for res_name, variant in sorted(group.variants.items()):
            label = f"  {variant.registry_name:<35} {variant.data_type:<5} {variant.category:<10} {len(variant.variables)} vars, {variant.file_count} files"
            click.echo(label)
            to_register.append(variant)

    click.echo()

    if not auto:
        if not click.confirm(f"Register {len(to_register)} dataset(s)?"):
            return

    def _multi_var_handler(var_name, sub_dir, all_vars):
        """Prompt user to pick a variable when NC file has multiple data variables."""
        click.echo()
        click.secho(f"  Multiple variables in {sub_dir}/ (evaluating: {var_name}):", fg="yellow")
        for i, v in enumerate(all_vars, 1):
            desc = v.get("long_name") or v.get("standard_name") or ""
            if desc:
                desc = f"  — {desc}"
            click.echo(f"    [{i}] {v['name']:<20} {v['unit']:<15} {v['dims']}{desc}")
        if auto:
            click.echo(f"    → Auto-selected: {all_vars[0]['name']}")
            return all_vars[0]["name"]
        choice = click.prompt("  Select variable number", type=int, default=1)
        idx = max(0, min(choice - 1, len(all_vars) - 1))
        return all_vars[idx]["name"]

    def _register_progress(msg):
        click.secho(f"  ✓{msg.lstrip()}", fg="green")

    register_scanned_datasets_batch(
        to_register,
        on_multi_var=_multi_var_handler,
        on_progress=_register_progress,
    )
    registered = len(to_register)

    # Clear registry cache so subsequent lookups see newly registered datasets
    from openbench.data.registry.manager import clear_registry_cache

    clear_registry_cache()

    click.echo()
    click.secho(f"Registered {registered} dataset(s).", fg="green", bold=True)
    click.echo("Verify: openbench data list")

    # Warn about unverified defaults
    click.echo()
    click.secho(
        "Note: Some fields (tim_res, grid_res, years) may be defaults and not verified from data.",
        fg="yellow",
    )
    click.echo("Use 'openbench data show <name>' to check, and 'openbench data register <name>' to fix.")


@data.command()
@click.argument("name")
def optimize(name):
    """Convert dataset to zarr for faster reads."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)
    if ref is None:
        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)

    if not ref.root_dir:
        click.secho(f"No local path configured for {name}. Download it first.", fg="red")
        raise SystemExit(1)

    try:
        import xarray as xr
    except ImportError:
        click.secho("xarray is required for optimization.", fg="red")
        raise SystemExit(1)

    from pathlib import Path

    root = Path(ref.root_dir)
    if not root.exists():
        click.secho(f"Data directory not found: {root}", fg="red")
        raise SystemExit(1)

    zarr_dir = root.parent / f"{root.name}.zarr"
    if zarr_dir.exists():
        click.echo(f"Zarr store already exists: {zarr_dir}")
        if not click.confirm("Overwrite?"):
            return

    # Find all NetCDF files
    from openbench.data.coordinates import glob_nc

    nc_files = glob_nc(root, recursive=True)

    if not nc_files:
        click.secho(f"No NetCDF files found in {root}", fg="yellow")
        return

    click.echo(f"Found {len(nc_files)} NetCDF files")
    click.echo(f"Converting to zarr: {zarr_dir}")

    try:
        # Open all files as a single dataset and save as zarr
        ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
        ds.to_zarr(str(zarr_dir), mode="w")
        ds.close()

        # Report size comparison
        import os

        nc_size = sum(os.path.getsize(f) for f in nc_files)
        zarr_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(zarr_dir)
            for f in fns
        )
        click.secho("✓ Converted to zarr", fg="green")
        click.echo(f"  NetCDF: {nc_size / 1e9:.1f} GB")
        click.echo(f"  Zarr:   {zarr_size / 1e9:.1f} GB")
    except Exception as e:
        click.secho(f"Conversion failed: {e}", fg="red")
        raise SystemExit(1)


@data.command("generate-station-list")
@click.argument("dataset_dir", type=click.Path(exists=True))
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
      openbench data generate-station-list /data/PLUMBER2/dataset/
      openbench data generate-station-list /data/GRDC/ -o grdc_stations.csv
    """
    from pathlib import Path

    from openbench.data.registry.scanner import generate_station_list as gen_list

    dataset_path = Path(dataset_dir)
    output_path = Path(output) if output else None

    try:
        csv_path = gen_list(dataset_path, output_path)
        import pandas as pd

        df = pd.read_csv(csv_path)
        click.secho(f"✓ Generated {csv_path}", fg="green")
        click.echo(f"  Stations: {len(df)}")
        if "LON" in df.columns and "LAT" in df.columns:
            click.echo(f"  Lon range: [{df['LON'].min():.2f}, {df['LON'].max():.2f}]")
            click.echo(f"  Lat range: [{df['LAT'].min():.2f}, {df['LAT'].max():.2f}]")
        if "SYEAR" in df.columns and "EYEAR" in df.columns:
            click.echo(f"  Year range: [{df['SYEAR'].min()}, {df['EYEAR'].max()}]")
        elif "SYEAR" in df.columns:
            click.echo(f"  Start year: {df['SYEAR'].min()}-{df['SYEAR'].max()}")
    except Exception as e:
        click.secho(f"Failed: {e}", fg="red")
        raise SystemExit(1)
