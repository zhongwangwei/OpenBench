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
    """Download reference datasets by name."""
    click.echo("Dataset download not yet implemented (requires hosted data repository).")
    click.echo(f"Requested: {', '.join(names)}")


@data.command()
def status():
    """Show local dataset cache status."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    refs = mgr.list_references()
    click.echo(f"Registry: {len(refs)} datasets available")
    click.echo("Local cache: not yet implemented")


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)
    if ref is None:
        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)
    if ref.root_dir:
        click.echo(ref.root_dir)
    else:
        click.echo(f"No local path configured for {name}")


@data.command()
@click.argument("name")
@click.option("--root-dir", required=True, help="Root directory containing data files.")
@click.option("--data-type", type=click.Choice(["grid", "stn"]), default="grid", help="Data type.")
@click.option("--tim-res", type=click.Choice(["Month", "Day", "Hour", "Year"]), default="Month")
@click.option("--grid-res", type=float, default=None, help="Grid resolution in degrees.")
@click.option("--category", default="Other", help="Category: Water, Carbon, Energy, etc.")
@click.option("--years", nargs=2, type=int, default=None, help="Start and end year.")
@click.option("--variable", "-v", multiple=True, help="Variable: 'VarName:ncname:unit' (repeatable).")
def register(name, root_dir, data_type, tim_res, grid_res, category, years, variable):
    """Register a new reference dataset into the local registry.

    Example:
        openbench data register MyData --root-dir /data/myref \\
            --data-type grid --grid-res 0.5 --tim-res Month \\
            --years 2000 2020 --category Water \\
            -v "Evapotranspiration:ET:mm day-1" \\
            -v "Runoff:RNOF:mm day-1"
    """
    import yaml
    from pathlib import Path

    from openbench.data.registry.manager import get_writable_registry_dir

    registry_dir = get_writable_registry_dir()
    catalog_path = registry_dir / "reference_catalog.yaml"

    # Check if already in catalog
    existing_catalog = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            existing_catalog = yaml.safe_load(f) or {}
    if name in existing_catalog:
        if not click.confirm(f"'{name}' already registered. Overwrite?"):
            return

    # Parse variable mappings
    variables = {}
    for v in variable:
        parts = v.split(":")
        if len(parts) < 2:
            click.secho(f"Invalid variable format: '{v}'. Use 'VarName:ncname:unit'", fg="red")
            raise SystemExit(1)
        var_name = parts[0].strip()
        nc_name = parts[1].strip()
        unit = parts[2].strip() if len(parts) > 2 else ""
        variables[var_name] = {"varname": nc_name, "varunit": unit}

    if not variables:
        # Interactive variable entry
        click.echo("\nAdd variables (empty name to finish):")
        while True:
            var_name = click.prompt("  Standard variable name (e.g., Evapotranspiration)", default="")
            if not var_name:
                break
            nc_name = click.prompt(f"  Variable name in NetCDF file", default=var_name)
            unit = click.prompt(f"  Unit", default="")
            prefix = click.prompt(f"  File prefix", default="")
            suffix = click.prompt(f"  File suffix", default="")
            entry = {"varname": nc_name, "varunit": unit}
            if prefix:
                entry["prefix"] = prefix
            if suffix:
                entry["suffix"] = suffix
            variables[var_name] = entry

    if not variables:
        click.secho("No variables defined. Registration cancelled.", fg="yellow")
        return

    # Build descriptor
    descriptor = {
        "name": name,
        "description": f"{name} reference dataset",
        "category": category,
        "data_type": data_type,
        "tim_res": tim_res,
        "data_groupby": "Year",
        "timezone": 0,
        "years": list(years) if years else [2000, 2020],
        "variables": variables,
        "root_dir": root_dir,
    }
    if grid_res is not None:
        descriptor["grid_res"] = grid_res

    existing_catalog[name] = descriptor
    with open(catalog_path, "w") as f:
        yaml.dump(existing_catalog, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.secho(f"✓ Registered '{name}' to {catalog_path}", fg="green")
    click.echo(f"  Variables: {', '.join(variables.keys())}")
    click.echo(f"  Data: {root_dir}")
    click.echo(f"\nVerify: openbench data list --variable {list(variables.keys())[0]}")


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

        from openbench.data.registry.scanner import _tim_res_rank

        best_rank = max(_tim_res_rank(r.tim_res) for r in variants.values())

        for res_label, ref in sorted(variants.items()):
            rank = _tim_res_rank(ref.tim_res)
            is_best = rank >= best_rank
            marker = " ← auto-selected (highest frequency)" if is_best else ""
            status = click.style(f"[{res_label}]", bold=is_best)

            click.echo(f"  {status} {ref.name}")
            click.echo(f"    Type: {ref.data_type}, Grid: {ref.grid_res or 'N/A'}°, Time: {ref.tim_res}{marker}")
            years = f"{ref.years[0]}-{ref.years[1]}" if ref.years else "N/A"
            click.echo(f"    Years: {years}, Variables: {len(ref.variables)}")
            if ref.root_dir:
                click.echo(f"    Path: {ref.root_dir}")
            click.echo()

        click.echo("In openbench.yaml, use either:")
        click.echo(f"  reference:")
        click.echo(f"    Evapotranspiration: {name}            # auto-select best resolution")
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
    click.secho(f"{'Variable':<35} {'NetCDF name':<20} {'Unit'}", bold=True)
    click.echo("─" * 70)
    for var_name, mapping in sorted(ref.variables.items()):
        click.echo(f"{var_name:<35} {mapping.varname:<20} {mapping.varunit}")


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
    from openbench.data.registry.scanner import find_new_datasets, register_scanned_dataset

    click.echo(f"Scanning {ref_root}...")
    new_groups = find_new_datasets(ref_root)

    if not new_groups:
        click.secho("No new datasets found.", fg="yellow")
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

    # Try to find existing descriptors for merging
    from openbench.data.registry.manager import RegistryManager

    mgr = RegistryManager()
    registered = 0
    for variant in to_register:
        # Look for existing metadata: try exact name, then base name (without suffix)
        existing = mgr.get_reference(variant.name) or mgr.get_reference(variant.registry_name)
        if not existing:
            # Try base name (e.g., GLEAM_v4.2a from old built-in entries)
            existing = mgr.get_reference(variant.name)
        existing_dict = None
        if existing:
            existing_dict = {
                "variables": {
                    vn: {"varname": vm.varname, "varunit": vm.varunit,
                         "prefix": vm.prefix, "suffix": vm.suffix}
                    for vn, vm in existing.variables.items()
                }
            }
        path = register_scanned_dataset(variant, existing_descriptor=existing_dict)
        click.secho(f"  ✓ {variant.registry_name}", fg="green")
        registered += 1

    click.echo()
    click.secho(f"Registered {registered} dataset(s).", fg="green", bold=True)
    click.echo("Verify: openbench data list")


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

    import glob
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
    nc_files = sorted(glob.glob(str(root / "**" / "*.nc"), recursive=True))
    nc_files += sorted(glob.glob(str(root / "**" / "*.nc4"), recursive=True))

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
