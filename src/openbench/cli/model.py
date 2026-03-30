"""openbench model commands."""

import click
import yaml


@click.group()
def model():
    """Manage model profiles."""


@model.command("list")
def list_models():
    """List all available model profiles."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    models = mgr.list_models()

    click.secho(f"{'Name':<20} {'Type':<6} {'Res':<8} {'Variables'}", bold=True)
    click.echo("─" * 60)
    for m in models:
        res = f"{m.grid_res}°" if m.grid_res else "?"
        click.echo(f"{m.name:<20} {m.data_type:<6} {res:<8} {len(m.variables)}")

    click.echo(f"\nTotal: {len(models)} model profiles")


@model.command()
@click.argument("name")
def show(name):
    """Show variable mappings for a model profile."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    m = mgr.get_model(name)
    if m is None:
        click.secho(f"Model profile not found: {name}", fg="red")
        raise SystemExit(1)

    click.secho(f"{m.name}", bold=True)
    click.echo(f"Description: {m.description}")
    click.echo(f"Type: {m.data_type}, Resolution: {m.grid_res}°, Time: {m.tim_res}")
    click.echo()
    click.secho(f"{'Variable':<35} {'File varname':<30} {'Unit'}", bold=True)
    click.echo("─" * 80)
    for var_name, mapping in sorted(m.variables.items()):
        vn = mapping.varname
        if isinstance(vn, list):
            vn_str = " → ".join(vn)
        else:
            vn_str = str(vn)
        click.echo(f"{var_name:<35} {vn_str:<30} {mapping.varunit}")


@model.command()
@click.argument("name")
@click.option("--data-type", type=click.Choice(["grid", "stn"]), default=None)
@click.option("--grid-res", type=float, default=None)
@click.option("--tim-res", type=click.Choice(["Month", "Day", "Hour", "Year"]), default=None)
@click.option("--description", default=None)
@click.option("-v", "--variable", multiple=True,
              help="Variable: 'StdName:ncname:unit' or 'StdName:name1,name2:unit' for fallback. Repeatable.")
@click.option("--append-only", is_flag=True, help="Only add new variables, skip existing ones.")
def register(name, data_type, grid_res, tim_res, description, variable, append_only):
    """Register or update a model profile.

    Creates a new profile or updates an existing one.
    Variables are appended by default; use --overwrite to replace existing.

    \b
    Examples:
      # Create new model
      openbench model register MyModel --data-type grid --grid-res 0.5 --tim-res Month \\
        -v "Evapotranspiration:ET:mm day-1" \\
        -v "GPP:gpp,assim:gC m-2 s-1"

      # Add or update a variable
      openbench model register CoLM2024 -v "Snow_Depth:f_snowdp:m"

      # Only add new, don't touch existing
      openbench model register CoLM2024 --append-only -v "GPP:f_gpp:g m-2 s-1"
    """
    from pathlib import Path

    from openbench.data.registry.manager import RegistryManager, get_writable_registry_dir

    registry_dir = get_writable_registry_dir()
    catalog_path = registry_dir / "model_catalog.yaml"

    # Load existing catalog
    catalog = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f) or {}

    # Load existing profile if updating
    existing = catalog.get(name, {})
    is_new = name not in catalog

    if is_new and not data_type:
        data_type = click.prompt("Data type", type=click.Choice(["grid", "stn"]), default="grid")
    if is_new and grid_res is None and (data_type or existing.get("data_type")) == "grid":
        grid_res = click.prompt("Grid resolution (degrees)", type=float, default=0.5)
    if is_new and tim_res is None:
        tim_res = click.prompt("Time resolution", type=click.Choice(["Month", "Day", "Hour", "Year"]), default="Month")

    # Build/update the profile
    profile = existing.copy()
    profile["name"] = name
    if description:
        profile["description"] = description
    elif "description" not in profile:
        profile["description"] = f"{name} model profile"
    if data_type:
        profile["data_type"] = data_type
    elif "data_type" not in profile:
        profile["data_type"] = "grid"
    if grid_res is not None:
        profile["grid_res"] = grid_res
    if tim_res:
        profile["tim_res"] = tim_res
    elif "tim_res" not in profile:
        profile["tim_res"] = "Month"

    # Parse and merge variables
    existing_vars = profile.get("variables", {})
    new_vars = {}
    for v in variable:
        parts = v.split(":")
        if len(parts) < 2:
            click.secho(f"Invalid format: '{v}'. Use 'StdName:ncname:unit'", fg="red")
            raise SystemExit(1)

        std_name = parts[0].strip()
        nc_names = parts[1].strip()
        unit = parts[2].strip() if len(parts) > 2 else ""

        # Support fallback: "f_gpp,f_assim" → ["f_gpp", "f_assim"]
        if "," in nc_names:
            varname = [n.strip() for n in nc_names.split(",")]
        else:
            varname = nc_names

        new_vars[std_name] = {"varname": varname, "varunit": unit}

    if not variable and is_new:
        # Interactive variable entry
        click.echo("\nAdd variables (empty name to finish):")
        while True:
            std_name = click.prompt("  Standard variable name (e.g., Evapotranspiration)", default="")
            if not std_name:
                break
            nc_name = click.prompt(f"  NetCDF variable name(s), comma-separated for fallback", default=std_name)
            unit = click.prompt(f"  Unit", default="")
            if "," in nc_name:
                varname = [n.strip() for n in nc_name.split(",")]
            else:
                varname = nc_name
            new_vars[std_name] = {"varname": varname, "varunit": unit}

    # Merge variables — default: overwrite existing; --append-only: skip existing
    updated_keys = []
    added_keys = []
    skipped_keys = []

    merged_vars = dict(existing_vars)
    for k, v in new_vars.items():
        if k in merged_vars:
            if append_only:
                skipped_keys.append(k)
                continue
            updated_keys.append(k)
        else:
            added_keys.append(k)
        merged_vars[k] = v

    profile["variables"] = merged_vars

    # Save
    catalog[name] = profile
    with open(catalog_path, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Report
    if is_new:
        click.secho(f"✓ Created model profile '{name}' ({len(merged_vars)} variables)", fg="green")
    else:
        parts = []
        if added_keys:
            parts.append(f"{len(added_keys)} added")
        if updated_keys:
            parts.append(f"{len(updated_keys)} updated")
        if skipped_keys:
            parts.append(f"{len(skipped_keys)} skipped (use without --append-only to update)")
        click.secho(f"✓ Updated '{name}': {', '.join(parts)} ({len(merged_vars)} total)", fg="green")

    click.echo(f"Verify: openbench model show {name}")


@model.command()
@click.argument("name")
@click.argument("variable_name")
def remove_var(name, variable_name):
    """Remove a variable from a model profile.

    Example: openbench model remove-var CoLM2024 Snow_Depth
    """
    from openbench.data.registry.manager import get_writable_registry_dir

    registry_dir = get_writable_registry_dir()
    catalog_path = registry_dir / "model_catalog.yaml"

    catalog = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f) or {}

    if name not in catalog:
        click.secho(f"Model not found: {name}", fg="red")
        raise SystemExit(1)

    variables = catalog[name].get("variables", {})
    if variable_name not in variables:
        click.secho(f"Variable '{variable_name}' not in {name}", fg="red")
        raise SystemExit(1)

    del variables[variable_name]
    catalog[name]["variables"] = variables

    with open(catalog_path, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.secho(f"✓ Removed '{variable_name}' from {name} ({len(variables)} remaining)", fg="green")
