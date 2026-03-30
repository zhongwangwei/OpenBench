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
    click.secho(f"{'Variable':<35} {'Source':<25} {'Unit':<15} {'Notes'}", bold=True)
    click.echo("─" * 100)
    for var_name, mapping in sorted(m.variables.items()):
        # Determine source type and display
        raw = m.variables[var_name]
        # Check for compute/filter (stored in the raw YAML dict, not in VariableMapping)
        # We need to access the raw data through the catalog
        vn = mapping.varname
        if isinstance(vn, list):
            vn_str = vn[0]
        else:
            vn_str = str(vn)

        notes_parts = []
        if mapping.fallbacks:
            for fb in mapping.fallbacks:
                conv = f", {fb.convert}" if fb.convert else ""
                notes_parts.append(f"fallback: {fb.varname} [{fb.varunit}{conv}]")
        if mapping.compute:
            expr = mapping.compute if len(mapping.compute) < 40 else mapping.compute[:37] + "..."
            notes_parts.append(f"compute: {expr}")
        if mapping.filter:
            notes_parts.append(f"filter: {mapping.filter}")
        notes = " | ".join(notes_parts)

        click.echo(f"{var_name:<35} {vn_str:<25} {mapping.varunit:<15} {notes}")


@model.command()
@click.argument("name")
@click.option("--data-type", type=click.Choice(["grid", "stn"]), default=None)
@click.option("--grid-res", type=float, default=None)
@click.option("--tim-res", type=click.Choice(["Month", "Day", "Hour", "Year"]), default=None)
@click.option("--description", default=None)
@click.option("-v", "--variable", multiple=True,
              help="'StdName:ncname:unit' (repeatable). Overwrites if exists.")
@click.option("-f", "--fallback", multiple=True,
              help="'StdName:fallback_name:fallback_unit:conversion' (repeatable).")
@click.option("--append-only", is_flag=True, help="Only add new variables, skip existing ones.")
def register(name, data_type, grid_res, tim_res, description, variable, fallback, append_only):
    """Register or update a model profile.

    \b
    Examples:
      # Create new model
      openbench model register MyModel --data-type grid --grid-res 0.5 \\
        -v "Evapotranspiration:ET:mm day-1"

      # Add primary + fallback with conversion
      openbench model register CoLM2024 \\
        -v "Gross_Primary_Productivity:f_gpp:g m-2 s-1" \\
        -f "Gross_Primary_Productivity:f_assim:mol m-2 s-1:value * 12.011"

      # Just add a simple variable
      openbench model register CoLM2024 -v "Snow_Depth:f_snowdp:m"
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

    # Parse primary variables
    existing_vars = profile.get("variables", {})
    new_vars = {}
    for v in variable:
        parts = v.split(":")
        if len(parts) < 2:
            click.secho(f"Invalid format: '{v}'. Use 'StdName:ncname:unit'", fg="red")
            raise SystemExit(1)

        std_name = parts[0].strip()
        nc_name = parts[1].strip()
        unit = parts[2].strip() if len(parts) > 2 else ""

        new_vars[std_name] = {"varname": nc_name, "varunit": unit}

    # Parse fallback definitions → attach to corresponding variable
    for fb_def in fallback:
        parts = fb_def.split(":")
        if len(parts) < 3:
            click.secho(f"Invalid fallback: '{fb_def}'. Use 'StdName:fallback_name:fallback_unit[:conversion]'", fg="red")
            raise SystemExit(1)

        std_name = parts[0].strip()
        fb_name = parts[1].strip()
        fb_unit = parts[2].strip()
        fb_convert = parts[3].strip() if len(parts) > 3 else ""

        fb_entry = {"varname": fb_name, "varunit": fb_unit}
        if fb_convert:
            fb_entry["convert"] = fb_convert

        # Attach to the variable (create if not in new_vars, merge with existing)
        if std_name in new_vars:
            if "fallbacks" not in new_vars[std_name]:
                new_vars[std_name]["fallbacks"] = []
            new_vars[std_name]["fallbacks"].append(fb_entry)
        elif std_name in existing_vars:
            # Fallback for existing variable that's not being updated
            if std_name not in new_vars:
                new_vars[std_name] = dict(existing_vars[std_name])
            if "fallbacks" not in new_vars[std_name]:
                new_vars[std_name]["fallbacks"] = []
            new_vars[std_name]["fallbacks"].append(fb_entry)
        else:
            click.secho(f"Warning: fallback for '{std_name}' but no primary variable defined. Use -v first.", fg="yellow")

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
