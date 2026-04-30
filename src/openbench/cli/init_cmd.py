"""openbench init command — interactive config generator."""

import click
import yaml


@click.command("init")
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def init_cmd(output):
    """Interactively generate an openbench.yaml config file."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()

    click.secho("OpenBench Configuration Wizard", bold=True)
    click.echo()

    # Project settings
    click.secho("1. Project Settings", bold=True)
    name = click.prompt("  Project name", default="my-evaluation")
    output_dir = click.prompt("  Output directory", default="./output")
    syear = click.prompt("  Start year", type=int, default=2004)
    eyear = click.prompt("  End year", type=int, default=2010)

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
    idx = 1
    for cat in sorted(categories.keys()):
        click.echo(f"  {cat}:")
        for var in sorted(categories[cat]):
            click.echo(f"    [{idx}] {var}")
            var_list.append(var)
            idx += 1

    click.echo()
    selection = click.prompt(
        "  Select variables (comma-separated numbers, or 'all')",
        default="all",
    )
    if selection.strip().lower() == "all":
        selected_vars = var_list
    else:
        # Reject "0" and any negative input — int("0") - 1 = -1 would silently
        # select the last item via Python negative indexing.
        indices = []
        for x in selection.split(","):
            s = x.strip()
            if not s.isdigit():
                continue
            n = int(s)
            if n < 1:
                click.secho(f"  Ignoring out-of-range selection: {s}", fg="yellow")
                continue
            indices.append(n - 1)
        selected_vars = [var_list[i] for i in indices if 0 <= i < len(var_list)]

    if not selected_vars:
        click.secho("  No variables selected. Using all.", fg="yellow")
        selected_vars = var_list

    # Reference selection
    click.echo()
    click.secho("3. Reference Data Sources", bold=True)
    reference = {}
    for var in list(selected_vars):
        available = mgr.references_for_variable(var)
        if len(available) == 1:
            reference[var] = available[0].name
            click.echo(f"  {var} -> {available[0].name} (only option)")
        elif len(available) > 1:
            click.echo(f"  {var} - available sources:")
            for i, ref in enumerate(available, 1):
                click.echo(f"    [{i}] {ref.name} ({ref.data_type}, {ref.tim_res})")
            choice = click.prompt(f"  Select for {var}", type=int, default=1)
            reference[var] = available[max(0, min(choice - 1, len(available) - 1))].name
        else:
            click.secho(f"  {var} - no reference data available, skipping", fg="yellow")
            selected_vars.remove(var)

    # Simulation selection
    click.echo()
    click.secho("4. Simulation Models", bold=True)
    models = mgr.list_models()
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

        # Resolve model name
        if model_input.isdigit():
            midx = int(model_input) - 1
            if 0 <= midx < len(models):
                model_name = models[midx].name
            else:
                click.secho("  Invalid selection", fg="red")
                continue
        else:
            model_name = model_input

        root_dir = click.prompt(f"  Data root directory for {model_name}")
        label = click.prompt("  Label for this run", default=model_name)

        simulation[label] = {"model": model_name, "root_dir": root_dir}
        click.echo(f"  Added: {label}")

    if not simulation:
        click.secho("  No simulations added. Adding placeholder.", fg="yellow")
        simulation["MyModel"] = {"model": "MyModel", "root_dir": "/path/to/data"}

    # Options
    click.echo()
    click.secho("5. Options", bold=True)
    comparison = click.confirm("  Enable comparison?", default=True)
    statistics = click.confirm("  Enable statistics?", default=False)

    # Build config
    # NOTE: reference uses flat var→source mapping (matches loader._build_reference);
    # not {"sources": {...}} which loader rejects as "reference.sources must be a string".
    config = {
        "project": {
            "name": name,
            "output_dir": output_dir,
            "years": [syear, eyear],
        },
        "evaluation": {"variables": selected_vars},
        "reference": reference,
        "simulation": simulation,
    }

    if comparison:
        config["comparison"] = {"enabled": True}
    if statistics:
        config["statistics"] = {"enabled": True}

    # Write
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.echo()
    click.secho(f"Generated {output}", fg="green", bold=True)
    click.echo(f"  Next: openbench check {output}")
