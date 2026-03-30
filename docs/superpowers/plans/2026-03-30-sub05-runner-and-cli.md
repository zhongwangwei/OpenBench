# Sub-project 5: Runner Migration + CLI Completion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the runner system (local + remote execution) and finish remaining CLI commands (`openbench init`). At the end, `openbench run` orchestrates actual evaluation, and the remote infrastructure is in place.

**Architecture:** Local runner directly calls migrated evaluation code via the config adapter. Remote runner is migrated from openbench-wizard with SSH/SFTP support. The `openbench init` command provides an interactive CLI wizard for generating configs.

**Tech Stack:** Python, click, paramiko (optional), PyYAML

**Working Directory:** `/Volumes/Data01/Openbench`

---

### Task 1: Migrate Remote SSH Infrastructure

**Files:**
- Copy + adapt: 5 SSH modules from openbench-wizard/core/

- [ ] **Step 1: Copy SSH modules**

```bash
cd /Volumes/Data01/Openbench
cp openbench-wizard/core/ssh_manager.py src/openbench/remote/ssh.py
cp openbench-wizard/core/credential_manager.py src/openbench/remote/credentials.py
cp openbench-wizard/core/connection_manager.py src/openbench/remote/connections.py
cp openbench-wizard/core/sync_engine.py src/openbench/remote/sync.py
cp openbench-wizard/core/storage.py src/openbench/remote/storage.py
```

- [ ] **Step 2: Fix imports**

In all 5 files, replace:
- `from core.ssh_manager` → `from openbench.remote.ssh`
- `from core.credential_manager` → `from openbench.remote.credentials`
- `from core.connection_manager` → `from openbench.remote.connections`
- `from core.sync_engine` → `from openbench.remote.sync`
- `from core.storage` → `from openbench.remote.storage`
- `from core.` → `from openbench.remote.` (general pattern)
- Any `from PySide6` imports should be wrapped in try/except (remote modules shouldn't depend on Qt)

- [ ] **Step 3: Update `src/openbench/remote/__init__.py`**

```python
"""SSH infrastructure for remote execution (requires openbench[remote]).

This package requires paramiko. If not installed, importing submodules
will raise ImportError with an installation hint.
"""


def _check_remote_deps():
    """Check that remote dependencies are available."""
    try:
        import paramiko  # noqa: F401
    except ImportError:
        raise ImportError(
            "Remote execution requires paramiko. Install with: pip install 'openbench[remote]'"
        ) from None
```

- [ ] **Step 4: Commit**

```bash
git add src/openbench/remote/
git commit -m "feat(remote): migrate SSH infrastructure from openbench-wizard"
```

---

### Task 2: Migrate Remote Runner

**Files:**
- Copy + adapt: remote runner from openbench-wizard

- [ ] **Step 1: Copy remote runner**

```bash
cp openbench-wizard/core/remote_runner.py src/openbench/runner/remote.py
```

- [ ] **Step 2: Fix imports**

Replace:
- `from core.ssh_manager` → `from openbench.remote.ssh`
- `from core.remote_runner` → remove (self-reference)
- Any Qt signal imports → replace with simple callback pattern
- Remove PySide6 dependency (use plain Python callbacks instead of Qt signals)

The remote runner should work without Qt. Replace Qt signals with a simple callback interface:

```python
# Instead of: self.progress_updated.emit(progress)
# Use:        if self._on_progress: self._on_progress(progress)
```

- [ ] **Step 3: Commit**

```bash
git add src/openbench/runner/remote.py
git commit -m "feat(runner): migrate remote runner from openbench-wizard"
```

---

### Task 3: Implement `openbench init` Interactive Config Generator

**Files:**
- Modify: `src/openbench/cli/init_cmd.py`

- [ ] **Step 1: Write the interactive init command**

Replace `src/openbench/cli/init_cmd.py` entirely:

```python
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
    all_variables = set()
    for ref in all_refs:
        all_variables.update(ref.variables.keys())

    # Group by rough category
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
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        selected_vars = [var_list[i] for i in indices if 0 <= i < len(var_list)]

    if not selected_vars:
        click.secho("  No variables selected. Using all.", fg="yellow")
        selected_vars = var_list

    # Reference selection
    click.echo()
    click.secho("3. Reference Data Sources", bold=True)
    reference = {}
    for var in selected_vars:
        available = mgr.references_for_variable(var)
        if len(available) == 1:
            reference[var] = available[0].name
            click.echo(f"  {var} → {available[0].name} (only option)")
        elif len(available) > 1:
            click.echo(f"  {var} — available sources:")
            for i, ref in enumerate(available, 1):
                click.echo(f"    [{i}] {ref.name} ({ref.data_type}, {ref.tim_res})")
            choice = click.prompt(f"  Select for {var}", type=int, default=1)
            reference[var] = available[max(0, min(choice - 1, len(available) - 1))].name
        else:
            click.secho(f"  {var} — no reference data available, skipping", fg="yellow")
            selected_vars = [v for v in selected_vars if v != var]

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
            idx = int(model_input) - 1
            if 0 <= idx < len(models):
                model_name = models[idx].name
            else:
                click.secho("  Invalid selection", fg="red")
                continue
        else:
            model_name = model_input

        root_dir = click.prompt(f"  Data root directory for {model_name}")
        label = click.prompt(f"  Label for this run", default=model_name)

        entry = {"model": model_name, "root_dir": root_dir}
        simulation[label] = entry
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
    click.secho(f"✓ Generated {output}", fg="green", bold=True)
    click.echo(f"  Next: openbench check {output}")
```

- [ ] **Step 2: Test**

```bash
echo "" | openbench init -o /tmp/test-init.yaml
# Or test interactively: openbench init
```

- [ ] **Step 3: Commit**

```bash
git add src/openbench/cli/init_cmd.py
git commit -m "feat(cli): implement interactive openbench init config generator"
```

---

### Task 4: Final Verification

- [ ] **Step 1: Run lint**

```bash
ruff check src/ tests/
ruff format src/ tests/
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 3: Verify all CLI commands**

```bash
openbench --help
openbench run tests/test_config/fixtures/full.yaml --dry-run
openbench check tests/test_config/fixtures/minimal.yaml
openbench data list | head -5
openbench model list
openbench model show CoLM2024
openbench migrate tests/test_config/fixtures/old_json/main.json -o /tmp/test-migrate.yaml
openbench version
```

- [ ] **Step 4: Commit and tag**

```bash
git add -A
git commit -m "chore: SP5 final cleanup"
git tag -a v3.0.0a5 -m "Sub-project 5 complete: runner migration and CLI completion"
```
