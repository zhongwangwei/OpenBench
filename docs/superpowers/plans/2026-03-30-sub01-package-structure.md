# Sub-project 1: Package Structure + pyproject.toml + CI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the unified `openbench` package skeleton with `src` layout, `pyproject.toml` with optional extras, and GitHub Actions CI — no functional code yet, just the scaffold that all subsequent sub-projects build on.

**Architecture:** Merge the two repositories (`OpenBench-wei` and `openbench-wizard`) into a single `src/openbench/` package using the modern Python `src` layout. The build system uses `hatchling`. Optional dependencies are split into extras (`[gui]`, `[remote]`, `[report]`, `[dev]`, `[all]`). CI runs lint (ruff) and tests (pytest) on push.

**Tech Stack:** Python 3.10+, hatchling (build), click (CLI), ruff (lint), pytest (test), GitHub Actions (CI)

**Spec:** `docs/superpowers/specs/2026-03-30-openbench-unification-design.md`

**Working Directory:** `/Volumes/Data01/Openbench`

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/openbench/__init__.py` | Package version + metadata |
| Create | `src/openbench/__main__.py` | `python -m openbench` support |
| Create | `src/openbench/cli/__init__.py` | CLI package |
| Create | `src/openbench/cli/main.py` | Click command group + `version` command |
| Create | `src/openbench/config/__init__.py` | Config package placeholder |
| Create | `src/openbench/core/__init__.py` | Core package placeholder |
| Create | `src/openbench/data/__init__.py` | Data package placeholder |
| Create | `src/openbench/data/registry/__init__.py` | Registry package placeholder |
| Create | `src/openbench/visualization/__init__.py` | Visualization package placeholder |
| Create | `src/openbench/runner/__init__.py` | Runner package placeholder |
| Create | `src/openbench/remote/__init__.py` | Remote package placeholder |
| Create | `src/openbench/gui/__init__.py` | GUI package placeholder (lazy import guard) |
| Create | `src/openbench/util/__init__.py` | Util package placeholder |
| Create | `pyproject.toml` | Build config, deps, extras, scripts |
| Create | `tests/__init__.py` | Test package |
| Create | `tests/test_smoke.py` | Smoke test: import + CLI entry point |
| Create | `.github/workflows/ci.yml` | GitHub Actions CI |
| Create | `.gitignore` | Python + project-specific ignores |
| Keep | `OpenBench-wei/` | Existing code untouched (referenced later) |
| Keep | `openbench-wizard/` | Existing code untouched (referenced later) |

---

### Task 1: Initialize Git Repository

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
*.egg
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project
output/
*.nc
*.nc4
cache/
.pytest_cache/
.ruff_cache/
htmlcov/
.coverage
```

- [ ] **Step 2: Initialize repo and make initial commit**

Run:
```bash
cd /Volumes/Data01/Openbench
git init
git add .gitignore
git commit -m "chore: initialize openbench unified repository"
```

Expected: Clean commit with `.gitignore` only.

---

### Task 2: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/openbench"]

[project]
name = "openbench"
version = "3.0.0a1"
description = "Open Source Land Surface Model Benchmarking System"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [
    { name = "Zhongwang Wei" },
    { name = "CoLM-SYSU" },
]
keywords = ["land surface model", "benchmarking", "evaluation", "climate", "earth science"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]

dependencies = [
    "xarray>=0.19.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "netCDF4>=1.5.7",
    "matplotlib>=3.4.0",
    "cartopy>=0.20.0",
    "dask>=2022.1.0",
    "joblib>=1.1.0",
    "flox>=0.5.0",
    "PyYAML>=6.0",
    "jinja2>=3.0.0",
    "click>=8.0",
    "platformdirs>=3.0",
]

[project.optional-dependencies]
gui = ["PySide6>=6.5.0"]
remote = ["paramiko>=3.0.0", "cryptography>=41.0.0"]
report = ["xhtml2pdf>=0.2.5"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.4.0",
]
all = ["openbench[gui,remote,report]"]

[project.scripts]
openbench = "openbench.cli.main:cli"

[project.urls]
Homepage = "https://github.com/CoLM-SYSU/OpenBench"
Documentation = "https://openbench.readthedocs.io"
Repository = "https://github.com/CoLM-SYSU/OpenBench"
Issues = "https://github.com/CoLM-SYSU/OpenBench/issues"

[tool.ruff]
target-version = "py310"
line-length = 120
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Commit pyproject.toml**

Run:
```bash
cd /Volumes/Data01/Openbench
git add pyproject.toml
git commit -m "chore: add pyproject.toml with hatchling build system and extras"
```

Expected: Clean commit.

---

### Task 3: Create Package Skeleton

**Files:**
- Create: `src/openbench/__init__.py`
- Create: `src/openbench/__main__.py`
- Create: `src/openbench/cli/__init__.py`
- Create: `src/openbench/cli/main.py`
- Create: `src/openbench/config/__init__.py`
- Create: `src/openbench/core/__init__.py`
- Create: `src/openbench/data/__init__.py`
- Create: `src/openbench/data/registry/__init__.py`
- Create: `src/openbench/visualization/__init__.py`
- Create: `src/openbench/runner/__init__.py`
- Create: `src/openbench/remote/__init__.py`
- Create: `src/openbench/gui/__init__.py`
- Create: `src/openbench/util/__init__.py`

- [ ] **Step 1: Create directory structure**

Run:
```bash
cd /Volumes/Data01/Openbench
mkdir -p src/openbench/cli
mkdir -p src/openbench/config
mkdir -p src/openbench/core
mkdir -p src/openbench/data/registry
mkdir -p src/openbench/visualization
mkdir -p src/openbench/runner
mkdir -p src/openbench/remote
mkdir -p src/openbench/gui
mkdir -p src/openbench/util
mkdir -p tests
```

- [ ] **Step 2: Write src/openbench/__init__.py**

```python
"""OpenBench: Open Source Land Surface Model Benchmarking System."""

__version__ = "3.0.0a1"
__author__ = "Zhongwang Wei, CoLM-SYSU"
__title__ = "OpenBench"
__description__ = "Land Surface Model Benchmarking System"
__license__ = "MIT"
```

- [ ] **Step 3: Write src/openbench/__main__.py**

```python
"""Allow running openbench as `python -m openbench`."""

from openbench.cli.main import cli

cli()
```

- [ ] **Step 4: Write src/openbench/cli/__init__.py**

```python
"""OpenBench command-line interface."""
```

- [ ] **Step 5: Write src/openbench/cli/main.py**

```python
"""OpenBench CLI entry point."""

import click

from openbench import __version__


@click.group()
@click.version_option(version=__version__, prog_name="openbench")
def cli():
    """OpenBench: Land Surface Model Benchmarking System."""


@cli.command()
def version():
    """Show version information."""
    click.echo(f"openbench {__version__}")
```

- [ ] **Step 6: Write placeholder __init__.py for all sub-packages**

Each of these files gets a one-line docstring:

`src/openbench/config/__init__.py`:
```python
"""Configuration loading, validation, and schema."""
```

`src/openbench/core/__init__.py`:
```python
"""Evaluation engine: metrics, scores, comparison, statistics."""
```

`src/openbench/data/__init__.py`:
```python
"""Data pipeline: preprocessing, caching, I/O."""
```

`src/openbench/data/registry/__init__.py`:
```python
"""Dataset registry and model profiles."""
```

`src/openbench/visualization/__init__.py`:
```python
"""Plotting and figure generation."""
```

`src/openbench/runner/__init__.py`:
```python
"""Local and remote evaluation runners."""
```

`src/openbench/remote/__init__.py`:
```python
"""SSH infrastructure for remote execution (requires openbench[remote])."""
```

`src/openbench/gui/__init__.py`:
```python
"""Wizard GUI (requires openbench[gui]).

This package requires PySide6. If not installed, importing submodules
will raise ImportError with an installation hint.
"""


def _check_gui_deps():
    """Check that GUI dependencies are available."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        raise ImportError(
            "GUI requires PySide6. Install with: pip install 'openbench[gui]'"
        ) from None
```

`src/openbench/util/__init__.py`:
```python
"""Shared utilities: logging, parallelism, memory, progress, reports."""
```

- [ ] **Step 7: Write tests/__init__.py**

```python
"""OpenBench test suite."""
```

- [ ] **Step 8: Commit skeleton**

Run:
```bash
cd /Volumes/Data01/Openbench
git add src/ tests/__init__.py
git commit -m "feat: create openbench package skeleton with src layout"
```

Expected: Clean commit with all placeholder files.

---

### Task 4: Write Smoke Tests

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write smoke tests**

```python
"""Smoke tests to verify package structure and CLI entry point."""


def test_import_openbench():
    """Verify that the openbench package can be imported."""
    import openbench

    assert openbench.__version__ == "3.0.0a1"
    assert openbench.__title__ == "OpenBench"


def test_import_subpackages():
    """Verify that all sub-packages can be imported."""
    import openbench.config
    import openbench.core
    import openbench.data
    import openbench.data.registry
    import openbench.visualization
    import openbench.runner
    import openbench.remote
    import openbench.util


def test_cli_entry_point():
    """Verify that the CLI group is callable."""
    from click.testing import CliRunner

    from openbench.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "3.0.0a1" in result.output


def test_cli_help():
    """Verify that --help works."""
    from click.testing import CliRunner

    from openbench.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "OpenBench" in result.output


def test_cli_version_option():
    """Verify that --version works."""
    from click.testing import CliRunner

    from openbench.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "3.0.0a1" in result.output


def test_gui_import_guard():
    """Verify that GUI import guard gives a helpful error when PySide6 is missing."""
    from openbench.gui import _check_gui_deps

    # This test passes if PySide6 IS installed (no error),
    # or if PySide6 is NOT installed (ImportError with hint).
    try:
        _check_gui_deps()
    except ImportError as e:
        assert "openbench[gui]" in str(e)
```

- [ ] **Step 2: Install package in dev mode and run tests**

Run:
```bash
cd /Volumes/Data01/Openbench
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```

Expected:
```
tests/test_smoke.py::test_import_openbench PASSED
tests/test_smoke.py::test_import_subpackages PASSED
tests/test_smoke.py::test_cli_entry_point PASSED
tests/test_smoke.py::test_cli_help PASSED
tests/test_smoke.py::test_cli_version_option PASSED
tests/test_smoke.py::test_gui_import_guard PASSED
```

- [ ] **Step 3: Verify CLI works from command line**

Run:
```bash
openbench version
openbench --help
openbench --version
```

Expected:
```
openbench 3.0.0a1
```

- [ ] **Step 4: Commit tests**

Run:
```bash
cd /Volumes/Data01/Openbench
git add tests/test_smoke.py
git commit -m "test: add smoke tests for package import and CLI entry point"
```

---

### Task 5: Add CLI Stub Commands

**Files:**
- Modify: `src/openbench/cli/main.py`
- Create: `src/openbench/cli/run.py`
- Create: `src/openbench/cli/check.py`
- Create: `src/openbench/cli/data.py`
- Create: `src/openbench/cli/model.py`
- Create: `src/openbench/cli/migrate.py`
- Create: `src/openbench/cli/init_cmd.py`
- Create: `src/openbench/cli/gui.py`
- Create: `tests/test_cli_stubs.py`

- [ ] **Step 1: Write tests for CLI stub commands**

Create `tests/test_cli_stubs.py`:

```python
"""Test that all CLI commands are registered and show help."""

from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()


def test_run_help():
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run evaluation" in result.output


def test_check_help():
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0
    assert "Validate" in result.output


def test_data_list_help():
    result = runner.invoke(cli, ["data", "list", "--help"])
    assert result.exit_code == 0


def test_data_download_help():
    result = runner.invoke(cli, ["data", "download", "--help"])
    assert result.exit_code == 0


def test_data_status_help():
    result = runner.invoke(cli, ["data", "status", "--help"])
    assert result.exit_code == 0


def test_data_path_help():
    result = runner.invoke(cli, ["data", "path", "--help"])
    assert result.exit_code == 0


def test_data_optimize_help():
    result = runner.invoke(cli, ["data", "optimize", "--help"])
    assert result.exit_code == 0


def test_model_list_help():
    result = runner.invoke(cli, ["model", "list", "--help"])
    assert result.exit_code == 0


def test_model_show_help():
    result = runner.invoke(cli, ["model", "show", "--help"])
    assert result.exit_code == 0


def test_model_create_help():
    result = runner.invoke(cli, ["model", "create", "--help"])
    assert result.exit_code == 0


def test_migrate_help():
    result = runner.invoke(cli, ["migrate", "--help"])
    assert result.exit_code == 0


def test_init_help():
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0


def test_gui_help():
    result = runner.invoke(cli, ["gui", "--help"])
    assert result.exit_code == 0


def test_all_commands_registered():
    """Verify all expected commands are in the CLI group."""
    command_names = set(cli.commands.keys())
    expected = {"run", "check", "data", "model", "migrate", "init", "gui", "version"}
    assert expected == command_names, f"Missing: {expected - command_names}, Extra: {command_names - expected}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Volumes/Data01/Openbench
pytest tests/test_cli_stubs.py -v
```

Expected: All FAIL (commands not registered).

- [ ] **Step 3: Write src/openbench/cli/run.py**

```python
"""openbench run command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Check only, don't execute.")
@click.option("--cores", type=int, default=None, help="Override number of CPU cores.")
@click.option("--variables", multiple=True, help="Run only specified variables.")
@click.option("--remote", default=None, help="Remote host or saved profile name.")
def run(config, dry_run, cores, variables, remote):
    """Run evaluation from a config file."""
    click.echo(f"Not yet implemented. Config: {config}")
```

- [ ] **Step 4: Write src/openbench/cli/check.py**

```python
"""openbench check command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
def check(config):
    """Validate config file and check data availability."""
    click.echo(f"Not yet implemented. Config: {config}")
```

- [ ] **Step 5: Write src/openbench/cli/data.py**

```python
"""openbench data commands."""

import click


@click.group()
def data():
    """Manage reference datasets."""


@data.command("list")
def list_datasets():
    """List all available reference datasets."""
    click.echo("Not yet implemented.")


@data.command()
@click.argument("names", nargs=-1, required=True)
def download(names):
    """Download reference datasets by name."""
    click.echo(f"Not yet implemented. Datasets: {', '.join(names)}")


@data.command()
def status():
    """Show local dataset cache status."""
    click.echo("Not yet implemented.")


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset."""
    click.echo(f"Not yet implemented. Dataset: {name}")


@data.command()
@click.argument("name")
def optimize(name):
    """Convert dataset to zarr for faster reads."""
    click.echo(f"Not yet implemented. Dataset: {name}")
```

- [ ] **Step 6: Write src/openbench/cli/model.py**

```python
"""openbench model commands."""

import click


@click.group()
def model():
    """Manage model profiles."""


@model.command("list")
def list_models():
    """List all available model profiles."""
    click.echo("Not yet implemented.")


@model.command()
@click.argument("name")
def show(name):
    """Show variable mappings for a model profile."""
    click.echo(f"Not yet implemented. Model: {name}")


@model.command()
def create():
    """Interactively create a new model profile."""
    click.echo("Not yet implemented.")
```

- [ ] **Step 7: Write src/openbench/cli/migrate.py**

```python
"""openbench migrate command."""

import click


@click.command()
@click.argument("old_config", type=click.Path(exists=True))
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def migrate(old_config, output):
    """Convert old JSON/NML config to unified YAML."""
    click.echo(f"Not yet implemented. Input: {old_config}, Output: {output}")
```

- [ ] **Step 8: Write src/openbench/cli/init_cmd.py**

```python
"""openbench init command."""

import click


@click.command("init")
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def init_cmd(output):
    """Interactively generate an openbench.yaml config file."""
    click.echo(f"Not yet implemented. Output: {output}")
```

- [ ] **Step 9: Write src/openbench/cli/gui.py**

```python
"""openbench gui command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option("--remote", is_flag=True, help="Start in remote mode.")
def gui(config, remote):
    """Launch the OpenBench graphical interface."""
    try:
        from openbench.gui import _check_gui_deps

        _check_gui_deps()
    except ImportError as e:
        raise click.ClickException(str(e))

    click.echo(f"Not yet implemented. Config: {config}")
```

- [ ] **Step 10: Update src/openbench/cli/main.py to register all commands**

Replace the entire file:

```python
"""OpenBench CLI entry point."""

import click

from openbench import __version__


@click.group()
@click.version_option(version=__version__, prog_name="openbench")
def cli():
    """OpenBench: Land Surface Model Benchmarking System."""


@cli.command()
def version():
    """Show version information."""
    click.echo(f"openbench {__version__}")


# Register sub-commands
from openbench.cli.check import check  # noqa: E402
from openbench.cli.data import data  # noqa: E402
from openbench.cli.gui import gui  # noqa: E402
from openbench.cli.init_cmd import init_cmd  # noqa: E402
from openbench.cli.migrate import migrate  # noqa: E402
from openbench.cli.model import model  # noqa: E402
from openbench.cli.run import run  # noqa: E402

cli.add_command(run)
cli.add_command(check)
cli.add_command(data)
cli.add_command(model)
cli.add_command(migrate)
cli.add_command(init_cmd)
cli.add_command(gui)
```

- [ ] **Step 11: Run all tests**

Run:
```bash
cd /Volumes/Data01/Openbench
pytest tests/ -v
```

Expected: All tests PASS (smoke + CLI stubs).

- [ ] **Step 12: Verify CLI from command line**

Run:
```bash
openbench --help
openbench data --help
openbench model --help
```

Expected: All commands listed with descriptions.

- [ ] **Step 13: Commit**

Run:
```bash
cd /Volumes/Data01/Openbench
git add src/openbench/cli/ tests/test_cli_stubs.py
git commit -m "feat: add CLI stub commands for all openbench subcommands"
```

---

### Task 6: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short
```

- [ ] **Step 2: Verify lint passes locally**

Run:
```bash
cd /Volumes/Data01/Openbench
ruff check src/ tests/
ruff format --check src/ tests/
```

Expected: No errors. If formatting issues, run `ruff format src/ tests/` first.

- [ ] **Step 3: Commit CI**

Run:
```bash
cd /Volumes/Data01/Openbench
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for lint and test matrix"
```

---

### Task 7: Add README and LICENSE

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write README.md**

```markdown
# OpenBench

> Open Source Land Surface Model Benchmarking System

OpenBench is a fully automated, cross-platform framework for benchmarking land surface models (LSMs) against curated reference datasets with consistent metrics, visualizations, and reports.

## Installation

```bash
# Core (CLI-only, HPC-friendly)
pip install openbench

# With GUI
pip install "openbench[gui]"

# With remote SSH execution
pip install "openbench[remote]"

# Everything
pip install "openbench[all]"
```

Also available via conda:

```bash
conda install -c conda-forge openbench
```

## Quick Start

```bash
# Generate a config interactively
openbench init

# Validate your config
openbench check openbench.yaml

# Run evaluation
openbench run openbench.yaml

# Launch GUI
openbench gui
```

## Manage Data

```bash
# List available reference datasets
openbench data list

# Download datasets
openbench data download GLEAM_v4.2a FLUXCOM

# List model profiles
openbench model list
```

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependency list

## License

MIT License. See [LICENSE](LICENSE) for details.
```

- [ ] **Step 2: Write LICENSE**

```
MIT License

Copyright (c) 2023-2026 Zhongwang Wei, CoLM-SYSU

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Write CHANGELOG.md**

```markdown
# Changelog

## [3.0.0a1] - Unreleased

### Added
- Unified package structure (`src/openbench/`)
- CLI entry point with subcommands: `run`, `check`, `data`, `model`, `migrate`, `init`, `gui`, `version`
- Optional extras: `[gui]`, `[remote]`, `[report]`, `[all]`
- GitHub Actions CI (lint + test matrix)

### Changed
- Merged `OpenBench-wei` and `openbench-wizard` into single package
- Switched build system to `hatchling`
- Configuration format: YAML only (JSON and Fortran NML deprecated)
```

- [ ] **Step 4: Commit**

Run:
```bash
cd /Volumes/Data01/Openbench
git add README.md LICENSE CHANGELOG.md
git commit -m "docs: add README, LICENSE, and CHANGELOG"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Clean install test**

Run:
```bash
cd /Volumes/Data01/Openbench
pip install -e ".[dev]"
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Verify full CLI**

Run:
```bash
openbench --version
openbench --help
openbench run --help
openbench check --help
openbench data --help
openbench data list
openbench model --help
openbench model list
openbench migrate --help
openbench init --help
openbench gui --help
```

Expected: All commands respond with help text or "Not yet implemented." messages.

- [ ] **Step 3: Verify package builds**

Run:
```bash
cd /Volumes/Data01/Openbench
pip install build
python -m build
ls dist/
```

Expected: `dist/openbench-3.0.0a1.tar.gz` and `dist/openbench-3.0.0a1-py3-none-any.whl` created.

- [ ] **Step 4: Run lint**

Run:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

Expected: No errors.

- [ ] **Step 5: Final commit (if any fixes needed)**

Run:
```bash
cd /Volumes/Data01/Openbench
git status
# If changes: git add ... && git commit -m "fix: address lint/test issues"
```

- [ ] **Step 6: Tag the milestone**

Run:
```bash
cd /Volumes/Data01/Openbench
git tag -a v3.0.0a1 -m "Sub-project 1 complete: package skeleton with CLI stubs and CI"
```
