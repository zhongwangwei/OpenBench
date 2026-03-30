# OpenBench

> Open Source Land Surface Model Benchmarking System

OpenBench is a fully automated, cross-platform framework for benchmarking land surface models (LSMs) against curated reference datasets with consistent metrics, visualizations, and reports.

## Key Features

- **Unified configuration** — single `openbench.yaml` replaces the 4-6 file setup of earlier versions
- **Data registry** — 66 reference datasets (grid and station) ready to list and download
- **Model profiles** — built-in variable mappings for CoLM2024, CLM5, and ERA5-Land
- **CLI** — `openbench run`, `check`, `init`, `data`, `model`, `migrate`, and more
- **GUI wizard** — interactive configuration and run management (optional `[gui]` extra)
- **Remote execution** — SSH-based job submission to HPC clusters (optional `[remote]` extra)
- **Migration tool** — convert existing JSON / Fortran NML configs to the new YAML format

## Installation

```bash
# Core install (CLI-only, HPC-friendly)
pip install openbench

# With GUI
pip install "openbench[gui]"

# With remote SSH execution
pip install "openbench[remote]"

# With PDF report generation
pip install "openbench[report]"

# Everything
pip install "openbench[all]"
```

Using `uv`:

```bash
uv pip install "openbench[all]"
```

Using conda (once available on conda-forge):

```bash
conda install -c conda-forge openbench
```

## Quick Start

```bash
# 1. Generate a config interactively
openbench init

# 2. Validate the config and check data availability
openbench check openbench.yaml

# 3. Run the evaluation
openbench run openbench.yaml
```

## CLI Reference

| Command | Description |
|---|---|
| `openbench run CONFIG` | Run evaluation from a YAML config file |
| `openbench check CONFIG` | Validate config and check reference data availability |
| `openbench init` | Interactively generate an `openbench.yaml` config |
| `openbench data list` | List all available reference datasets |
| `openbench data download DATASET...` | Download one or more reference datasets |
| `openbench model list` | List built-in model profiles |
| `openbench model show MODEL` | Show variable mappings for a model profile |
| `openbench migrate OLD_CONFIG` | Convert a JSON or Fortran NML config to YAML |
| `openbench gui` | Launch the GUI wizard (requires `[gui]` extra) |
| `openbench version` | Print version string |

## Minimal Configuration Example

```yaml
# openbench.yaml
project:
  name: my_run
  output_dir: ./output
  years: [2010, 2014]
  lat_range: [-90, 90]
  lon_range: [-180, 180]

evaluation:
  variables:
    - Evapotranspiration
    - SoilMoisture

simulations:
  CoLM2024_test:
    model: CoLM2024
    root_dir: /data/simulations/colm2024

reference:
  Evapotranspiration: GLEAM_v4.2a
  SoilMoisture: ESA_CCI_SM_v9.0
```

Advanced features supported in the config:

- `_defaults` block — merge shared settings into simulation entries
- `!include path/to/partial.yaml` — split large configs across files
- `time_alignment: intersection | per_pair | strict` — control how model/reference time windows are matched

## Data Management

```bash
# Browse all 66 reference datasets
openbench data list

# Download specific datasets
openbench data download GLEAM_v4.2a ESA_CCI_SM_v9.0

# Example output of `openbench data list`:
# Name                           Category     Type   Res      Years          Variables
# ────────────────────────────────────────────────────────────────────────────────────
# GLEAM_v4.2a                    Water        grid   0.25°    1980-2023      8
# ESA_CCI_SM_v9.0                Water        grid   0.25°    1978-2023      1
# FLUXCOM                        Energy       grid   0.5°     2001-2019      3
# ...                            ...          ...    ...      ...            ...
# Total: 66 datasets
```

## Model Profiles

```bash
# List built-in profiles
openbench model list

# Show variable mappings for a profile
openbench model show CoLM2024

# Example output of `openbench model list`:
# Name                 Type   Res      Variables
# ────────────────────────────────────────────────────────────
# CLM5                 grid   1.25°    12
# CoLM2024             grid   0.5°     10
# ERA5-Land            grid   0.1°     12
#
# Total: 3 model profiles
```

When `simulations.<name>.model` matches a known profile, variable file-name patterns and unit conversions are applied automatically. Fields in the simulation entry override the profile.

## Migrating from Earlier Versions

```bash
# Convert an existing JSON config
openbench migrate old_config.json -o openbench.yaml

# Convert a Fortran NML config
openbench migrate main.nml -o openbench.yaml
```

Evaluation results produced by v3.0 are numerically identical to v2.0.

## Requirements

- Python 3.10 or newer
- Core dependencies: xarray, numpy, scipy, pandas, netCDF4, matplotlib, cartopy, dask, PyYAML, click
- See `pyproject.toml` for the full pinned dependency list

## Links

- Source: <https://github.com/CoLM-SYSU/OpenBench>
- Documentation: <https://openbench.readthedocs.io>
- Issues: <https://github.com/CoLM-SYSU/OpenBench/issues>

## License

MIT License. See [LICENSE](LICENSE) for details.
