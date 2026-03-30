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
