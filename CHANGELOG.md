# Changelog

## [3.0.0b1] - 2026-06-07

First public beta of the 3.0 line. APIs and config schema may still change
before the 3.0.0 final release.

### Added
- Unified package structure (`src/openbench/`) merging OpenBench-wei and openbench-wizard
- Single YAML configuration file (`openbench.yaml`) replacing 4-6 file setup
- Data registry with 101 reference datasets (69 grid + 32 station)
- 22 model profiles (CoLM2024, CLM5, NoahMP5, ERA5-Land, …)
- Bundled IGBP / PFT / Köppen classification masks for group-by analysis
- CLI commands: `run`, `check`, `init`, `ref`, `sim`, `model`, `migrate`, `cache`, `gui`, `version`
- Interactive config generator (`openbench init`)
- Config migration tool (`openbench migrate`) for old JSON/NML formats
- `_defaults` merge support for simulation configs
- `!include` tag support in YAML configs
- Three time alignment strategies: intersection, per_pair, strict
- Config adapter bridging new and legacy evaluation engine formats
- SSH remote execution infrastructure (requires `openbench[remote]`)
- GUI wizard (requires `openbench[gui]`)
- Optional extras: `[gui]`, `[remote]`, `[report]`, `[all]`
- GitHub Actions CI (lint + test matrix)
- 70+ tests covering config, registry, metrics, scores, CLI

### Changed
- Build system: hatchling (was setuptools/manual)
- Config format: YAML only (JSON and Fortran NML deprecated)
- Package name: `colm-openbench` on PyPI
- Python requirement: >=3.10

### Migration
- Use `openbench migrate old-config.json -o openbench.yaml` to convert existing configs
- Evaluation results are numerically identical to v2.0
