# Changelog

## [3.0.0a1] - 2026-03-30

### Added
- Unified package structure (`src/openbench/`) merging OpenBench-wei and openbench-wizard
- Single YAML configuration file (`openbench.yaml`) replacing 4-6 file setup
- Data registry with 66 reference datasets (38 grid + 28 station)
- Model profiles for CoLM2024, CLM5, ERA5-Land
- CLI commands: `run`, `check`, `init`, `data`, `model`, `migrate`, `gui`, `version`
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
- Package name: `openbench` on PyPI
- Python requirement: >=3.10

### Migration
- Use `openbench migrate old-config.json -o openbench.yaml` to convert existing configs
- Evaluation results are numerically identical to v2.0
