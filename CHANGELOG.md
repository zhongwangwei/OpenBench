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
