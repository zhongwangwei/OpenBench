# Changelog

## [3.0.0b2] - 2026-06-09

Bug-fix and hardening release over 3.0.0b1 (deep code review + full cross-platform CI).

### Fixed
- Target diagram plotted total RMSE on the uRMSD axis; now passes centered CRMSD so
  points satisfy RMSD² = bias² + uRMSD² (with an invariant guard)
- `absolute_percent_bias` normalizes by `|Σo|` so a negative observed sum no longer yields a negative APB
- ANOVA default `analysis_type` corrected to `oneway` (was the non-accepted `one-way`)
- `br2` uses `|slope|` (Krause 2005), keeping it in `[0, r²]` for negative slopes
- SMPI grid path applies area weights and uses a consistent bootstrap dimension
- Taylor grid summary computes std/correlation/CRMSD over one pairwise finite mask
- NetCDF writes are serialized with a lock (netCDF4/HDF5 is not thread-safe) — fixes a segfault
- Registry catalog writes hold a cross-process lock; `delete_reference` writes a tombstone
- Config rejects non-mapping `project`/`evaluation`/`simulation` sections with a clean error
- Cross-platform path handling (POSIX paths in catalogs/configs; Windows file reads)

### Changed
- Version is a single source of truth (`__init__.__version__`, hatchling dynamic)
- The full test suite (84 files) now runs on CI across Linux/macOS/Windows

### Added
- 30-minute (half-hour) time-resolution detection
- Expanded reference dataset metadata; `CITATION.cff`

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
