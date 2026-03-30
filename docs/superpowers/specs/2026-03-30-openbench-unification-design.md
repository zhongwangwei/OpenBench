# OpenBench Unification Design Spec

**Date:** 2026-03-30
**Status:** Draft
**Author:** zhongwangwei + Claude

## 1. Overview

Merge `OpenBench-wei` (core benchmarking framework, ~53k lines) and `openbench-wizard` (GUI/CLI frontend, ~23k lines) into a single unified Python package `openbench`. The goals are:

1. **Simplify installation** — optional extras (`[gui]`, `[remote]`, `[all]`), publish to PyPI + conda-forge
2. **Simplify configuration** — single YAML file with smart defaults, data registry, model profiles
3. **Improve performance** — zarr caching, variable-level parallelism, incremental re-computation
4. **Support multiple usage modes** — CLI (`openbench run`), GUI (`openbench gui`), Python API
5. **Support multiple installation methods** — `pip`, `uv`, `conda`

## 2. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Package name | `openbench` | Established brand, short, memorable |
| Config format | YAML only | Drop JSON (deprecated) and Fortran NML (deprecated); provide `openbench migrate` |
| Python version | >= 3.10 | HPC compatibility |
| Visualization | Keep as-is | 25+ modules work, not worth rewriting |
| Testing | Incremental | Critical paths first, expand later |
| Remote SSH | Keep and improve | Critical for HPC users |
| Time alignment | 3 strategies (`intersection`, `per_pair`, `strict`) | Flexible multi-model comparison |

## 3. Package Structure

```
openbench/                          # Repository root
├── src/
│   └── openbench/
│       ├── __init__.py             # Version, public API
│       ├── __main__.py             # python -m openbench
│       ├── cli/                    # CLI layer (click-based)
│       │   ├── __init__.py
│       │   ├── main.py             # Entry point, command group
│       │   ├── run.py              # openbench run
│       │   ├── check.py            # openbench check
│       │   ├── data.py             # openbench data list/download/status/path/optimize
│       │   ├── model.py            # openbench model list/show/create
│       │   ├── migrate.py          # openbench migrate (JSON/NML -> YAML)
│       │   ├── init.py             # openbench init (interactive config generator)
│       │   └── gui.py              # openbench gui (lazy-imports PySide6)
│       ├── config/                 # Configuration system
│       │   ├── __init__.py
│       │   ├── schema.py           # Dataclass-based config schema with defaults
│       │   ├── loader.py           # YAML-only loading + validation
│       │   └── migration.py        # JSON/NML -> YAML converter
│       ├── data/                   # Data pipeline
│       │   ├── __init__.py
│       │   ├── pipeline.py         # DataPipeline orchestrator
│       │   ├── cache.py            # CacheSystem (memory + disk + zarr)
│       │   ├── climatology.py      # Climatology processing
│       │   ├── preprocessing.py    # File checking, preprocessing
│       │   ├── regrid/             # Regridding methods (as-is)
│       │   ├── io.py               # File I/O, unit conversion, time utils
│       │   └── registry/           # Dataset & model registry
│       │       ├── __init__.py
│       │       ├── manager.py      # RegistryManager class
│       │       ├── catalog.yaml    # Dataset catalog (70+ entries)
│       │       ├── references/     # Reference dataset descriptors (70+ YAML files)
│       │       │   ├── GLEAM_v4.2a.yaml
│       │       │   ├── FLUXCOM.yaml
│       │       │   └── ...
│       │       └── models/         # Model profile descriptors
│       │           ├── CoLM2014.yaml
│       │           ├── CoLM2024.yaml
│       │           ├── CLM5.yaml
│       │           ├── NOAH.yaml
│       │           ├── JULES.yaml
│       │           └── ERA5-Land.yaml
│       ├── core/                   # Evaluation engine
│       │   ├── __init__.py
│       │   ├── metrics.py          # 25+ metrics (as-is)
│       │   ├── scores.py           # Normalized scores (as-is)
│       │   ├── evaluation.py       # Grid + station evaluation engines
│       │   ├── comparison.py       # Multi-model comparison
│       │   ├── statistics/         # 20+ stat modules (as-is)
│       │   └── groupby.py          # Land-cover + climate zone grouping
│       ├── visualization/          # Plotting (as-is from OpenBench-wei)
│       │   ├── __init__.py
│       │   ├── Fig_*.py            # All 25+ figure modules unchanged
│       │   ├── toolbox.py
│       │   └── cmaps/
│       ├── runner/                 # Execution engines
│       │   ├── __init__.py
│       │   ├── local.py            # Local evaluation runner
│       │   └── remote.py           # SSH runner (requires openbench[remote])
│       ├── remote/                 # SSH infrastructure (optional)
│       │   ├── __init__.py
│       │   ├── ssh.py              # SSHManager
│       │   ├── credentials.py      # Encrypted credential storage
│       │   ├── connections.py      # Connection profiles
│       │   ├── sync.py             # File sync engine
│       │   └── storage.py          # Abstract + Local/Remote storage
│       ├── gui/                    # Wizard GUI (requires openbench[gui])
│       │   ├── __init__.py
│       │   ├── app.py              # Application bootstrap
│       │   ├── main_window.py
│       │   ├── controller.py       # Wizard controller
│       │   ├── pages/              # 7 wizard pages (simplified from 11)
│       │   │   ├── page_environment.py   # Local/remote setup
│       │   │   ├── page_project.py       # Name, years, area
│       │   │   ├── page_variables.py     # Variables + reference selection
│       │   │   ├── page_simulation.py    # Models + paths
│       │   │   ├── page_options.py       # Metrics, scores, comparison, stats
│       │   │   ├── page_preview.py       # YAML preview + check
│       │   │   └── page_run.py           # Progress + logs
│       │   ├── widgets/
│       │   ├── dialogs/
│       │   └── styles/
│       └── util/                   # Shared utilities
│           ├── __init__.py
│           ├── logging.py
│           ├── parallel.py
│           ├── memory.py
│           ├── progress.py
│           ├── report.py
│           └── exceptions.py
├── tests/
│   ├── test_config/
│   ├── test_core/
│   ├── test_data/
│   ├── test_runner/
│   ├── test_remote/
│   └── test_gui/
├── docs/
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

## 4. Installation & Packaging

### 4.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "openbench"
version = "3.0.0"
description = "Open Source Land Surface Model Benchmarking System"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "CoLM-SYSU" }]
keywords = ["land surface model", "benchmarking", "evaluation", "climate"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
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
dev = ["pytest>=7.0", "pytest-cov", "ruff"]
all = ["openbench[gui,remote,report]"]

[project.scripts]
openbench = "openbench.cli.main:cli"

[project.urls]
Homepage = "https://github.com/CoLM-SYSU/OpenBench"
Documentation = "https://openbench.readthedocs.io"
Repository = "https://github.com/CoLM-SYSU/OpenBench"
```

### 4.2 Installation Commands

```bash
# Core only (HPC, CLI-only)
pip install openbench
uv pip install openbench
conda install -c conda-forge openbench

# With GUI
pip install "openbench[gui]"

# With remote SSH execution
pip install "openbench[remote]"

# Everything
pip install "openbench[all]"

# Development
pip install -e ".[dev,all]"
```

### 4.3 User-Level Directory

Managed via `platformdirs`:

```
~/.openbench/                       # or platform-appropriate location
├── config.yaml                     # Global preferences
│   # data_root: /shared/data/openbench   (optional: HPC shared dir)
├── models/                         # User-defined model profiles
│   └── MyModel.yaml
├── datasets/                       # Downloaded reference data cache
│   ├── GLEAM_v4.2a/
│   └── FLUXCOM/
└── connections.yaml                # Saved SSH profiles (when using [remote])
```

## 5. Configuration System

### 5.1 Unified openbench.yaml

Single file replaces 4-6 files. Smart defaults minimize required fields.

**Minimal config:**

```yaml
project:
  name: my-evaluation
  output_dir: ./output
  years: [2004, 2010]

evaluation:
  variables: [Evapotranspiration, Latent_Heat, GPP]

reference:
  Evapotranspiration: GLEAM_v4.2a
  Latent_Heat: FLUXCOM
  GPP: FLUXCOM

simulation:
  CoLM2024:
    model: CoLM2024
    root_dir: /data/CoLM2024/output
```

**Full config (all options):**

```yaml
project:
  name: multi-model-comparison
  output_dir: ./output
  years: [2004, 2010]
  min_year_threshold: 3
  lat_range: [-60, 90]
  lon_range: [-180, 180]

evaluation:
  variables:
    - Evapotranspiration
    - Latent_Heat
    - Sensible_Heat
    - GPP
    - Net_Ecosystem_Exchange
    - Soil_Moisture
    - Runoff

reference:
  Evapotranspiration: GLEAM_v4.2a
  Latent_Heat: FLUXCOM
  Sensible_Heat: FLUXCOM
  GPP: FLUXCOM
  Net_Ecosystem_Exchange: FLUXCOM
  Soil_Moisture: SMAP
  Runoff: GRUN

simulation:
  _defaults:
    data_type: grid
    grid_res: 0.5
    tim_res: Month

  CoLM2014:
    model: CoLM2014
    root_dir: /data/CoLM2014/output

  CoLM2024:
    model: CoLM2024
    root_dir: /data/CoLM2024/output

  CLM5:
    model: CLM5
    root_dir: /data/CLM5/output
    variables:
      GPP: { varname: FPSN }           # Override for this model

metrics: [bias, RMSE, ubRMSE, correlation, mean_absolute_error]
scores: [nBiasScore, nRMSEScore, nPhaseScore, nSpatialScore]

comparison:
  enabled: true

statistics:
  enabled: false

options:
  num_cores: 48                         # default: auto-detect
  time_alignment: intersection          # intersection | per_pair | strict
  unified_mask: true
  generate_report: true
  IGBP_groupby: false
  PFT_groupby: false
  climate_zone_groupby: false
```

### 5.2 Config Schema (dataclass-based)

Config is validated at load time using Python dataclasses. Each field has a type, default, and description. Validation errors produce clear, actionable messages.

Example validation output:

```
$ openbench check openbench.yaml

Config validation:
  ✓ YAML syntax valid
  ✓ Schema validation passed
  ✓ Year range [2004, 2010] valid

Reference data:
  ✓ GLEAM_v4.2a -> ~/.openbench/datasets/GLEAM_v4.2a/ (2.1 GB)
  ✓ FLUXCOM -> /shared/data/openbench/FLUXCOM/ (shared)

Simulation data:
  ✓ CoLM2024 model profile loaded (built-in)
  ✓ /data/CoLM2024/output/ exists
  ✓ Variable ET found in files (2004-2010)

Time coverage:
  ✓ All datasets cover [2004, 2010]
  Time alignment: intersection -> effective range: 2004-2010

Metrics & scores:
  ✓ All requested metrics available

Result: 0 errors, 0 warnings. Ready to run.
```

### 5.3 Migration Tool

Converts old JSON/NML configs to new unified YAML:

```bash
$ openbench migrate nml/nml-json/main-LowRes.json -o openbench.yaml
Reading main config: nml/nml-json/main-LowRes.json
Reading ref config: nml/nml-json/ref-LowRes.json
  Resolving 40+ variable definition files...
Reading sim config: nml/nml-json/sim.json
Reading stats config: nml/nml-json/stats.yaml
Reading figure config: nml/nml-json/figlib.yaml

✓ Converted 6 config files -> openbench.yaml
✓ Resolved 12 reference sources to registry names
✓ Detected 2 simulation models -> matched to CoLM2014, CLM5
⚠ 1 unknown model -> inline config preserved, consider: openbench model create
```

### 5.4 Simulation `_defaults` Merge Behavior

When `_defaults` is present under `simulation`, each model entry inherits all `_defaults` fields. Model-level fields override `_defaults`. For nested keys like `variables`, the merge is **shallow per variable** — specifying a variable in the model replaces that variable's entire mapping, but other variables are inherited from `_defaults`.

```yaml
simulation:
  _defaults:
    data_type: grid
    variables:
      GPP: { varname: GPP, varunit: "gC m-2 day-1" }
      ET:  { varname: ET, varunit: "mm day-1" }

  CLM5:
    model: CLM5
    root_dir: /data/CLM5
    variables:
      GPP: { varname: FPSN }    # Replaces GPP entirely; ET inherited from _defaults
```

### 5.5 !include Support

Power users can split configs:

```yaml
reference: !include ref.yaml
simulation: !include sim.yaml
# or load all YAML files from a directory:
simulation: !include sim/*.yaml
```

## 6. Data Registry & Model Profile System

### 6.1 Reference Dataset Registry

70+ curated reference datasets ship as YAML descriptors in the package. Each descriptor contains:

```yaml
# src/openbench/data/registry/references/GLEAM_v4.2a.yaml
name: GLEAM_v4.2a
description: "Global Land Evaporation Amsterdam Model v4.2a"
data_type: grid
grid_res: 0.25
tim_res: Month
data_groupby: Year
timezone: 0
years: [1980, 2023]
variables:
  Evapotranspiration:
    varname: E
    varunit: "mm day-1"
    prefix: "E_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Evapotranspiration/GLEAM_v4.2a"
```

### 6.2 Dataset Catalog

Master index with download URLs, sizes, and checksums:

```yaml
# src/openbench/data/registry/catalog.yaml
datasets:
  GLEAM_v4.2a:
    description: "Global Land Evaporation Amsterdam Model v4.2a"
    category: Water
    variables: [Evapotranspiration]
    spatial: { type: grid, resolution: 0.25 }
    temporal: { resolution: Month, range: [1980, 2023] }
    size: "2.1 GB"
    url: "https://data.example.org/openbench/GLEAM_v4.2a.tar.gz"
    sha256: "abc123..."
```

### 6.3 Model Profiles

Built-in profiles for common LSMs, user-extensible:

```yaml
# src/openbench/data/registry/models/CoLM2024.yaml
name: CoLM2024
description: "Common Land Model 2024"
data_type: grid
grid_res: 0.5
tim_res: Month
variables:
  Evapotranspiration: { varname: ET, varunit: "mm day-1", prefix: ET_, suffix: "" }
  Latent_Heat: { varname: Qle, varunit: "W m-2" }
  Sensible_Heat: { varname: Qh, varunit: "W m-2" }
  GPP: { varname: GPP, varunit: "gC m-2 day-1" }
  # ... all supported variables
```

### 6.4 Data Path Resolution

When a reference source is specified (e.g., `GLEAM_v4.2a`), resolution order:

1. `options.data_root` (HPC shared directory, if configured)
2. `~/.openbench/datasets/` (user downloaded cache)
3. Not found -> prompt `openbench data download GLEAM_v4.2a`

### 6.5 CLI Commands

```bash
# Dataset management
openbench data list                    # List all datasets with status
openbench data download GLEAM_v4.2a    # Download specific dataset
openbench data status                  # Show local cache status
openbench data path GLEAM_v4.2a        # Print local path
openbench data optimize GLEAM_v4.2a    # Convert to zarr for fast reads

# Model profile management
openbench model list                   # List all model profiles
openbench model show CoLM2024          # Show variable mappings
openbench model create                 # Interactive creation
```

## 7. CLI Design

### 7.1 Command Tree

```
openbench
├── run <config>                    # Run evaluation
│   ├── --dry-run                   #   Check only, don't execute
│   ├── --cores N                   #   Override core count
│   ├── --variables VAR...          #   Run subset of variables
│   └── --remote HOST|PROFILE       #   Execute on remote host
├── check <config>                  # Validate config + data
├── gui [config]                    # Launch GUI
│   └── --remote                    #   Start in remote mode
├── init                            # Interactive config generator
├── data                            # Dataset management
│   ├── list
│   ├── download <name...>
│   ├── status
│   ├── path <name>
│   └── optimize <name>
├── model                           # Model profile management
│   ├── list
│   ├── show <name>
│   └── create
├── migrate <old-config>            # Convert old config to new YAML
│   └── -o <output>
└── version                         # Version info
```

### 7.2 Interactive Config Generator

`openbench init` provides a step-by-step CLI wizard that mirrors the GUI flow:

1. Project name + output directory
2. Year range
3. Variable selection (categorized: Carbon/Water/Energy)
4. Reference source selection (from registry, shows download status)
5. Simulation model selection (from profiles or create new)
6. Simulation data paths
7. Options (comparison, statistics, metrics, scores)
8. Generate `openbench.yaml`

## 8. GUI Integration

### 8.1 Architecture Change

```
Before: GUI -> generate YAML -> spawn subprocess -> parse log output -> display progress
After:  GUI -> build config dict -> call core API -> native progress callbacks -> display progress
```

### 8.2 Page Simplification (11 -> 7)

| New Page | Replaces | Changes |
|---|---|---|
| PageEnvironment | PageRuntime | Simplified, same local/remote toggle |
| PageProject | PageGeneral | Name, years, area, basic options |
| PageVariables | PageEvaluation + PageRefData | Variable selection + reference source from registry in one view |
| PageSimulation | PageSimData | Model profile selection + paths, much simpler with model profiles |
| PageOptions | PageMetrics + PageScores + PageComparisons + PageStatistics | All optional settings in one tabbed page |
| PagePreview | PagePreview | YAML preview + `openbench check` integration |
| PageRun | PageRunMonitor | Direct API progress callbacks instead of log parsing |

### 8.3 Lazy Import

```bash
$ pip install openbench          # No GUI deps
$ openbench gui
Error: GUI requires PySide6. Install with: pip install "openbench[gui]"
```

## 9. Performance Optimization

### 9.1 Data I/O

| Optimization | Description | Expected Impact |
|---|---|---|
| zarr caching | `openbench data optimize` converts reference data to zarr | Read speed 5-10x |
| Pipeline data reuse | Same file read once for multiple variables | Reduces I/O by 30-50% |
| Lazy dask loading | Enforce lazy loading, remove premature `.load()` calls | Reduce memory |
| Smart time slicing | Only read data within effective time window | Skip unused years |

### 9.2 Parallel Computation

| Optimization | Description | Expected Impact |
|---|---|---|
| Variable-level parallelism | Different variables evaluated in parallel | 3-5x on multi-core |
| Adaptive scheduling | Small datasets serial, large datasets parallel | Reduce joblib overhead |
| Memory-aware workers | Limit workers based on available memory (psutil) | Prevent OOM |

### 9.3 Caching & Incremental Re-computation

| Cache Level | Contents | Invalidation |
|---|---|---|
| L1: Memory | Current run intermediate results | Run ends |
| L2: Disk | Metrics/scores results | Data or config change (content hash) |
| L3: zarr | Pre-processed reference data | Source data update |

Re-run skips unchanged computations:

```
$ openbench run openbench.yaml
Checking cache...
  ✓ Evapotranspiration cached (2 min ago), skipping
  ✓ Latent_Heat cached, skipping
  ~ GPP config changed, recalculating...
Done in 45s (full run: 8min)
```

### 9.4 Memory Management

Stream processing: load variable A -> evaluate A -> plot A -> release A -> load B -> ...

Peak memory: O(single variable) instead of O(all variables).

### 9.5 Visualization

- Only generate enabled plots (skip disabled ones)
- Figure generation in separate process pool (matplotlib is not thread-safe but multi-process safe)

## 10. Time Alignment for Multi-Model Comparison

Three strategies for handling different temporal coverage across simulations:

```yaml
options:
  time_alignment: intersection    # default
```

| Strategy | Behavior | Use Case |
|---|---|---|
| `intersection` | All datasets' common time range; guarantees comparability | Multi-model comparison (default) |
| `per_pair` | Each sim-ref pair uses its own overlap; different models may have different eval periods | Maximize data utilization per model |
| `strict` | All data must cover config `years` exactly; error if any gap | Strict benchmarking |

`openbench check` reports time coverage and effective range for the chosen strategy.

## 11. Remote Execution (openbench[remote])

Migrated from openbench-wizard with the same capabilities:

- SSH connection management with profiles
- Encrypted credential storage (Fernet cipher)
- Jump node / bastion host support
- SFTP file transfer with progress
- Real-time log streaming
- File sync engine with local caching

Key improvement: remote runner calls core API directly instead of parsing subprocess logs.

## 12. Testing Strategy

### Phase 1 (during restructuring): Critical paths

- Config loading + validation + schema
- Metrics calculations (numerical correctness)
- Scores calculations (numerical correctness)
- Data pipeline (coordinate standardization, time alignment)
- Registry resolution (reference + model profiles)
- Migration tool (JSON/NML -> YAML conversion)

### Phase 2 (post-restructuring): Expansion

- Runner (local + remote)
- CLI commands (integration tests)
- GUI (basic smoke tests)
- End-to-end with sample datasets

## 13. Sub-Project Decomposition

### Phase 1 -- Foundation (no functional changes)

**Sub-project 1: Package structure + pyproject.toml + CI**
- Create `src/openbench/` layout
- Configure `pyproject.toml` with extras
- Set up GitHub Actions CI (lint + test)
- Estimated scope: Small

**Sub-project 2: Configuration system**
- Implement dataclass-based schema
- YAML-only loader with validation
- Migration tool (JSON/NML -> YAML)
- `!include` support
- Estimated scope: Medium

**Sub-project 3: Data registry + Model profiles**
- RegistryManager class
- Convert 70+ variable definition files to reference descriptors
- Create built-in model profiles (CoLM2014, CoLM2024, CLM5, etc.)
- catalog.yaml with download metadata
- `openbench data` and `openbench model` CLI commands
- Estimated scope: Medium

### Phase 2 -- Core Migration (functional parity)

**Sub-project 4: Core engine migration**
- Move metrics, scores, evaluation, comparison, statistics, visualization
- Adapt imports to new package structure
- Integrate with new config system and registry
- Variable-level parallelism
- Stream processing for memory optimization
- Estimated scope: Large

**Sub-project 5: Runner migration**
- Local runner with direct API integration (not subprocess)
- Remote runner with SSH support
- Progress callback system
- Estimated scope: Medium

**Sub-project 6: CLI implementation**
- `openbench run/check/init/migrate/data/model/gui/version`
- Click-based command structure
- Interactive `openbench init` wizard
- Estimated scope: Medium

### Phase 3 -- Frontend + Quality

**Sub-project 7: GUI migration**
- Simplify from 11 pages to 7 pages
- Integrate with core API (remove log parsing)
- Registry-aware data source selection
- Model profile-aware simulation config
- Estimated scope: Large

**Sub-project 8: Test suite**
- Critical path tests (config, metrics, scores, pipeline, registry, migration)
- Integration tests for CLI
- Estimated scope: Medium

**Sub-project 9: Release**
- PyPI publishing workflow
- conda-forge recipe
- Documentation (README, user guide, API docs)
- CHANGELOG
- Estimated scope: Small

## 14. Migration Path for Existing Users

```bash
# Step 1: Install
pip install openbench

# Step 2: Migrate configs
openbench migrate nml/nml-json/main-LowRes.json -o openbench.yaml

# Step 3: Verify
openbench check openbench.yaml

# Step 4: Run
openbench run openbench.yaml
```

### Backward Compatibility Guarantees

- `openbench migrate` provides lossless conversion from old config formats
- Evaluation results (metrics, scores) are numerically identical to v2.0
- Output directory structure remains compatible
