# OpenBench

> Open Source Land Surface Model Benchmarking System

[![PyPI version](https://img.shields.io/pypi/v/colm-openbench?include_prereleases)](https://pypi.org/project/colm-openbench/)
[![Python versions](https://img.shields.io/pypi/pyversions/colm-openbench)](https://pypi.org/project/colm-openbench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![User's Guide](https://img.shields.io/badge/docs-User's%20Guide%20(PDF)-blue)](docs/manual/OpenBench_UsersGuide_EN.pdf)

OpenBench is a fully automated, cross-platform framework for benchmarking land
surface models (LSMs) against curated reference datasets. From a single YAML
config it preprocesses model output and references onto a common grid/time
window, computes a consistent metric and score suite, renders publication-ready
comparison figures, and assembles an HTML/PDF report — for any number of
variables, models, and references at once.

```bash
pip install colm-openbench
openbench smoke-test   # verify the installed package with bundled sample data
openbench init           # generate openbench.yaml interactively
openbench check openbench.yaml   # validate config + data availability
openbench run   openbench.yaml   # evaluate, plot, report
```

---

## Table of Contents

- [Why OpenBench](#why-openbench)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Metrics & Scores](#metrics--scores)
- [Comparison & Visualization](#comparison--visualization)
- [Classification Group-By (IGBP / PFT / Köppen)](#classification-group-by-igbp--pft--köppen)
- [Reference Data Management](#reference-data-management)
- [Model Profiles](#model-profiles)
- [Output Structure](#output-structure)
- [Performance Tuning](#performance-tuning)
- [Migrating from Earlier Versions](#migrating-from-earlier-versions)
- [Requirements](#requirements)
- [Citation](#citation)
- [Links](#links)
- [License](#license)

---

## Why OpenBench

Evaluating an LSM normally means hand-wiring regridding, time alignment, masking,
metric code, and plotting for every model/reference pair — and redoing it when a
file changes. OpenBench turns that into one declarative config:

- **One config, many comparisons.** Variables × models × references are evaluated
  as a full cartesian product, each pair kept in its own output so nothing
  clobbers anything else.
- **Consistent science.** The same regridding, masking, weighting, and metric
  definitions apply to every pair, so cross-model comparisons are fair by
  construction (cumulative unified masking enforces identical spatial coverage
  across models).
- **Grid *and* station** references are both first-class.
- **Incremental & resumable.** Results are cached by a content fingerprint of the
  config, code, and input files; an interrupted run safely resumes without
  recomputing finished work or producing partial summaries.
- **Reproducible.** Resolved configs, content fingerprints, and resumable caches
  make runs auditable and repeatable.

## Key Features

- **Unified configuration** — a single `openbench.yaml` replaces the 4–6 file
  setup of earlier versions, with `_defaults` merging and `!include` splitting.
- **Reference data registry** — **101** built-in reference datasets (grid and
  station) you can list, inspect, scan, and register.
- **Model profiles** — **22** built-in profiles (CoLM2024, CLM5, NoahMP5,
  JULES7, VIC5, ERA5-Land, GLDAS2, MATSIRO, CaMa-Flood, WRF, …) supplying
  variable name mappings and unit conversions automatically.
- **Rich science suite** — 25+ point-wise metrics, 8 normalized skill scores, and
  17 statistical analysis modules.
- **Comparison & visualization** — Taylor, Target, Portrait, heat-map, parallel
  coordinates, KDE, ridgeline, and more — generated automatically.
- **Classification group-by** — per-class aggregation by IGBP land cover, PFT, or
  Köppen climate zone; the masks ship with the package, so it works out of the box.
- **CLI-first** — `run`, `check`, `init`, `ref`, `sim`, `model`, `migrate`,
  `cache`, `smoke-test`, `gui`.
- **GUI wizard** — interactive configuration and run management (optional `[gui]`).
- **Remote execution** — SSH-based job submission to HPC clusters (optional
  `[remote]`).
- **Migration tool** — convert existing JSON / Fortran NML configs to YAML.

## Installation

Recommended clean environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install from PyPI:

```bash
# Default install — fully featured (CLI, statistics, plotting, PDF/HTML
# reports, legacy migration). Best for HPC/servers.
pip install colm-openbench

# Optional: graphical configuration wizard (PySide6/Qt)
pip install "colm-openbench[gui]"

# Optional: remote SSH execution (paramiko)
pip install "colm-openbench[remote]"

# Optional: both of the above
pip install "colm-openbench[all]"
```

Using `uv`:

```bash
uv pip install "colm-openbench[all]"
```

### conda / conda-forge

OpenBench ships a conda-forge recipe (`conda/meta.yaml`, `noarch`: sourced from
PyPI, with Cartopy/NetCDF/PROJ resolved by conda). Once published to
conda-forge, a single command installs everything (no local compilation):

```bash
conda install -c conda-forge colm-openbench
mamba install -c conda-forge colm-openbench   # or the faster mamba
```

> **Note:** the current release is a Beta pre-release (`3.0.0b1`) and is **not on
> conda-forge yet**. Until it is, use one of the two paths below.

If your platform does not have wheels for geospatial dependencies such as
Cartopy, install the scientific stack from conda-forge first, then install
OpenBench with pip:

```bash
mamba create -n openbench -c conda-forge python=3.12 cartopy netcdf4
mamba activate openbench
pip install colm-openbench
```

Or build a local conda package straight from the bundled recipe:

```bash
conda install -n base conda-build
conda build conda/
conda install -c local colm-openbench
```

Install from a GitHub checkout for development or local testing:

```bash
git clone https://github.com/CoLM-SYSU/OpenBench.git
cd OpenBench
python -m pip install -e ".[dev]"
```

> The IGBP / PFT / Köppen classification masks needed for group-by analysis are
> bundled in the package (compressed, ~1.1 MB total) — no extra download.

Verify an installation before preparing your own data:

```bash
openbench smoke-test
```

This unpacks a bundled `Initial_test` fixture and runs `openbench check` against
the original grid and station cases: three variables, two simulation cases, and
their paired reference sources. Use `openbench smoke-test --run` for a full
evaluation smoke run; the full run may let Cartopy download Natural Earth map
assets on a clean machine.

## Quick Start

```bash
# 1. Generate a config interactively (picks variables, references, models)
openbench init

# 2. Validate the config and check reference data availability
openbench check openbench.yaml

# 3. Run the evaluation (metrics + scores + comparison figures + report)
openbench run openbench.yaml

# Re-run after editing the config — only what changed is recomputed
openbench run openbench.yaml

# Force a full re-run, ignoring the incremental cache
openbench run openbench.yaml --force
```

Results land in `output/<project_name>/`; start with `reports/report.html`.

## How It Works

```
config.yaml
   │
   ▼
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────┐
│ Preprocess  │ → │  Evaluate    │ → │  Compare      │ → │  Report    │
│ regrid /    │   │  metrics &   │   │  cross-model  │   │  HTML/PDF  │
│ time-align /│   │  scores per  │   │  figures &    │   │            │
│ unified mask│   │  (var,sim,ref)│   │  statistics   │   │            │
└─────────────┘   └──────────────┘   └───────────────┘   └────────────┘
        every stage is incrementally cached and resumable
```

1. **Preprocess** — each model and reference is regridded to a common resolution,
   clipped to the shared time window, and (optionally) masked so all models share
   identical valid coverage.
2. **Evaluate** — for every `(variable, simulation, reference)` triple, OpenBench
   computes the configured metrics and scores (grid → per-cell NetCDF, station →
   per-station CSV).
3. **Compare** — when `comparison.enabled`, results across all pairs feed shared
   figures (Taylor, portrait, heat-map, …) and optional statistics.
4. **Report** — everything is assembled into an HTML report (PDF with the
   `[report]` extra).

## Configuration

A minimal config:

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
    - Latent_Heat

reference:
  Evapotranspiration: ET_Xu_etal_2025_LowRes
  Latent_Heat: FLUXCOM

simulation:
  CoLM2024_test:
    model: CoLM2024            # matches a built-in profile → auto var mapping
    root_dir: /data/simulations/colm2024
```

A fuller config showing per-simulation overrides, metric/score subsets, and
comparison/statistics toggles:

```yaml
project:
  name: multi-model
  output_dir: ./output
  years: [2000, 2020]
  grid_res: 0.5
  num_cores: 16
  time_alignment: per_pair      # intersection | per_pair | strict
  unified_mask: true            # identical valid coverage across models
  weight: area                  # area (cos-lat) | mass | none

evaluation:
  variables: [Evapotranspiration, Gross_Primary_Productivity, Latent_Heat]

reference:
  Evapotranspiration: ET_Xu_etal_2025_LowRes
  Gross_Primary_Productivity: FLUXCOM-X-BASE_LowRes
  Latent_Heat: FLUXCOM

simulation:
  CoLM2024:
    model: CoLM2024
    root_dir: /data/CoLM2024
    variables:
      Gross_Primary_Productivity:
        varname: GPP            # override the profile's file/variable name
  CLM5:
    model: CLM5
    root_dir: /data/CLM5
    variables:
      Gross_Primary_Productivity:
        varname: FPSN

metrics: [bias, RMSE, correlation, KGE]
scores:  [nBiasScore, nRMSEScore, Overall_Score]

comparison:
  enabled: true

statistics:
  enabled: true
  items: [Z_Score, ANOVA]
```

### Key `project` options

| Option | Values / default | Meaning |
|---|---|---|
| `years` | `[start, end]` | Evaluation time window |
| `lat_range` / `lon_range` | `[min, max]` | Spatial domain |
| `grid_res` | float (degrees) | Common grid the data is regridded to |
| `num_cores` | int / `null` | Parallel workers (`null`/`0` → all CPUs) |
| `time_alignment` | `intersection` (default) / `per_pair` / `strict` | How model & reference time windows are matched |
| `regrid_backend` | `openbench_conservative` (default) / `cdo_remapcon` / `xesmf_conservative` / `basic_interpolation` | Regridding engine |
| `unified_mask` | `true` (default) | Cumulative NaN mask across sims for cross-model fairness |
| `weight` | `area` (default when omitted) / `mass` / `none` | Spatial aggregation weighting |
| `IGBP_groupby` / `PFT_groupby` / `climate_zone_groupby` | `false` | Per-class aggregation (see below) |
| `generate_report` | `true` | Emit HTML/PDF report |

### Advanced config helpers

- `_defaults` block — merge shared settings into every `simulation` entry.
- `!include path/to/partial.yaml` — split large configs across files.
- Environment variables override YAML at runtime (handy for quick experiments) —
  see [Performance Tuning](#performance-tuning).

## CLI Reference

Run `openbench <command> --help` for full options on any command.

### Top-level commands

| Command | Description |
|---|---|
| `openbench run CONFIG` | Run evaluation, comparison, and report from a YAML config |
| `openbench check CONFIG` | Validate config and check reference/data availability |
| `openbench smoke-test` | Validate the installed package with bundled `Initial_test` sample data |
| `openbench init` | Interactively generate an `openbench.yaml` |
| `openbench ref ...` | Manage the reference dataset registry (see below) |
| `openbench model ...` | Manage model profiles (see below) |
| `openbench sim scan DIR` | Scan directories for simulation cases → config fragments |
| `openbench cache ...` | Inspect/clear runtime caches (see below) |
| `openbench migrate OLD` | Convert a JSON or Fortran NML config to YAML |
| `openbench gui` | Launch the GUI wizard (requires `[gui]` extra) |
| `openbench version` / `--version` | Print version |

Useful `run` flags: `--force` (ignore cache), `--comparison-only` (re-run only
comparisons on existing outputs), `--dry-run`, `--cores N`,
`--variable VAR` (repeatable), `--output-dir DIR`, `--dump-config`.

### `openbench ref` — reference registry

| Subcommand | Description |
|---|---|
| `ref list` | List all available reference datasets |
| `ref show NAME` | Show details of a dataset |
| `ref scan DIR` | Scan a directory and auto-register datasets you already have |
| `ref register NAME` | Register or update a reference dataset |
| `ref register-profile` | Register a reference profile (variable mappings) |
| `ref path NAME` | Print the local path for a dataset |
| `ref optimize` | Convert a dataset to Zarr for faster reads |
| `ref generate-station-list` | Auto-generate a station-list CSV from NetCDF files |
| `ref delete NAME` | Delete a user reference entry/overlay |
| `ref convert-old` | Convert an old-format reference YAML to v3 |

### `openbench model` — model profiles

| Subcommand | Description |
|---|---|
| `model list` | List all available model profiles |
| `model show NAME` | Show variable mappings for a profile |
| `model status` | Show model registry status |
| `model register` / `rename` / `delete` | Manage user profiles |
| `model export` / `import` | Move a profile in/out as standalone YAML |
| `model alias` | List or create user model aliases |
| `model remove-var` | Remove a variable from a profile |
| `model validate` | Check a profile for completeness issues |
| `model path NAME` | Show whether a profile is bundled or user-defined |

### `openbench cache` — runtime caches

| Subcommand | Description |
|---|---|
| `cache status` | Show cache status (e.g. `--regrid` for the regrid-weight cache) |
| `cache clear` | Clear selected cache files |

## Metrics & Scores

OpenBench computes **25+ point-wise metrics** per `(variable, simulation,
reference)` pair. By default all applicable metrics are produced; restrict with
the top-level `metrics:` list.

| Metric | Range | Meaning |
|---|---|---|
| `bias`, `percent_bias` | (−∞, ∞) | Mean (relative) difference; 0 = unbiased |
| `RMSE` | [0, ∞) | Root-mean-square error |
| `ubRMSE` / `CRMSD` | [0, ∞) | Bias-removed (centered) RMSE |
| `mean_absolute_error` | [0, ∞) | Mean absolute difference |
| `correlation` | [−1, 1] | Pearson correlation |
| `correlation_R2` | [0, 1] | Explained-variance fraction |
| `NSE` | (−∞, 1] | Nash–Sutcliffe efficiency |
| `KGE`, `KGESS` | (−∞, 1] | Kling–Gupta efficiency / skill score |
| `index_agreement` | [0, 1] | Willmott index of agreement |
| `rv` | [0, ∞) | Variance ratio σ_sim / σ_obs |
| `kappa_coeff` | [−1, 1] | Categorical agreement |

**8 normalized skill scores** rescale metrics onto a comparable 0–1 axis
(1 = best): `nBiasScore`, `nRMSEScore`, `nPhaseScore`, `nIavScore`,
`nSpatialScore`, `nSeasonalityScore`, `index_agreement`, and the combined
`Overall_Score` that drives portrait plots and rankings.

Beyond metrics/scores, the `statistics` block exposes **17 analysis modules**
(correlation, ANOVA, Mann–Kendall trend, three-cornered hat, Hellinger distance,
partial least squares, false discovery rate, …).

## Comparison & Visualization

When `comparison.enabled: true`, OpenBench renders cross-model figures
automatically (PNG, with the underlying data saved alongside):

- **Taylor diagram** & **Target diagram** — correlation/variance/CRMSD at a glance
- **Portrait plot** (seasonal) — model × variable score grid
- **Heat map** & **land-cover-based heat map**
- **Parallel coordinates**, **kernel density estimate**, **ridgeline**,
  **whisker** plots
- **Difference plots**, **relative score**, **radar map**
- **Single Model Performance Index (SMPI)**, **Mann–Kendall trend**,
  **three-cornered hat**, **functional response**, **PLSR**

## Classification Group-By (IGBP / PFT / Köppen)

OpenBench can aggregate scores per land-cover or climate class. Enable any of:

```yaml
project:
  IGBP_groupby: true          # MODIS/IGBP land cover (17 classes)
  PFT_groupby: true           # Plant Functional Type (16 classes)
  climate_zone_groupby: true  # Köppen climate zone (30 classes)
```

The classification masks (`IGBP.nc`, `PFT.nc`, `Climate_zone.nc`) are **bundled
with the package** (compressed integer masks, ~1.1 MB total), so group-by works
out of the box with no extra download. To use your own / higher-resolution masks,
override the bundled versions via either:

```bash
export OPENBENCH_DATASET_DIR=/path/to/your/dataset   # highest priority
# or place files in ./dataset/<name>.nc relative to where you run openbench
```

Per-class results appear under
`output/<run>/comparisons/{IGBP,PFT,Climate_zone}_groupby/`.

## Reference Data Management

```bash
# Browse all 101 reference datasets
openbench ref list

# Inspect a single dataset (a base name resolves to its resolution family)
openbench ref show GLEAM_v4.2a

# Scan a directory and auto-register reference datasets you already have
openbench ref scan /Volumes/work/Reference
```

Example `openbench ref list` output:

```
Name                           Category     Type   Res      Years          Variables
────────────────────────────────────────────────────────────────────────────────────
AH4GUC_LowRes                  Urban        grid   0.5°     2010-2010      1
CERES_EBAF_Ed4.2_LowRes        Energy       grid   0.5°     ...            7
CLARA_3_LowRes                 Energy       grid   0.5°     1979-2024      8
CN05.1_MidRes                  Meteorology  grid   0.25°    1961-2021      2
Caravan_Daily                  Water        stn    stn      1950-2023      1
...                            ...          ...    ...      ...            ...
Total: 101 datasets
```

Many datasets come in multiple resolutions (`_LowRes`, `_MidRes`); a base name
like `GLEAM_v4.2a` resolves to its family, or you can pin a specific one such as
`ET_Xu_etal_2025_LowRes`.

## Model Profiles

```bash
openbench model list            # list built-in profiles
openbench model show CoLM2024   # variable mappings for a profile
```

Example `openbench model list` output:

```
Name                 Type   Res      Variables Complete
────────────────────────────────────────────────────────────
CoLM2024             grid   0.5°     52        ✓
CLM5                 grid   1.25°    69        ⚠ 16
ERA5-Land            grid   0.1°     12        ✓
GLDAS                grid   0.25°    11        ✓
NoahMP5              grid   ...      70        ⚠ 34
...
Total: 22 model profiles
```

When `simulation.<name>.model` matches a known profile, variable file-name
patterns and unit conversions are applied automatically. Fields in the
simulation entry override the profile.

## Output Structure

```
output/<project_name>/
├── config.yaml          # the fully-resolved config actually used
├── run.log              # complete run log
├── data/                # preprocessed (regridded, clipped) sim/ref NetCDF
├── metrics/             # per-(var,sim,ref) metrics — grid: NetCDF, station: CSV
├── scores/              # per-(var,sim,ref) normalized scores
├── figures/             # per-pair visualizations (PNG)
├── comparisons/         # cross-model figures + group-by results (if enabled)
├── statistics/          # statistical analyses (if enabled)
├── reports/
│   ├── report.html
│   └── report.pdf       # only with the [report] extra
└── .openbench_cache.json # incremental cache index
```

Grid evaluations write one NetCDF per metric
(`<var>_ref_<R>_sim_<S>_<metric>.nc`, dims `(lat, lon)` or `(time, lat, lon)`);
station evaluations write an aggregated CSV
(`<var>_stn_<R>_<S>_evaluations.csv`, one row per station). Read either directly
with `xarray` / `pandas` for downstream analysis.

## Performance Tuning

For high-resolution grids or many simulation files, tune IO and Dask in
`openbench.yaml`:

```yaml
project:
  num_cores: 16

  io:
    # true enables zlib for numeric NetCDF outputs; level 1 is the fast default.
    netcdf_compression: true
    netcdf_compression_level: 1

    # auto/null: use the resource planner; 0: disable; N: force N files per batch.
    mfdataset_batch_size: auto
    mfdataset_auto_batch_min_files: 200
    mfdataset_auto_batch_min_size_mb: 1024
    mfdataset_auto_batch_max_size: 100
    mfdataset_auto_batch_memory_fraction: 0.25

  dask:
    enabled: true
    n_workers: 4
    threads_per_worker: 1
    processes: true
    memory_limit: auto
```

Environment variables override YAML and take precedence, which is useful for
temporary benchmarking:

```bash
OPENBENCH_NETCDF_COMPRESSION=1 OPENBENCH_NETCDF_COMP_LEVEL=1 \
OPENBENCH_MFDATASET_BATCH_SIZE=50 \
OPENBENCH_DASK=1 OPENBENCH_DASK_N_WORKERS=4 \
openbench run openbench.yaml --force
```

For repeatable measurements, run the same config from a clean output directory
and keep the resolved config, run log, and environment metadata with the output:

```bash
openbench run openbench.yaml --force --dump-config
```

## Migrating from Earlier Versions

```bash
# Convert an existing JSON config
openbench migrate old_config.json -o openbench.yaml

# Convert a Fortran NML config (requires the [migration] extra)
openbench migrate main.nml -o openbench.yaml
```

The v3.0 migration path is designed to preserve the intended evaluation setup
while making defaults and generated YAML explicit. For publication comparisons,
archive the resolved `config.yaml` and rerun a small overlap case before mixing
old and new result sets.

## Requirements

- Python 3.10 or newer
- Core dependencies: xarray, numpy, scipy, pandas, netCDF4, matplotlib,
  cartopy, dask, joblib, flox, PyYAML, Jinja2, click, tqdm, packaging
- Optional extras: `gui` (PySide6), `remote` (paramiko), `report` (xhtml2pdf),
  `migration` (f90nml)
- See [`pyproject.toml`](pyproject.toml) for the full pinned dependency list

## Citation

If you use OpenBench in scientific work, please cite:

> Wei, Z., Xu, Q., Bai, F., Xu, X., Wei, Z., Dong, W., Liang, H., Wei, N., Lu, X., Li, L., Zhang, S., Yuan, H., Liu, L., and Dai, Y.: OpenBench: a land model evaluation system, Geosci. Model Dev., 18, 6517–6540, <https://doi.org/10.5194/gmd-18-6517-2025>, 2025.

```bibtex
@article{wei2025openbench,
  author  = {Wei, Z. and Xu, Q. and Bai, F. and Xu, X. and Wei, Z. and Dong, W.
             and Liang, H. and Wei, N. and Lu, X. and Li, L. and Zhang, S.
             and Yuan, H. and Liu, L. and Dai, Y.},
  title   = {{OpenBench}: a land model evaluation system},
  journal = {Geoscientific Model Development},
  volume  = {18},
  pages   = {6517--6540},
  year    = {2025},
  doi     = {10.5194/gmd-18-6517-2025}
}
```

## Links

- Source: <https://github.com/CoLM-SYSU/OpenBench>
- PyPI: <https://pypi.org/project/colm-openbench/>
- User's Guide (PDF): [English](docs/manual/OpenBench_UsersGuide_EN.pdf) · [中文](docs/manual/OpenBench_UsersGuide.pdf)
- Issues: <https://github.com/CoLM-SYSU/OpenBench/issues>

## License

MIT License. See [LICENSE](LICENSE) for details.
