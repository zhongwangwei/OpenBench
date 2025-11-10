# OpenBench: The Open Source Land Surface Model Benchmarking System

> A fully automated, cross-platform framework for benchmarking land surface models (LSMs) against curated reference datasets with consistent metrics, visualizations, and reports.

OpenBench standardizes every stage of LSM evaluation—configuration management, preprocessing, validation, scoring, visualization, and reporting—so researchers can focus on insights instead of wiring. The project is actively maintained and targets Windows, Linux, and macOS with emphasis on reproducibility and modularity.

## Contents
- [Highlights](#highlights)
- [Core Capabilities](#core-capabilities)
- [Architecture & Repository Layout](#architecture--repository-layout)
- [Getting Started](#getting-started)
- [Running Benchmarks](#running-benchmarks)
- [Configuration Reference](#configuration-reference)
- [Outputs & Reports](#outputs--reports)
- [Development & Testing](#development--testing)
- [Troubleshooting & Performance](#troubleshooting--performance)
- [Customization & Extensibility](#customization--extensibility)
- [Contributing](#contributing)
- [Version History](#version-history)
- [License & Credits](#license--credits)

## Highlights
- **Latest Release:** v2.0 (July 2025)
- **Multi-format configs:** JSON, YAML, and Fortran Namelist with automatic detection.
- **Modular architecture:** Nine cohesive subsystems (config, data pipeline, evaluation, scoring, visualization, etc.) with dependency injection.
- **Parallel & cached:** Adaptive worker allocation, memory-aware scheduling, and multi-level caching.
- **Unified logging:** Clean console progress plus detailed file logs with structured context.
- **GUI + API (preview):** Desktop interface and Python API are in active development and not yet feature-complete.
- **Climate analytics:** Built-in Köppen climate zone group-by and land-cover grouping tools.

## Core Capabilities
- **Multi-model comparisons** for CLM, CoLM, and other LSM outputs using common scoring metrics.
- **Data ingestion** for gridded datasets and station observations with configurable preprocessing.
- **Advanced evaluation metrics** (bias, RMSE, correlation, Taylor scores, Model Fidelity Metric) and customizable scoring pipelines.
- **Automated visualization** (maps, time series, scatter, Taylor diagrams) with reproducible styling.
- **Resilient error handling** that surfaces actionable diagnostics without halting complete runs.
- **Smart resource management** leveraging psutil when present, plus automatic cleanup of temp assets.
- **Extensible plugin points** for new metrics, filters, cache strategies, and report templates.

## Architecture & Repository Layout
OpenBench is organized as a Python package with focused subpackages. Refer to `AGENTS.md` for contributor-specific conventions.

| Area | Location | Responsibility |
| --- | --- | --- |
| Configuration | `openbench/config/` | Readers, writers, and validation helpers for JSON/YAML/NML namelists. |
| Data Pipeline | `openbench/data/` | Preprocessing, caching (`Mod_CacheSystem`), climatology helpers, and regridding utilities. |
| Core Logic | `openbench/core/` | Evaluation, metrics, scoring, comparison, and statistics engines. |
| Utilities | `openbench/util/` | Logging, progress monitors, report generation, interface definitions, memory management. |
| Visualization | `openbench/visualization/` | Plotting routines for grid, station, land-cover, and climate-zone views. |
| GUI | `GUI/` | Standalone desktop interface built on the same APIs. |
| Config Samples | `nml/` | Format-specific example namelists (`nml-json/`, `nml-yaml/`, `nml-Fortran/`). |
| Datasets | `dataset/` | Reference data and curated simulations (not versioned by default). |
| Outputs | `output/` | Generated metrics, scores, figures, reports, logs, scratch, and tmp folders. |

```
OpenBench/
├── openbench/
│   ├── config/            # Config readers/writers
│   ├── core/              # Evaluation, metrics, scoring, comparison
│   ├── data/              # Preprocessing, caching, climatology
│   ├── util/              # Logging, validation, reports, helpers
│   ├── visualization/     # Plotting modules
│   ├── openbench.py       # CLI entry point
│   └── openbench_api.py   # High-level Python API
├── preprocessing/         # Conversion scripts and station prep workflows
├── GUI/                   # Desktop UI
├── dataset/               # Reference and simulation data (user-provided)
├── nml/                   # Sample configuration sets (JSON/YAML/NML)
├── output/                # Evaluation artifacts (gitignored)
├── docs/ / doc/           # User guides and PDF manuals
└── AGENTS.md              # Contributor guide (coding and PR conventions)
```

## Getting Started

### Requirements
- Python **3.10+** (3.11 tested). Use a virtual environment to isolate dependencies.
- Core libraries from `requirements.txt`: `xarray`, `numpy`, `pandas`, `netCDF4`, `dask`, `cartopy`, `matplotlib`, `scipy`, `joblib`, `flox`, `jinja2`, `xhtml2pdf`.
- Optional: `psutil` for enhanced memory telemetry, `f90nml` for NML parsing, `PyYAML` for YAML support (auto-detected), `cdo` binary for advanced regridding (Linux/macOS).

### Setup
1. Clone and enter the repository:
   ```bash
   git clone https://github.com/zhongwangwei/OpenBench.git
   cd OpenBench
   ```
2. Create/activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. (Optional) Install extras:
   ```bash
   pip install psutil f90nml pyyaml pytest
   ```
5. Populate `dataset/` with reference and simulation data (see docs in `doc/` and `docs/` for acquisition guidance).

## Running Benchmarks

### Command-Line Workflow
Use the main driver `openbench/openbench.py`. The file format is inferred from the extension.

```bash
# JSON example
python openbench/openbench.py nml/nml-json/main-Debug.json

# YAML example
python openbench/openbench.py nml/nml-yaml/main-Debug.yaml

# Fortran Namelist example
python openbench/openbench.py nml/nml-Fortran/main-Debug.nml
```

Common options:
- Duplicate an existing `main-*.json|yaml|nml` and adjust paths to your datasets.
- Use absolute paths if launching from outside the repo.
- Provide a custom output directory inside the configuration (`general.basedir`).

### GUI Preview (Experimental)
```
python GUI/GUI_openbench.py
```
The desktop interface is still under development, may be missing features, and is not yet recommended for production workflows.

### API Preview (Experimental)
```python
from openbench.openbench_api import OpenBench

ob = OpenBench.from_config("nml/nml-json/main-Debug.json")
ob.run()  # Executes the full pipeline
summary = ob.results
```
The programmatic API is evolving and subject to breaking changes; treat it as a preview interface until the formal release notes mark it stable.

### Preprocessing Utilities
- `preprocessing/convert_nml_to_yaml_json/convert_nml_to_yaml_json.py`: regenerates synchronized JSON/YAML configs from Fortran namelists.
- `preprocessing/get_stn_*`: prepares station-based datasets for ingestion.

## Configuration Reference
Configurations are split into complementary files:

- **Main (`main-*`)**: top-level metadata, run name, output directory, toggles for evaluation modules.
- **Reference (`ref-*`)**: lists reference products and variable mappings.
- **Simulation (`sim-*`)**: describes model outputs, units, spatial/temporal metadata.
- **Optional extras**: land-cover groups, climate zones, plotting presets.

Example JSON:
```json
{
  "general": {
    "basename": "debug",
    "basedir": "./output",
    "reference_nml": "./nml/nml-json/ref-Debug.json",
    "simulation_nml": "./nml/nml-json/sim-Debug.json"
  },
  "evaluation_items": {
    "Evapotranspiration": true,
    "Latent_Heat": true
  }
}
```

Example YAML:
```yaml
general:
  basename: debug
  basedir: ./output
  reference_nml: ./nml/nml-yaml/ref-Debug.yaml
  simulation_nml: ./nml/nml-yaml/sim-Debug.yaml
evaluation_items:
  Evapotranspiration: true
  Latent_Heat: true
```

Example Fortran Namelist:
```fortran
&general
  basename = debug
  basedir = ./output
  reference_nml = ./nml/nml-Fortran/ref-Debug.nml
  simulation_nml = ./nml/nml-Fortran/sim-Debug.nml
/
```

### Tips
- Stick to forward slashes for portability.
- Keep dataset-relative paths (`./dataset/...`) for easier sharing.
- Use `openbench/util/Mod_ConfigCheck.py` helpers (invoked automatically) for early validation.

## Outputs & Reports
Results are stored beneath `output/<basename>/`:

```
output/debug/
├── output/
│   ├── metrics/        # CSV/JSON/NetCDF metric summaries
│   ├── scores/         # Aggregated scoring tables
│   ├── data/           # Processed intermediate datasets
│   ├── figures/        # Maps, scatter plots, Taylor diagrams
│   └── comparisons/    # Cross-model comparisons
├── log/                # Timestamped log files
├── scratch/            # Working data
└── tmp/                # Temporary assets
```

Key characteristics:
- Metrics and scores are organized per variable and evaluation scope (grid, station, climate zone).
- Figures mirror the evaluation structure for quick inspection.
- Logs include both console-friendly summaries and structured entries (JSON if enabled).
- Automatic cleanup removes stale scratch/tmp directories between runs.

## Development & Testing
- Adhere to `AGENTS.md` for coding style, naming, and PR expectations (PEP 8, 4-space indentation, descriptive names).
- Prefer colocating new utilities with similar modules (e.g., new evaluation logic under `openbench/core/evaluation/`).
- Recommended workflow:
  ```bash
  # Lint / style (optional)
  python -m compileall openbench

  # Functional smoke tests
  python openbench/openbench.py nml/nml-json/main-Debug.json
  python openbench/openbench.py nml/nml-yaml/main-Debug.yaml
  ```
- Automated tests (if added) should live under `tests/` and use `pytest`. Mirror the package structure (`tests/core/test_metrics.py`, etc.) and keep fixtures lightweight (mocked xarray datasets).
- Document reproducibility steps in PRs: configuration used, dataset subset, observed metrics.

## Troubleshooting & Performance

### Platform Notes
- **Windows:** `cdo` is optional and skipped automatically. Prefer forward slashes in configs. Lower worker counts if memory constrained.
- **Linux/macOS:** Install `cdo` via `apt`, `yum`, or `brew` for advanced regridding. Ensure user write access to `output/`.

### Configuration Issues
- Missing file errors typically stem from relative path mismatches—start from repository root or convert to absolute paths.
- Comments in NML files are stripped automatically; keep formatting consistent.
- Use the preprocessing scripts to convert Fortran namelists whenever reference files change.

### Performance Tips
- Override `general.max_workers` when working on small machines; OpenBench will otherwise auto-detect cores.
- Enable caching and reuse `output/<case>/tmp` for multi-run experiments; caches are invalidated when configs change.
- Large evaluations benefit from chunked datasets (via `dask`) and the optional `psutil` monitor for early warnings.

### Cartopy & Offline Assets
Cartopy downloads coastline data on first run. For offline/HPC clusters:
```python
import cartopy
print(cartopy.config["data_dir"])
```
Populate that directory with Natural Earth datasets (see README instructions in `doc/CLIMATOLOGY_EVALUATION.md`).

## Customization & Extensibility
- **Custom metrics**: Implement new functions under `openbench/core/metrics/` and register them with the evaluation engine.
- **Reporting**: Extend `openbench/util/Mod_ReportGenerator.py` templates (Jinja2) for custom PDF/HTML layouts.
- **Filters & preprocessing**: Drop scripts in `openbench/data/custom/` or extend `Mod_DataPipeline.py` for specialized QC.
- **API integrations (preview)**: Leverage the evolving `OpenBench` class carefully when integrating with workflow managers (Airflow, Snakemake, etc.).

## Contributing
We welcome issues, feature proposals, and pull requests.

1. **Discuss first**: Open an issue for large changes to align on scope.
2. **Fork & branch**: `feature/<topic>` or `bugfix/<issue-number>`.
3. **Follow style**: PEP 8, meaningful names, logging via `Mod_LoggingSystem`, and refer to `AGENTS.md` for directory-specific expectations.
4. **Test**: Run at least one JSON and one YAML scenario (plus any new pytest suites).
5. **Commit messages**: Use short, imperative summaries (`Add Model Fidelity Metric support`).
6. **Pull request checklist**:
   - Clear description and screenshots of new visuals (if applicable).
   - Commands/configs used for validation.
   - Note any dataset or config updates that require downstream reruns.

## Version History
- **v2.0 (July 2025)**  
  - Multi-format config detection, enhanced modular architecture, smart parallel engine, multi-level caching, structured logging, Köppen climate analysis, GUI refresh, dataset directory restructuring.
- **v1.0 (June 2025)**  
  - Initial open-source release with JSON configs and baseline evaluation workflow.

## Citation
If you use OpenBench in scientific work, please cite:

Wei, Z., Xu, Q., Bai, F., Xu, X., Wei, Z., Dong, W., Liang, H., Wei, N., Lu, X., Li, L. and Zhang, S., 2025. OpenBench: a land model evaluation system. *Geoscientific Model Development*, 18(18), 6517-6540.

## License & Credits
- Licensed under the **MIT License** (see `LICENSE`).
- Cite “OpenBench: The Open Source Land Surface Model Benchmarking System” in publications referencing this toolkit.
- Primary contacts: Zhongwang Wei (zhongwang007@gmail.com) and the OpenBench contributor team listed in `docs/` and commit history.

For deeper guidance, explore the PDF manuals in `doc/` and `docs/`, plus the contributor instructions in `AGENTS.md`. Happy benchmarking!
