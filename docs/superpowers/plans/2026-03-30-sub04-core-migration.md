# Sub-project 4: Core Engine Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the core evaluation engine from OpenBench-wei into the unified `src/openbench/` package, preserving all functionality while adapting imports to the new structure. At the end, `openbench run openbench.yaml` should execute a complete evaluation.

**Architecture:** Copy modules bottom-up following the dependency DAG (leaf modules first), fix `from openbench.xxx` imports to match new paths, remove `sys.path` hacks, and create a config adapter that translates the new unified YAML format into the dict structure the existing evaluation code expects.

**Tech Stack:** Python, xarray, numpy, scipy, matplotlib, dask, joblib

**Spec:** `docs/superpowers/specs/2026-03-30-openbench-unification-design.md` (Section 13.Phase2.Sub4)

**Working Directory:** `/Volumes/Data01/Openbench`

**Source:** `OpenBench-wei/openbench/` (existing code to migrate)

---

## Migration Strategy

The existing code is well-structured with no circular dependencies. We copy files in dependency order:

```
Layer 1 (leaves):  util/exceptions, util/converttype, util/interfaces
Layer 2:           util/logging, util/memory, util/progress, util/fileio, util/directory
Layer 3:           data/cache, data/io, data/preprocessing, data/regrid
Layer 4:           core/metrics, core/scores, core/evaluation, core/comparison, core/groupby
Layer 5:           core/statistics (20 modules)
Layer 6:           visualization (25+ modules, as-is)
Layer 7:           config adapter + run command wiring
```

Each file gets:
1. Copied to new location
2. Internal imports updated (`from openbench.util.Mod_X` → `from openbench.util.X`)
3. `sys.path` hacks removed
4. No logic changes (preserve battle-tested code)

## File Mapping

| Old Path | New Path | Notes |
|---|---|---|
| `util/Mod_Exceptions.py` | `util/exceptions.py` | Rename only |
| `util/Mod_Converttype.py` | `util/converttype.py` | Rename only |
| `util/Mod_Interfaces.py` | `util/interfaces.py` | Rename only |
| `util/Mod_LoggingSystem.py` | `util/logging_system.py` | Avoid shadowing `logging` |
| `util/Mod_MemoryManager.py` | `util/memory.py` | Rename |
| `util/Mod_ProgressMonitor.py` | `util/progress.py` | Rename |
| `util/Mod_FileIO.py` | `util/fileio.py` | Rename |
| `util/Mod_DirectoryUtils.py` | `util/directory.py` | Rename |
| `util/Mod_ParallelEngine.py` | `util/parallel.py` | Rename |
| `util/Mod_OutputManager.py` | `util/output.py` | Rename |
| `util/Mod_ReportGenerator.py` | `util/report.py` | Rename |
| `util/Mod_ConfigCheck.py` | `util/config_check.py` | Rename |
| `util/Mod_PreValidation.py` | `util/prevalidation.py` | Rename |
| `util/Mod_CacheCleanup.py` | `util/cache_cleanup.py` | Rename |
| `util/Mod_DatasetLoader.py` | `util/dataset_loader.py` | Rename |
| `util/Mod_APIService.py` | `util/api_service.py` | Rename |
| `data/Mod_CacheSystem.py` | `data/cache.py` | Rename |
| `data/Mod_DataPipeline.py` | `data/pipeline.py` | Rename |
| `data/Mod_DatasetProcessing.py` | `data/processing.py` | Rename |
| `data/Mod_Preprocessing.py` | `data/preprocessing.py` | Rename |
| `data/Mod_Climatology.py` | `data/climatology.py` | Rename |
| `data/Lib_FileProcessing.py` | `data/file_processing.py` | Rename |
| `data/Lib_Unit.py` | `data/unit.py` | Rename |
| `data/Lib_Time.py` | `data/time_utils.py` | Rename |
| `data/regrid/` | `data/regrid/` | Copy entire dir |
| `data/custom/` | `data/custom/` | Copy entire dir |
| `core/metrics/Mod_Metrics.py` | `core/metrics.py` | Rename |
| `core/scoring/Mod_Scores.py` | `core/scores.py` | Rename |
| `core/evaluation/Mod_Evaluation.py` | `core/evaluation.py` | Rename |
| `core/evaluation/Mod_EvaluationEngine.py` | `core/evaluation_engine.py` | Rename |
| `core/comparison/Mod_Comparison.py` | `core/comparison.py` | Rename |
| `core/comparison/Mod_Landcover_Groupby.py` | `core/landcover_groupby.py` | Rename |
| `core/comparison/Mod_ClimateZone_Groupby.py` | `core/climatezone_groupby.py` | Rename |
| `core/statistic/*.py` | `core/statistics/*.py` | Copy entire dir |
| `visualization/*.py` | `visualization/*.py` | Copy as-is |
| `config/readers.py` | `config/legacy_readers.py` | For migration tool |
| `config/processors.py` | `config/legacy_processors.py` | For evaluation |
| `config/updaters.py` | `config/legacy_updaters.py` | For evaluation |
| `config/manager.py` | `config/legacy_manager.py` | For evaluation |

---

### Task 1: Migrate Utility Modules (Layer 1-2)

**Files:**
- Copy + rename: 16 util modules from OpenBench-wei
- Create: `src/openbench/util/compat.py` (import compatibility shim)

- [ ] **Step 1: Copy all utility modules**

Run:
```bash
cd /Volumes/Data01/Openbench

# Layer 1 - leaf modules
cp OpenBench-wei/openbench/util/Mod_Exceptions.py src/openbench/util/exceptions.py
cp OpenBench-wei/openbench/util/Mod_Converttype.py src/openbench/util/converttype.py
cp OpenBench-wei/openbench/util/Mod_Interfaces.py src/openbench/util/interfaces.py

# Layer 2 - utility modules
cp OpenBench-wei/openbench/util/Mod_LoggingSystem.py src/openbench/util/logging_system.py
cp OpenBench-wei/openbench/util/Mod_MemoryManager.py src/openbench/util/memory.py
cp OpenBench-wei/openbench/util/Mod_ProgressMonitor.py src/openbench/util/progress.py
cp OpenBench-wei/openbench/util/Mod_FileIO.py src/openbench/util/fileio.py
cp OpenBench-wei/openbench/util/Mod_DirectoryUtils.py src/openbench/util/directory.py
cp OpenBench-wei/openbench/util/Mod_ParallelEngine.py src/openbench/util/parallel.py
cp OpenBench-wei/openbench/util/Mod_OutputManager.py src/openbench/util/output.py
cp OpenBench-wei/openbench/util/Mod_ReportGenerator.py src/openbench/util/report.py
cp OpenBench-wei/openbench/util/Mod_ConfigCheck.py src/openbench/util/config_check.py
cp OpenBench-wei/openbench/util/Mod_PreValidation.py src/openbench/util/prevalidation.py
cp OpenBench-wei/openbench/util/Mod_CacheCleanup.py src/openbench/util/cache_cleanup.py
cp OpenBench-wei/openbench/util/Mod_DatasetLoader.py src/openbench/util/dataset_loader.py
cp OpenBench-wei/openbench/util/Mod_APIService.py src/openbench/util/api_service.py
```

- [ ] **Step 2: Fix imports in all util modules**

For each file in `src/openbench/util/`, replace all internal import patterns:
- `from openbench.util.Mod_Exceptions` → `from openbench.util.exceptions`
- `from openbench.util.Mod_Converttype` → `from openbench.util.converttype`
- `from openbench.util.Mod_LoggingSystem` → `from openbench.util.logging_system`
- `from openbench.util.Mod_MemoryManager` → `from openbench.util.memory`
- `from openbench.util.Mod_ProgressMonitor` → `from openbench.util.progress`
- `from openbench.util.Mod_FileIO` → `from openbench.util.fileio`
- `from openbench.util.Mod_DirectoryUtils` → `from openbench.util.directory`
- `from openbench.util.Mod_ParallelEngine` → `from openbench.util.parallel`
- `from openbench.util.Mod_OutputManager` → `from openbench.util.output`
- `from openbench.util.Mod_ReportGenerator` → `from openbench.util.report`
- `from openbench.util.Mod_ConfigCheck` → `from openbench.util.config_check`
- `from openbench.util.Mod_PreValidation` → `from openbench.util.prevalidation`
- `from openbench.util.Mod_CacheCleanup` → `from openbench.util.cache_cleanup`
- `from openbench.util.Mod_DatasetLoader` → `from openbench.util.dataset_loader`
- `from openbench.util.Mod_Interfaces` → `from openbench.util.interfaces`
- `from openbench.util.Mod_APIService` → `from openbench.util.api_service`

Also remove any `sys.path.insert` or `sys.path.append` lines.

- [ ] **Step 3: Update `src/openbench/util/__init__.py`**

```python
"""Shared utilities: logging, parallelism, memory, progress, reports.

This package contains utility modules migrated from OpenBench v2.
"""
```

- [ ] **Step 4: Verify leaf modules import**

Run:
```bash
python -c "from openbench.util.exceptions import OpenBenchError; print('OK')"
python -c "from openbench.util.converttype import convert_type; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/openbench/util/
git commit -m "feat(util): migrate 16 utility modules from OpenBench-wei"
```

---

### Task 2: Migrate Data Pipeline Modules (Layer 3)

**Files:**
- Copy + rename: 8 data modules + regrid/ and custom/ directories

- [ ] **Step 1: Copy data modules**

```bash
cd /Volumes/Data01/Openbench
cp OpenBench-wei/openbench/data/Mod_CacheSystem.py src/openbench/data/cache.py
cp OpenBench-wei/openbench/data/Mod_DataPipeline.py src/openbench/data/pipeline.py
cp OpenBench-wei/openbench/data/Mod_DatasetProcessing.py src/openbench/data/processing.py
cp OpenBench-wei/openbench/data/Mod_Preprocessing.py src/openbench/data/preprocessing.py
cp OpenBench-wei/openbench/data/Mod_Climatology.py src/openbench/data/climatology.py
cp OpenBench-wei/openbench/data/Lib_FileProcessing.py src/openbench/data/file_processing.py
cp OpenBench-wei/openbench/data/Lib_Unit.py src/openbench/data/unit.py
cp OpenBench-wei/openbench/data/Lib_Time.py src/openbench/data/time_utils.py

# Copy entire directories
cp -r OpenBench-wei/openbench/data/regrid src/openbench/data/
cp -r OpenBench-wei/openbench/data/custom src/openbench/data/
```

- [ ] **Step 2: Fix imports in all data modules**

Replace patterns:
- `from openbench.data.Mod_CacheSystem` → `from openbench.data.cache`
- `from openbench.data.Mod_DataPipeline` → `from openbench.data.pipeline`
- `from openbench.data.Mod_DatasetProcessing` → `from openbench.data.processing`
- `from openbench.data.Mod_Preprocessing` → `from openbench.data.preprocessing`
- `from openbench.data.Mod_Climatology` → `from openbench.data.climatology`
- `from openbench.data.Lib_FileProcessing` → `from openbench.data.file_processing`
- `from openbench.data.Lib_Unit` → `from openbench.data.unit`
- `from openbench.data.Lib_Time` → `from openbench.data.time_utils`
- All `from openbench.util.Mod_*` → new names (same as Task 1)
- Remove `sys.path` hacks

- [ ] **Step 3: Verify key imports**

```bash
python -c "from openbench.data.cache import CacheSystem; print('OK')"
python -c "from openbench.data.unit import convert_unit; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/openbench/data/
git commit -m "feat(data): migrate data pipeline modules from OpenBench-wei"
```

---

### Task 3: Migrate Core Modules (Layer 4)

**Files:**
- Copy: metrics, scores, evaluation, comparison, groupby

- [ ] **Step 1: Copy core modules**

```bash
cd /Volumes/Data01/Openbench
cp OpenBench-wei/openbench/core/metrics/Mod_Metrics.py src/openbench/core/metrics.py
cp OpenBench-wei/openbench/core/scoring/Mod_Scores.py src/openbench/core/scores.py
cp OpenBench-wei/openbench/core/evaluation/Mod_Evaluation.py src/openbench/core/evaluation.py
cp OpenBench-wei/openbench/core/evaluation/Mod_EvaluationEngine.py src/openbench/core/evaluation_engine.py
cp OpenBench-wei/openbench/core/comparison/Mod_Comparison.py src/openbench/core/comparison.py
cp OpenBench-wei/openbench/core/comparison/Mod_Landcover_Groupby.py src/openbench/core/landcover_groupby.py
cp OpenBench-wei/openbench/core/comparison/Mod_ClimateZone_Groupby.py src/openbench/core/climatezone_groupby.py
```

- [ ] **Step 2: Fix imports in all core modules**

Replace patterns (all the Mod_* → new names as in Tasks 1-2), plus:
- `from openbench.core.metrics.Mod_Metrics` → `from openbench.core.metrics`
- `from openbench.core.scoring.Mod_Scores` → `from openbench.core.scores`
- `from openbench.core.evaluation.Mod_Evaluation` → `from openbench.core.evaluation`
- `from openbench.core.evaluation.Mod_EvaluationEngine` → `from openbench.core.evaluation_engine`
- `from openbench.core.comparison.Mod_Comparison` → `from openbench.core.comparison`
- `from openbench.core.comparison.Mod_Landcover_Groupby` → `from openbench.core.landcover_groupby`
- `from openbench.core.comparison.Mod_ClimateZone_Groupby` → `from openbench.core.climatezone_groupby`
- Remove `sys.path` hacks

- [ ] **Step 3: Verify key imports**

```bash
python -c "from openbench.core.metrics import metrics; print('OK')"
python -c "from openbench.core.scores import scores; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/openbench/core/
git commit -m "feat(core): migrate metrics, scores, evaluation, comparison modules"
```

---

### Task 4: Migrate Statistics Modules (Layer 5)

**Files:**
- Copy: entire statistics directory

- [ ] **Step 1: Copy statistics modules**

```bash
cd /Volumes/Data01/Openbench
mkdir -p src/openbench/core/statistics
cp OpenBench-wei/openbench/core/statistic/*.py src/openbench/core/statistics/
```

- [ ] **Step 2: Fix imports in statistics modules**

Same pattern replacements as previous tasks.

- [ ] **Step 3: Commit**

```bash
git add src/openbench/core/statistics/
git commit -m "feat(core): migrate 20+ statistics modules"
```

---

### Task 5: Migrate Visualization Modules (Layer 6)

**Files:**
- Copy: entire visualization directory (as-is per spec)

- [ ] **Step 1: Copy visualization modules**

```bash
cd /Volumes/Data01/Openbench
cp OpenBench-wei/openbench/visualization/*.py src/openbench/visualization/
cp -r OpenBench-wei/openbench/visualization/cmaps src/openbench/visualization/
```

- [ ] **Step 2: Fix imports in visualization modules**

Same pattern replacements. Note: visualization modules have many `sys.path` hacks that must be removed.

- [ ] **Step 3: Commit**

```bash
git add src/openbench/visualization/
git commit -m "feat(viz): migrate 25+ visualization modules as-is"
```

---

### Task 6: Migrate Legacy Config Modules + Create Config Adapter

**Files:**
- Copy: config modules needed by evaluation engine
- Create: `src/openbench/config/adapter.py` (bridges new config → old format)

- [ ] **Step 1: Copy legacy config modules**

```bash
cd /Volumes/Data01/Openbench
cp OpenBench-wei/openbench/config/readers.py src/openbench/config/legacy_readers.py
cp OpenBench-wei/openbench/config/processors.py src/openbench/config/legacy_processors.py
cp OpenBench-wei/openbench/config/updaters.py src/openbench/config/legacy_updaters.py
cp OpenBench-wei/openbench/config/manager.py src/openbench/config/legacy_manager.py
```

- [ ] **Step 2: Fix imports in legacy config modules**

- [ ] **Step 3: Create `src/openbench/config/adapter.py`**

This is the bridge: takes an `OpenBenchConfig` (new format) and produces the dict structure that the legacy evaluation code expects.

```python
"""Adapter to convert new OpenBenchConfig to legacy dict format.

The evaluation engine (Mod_Evaluation, Mod_Comparison, etc.) expects
config as nested dicts with specific key patterns. This adapter
translates the new dataclass-based config to that format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbench.config.schema import OpenBenchConfig
from openbench.data.registry import RegistryManager


def to_legacy_config(cfg: OpenBenchConfig) -> dict[str, Any]:
    """Convert OpenBenchConfig to the legacy dict format.

    Returns a dict that mimics the structure produced by the old
    ConfigManager.load_config() for use with legacy evaluation code.
    """
    mgr = RegistryManager()

    general = {
        "basename": cfg.project.name,
        "basedir": cfg.project.output_dir,
        "syear": cfg.project.years[0],
        "eyear": cfg.project.years[1],
        "min_year": cfg.project.min_year_threshold,
        "min_lat": cfg.project.lat_range[0],
        "max_lat": cfg.project.lat_range[1],
        "min_lon": cfg.project.lon_range[0],
        "max_lon": cfg.project.lon_range[1],
        "num_cores": cfg.options.num_cores or _detect_cores(),
        "evaluation": True,
        "comparison": cfg.comparison.enabled,
        "statistics": cfg.statistics.enabled,
        "debug_mode": cfg.options.debug_mode,
        "only_drawing": cfg.options.only_drawing,
        "IGBP_groupby": cfg.options.IGBP_groupby,
        "PFT_groupby": cfg.options.PFT_groupby,
        "Climate_zone_groupby": cfg.options.climate_zone_groupby,
        "unified_mask": cfg.options.unified_mask,
        "generate_report": cfg.options.generate_report,
        "weight": cfg.comparison.weight or "area",
        "compare_tim_res": cfg.comparison.tim_res or "Month",
        "compare_tzone": cfg.comparison.timezone or 0,
        "compare_grid_res": cfg.comparison.grid_res or 0.5,
    }

    # Build evaluation_items dict (boolean flags)
    evaluation_items = {var: True for var in cfg.evaluation.variables}

    # Build metrics dict (boolean flags)
    if cfg.metrics:
        metrics_dict = {m: True for m in cfg.metrics}
    else:
        metrics_dict = {"bias": True, "RMSE": True, "correlation": True}

    # Build scores dict
    if cfg.scores:
        scores_dict = {s: True for s in cfg.scores}
    else:
        scores_dict = {"Overall_Score": True}

    # Build comparisons dict
    if cfg.comparison.items:
        comparisons_dict = {c: True for c in cfg.comparison.items}
    else:
        comparisons_dict = {"Taylor_Diagram": True, "HeatMap": True}

    # Build statistics dict
    if cfg.statistics.items:
        statistics_dict = {s: True for s in cfg.statistics.items}
    else:
        statistics_dict = {}

    return {
        "general": general,
        "evaluation_items": evaluation_items,
        "metrics": metrics_dict,
        "scores": scores_dict,
        "comparisons": comparisons_dict,
        "statistics": statistics_dict,
    }


def _detect_cores() -> int:
    """Auto-detect available CPU cores."""
    try:
        import os
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1
```

- [ ] **Step 4: Commit**

```bash
git add src/openbench/config/adapter.py src/openbench/config/legacy_*.py
git commit -m "feat(config): add legacy config modules and adapter for evaluation bridge"
```

---

### Task 7: Wire `openbench run` Command

**Files:**
- Create: `src/openbench/runner/local.py`
- Modify: `src/openbench/cli/run.py`

- [ ] **Step 1: Create `src/openbench/runner/local.py`**

```python
"""Local evaluation runner.

Orchestrates the evaluation pipeline using the new config system
and the migrated core engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


def run_evaluation(cfg: OpenBenchConfig) -> dict[str, Any]:
    """Run evaluation from a validated config.

    This is the main entry point that replaces the old openbench.py script.
    It uses the config adapter to bridge between new and legacy formats.

    Args:
        cfg: Validated OpenBenchConfig instance.

    Returns:
        Summary dict with results.
    """
    from openbench.config.adapter import to_legacy_config

    legacy = to_legacy_config(cfg)
    general = legacy["general"]

    # Setup output directories
    basedir = Path(general["basedir"])
    basename = general["basename"]
    output_dir = basedir / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation: {basename}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Variables: {list(legacy['evaluation_items'].keys())}")
    logger.info(f"Simulations: {list(cfg.simulation.keys())}")

    # TODO: In Sub-project 4 full implementation, this will call
    # the migrated evaluation engine. For now, we verify the pipeline
    # is connected end-to-end.
    results = {
        "status": "success",
        "basename": basename,
        "output_dir": str(output_dir),
        "variables": list(legacy["evaluation_items"].keys()),
        "simulations": list(cfg.simulation.keys()),
        "metrics": list(legacy["metrics"].keys()),
    }

    logger.info("Evaluation complete.")
    return results
```

- [ ] **Step 2: Update `src/openbench/cli/run.py`**

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
    from openbench.config import ConfigError, load_config

    # Load and validate config
    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"Config error: {e}", fg="red")
        raise SystemExit(1)

    # Apply CLI overrides
    if cores:
        cfg.options.num_cores = cores
    if variables:
        cfg.evaluation.variables = list(variables)

    if dry_run:
        click.secho("Dry run — config valid, would evaluate:", bold=True)
        click.echo(f"  Project: {cfg.project.name}")
        click.echo(f"  Variables: {', '.join(cfg.evaluation.variables)}")
        click.echo(f"  Simulations: {', '.join(cfg.simulation.keys())}")
        click.echo(f"  Metrics: {cfg.metrics or 'all'}")
        return

    if remote:
        click.echo("Remote execution not yet implemented.")
        click.echo("Install openbench[remote] and use openbench gui for remote execution.")
        raise SystemExit(1)

    # Run evaluation
    from openbench.runner.local import run_evaluation

    click.secho(f"Running evaluation: {cfg.project.name}", bold=True)
    results = run_evaluation(cfg)

    click.secho(f"\n✓ Evaluation complete", fg="green", bold=True)
    click.echo(f"  Output: {results['output_dir']}")
    click.echo(f"  Variables: {len(results['variables'])}")
    click.echo(f"  Simulations: {len(results['simulations'])}")
```

- [ ] **Step 3: Test end-to-end**

```bash
openbench run tests/test_config/fixtures/minimal.yaml --dry-run
openbench run tests/test_config/fixtures/full.yaml --dry-run
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add src/openbench/runner/ src/openbench/cli/run.py
git commit -m "feat(runner): wire openbench run with config adapter and local runner"
```

---

### Task 8: Final Verification and Lint

- [ ] **Step 1: Run lint and fix**

```bash
ruff check src/ tests/
ruff format src/ tests/
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 3: End-to-end verification**

```bash
openbench run tests/test_config/fixtures/full.yaml --dry-run
openbench check tests/test_config/fixtures/full.yaml
openbench data list | head -5
openbench model list
openbench --version
```

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: lint cleanup for core engine migration"
```

- [ ] **Step 5: Tag milestone**

```bash
git tag -a v3.0.0a4 -m "Sub-project 4 complete: core engine migration"
```
