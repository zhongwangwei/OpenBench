# Sub-project 2: Configuration System

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the new unified YAML-only configuration system with dataclass-based schema, validation, smart defaults, `_defaults` merge for simulation, `!include` support, and a migration tool to convert old JSON/NML configs.

**Architecture:** Config is defined as a hierarchy of Python dataclasses. A single `openbench.yaml` replaces the old 4-6 file setup. The loader reads YAML, validates against the schema, resolves `_defaults` and `!include`, and returns a typed `OpenBenchConfig` object. The migration tool reads old JSON/NML configs and produces the new format.

**Tech Stack:** Python dataclasses, PyYAML, click (for CLI), pytest

**Spec:** `docs/superpowers/specs/2026-03-30-openbench-unification-design.md` (Sections 5, 13.Phase1.Sub2)

**Working Directory:** `/Volumes/Data01/Openbench`

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/openbench/config/schema.py` | Dataclass definitions for all config sections |
| Create | `src/openbench/config/loader.py` | YAML loading, `!include` support, validation, `_defaults` merge |
| Create | `src/openbench/config/migration.py` | Convert old JSON/NML configs to new YAML |
| Modify | `src/openbench/config/__init__.py` | Public API exports |
| Modify | `src/openbench/cli/check.py` | Wire up `openbench check` to config validation |
| Modify | `src/openbench/cli/migrate.py` | Wire up `openbench migrate` to migration tool |
| Create | `tests/test_config/__init__.py` | Test package |
| Create | `tests/test_config/test_schema.py` | Schema dataclass tests |
| Create | `tests/test_config/test_loader.py` | Loader + validation tests |
| Create | `tests/test_config/test_migration.py` | Migration tool tests |
| Create | `tests/test_config/fixtures/` | Test fixture YAML files |

---

### Task 1: Define Config Schema Dataclasses

**Files:**
- Create: `src/openbench/config/schema.py`
- Create: `tests/test_config/__init__.py`
- Create: `tests/test_config/test_schema.py`

- [ ] **Step 1: Write test file `tests/test_config/test_schema.py`**

```python
"""Tests for config schema dataclasses."""

from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
    SimulationEntry,
    StatisticsConfig,
)


def test_project_config_defaults():
    p = ProjectConfig(name="test", output_dir="./output", years=[2004, 2010])
    assert p.name == "test"
    assert p.output_dir == "./output"
    assert p.years == [2004, 2010]
    assert p.min_year_threshold == 3
    assert p.lat_range == [-90.0, 90.0]
    assert p.lon_range == [-180.0, 180.0]


def test_project_config_custom():
    p = ProjectConfig(
        name="custom",
        output_dir="/data/out",
        years=[2000, 2020],
        min_year_threshold=5,
        lat_range=[-60, 90],
        lon_range=[-180, 180],
    )
    assert p.min_year_threshold == 5
    assert p.lat_range == [-60, 90]


def test_evaluation_config():
    e = EvaluationConfig(variables=["Evapotranspiration", "GPP"])
    assert len(e.variables) == 2


def test_simulation_entry_minimal():
    s = SimulationEntry(model="CoLM2024", root_dir="/data/CoLM2024")
    assert s.model == "CoLM2024"
    assert s.root_dir == "/data/CoLM2024"
    assert s.data_type is None
    assert s.variables is None


def test_simulation_entry_with_overrides():
    s = SimulationEntry(
        model="CLM5",
        root_dir="/data/CLM5",
        data_type="grid",
        grid_res=1.0,
        tim_res="Month",
        variables={"GPP": {"varname": "FPSN"}},
    )
    assert s.data_type == "grid"
    assert s.grid_res == 1.0
    assert s.variables["GPP"]["varname"] == "FPSN"


def test_options_config_defaults():
    o = OptionsConfig()
    assert o.num_cores is None
    assert o.time_alignment == "intersection"
    assert o.unified_mask is True
    assert o.generate_report is True
    assert o.IGBP_groupby is False
    assert o.PFT_groupby is False
    assert o.climate_zone_groupby is False
    assert o.debug_mode is False
    assert o.only_drawing is False


def test_comparison_config_defaults():
    c = ComparisonConfig()
    assert c.enabled is False


def test_statistics_config_defaults():
    s = StatisticsConfig()
    assert s.enabled is False


def test_openbench_config_minimal():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="test", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference={"Evapotranspiration": "GLEAM_v4.2a"},
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data")},
    )
    assert cfg.project.name == "test"
    assert cfg.metrics is None
    assert cfg.scores is None
    assert cfg.comparison.enabled is False
    assert cfg.statistics.enabled is False
    assert cfg.options.time_alignment == "intersection"


def test_openbench_config_full():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="full", output_dir="./out", years=[2000, 2020]),
        evaluation=EvaluationConfig(variables=["GPP", "Latent_Heat"]),
        reference={"GPP": "FLUXCOM", "Latent_Heat": "FLUXCOM"},
        simulation={
            "CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data/colm"),
            "CLM5": SimulationEntry(model="CLM5", root_dir="/data/clm5"),
        },
        metrics=["bias", "RMSE", "correlation"],
        scores=["nBiasScore", "nRMSEScore"],
        comparison=ComparisonConfig(enabled=True),
        statistics=StatisticsConfig(enabled=True, items=["Z_Score", "ANOVA"]),
        options=OptionsConfig(num_cores=16, time_alignment="per_pair"),
    )
    assert len(cfg.simulation) == 2
    assert cfg.metrics == ["bias", "RMSE", "correlation"]
    assert cfg.comparison.enabled is True
    assert cfg.options.num_cores == 16
    assert cfg.options.time_alignment == "per_pair"
```

- [ ] **Step 2: Write `tests/test_config/__init__.py`**

```python
"""Config test package."""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config/test_schema.py -v`
Expected: FAIL (schema module not found)

- [ ] **Step 4: Write `src/openbench/config/schema.py`**

```python
"""OpenBench configuration schema defined as dataclasses.

Each section of openbench.yaml maps to a dataclass with typed fields
and sensible defaults. The top-level OpenBenchConfig holds everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectConfig:
    """Project identification and spatial-temporal bounds."""

    name: str
    output_dir: str
    years: list[int]  # [start_year, end_year]
    min_year_threshold: int = 3
    lat_range: list[float] = field(default_factory=lambda: [-90.0, 90.0])
    lon_range: list[float] = field(default_factory=lambda: [-180.0, 180.0])


@dataclass
class EvaluationConfig:
    """Which variables to evaluate."""

    variables: list[str]


@dataclass
class SimulationEntry:
    """A single simulation model entry.

    When 'model' matches a known model profile, variable mappings are
    resolved from the registry. Fields here override the profile.
    """

    model: str
    root_dir: str
    data_type: Optional[str] = None
    grid_res: Optional[float] = None
    tim_res: Optional[str] = None
    variables: Optional[dict[str, dict[str, Any]]] = None


@dataclass
class ComparisonConfig:
    """Multi-model comparison settings."""

    enabled: bool = False
    items: Optional[list[str]] = None
    weight: Optional[str] = None
    tim_res: Optional[str] = None
    timezone: Optional[float] = None
    grid_res: Optional[float] = None


@dataclass
class StatisticsConfig:
    """Statistical analysis settings."""

    enabled: bool = False
    items: Optional[list[str]] = None


@dataclass
class OptionsConfig:
    """Global options with sensible defaults."""

    num_cores: Optional[int] = None  # None = auto-detect
    time_alignment: str = "intersection"  # intersection | per_pair | strict
    unified_mask: bool = True
    generate_report: bool = True
    IGBP_groupby: bool = False
    PFT_groupby: bool = False
    climate_zone_groupby: bool = False
    debug_mode: bool = False
    only_drawing: bool = False


@dataclass
class OpenBenchConfig:
    """Top-level configuration container.

    Maps directly to openbench.yaml structure.
    """

    project: ProjectConfig
    evaluation: EvaluationConfig
    reference: dict[str, str]  # variable_name -> registry source name
    simulation: dict[str, SimulationEntry]  # label -> entry

    metrics: Optional[list[str]] = None  # None = all available
    scores: Optional[list[str]] = None  # None = all available
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config/test_schema.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/openbench/config/schema.py tests/test_config/
git commit -m "feat(config): add dataclass-based config schema with defaults"
```

---

### Task 2: Implement YAML Loader with Validation

**Files:**
- Create: `src/openbench/config/loader.py`
- Create: `tests/test_config/test_loader.py`
- Create: `tests/test_config/fixtures/minimal.yaml`
- Create: `tests/test_config/fixtures/full.yaml`
- Create: `tests/test_config/fixtures/with_defaults.yaml`
- Create: `tests/test_config/fixtures/invalid_years.yaml`
- Create: `tests/test_config/fixtures/missing_project.yaml`

- [ ] **Step 1: Create test fixture YAML files**

`tests/test_config/fixtures/minimal.yaml`:
```yaml
project:
  name: test-minimal
  output_dir: ./output
  years: [2004, 2010]

evaluation:
  variables:
    - Evapotranspiration

reference:
  Evapotranspiration: GLEAM_v4.2a

simulation:
  CoLM2024:
    model: CoLM2024
    root_dir: /data/CoLM2024
```

`tests/test_config/fixtures/full.yaml`:
```yaml
project:
  name: test-full
  output_dir: ./output
  years: [2000, 2020]
  min_year_threshold: 5
  lat_range: [-60, 90]
  lon_range: [-180, 180]

evaluation:
  variables:
    - Evapotranspiration
    - GPP
    - Latent_Heat

reference:
  Evapotranspiration: GLEAM_v4.2a
  GPP: FLUXCOM
  Latent_Heat: FLUXCOM

simulation:
  CoLM2024:
    model: CoLM2024
    root_dir: /data/CoLM2024

  CLM5:
    model: CLM5
    root_dir: /data/CLM5
    variables:
      GPP:
        varname: FPSN

metrics: [bias, RMSE, correlation]
scores: [nBiasScore, nRMSEScore]

comparison:
  enabled: true

statistics:
  enabled: true
  items: [Z_Score, ANOVA]

options:
  num_cores: 16
  time_alignment: per_pair
  unified_mask: true
  debug_mode: false
```

`tests/test_config/fixtures/with_defaults.yaml`:
```yaml
project:
  name: test-defaults
  output_dir: ./output
  years: [2004, 2010]

evaluation:
  variables: [Evapotranspiration]

reference:
  Evapotranspiration: GLEAM_v4.2a

simulation:
  _defaults:
    data_type: grid
    grid_res: 0.5
    tim_res: Month

  CoLM2014:
    model: CoLM2014
    root_dir: /data/CoLM2014

  CoLM2024:
    model: CoLM2024
    root_dir: /data/CoLM2024

  CLM5:
    model: CLM5
    root_dir: /data/CLM5
    variables:
      Evapotranspiration:
        varname: QFLX_EVAP_TOT
```

`tests/test_config/fixtures/invalid_years.yaml`:
```yaml
project:
  name: bad
  output_dir: ./output
  years: [2020, 2010]

evaluation:
  variables: [GPP]

reference:
  GPP: FLUXCOM

simulation:
  CoLM2024:
    model: CoLM2024
    root_dir: /data
```

`tests/test_config/fixtures/missing_project.yaml`:
```yaml
evaluation:
  variables: [GPP]

reference:
  GPP: FLUXCOM

simulation:
  CoLM2024:
    model: CoLM2024
    root_dir: /data
```

- [ ] **Step 2: Write `tests/test_config/test_loader.py`**

```python
"""Tests for config loader."""

from pathlib import Path

import pytest

from openbench.config.loader import ConfigError, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_minimal():
    cfg = load_config(FIXTURES / "minimal.yaml")
    assert cfg.project.name == "test-minimal"
    assert cfg.project.years == [2004, 2010]
    assert cfg.evaluation.variables == ["Evapotranspiration"]
    assert cfg.reference["Evapotranspiration"] == "GLEAM_v4.2a"
    assert "CoLM2024" in cfg.simulation
    assert cfg.simulation["CoLM2024"].model == "CoLM2024"
    # Defaults applied
    assert cfg.options.time_alignment == "intersection"
    assert cfg.comparison.enabled is False


def test_load_full():
    cfg = load_config(FIXTURES / "full.yaml")
    assert cfg.project.name == "test-full"
    assert cfg.project.min_year_threshold == 5
    assert cfg.project.lat_range == [-60, 90]
    assert len(cfg.evaluation.variables) == 3
    assert len(cfg.simulation) == 2
    assert cfg.metrics == ["bias", "RMSE", "correlation"]
    assert cfg.scores == ["nBiasScore", "nRMSEScore"]
    assert cfg.comparison.enabled is True
    assert cfg.statistics.enabled is True
    assert cfg.statistics.items == ["Z_Score", "ANOVA"]
    assert cfg.options.num_cores == 16
    assert cfg.options.time_alignment == "per_pair"


def test_load_with_defaults_merge():
    cfg = load_config(FIXTURES / "with_defaults.yaml")
    # _defaults should be merged into each entry
    assert "_defaults" not in cfg.simulation
    assert cfg.simulation["CoLM2014"].data_type == "grid"
    assert cfg.simulation["CoLM2014"].grid_res == 0.5
    assert cfg.simulation["CoLM2014"].tim_res == "Month"
    assert cfg.simulation["CoLM2024"].data_type == "grid"
    # CLM5 has variable override but inherits data_type from _defaults
    assert cfg.simulation["CLM5"].data_type == "grid"
    assert cfg.simulation["CLM5"].variables is not None
    assert cfg.simulation["CLM5"].variables["Evapotranspiration"]["varname"] == "QFLX_EVAP_TOT"


def test_invalid_years_raises():
    with pytest.raises(ConfigError, match="start year.*must be.*end year"):
        load_config(FIXTURES / "invalid_years.yaml")


def test_missing_project_raises():
    with pytest.raises(ConfigError, match="project"):
        load_config(FIXTURES / "missing_project.yaml")


def test_invalid_time_alignment():
    """Construct a config dict with bad time_alignment."""
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
        "options": {"time_alignment": "invalid_value"},
    }
    with pytest.raises(ConfigError, match="time_alignment"):
        _build_config(raw)


def test_unreferenced_variable_warning():
    """Variable in evaluation but not in reference should warn."""
    from openbench.config.loader import _build_config

    raw = {
        "project": {"name": "t", "output_dir": ".", "years": [2000, 2010]},
        "evaluation": {"variables": ["GPP", "Latent_Heat"]},
        "reference": {"GPP": "FLUXCOM"},
        "simulation": {"M": {"model": "M", "root_dir": "/d"}},
    }
    with pytest.raises(ConfigError, match="Latent_Heat"):
        _build_config(raw)


def test_load_nonexistent_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("/nonexistent/file.yaml"))


def test_load_non_yaml():
    """Reject non-YAML files."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b'{"key": "value"}')
        f.flush()
        with pytest.raises(ConfigError, match="YAML"):
            load_config(Path(f.name))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config/test_loader.py -v`
Expected: FAIL (loader module not found)

- [ ] **Step 4: Write `src/openbench/config/loader.py`**

```python
"""YAML configuration loader with validation and _defaults merge.

Public API:
    load_config(path) -> OpenBenchConfig
    ConfigError - raised on validation failure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
    SimulationEntry,
    StatisticsConfig,
)

VALID_TIME_ALIGNMENTS = {"intersection", "per_pair", "strict"}


class ConfigError(Exception):
    """Raised when config loading or validation fails."""


class _IncludeLoader(yaml.SafeLoader):
    """YAML loader with !include tag support."""


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    """Handle !include tags in YAML files."""
    path_str = loader.construct_scalar(node)
    # Resolve relative to the YAML file's directory
    base_dir = Path(loader.name).parent if hasattr(loader, "name") else Path(".")
    target = base_dir / path_str

    if "*" in path_str:
        # Glob pattern: !include sim/*.yaml
        import glob

        result = {}
        for match in sorted(glob.glob(str(target))):
            with open(match) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    result.update(data)
        return result
    else:
        # Single file: !include ref.yaml
        with open(target) as f:
            return yaml.safe_load(f)


_IncludeLoader.add_constructor("!include", _include_constructor)


def load_config(path: str | Path) -> OpenBenchConfig:
    """Load and validate an openbench.yaml config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated OpenBenchConfig instance.

    Raises:
        ConfigError: If the file is invalid or validation fails.
    """
    path = Path(path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ConfigError(
            f"Only YAML config files are supported (got {path.suffix}). "
            "Use 'openbench migrate' to convert old JSON/NML configs."
        )

    try:
        with open(path) as f:
            raw = yaml.load(f, Loader=_IncludeLoader)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> OpenBenchConfig:
    """Build and validate an OpenBenchConfig from a raw dict."""
    # --- project (required) ---
    if "project" not in raw:
        raise ConfigError("Missing required section: 'project'")
    project = _build_project(raw["project"])

    # --- evaluation (required) ---
    if "evaluation" not in raw:
        raise ConfigError("Missing required section: 'evaluation'")
    evaluation = _build_evaluation(raw["evaluation"])

    # --- reference (required) ---
    if "reference" not in raw:
        raise ConfigError("Missing required section: 'reference'")
    reference = raw["reference"]
    if not isinstance(reference, dict):
        raise ConfigError("'reference' must be a mapping of variable -> source name")

    # Check all evaluation variables have a reference
    for var in evaluation.variables:
        if var not in reference:
            raise ConfigError(
                f"Variable '{var}' is in evaluation.variables but has no entry in 'reference'. "
                f"Add: reference.{var}: <source_name>"
            )

    # --- simulation (required) ---
    if "simulation" not in raw:
        raise ConfigError("Missing required section: 'simulation'")
    simulation = _build_simulation(raw["simulation"])

    # --- optional sections ---
    metrics = raw.get("metrics")
    scores = raw.get("scores")
    comparison = _build_comparison(raw.get("comparison", {}))
    statistics = _build_statistics(raw.get("statistics", {}))
    options = _build_options(raw.get("options", {}))

    return OpenBenchConfig(
        project=project,
        evaluation=evaluation,
        reference=reference,
        simulation=simulation,
        metrics=metrics,
        scores=scores,
        comparison=comparison,
        statistics=statistics,
        options=options,
    )


def _build_project(raw: dict[str, Any]) -> ProjectConfig:
    """Build and validate ProjectConfig."""
    required = ["name", "output_dir", "years"]
    for key in required:
        if key not in raw:
            raise ConfigError(f"Missing required field: project.{key}")

    years = raw["years"]
    if not isinstance(years, list) or len(years) != 2:
        raise ConfigError("project.years must be a list of [start_year, end_year]")
    if years[0] > years[1]:
        raise ConfigError(f"project.years start year ({years[0]}) must be <= end year ({years[1]})")

    return ProjectConfig(
        name=raw["name"],
        output_dir=raw["output_dir"],
        years=years,
        min_year_threshold=raw.get("min_year_threshold", 3),
        lat_range=raw.get("lat_range", [-90.0, 90.0]),
        lon_range=raw.get("lon_range", [-180.0, 180.0]),
    )


def _build_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    """Build and validate EvaluationConfig."""
    if "variables" not in raw:
        raise ConfigError("Missing required field: evaluation.variables")
    variables = raw["variables"]
    if not isinstance(variables, list) or len(variables) == 0:
        raise ConfigError("evaluation.variables must be a non-empty list")
    return EvaluationConfig(variables=variables)


def _build_simulation(raw: dict[str, Any]) -> dict[str, SimulationEntry]:
    """Build simulation entries with _defaults merge."""
    defaults = raw.pop("_defaults", {}) if isinstance(raw, dict) else {}
    result = {}

    for label, entry in raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"simulation.{label} must be a mapping")

        # Merge defaults: defaults are base, entry overrides
        merged = {**defaults, **entry}

        # For 'variables', do shallow-per-variable merge
        if "variables" in defaults and "variables" in entry:
            merged_vars = {**defaults.get("variables", {}), **entry["variables"]}
            merged["variables"] = merged_vars

        if "model" not in merged:
            raise ConfigError(f"simulation.{label} must have a 'model' field")
        if "root_dir" not in merged:
            raise ConfigError(f"simulation.{label} must have a 'root_dir' field")

        result[label] = SimulationEntry(
            model=merged["model"],
            root_dir=merged["root_dir"],
            data_type=merged.get("data_type"),
            grid_res=merged.get("grid_res"),
            tim_res=merged.get("tim_res"),
            variables=merged.get("variables"),
        )

    return result


def _build_comparison(raw: Any) -> ComparisonConfig:
    """Build ComparisonConfig from raw dict or None."""
    if not raw or not isinstance(raw, dict):
        return ComparisonConfig()
    return ComparisonConfig(
        enabled=raw.get("enabled", False),
        items=raw.get("items"),
        weight=raw.get("weight"),
        tim_res=raw.get("tim_res"),
        timezone=raw.get("timezone"),
        grid_res=raw.get("grid_res"),
    )


def _build_statistics(raw: Any) -> StatisticsConfig:
    """Build StatisticsConfig from raw dict or None."""
    if not raw or not isinstance(raw, dict):
        return StatisticsConfig()
    return StatisticsConfig(
        enabled=raw.get("enabled", False),
        items=raw.get("items"),
    )


def _build_options(raw: Any) -> OptionsConfig:
    """Build and validate OptionsConfig."""
    if not raw or not isinstance(raw, dict):
        return OptionsConfig()

    time_alignment = raw.get("time_alignment", "intersection")
    if time_alignment not in VALID_TIME_ALIGNMENTS:
        raise ConfigError(
            f"options.time_alignment must be one of {VALID_TIME_ALIGNMENTS}, got '{time_alignment}'"
        )

    return OptionsConfig(
        num_cores=raw.get("num_cores"),
        time_alignment=time_alignment,
        unified_mask=raw.get("unified_mask", True),
        generate_report=raw.get("generate_report", True),
        IGBP_groupby=raw.get("IGBP_groupby", False),
        PFT_groupby=raw.get("PFT_groupby", False),
        climate_zone_groupby=raw.get("climate_zone_groupby", False),
        debug_mode=raw.get("debug_mode", False),
        only_drawing=raw.get("only_drawing", False),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config/test_loader.py -v`
Expected: All PASS

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS (schema + loader + smoke + cli stubs)

- [ ] **Step 7: Commit**

```bash
git add src/openbench/config/loader.py tests/test_config/test_loader.py tests/test_config/fixtures/
git commit -m "feat(config): add YAML loader with validation and _defaults merge"
```

---

### Task 3: Wire Up `openbench check` CLI Command

**Files:**
- Modify: `src/openbench/cli/check.py`
- Modify: `src/openbench/config/__init__.py`

- [ ] **Step 1: Update `src/openbench/config/__init__.py`**

```python
"""Configuration loading, validation, and schema.

Public API:
    load_config(path) -> OpenBenchConfig
    ConfigError - validation error
    OpenBenchConfig, ProjectConfig, ... - schema dataclasses
"""

from openbench.config.loader import ConfigError, load_config
from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    OptionsConfig,
    ProjectConfig,
    SimulationEntry,
    StatisticsConfig,
)

__all__ = [
    "load_config",
    "ConfigError",
    "OpenBenchConfig",
    "ProjectConfig",
    "EvaluationConfig",
    "SimulationEntry",
    "ComparisonConfig",
    "StatisticsConfig",
    "OptionsConfig",
]
```

- [ ] **Step 2: Update `src/openbench/cli/check.py`**

```python
"""openbench check command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
def check(config):
    """Validate config file and check data availability."""
    from openbench.config import ConfigError, load_config

    try:
        cfg = load_config(config)
    except ConfigError as e:
        click.secho(f"  ✗ {e}", fg="red")
        raise SystemExit(1)

    click.secho("Config validation:", bold=True)
    click.secho("  ✓ YAML syntax valid", fg="green")
    click.secho("  ✓ Schema validation passed", fg="green")
    click.secho(
        f"  ✓ Year range [{cfg.project.years[0]}, {cfg.project.years[1]}] valid",
        fg="green",
    )

    click.secho(f"\nReference data ({len(cfg.reference)} sources):", bold=True)
    for var, source in cfg.reference.items():
        click.secho(f"  ✓ {var} → {source}", fg="green")

    click.secho(f"\nSimulation data ({len(cfg.simulation)} models):", bold=True)
    for label, entry in cfg.simulation.items():
        click.secho(f"  ✓ {label} (model: {entry.model}, root: {entry.root_dir})", fg="green")

    if cfg.metrics:
        click.secho(f"\nMetrics: {', '.join(cfg.metrics)}", bold=True)
    if cfg.scores:
        click.secho(f"Scores: {', '.join(cfg.scores)}", bold=True)

    click.secho(f"\nOptions:", bold=True)
    click.secho(f"  Time alignment: {cfg.options.time_alignment}")
    click.secho(f"  Unified mask: {cfg.options.unified_mask}")
    click.secho(f"  Comparison: {cfg.comparison.enabled}")
    click.secho(f"  Statistics: {cfg.statistics.enabled}")

    click.secho(f"\n✓ Config valid. Ready to run.", fg="green", bold=True)
```

- [ ] **Step 3: Test manually**

Run:
```bash
openbench check tests/test_config/fixtures/minimal.yaml
```

Expected: Green output showing validation passed.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/openbench/config/__init__.py src/openbench/cli/check.py
git commit -m "feat(cli): wire openbench check to config loader with validation output"
```

---

### Task 4: Implement Migration Tool

**Files:**
- Create: `src/openbench/config/migration.py`
- Create: `tests/test_config/test_migration.py`
- Create: `tests/test_config/fixtures/old_json/main.json`
- Create: `tests/test_config/fixtures/old_json/ref.json`
- Create: `tests/test_config/fixtures/old_json/sim.json`
- Create: `tests/test_config/fixtures/old_json/ref_def/GLEAM.json`
- Create: `tests/test_config/fixtures/old_json/sim_def/case1.json`
- Modify: `src/openbench/cli/migrate.py`

- [ ] **Step 1: Create old-format test fixtures**

`tests/test_config/fixtures/old_json/main.json`:
```json
{
    "general": {
        "basename": "test-migrate",
        "basedir": "./output",
        "syear": 2004,
        "eyear": 2010,
        "min_year": 3,
        "min_lon": -180,
        "max_lon": 180,
        "min_lat": -90,
        "max_lat": 90,
        "reference_nml": "./tests/test_config/fixtures/old_json/ref.json",
        "simulation_nml": "./tests/test_config/fixtures/old_json/sim.json",
        "num_cores": 8,
        "evaluation": true,
        "comparison": true,
        "statistics": false,
        "IGBP_groupby": false,
        "PFT_groupby": false,
        "Climate_zone_groupby": false,
        "unified_mask": true,
        "generate_report": true
    },
    "evaluation_items": {
        "Evapotranspiration": true,
        "GPP": false,
        "Latent_Heat": true
    },
    "metrics": {
        "bias": true,
        "RMSE": true,
        "correlation": true,
        "percent_bias": false
    },
    "scores": {
        "nBiasScore": true,
        "nRMSEScore": true,
        "Overall_Score": false
    },
    "comparisons": {
        "Taylor_Diagram": true,
        "HeatMap": false
    }
}
```

`tests/test_config/fixtures/old_json/ref.json`:
```json
{
    "general": {
        "Evapotranspiration_ref_source": "GLEAM_v4.2a",
        "Latent_Heat_ref_source": "FLUXCOM"
    },
    "def_nml": {
        "GLEAM_v4.2a": "./tests/test_config/fixtures/old_json/ref_def/GLEAM.json",
        "FLUXCOM": "./tests/test_config/fixtures/old_json/ref_def/GLEAM.json"
    }
}
```

`tests/test_config/fixtures/old_json/sim.json`:
```json
{
    "general": {
        "Evapotranspiration_sim_source": ["CoLM2024"],
        "Latent_Heat_sim_source": ["CoLM2024"]
    },
    "def_nml": {
        "CoLM2024": "./tests/test_config/fixtures/old_json/sim_def/case1.json"
    }
}
```

`tests/test_config/fixtures/old_json/ref_def/GLEAM.json`:
```json
{
    "general": {
        "root_dir": "./dataset/Reference/Grid",
        "data_type": "grid",
        "syear": 1980,
        "eyear": 2023,
        "tim_res": "Month",
        "grid_res": 0.25,
        "timezone": 0,
        "data_groupby": "Year"
    },
    "Evapotranspiration": {
        "sub_dir": "Water/Evapotranspiration/GLEAM",
        "varname": "E",
        "varunit": "mm day-1",
        "prefix": "E_",
        "suffix": "_GLEAM"
    }
}
```

`tests/test_config/fixtures/old_json/sim_def/case1.json`:
```json
{
    "general": {
        "root_dir": "/data/CoLM2024/output",
        "data_type": "grid",
        "syear": 2004,
        "eyear": 2010,
        "tim_res": "Month",
        "grid_res": 0.5,
        "timezone": 0,
        "data_groupby": "Year"
    },
    "Evapotranspiration": {
        "varname": "ET",
        "varunit": "mm day-1",
        "prefix": "",
        "suffix": ""
    },
    "Latent_Heat": {
        "varname": "Qle",
        "varunit": "W m-2",
        "prefix": "",
        "suffix": ""
    }
}
```

- [ ] **Step 2: Write `tests/test_config/test_migration.py`**

```python
"""Tests for config migration tool."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from openbench.config.migration import migrate_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_migrate_json_config():
    """Migrate old JSON config set to new YAML format."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        output_path = Path(f.name)

    result = migrate_config(FIXTURES / "old_json" / "main.json", output_path)

    assert output_path.exists()
    with open(output_path) as f:
        data = yaml.safe_load(f)

    # Check project section
    assert data["project"]["name"] == "test-migrate"
    assert data["project"]["output_dir"] == "./output"
    assert data["project"]["years"] == [2004, 2010]

    # Check evaluation - only true items
    assert "Evapotranspiration" in data["evaluation"]["variables"]
    assert "Latent_Heat" in data["evaluation"]["variables"]
    assert "GPP" not in data["evaluation"]["variables"]

    # Check reference
    assert "Evapotranspiration" in data["reference"]

    # Check simulation
    assert "CoLM2024" in data["simulation"]
    assert data["simulation"]["CoLM2024"]["root_dir"] == "/data/CoLM2024/output"

    # Check metrics - only true ones
    assert "bias" in data["metrics"]
    assert "RMSE" in data["metrics"]
    assert "percent_bias" not in data["metrics"]

    # Check scores - only true ones
    assert "nBiasScore" in data["scores"]
    assert "Overall_Score" not in data["scores"]

    # Check comparison
    assert data["comparison"]["enabled"] is True

    # Check options
    assert data["options"]["num_cores"] == 8

    # Validate the output can be loaded by the new loader
    from openbench.config.loader import load_config

    cfg = load_config(output_path)
    assert cfg.project.name == "test-migrate"

    output_path.unlink()


def test_migrate_returns_summary():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        output_path = Path(f.name)

    result = migrate_config(FIXTURES / "old_json" / "main.json", output_path)

    assert "files_read" in result
    assert result["files_read"] >= 1
    assert "variables" in result
    assert "simulations" in result

    output_path.unlink()


def test_migrate_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        migrate_config(Path("/no/such/file.json"), Path("/tmp/out.yaml"))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config/test_migration.py -v`
Expected: FAIL (migration module not found)

- [ ] **Step 4: Write `src/openbench/config/migration.py`**

```python
"""Migrate old JSON/NML configs to the new unified YAML format.

Reads the old multi-file config structure (main + ref + sim + variable defs)
and produces a single openbench.yaml in the new format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def migrate_config(main_config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convert old-format config to new unified YAML.

    Args:
        main_config_path: Path to the old main config file (JSON, YAML, or NML).
        output_path: Where to write the new openbench.yaml.

    Returns:
        Summary dict with migration statistics.

    Raises:
        FileNotFoundError: If the main config file doesn't exist.
    """
    main_config_path = Path(main_config_path)
    output_path = Path(output_path)

    if not main_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {main_config_path}")

    # Read main config
    main = _read_old_config(main_config_path)
    files_read = 1

    general = main.get("general", {})
    base_dir = main_config_path.parent

    # Read reference config
    ref_sources = {}
    ref_nml_path = general.get("reference_nml")
    if ref_nml_path:
        ref_config = _read_old_config(_resolve_path(ref_nml_path, base_dir))
        files_read += 1
        ref_general = ref_config.get("general", {})
        ref_def_nml = ref_config.get("def_nml", {})

        # Extract source names per variable
        for key, value in ref_general.items():
            if key.endswith("_ref_source"):
                var_name = key.replace("_ref_source", "")
                ref_sources[var_name] = value

    # Read simulation config
    sim_entries = {}
    sim_nml_path = general.get("simulation_nml")
    sim_def_configs = {}
    if sim_nml_path:
        sim_config = _read_old_config(_resolve_path(sim_nml_path, base_dir))
        files_read += 1
        sim_general = sim_config.get("general", {})
        sim_def_nml = sim_config.get("def_nml", {})

        # Collect all unique simulation source names
        all_sim_sources = set()
        for key, value in sim_general.items():
            if key.endswith("_sim_source"):
                sources = value if isinstance(value, list) else [value]
                all_sim_sources.update(sources)

        # Read each simulation definition file
        for source_name in all_sim_sources:
            if source_name in sim_def_nml:
                def_path = _resolve_path(sim_def_nml[source_name], base_dir)
                if def_path.exists():
                    sim_def = _read_old_config(def_path)
                    files_read += 1
                    sim_def_configs[source_name] = sim_def

                    sim_general_def = sim_def.get("general", {})
                    variables = {}
                    for var_key, var_val in sim_def.items():
                        if var_key != "general" and isinstance(var_val, dict):
                            variables[var_key] = var_val

                    sim_entries[source_name] = {
                        "model": source_name,
                        "root_dir": sim_general_def.get("root_dir", sim_general_def.get("dir", "")),
                        "data_type": sim_general_def.get("data_type"),
                        "grid_res": sim_general_def.get("grid_res"),
                        "tim_res": sim_general_def.get("tim_res"),
                    }
                    if variables:
                        sim_entries[source_name]["variables"] = variables

    # Build new config
    eval_items = main.get("evaluation_items", {})
    enabled_variables = [k for k, v in eval_items.items() if v]

    metrics_dict = main.get("metrics", {})
    enabled_metrics = [k for k, v in metrics_dict.items() if v]

    scores_dict = main.get("scores", {})
    enabled_scores = [k for k, v in scores_dict.items() if v]

    comparisons_dict = main.get("comparisons", {})
    enabled_comparisons = [k for k, v in comparisons_dict.items() if v]

    statistics_dict = main.get("statistics_items", main.get("statistics", {}))
    if isinstance(statistics_dict, dict):
        enabled_statistics = [k for k, v in statistics_dict.items() if isinstance(v, bool) and v]
    else:
        enabled_statistics = []

    # Filter reference to only enabled variables
    filtered_ref = {var: ref_sources[var] for var in enabled_variables if var in ref_sources}

    new_config: dict[str, Any] = {
        "project": {
            "name": general.get("basename", "migrated"),
            "output_dir": general.get("basedir", "./output"),
            "years": [general.get("syear", 2000), general.get("eyear", 2020)],
        },
        "evaluation": {
            "variables": enabled_variables,
        },
        "reference": filtered_ref,
        "simulation": sim_entries if sim_entries else {"default": {"model": "unknown", "root_dir": "."}},
    }

    if enabled_metrics:
        new_config["metrics"] = enabled_metrics
    if enabled_scores:
        new_config["scores"] = enabled_scores

    comparison_enabled = general.get("comparison", False)
    if comparison_enabled or enabled_comparisons:
        comp: dict[str, Any] = {"enabled": bool(comparison_enabled)}
        if enabled_comparisons:
            comp["items"] = enabled_comparisons
        new_config["comparison"] = comp

    statistics_enabled = general.get("statistics", False)
    if isinstance(statistics_enabled, bool) and (statistics_enabled or enabled_statistics):
        stat: dict[str, Any] = {"enabled": bool(statistics_enabled)}
        if enabled_statistics:
            stat["items"] = enabled_statistics
        new_config["statistics"] = stat

    # Options
    options: dict[str, Any] = {}
    if general.get("num_cores"):
        options["num_cores"] = general["num_cores"]
    if general.get("unified_mask") is not None:
        options["unified_mask"] = general["unified_mask"]
    if general.get("generate_report") is not None:
        options["generate_report"] = general["generate_report"]
    if general.get("IGBP_groupby"):
        options["IGBP_groupby"] = True
    if general.get("PFT_groupby"):
        options["PFT_groupby"] = True
    if general.get("Climate_zone_groupby"):
        options["climate_zone_groupby"] = True
    if options:
        new_config["options"] = options

    # Add min_year_threshold and spatial bounds if non-default
    if general.get("min_year"):
        new_config["project"]["min_year_threshold"] = general["min_year"]
    lat_range = [general.get("min_lat", -90), general.get("max_lat", 90)]
    lon_range = [general.get("min_lon", -180), general.get("max_lon", 180)]
    if lat_range != [-90, 90]:
        new_config["project"]["lat_range"] = lat_range
    if lon_range != [-180, 180]:
        new_config["project"]["lon_range"] = lon_range

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return {
        "files_read": files_read,
        "variables": enabled_variables,
        "simulations": list(sim_entries.keys()),
        "metrics": enabled_metrics,
        "scores": enabled_scores,
    }


def _read_old_config(path: Path) -> dict:
    """Read an old config file (JSON or YAML)."""
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix == ".json":
            return json.load(f)
        elif suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        elif suffix == ".nml":
            # Minimal NML support: try f90nml if available, otherwise error
            try:
                import f90nml

                nml = f90nml.read(path)
                return dict(nml)
            except ImportError:
                raise ImportError(
                    "Migrating Fortran NML files requires f90nml: pip install f90nml"
                )
        else:
            raise ValueError(f"Unsupported config format: {suffix}")


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir if not absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return base_dir / p
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config/test_migration.py -v`
Expected: All PASS

- [ ] **Step 6: Wire up `openbench migrate` CLI command**

Update `src/openbench/cli/migrate.py`:

```python
"""openbench migrate command."""

import click


@click.command()
@click.argument("old_config", type=click.Path(exists=True))
@click.option("-o", "--output", default="openbench.yaml", help="Output file path.")
def migrate(old_config, output):
    """Convert old JSON/NML config to unified YAML."""
    from pathlib import Path

    from openbench.config.migration import migrate_config

    try:
        result = migrate_config(Path(old_config), Path(output))
    except Exception as e:
        click.secho(f"Migration failed: {e}", fg="red")
        raise SystemExit(1)

    click.secho(f"✓ Read {result['files_read']} config files", fg="green")
    click.secho(f"✓ {len(result['variables'])} evaluation variables", fg="green")
    click.secho(f"✓ {len(result['simulations'])} simulation models", fg="green")
    click.secho(f"✓ Written to {output}", fg="green", bold=True)
    click.echo(f"\nNext: openbench check {output}")
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/openbench/config/migration.py src/openbench/cli/migrate.py tests/test_config/test_migration.py tests/test_config/fixtures/old_json/
git commit -m "feat(config): add migration tool to convert old JSON/NML to new YAML"
```

---

### Task 5: Final Integration Test and Lint

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run lint**

Run:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

If format issues: `ruff format src/ tests/` then re-check.

- [ ] **Step 3: Test openbench check end-to-end**

Run:
```bash
openbench check tests/test_config/fixtures/minimal.yaml
openbench check tests/test_config/fixtures/full.yaml
```

Expected: Green validation output for both.

- [ ] **Step 4: Test openbench migrate end-to-end**

Run:
```bash
openbench migrate tests/test_config/fixtures/old_json/main.json -o /tmp/migrated.yaml
cat /tmp/migrated.yaml
openbench check /tmp/migrated.yaml
```

Expected: Migration succeeds, output YAML is valid, check passes.

- [ ] **Step 5: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: address lint and integration issues in config system"
```

Only commit if there were changes to fix.

- [ ] **Step 6: Tag milestone**

```bash
git tag -a v3.0.0a2 -m "Sub-project 2 complete: config system with schema, loader, migration"
```
