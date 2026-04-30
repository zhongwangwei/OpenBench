# OpenBench Naming Conventions

## Year Range

| Context | Convention | Example |
|---------|-----------|---------|
| New config schema (`config/schema.py`) | `years: list[int]` | `[2004, 2010]` |
| Adapter output (`config/adapter.py`) | `syear`, `eyear` | `syear=2004, eyear=2010` |
| Evaluator runtime (`core/evaluation.py`) | `syear`, `eyear` | From GeneralInfoReader |

**Rule:** New code uses `years` (list). Adapter and evaluator-facing runtime structures still use `syear`/`eyear` because the evaluation engine depends on these exact names.

## Variable Names

| Context | Convention | Example |
|---------|-----------|---------|
| Standard (directory/config) | Underscore-separated | `Evapotranspiration`, `Latent_Heat` |
| NetCDF variable name | Dataset-specific | `E`, `Qle`, `slhf` |

## Module Naming

| Context | Convention | Example |
|---------|-----------|---------|
| New modules | `snake_case.py` | `local.py`, `cache.py`, `schema.py` |
| Migrated compatibility modules | Original name preserved | `Mod_Statistics.py`, `Fig_*.py` |

**Rule:** New code uses `snake_case`. Older migrated files keep their original names to avoid breaking internal references.

## Config Key Naming

| Context | Convention |
|---------|-----------|
| New config (`openbench.yaml`) | `snake_case` (`output_dir`, `time_alignment`) |
| Adapter output for evaluator/runtime inputs | Historical evaluator keys (`compare_tim_res`, `IGBP_groupby`) |

## Resolution Suffixes

Registry datasets use resolution suffixes:
- `GLEAM_v4.2a_LowRes`
- `GLEAM_v4.2a_MidRes`
- `GLEAM_v4.2a_HigRes`

Base name (without suffix) is used for auto-resolution only.
