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

Logical aliases are accepted case-insensitively: `Transpiration` becomes
`Canopy_Transpiration`, `Canopy_Interception` becomes `Canopy_Evaporation`,
and `Runoff` becomes `Total_Runoff`. Surface and
subsurface runoff remain separate variables. Native NetCDF names are not renamed.
This legacy interception label denotes evaporation loss, not precipitation
intercepted onto the canopy (for example, CLM `QINTR` is not `QVEGE`).
Conflicting definitions are rejected; logical aliases do not merge data folders.

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
- `GLEAM_v4.3a_LowRes`
- `GLEAM_v4.3a_MidRes`
- `GLEAM_v4.3a_HigRes`

Base name (without suffix) is used for auto-resolution only.

## Root-Zone Soil Temperature

`CoLM2024` (including its `CoLM` alias) derives `Root_Zone_Soil_Temperature`
in kelvin from native `f_t_soisno`. It requires the default 15-position
`soilsnow` dimension: five snow positions followed by ten soil layers, with
`t_soisno` enabled in the history namelist. Weights are 0–1 m layer-thickness
intersections, including only the overlapping part of soil layer 8, not
vegetation root fractions. Missing values in any used layer propagate; snow
and deeper layers do not participate. Custom vertical grids need a custom
mapping. The native grid and history dimensions follow
[CoLM2024.1](https://github.com/CoLM-SYSU/CoLM/blob/0db2bacfb863398074d57d680e0264e4907aaf4b/main/MOD_Vars_Global.F90#L137-L153).

The SMAP reference root-zone temperature is a different, explicitly approximate
diagnostic: nominal thermal-layer overlaps with the top boundary assumed at
zero. Surface soil temperature is the first model thermal layer, not skin
temperature or a uniform 0–5 cm mean. Do not substitute either diagnostic for
a native observed temperature without accounting for these definitions.
