# OpenBench Naming Conventions

## Year Range

| Context | Convention | Example |
|---------|-----------|---------|
| New config schema (`config/schema.py`) | `years: list[int]` | `[2004, 2010]` |
| Adapter output (`config/adapter.py`) | `syear`, `eyear` | `syear=2004, eyear=2010` |
| Evaluator runtime (`core/evaluation.py`) | `syear`, `eyear` | From GeneralInfoReader |

**Rule:** New code uses `years` (list). Adapter and evaluator-facing runtime structures still use `syear`/`eyear` because the evaluation engine depends on these exact names.

Reference variable mappings may specify their own `years: [start, end]`; otherwise
they inherit the dataset coverage. Explicit config variable overrides win over
config dataset overrides, then catalog variable coverage, then catalog dataset
coverage. Each selected binding passes its own years to the evaluator; strict
time alignment still checks the requested project years. Narrowing registration
does not delete out-of-range files.

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

## Station Water-Surface Elevation Datum

`Water_Surface_Elevation` is the sole logical variable eligible for the opt-in
EGM2008-to-EGM96 station correction. It is not an alias for water depth,
`River_Water_Level`, `Lake_Water_Level` or dam-structure elevation.
Observations and model output must use `m`; the model/evaluation datum is EGM96.
Configure `vertical_datum` only on the reference variable with `source_geoid`
(EGM2008 PGM) and `target_geoid` (EGM96 PGM). Simulation overrides are rejected
by the YAML loader. The correction is `H96 = H2008 + N2008 - N96`, in memory
before metrics and plots; source and cached NetCDF files remain unchanged.
With this option, station-only `only_drawing` refreshes metrics/plots from those
cached inputs rather than reusing potentially uncorrected tables. Normal station
comparisons use the same correction when loading a pair. With selected datum
correction, `--comparison-only` is rejected; `only_drawing` with enabled comparisons
or report generation is also rejected before any task runs. Use normal evaluation
(without `--comparison-only`, with `only_drawing: false`) to regenerate comparison
aggregates, or set both `comparison.enabled: false` and
`project.generate_report: false` for station-only redraws. Other variables and ordinary
`only_drawing` behavior are unchanged.

The GUI lists the variable, but geoid paths remain manual configuration fields;
GUI import/export preserves them. No bundled observation dataset is invented
while the station data are unavailable. See
[station vertical datum configuration](../../docs/manual/station_vertical_datum.md).

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

## Reference physical-definition boundaries

Reference availability does not imply a compatible default model output. Supply
an explicit custom model mapping with the same physical definition and area basis
for the following evaluation items; OpenBench does not invent these mappings.

- `Ground_Frost_Frequency` (CRU TS 4.08) is the monthly number of frost days,
  in `days`, potentially fractional because it is an estimated field. Use a
  same-definition preaggregated monthly model count, not the monthly mean of a
  daily indicator or an unverified minimum-temperature threshold.
- HOMTS `Soil_Temperature_{5,10,15,20,40}cm` are temperatures at point depths,
  not layer means or a 0–1 m root-zone mean. The historical physical directory
  `Root_Zone_Soil_Temperature/HOMTS` is retained without changing this definition.
  The existing surface 0 cm binding is unchanged; the alternate 0 cm file is not
  registered again. Model point depths require verified depth coordinates and
  an explicit mapping, not reuse of a top-layer or root-zone average.
  The bundled CoLM2024 mapping is a derived point-temperature approximation:
  bounded linear interpolation between soil centers from the standard
  [CoLM2024.1 grid](https://github.com/CoLM-SYSU/CoLM/blob/0db2bacfb863398074d57d680e0264e4907aaf4b/main/MOD_Vars_Global.F90#L137-L153),
  using native `f_t_soisno` in K with exactly 15 `soilsnow` slots (5 snow + 10 soil).
  It does not use `f_t_soisno_5cm`, which is a layer mean, or extrapolate.
  This is conditional on that standard layout; an actual user output has not
  been validated. The default monthly model cadence requires monthly aggregation
  of HOMTS; monthly output cannot recover daily temperatures.
- `UpCH4_v0.4_WAD2M_LowRes.Wetland_Methane_Emission` is the ensemble-mean
  freshwater vegetated-wetland contribution per **total grid-cell area**.
  Wetland fraction has already been applied: do not apply it again. Open-water
  and coastal wetlands are excluded. It is not generic all-source `Methane`.
  A model may use its own wetland fraction, but must provide the corresponding
  wetland contribution per total grid-cell area. Raw products and uncertainty
  fields are intentionally excluded from automatic registration.
- HSWUD v8 `Domestic_Water_Withdrawal`, `Manufacturing_Water_Withdrawal`,
  `Irrigation_Water_Withdrawal`, and `Thermoelectric_Water_Withdrawal` are gross
  withdrawals expressed as `mm day-1` over total grid-cell area. They are not
  consumptive use. Irrigation includes conveyance losses and is not interchangeable
  with net irrigation applied to soil (`Total_Irrigation_Amount`); thermoelectric
  withdrawal refers to cooling water. Only processed v8 files are scan candidates.

HSWUD v8 retains its published unequal latitude/longitude spacings
(0.1002857143 / 0.1001639344 degrees). A scalar `grid_res` is deliberately
omitted rather than advertising an exact 0.1-degree grid. Derived full-cell
areas use the source coordinate midpoints and WGS84 geometry, with assumptions
recorded in the NetCDF; no coordinate relocation or implicit regridding is done.
