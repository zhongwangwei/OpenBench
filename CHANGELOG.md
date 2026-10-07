# Changelog

## [Unreleased]

### Added
- Rescanning checks preserved station-matching variable names against the
  dataset, warning about missing fields with available names while keeping
  user settings unchanged. A malformed block (not a mapping, or a file or
  variable name that is not text) is reported the same way and never stops
  the rescan of the other datasets.
- `openbench init` accepts comma-separated reference numbers or names for each
  variable. All selected references are checked and evaluated separately;
  single selections and `0` to skip remain supported.
- River sediment references OpenBench_Sediment_Daily,
  OpenBench_Sediment_Monthly and OpenBench_Sediment_Annual (SedRef v1.0.0,
  CC BY 4.0) with three evaluation items:
  `Discharge_For_Sediment` (river discharge at the sediment gauges, kept apart
  from `Streamflow`), `Suspended_Sediment_Concentration` (mg L-1) and
  `Suspended_Sediment_Load` (t d-1). They are matched to the river network
  like Streamflow: CaMa allocation, the fixed minimum upstream area per
  resolution, allocation error at most 0.2, and the `_dist` fallback. The data
  are not bundled; they go under
  `${OPENBENCH_REF_ROOT}/Station/Water/Sediment/<Daily|Monthly|Annual>`. The
  GUI lists the items in a new Sediment group.
- CoLM2024 maps the sediment items to Grid_RiverLake routing output:
  `f_discharge`, and the three size classes of CoLM's standard sediment
  parameters (`f_sedcon_1..3`, `f_sedout_1..3`) summed and multiplied by its
  default grain density of 2650 kg m-3. A run with another number of classes
  or another density needs its own `compute` expression.
- Compute expressions can sum numbered parts with
  `ds.sum_prefix('f_sedcon_', 3)`. Exactly parts 1..3 must be in the data
  computed: a missing part (for example one kept in another file) or a
  higher-numbered one stops the evaluation instead of giving a partial sum.
  File lookup and simulation scanning treat the parts as dependencies, like
  `ds['name']`.
- Unit conversions for sediment concentration (mg L-1, g m-3, kg m-3, g L-1)
  and load (t d-1, kg s-1, kg d-1, t yr-1).
- Thirty reference datasets for land-surface CH4 emission and the areas that
  emit it: FLUXNET-CH4 monthly tower fluxes (Methane), the 22 GCP-CH4
  wetland runs and their ensemble mean (Wetland_Methane_Emission,
  Wetland_Fraction), WAD2M and GIEMS-MC wetland fraction, Johnson et al.
  (2022) lake CH4 flux and lake fraction, MIRCA2000 rice area, GRPI rice CH4
  flux and GLWD v2.0 lake, wetland and rice fraction. The data are not
  bundled. Of the new items only Methane can be selected in the GUI so far.
- `KGEln`, the Kling-Gupta efficiency with log ratios of variability and
  mean, so that over- and underestimation by the same factor weigh the same.
- A Streamflow station dataset registered as `<name>_full.nc` falls back to
  its redistributable subset `<name>_dist.nc` in the same directory when the
  full file is absent, so the OpenBench_Streamflow references run from either
  file. `openbench init` accepts either file too.

### Changed
- Station matching reads the evaluated item's `varname` from the dataset,
  falling back to `station_matching.discharge_var`, and writes the station
  files under that name, so one dataset can serve several items. The
  Streamflow references now give their dataset variable as `varname` (for
  example `Disch` for GRDC), which also names their output files
  (`Streamflow_ref_GRDC_Daily_Disch.nc`). A user catalog that still says
  `discharge` keeps working through the fallback, which only Streamflow has:
  a sediment item whose variable is missing from the dataset stops with an
  error instead of reading discharge in its place.
- The OpenBench_Streamflow_Monthly and OpenBench_Streamflow_Daily references
  now list years 1806-2026, and OpenBench_Streamflow_Hourly 1909-2026, the
  coverage of the 2026-10-06 release.
- OpenBench_Streamflow_Daily and OpenBench_Streamflow_Hourly read the upstream
  area from `upstream_area`, as OpenBench_Streamflow_Monthly does. A station
  dataset without its configured area variable now falls back to
  `upstream_area` or `area`, and warns when it has neither. It used to skip
  the minimum upstream area without a word.
- CH4 fluxes in kg CH4 m-2 s-1; g CH4 m-2 per second, day or year;
  mg CH4 m-2 d-1; and umol or nmol CH4 m-2 s-1 now convert to gC m-2 day-1.
  Before, they kept their declared unit. A bare nmol m-2 s-1 is read as a
  carbon flux, like umol m-2 s-1.
- Molar carbon fluxes (mol m-2 s-1 and the umol m-2 s-1 spellings) now use
  12.011 g mol-1 for carbon, as the CO2 and CH4 entries already did. Results
  in umol units, such as tower GPP, rise by 0.09 %.
- Streamflow station matching now drops gauges whose reported upstream area
  is too small for the river to be resolved at the simulation resolution:
  3000 km² at 15min, 500 km² at 06min, 350 km² at 05min, 150 km² at 03min and
  100 km² at 01min. The minimum can no longer be set in the catalog;
  `station_matching.min_uparea` is ignored with a warning. A gauge without a
  reported area passes this check but must still pass the others. Station
  matching, including the `direct` method, now fails with a clear error for
  other simulation resolutions.
- CaMA station matching (`cama_allocation`) now keeps only gauges whose
  allocation error `cama_alloc_err_<res>` is known and at most 0.2 in absolute
  value, compared at the precision it is stored in. Gauges with a missing
  (NaN) error used to pass and are now dropped, and an error of exactly 0.2
  stored as float32 is no longer dropped. The limit can no longer be set in
  the catalog; `station_matching.area_error_threshold` is ignored with a
  warning.

### Fixed
- Station evaluation no longer warns "time coordinates required
  normalization" for every station. Simulations extracted from a grid are
  stamped at the end of each period and station references in its middle;
  when both hold one value per comparison period they are aligned by period
  quietly. A series with several values in one period is now skipped as a
  data gap with that reason; it used to stop the evaluation with a pandas
  indexing error.
- Station time-series plots draw every station with the same line width and
  marker size. Both used to be divided by the length of the station's record,
  so a 1-year monthly record was drawn 12 points wide and a 10-year daily
  record almost invisibly. `obs_lineswidth`/`sim_lineswidth` and
  `obs_markersize`/`sim_markersize` in `plot_stn.yaml` are now plain sizes in
  points (1 and 4); older totals such as 144 and 432 are read as they were
  meant for a 144-step record. Markers are left out of records longer than
  `marker_max_points` (200) steps, where they hid the line.
- Station time-series plots keep a marker on a value whose neighbours are both
  missing. Records longer than `marker_max_points` are drawn without markers,
  so such a value formed no line segment and vanished; a record with a value
  every other day was not drawn at all. Lines still never cross missing values.
- Rescanning the reference root registers OpenBench_Streamflow_Daily from its
  `_dist.nc` file too, and OpenBench_Streamflow_Monthly with
  `discharge_var: discharge`. The scan profile said `streamflow`, so a
  rescanned monthly reference could not be matched.
- In station time-series plots the RMSE/R/KGESS line now sits on its own row
  above the plot, right-aligned, under the title. It was placed at a fixed 60 %
  of the width on the title's row, so a long station id or coordinate ran into
  it. The title shows the station id as it is (`01010000_USGS`, not
  `01010000_Usgs`).
- Metric maps no longer cut off negative APFB, dr and cp values. Like NSE and
  KGE they are now drawn on -1 to 1, with values beyond shown by the colour
  bar's end arrows. The MFM components keep their colour bar within 0 to 1.
- KGEln, APFB, br2, cp, dr and the three MFM components no longer log
  "Unknown metric unit" for every plot.
- A metric map whose values reach exactly one end of the colour bar and pass
  the other end now gets the arrow for the end that is passed. Before, it got
  no arrow, and with `show_method: interpolate` the areas beyond that end
  were left blank.
- A failed Streamflow station match (missing dataset, unsupported resolution,
  or no gauge passing the limits) now stops that evaluation with the original
  error. It used to be logged and the evaluation continued with the default
  filter, which could keep a previously loaded station list the matcher never
  checked.

## [3.0.6] - 2026-10-05

### Changed
- Reports now also write `reports/<name>_standalone.html`, a single file with
  every figure embedded, so the report keeps its figures when copied or sent.
  Embedded figures are downscaled to 1200 px wide and stored as JPEG when that
  is much smaller than PNG; `evaluation_report.html` still links the
  full-resolution files under `reports/figures/`.
- PDF reports are no longer generated. The `report` extra (xhtml2pdf) is now
  empty and kept only so `pip install colm-openbench[report]` still resolves.
- An evaluation pair now fails with a clear error when its reference and
  simulation units convert to different base units (for example `W m-2`
  against `mm day-1`), instead of computing metrics from incomparable values.
  Units the converter does not recognize are still compared as declared.
- Catalog and config variables accept `accumulated: year` or
  `accumulated: run` for running totals. `year` is for totals that restart on
  1 January, and `run` for totals counted from the start of the simulation.
  After the data are assembled per year, and before resampling or unit
  conversion, the totals are turned into the amount added in each time step.
  A step that cannot be recovered (a year that starts after January, the first
  step of a run, or a restart) is left missing.
- Preprocessing warns once per source when a file's `units` attribute names a
  different unit than the declared `varunit` (for example `mm.month-1` against
  `mm day-1`).

### Fixed
- Carbon stock spellings `kgC m-2`, `kg C m-2`, `gC m-2` and `g C m-2`
  now resolve to one base unit, avoiding false incompatible-unit errors.
- Target diagrams pool equally weighted station/cell errors using centred
  variance and the spread of station/cell biases. This preserves the RMSD
  identity without losing small centred errors when bias dominates.
- Since 3.0.0, every variable declared as `W m-2` or `w m-2` was converted to
  `mm day-1` as if it were evaporation, while `W/m2` and `watt/m2` stayed in
  W m-2. Sensible heat and radiation were therefore reported in mm day-1, and a
  CoLM simulation (`w m-2`) evaluated against the OpenBench_FLUX station
  references (`W/m2`) was about 29 times too small for latent heat, sensible
  heat, net radiation, ground heat and the radiation components. W m-2 now
  stays W m-2 in every spelling; evaporation and transpiration items declared
  in W m-2 are still converted to mm day-1, and figures are labelled with the
  unit the data were converted to (#211). Cached results from earlier versions
  are invalidated.
- Bundled registry units that did not match the data:
  - ERA5LAND precipitation (stored in m hr-1) was declared `mm`, and its
    runoff (daily totals in m) `mm`.
  - ERA5-Land model output (monthly means of daily totals) was declared `m`
    and `J m-2`. It now uses `m day-1` and `J m-2 day-1`. Evaporation, latent
    heat and sensible heat are negated, because ECMWF fluxes are positive
    downward.
  - GLDAS runoff, a 3-hour accumulation, was declared `kg m-2`.
  - WRF evapotranspiration, computed in mm day-1, was declared `mm`.
  - GRAiCE water storage change was declared `cm` instead of
    `cm of equivalent water thickness`.
  - Several spellings were not recognised, so the data passed through
    unconverted. These include `mm/s` (NoahMP5 runoff; CLM5, E3SM and ELM
    irrigation), `degrees Celsius` (CRU temperatures), `mm d-1` (MSWEP),
    `m of water equivalent`, `m3/m3`, `g C m-2 yr-1`, `kPa` and `Mg ha-1`.
  - JULES7 GPP and respiration are carbon fluxes (`kg c m-2 s-1`).
  - ecLand GPP, respiration and NEE are CO2 mass fluxes, positive downward.
    They now use the new `kg co2 m-2 s-1`, and respiration and NEE are
    negated.
  - NoahMP5:
    - `SnowDepth` is in m, not mm.
    - `EvapSoilSfcLiq` is in m s-1.
    - `NetEcoExchange` is in g CO2 m-2 s-1.
    - Latent heat used only the vegetated-ground flux and is now the total.
    - Transpiration pointed at ground evaporation.
    - Downward shortwave and longwave radiation were swapped.
  - TE snow water equivalent unit `kg m2-1` is now `kg m-2`.
  - VIC5 fluxes declared `mm step-1` are `mm day-1` for the catalog's daily
    output. The CLM5/ELM/E3SM snow cover fraction `FSNO_EFF` was declared
    `m s-1 wind` and is now `unitless`.
  - Removed entries whose values cannot be converted to what they are compared
    with:
    - CLM5/ELM/E3SM burned area (fraction per second).
    - NoahMP5 ecosystem respiration (`RespirationSoil` has no documented unit
      and may include root respiration).
    - CRU frost-day frequency and the HOMTS root-zone soil temperature (unit
      `TS`).
    - The HSWUD water use, UpCH4 wetland methane and ESA CCI burned area
      datasets.
  - BCC_AVIM snow depth pointed at the snow water equivalent `H2OSNO` and now
    uses `SNOWDP` in m.
  - VIC5 soil moisture pointed at `OUT_SOIL_MOIST_FRAC`, which VIC 5 does not
    write. Surface soil moisture now uses layer 0 of
    `OUT_SOIL_LIQ_FRAC + OUT_SOIL_ICE_FRAC`; root-zone soil moisture is left
    unmapped.
  - CoLM `f_sum_irrig` is a year-to-date total (`accumulated: year`, monthly
    amounts in `mm month-1`).
  - WRF precipitation is now `RAINNC + RAINC`. Both accumulate from the start
    of the run (`accumulated: run`, `mm hr-1`).
  - JULES7 `rflow` is a flow per unit grid-box area and is multiplied by the
    cell area to give m3 s-1.
  - CLM5/ELM/E3SM rice yield pointed at `GRAINC_TO_FOOD`, the grain carbon
    flux of all crops in a grid cell, and is no longer mapped.
  - TE water storage change, a volume difference of `STORGE` in m3, is no
    longer mapped.
- A `compute` expression written inline in a config's simulation variables
  was ignored. Only catalog expressions took effect, because the processor
  read `sim_compute`/`ref_compute` while the config passes the expression
  keyed by source name. Expressions from either place are now applied.
- `openbench smoke-test --run` reported Evapotranspiration against GLEAM 4.2a
  about 30 times too high: the bundled fixture stores monthly totals
  (`mm.month-1`) but the smoke catalog declared `mm day-1`, so no conversion
  was applied. The full GLEAM 4.2a reference set is stored in `mm day-1` and
  was not affected.
- Calendar-aware unit conversions (`mm month-1` → `mm day-1`,
  `mm day-1` → `mm year-1`) dropped the variable name of a DataArray, so the
  preprocessed file stored it as `__xarray_dataarray_variable__` and station
  evaluations against such a grid reference failed with
  `Variable '<name>' not found in ref dataset`. Bundled references declared in
  `mm month-1` include GGMSEUD and GIWUED (`Total_Irrigation_Amount`).

## [3.0.5] - 2026-09-30

Compatibility and fix release for xarray 2026.9 and station comparisons.

### Changed
- Map ticks and gridlines follow the plotted extent (global, regional or a
  user-set `min_lon`/`max_lon`/`min_lat`/`max_lat`) instead of fixed global
  intervals, through one shared helper used by all map figures (#210).
  Single-point or dateline-crossing extents (min >= max) still render,
  without ticks.

### Fixed
- Map figures failed with xarray 2026.9.0 when large grids were downsampled
  for plotting (`DataArrayCoarsen` reductions no longer accept `skipna`).
- The comparison phase converted `compare_tim_res` to invalid pandas aliases
  (`1DE`, `1HE`, `1WE`), so station comparisons could only align differently
  stamped sim/ref series through a growing list of special cases, and
  multi-step resolutions such as `3month` skipped that fallback entirely. It
  now uses valid pandas frequencies (`1D`, `6h`, `3ME`, `1YE`, `1W`), shared by
  normal and drawing-only runs, and station time alignment parses the step and
  unit.
- `conda/meta.yaml` now carries the checksum of the sdist published with the
  GitHub release.

## [3.0.4] - 2026-09-30

Interactive `openbench init` now asks for the evaluation settings it used to
fill in silently, and `openbench run --resume` continues long grid-to-station
runs without redoing station preprocessing.

### Added
- New wizard step "Domain, Resolution & Runtime": latitude/longitude range,
  target `tim_res` and `grid_res` (defaults inferred from the selected
  references and simulations), `time_alignment`, IGBP/PFT/climate-zone
  group-by, and `num_cores` (`auto` = all CPU cores). Answers are written to
  `project` as active values. When simulations disagree on `tim_res` or
  `grid_res`, a target is required (`none` would fail `openbench check`), and
  an unsupported inferred `tim_res` (e.g. `2-Day`) is not offered as the
  default.
- New wizard step "Metrics, Scores & Analyses": pick metrics, scores,
  comparison figures and statistics methods by number or name from numbered
  lists, with defaults marked `*`.
- `openbench init` asks to confirm scanned simulation cases before station
  lists are materialized; answering no returns to the simulation roots prompt.
- `openbench run --resume` reuses a task's station preprocessing when its
  completion marker matches the current inputs and configuration and every
  expected station artifact exists (a non-empty `.nc` or a `.skip.txt`
  data-gap marker). Metrics and scores are always recomputed; tasks that do
  not qualify are preprocessed normally, and the run manifest records which
  tasks were resumed. `--resume` cannot be combined with `--force`,
  `--comparison-only` or `project.only_drawing`.
- CaMa `Total_Runoff` is converted from a per-cell volume flux (`m3 s-1`) to
  `mm day-1` using the grid-cell area of the input grid. Catalog `compute`
  expressions may now use `np.sin` and `.diff()`.

### Changed
- Every init prompt explains the setting and shows its default; Enter keeps
  the default. Yes/no prompts read `[yes/no/back, Enter = …]`.
- `b` is accepted as `back` in all init prompts and in yes/no and numbered
  prompts of other wizards. Free-text data-name prompts (NetCDF variable
  names, units, globs) still require the full word `back`.
- Going back skips comparison/statistics item prompts for disabled phases.
- `--refresh-ref` help now states that init never rescans without it, and init
  prints a hint when the reference catalog was not rescanned.
- Faster grid preprocessing, conservative regridding and station-grid
  extraction. Station matching reads only candidate time windows, and
  `openbench check` scans simulation time coverage in parallel using
  `num_cores`.
- Preprocessing outputs (yearly scratch files and flat sim/ref NetCDF) follow
  `OPENBENCH_NETCDF_COMPRESSION` instead of always being written uncompressed.
- Stations without overlapping finite sim/ref values are no longer filtered
  when the station list is loaded; evaluation still skips them. The old filter
  was bypassed whenever any station had a `.skip.txt` marker and reopened
  every station file.

### Fixed
- Keep station IDs as text when reading station lists and station metadata,
  so IDs with leading zeros (e.g. `0000000009463`) are no longer turned into
  integers and lost. ID-like columns are recognised in any case (`ID`, `Id`,
  `site_id`, ...).
- Match station IDs regardless of zero padding (`0000000009463` == `9463`)
  when merging simulation and reference station lists, selecting stations
  from merged NetCDF files, and looking up sidecar metadata. HydroWeb also
  finds station files named by the unpadded ID.
- Grid evaluation found no overlapping timestamps when a monthly reference kept
  mid-month labels while a daily simulation was resampled to month-end. Data
  already at the target frequency is now resampled to the shared time labels.
- Station comparisons (Taylor and Target diagrams, SMPI, seasonal portraits,
  scenario correlation, ...) at daily, hourly or yearly `compare_tim_res`
  reported stations as having no overlapping time steps when sim and ref used
  different timestamp conventions (e.g. 00 UTC vs 12 UTC daily values). The
  comparison phase stores the resolution as `1DE`/`1HE`/`1YE`, which the
  timestamp normalization fallback did not recognise; monthly (`1ME`) was
  already handled in 3.0.3.
- Station IDs duplicated across sources get a filename-safe qualifier (the
  previous `::` separator is invalid in Windows filenames). Only duplicated
  IDs are qualified.
- Seasonal portrait plots: annual-cycle scores need all 12 months, so they are
  reported as N/A for three-month seasons with a warning, and the remaining
  outputs still render. Constant metric ranges no longer break the colorbar.
- Parallel-coordinate plots apply the 5–95% outlier trim only when a metric
  has more than two finite values.
- SWAMPS_v3.2 catalog entries: corrected root directories, unit `percent`,
  daily LowRes data and 1992–2020 MidRes years. A user catalog under
  `~/.openbench/references/` overrides the bundled one, so update the entry
  there too.

## [3.0.3] - 2026-09-18

Feature and reliability release for station-mode preprocessing and comparisons.

### Added
- Support a derived-variable fallback for station datasets when the configured
  variable is absent, and normalize `1ME`/`1MS` monthly station timestamps
  before alignment (#198).

### Fixed
- Run catalog compute (sign flips, unit conversions, PFT aggregation) on
  station-mode model output before falling back to the raw variable, matching
  the order grid preprocessing already uses.
- Read relabelled flat files by their catalog item name in the SMPI,
  Mann-Kendall reference, mass-weighted score and groupby readers, instead of
  the raw configured varname.
- Harden KDE and ridgeline plots against sparse and constant data; update
  streamflow dataset configuration and plot defaults.
- Only clip Parallel Coordinates metric quantiles when enough finite values
  are present, rather than counting NaNs toward the sample size (#199).

### Packaging
- Published GitHub release v3.0.3, PyPI 3.0.3 (uploaded manually with
  `twine`, since the `publish.yml` trusted-publishing workflow is not yet
  registered on PyPI for this repository) and updated the pending
  conda-forge recipe in
  [conda-forge/staged-recipes#34807](https://github.com/conda-forge/staged-recipes/pull/34807)
  to 3.0.3, dropping the Windows console-encoding patch the source no
  longer needs. conda-forge review/merge is still pending.

## [3.0.2] - 2026-09-14

Patch release for station evaluation and comparison reliability.

### Fixed
- Retain station comparison rows with unavailable values and explicit reasons,
  without excluding valid grid pairs in mixed grid/station configurations.
- Distinguish known station data gaps from processing failures; report partial
  evaluation success and do not cache it as a complete result.
- Align station timestamps before using configured-resolution normalization,
  preserving exact non-Gregorian calendar coordinates and singleton time axes.
- Include station results in Correlation, Basic, seasonal and tail-statistic
  comparisons and drawing-only runs; preserve undefined statistics as NA.
- Restrict Relative Score to configured compatible sources, retain undefined
  station results, and use signed z-score color scales rather than score bounds.
- Preserve per-statistic valid sample counts in group-by tables and figures.
- Resolve Diff Plot color-normalization conflicts and retain Matplotlib 3.4+
  Whisker Plot compatibility; honor station plot limits and neutral NA markers.
- Fall back directly from missing PLUMBER2 `Qle_cor` / `Qh_cor` to `Qle` / `Qh`,
  and accept the supported carbon-flux and temperature unit aliases.

### Packaging
- Advance the GitHub source and Conda recipe to 3.0.2. The source already includes
  the Windows console fix, so its old Conda backport patch is no longer needed.
- This GitHub update does not publish a package to PyPI or conda-forge.

## [3.0.1] - 2026-09-11

Patch release over 3.0.0 for release metadata and CLI output compatibility.

### Fixed
- Prevent Unicode output crashes in redirected CP1252 and other legacy-encoded
  CLI streams while preserving UTF-8 output on normal terminals.
- Backport the Windows-default `surrogateescape` correction into Conda builds
  of the immutable GitHub 3.0.1 source; the PyPI 3.0.1 release includes it directly.
- Correct distribution metadata and bundled third-party license coverage for
  vendored `cmaps` and NCL color-table resources.

### Packaging
- Keep Conda-forge availability documented as pending review in
  [conda-forge/staged-recipes#34807](https://github.com/conda-forge/staged-recipes/pull/34807);
  no conda-forge channel package has been published yet.

## [3.0.0] - 2026-09-11

First stable 3.0 release, including the main-branch fixes validated across Linux,
macOS and Windows on Python 3.10–3.12. The opt-in uncertainty-aware pipeline
remains on its separate development branch.

### Fixed
- Shared grid preprocessing no longer depends on which reference is evaluated
  first; per-pair masked references survive cache reuse and post-processing errors.
- Statistics respect preprocessed units, finite observations and coordinate
  alignment; ANOVA, PLSR and Three-Cornered Hat receive the correct source layout.
- Non-Gregorian monthly conversions, computed reference variables and mixed
  flat/resolution-specific reference directories retain their intended data.
- Remote configuration changes apply atomically without retaining another
  host's passwords; recursive deletes clear matching cached and pending files.

### Maintenance
- Remove obsolete processing scaffolding and consolidate calendar and saved
  credential handling without removing public compatibility entry points.
- Share wheel/sdist resource checks between CI and publishing, require explicitly
  selected artifacts to exist, and document fresh-build release verification.
- Remove redundant comments and duplicate test scaffolding while retaining
  meaningful scientific documentation and deterministic regression coverage.
- Align the Conda recipe with the stable PyPI version and verify its dependency,
  command-line and bundled-data installation contracts.

## [3.0.0b16] - 2026-08-31

Beta release over 3.0.0b15 for remote workflow reliability, expanded evaluation
methods, and complete output packaging.

### Added
- Evaluation-guide appendix metrics and methods, including deterministic,
  categorical, uncertainty, trend, and distribution diagnostics.
- Live remote CPU and memory monitoring tied to the active OpenBench process.

### Changed
- Index of Agreement is available through the metric workflow.
- Generated figures are published under the advertised top-level `figures/`
  directory while existing metric, comparison, and report paths remain valid.

### Fixed
- Remote installation, Conda environment creation, SSH host selection, output
  paths, project confirmation, and local/remote dataset discovery.
- Remote progress now waits for the real evaluation process across jump hosts
  instead of reporting premature success or completion.
- Legacy flat reference layouts, CLM multi-stream file prefixes, and simulation
  data scanning now preserve the files selected by the user.
- Appendix NetCDF execution, GUI/Qt CI stability, and repository formatting.

## [3.0.0b15] - 2026-08-26

Beta release over 3.0.0b14 for GUI/runtime reliability, truthful reporting,
station evaluation fixes, and preprocessing performance.

### Added
- Content-addressed xESMF weight caching to reuse identical regridding weights.
- GUI progress events and bounded report summaries for safer long-running local
  and remote evaluations.

### Changed
- Local Dask evaluation avoids distributed NetCDF/HDF5 worker writes and keeps
  scanned simulation case cards readable.
- Shared preprocessing, reference masking, MFM component work, and multi-model
  comparisons reuse expensive intermediate work where results are equivalent.

### Fixed
- GUI local and remote workflows now fail truthfully instead of silently falling
  back, stalling, or running on the wrong target.
- Reference and simulation scanning preserve requested datasets, station lists,
  paths, and inline metadata more reliably.
- Time alignment, finite-observation masks, singleton conservative regrids, and
  station missing-value handling no longer silently alter scientific results.
- Large NetCDF/CSV reports remain bounded and avoid fabricated or incomplete
  summaries.
- Windows path normalization, environment discovery, CPU quota handling, and CI
  formatting stability.
- Station evaluation process updates from PRs #167 and #170.

## [3.0.0b14] - 2026-08-22

Beta release over 3.0.0b13 for actionable GUI preflight diagnostics.

### Fixed
- GUI configuration-check failures now include the complete CLI check output,
  so validation details are not hidden by the final summary lines.

## [3.0.0b13] - 2026-08-22

Beta release over 3.0.0b12 for safer configuration, registration, and GUI
workflows.

### Added
- Chinese/English GUI switching with immediate page retranslation.
- Back navigation throughout interactive CLI setup and registration wizards.

### Changed
- GUI evaluations now run the same CLI preflight check before execution.
- Reference rescans distinguish new and registered datasets, default to new
  entries, and enrich only explicitly selected registered datasets.
- CLI registration accepts quoted whitespace-containing units in named
  `key=value` variable specifications and provides actionable shell guidance.

### Fixed
- CLI and GUI preflight reject evaluations whose configured years do not
  overlap the available data period.
- Reference rescans preserve manual catalog corrections and keep remote refresh
  work responsive and narrowly targeted.
- Multi-step CLI back navigation retains previously entered values across
  variables and empty intermediate steps.
- Cross-platform CI behavior for paths, generated artifacts, and GUI tests.

## [3.0.0b12] - 2026-08-18

Beta release over 3.0.0b11 for reliable generated station configuration.

### Fixed
- `openbench init` and `openbench sim scan` now write absolute station
  `fulllist` paths so generated YAML works from any current directory.

## [3.0.0b11] - 2026-08-13

Beta release over 3.0.0b10 for GUI dataset discovery, registry coverage, and
statistical evaluation correctness.

### Added
- Expanded Crop reference catalog coverage and CoLM2024 model metadata.
- Run-manifest and configuration-key validation for more reproducible runs.

### Changed
- GUI simulation scans retain per-case grid/station metadata and prepare station
  lists before local preview/export.
- SMPI remains available through its dedicated comparison workflow instead of
  the single-value metric selector.

### Fixed
- GUI navigation now persists edits, registered reference paths are no longer
  overwritten by scan roots, and simulation discovery follows the selected run.
- Reference station-list resolution, Streamflow catalog entries, units, and
  registry behavior were corrected.
- Mann-Kendall handles tied short series, categorical kappa rejects continuous
  inputs, and SMPI bootstrap intervals are consistent for NumPy and Dask data.

## [3.0.0b10] - 2026-07-18

Beta release over 3.0.0b9 for station-reference routing correctness.

### Changed
- Renamed the active configuration runtime module from `legacy_processors.py`
  to `runtime_info.py` to reflect that it remains on the main execution path.

### Fixed
- Non-Streamflow station references no longer enter the discharge/CaMA-specific
  `station_matching` engine.
- Reference scans remove stale non-Streamflow matching blocks and generate
  portable station-list catalogs instead.

## [3.0.0b9] - 2026-07-16

Beta release over 3.0.0b8 for registry coverage and CI stability.

### Changed
- Station fulllists generated by scans are stored under `OPENBENCH_HOME`,
  allowing shared reference roots to remain read-only.
- Expanded bundled model and reference metadata.

### Fixed
- Compute validation now accepts keyword arguments such as `isel(lake=0)`.
- Restored required station-matching metadata and removed incomplete generated
  station catalog entries.
- Simulation scanning retains the TE `Albedo` compatibility key.
- NetCDF evaluation writes no longer retain lazy references to closed source
  datasets.
- Ruff formatting for the FFT component metric.

## [3.0.0b8] - 2026-07-08

Patch release over 3.0.0b7 for model coverage, audit fixes, and CI stability.

### Added
- Bundled ECLand and LM4 model profiles and aliases.

### Fixed
- Station simulation scanning now loads merged NetCDF sources before writing the
  target file, avoiding platform-sensitive HDF5/netCDF CI hangs.
- Hardened audit fixes for compute sandboxing, remote uploads, registry locking,
  station lists, time decoding, station matching, and unit conversions.

## [3.0.0b7] - 2026-06-11

Patch release over 3.0.0b6 for GUI selection behavior.

### Fixed
- Multi-resolution reference selection now allows lower-frequency variants such
  as LowRes/Month instead of forcing the highest-frequency available variant.
- Simulation dataset scanning no longer auto-selects every model-profile
  variable or expands the user's Evaluation Variables selection in the preview.

## [3.0.0b6] - 2026-06-11

Patch release over 3.0.0b5 for GUI wizard flow alignment.

### Changed
- GUI setup now follows the CLI/config order: choose Evaluation Variables first,
  then configure reference datasets and simulation datasets for those variables.

## [3.0.0b5] - 2026-06-11

Patch release over 3.0.0b4 for GUI reference selection and remote SSH hardening.

### Fixed
- Multi-resolution reference selection no longer crashes when the dialog is
  populated from registry `ReferenceDataset` entries, which do not carry the
  scan-only `file_count` field.
- SSH output streaming falls back to short polling when `select.select()`
  rejects Paramiko-like channels on platforms such as Windows.

## [3.0.0b4] - 2026-06-11

Remote-workflow hardening release: the GUI's remote mode was reworked end to
end (responsive SSH layer + three review rounds over it).

### Fixed
- Conda activation on remote hosts uses the POSIX `.` command — the `source`
  bashism broke every conda-env remote flow on hosts whose `/bin/sh` is
  dash/ash (commands are wrapped in `sh -c` for csh/tcsh login shells)
- Remote paths starting with `~` (including the documented `~/OpenBench`
  default) now expand to `$HOME` everywhere a path reaches the remote shell:
  the evaluation run command, pip dependency step, conda create, the
  OpenBench install check, the sync engine, scanners, validators, NML/model
  editors and file browsers. Bare `shlex.quote` turned them into literal-`~`
  paths that no remote shell resolves
- Remote dataset scanning trusts the inspection results shipped from the
  remote host instead of coincidentally same-named local directories
  (wrong-machine inspection on shared mounts), and station fulllists are
  generated remotely for `stn` references
- A dropped SSH session is reported as "Connection Lost" instead of the
  misleading "Remote directory not found"
- The remote scan bootstrap `expanduser()`s the checkout path so a plain-git
  `~/OpenBench` install is importable even without the pip step

### Changed
- Every SSH call made from the GUI thread now routes through a responsive
  worker layer: the window keeps painting during connects, scans and
  installs; long installs stream output into Esc-guarded progress dialogs;
  scans and installs are cancellable mid-command
- Remote installs run `pip install -e` for dependencies (the old
  requirements.yml flow targeted a file that no longer exists)
- Browse start-path resolution and symlink-target classification each probe
  the remote host in a single compound round trip (was up to N serial
  round trips per click)
- The Runtime page's local git install/update streams through a worker
  thread with a stoppable subprocess (no more event-loop pumping)

## [3.0.0b3] - 2026-06-09

### Changed
- `pip install colm-openbench` is now fully featured by default: statistics
  (scikit-learn, statsmodels), plotting (seaborn), legacy migration (f90nml)
  and HTML reports ship in the base install. Only `[gui]` (PySide6), `[remote]`
  (paramiko) and `[report]` (xhtml2pdf — needs system cairo) remain optional;
  `[all]` = gui + remote + report.

### Added
- Reference datasets: `Rodai2025_NPP` (station NPP) and lake stations
  `G_REALM_LakeLevel`, `GLAST_LakeSurfaceWaterTemperature`, `ReaLSAT_LakeArea`
- 30-minute (half-hour) time-resolution detection

### Docs
- User's Guide (CN+EN): climatology added to the `tim_res` field table; a
  multi-model `simulation` example; `timezone` flagged as not-yet-implemented;
  default `io`/`dask` sub-blocks shown; detailed cases for unresolvable
  reference profiles and `sim scan` model inference

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
