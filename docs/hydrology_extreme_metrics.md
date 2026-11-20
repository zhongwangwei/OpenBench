# Hydrological extreme-flow metrics and inundation CSI

These metrics evaluate high-flow quantiles, flood-peak timing, and inundation
agreement. They appear in the Metrics GUI and configuration under
**Hydrology** or **Inundation** and do not require a particular simulation model.

## Flow quantiles

- **Q95_Bias**, **Q99_Bias** (%) compare the 0.95 and 0.99 *non-exceedance*
  percentiles of modeled vs. observed discharge:
  100 * (quantile(sim, p) - quantile(obs, p)) / quantile(obs, p).
- They are **high-flow quantiles** and must not be confused with the Q95
  exceedance flow-duration notation sometimes used for low flow.
- Only dates with finite values on *both sides* are used. If the observed
  quantile is non-positive, or fewer than two pairs remain, the value is NaN.
- These are different from PBIAS_HF, which sums biases **on observed high-flow
  dates** rather than comparing two marginal flow percentiles.
- Difference maps for Q95/Q99 Bias use percentage points (`pp`). Their default
  color limits use the robust data range, with colorbar extensions marking
  outliers. Explicit user-specified limits remain unchanged.

## Event-based flood peak timing

- **PeakTimingError** (days, ideal 0): mean absolute difference between
  matched observed/simulated local flood-peak dates.
- **PeakTimingMatchRate** (fraction 0-1): fraction of independently detected
  observed flood peaks matched to a model peak. Always examine alongside
  PeakTimingError: unmatched events are not included in the timing mean.
- Defaults: observed Q95 threshold; at least 15 days between observed
  independent events; match the largest available simulated local maximum
  within +/- 7 days, one-to-one, prioritizing the strongest observed floods.
  The method requires >= 30 common finite daily/subdaily samples. All matching
  occurs after OpenBench's ordinary time alignment.
- Monthly and annual flow series are *not* suitable for this event metric and
  yield NaN; use KGE, flow quantiles, or an appropriate seasonal metric there.
- Defaults are adjustable through the programmatic peak_timing_stats helper;
  the current GUI uses the documented defaults.

## Inundation Fraction: distinguish two CSI meanings

- **Inundation_CSI** is the **per-cell temporal CSI**. On each grid cell,
  across valid paired timestamps, it is wet/wet count divided by wet-on-either
  count. This yields the normal (lat, lon) NetCDF evaluation map.
- When Inundation_CSI is evaluated on a gridded **Inundation_Fraction** item,
  OpenBench **also writes**:

  - `metrics/<item>_ref_<ref>_sim_<sim>_Inundation_CSI<vkey>_spatial_timeseries.csv`
  - `metrics/<item>_ref_<ref>_sim_<sim>_Inundation_CSI<vkey>_spatial_summary.csv`

  Here `<vkey>` is the same optional variable suffix used in NetCDF outputs.
- The first CSV has the **per-timestamp spatial CSI**, formed from wet/wet area
  divided by the union of wet areas over the evaluated domain. The summary
  contains time-mean CSI and area-time pooled CSI. They are NOT the same
  quantity as per-cell temporal CSI.
- Default threshold: **fraction > 0.05 is wet**, applied identically to
  modeled and observed inundation fractions. Inputs must be in 0-1 fraction,
  NOT 0-100 percent or flooded area in m2. Paired missing cells are excluded.
  A paired out-of-range fraction invalidates that pixel's temporal CSI;
  spatial CSI excludes out-of-range pairs at each timestamp.
  All-dry pixels/months have undefined (NaN) CSI, not an artificial perfect 1.
- Spatial CSI uses cos(latitude) cell-area weights, appropriate to a
  regularly spaced rectilinear lat/lon grid. It is not appropriate for a
  projected/equal-area source grid before that source is correctly regridded.
- Match GIEMS vs. model spatial footprints, land/water definitions,
  resolution and monthly time labeling before interpreting CSI.

## Comparing simulations

For two simulations, first use identical station IDs and **common valid
timestamps**. Positive delta-KGE indicates better skill; positive
difference in absolute quantile bias or absolute timing error means worse
skill. Compare changes in errors, not signed PBIAS differences alone.

## Compatibility with current OpenBench main

Metrics are selected globally across all evaluation variables.
`Inundation_CSI` is **automatically skipped** for variables other than
`Inundation_Fraction`, avoiding errors for mixed streamflow/inundation
evaluations. Q95/Q99 Bias and flood peak timing statistics are only computed
for Streamflow; they are skipped for all other variables. The preflight CLI
check rejects a CSI selection if no inundation-fraction variable is requested,
or flow metrics if no Streamflow is requested. It also rejects event-peak
timing metrics when the evaluation resolution is monthly or annual.

Flood timing accepts daily or finer comparison resolutions, including the
existing `3Hour` and `6Hour` settings. It requires Gregorian-compatible dates;
the timing calculation is independent of datetime storage precision.
