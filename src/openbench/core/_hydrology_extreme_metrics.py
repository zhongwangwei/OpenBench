"""Hydrological extremes and inundation overlap statistics for OpenBench.

All calculations use paired finite observations and simulations. A spatial CSI
is distinct from a pixel's temporal wet/dry CSI; both are provided explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks


def quantile_flow_bias(sim: xr.DataArray, obs: xr.DataArray, quantile: float) -> xr.DataArray:
    """Percent bias between the two paired-sample *high* flow quantiles."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    sim, obs = xr.align(sim, obs, join="inner")
    paired = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim.where(paired), obs.where(paired)
    if sim.chunks is not None:
        sim = sim.chunk({"time": -1})
    if obs.chunks is not None:
        obs = obs.chunk({"time": -1})
    qs = sim.quantile(quantile, dim="time", skipna=True)
    qo = obs.quantile(quantile, dim="time", skipna=True)
    result = xr.where((paired.sum("time") >= 2) & (qo > 0), 100.0 * (qs - qo) / qo, np.nan)
    return result.drop_vars("quantile", errors="ignore")


def _peak_stats_1d(sim, obs, days, threshold_quantile, separation_days, window_days):
    """Return matched-event mean absolute timing error (days) and match fraction.

    Independent observed events are local maxima above observed Q95, separated
    by >= 15 days by default. Each is matched once to the largest local model
    maximum in a +/- 7 day window, never to a peak already used elsewhere.
    """
    finite = np.isfinite(sim) & np.isfinite(obs) & np.isfinite(days)
    if finite.sum() < 30:
        return np.nan, np.nan
    sim, obs, days = sim[finite], obs[finite], days[finite]
    intervals = np.diff(days)
    if np.any(intervals <= 0) or np.median(intervals) > 1.5:
        return np.nan, np.nan  # Designed for daily or finer time series

    threshold = float(np.quantile(obs, threshold_quantile))
    if not np.isfinite(threshold) or np.max(obs) <= 0:
        return np.nan, np.nan

    def complete_neighbors(indices):
        return np.asarray(
            [
                i
                for i in indices
                if i > 0 and i < len(days) - 1 and days[i] - days[i - 1] <= 1.5 and days[i + 1] - days[i] <= 1.5
            ],
            dtype=int,
        )

    observed_peaks = complete_neighbors(find_peaks(obs)[0])
    observed_peaks = observed_peaks[(obs[observed_peaks] > 0) & (obs[observed_peaks] >= threshold)]
    if not observed_peaks.size:
        return np.nan, np.nan

    # Retain strongest peaks first, so near-duplicate observed flood crests do
    # not appear as independent events. Separation is in days, not row indices.
    independent = []
    for idx in observed_peaks[np.argsort(-obs[observed_peaks])]:
        if all(abs(days[idx] - days[other]) >= separation_days for other in independent):
            independent.append(int(idx))

    model_peaks = complete_neighbors(find_peaks(sim)[0])
    used = set()
    errors = []
    for idx in independent:
        nearby = [int(j) for j in model_peaks if int(j) not in used and abs(days[j] - days[idx]) <= window_days]
        if not nearby:
            continue
        # Largest local crest best represents the flood event; tie-break by
        # timing distance rather than relying on data ordering.
        chosen = min(nearby, key=lambda j: (-sim[j], abs(days[j] - days[idx])))
        used.add(chosen)
        errors.append(abs(float(days[chosen] - days[idx])))

    return (float(np.mean(errors)) if errors else np.nan, float(len(errors) / len(independent)))


def peak_timing_stats(
    sim: xr.DataArray,
    obs: xr.DataArray,
    threshold_quantile: float = 0.95,
    separation_days: float = 15.0,
    window_days: float = 7.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Absolute flood-peak timing error and matched-event fraction.

    Missing matches do not contribute to timing error; consult match fraction
    alongside it. Requires a Gregorian-compatible daily/subdaily time axis.
    """
    if not 0 < threshold_quantile < 1 or separation_days <= 0 or window_days <= 0:
        raise ValueError("Invalid peak detection quantile or time windows")
    sim, obs = xr.align(sim, obs, join="inner")
    if "time" not in sim.coords:
        raise ValueError("PeakTimingError requires a time coordinate")
    try:
        dates = pd.to_datetime(sim.time.values)
        # Timedelta division respects the datetime unit (s/ms/us/ns), unlike
        # interpreting asi8 as nanoseconds on every pandas version.
        day_values = np.asarray((dates - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("PeakTimingError requires Gregorian-compatible datetimes") from exc

    time_days = xr.DataArray(day_values, coords={"time": sim.time}, dims=["time"])
    valid = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim.where(valid), obs.where(valid)
    if sim.chunks is not None:
        sim = sim.chunk({"time": -1})
    if obs.chunks is not None:
        obs = obs.chunk({"time": -1})
    return xr.apply_ufunc(
        _peak_stats_1d,
        sim,
        obs,
        time_days,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[[], []],
        kwargs={
            "threshold_quantile": threshold_quantile,
            "separation_days": separation_days,
            "window_days": window_days,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )


def _valid_fraction_pairs(sim: xr.DataArray, obs: xr.DataArray):
    sim, obs = xr.align(sim, obs, join="inner")
    finite = np.isfinite(sim) & np.isfinite(obs)
    # Do not treat a 0-100 percentage or an area (m2) as a 0-1 fraction.
    domain = (sim >= -1e-6) & (sim <= 1 + 1e-6) & (obs >= -1e-6) & (obs <= 1 + 1e-6)
    return sim.clip(0, 1), obs.clip(0, 1), finite, domain


def inundation_temporal_csi(sim: xr.DataArray, obs: xr.DataArray, threshold: float = 0.05):
    """Per-pixel temporal critical success index, averaged over paired times.

    Pixels where neither series is ever wet have an undefined CSI (NaN), not 1.
    Reject entire pixel series with out-of-domain paired fraction values.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("Inundation CSI threshold must be strictly between 0 and 1")
    sim, obs, finite, domain = _valid_fraction_pairs(sim, obs)
    valid = finite & domain
    sim_wet = sim > threshold
    obs_wet = obs > threshold
    hits = (sim_wet & obs_wet & valid).sum("time")
    unions = ((sim_wet | obs_wet) & valid).sum("time")
    bad_pixel = (finite & ~domain).any("time")
    return xr.where((unions > 0) & ~bad_pixel, hits / unions, np.nan)


def inundation_spatial_csi(sim: xr.DataArray, obs: xr.DataArray, threshold: float = 0.05):
    """Area-weighted spatial inundation CSI for each matched timestamp.

    The weights cos(latitude) approximate cell area for a regular lat/lon
    grid. Caller must align model/reference to the same grid and valid domain.
    Returns a Dataset with CSI, hits, false alarms, misses, and valid weights.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("Inundation CSI threshold must be strictly between 0 and 1")
    sim, obs, finite, domain = _valid_fraction_pairs(sim, obs)
    if not {"time", "lat", "lon"}.issubset(sim.dims):
        raise ValueError("Spatial inundation CSI requires time, lat, lon dimensions")
    if sim.lat.ndim != 1 or sim.lon.ndim != 1:
        raise ValueError("Spatial inundation CSI requires a rectilinear lat/lon grid")
    weights = xr.DataArray(
        np.maximum(np.cos(np.deg2rad(sim.lat.values)), 0.0),
        dims=["lat"],
        coords={"lat": sim.lat},
    )
    valid = finite & domain
    s_wet, o_wet = sim > threshold, obs > threshold
    spatial_dims = ("lat", "lon")

    def area_of(mask):
        return xr.where(mask, weights, 0.0).sum(dim=spatial_dims)

    hits = area_of(valid & s_wet & o_wet)
    false_alarms = area_of(valid & s_wet & ~o_wet)
    misses = area_of(valid & ~s_wet & o_wet)
    valid_weight = area_of(valid)
    union = hits + false_alarms + misses
    return xr.Dataset(
        {
            "CSI_spatial": xr.where(union > 0, hits / union, np.nan),
            "hit_weight": hits,
            "false_alarm_weight": false_alarms,
            "miss_weight": misses,
            "valid_weight": valid_weight,
        }
    )
