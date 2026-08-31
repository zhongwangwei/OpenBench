"""Appendix methods whose inputs or outputs do not fit pairwise scalar metrics."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats


def uncertainty_factors(lower, upper, observation, dim="time"):
    """Return uncertainty-band width (R-factor) and coverage (p-factor)."""
    lower, upper, observation = xr.align(lower, upper, observation, join="inner")
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(observation)
    valid &= upper >= lower
    lower = lower.where(valid)
    upper = upper.where(valid)
    observation = observation.where(valid)
    obs_std = observation.std(dim=dim)
    r_factor = xr.where(obs_std > 0, (upper - lower).mean(dim=dim) / obs_std, np.nan)
    p_factor = ((observation >= lower) & (observation <= upper)).where(valid).mean(dim=dim)
    return xr.Dataset({"R_factor": r_factor, "p_factor": p_factor})


def ideal_point_error(rmse, r_squared, pbias, dim="candidate"):
    """Calculate IPE across a fixed candidate-model dimension."""
    rmse, r_squared, pbias = xr.align(rmse, r_squared, pbias, join="inner")
    rmse_max = rmse.max(dim=dim)
    r2_span = 1.0 - r_squared.min(dim=dim)
    pbias_max = np.abs(pbias).max(dim=dim)

    total = xr.zeros_like(rmse, dtype=float)
    count = xr.zeros_like(rmse, dtype=float)
    for value, denominator in (
        (rmse, rmse_max),
        (1.0 - r_squared, r2_span),
        (np.abs(pbias), pbias_max),
    ):
        usable = np.isfinite(denominator) & (denominator > 0)
        total = total + xr.where(usable, (value / denominator) ** 2, 0.0)
        count = count + xr.where(usable, 1.0, 0.0)
    return xr.where(count > 0, 1.0 - np.sqrt(total / count), np.nan).rename("IPE")


def contingency_scores(forecast, outcome, dim="time"):
    """Return CSI, HSS, POD, and FAR for already-binarized events."""
    forecast, outcome = _binary_inputs(forecast, outcome)
    domain = _binary_domain(forecast, outcome, dim)
    hit = ((forecast == 1) & (outcome == 1)).sum(dim=dim)
    false_alarm = ((forecast == 1) & (outcome == 0)).sum(dim=dim)
    miss = ((forecast == 0) & (outcome == 1)).sum(dim=dim)
    correct_negative = ((forecast == 0) & (outcome == 0)).sum(dim=dim)
    csi_denominator = hit + false_alarm + miss
    pod_denominator = hit + miss
    far_denominator = hit + false_alarm
    hss_denominator = (hit + miss) * (miss + correct_negative) + (hit + false_alarm) * (false_alarm + correct_negative)
    return xr.Dataset(
        {
            "CSI": hit / csi_denominator.where(csi_denominator > 0),
            "HSS": 2.0 * (hit * correct_negative - false_alarm * miss) / hss_denominator.where(hss_denominator > 0),
            "POD": hit / pod_denominator.where(pod_denominator > 0),
            "FAR": false_alarm / far_denominator.where(far_denominator > 0),
        }
    ).where(domain)


def taylor_skill_score(simulation, observation, reference_correlation, dim="time"):
    """Return the appendix Taylor skill score for an explicit reference correlation."""
    simulation, observation = xr.align(simulation, observation, join="inner")
    valid = np.isfinite(simulation) & np.isfinite(observation)
    simulation = simulation.where(valid)
    observation = observation.where(valid)
    correlation = xr.corr(simulation, observation, dim=dim)
    obs_std = observation.std(dim=dim)
    std_ratio = xr.where(obs_std > 0, simulation.std(dim=dim) / obs_std, np.nan)
    denominator = (1.0 + reference_correlation) ** 4 * (std_ratio + 1.0 / std_ratio) ** 2
    return xr.where(
        (reference_correlation > -1) & (std_ratio > 0),
        4.0 * (1.0 + correlation) ** 4 / denominator,
        np.nan,
    ).rename("TSS")


def brier_score(probability, outcome, dim="time"):
    """Return the Brier score for binary-event probabilities."""
    probability, outcome = _probability_inputs(probability, outcome)
    result = ((probability - outcome) ** 2).mean(dim=dim)
    return result.where(_probability_domain(probability, outcome, dim)).rename("BS")


def brier_decomposition(probability, outcome, dim="time", bins=10):
    """Return Brier score, reliability, resolution, and uncertainty."""
    probability, outcome = _probability_inputs(probability, outcome)
    if bins < 1:
        raise ValueError("bins must be at least 1")
    probability, outcome = _single_chunk(probability, outcome, dim=dim)

    def _components(p, y):
        mask = np.isfinite(p) & np.isfinite(y)
        p = p[mask]
        y = y[mask]
        if p.size == 0:
            return np.full(5, np.nan)
        if ((p < 0) | (p > 1)).any() or not np.isin(y, [0, 1]).all():
            return np.full(5, np.nan)
        group = np.minimum((p * bins).astype(int), bins - 1)
        base_rate = y.mean()
        reliability = 0.0
        resolution = 0.0
        for index in range(bins):
            selected = group == index
            if not selected.any():
                continue
            weight = selected.mean()
            reliability += weight * (p[selected].mean() - y[selected].mean()) ** 2
            resolution += weight * (y[selected].mean() - base_rate) ** 2
        uncertainty = base_rate * (1.0 - base_rate)
        score = np.mean((p - y) ** 2)
        residual = score - (reliability - resolution + uncertainty)
        return np.array([score, reliability, resolution, uncertainty, residual])

    values = xr.apply_ufunc(
        _components,
        probability,
        outcome,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["brier_component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"brier_component": 5}},
    )
    return xr.Dataset(
        {
            "BS": values.isel(brier_component=0, drop=True),
            "reliability": values.isel(brier_component=1, drop=True),
            "resolution": values.isel(brier_component=2, drop=True),
            "uncertainty": values.isel(brier_component=3, drop=True),
            "binning_residual": values.isel(brier_component=4, drop=True),
        }
    )


def crps_ensemble(ensemble, observation, member_dim="member", dim="time"):
    """Return the empirical ensemble CRPS, averaged over *dim*."""
    if member_dim not in ensemble.dims:
        raise ValueError(f"ensemble must contain a {member_dim!r} dimension")
    ensemble, observation = xr.align(ensemble, observation, join="inner", exclude={member_dim})
    observation = observation.where(np.isfinite(observation))
    ensemble = ensemble.where(np.isfinite(ensemble) & np.isfinite(observation))
    first = np.abs(ensemble - observation).mean(dim=member_dim)
    other_dim = f"{member_dim}_other"
    pairwise = np.abs(ensemble - ensemble.rename({member_dim: other_dim})).mean(dim=[member_dim, other_dim])
    result = first - 0.5 * pairwise
    return result.mean(dim=dim).rename("CRPS") if dim is not None else result.rename("CRPS")


def roc_auc(probability, outcome, dim="time"):
    """Return ROC area under the curve for binary outcomes."""
    probability, outcome = _probability_inputs(probability, outcome)
    probability, outcome = _single_chunk(probability, outcome, dim=dim)

    def _auc(p, y):
        mask = np.isfinite(p) & np.isfinite(y)
        p = p[mask]
        y = y[mask]
        if ((p < 0) | (p > 1)).any() or not np.isin(y, [0, 1]).all():
            return np.nan
        y = y.astype(int)
        positive = y == 1
        n_positive = positive.sum()
        n_negative = y.size - n_positive
        if n_positive == 0 or n_negative == 0:
            return np.nan
        ranks = stats.rankdata(p, method="average")
        return (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)

    return xr.apply_ufunc(
        _auc,
        probability,
        outcome,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).rename("AUC")


def roc_curve(probability, outcome):
    """Return an exact ROC curve for one-dimensional probability inputs."""
    probability, outcome = _probability_inputs(probability, outcome)
    if probability.ndim != 1 or outcome.ndim != 1:
        raise ValueError("roc_curve accepts one-dimensional inputs; use roc_auc for gridded data")
    p = np.asarray(probability)
    y = np.asarray(outcome)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must be binary values 0 or 1")
    y = y.astype(int)
    positive = np.count_nonzero(y == 1)
    negative = np.count_nonzero(y == 0)
    if positive == 0 or negative == 0:
        raise ValueError("roc_curve requires both positive and negative outcomes")
    thresholds = np.r_[np.inf, np.unique(p)[::-1]]
    tpr = np.array([np.count_nonzero((p >= threshold) & (y == 1)) / positive for threshold in thresholds])
    fpr = np.array([np.count_nonzero((p >= threshold) & (y == 0)) / negative for threshold in thresholds])
    return xr.Dataset(
        {"TPR": ("threshold", tpr), "FPR": ("threshold", fpr)},
        coords={"threshold": thresholds},
    )


def fit_gev(data, dim="time", min_samples=3):
    """Fit a GEV distribution to block maxima along *dim*."""
    data = _single_chunk(data, dim=dim)[0]

    def _fit(values):
        values = values[np.isfinite(values)]
        if values.size < min_samples or np.ptp(values) == 0:
            return np.full(3, np.nan)
        shape_scipy, location, scale = stats.genextreme.fit(values)
        return np.array([-shape_scipy, location, scale])

    values = _fit_distribution(data, dim, _fit)
    return xr.Dataset(
        {
            "shape": values.isel(distribution_parameter=0, drop=True),
            "location": values.isel(distribution_parameter=1, drop=True),
            "scale": values.isel(distribution_parameter=2, drop=True),
        }
    )


def fit_gpd(data, threshold, dim="time", min_samples=3):
    """Fit a GPD to threshold excesses along *dim*."""
    if not isinstance(threshold, xr.DataArray):
        threshold = xr.DataArray(threshold)
    data, threshold = xr.align(data, threshold, join="inner", exclude={dim})
    data = _single_chunk(data, dim=dim)[0]

    def _fit(values, cutoff):
        values = values[np.isfinite(values)]
        excess = values[values > cutoff] - cutoff
        if excess.size < min_samples or np.ptp(excess) == 0:
            return np.full(2, np.nan)
        shape, _, scale = stats.genpareto.fit(excess, floc=0.0)
        return np.array([shape, scale])

    values = xr.apply_ufunc(
        _fit,
        data,
        threshold,
        input_core_dims=[[dim], []],
        output_core_dims=[["distribution_parameter"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"distribution_parameter": 2}},
    )
    return xr.Dataset(
        {
            "shape": values.isel(distribution_parameter=0, drop=True),
            "scale": values.isel(distribution_parameter=1, drop=True),
            "threshold": threshold,
        }
    )


def _probability_inputs(probability, outcome):
    probability, outcome = xr.align(probability, outcome, join="inner")
    valid = np.isfinite(probability) & np.isfinite(outcome)
    probability = probability.where(valid)
    outcome = outcome.where(valid)
    return probability, outcome


def _binary_inputs(forecast, outcome):
    forecast, outcome = xr.align(forecast, outcome, join="inner")
    valid = np.isfinite(forecast) & np.isfinite(outcome)
    forecast = forecast.where(valid)
    outcome = outcome.where(valid)
    return forecast, outcome


def _probability_domain(probability, outcome, dim):
    valid = probability.notnull() & outcome.notnull()
    probability_ok = probability.isnull() | ((probability >= 0) & (probability <= 1))
    outcome_ok = outcome.isnull() | outcome.isin([0, 1])
    return valid.any(dim=dim) & probability_ok.all(dim=dim) & outcome_ok.all(dim=dim)


def _binary_domain(forecast, outcome, dim):
    valid = forecast.notnull() & outcome.notnull()
    forecast_ok = forecast.isnull() | forecast.isin([0, 1])
    outcome_ok = outcome.isnull() | outcome.isin([0, 1])
    return valid.any(dim=dim) & forecast_ok.all(dim=dim) & outcome_ok.all(dim=dim)


def _single_chunk(*arrays, dim):
    return tuple(
        array.chunk({dim: -1}) if array.chunks is not None and dim in array.dims else array for array in arrays
    )


def _fit_distribution(data, dim, function):
    return xr.apply_ufunc(
        function,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[["distribution_parameter"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"distribution_parameter": 3}},
    )
