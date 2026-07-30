"""Reusable uncertainty calculations for aggregate OpenBench metrics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from openbench.config.schema import UNCERTAINTY_METRIC_DIRECTIONS

MIN_VALID_SAMPLES = 8


def derived_seed(seed: int, *parts: object) -> int:
    """Derive a stable independent NumPy seed from a user seed and context."""
    digest = hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def moving_block_indices(
    sample_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw circular moving blocks until *sample_count* positions are filled."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    block_length = min(block_length, sample_count)
    starts = rng.integers(0, sample_count, size=math.ceil(sample_count / block_length))
    return np.concatenate([(start + np.arange(block_length)) % sample_count for start in starts])[:sample_count]


def _paired_values(sim: Any, ref: Any) -> tuple[np.ndarray, np.ndarray]:
    sim_values = np.asarray(sim, dtype=float).reshape(-1)
    ref_values = np.asarray(ref, dtype=float).reshape(-1)
    if sim_values.size != ref_values.size:
        raise ValueError("simulation and reference samples must have the same length")
    valid = np.isfinite(sim_values) & np.isfinite(ref_values)
    return sim_values[valid], ref_values[valid]


def metric_value(metric: str, sim: Any, ref: Any) -> float:
    """Evaluate one supported deterministic metric with paired NaN handling."""
    if metric not in UNCERTAINTY_METRIC_DIRECTIONS:
        raise ValueError(f"unsupported uncertainty metric: {metric}")
    sim_values, ref_values = _paired_values(sim, ref)
    if sim_values.size < 2:
        return math.nan

    diff = sim_values - ref_values
    if metric == "bias":
        return float(np.mean(diff))
    if metric == "percent_bias":
        denominator = np.sum(ref_values)
        return float(100 * np.sum(diff) / denominator) if denominator != 0 else math.nan
    if metric == "absolute_percent_bias":
        denominator = abs(np.sum(ref_values))
        return float(100 * abs(np.sum(diff)) / denominator) if denominator != 0 else math.nan
    if metric == "RMSE":
        return float(np.sqrt(np.mean(diff**2)))
    if metric in {"ubRMSE", "CRMSD"}:
        return float(np.sqrt(np.mean(((sim_values - sim_values.mean()) - (ref_values - ref_values.mean())) ** 2)))
    if metric == "mean_absolute_error":
        return float(np.mean(np.abs(diff)))

    ref_variance_sum = np.sum((ref_values - ref_values.mean()) ** 2)
    if metric == "NSE":
        return float(1 - np.sum(diff**2) / ref_variance_sum) if ref_variance_sum != 0 else math.nan
    if metric == "ubNSE":
        centered_diff = (sim_values - sim_values.mean()) - (ref_values - ref_values.mean())
        return float(1 - np.sum(centered_diff**2) / ref_variance_sum) if ref_variance_sum != 0 else math.nan

    sim_std = np.std(sim_values)
    ref_std = np.std(ref_values)
    correlation = (
        float(np.corrcoef(sim_values, ref_values)[0, 1]) if sim_std != 0 and ref_std != 0 else math.nan
    )
    if metric == "correlation":
        return correlation
    if metric == "correlation_R2":
        return correlation**2
    if metric in {"KGE", "KGESS"}:
        ref_mean = np.mean(ref_values)
        if not np.isfinite(correlation) or ref_std == 0 or ref_mean == 0:
            return math.nan
        kge = 1 - math.sqrt((correlation - 1) ** 2 + (sim_std / ref_std - 1) ** 2 + (sim_values.mean() / ref_mean - 1) ** 2)
        return float((kge + 0.41) / 1.41) if metric == "KGESS" else float(kge)
    if metric == "L":
        return float(np.exp(-5 * np.sum(diff**2) / ref_variance_sum)) if ref_variance_sum != 0 else math.nan
    if metric == "index_agreement":
        denominator = np.sum((np.abs(sim_values - ref_values.mean()) + np.abs(ref_values - ref_values.mean())) ** 2)
        return float(1 - np.sum(diff**2) / denominator) if denominator != 0 else math.nan
    raise AssertionError(metric)


def quality_value(metric: str, value: float) -> float:
    """Convert a metric to a common higher-is-better scale."""
    direction = UNCERTAINTY_METRIC_DIRECTIONS[metric]
    if direction == "lower":
        return -value
    if direction == "zero":
        return -abs(value)
    return value


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else math.nan


def _interval_summary(
    estimate: float,
    samples: Sequence[float],
    *,
    confidence_level: float,
    sample_count: int,
    n_resamples: int,
    block_length: int,
    method: str,
) -> dict[str, Any]:
    valid = np.asarray(samples, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not np.isfinite(estimate) or valid.size < 2:
        return {
            "status": "insufficient_data",
            "estimate": None,
            "lower": None,
            "upper": None,
            "standard_error": None,
            "sample_count": sample_count,
            "valid_resamples": int(valid.size),
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "block_length": block_length,
            "method": method,
        }
    alpha = (1 - confidence_level) / 2
    return {
        "status": "available",
        "estimate": float(estimate),
        "lower": float(np.quantile(valid, alpha)),
        "upper": float(np.quantile(valid, 1 - alpha)),
        "standard_error": float(np.std(valid, ddof=1)),
        "sample_count": sample_count,
        "valid_resamples": int(valid.size),
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "block_length": block_length,
        "method": method,
    }


def bootstrap_metric(
    sim: Any,
    ref: Any,
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    block_length: int | None,
    seed: int,
) -> dict[str, Any]:
    """Estimate a percentile CI by paired circular moving-block bootstrap."""
    sim_values, ref_values = _paired_values(sim, ref)
    sample_count = int(sim_values.size)
    resolved_block = min(block_length or max(1, round(sample_count ** (1 / 3))), max(1, sample_count))
    if sample_count < MIN_VALID_SAMPLES:
        return _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=resolved_block,
            method="moving_block_bootstrap",
        )

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        positions = moving_block_indices(sample_count, resolved_block, rng)
        samples[index] = metric_value(metric, sim_values[positions], ref_values[positions])
    return _interval_summary(
        metric_value(metric, sim_values, ref_values),
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=resolved_block,
        method="moving_block_bootstrap",
    )


def paired_metric_difference(
    sim_a: Any,
    sim_b: Any,
    ref: Any,
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    block_length: int | None,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the quality difference between two simulations against one reference."""
    a = np.asarray(sim_a, dtype=float).reshape(-1)
    b = np.asarray(sim_b, dtype=float).reshape(-1)
    o = np.asarray(ref, dtype=float).reshape(-1)
    if a.size != b.size or a.size != o.size:
        raise ValueError("paired model difference inputs must have the same length")
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(o)
    a, b, o = a[valid], b[valid], o[valid]
    sample_count = int(a.size)
    resolved_block = min(block_length or max(1, round(sample_count ** (1 / 3))), max(1, sample_count))
    if sample_count < MIN_VALID_SAMPLES:
        return _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=resolved_block,
            method="paired_moving_block_bootstrap",
        )

    def difference(positions: Any = slice(None)) -> float:
        return quality_value(metric, metric_value(metric, a[positions], o[positions])) - quality_value(
            metric, metric_value(metric, b[positions], o[positions])
        )

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        samples[index] = difference(moving_block_indices(sample_count, resolved_block, rng))
    return _interval_summary(
        difference(),
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=resolved_block,
        method="paired_moving_block_bootstrap",
    )


def bootstrap_network_metric(
    station_pairs: Sequence[tuple[Any, Any]],
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    block_length: int | None,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a station-network mean metric using temporal blocks within stations."""
    pairs = [_paired_values(sim, ref) for sim, ref in station_pairs]
    pairs = [(sim, ref) for sim, ref in pairs if sim.size >= MIN_VALID_SAMPLES]
    sample_count = sum(sim.size for sim, _ in pairs)
    if not pairs:
        result = _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=block_length or 1,
            method="station_network_moving_block_bootstrap",
        )
        result["station_count"] = 0
        return result

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        station_values = []
        for sim, ref in pairs:
            resolved_block = min(block_length or max(1, round(sim.size ** (1 / 3))), sim.size)
            positions = moving_block_indices(sim.size, resolved_block, rng)
            station_values.append(metric_value(metric, sim[positions], ref[positions]))
        samples[index] = _finite_mean(station_values)

    result = _interval_summary(
        _finite_mean([metric_value(metric, sim, ref) for sim, ref in pairs]),
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=block_length or 0,
        method="station_network_moving_block_bootstrap",
    )
    result["station_count"] = len(pairs)
    return result


def paired_network_metric_difference(
    station_triplets: Sequence[tuple[Any, Any, Any]],
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    block_length: int | None,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired station-network quality difference."""
    triplets = []
    for sim_a, sim_b, ref in station_triplets:
        a = np.asarray(sim_a, dtype=float).reshape(-1)
        b = np.asarray(sim_b, dtype=float).reshape(-1)
        o = np.asarray(ref, dtype=float).reshape(-1)
        if a.size != b.size or a.size != o.size:
            raise ValueError("paired station inputs must have the same length")
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(o)
        if np.count_nonzero(valid) >= MIN_VALID_SAMPLES:
            triplets.append((a[valid], b[valid], o[valid]))

    sample_count = sum(a.size for a, _, _ in triplets)
    if not triplets:
        result = _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=block_length or 1,
            method="paired_station_network_moving_block_bootstrap",
        )
        result["station_count"] = 0
        return result

    def network_difference(resample: bool, rng: np.random.Generator | None = None) -> float:
        station_values = []
        for a, b, o in triplets:
            positions: Any = slice(None)
            if resample:
                resolved_block = min(block_length or max(1, round(a.size ** (1 / 3))), a.size)
                positions = moving_block_indices(a.size, resolved_block, rng)
            station_values.append(
                quality_value(metric, metric_value(metric, a[positions], o[positions]))
                - quality_value(metric, metric_value(metric, b[positions], o[positions]))
            )
        return _finite_mean(station_values)

    rng = np.random.default_rng(seed)
    samples = [network_difference(True, rng) for _ in range(n_resamples)]
    result = _interval_summary(
        network_difference(False),
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=block_length or 0,
        method="paired_station_network_moving_block_bootstrap",
    )
    result["station_count"] = len(triplets)
    return result


def verdict_from_reference_differences(
    differences: dict[str, dict[str, Any]],
    *,
    simulation_a: str,
    simulation_b: str,
) -> dict[str, Any]:
    """Classify a model pair without pooling references."""
    available = {name: result for name, result in differences.items() if result.get("status") == "available"}
    if not differences or len(available) != len(differences):
        status = "insufficient_data"
        winner = None
    else:
        signs = {int(np.sign(result["estimate"])) for result in available.values() if result["estimate"] != 0}
        if len(signs) > 1:
            status = "reference_sensitive"
            winner = None
        elif all(result["lower"] > 0 for result in available.values()):
            status = "robustly_better"
            winner = simulation_a
        elif all(result["upper"] < 0 for result in available.values()):
            status = "robustly_better"
            winner = simulation_b
        else:
            status = "indistinguishable"
            winner = None
    return {
        "status": status,
        "simulation_a": simulation_a,
        "simulation_b": simulation_b,
        "winner": winner,
        "references": differences,
    }
