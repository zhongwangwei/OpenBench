"""Reusable uncertainty calculations for aggregate OpenBench metrics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from openbench.config.schema import UNCERTAINTY_METRIC_DIRECTIONS

MIN_VALID_SAMPLES = 8
BOOTSTRAP_BATCH_SIZE = 128


def derived_seed(seed: int, *parts: object) -> int:
    """Derive a stable independent NumPy seed from a user seed and context."""
    digest = hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _numeric_time_deltas(time: Any) -> np.ndarray | None:
    values = np.asarray(time).reshape(-1)
    deltas = []
    for earlier, later in zip(values[:-1], values[1:]):
        try:
            delta = later - earlier
            if isinstance(delta, np.timedelta64):
                value = float(delta / np.timedelta64(1, "ns"))
            elif hasattr(delta, "total_seconds"):
                value = float(delta.total_seconds())
            else:
                value = float(delta)
        except (TypeError, ValueError, OverflowError):
            return None
        deltas.append(value)
    return np.asarray(deltas, dtype=float)


def _paired_values_and_segments(
    arrays: Sequence[Any],
    time: Any | None = None,
) -> tuple[list[np.ndarray], list[slice]]:
    values = [np.asarray(array, dtype=float).reshape(-1) for array in arrays]
    if not values or any(array.size != values[0].size for array in values[1:]):
        raise ValueError("paired samples must have the same length")
    if time is not None and np.asarray(time).size != values[0].size:
        raise ValueError("time coordinates and paired samples must have the same length")

    valid = np.logical_and.reduce([np.isfinite(array) for array in values])
    valid_positions = np.flatnonzero(valid)
    filtered = [array[valid] for array in values]
    if valid_positions.size == 0:
        return filtered, []

    time_deltas = _numeric_time_deltas(time) if time is not None else None
    expected_step = None
    if time_deltas is not None:
        positive = time_deltas[np.isfinite(time_deltas) & (time_deltas > 0)]
        if positive.size:
            expected_step = float(np.median(positive))

    breaks = [0]
    for compressed_index, (previous, current) in enumerate(zip(valid_positions[:-1], valid_positions[1:]), start=1):
        discontinuous = current != previous + 1
        if expected_step is not None:
            delta = time_deltas[previous]
            discontinuous = discontinuous or not np.isfinite(delta) or delta <= 0 or delta > expected_step * 1.5
        if discontinuous:
            breaks.append(compressed_index)
    breaks.append(int(valid_positions.size))
    return filtered, [slice(start, stop) for start, stop in zip(breaks[:-1], breaks[1:]) if stop > start]


def segmented_block_indices(
    segments: Sequence[slice],
    sample_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Draw non-circular blocks without crossing a contiguous-segment boundary."""
    matrix, block_sizes = segmented_block_index_matrix(
        segments,
        sample_count,
        block_length,
        1,
        rng,
    )
    return matrix[0], block_sizes


def segmented_block_index_matrix(
    segments: Sequence[slice],
    sample_count: int,
    block_length: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Draw stratified non-circular blocks for several bootstrap resamples."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    usable = [segment for segment in segments if int(segment.stop) > int(segment.start)]
    if not usable:
        raise ValueError("at least one non-empty segment is required")
    if sum(int(segment.stop) - int(segment.start) for segment in usable) != sample_count:
        raise ValueError("sample_count must equal the total segment length")

    resampled_segments: list[np.ndarray] = []
    block_sizes: list[int] = []
    for segment in usable:
        segment_length = int(segment.stop) - int(segment.start)
        actual_length = min(block_length, segment_length)
        block_count = math.ceil(segment_length / actual_length)
        starts = rng.integers(
            int(segment.start),
            int(segment.stop) - actual_length + 1,
            size=(n_resamples, block_count),
        )
        offsets = np.arange(actual_length)
        indices = (starts[..., None] + offsets).reshape(n_resamples, -1)[:, :segment_length]
        resampled_segments.append(indices)
        block_sizes.extend(
            [actual_length] * (block_count - 1) + [segment_length - actual_length * (block_count - 1)]
        )
    return np.concatenate(resampled_segments, axis=1), block_sizes


def _resolved_block_length(sample_count: int, segments: Sequence[slice], requested: int | None) -> int:
    longest_segment = max((int(segment.stop) - int(segment.start) for segment in segments), default=1)
    automatic = max(1, round(max(1, sample_count) ** (1 / 3)))
    return min(requested or automatic, longest_segment)


def _paired_values(sim: Any, ref: Any) -> tuple[np.ndarray, np.ndarray]:
    values, _ = _paired_values_and_segments((sim, ref))
    return values[0], values[1]


def metric_value(metric: str, sim: Any, ref: Any) -> float:
    """Evaluate one supported deterministic metric with paired NaN handling."""
    if metric not in UNCERTAINTY_METRIC_DIRECTIONS:
        raise ValueError(f"unsupported uncertainty metric: {metric}")
    sim_values, ref_values = _paired_values(sim, ref)
    if sim_values.size < 2:
        return math.nan
    return float(_metric_values(metric, sim_values, ref_values))


def _metric_values(metric: str, sim_values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    """Evaluate one metric along the last axis of paired finite arrays."""
    if metric not in UNCERTAINTY_METRIC_DIRECTIONS:
        raise ValueError(f"unsupported uncertainty metric: {metric}")
    diff = sim_values - ref_values
    if metric == "bias":
        return np.mean(diff, axis=-1)
    if metric == "percent_bias":
        denominator = np.sum(ref_values, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator != 0, 100 * np.sum(diff, axis=-1) / denominator, np.nan)
    if metric == "absolute_percent_bias":
        denominator = np.abs(np.sum(ref_values, axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator != 0, 100 * np.abs(np.sum(diff, axis=-1)) / denominator, np.nan)
    if metric == "RMSE":
        return np.sqrt(np.mean(diff**2, axis=-1))
    sim_centered = sim_values - np.mean(sim_values, axis=-1, keepdims=True)
    ref_centered = ref_values - np.mean(ref_values, axis=-1, keepdims=True)
    if metric in {"ubRMSE", "CRMSD"}:
        return np.sqrt(np.mean((sim_centered - ref_centered) ** 2, axis=-1))
    if metric == "mean_absolute_error":
        return np.mean(np.abs(diff), axis=-1)

    ref_variance_sum = np.sum(ref_centered**2, axis=-1)
    if metric == "NSE":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ref_variance_sum != 0, 1 - np.sum(diff**2, axis=-1) / ref_variance_sum, np.nan)
    if metric == "ubNSE":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                ref_variance_sum != 0,
                1 - np.sum((sim_centered - ref_centered) ** 2, axis=-1) / ref_variance_sum,
                np.nan,
            )

    sim_variance_sum = np.sum(sim_centered**2, axis=-1)
    correlation_denominator = np.sqrt(sim_variance_sum * ref_variance_sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(
            correlation_denominator != 0,
            np.sum(sim_centered * ref_centered, axis=-1) / correlation_denominator,
            np.nan,
        )
    if metric == "correlation":
        return correlation
    if metric == "correlation_R2":
        return correlation**2
    if metric in {"KGE", "KGESS"}:
        sim_std = np.std(sim_values, axis=-1)
        ref_std = np.std(ref_values, axis=-1)
        sim_mean = np.mean(sim_values, axis=-1)
        ref_mean = np.mean(ref_values, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kge = 1 - np.sqrt(
                (correlation - 1) ** 2 + (sim_std / ref_std - 1) ** 2 + (sim_mean / ref_mean - 1) ** 2
            )
        kge = np.where(np.isfinite(correlation) & (ref_std != 0) & (ref_mean != 0), kge, np.nan)
        return (kge + 0.41) / 1.41 if metric == "KGESS" else kge
    if metric == "L":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                ref_variance_sum != 0,
                np.exp(-5 * np.sum(diff**2, axis=-1) / ref_variance_sum),
                np.nan,
            )
    if metric == "index_agreement":
        ref_mean = np.mean(ref_values, axis=-1, keepdims=True)
        denominator = np.sum((np.abs(sim_values - ref_mean) + np.abs(ref_values - ref_mean)) ** 2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator != 0, 1 - np.sum(diff**2, axis=-1) / denominator, np.nan)
    raise AssertionError(metric)


def quality_value(metric: str, value: float) -> float:
    """Convert a metric to a common higher-is-better scale."""
    return float(_quality_values(metric, np.asarray(value)))


def _quality_values(metric: str, values: np.ndarray) -> np.ndarray:
    direction = UNCERTAINTY_METRIC_DIRECTIONS[metric]
    if direction == "lower":
        return -values
    if direction == "zero":
        return -np.abs(values)
    return values


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
    segment_count: int = 1,
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
            "valid_pair_count": sample_count,
            "segment_count": segment_count,
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
        "valid_pair_count": sample_count,
        "segment_count": segment_count,
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
    time: Any | None = None,
) -> dict[str, Any]:
    """Estimate a percentile CI using paired gap-aware moving blocks."""
    values, segments = _paired_values_and_segments((sim, ref), time)
    sim_values, ref_values = values
    sample_count = int(sim_values.size)
    resolved_block = _resolved_block_length(sample_count, segments, block_length)
    if sample_count < MIN_VALID_SAMPLES:
        return _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=resolved_block,
            method="segmented_moving_block_bootstrap",
            segment_count=len(segments),
        )

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, n_resamples)
        positions, _ = segmented_block_index_matrix(
            segments,
            sample_count,
            resolved_block,
            stop - start,
            rng,
        )
        samples[start:stop] = _metric_values(metric, sim_values[positions], ref_values[positions])
    return _interval_summary(
        metric_value(metric, sim_values, ref_values),
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=resolved_block,
        method="segmented_moving_block_bootstrap",
        segment_count=len(segments),
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
    time: Any | None = None,
) -> dict[str, Any]:
    """Bootstrap the quality difference between two simulations against one reference."""
    values, segments = _paired_values_and_segments((sim_a, sim_b, ref), time)
    a, b, o = values
    sample_count = int(a.size)
    resolved_block = _resolved_block_length(sample_count, segments, block_length)
    if sample_count < MIN_VALID_SAMPLES:
        return _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=resolved_block,
            method="paired_segmented_moving_block_bootstrap",
            segment_count=len(segments),
        )

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, n_resamples)
        positions, _ = segmented_block_index_matrix(
            segments,
            sample_count,
            resolved_block,
            stop - start,
            rng,
        )
        samples[start:stop] = _quality_values(metric, _metric_values(metric, a[positions], o[positions])) - (
            _quality_values(metric, _metric_values(metric, b[positions], o[positions]))
        )
    estimate = quality_value(metric, metric_value(metric, a, o)) - quality_value(metric, metric_value(metric, b, o))
    return _interval_summary(
        estimate,
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=resolved_block,
        method="paired_segmented_moving_block_bootstrap",
        segment_count=len(segments),
    )


def bootstrap_network_metric(
    station_pairs: Sequence[tuple[Any, ...]],
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    block_length: int | None,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a station-network mean metric using temporal blocks within stations."""
    pairs = []
    for pair in station_pairs:
        if len(pair) not in {2, 3}:
            raise ValueError("station pairs must contain simulation, reference, and optional time")
        values, segments = _paired_values_and_segments(pair[:2], pair[2] if len(pair) == 3 else None)
        sim, ref = values
        if sim.size >= MIN_VALID_SAMPLES:
            pairs.append((sim, ref, segments, _resolved_block_length(sim.size, segments, block_length)))
    sample_count = sum(sim.size for sim, _, _, _ in pairs)
    segment_count = sum(len(segments) for _, _, segments, _ in pairs)
    if not pairs:
        result = _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=block_length or 1,
            method="station_network_segmented_moving_block_bootstrap",
            segment_count=segment_count,
        )
        result["station_count"] = 0
        return result

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, n_resamples)
        batch_size = stop - start
        total = np.zeros(batch_size, dtype=float)
        count = np.zeros(batch_size, dtype=int)
        for sim, ref, segments, resolved_block in pairs:
            positions, _ = segmented_block_index_matrix(
                segments,
                sim.size,
                resolved_block,
                batch_size,
                rng,
            )
            values = _metric_values(metric, sim[positions], ref[positions])
            finite = np.isfinite(values)
            total[finite] += values[finite]
            count[finite] += 1
        samples[start:stop] = np.divide(total, count, out=np.full(batch_size, np.nan), where=count > 0)

    resolved_blocks = [resolved for _, _, _, resolved in pairs]
    result = _interval_summary(
        _finite_mean([metric_value(metric, sim, ref) for sim, ref, _, _ in pairs]),
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=max(resolved_blocks),
        method="station_network_segmented_moving_block_bootstrap",
        segment_count=segment_count,
    )
    result["station_count"] = len(pairs)
    result["minimum_block_length"] = min(resolved_blocks)
    return result


def paired_network_metric_difference(
    station_triplets: Sequence[tuple[Any, ...]],
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    block_length: int | None,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired station-network quality difference."""
    triplets = []
    for triplet in station_triplets:
        if len(triplet) not in {3, 4}:
            raise ValueError("station triplets must contain two simulations, reference, and optional time")
        values, segments = _paired_values_and_segments(triplet[:3], triplet[3] if len(triplet) == 4 else None)
        a, b, o = values
        if a.size >= MIN_VALID_SAMPLES:
            triplets.append((a, b, o, segments, _resolved_block_length(a.size, segments, block_length)))

    sample_count = sum(a.size for a, _, _, _, _ in triplets)
    segment_count = sum(len(segments) for _, _, _, segments, _ in triplets)
    if not triplets:
        result = _interval_summary(
            math.nan,
            [],
            confidence_level=confidence_level,
            sample_count=sample_count,
            n_resamples=n_resamples,
            block_length=block_length or 1,
            method="paired_station_network_segmented_moving_block_bootstrap",
            segment_count=segment_count,
        )
        result["station_count"] = 0
        return result

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, n_resamples)
        batch_size = stop - start
        total = np.zeros(batch_size, dtype=float)
        count = np.zeros(batch_size, dtype=int)
        for a, b, o, segments, resolved_block in triplets:
            positions, _ = segmented_block_index_matrix(
                segments,
                a.size,
                resolved_block,
                batch_size,
                rng,
            )
            values = _quality_values(metric, _metric_values(metric, a[positions], o[positions])) - _quality_values(
                metric, _metric_values(metric, b[positions], o[positions])
            )
            finite = np.isfinite(values)
            total[finite] += values[finite]
            count[finite] += 1
        samples[start:stop] = np.divide(total, count, out=np.full(batch_size, np.nan), where=count > 0)

    estimate = _finite_mean(
        [
            quality_value(metric, metric_value(metric, a, o))
            - quality_value(metric, metric_value(metric, b, o))
            for a, b, o, _, _ in triplets
        ]
    )
    resolved_blocks = [resolved for _, _, _, _, resolved in triplets]
    result = _interval_summary(
        estimate,
        samples,
        confidence_level=confidence_level,
        sample_count=sample_count,
        n_resamples=n_resamples,
        block_length=max(resolved_blocks),
        method="paired_station_network_segmented_moving_block_bootstrap",
        segment_count=segment_count,
    )
    result["station_count"] = len(triplets)
    result["minimum_block_length"] = min(resolved_blocks)
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
