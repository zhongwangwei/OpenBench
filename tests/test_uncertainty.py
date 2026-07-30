import numpy as np

from openbench.core.uncertainty import (
    _metric_values,
    bootstrap_metric,
    bootstrap_network_metric,
    metric_value,
    paired_metric_difference,
    segmented_block_index_matrix,
    segmented_block_indices,
    verdict_from_reference_differences,
)


def test_segmented_block_indices_never_cross_gap():
    indices, block_sizes = segmented_block_indices(
        [slice(0, 4), slice(4, 9)],
        sample_count=9,
        block_length=3,
        rng=np.random.default_rng(5),
    )
    offset = 0
    for size in block_sizes:
        block = indices[offset : offset + size]
        assert np.all(np.diff(block) == 1)
        assert np.all(block < 4) or np.all(block >= 4)
        offset += size


def test_segmented_block_matrix_preserves_each_segment_size():
    indices, _ = segmented_block_index_matrix(
        [slice(0, 4), slice(4, 9)],
        sample_count=9,
        block_length=3,
        n_resamples=5,
        rng=np.random.default_rng(5),
    )

    assert indices.shape == (5, 9)
    assert np.all(indices[:, :4] < 4)
    assert np.all(indices[:, 4:] >= 4)


def test_vectorized_metrics_match_scalar_metrics():
    ref = np.arange(1, 21, dtype=float)
    simulations = np.stack([ref + np.sin(ref), ref * 1.1 + 2])
    references = np.stack([ref, ref])

    for metric in (
        "bias",
        "percent_bias",
        "absolute_percent_bias",
        "RMSE",
        "ubRMSE",
        "CRMSD",
        "mean_absolute_error",
        "NSE",
        "ubNSE",
        "correlation",
        "correlation_R2",
        "KGE",
        "KGESS",
        "L",
        "index_agreement",
    ):
        expected = [metric_value(metric, sim, obs) for sim, obs in zip(simulations, references)]
        np.testing.assert_allclose(_metric_values(metric, simulations, references), expected)


def test_bootstrap_metric_is_reproducible_and_pairwise_nan_safe():
    ref = np.arange(20, dtype=float)
    sim = ref + np.sin(ref)
    sim[4] = np.nan
    kwargs = dict(
        metric="RMSE",
        n_resamples=50,
        confidence_level=0.9,
        block_length=3,
        seed=11,
    )

    first = bootstrap_metric(sim, ref, **kwargs)
    second = bootstrap_metric(sim, ref, **kwargs)

    assert first == second
    assert first["status"] == "available"
    assert first["sample_count"] == 19
    assert first["lower"] <= first["upper"]
    assert np.isfinite(first["estimate"])


def test_bootstrap_metric_splits_irregular_time_gaps():
    time = np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
            "2000-04-01",
            "2000-05-01",
            "2000-09-01",
            "2000-10-01",
            "2000-11-01",
            "2000-12-01",
            "2001-01-01",
        ],
        dtype="datetime64[D]",
    )
    ref = np.arange(time.size, dtype=float)
    result = bootstrap_metric(
        ref + np.sin(ref),
        ref,
        "RMSE",
        n_resamples=20,
        confidence_level=0.9,
        block_length=12,
        seed=2,
        time=time,
    )

    assert result["status"] == "available"
    assert result["segment_count"] == 2
    assert result["valid_pair_count"] == 10
    assert result["block_length"] == 5
    assert result["method"] == "segmented_moving_block_bootstrap"


def test_bootstrap_metric_reports_insufficient_data():
    result = bootstrap_metric(
        [1, 2, np.nan],
        [1, 2, 3],
        "bias",
        n_resamples=10,
        confidence_level=0.95,
        block_length=None,
        seed=1,
    )
    assert result["status"] == "insufficient_data"
    assert result["estimate"] is None
    assert result["sample_count"] == 2


def test_paired_difference_uses_metric_direction():
    ref = np.arange(1, 41, dtype=float)
    better = ref + 0.1
    worse = ref + 2.0
    result = paired_metric_difference(
        better,
        worse,
        ref,
        "RMSE",
        n_resamples=100,
        confidence_level=0.9,
        block_length=4,
        seed=3,
    )
    assert result["lower"] > 0


def test_station_network_bootstrap_reports_aggregate_only():
    ref = np.arange(1, 21, dtype=float)
    result = bootstrap_network_metric(
        [(ref + 1, ref), (ref + 2, ref)],
        "bias",
        n_resamples=20,
        confidence_level=0.9,
        block_length=3,
        seed=9,
    )
    assert result["status"] == "available"
    assert result["station_count"] == 2
    assert result["estimate"] == 1.5
    assert result["segment_count"] == 2


def test_verdicts_do_not_pool_references():
    robust = verdict_from_reference_differences(
        {
            "ref_a": {"status": "available", "estimate": 1.0, "lower": 0.2, "upper": 1.8},
            "ref_b": {"status": "available", "estimate": 0.7, "lower": 0.1, "upper": 1.2},
        },
        simulation_a="A",
        simulation_b="B",
    )
    sensitive = verdict_from_reference_differences(
        {
            "ref_a": {"status": "available", "estimate": 1.0, "lower": 0.2, "upper": 1.8},
            "ref_b": {"status": "available", "estimate": -0.7, "lower": -1.2, "upper": -0.1},
        },
        simulation_a="A",
        simulation_b="B",
    )
    uncertain = verdict_from_reference_differences(
        {"ref_a": {"status": "available", "estimate": 0.1, "lower": -0.2, "upper": 0.3}},
        simulation_a="A",
        simulation_b="B",
    )
    incomplete = verdict_from_reference_differences(
        {
            "ref_a": {"status": "available", "estimate": 1.0, "lower": 0.2, "upper": 1.8},
            "ref_b": {"status": "insufficient_data"},
        },
        simulation_a="A",
        simulation_b="B",
    )

    assert robust["status"] == "robustly_better"
    assert robust["winner"] == "A"
    assert sensitive["status"] == "reference_sensitive"
    assert uncertain["status"] == "indistinguishable"
    assert incomplete["status"] == "insufficient_data"
