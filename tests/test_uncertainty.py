import numpy as np

from openbench.core.uncertainty import (
    bootstrap_metric,
    bootstrap_network_metric,
    moving_block_indices,
    paired_metric_difference,
    verdict_from_reference_differences,
)


def test_moving_block_indices_preserve_contiguous_blocks():
    indices = moving_block_indices(10, 3, np.random.default_rng(2))
    assert len(indices) == 10
    for start in range(0, 9, 3):
        assert np.all((np.diff(indices[start : start + 3]) % 10) == 1)


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
    assert first["lower"] <= first["estimate"] <= first["upper"]


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
