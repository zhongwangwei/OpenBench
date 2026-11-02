"""Configuration-only metric/score advisory tests (no NetCDF required)."""

from types import SimpleNamespace

import pytest

from openbench.cli._metric_guidance import METRIC_LIMITS, SCORE_LIMITS, metric_score_findings
from openbench.core.registry import IMPLEMENTED_METRICS, IMPLEMENTED_SCORES


def _cfg(years=(2001, 2001), *, tim_res="Day", metrics=None, scores=None):
    return SimpleNamespace(
        project=SimpleNamespace(years=list(years), tim_res=tim_res),
        metrics=metrics,
        scores=scores,
    )


def test_guidance_covers_every_selectable_metric_and_score():
    assert set(METRIC_LIMITS) == IMPLEMENTED_METRICS
    assert set(SCORE_LIMITS) == IMPLEMENTED_SCORES
    assert all(METRIC_LIMITS.values()) and all(SCORE_LIMITS.values())


def test_single_year_iav_warns_without_blocking_normal_metrics():
    warnings, notes = metric_score_findings(
        _cfg(metrics=["bias", "RMSE"], scores=["nIavScore", "Overall_Score"])
    )
    assert any("nIavScore:" in message and "only one year" in message for message in warnings)
    assert any("Overall_Score:" in message and "IAV" in message for message in warnings)
    assert any("bias:" in note for note in notes)
    assert any("RMSE:" in note for note in notes)
    assert not any("bias:" in message or "RMSE:" in message for message in warnings)
    assert any("raw daily samples" in message for message in warnings)


def test_one_year_monthly_phase_is_not_rejected():
    warnings, notes = metric_score_findings(
        _cfg(tim_res="Month", metrics=[], scores=["nPhaseScore", "nSeasonalityScore"])
    )
    assert warnings == []
    assert any("12 calendar months" in note for note in notes)


def test_yearly_resolution_flags_seasonality_and_single_sample_temporal_metrics():
    warnings, _ = metric_score_findings(
        _cfg(tim_res="Year", metrics=["correlation", "bias"], scores=["nPhaseScore"])
    )
    assert any("nPhaseScore:" in text and "12-month" in text for text in warnings)
    assert any("correlation:" in text and "single Year-resolution" in text for text in warnings)
    assert not any("bias:" in text for text in warnings)


def test_climatology_uses_existing_compatibility_rules():
    warnings, _ = metric_score_findings(
        _cfg(years=(2001, 2005), tim_res="climatology-year",
             metrics=["bias", "correlation", "APFB"], scores=["nIavScore", "Overall_Score"])
    )
    for name in ["correlation", "APFB", "nIavScore", "Overall_Score"]:
        assert any(f"{name}: climatology-year is unsupported" in text for text in warnings)
    assert not any("bias:" in text for text in warnings)


def test_climatology_month_overall_notes_missing_iav_component():
    warnings, _ = metric_score_findings(
        _cfg(years=(2001, 2005), tim_res="climatology-month",
             metrics=[], scores=["Overall_Score"])
    )
    assert any("one representative annual cycle" in text for text in warnings)


def test_explicit_empty_selections_do_not_enable_defaults():
    warnings, notes = metric_score_findings(_cfg(metrics=[], scores=[]))
    assert warnings == []
    assert not any(note.startswith("bias:") or note.startswith("Overall_Score:") for note in notes)


def test_omitted_selections_use_runtime_defaults_and_unknown_names_are_ignored():
    _, notes = metric_score_findings(_cfg(tim_res=None))
    assert any(note.startswith("bias:") for note in notes)
    assert any(note.startswith("Overall_Score:") for note in notes)
    _, notes = metric_score_findings(_cfg(metrics=["not-a-metric"], scores=[]))
    assert not any(note.startswith("not-a-metric:") for note in notes)


@pytest.mark.parametrize("year_range", [(2001, 2002), (2001, 2005)])
def test_multi_year_iav_guidance_is_not_a_one_year_error(year_range):
    warnings, _ = metric_score_findings(
        _cfg(years=year_range, tim_res="Month", metrics=[], scores=["nIavScore"])
    )
    assert not any("only one year" in text for text in warnings)
