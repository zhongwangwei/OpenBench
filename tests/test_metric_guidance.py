"""Configuration-only metric/score preflight tests (no NetCDF required)."""

from types import SimpleNamespace

import pytest

from openbench.cli._metric_guidance import metric_score_errors


def _cfg(years=(2001, 2001), *, tim_res="Day", metrics=None, scores=None):
    return SimpleNamespace(
        project=SimpleNamespace(years=list(years), tim_res=tim_res),
        metrics=metrics,
        scores=scores,
    )


def test_single_year_iav_is_an_error_but_overall_score_can_reweight():
    errors = metric_score_errors(
        _cfg(metrics=["bias", "RMSE"], scores=["nIavScore", "Overall_Score"]), "Month"
    )

    assert len(errors) == 1
    assert "nIavScore cannot be calculated" in errors[0]
    assert "project.years [2001, 2001]" in errors[0]
    assert not any("Overall_Score" in message for message in errors)


def test_single_year_overall_score_alone_is_allowed():
    errors = metric_score_errors(_cfg(metrics=[], scores=["Overall_Score"]), "Month")

    assert errors == []


def test_one_year_monthly_phase_and_seasonality_are_allowed():
    errors = metric_score_errors(
        _cfg(metrics=[], scores=["nPhaseScore", "nSeasonalityScore"]), "Month"
    )

    assert errors == []


def test_yearly_resolution_rejects_seasonality_and_single_sample_metrics():
    errors = metric_score_errors(
        _cfg(tim_res="Year", metrics=["correlation", "bias"], scores=["nPhaseScore"]), "Year"
    )

    assert any("nPhaseScore cannot be calculated" in text and "12-month" in text for text in errors)
    assert any("correlation cannot be calculated" in text and "one temporal sample" in text for text in errors)
    assert not any("bias" in text for text in errors)


def test_climatology_uses_existing_compatibility_rules():
    errors = metric_score_errors(
        _cfg(
            years=(2001, 2005),
            tim_res="climatology-year",
            metrics=["bias", "correlation", "APFB"],
            scores=["nIavScore", "Overall_Score"],
        ),
        "climatology-year",
    )

    for name in ["correlation", "APFB", "nIavScore", "Overall_Score"]:
        assert any(f"{name} cannot be calculated" in text for text in errors)
    assert not any(text.startswith("bias ") for text in errors)


def test_monthly_climatology_allows_reweighted_overall_score():
    errors = metric_score_errors(
        _cfg(years=(2001, 2005), metrics=[], scores=["Overall_Score"]), "climatology-month"
    )

    assert errors == []


def test_explicit_empty_selections_do_not_enable_defaults():
    assert metric_score_errors(_cfg(metrics=[], scores=[]), "Year") == []


def test_omitted_selections_keep_runtime_defaults():
    errors = metric_score_errors(_cfg(metrics=None, scores=None), "Year")

    assert any(message.startswith("correlation cannot be calculated") for message in errors)
    assert not any("Overall_Score" in message for message in errors)


@pytest.mark.parametrize("year_range", [(2001, 2002), (2001, 2005)])
def test_multi_year_iav_has_no_static_error(year_range):
    errors = metric_score_errors(
        _cfg(years=year_range, tim_res="Month", metrics=[], scores=["nIavScore"]), "Month"
    )

    assert errors == []
